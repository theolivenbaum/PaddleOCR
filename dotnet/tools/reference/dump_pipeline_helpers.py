"""Dumps the three default-path pipeline helpers that had only hand-written tests.

`filter_overlap_boxes` decides which of two overlapping regions survives, `convert_otsl_to_html`
turns the model's table markup into HTML, `truncate_repetitive_content` cuts a block's output
short when the decoder has fallen into a loop, and `crop_margin` trims a formula's surround. All
four run on the default path, and all four are pure functions of their arguments, so they are
compared against upstream directly.
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from _common import save_exact  # noqa: E402


def box(label, coordinate, polygon=None):
    entry = {"label": label, "coordinate": list(coordinate), "score": 0.9, "cls_id": 0}
    if polygon is not None:
        entry["polygon_points"] = [list(p) for p in polygon]
    return entry


def overlap_cases():
    """Layouts that reach each arm of the drop rule.

    The rule is not symmetric: an `inline_formula` is dropped at a lower threshold than anything
    else, a pairing of a picture-like label with a different label is spared entirely unless a
    table is involved, and a region under six pixels on a side goes regardless of what it
    overlaps. The polygon arm only fires when the shape mode is not `rect`.
    """
    return [
        ("nothing_overlaps", "auto", [
            box("text", (0, 0, 100, 40)),
            box("text", (0, 60, 100, 100)),
        ]),
        ("smaller_inside_larger", "auto", [
            box("text", (0, 0, 200, 200)),
            box("text", (20, 20, 60, 60)),
        ]),
        ("larger_second", "auto", [
            box("text", (20, 20, 60, 60)),
            box("text", (0, 0, 200, 200)),
        ]),
        ("thin_sliver_dropped", "auto", [
            box("text", (0, 0, 200, 4)),
            box("text", (0, 60, 200, 200)),
        ]),
        ("inline_formula_lower_threshold", "auto", [
            box("text", (0, 0, 200, 100)),
            box("inline_formula", (0, 0, 120, 100)),
        ]),
        ("inline_formula_both", "auto", [
            box("inline_formula", (0, 0, 200, 100)),
            box("inline_formula", (10, 0, 190, 100)),
        ]),
        ("picture_over_text_is_spared", "auto", [
            box("image", (0, 0, 200, 200)),
            box("text", (10, 10, 190, 190)),
        ]),
        ("table_over_text_is_not_spared", "auto", [
            box("table", (0, 0, 200, 200)),
            box("text", (10, 10, 190, 190)),
        ]),
        ("table_over_image_is_spared", "auto", [
            box("table", (0, 0, 200, 200)),
            box("image", (10, 10, 190, 190)),
        ]),
        ("reference_label_removed", "auto", [
            box("reference", (0, 0, 100, 40)),
            box("text", (0, 60, 100, 100)),
        ]),
        # Boxes overlap past the threshold but their outlines barely touch: the polygon arm keeps
        # both. The same pair in `rect` mode drops one, which is the point of the pairing.
        ("polygons_disagree_with_boxes", "auto", [
            box("text", (0, 0, 200, 200), [(0, 0), (100, 0), (100, 100), (0, 100)]),
            box("text", (10, 10, 190, 190), [(110, 110), (200, 110), (200, 200), (110, 200)]),
        ]),
        ("polygons_ignored_in_rect_mode", "rect", [
            box("text", (0, 0, 200, 200), [(0, 0), (100, 0), (100, 100), (0, 100)]),
            box("text", (10, 10, 190, 190), [(110, 110), (200, 110), (200, 200), (110, 200)]),
        ]),
        ("chain_of_three", "auto", [
            box("text", (0, 0, 300, 300)),
            box("text", (10, 10, 290, 290)),
            box("text", (20, 20, 280, 280)),
        ]),
    ]


def otsl_cases():
    return [
        ("simple_grid", "<fcel>A<fcel>B<nl><fcel>1<fcel>2<nl>"),
        ("empty_cells", "<fcel>A<ecel><nl><ecel><fcel>2<nl>"),
        ("column_span", "<fcel>A<lcel><nl><fcel>1<fcel>2<nl>"),
        ("row_span", "<fcel>A<fcel>B<nl><ucel><fcel>2<nl>"),
        ("both_spans", "<fcel>A<lcel><fcel>C<nl><ucel><xcel><fcel>3<nl>"),
        ("ragged_rows_padded", "<fcel>A<fcel>B<fcel>C<nl><fcel>1<nl>"),
        ("no_trailing_newline", "<fcel>A<fcel>B<nl><fcel>1<fcel>2"),
        ("markup_inside_a_cell", "<fcel>a<sub>1</sub><fcel>$x^2$<nl><fcel>1<fcel>2<nl>"),
        ("single_cell", "<fcel>only<nl>"),
        ("empty", ""),
    ]


def truncation_cases():
    """Content long enough to reach the guard at all — under 3000 characters it is returned as is."""
    pad = "The quick brown fox jumps over the lazy dog. "
    return [
        ("short_is_untouched", "a short line"),
        ("long_but_varied", "".join(f"line {i} of ordinary prose\n" for i in range(200))),
        ("repeating_suffix", ("Introductory prose that is not part of the loop. "
                              + "and so on and so forth " * 200)),
        ("whole_string_repeats", "abcdefghij" * 400),
        ("repeated_line", "the same line over and over\n" * 200),
        ("repeated_line_diluted", ("the same line over and over\n" * 100)
                                  + "".join(f"distinct line {i}\n" for i in range(100))),
        ("blank", " " * 4000),
        ("long_single_line_no_loop", pad * 100),
    ]


def margin_cases():
    """Formula crops, chosen for what the normalisation step does to each.

    `crop_margin` stretches the grey to the full range before thresholding, so the crop it makes
    depends on the image's own contrast: a faint mark on a grey ground is trimmed exactly as a
    black one on white is, and a flat image is returned untouched. The colour conversion is worth
    exercising too — upstream asks OpenCV for a BGR-to-grey on an array that is RGB, so the red
    and blue weights land on the wrong channels.
    """
    def canvas(width, height, colour):
        return np.full((height, width, 3), colour, dtype=np.uint8)

    plain = canvas(60, 40, 255)
    plain[12:28, 15:45] = 20

    faint = canvas(60, 40, 160)
    faint[12:28, 15:45] = 120

    coloured = canvas(60, 40, 255)
    coloured[12:28, 15:45] = (200, 30, 30)

    edge_to_edge = canvas(60, 40, 255)
    edge_to_edge[:, :] = 0

    thin = canvas(60, 40, 255)
    thin[20:21, 15:45] = 0

    off_centre = canvas(80, 50, 255)
    off_centre[2:10, 60:78] = 40

    return [
        ("black_on_white", plain),
        ("faint_on_grey", faint),
        ("coloured_glyphs", coloured),
        ("flat", canvas(60, 40, 128)),
        ("all_foreground", edge_to_edge),
        ("one_pixel_tall", thin),
        ("off_centre", off_centre),
    ]


def main() -> None:
    from paddlex.inference.pipelines.paddleocr_vl.uilts import (
        convert_otsl_to_html,
        crop_margin,
        filter_overlap_boxes,
        truncate_repetitive_content,
    )

    overlaps = []
    for name, mode, boxes in overlap_cases():
        kept = filter_overlap_boxes({"boxes": boxes}, mode)["boxes"]
        overlaps.append({
            "name": name,
            "shape_mode": mode,
            "boxes": boxes,
            "kept": [b["coordinate"] for b in kept],
            "kept_labels": [b["label"] for b in kept],
        })
        print(f"overlap {name}: kept {len(kept)}/{len(boxes)}")

    tables = []
    for name, otsl in otsl_cases():
        tables.append({"name": name, "otsl": otsl, "html": convert_otsl_to_html(otsl)})
        print(f"otsl {name}: {len(tables[-1]['html'])} chars")

    truncations = []
    for name, content in truncation_cases():
        result = truncate_repetitive_content(content)
        truncations.append({"name": name, "content": content, "truncated": result})
        print(f"truncate {name}: {len(content)} -> {len(result)}")

    margins = []
    arrays = {}
    for index, (name, image) in enumerate(margin_cases()):
        cropped = crop_margin(image)
        arrays[f"margin_in_{index}"] = np.ascontiguousarray(image)
        arrays[f"margin_out_{index}"] = np.ascontiguousarray(cropped)
        margins.append({
            "name": name,
            "input_key": f"margin_in_{index}",
            "output_key": f"margin_out_{index}",
        })
        print(f"crop_margin {name}: {image.shape[1]}x{image.shape[0]} -> "
              f"{cropped.shape[1]}x{cropped.shape[0]}")

    payload = json.dumps(
        {"overlaps": overlaps, "tables": tables, "truncations": truncations, "margins": margins},
        ensure_ascii=False).encode("utf-8")
    save_exact("pipeline_helpers.npz", cases=np.frombuffer(payload, dtype=np.uint8), **arrays)


if __name__ == "__main__":
    main()
