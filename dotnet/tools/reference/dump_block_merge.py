"""Dumps upstream's block grouping and the images it stacks from each group.

`merge_blocks` decides which consecutive blocks are one paragraph split across columns or around
a figure, and `merge_images` stacks each group into a single image for the model to read. Both
run on the default path, so both are compared directly rather than by eye: the grouping
decisions, the alignment chosen for each join, and the stacked pixels.
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

# The labels the pipeline refuses to merge: the pictures it keeps whole, plus tables.
NON_MERGE_LABELS = ["image", "header_image", "footer_image", "chart", "seal", "table"]


def crop(width: int, height: int, seed: int) -> np.ndarray:
    """A deterministic block image, so the stacked pixels are worth comparing."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def block(label: str, box, seed: int) -> dict:
    x1, y1, x2, y2 = box
    return {"label": label, "box": list(box), "img": crop(x2 - x1, y2 - y1, seed), "seed": seed}


def cases():
    """Layouts that reach each branch of the grouping rules.

    Two rules join blocks. `is_updown_align` wants two text runs stacked with a small gap, one
    edge aligned and not the other, and — the part that is easy to miss — a picture or table
    straddling the span they would cover: that is the paragraph-split-around-a-figure case the
    rule exists for, and two ordinary consecutive paragraphs are deliberately not joined.
    `is_cross` wants two text runs side by side with a narrow gutter, which is a paragraph split
    across columns.
    """
    return [
        # Left edges aligned, right edges not, with a figure across the gap.
        ("updown_left_aligned", [
            block("text", (40, 100, 340, 200), 1),
            block("image", (250, 190, 420, 260), 2),
            block("text", (40, 210, 300, 310), 3),
        ]),
        # Right edges aligned instead.
        ("updown_right_aligned", [
            block("text", (80, 100, 340, 200), 4),
            block("image", (250, 190, 420, 260), 5),
            block("text", (40, 210, 340, 310), 6),
        ]),
        # The same geometry with nothing straddling the gap: two ordinary paragraphs, left alone.
        ("updown_without_figure", [
            block("text", (40, 100, 340, 200), 7),
            block("text", (40, 210, 300, 310), 8),
        ]),
        # Both edges aligned, so the exclusive-or fails and they stay apart.
        ("updown_both_edges_aligned", [
            block("text", (40, 100, 340, 200), 9),
            block("image", (250, 190, 420, 260), 10),
            block("text", (40, 210, 340, 310), 11),
        ]),
        # Too far apart vertically.
        ("updown_gap_too_large", [
            block("text", (40, 100, 340, 200), 12),
            block("image", (250, 190, 420, 320), 13),
            block("text", (40, 300, 300, 400), 14),
        ]),
        # Side by side with a narrow gutter.
        ("cross_columns", [
            block("text", (40, 100, 240, 300), 15),
            block("text", (250, 120, 450, 320), 16),
        ]),
        # Side by side but the gutter is wider than three tenths of the column.
        ("cross_gutter_too_wide", [
            block("text", (40, 100, 240, 300), 17),
            block("text", (380, 120, 580, 320), 18),
        ]),
        # Three runs chained by the up-down rule.
        ("three_run_group", [
            block("text", (40, 100, 340, 180), 19),
            block("image", (250, 170, 420, 380), 20),
            block("text", (40, 190, 300, 270), 21),
            block("text", (40, 280, 320, 360), 22),
        ]),
        # A group that forms and is then abandoned: stacked, it is more than three times as tall
        # as it is wide.
        ("aspect_ratio_abandons", [
            block("text", (40, 100, 140, 400), 23),
            block("image", (120, 390, 200, 460), 24),
            block("text", (40, 420, 130, 720), 25),
        ]),
        # The rule only joins text, so a heading above a paragraph stays its own block.
        ("heading_above_text", [
            block("paragraph_title", (40, 100, 340, 150), 26),
            block("image", (250, 140, 420, 200), 27),
            block("text", (40, 160, 300, 260), 28),
        ]),
    ]


def main() -> None:
    from paddlex.inference.pipelines.paddleocr_vl.uilts import merge_blocks

    records = []
    payload = {}

    for index, (name, blocks) in enumerate(cases()):
        merged = merge_blocks(
            [dict(b) for b in blocks], non_merge_labels=list(NON_MERGE_LABELS))

        entries = []
        for position, result in enumerate(merged):
            key = f"img_{index}_{position}"
            image = result.get("img")
            if image is not None:
                payload[key] = np.ascontiguousarray(image)

            entries.append({
                "label": result["label"],
                "box": [int(v) for v in result["box"]],
                "seed": result["seed"],
                "group_id": result.get("group_id", None),
                "merge_aligns": result.get("merge_aligns", None),
                "image_key": key if image is not None else None,
                "image_shape": list(image.shape) if image is not None else None,
            })

        inputs = []
        for position, b in enumerate(blocks):
            key = f"in_{index}_{position}"
            payload[key] = np.ascontiguousarray(b["img"])
            inputs.append({
                "label": b["label"],
                "box": b["box"],
                "seed": b["seed"],
                "image_key": key,
            })

        records.append({
            "name": name,
            "blocks": inputs,
            "result": entries,
        })

        print(f"{name}: order={[e['seed'] for e in entries]} "
              f"groups={[e['group_id'] for e in entries]} "
              f"aligns={[e['merge_aligns'] for e in entries if e['merge_aligns']]}")

    payload["cases"] = np.frombuffer(
        json.dumps(records, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
    save_exact("block_merge.npz", **payload)


if __name__ == "__main__":
    main()
