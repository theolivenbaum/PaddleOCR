"""Dumps reference polygons for the layout model's mask head.

Upstream's default `layout_shape_mode="auto"` turns each detection's mask into a polygon
(`extract_polygon_points_by_masks`, which is `mask2polygon` plus `extract_custom_vertices` plus
`_normalize_layout_polygon`) and then uses those polygons both to filter overlapping regions and
to crop the blocks. This records the inputs and the polygons upstream produces for them.

PaddleX is a read-only checkout here rather than an installed package, so importing it whole
fails on its metadata lookup; the functions under test are exec'd out of `processors.py`
instead, which is also what pins the reference to the exact source being ported.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from _common import save_exact  # noqa: E402

PADDLEX_DIR = os.environ.get("PADDLEX_DIR", "/home/user/ref/PaddleX")
PROCESSORS = os.path.join(
    PADDLEX_DIR, "paddlex/inference/models/layout_analysis/processors.py")
UILTS = os.path.join(PADDLEX_DIR, "paddlex/inference/pipelines/paddleocr_vl/uilts.py")


def load_upstream():
    """Exec's the polygon functions out of PaddleX without importing the package."""
    import cv2

    source = open(PROCESSORS).read().splitlines()

    # `is_convex` through `convert_polygon_to_quad`: everything the mask head's polygon path
    # needs, and nothing that reaches for PaddleX's own imports.
    start = next(i for i, line in enumerate(source) if line.startswith("def is_convex("))
    end = next(i for i, line in enumerate(source) if line.startswith("def restructured_boxes("))
    body = "\n".join(source[start:end])

    # `calculate_polygon_overlap_ratio` lives further down the same file; take it verbatim too.
    ratio_start = next(
        i for i, line in enumerate(source)
        if line.startswith("def calculate_polygon_overlap_ratio("))
    ratio_end = next(
        i for i, line in enumerate(source) if line.startswith("def calculate_bbox_area("))
    valid_start = next(i for i, line in enumerate(source) if line.startswith("def make_valid("))
    body = (
        "\n".join(source[valid_start:ratio_start])
        + "\n" + "\n".join(source[ratio_start:ratio_end])
        + "\n" + body
    )

    module = types.ModuleType("paddlex_layout_polygons")
    module.__dict__.update({
        "np": np,
        "cv2": cv2,
        "function_requires_deps": lambda *a, **k: (lambda f: f),
    })
    sys.modules[module.__name__] = module
    exec(compile(body, PROCESSORS, "exec"), module.__dict__)
    return module


def synthetic_case(seed: int, page_w: int, page_h: int, count: int):
    """Boxes with blobby 200x200 masks, in the layout detector's own coordinate conventions."""
    rng = np.random.default_rng(seed)
    boxes = []
    masks = []

    yy, xx = np.mgrid[0:200, 0:200]

    for _ in range(count):
        x1 = rng.integers(0, page_w - 40)
        y1 = rng.integers(0, page_h - 40)
        x2 = x1 + rng.integers(30, max(31, page_w - x1))
        y2 = y1 + rng.integers(30, max(31, page_h - y1))
        boxes.append([rng.integers(0, 25), rng.uniform(0.4, 1.0), x1, y1, x2, y2])

        mask = np.zeros((200, 200), dtype=np.int32)
        # The mask covers the whole page at 200x200; paint a blob roughly inside the box.
        cx = (x1 + x2) / 2 * 200 / page_w
        cy = (y1 + y2) / 2 * 200 / page_h
        rx = max(2.0, (x2 - x1) / 2 * 200 / page_w)
        ry = max(2.0, (y2 - y1) / 2 * 200 / page_h)
        for _ in range(rng.integers(1, 4)):
            ox, oy = rng.normal(0, rx / 3), rng.normal(0, ry / 3)
            mask |= (((xx - cx - ox) / rx) ** 2 + ((yy - cy - oy) / ry) ** 2 <= 1).astype(np.int32)

        masks.append(mask)

    return np.asarray(boxes, dtype=np.float32), np.asarray(masks, dtype=np.int32)


def main() -> None:
    upstream = load_upstream()
    modes = ["rect", "quad", "poly", "auto"]
    payload = {}

    cases = [
        (0, 900, 640, 5),
        (1, 1200, 900, 7),
        (2, 640, 1000, 4),
        (3, 800, 800, 9),
    ]

    for index, (seed, page_w, page_h, count) in enumerate(cases):
        boxes, masks = synthetic_case(seed, page_w, page_h, count)
        payload[f"boxes_{index}"] = boxes
        payload[f"masks_{index}"] = masks.astype(np.int32)
        payload[f"page_{index}"] = np.asarray([page_w, page_h], dtype=np.int32)

        # `scale_ratio` is the model input size over the page size; the function divides by 4
        # because the masks are a quarter of that resolution.
        scale_ratio = [800 / page_w, 800 / page_h]

        for mode in modes:
            polygons = upstream.extract_polygon_points_by_masks(
                boxes, masks, scale_ratio, mode)
            lengths = np.asarray([len(np.asarray(p)) for p in polygons], dtype=np.int32)
            flat = np.concatenate(
                [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polygons])
            payload[f"poly_{index}_{mode}_lengths"] = lengths
            payload[f"poly_{index}_{mode}"] = flat
            print(f"case {index} {mode}: {lengths.tolist()}")

    payload["count"] = np.asarray([len(cases)], dtype=np.int32)
    save_exact("layout_polygons.npz", **payload)


if __name__ == "__main__":
    main()
