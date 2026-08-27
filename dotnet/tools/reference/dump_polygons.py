"""Dumps Shapely and OpenCV results for the polygon geometry the layout stage uses.

`_normalize_layout_polygon` compares a region's polygon against its bounding rectangle and its
minimum-area quad, and `filter_overlap_boxes` compares two regions' polygons — all through
`calculate_polygon_overlap_ratio`, which is Shapely, and `convert_polygon_to_quad`, which is
`cv2.minAreaRect`. Both are reproduced in C#, so both need a reference.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from _common import save_exact  # noqa: E402


def shapes() -> list[np.ndarray]:
    """A mix of convex, concave, rotated and degenerate outlines."""
    out = [
        [[0, 0], [10, 0], [10, 8], [0, 8]],
        [[2, 1], [9, 3], [7, 9], [1, 6]],
        [[0, 0], [10, 0], [10, 10], [5, 4], [0, 10]],                 # concave notch
        [[0, 0], [6, 2], [12, 0], [12, 6], [6, 4], [0, 6]],           # two notches
        [[3, 0], [6, 3], [3, 6], [0, 3]],                             # diamond
        [[0, 0], [1, 0], [1, 1], [0, 1]],                             # unit square
        [[4, 4], [14, 4], [14, 12], [4, 12]],
        [[-5, -5], [5, -5], [5, 5], [-5, 5]],
        [[0, 0], [20, 1], [20, 3], [0, 2]],                           # thin, slightly rotated
        [[0, 0], [3, 0], [3, 3], [6, 3], [6, 6], [0, 6]],             # L
    ]

    rng = np.random.default_rng(7)
    for _ in range(8):
        n = int(rng.integers(5, 12))
        angles = np.sort(rng.uniform(0, 2 * np.pi, n))
        radii = rng.uniform(3, 14, n)
        centre = rng.uniform(-6, 20, 2)
        pts = np.stack(
            [centre[0] + radii * np.cos(angles), centre[1] + radii * np.sin(angles)], axis=1)
        out.append(pts)

    # float32, because that is the dtype the pipeline's polygons actually carry:
    # `_rect_from_box` and `convert_polygon_to_quad` both return float32, and a contour's
    # integer points widen to it. Comparing against float64 references would measure the
    # conversion rather than the geometry.
    return [np.asarray(p, dtype=np.float32) for p in out]


def main() -> None:
    import cv2
    from shapely.geometry import Polygon

    def valid(points):
        poly = Polygon(points)
        return poly if poly.is_valid else poly.buffer(0)

    polygons = shapes()
    payload = {}

    for index, polygon in enumerate(polygons):
        payload[f"poly_{index}"] = polygon

        quad = cv2.boxPoints(cv2.minAreaRect(polygon.astype(np.float32)))
        centre = quad.mean(axis=0)
        order = np.argsort(np.arctan2(quad[:, 1] - centre[1], quad[:, 0] - centre[0]))
        quad = quad[order]
        quad = np.roll(quad, -int(np.argmin(quad[:, 0] + quad[:, 1])), axis=0)
        payload[f"quad_{index}"] = quad.astype(np.float32)
        payload[f"area_{index}"] = np.asarray([valid(polygon).area], dtype=np.float64)

    # Every ordered pair, in all three ratio modes.
    count = len(polygons)
    ratios = np.zeros((count, count, 3), dtype=np.float64)
    intersections = np.zeros((count, count), dtype=np.float64)

    for i in range(count):
        for j in range(count):
            a, b = valid(polygons[i]), valid(polygons[j])
            inter = a.intersection(b).area
            union = a.union(b).area
            intersections[i, j] = inter
            ratios[i, j, 0] = inter / union if union else 0.0
            ratios[i, j, 1] = inter / min(a.area, b.area) if min(a.area, b.area) else 0.0
            ratios[i, j, 2] = inter / max(a.area, b.area) if max(a.area, b.area) else 0.0

    # `crop_by_boxes` masks a region's crop with cv2.fillPoly, so the raster needs a reference
    # too. Integer outlines, because that is what the crop path passes.
    fill_w, fill_h = 40, 32
    for index, polygon in enumerate(polygons):
        pts = np.round(polygon).astype(np.int32)
        canvas = np.zeros((fill_h, fill_w), dtype=np.uint8)
        cv2.fillPoly(canvas, [pts.reshape(-1, 1, 2)], 1)
        payload[f"fill_{index}"] = canvas
        payload[f"fillpts_{index}"] = pts

    payload["fill_size"] = np.asarray([fill_w, fill_h], dtype=np.int32)
    payload["ratios"] = ratios
    payload["intersections"] = intersections
    payload["count"] = np.asarray([count], dtype=np.int32)

    print(f"{count} polygons; intersection areas from {intersections.min():.3f} "
          f"to {intersections.max():.3f}")
    save_exact("polygons.npz", **payload)


if __name__ == "__main__":
    main()
