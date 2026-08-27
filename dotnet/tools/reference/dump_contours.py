"""Dumps OpenCV contour results for a set of masks.

`mask2polygon` in PaddleX leans on four OpenCV calls — `findContours`, `contourArea`,
`arcLength` and `approxPolyDP`. Each is reproduced in C#, so each needs a reference. The masks
below mix hand-written shapes that pin down the tracing conventions (diagonal steps, holes,
single pixels, touching corners) with random blobs that exercise the general case.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from _common import save_exact  # noqa: E402


def handwritten() -> list[np.ndarray]:
    shapes = [
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]],
        [[1, 0], [0, 1]],
        [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]],
        [[1]],
        [[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 1, 1, 1], [0, 0, 0, 1], [1, 1, 0, 1], [1, 1, 1, 1]],
    ]
    return [np.asarray(s, dtype=np.uint8) for s in shapes]


def blobs() -> list[np.ndarray]:
    """Random masks in the shape `mask2polygon` actually sees: a resized per-box crop."""
    out = []
    for seed, (h, w) in enumerate([(24, 40), (60, 45), (17, 90), (80, 80), (33, 21)]):
        rng = np.random.default_rng(seed + 100)
        mask = np.zeros((h, w), dtype=np.uint8)

        # A few overlapping filled ellipses, then a threshold, which is close to what a
        # detector's mask head produces once it is resized to the box.
        yy, xx = np.mgrid[0:h, 0:w]
        for _ in range(3):
            cy, cx = rng.integers(0, h), rng.integers(0, w)
            ry, rx = rng.integers(h // 6 + 1, h // 2 + 2), rng.integers(w // 6 + 1, w // 2 + 2)
            mask |= (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1).astype(np.uint8)

        out.append(mask)
    return out


def main() -> None:
    import cv2

    payload = {}
    masks = handwritten() + blobs()

    for index, mask in enumerate(masks):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        payload[f"mask_{index}"] = mask

        # Contours are stored flattened with a length per contour, since they vary in size.
        lengths = []
        points = []
        areas = []
        arcs = []
        approximations = []
        approximation_lengths = []

        # `mask2polygon` sorts by area and simplifies with 0.004 * the perimeter.
        for contour in contours:
            flat = contour.reshape(-1, 2)
            lengths.append(len(flat))
            points.append(flat)
            areas.append(cv2.contourArea(contour))
            arc = cv2.arcLength(contour, True)
            arcs.append(arc)
            approximate = cv2.approxPolyDP(contour, 0.004 * arc, True).reshape(-1, 2)
            approximation_lengths.append(len(approximate))
            approximations.append(approximate)

        payload[f"lengths_{index}"] = np.asarray(lengths, dtype=np.int32)
        payload[f"points_{index}"] = (
            np.concatenate(points).astype(np.int32) if points else np.zeros((0, 2), np.int32))
        payload[f"areas_{index}"] = np.asarray(areas, dtype=np.float64)
        payload[f"arcs_{index}"] = np.asarray(arcs, dtype=np.float64)
        payload[f"approx_lengths_{index}"] = np.asarray(approximation_lengths, dtype=np.int32)
        payload[f"approx_{index}"] = (
            np.concatenate(approximations).astype(np.int32)
            if approximations else np.zeros((0, 2), np.int32))

        print(f"mask {index}: {mask.shape} -> {len(contours)} contours, "
              f"{[len(p) for p in points]} points -> {approximation_lengths} after approx")

    payload["count"] = np.asarray([len(masks)], dtype=np.int32)
    save_exact("contours.npz", **payload)


if __name__ == "__main__":
    main()
