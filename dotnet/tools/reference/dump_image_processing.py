"""Dumps `PaddleOCRVLImageProcessor` inputs and outputs for the C# imaging parity tests.

Reference: `image_processing_paddleocr_vl.py` in the PaddleOCR-VL-1.6 checkpoint —
`smart_resize` + PIL bicubic resize + rescale + normalize + patchify.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from _common import load_processor, save, synthetic_image  # noqa: E402

# (width, height) pairs chosen to exercise: downscale, upscale to min_pixels, near-square,
# extreme aspect ratio, and an already-aligned size.
CASES = [
    (613, 457),
    (1920, 1080),
    (128, 96),
    (2048, 137),
    (280, 280),
    (57, 941),
]


def main() -> None:
    processor = load_processor().image_processor

    payload = {}
    for index, (width, height) in enumerate(CASES):
        image = synthetic_image(width, height, seed=index)
        rgb = np.asarray(image, dtype=np.uint8)

        out = processor.preprocess(images=[image], return_tensors="np")
        grid = np.asarray(out["image_grid_thw"], dtype=np.int64)[0]

        payload[f"case{index}_source"] = rgb
        payload[f"case{index}_pixel_values"] = np.asarray(out["pixel_values"], dtype=np.float32)
        payload[f"case{index}_grid"] = grid

        # Also dump the intermediate resized uint8 image so a resize mismatch is diagnosable
        # without wading through the normalised patches.
        from transformers.image_transforms import resize as hf_resize
        from transformers.image_utils import ChannelDimension, PILImageResampling

        target_h = int(grid[1]) * processor.patch_size
        target_w = int(grid[2]) * processor.patch_size
        resized = hf_resize(
            rgb,
            size=(target_h, target_w),
            resample=PILImageResampling.BICUBIC,
            input_data_format=ChannelDimension.LAST,
        )
        payload[f"case{index}_resized"] = np.asarray(resized, dtype=np.uint8)

    payload["case_count"] = np.asarray([len(CASES)], dtype=np.int64)
    save("image_processing.npz", **payload)


if __name__ == "__main__":
    main()
