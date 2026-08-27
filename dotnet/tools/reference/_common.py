"""Shared helpers for the reference dumpers.

These scripts load the real Hugging Face PaddleOCR-VL-1.6 checkpoint, run one stage of the
pipeline, and write inputs plus outputs to a ``.npz`` file that the C# parity tests read back.

The model is loaded in ``float32`` on purpose: the shipped weights are bfloat16, widening them
to float32 is lossless, and it makes the reference computation match the C# port's arithmetic
(float32 activations, float32 accumulation) so tolerances can stay tight.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np

MODEL_DIR = os.environ.get("PADDLEOCR_VL_DIR", "/home/user/ref/vl16")
FIXTURE_DIR = pathlib.Path(
    os.environ.get(
        "PADDLEOCR_SHARP_FIXTURES",
        pathlib.Path(__file__).resolve().parents[2] / "artifacts" / "fixtures",
    )
)


def fixture_path(name: str) -> pathlib.Path:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURE_DIR / name


def save(name: str, **arrays) -> pathlib.Path:
    path = fixture_path(name)
    cleaned = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.dtype == np.float64:
            array = array.astype(np.float32)
        if array.dtype == np.int64:
            array = array.astype(np.int64)
        cleaned[key] = np.ascontiguousarray(array)
    np.savez(path, **cleaned)
    print(f"wrote {path} ({', '.join(f'{k}{list(v.shape)}' for k, v in cleaned.items())})")
    return path


def load_model(dtype: str = "float32"):
    import torch
    from transformers import AutoModelForCausalLM

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]

    # `torch_dtype` was renamed to `dtype` in transformers 4.56; accept either.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        )
    model.eval()
    return model


def load_processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)


def synthetic_image(width: int, height: int, seed: int = 0):
    """A deterministic RGB image, used so fixtures do not depend on sample assets."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)

    # Add smooth structure so resampling differences actually show up in the comparison.
    yy, xx = np.mgrid[0:height, 0:width]
    ramp = (np.sin(xx / 17.0) * 60 + np.cos(yy / 11.0) * 60 + 128).astype(np.int32)
    blended = ((base.astype(np.int32) * 0.35) + (ramp[..., None] * 0.65)).clip(0, 255)
    return Image.fromarray(blended.astype(np.uint8), mode="RGB")


TEXT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def text_image(lines, width=760, font_size=28, margin=24):
    """Renders black text on white, so end-to-end runs have something real to read."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(TEXT_FONT, font_size)
    line_height = int(font_size * 1.45)
    height = margin * 2 + line_height * len(lines)

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        draw.text((margin, margin + index * line_height), line, fill=(0, 0, 0), font=font)
    return image


SAMPLE_LINES = [
    "PaddleOCR-VL pure C# port",
    "The quick brown fox jumps over",
    "the lazy dog. 0123456789",
]
