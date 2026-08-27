"""Dumps reference outputs for the two document pre-processing models.

Both ship as Paddle inference graphs, so this records the exact input tensors and the fetched
outputs; the C# tests replay them through the graph interpreter.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from _common import SAMPLE_LINES, save, synthetic_image, text_image  # noqa: E402

ORIENTATION_DIR = os.environ.get("PP_LCNET_DOC_ORI_DIR", "/home/user/ref/docori")
UNWARP_DIR = os.environ.get("UVDOC_DIR", "/home/user/ref/uvdoc")

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def predictor(directory):
    import paddle
    from paddle.inference import Config, create_predictor

    paddle.set_device("cpu")
    config = Config(
        os.path.join(directory, "inference.json"),
        os.path.join(directory, "inference.pdiparams"),
    )
    config.disable_gpu()
    config.switch_ir_optim(False)
    config.disable_glog_info()
    if hasattr(config, "disable_mkldnn"):
        config.disable_mkldnn()
    return create_predictor(config)


def run(engine, feed):
    for name in engine.get_input_names():
        handle = engine.get_input_handle(name)
        handle.reshape(feed[name].shape)
        handle.copy_from_cpu(feed[name])
    engine.run()
    return [engine.get_output_handle(name).copy_to_cpu() for name in engine.get_output_names()]


def orientation_input(rgb: np.ndarray) -> np.ndarray:
    """Resize the short side to 256, centre-crop 224, scale and normalise — from inference.yml."""
    import cv2

    height, width = rgb.shape[:2]
    scale = 256 / min(width, height)
    resized = cv2.resize(
        rgb,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )

    left = (resized.shape[1] - 224) // 2
    top = (resized.shape[0] - 224) // 2
    crop = resized[top:top + 224, left:left + 224]

    scaled = (crop.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(scaled.transpose(2, 0, 1)[None])


def main() -> None:
    payload = {}

    page = text_image(SAMPLE_LINES, width=760, font_size=28)
    upright = np.asarray(page, dtype=np.uint8)
    rotated = np.ascontiguousarray(np.rot90(upright, k=-1))

    engine = predictor(ORIENTATION_DIR)
    for name, image in (("upright", upright), ("rotated", rotated)):
        tensor = orientation_input(image)
        outputs = run(engine, {"x": tensor})
        payload[f"ori_{name}_source"] = image
        payload[f"ori_{name}_input"] = tensor
        payload[f"ori_{name}_logits"] = outputs[0]
        print(f"orientation/{name}: {np.round(outputs[0][0], 4)} -> class {int(outputs[0][0].argmax())}")

    warped = np.asarray(synthetic_image(256, 192, seed=5), dtype=np.uint8)
    tensor = np.ascontiguousarray((warped.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
    engine = predictor(UNWARP_DIR)
    outputs = run(engine, {"image": tensor})
    payload["uvdoc_source"] = warped
    payload["uvdoc_input"] = tensor
    payload["uvdoc_output"] = outputs[0]
    print(f"uvdoc: {outputs[0].shape}")

    save("preprocessing.npz", **payload)


if __name__ == "__main__":
    main()
