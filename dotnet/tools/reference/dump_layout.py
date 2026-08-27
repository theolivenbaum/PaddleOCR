"""Dumps PP-DocLayoutV3 reference inputs and outputs for the C# graph-interpreter tests.

Runs the shipped Paddle inference program with the preprocessing the PaddleX object-detection
predictor applies for this model (bicubic resize to 800x800 via OpenCV, scale by 1/255, HWC->CHW)
and records both the preprocessed tensor and every fetched output.

Feeding the recorded tensor back into the C# interpreter isolates graph parity from resize
parity, which is checked separately.
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

LAYOUT_DIR = os.environ.get("PP_DOCLAYOUT_V3_DIR", "/home/user/ref/layout")
TARGET = 800


def preprocess(rgb: np.ndarray):
    import cv2

    resized = cv2.resize(rgb, (TARGET, TARGET), interpolation=cv2.INTER_CUBIC)
    scaled = resized.astype(np.float32) * (1.0 / 255.0)
    chw = scaled.transpose(2, 0, 1)[None]
    # The detector's post-process recovers the original page size as
    # `im_shape / scale_factor`, so feeding the true size with a unit scale factor makes the
    # fetched boxes land directly in original-image coordinates.
    scale_factors = np.asarray([[1.0, 1.0]], dtype=np.float32)
    image_shape = np.asarray([[rgb.shape[0], rgb.shape[1]]], dtype=np.float32)
    return resized, np.ascontiguousarray(chw), scale_factors, image_shape


def main() -> None:
    import paddle
    from paddle.inference import Config, create_predictor

    paddle.set_device("cpu")

    config = Config(
        os.path.join(LAYOUT_DIR, "inference.json"),
        os.path.join(LAYOUT_DIR, "inference.pdiparams"),
    )
    config.disable_gpu()
    config.switch_ir_optim(False)
    # oneDNN's PIR bridge cannot convert this program's double-array attributes.
    if hasattr(config, "disable_mkldnn"):
        config.disable_mkldnn()
    config.disable_glog_info()
    predictor = create_predictor(config)

    page = text_image(
        [
            "A Layout Detection Sample",
            "",
            "This paragraph exists so the detector has",
            "a title, body text and a table to find on",
            "the page rather than random noise.",
        ],
        width=900,
        font_size=30,
    )
    rgb = np.asarray(page, dtype=np.uint8)

    resized, chw, scale_factors, image_shape = preprocess(rgb)

    feed = {
        "image": chw,
        "scale_factor": scale_factors,
        "im_shape": image_shape,
    }
    for name in predictor.get_input_names():
        handle = predictor.get_input_handle(name)
        handle.reshape(feed[name].shape)
        handle.copy_from_cpu(feed[name])

    predictor.run()

    payload = {
        "source": rgb,
        "resized": resized,
        "image": chw,
        "scale_factor": scale_factors,
        "im_shape": image_shape,
    }

    for index, name in enumerate(predictor.get_output_names()):
        array = predictor.get_output_handle(name).copy_to_cpu()
        payload[f"output{index}"] = array
        print(f"{name}: {array.shape} {array.dtype}")
        if array.ndim == 2 and array.shape[1] == 7 and array.shape[0] > 0:
            keep = array[array[:, 1] > 0.3]
            print(f"  {len(keep)} detections above 0.3")
            for row in keep[:8]:
                print("   ", np.round(row, 3))

    payload["output_count"] = np.asarray([len(predictor.get_output_names())], dtype=np.int64)

    # A second, smaller case keeps the fixture useful for a quick regression run.
    small = np.asarray(synthetic_image(320, 240, seed=11), dtype=np.uint8)
    small_resized, small_chw, small_scale, small_shape = preprocess(small)
    payload["small_source"] = small
    payload["small_resized"] = small_resized
    payload["small_image"] = small_chw
    payload["small_scale_factor"] = small_scale
    payload["small_im_shape"] = small_shape

    save("layout.npz", **payload)


if __name__ == "__main__":
    main()
