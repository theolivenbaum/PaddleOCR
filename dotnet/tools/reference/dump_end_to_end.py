"""Runs the upstream PaddleOCR-VL model over layout crops of a page and records the result.

The layout boxes come from the Paddle inference graph, exactly as the pipeline produces them; each
crop is then recognised with the Hugging Face model. The C# end-to-end test replays the same page
and compares block labels, boxes and recognised text.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from _common import load_model, load_processor, save, text_image  # noqa: E402

LAYOUT_DIR = os.environ.get("PP_DOCLAYOUT_V3_DIR", "/home/user/ref/layout")
TARGET = 800
THRESHOLD = 0.3

PAGE_LINES = [
    "Pure C# PaddleOCR-VL",
    "",
    "1. Introduction",
    "This page is rendered so the end-to-end pipeline has",
    "a real document to parse: a title, a numbered heading,",
    "body text, and the paragraph below.",
    "",
    "The quick brown fox jumps over the lazy dog. 0123456789",
]

LABELS = [
    "abstract", "algorithm", "aside_text", "chart", "content", "display_formula", "doc_title",
    "figure_title", "footer", "footer_image", "footnote", "formula_number", "header",
    "header_image", "image", "inline_formula", "number", "paragraph_title", "reference",
    "reference_content", "seal", "table", "text", "vertical_text", "vision_footnote",
]

PROMPTS = {
    "table": "Table Recognition:",
    "display_formula": "Formula Recognition:",
    "inline_formula": "Formula Recognition:",
}


def detect(rgb: np.ndarray):
    import cv2
    import paddle
    from paddle.inference import Config, create_predictor

    paddle.set_device("cpu")
    config = Config(
        os.path.join(LAYOUT_DIR, "inference.json"),
        os.path.join(LAYOUT_DIR, "inference.pdiparams"),
    )
    config.disable_gpu()
    config.switch_ir_optim(False)
    config.disable_glog_info()
    if hasattr(config, "disable_mkldnn"):
        config.disable_mkldnn()
    predictor = create_predictor(config)

    resized = cv2.resize(rgb, (TARGET, TARGET), interpolation=cv2.INTER_CUBIC)
    chw = np.ascontiguousarray((resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])

    feed = {
        "image": chw,
        "scale_factor": np.asarray([[1.0, 1.0]], dtype=np.float32),
        "im_shape": np.asarray([[rgb.shape[0], rgb.shape[1]]], dtype=np.float32),
    }
    for name in predictor.get_input_names():
        handle = predictor.get_input_handle(name)
        handle.reshape(feed[name].shape)
        handle.copy_from_cpu(feed[name])
    predictor.run()

    raw = predictor.get_output_handle(predictor.get_output_names()[0]).copy_to_cpu()
    keep = raw[(raw[:, 1] > THRESHOLD) & (raw[:, 0] > -1)]
    order = np.argsort(keep[:, 6]) if keep.shape[1] > 6 else np.arange(len(keep))
    return keep[order]


def main() -> None:
    import torch
    from PIL import Image

    page = text_image(PAGE_LINES, width=980, font_size=30)
    rgb = np.asarray(page, dtype=np.uint8)

    boxes = detect(rgb)
    print(f"{len(boxes)} blocks above {THRESHOLD}")

    processor = load_processor()
    model = load_model("float32")

    contents = []
    records = []

    with torch.no_grad():
        for row in boxes:
            label = LABELS[int(row[0])]
            x1, y1, x2, y2 = (int(np.floor(row[2])), int(np.floor(row[3])),
                              int(np.ceil(row[4])), int(np.ceil(row[5])))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(rgb.shape[1], x2), min(rgb.shape[0], y2)
            crop = Image.fromarray(rgb[y1:y2, x1:x2])

            prompt = PROMPTS.get(label, "OCR:")
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            batch = processor(images=[crop], text=[text], return_tensors="pt")

            generated = model.generate(**batch, max_new_tokens=512, do_sample=False, use_cache=True)
            new_tokens = generated[0, batch["input_ids"].shape[1]:]
            content = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

            print(f"  {label} [{x1},{y1},{x2},{y2}] -> {content!r}")
            contents.append(content)
            records.append([int(row[0]), float(row[1]), x1, y1, x2, y2, int(row[6])])

    payload = {
        "source": rgb,
        "boxes": np.asarray(records, dtype=np.float32),
        "block_count": np.asarray([len(records)], dtype=np.int64),
    }
    for index, content in enumerate(contents):
        payload[f"content{index}"] = np.frombuffer(content.encode("utf-8"), dtype=np.uint8)

    save("end_to_end.npz", **payload)


if __name__ == "__main__":
    main()
