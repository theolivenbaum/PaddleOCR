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

from _common import load_model, load_processor, save, table_page, text_image  # noqa: E402

LAYOUT_DIR = os.environ.get("PP_DOCLAYOUT_V3_DIR", "/home/user/ref/layout")
PADDLEX_DIR = os.environ.get("PADDLEX_DIR", "/home/user/ref/PaddleX")
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


def load_pipeline_helpers():
    """Loads the pipeline's post-processing helpers without importing all of PaddleX.

    `uilts.py` pulls in the whole inference stack when imported normally, but the OTSL conversion
    and the repetition guard are self-contained, so the relevant slice is executed on its own.
    """
    import itertools
    import re
    import types
    from collections import Counter
    from typing import Any, Dict, List, Tuple, Union

    import html
    from pydantic import BaseModel, computed_field, model_validator

    source = open(
        os.path.join(PADDLEX_DIR, "paddlex/inference/pipelines/paddleocr_vl/uilts.py"),
        encoding="utf-8",
    ).read()
    start = source.index("class TableCell(BaseModel):")
    end = source.index("def crop_margin(")

    module = types.ModuleType("paddleocr_vl_uilts_slice")
    sys.modules[module.__name__] = module

    namespace = module.__dict__
    namespace.update({
        "BaseModel": BaseModel,
        "computed_field": computed_field,
        "model_validator": model_validator,
        "itertools": itertools,
        "re": re,
        "html": html,
        "Counter": Counter,
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Tuple": Tuple,
        "Union": Union,
    })

    exec(compile(source[start:end], "uilts.py", "exec"), namespace)

    # The models were defined inside an exec'd namespace, so pydantic has to be told where to
    # resolve their annotations from before they can be instantiated.
    namespace["TableCell"].model_rebuild(_types_namespace=namespace)
    namespace["TableData"].model_rebuild(_types_namespace=namespace)

    return namespace["convert_otsl_to_html"], namespace["truncate_repetitive_content"]


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
    run("end_to_end.npz", text_image(PAGE_LINES, width=980, font_size=30))
    run("end_to_end_table.npz", table_page())


def run(fixture: str, page) -> None:
    import torch
    from PIL import Image

    rgb = np.asarray(page, dtype=np.uint8)

    boxes = detect(rgb)
    print(f"{len(boxes)} blocks above {THRESHOLD}")

    processor = load_processor()
    model = load_model("float32")
    convert_otsl_to_html, truncate_repetitive_content = load_pipeline_helpers()

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

            # Apply the same post-processing the pipeline does, so the fixture holds the
            # pipeline's block content rather than the model's raw output.
            content = truncate_repetitive_content(
                content, min_count=5000 if label == "table" else 50
            )
            if label == "table":
                html_str = convert_otsl_to_html(content)
                if html_str != "":
                    content = html_str

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

    save(fixture, **payload)


if __name__ == "__main__":
    main()
