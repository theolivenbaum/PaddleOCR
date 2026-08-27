"""Runs the real PaddleOCR-VL-1.6 pipeline on a page and records what it produced.

This is not a fixture generator — it exists so the C# port's output can be put beside the
Python original's for the same page. The pipeline is the genuine `PaddleOCRVL16Pipeline` with
the shipped 1.6 configuration, pointed at the local model checkouts.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from _common import SAMPLE_LINES, table_page, text_image  # noqa: E402

SHIPPED_CONFIG = os.environ.get(
    "PADDLEOCR_VL_PIPELINE_CONFIG",
    "/home/user/ref/PaddleX/paddlex/configs/pipelines/PaddleOCR-VL-1.6.yaml")
OUTPUT = os.environ.get("COMPARE_DIR", "/tmp/compare")

# Where each sub-model was downloaded to. Pointing the shipped configuration at these keeps the
# comparison honest: the Python side runs the same weights the C# side loads, and every other
# setting is whatever ships with the 1.6 pipeline.
MODEL_DIRS = {
    "LayoutDetection": os.environ.get("PADDLEOCR_LAYOUT_DIR", "/home/user/ref/layout"),
    "VLRecognition": os.environ.get("PADDLEOCR_VL_DIR", "/home/user/ref/vl16"),
    "DocOrientationClassify": os.environ.get("PADDLEOCR_DOCORI_DIR", "/home/user/ref/docori"),
    "DocUnwarping": os.environ.get("PADDLEOCR_UVDOC_DIR", "/home/user/ref/uvdoc"),
}


def local_config() -> str:
    """Rewrites the shipped 1.6 configuration to load the local checkouts, and returns its path."""
    import yaml

    with open(SHIPPED_CONFIG) as handle:
        config = yaml.safe_load(handle)

    def point_at_local(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, dict) and key in MODEL_DIRS:
                    value["model_dir"] = MODEL_DIRS[key]
                point_at_local(value)

    point_at_local(config)

    # The queued path spreads the stages over worker threads, which on a CPU-only box only makes
    # the timings harder to read. Nothing else is changed.
    config["use_queues"] = False

    path = os.path.join(OUTPUT, "pipeline.yaml")
    with open(path, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def pages():
    """The pages both implementations are asked to parse."""
    report = text_image(
        [
            "Pure C# PaddleOCR-VL",
            "",
            "1. Introduction",
            "This page is rendered so the pipeline has a real",
            "document to parse: a title, a numbered heading,",
            "body text, and the paragraph below.",
            "",
            "The quick brown fox jumps over the lazy dog. 0123456789",
        ],
        width=980,
        font_size=30,
    )
    return [("report", report), ("benchmark", table_page()), ("lines", text_image(SAMPLE_LINES))]


def main() -> None:
    from paddlex import create_pipeline

    os.makedirs(OUTPUT, exist_ok=True)
    pipeline = create_pipeline(pipeline=local_config(), device="cpu")

    summary = []

    for name, page in pages():
        path = os.path.join(OUTPUT, f"{name}.png")
        page.save(path)

        started = time.time()
        results = list(pipeline.predict(path))
        elapsed = time.time() - started

        result = results[0]
        markdown = result.markdown["markdown_texts"]
        blocks = result.json["res"]["parsing_res_list"]

        with open(os.path.join(OUTPUT, f"{name}.python.md"), "w") as handle:
            handle.write(markdown)

        with open(os.path.join(OUTPUT, f"{name}.python.json"), "w") as handle:
            json.dump(blocks, handle, ensure_ascii=False, indent=1, default=str)

        summary.append({"name": name, "seconds": round(elapsed, 1), "blocks": len(blocks)})
        print(f"{name}: {len(blocks)} blocks in {elapsed:.1f}s")
        for block in blocks:
            content = str(block.get("block_content", ""))
            print(f"   {block['block_label']:16} {block['block_bbox']} {content[:60]!r}")

    with open(os.path.join(OUTPUT, "python_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1)


if __name__ == "__main__":
    main()
