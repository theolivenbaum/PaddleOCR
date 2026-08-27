"""Dumps the markdown upstream renders for a fixed block list.

The pipeline's shipped markdown is the "pretty" variant: captions and images are wrapped in a
centred HTML div, images become width-scaled `<img>` tags, and tables get border and alignment
styling. `MarkdownConverter` and the format functions are exec'd from PaddleX directly, so the
reference is the real renderer; only `_build_handle_funcs_dict`'s wiring is restated here,
because it is a method on the Result class and reaches for its own settings dict.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from _common import save_exact  # noqa: E402

PADDLEX_DIR = os.environ.get("PADDLEX_DIR", "/home/user/ref/PaddleX")
CONVERTER_DIR = os.path.join(PADDLEX_DIR, "paddlex/inference/common/result/converter")


def load_upstream():
    """Exec's the two converter modules; both are self-contained apart from each other."""
    functions = types.ModuleType("markdown_format_funcs")
    functions.__dict__["__name__"] = "markdown_format_funcs"
    sys.modules["markdown_format_funcs"] = functions
    exec(
        compile(
            open(os.path.join(CONVERTER_DIR, "markdown_format_funcs.py")).read(),
            "markdown_format_funcs.py",
            "exec",
        ),
        functions.__dict__,
    )

    converter_source = open(os.path.join(CONVERTER_DIR, "markdown_converter.py")).read()
    converter_source = converter_source.replace(
        "from .markdown_format_funcs import merge_formula_and_number",
        "from markdown_format_funcs import merge_formula_and_number",
    )
    converter = types.ModuleType("markdown_converter")
    sys.modules["markdown_converter"] = converter
    exec(compile(converter_source, "markdown_converter.py", "exec"), converter.__dict__)

    return functions, converter


class Block:
    """The block protocol `MarkdownConverter` expects."""

    def __init__(self, label, content, bbox, image=None):
        self.label = label
        self.content = content
        self.bbox = bbox
        self.image = image


def handle_funcs(fn, settings, page_width):
    """Restatement of `PaddleOCRVLResult._build_handle_funcs_dict`."""
    pretty = settings.get("pretty", True)
    use_ocr_for_image_block = settings.get("use_ocr_for_image_block", False)
    use_seal_recognition = settings.get("use_seal_recognition", False)

    if pretty:
        text_func = lambda b: fn.format_centered_by_html(fn.format_text_plain(b))
        image_func = lambda b: fn.format_centered_by_html(
            fn.format_image_scaled_by_html(
                b, original_image_width=page_width, show_ocr_content=use_ocr_for_image_block),
            collapse_newlines=not use_ocr_for_image_block)
        seal_func = lambda b: fn.format_centered_by_html(
            fn.format_image_scaled_by_html(
                b, original_image_width=page_width, show_ocr_content=use_seal_recognition),
            collapse_newlines=not use_seal_recognition)
    else:
        text_func = lambda b: b.content
        image_func = lambda b: fn.format_image_plain(b, show_ocr_content=use_ocr_for_image_block)
        seal_func = lambda b: fn.format_image_plain(b, show_ocr_content=use_seal_recognition)

    chart_func = (
        fn.format_chart2html_table
        if settings.get("use_chart_recognition", False) else image_func)

    if not settings.get("use_layout_detection", True):
        seal_func = text_func

    if pretty:
        table_func = lambda b: "\n" + fn.format_table_center(b)
    else:
        table_func = lambda b: fn.simplify_table("\n" + b.content)

    result = fn.build_handle_funcs_dict(
        text_func=text_func,
        image_func=image_func,
        chart_func=chart_func,
        table_func=table_func,
        formula_func=lambda b: b.content,
        seal_func=seal_func)

    for label in settings.get("markdown_ignore_labels", []):
        result.pop(label, None)

    return result


def blocks():
    return [
        Block("doc_title", "A Document\nTitle", [40, 20, 760, 70]),
        Block("paragraph_title", "2.1 Method", [40, 90, 400, 120]),
        Block("text", "First line.\nSecond line.\n\nA new paragraph.", [40, 130, 760, 260]),
        Block("abstract", "Abstract This paper describes a thing.", [40, 270, 760, 330]),
        Block("content", "Chapter one-\nis here.\nChapter two.", [40, 340, 760, 400]),
        Block("figure_title", "Figure 1. A caption\nthat wraps.", [200, 410, 600, 440]),
        Block("image", "", [150, 450, 650, 700], image={"path": "imgs/img_0.png", "img": None}),
        Block("table", "<table><tr><th>A</th><td>1</td></tr></table>", [40, 710, 760, 800]),
        Block("display_formula", "$$x = y + 1$$", [200, 810, 600, 850]),
        Block("formula_number", "(1)", [700, 810, 760, 850]),
        Block("reference", "References\n[1] Someone.", [40, 860, 760, 940]),
        Block("algorithm", "\nAlgorithm 1\nstep\n", [40, 950, 760, 1010]),
        Block("chart", "A|B\n1|2\n3|4", [150, 1020, 650, 1200],
              image={"path": "imgs/chart_0.png", "img": None}),
        Block("seal", "A seal", [600, 1210, 760, 1300],
              image={"path": "imgs/seal_0.png", "img": None}),
        Block("vertical_text", "Side\nnote", [10, 400, 35, 700]),
        Block("header", "Running head", [40, 0, 760, 18]),
        Block("footer", "Page 3", [40, 1310, 760, 1330]),
        Block("number", "3", [740, 1310, 760, 1330]),
        Block("aside_text", "Margin note", [770, 400, 800, 700]),
        Block("footnote", "1. A footnote.", [40, 1290, 760, 1310]),
        Block("vision_footnote", "Source: somewhere", [150, 700, 650, 720]),
        Block("reference_content", "[2] Another.", [40, 940, 760, 960]),
        Block("inline_formula", "$a$", [300, 130, 320, 150]),
        Block("spotting", "Spotted text", [40, 130, 760, 260]),
    ]


CASES = {
    "pretty": {},
    "plain": {"pretty": False},
    "pretty_formula_number": {"show_formula_number": True},
    "pretty_ignore": {"markdown_ignore_labels": ["number", "footnote", "header", "footer",
                                                 "aside_text", "header_image", "footer_image"]},
    "pretty_chart": {"use_chart_recognition": True},
    "pretty_ocr_images": {"use_ocr_for_image_block": True, "use_seal_recognition": True},
}

PAGE_WIDTH = 800


def main() -> None:
    fn, converter = load_upstream()
    payload = {}

    for name, settings in CASES.items():
        funcs = handle_funcs(fn, settings, PAGE_WIDTH)
        result = converter.MarkdownConverter.convert(
            blocks(),
            handle_funcs_dict=funcs,
            show_formula_number=settings.get("show_formula_number", False),
            imgs_in_doc=None)
        # UTF-8 bytes rather than a numpy unicode array: the C# `.npy` reader speaks dtypes that
        # appear in tensors, and UCS-4 is not one of them.
        payload[name] = np.frombuffer(result["markdown_texts"].encode("utf-8"), dtype=np.uint8)
        print(f"--- {name} ({len(result['markdown_texts'])} chars)")

    payload["page_width"] = np.asarray([PAGE_WIDTH], dtype=np.int32)
    payload["cases"] = np.frombuffer("\n".join(CASES).encode("utf-8"), dtype=np.uint8)
    save_exact("markdown.npz", **payload)


if __name__ == "__main__":
    main()
