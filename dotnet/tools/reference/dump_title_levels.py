"""Dumps upstream's heading-level assignment for a document's paragraph titles.

`restructure_pages(relevel_titles=True)` decides each `paragraph_title`'s heading depth by
voting between three signals: the level implied by its numbering, the order its numbering style
first appeared, and a cluster of its text height. The module is self-contained apart from numpy
and scikit-learn, so it is exec'd here and driven with stand-in blocks.

The cluster map is dumped alongside the final levels so the C# side can be checked against the
levels with upstream's own clustering supplied, isolating the one part that cannot be reproduced
exactly — scikit-learn's KMeans.
"""

from __future__ import annotations

import json
import os
import sys
import types

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from _common import save_exact  # noqa: E402

PADDLEX_DIR = os.environ.get("PADDLEX_DIR", "/home/user/ref/PaddleX")
TITLE_LEVEL = os.path.join(
    PADDLEX_DIR, "paddlex/inference/pipelines/layout_parsing/title_level.py")


class Block:
    def __init__(self, label, content, bbox):
        self.label = label
        self.content = content
        self.bbox = list(bbox)
        self.title_level = None


def load_upstream():
    module = types.ModuleType("title_level")
    sys.modules["title_level"] = module
    exec(compile(open(TITLE_LEVEL).read(), "title_level.py", "exec"), module.__dict__)
    return module


def pages():
    """Headings covering every numbering style, the keyword list, and unnumbered titles."""

    def title(content, height, width=400):
        return Block("paragraph_title", content, (40, 100, 40 + width, 100 + height))

    return [
        [
            Block("doc_title", "A Study of Things", (40, 20, 700, 70)),
            title("Abstract", 30),
            title("1 Introduction", 30),
            title("1.1 Background", 24),
            title("1.2 Related Work", 24),
            title("1.2.1 Earlier Attempts", 20),
            Block("text", "Body text.", (40, 300, 700, 400)),
        ],
        [
            title("2 Method", 30),
            title("2.1 Setup", 24),
            title("(1) First step", 18),
            title("(2) Second step", 18),
            title("A. An appendix-style heading", 24),
            title("II. A roman heading", 30),
        ],
        [
            title("Results", 30),
            title("Discussion", 30),
            title("An unnumbered heading of ordinary size", 24),
            title("References", 30),
            title("附录", 30),
            title("第一章 总论", 30),
        ],
    ]


def main() -> None:
    upstream = load_upstream()
    blocks_by_page = pages()

    entries_before = []
    for page in blocks_by_page:
        for block in page:
            if block.label == "paragraph_title":
                height = upstream.get_title_height(block)
                symbol, level = upstream.get_symbol_and_level(block.content)
                entries_before.append({
                    "content": block.content,
                    "height": height,
                    "symbol": symbol,
                    "symbol_level": level,
                })

    cluster_entries = [{"height": e["height"]} for e in entries_before]
    cluster_map = upstream.cluster_global_heights(cluster_entries)

    upstream.assign_levels_to_parsing_res(blocks_by_page)

    records = {
        "entries": entries_before,
        "cluster_map": {str(k): int(v) for k, v in cluster_map.items()},
        "pages": [
            [
                {
                    "label": b.label,
                    "content": b.content,
                    "bbox": b.bbox,
                    "title_level": getattr(b, "title_level", None),
                }
                for b in page
            ]
            for page in blocks_by_page
        ],
    }

    for entry in entries_before:
        print(f"{entry['content'][:40]:42} h={entry['height']:3} "
              f"symbol={entry['symbol']} level={entry['symbol_level']}")
    print("cluster map:", records["cluster_map"])
    print("levels:", [b["title_level"] for page in records["pages"] for b in page
                      if b["label"] == "paragraph_title"])

    save_exact("title_levels.npz",
               data=np.frombuffer(
                   json.dumps(records, ensure_ascii=False).encode("utf-8"), dtype=np.uint8))


if __name__ == "__main__":
    main()
