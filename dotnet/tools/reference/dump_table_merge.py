"""Dumps upstream's decisions and output for merging tables across a page break.

`restructure_pages(merge_tables=True)` looks at the last table on one page and the first on the
next, decides whether they are one table split by the break, and if so appends the second's rows
to the first. `merge_table.py` is self-contained apart from BeautifulSoup, so it is exec'd here
and driven with stand-ins for the block and page objects it reads.
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
MERGE_TABLE = os.path.join(
    PADDLEX_DIR, "paddlex/inference/pipelines/layout_parsing/merge_table.py")


class Block:
    """The attributes `can_merge_tables` and `merge_tables_across_pages` read."""

    def __init__(self, label, content="", bbox=(0, 0, 100, 50)):
        self.label = label
        self.content = content
        self.bbox = list(bbox)
        self.global_block_id = 0
        self.global_group_id = 0


def load_upstream():
    module = types.ModuleType("merge_table")
    sys.modules["merge_table"] = module
    exec(compile(open(MERGE_TABLE).read(), "merge_table.py", "exec"), module.__dict__)
    return module


HEADER = "<tr><th>Name</th><th>Value</th></tr>"


def table(rows, header=True):
    body = (HEADER if header else "") + "".join(rows)
    return f"<table>{body}</table>"


def cases():
    """Pairs of pages, each described by its blocks, with the tables to be considered."""
    return [
        # A table continuing onto the next page, repeating its header.
        ("continues", [Block("table", table(["<tr><td>a</td><td>1</td></tr>"]), (10, 10, 210, 60))],
         [Block("table", table(["<tr><td>b</td><td>2</td></tr>"]), (12, 10, 210, 60))]),
        # Same, with a "Continued" caption before it on the second page.
        ("continued_caption",
         [Block("table", table(["<tr><td>a</td><td>1</td></tr>"]), (10, 10, 210, 60))],
         [Block("text", "Table 1 continued"),
          Block("table", table(["<tr><td>b</td><td>2</td></tr>"]), (10, 10, 210, 60))]),
        # A running head before it is fine.
        ("header_before",
         [Block("table", table(["<tr><td>a</td><td>1</td></tr>"]), (10, 10, 210, 60))],
         [Block("header", "Chapter 2"),
          Block("table", table(["<tr><td>b</td><td>2</td></tr>"]), (10, 10, 210, 60))]),
        # Ordinary prose after the first table means it had finished.
        ("prose_after",
         [Block("table", table(["<tr><td>a</td><td>1</td></tr>"]), (10, 10, 210, 60)),
          Block("text", "Some following prose.")],
         [Block("table", table(["<tr><td>b</td><td>2</td></tr>"]), (10, 10, 210, 60))]),
        # Different widths: not the same table.
        ("width_mismatch",
         [Block("table", table(["<tr><td>a</td><td>1</td></tr>"]), (10, 10, 210, 60))],
         [Block("table", table(["<tr><td>b</td><td>2</td></tr>"]), (10, 10, 400, 60))]),
        # Different column counts and no matching rows.
        ("column_mismatch",
         [Block("table", table(["<tr><td>a</td><td>1</td></tr>"]), (10, 10, 210, 60))],
         [Block("table",
                "<table><tr><th>A</th><th>B</th><th>C</th></tr><tr><td>x</td><td>y</td><td>z</td></tr></table>",
                (10, 10, 210, 60))]),
        # No repeated header on the second page.
        ("no_repeated_header",
         [Block("table", table(["<tr><td>a</td><td>1</td></tr>"]), (10, 10, 210, 60))],
         [Block("table", table(["<tr><td>b</td><td>2</td></tr>"], header=False), (10, 10, 210, 60))]),
        # Spanning cells, so the total column count has to account for colspan.
        ("colspan",
         [Block("table",
                "<table><tr><th colspan=\"2\">Both</th></tr><tr><td>a</td><td>1</td></tr></table>",
                (10, 10, 210, 60))],
         [Block("table",
                "<table><tr><th colspan=\"2\">Both</th></tr><tr><td>b</td><td>2</td></tr></table>",
                (10, 10, 210, 60))]),
        # Row spans, so the column count has to walk an occupancy grid rather than add up a row.
        ("rowspan",
         [Block("table",
                "<table><tr><th>Name</th><th>Value</th></tr>"
                "<tr><td rowspan=\"2\">a</td><td>1</td></tr><tr><td>2</td></tr></table>",
                (10, 10, 210, 60))],
         [Block("table",
                "<table><tr><th>Name</th><th>Value</th></tr>"
                "<tr><td rowspan=\"2\">b</td><td>3</td></tr><tr><td>4</td></tr></table>",
                (10, 10, 210, 60))]),
        # Full-width header text on one page, ASCII on the other: still the same header.
        ("full_width_header",
         [Block("table",
                "<table><tr><th>Ｎａｍｅ</th><th>Ｖａｌｕｅ</th></tr><tr><td>a</td><td>1</td></tr></table>",
                (10, 10, 210, 60))],
         [Block("table", table(["<tr><td>b</td><td>2</td></tr>"]), (10, 10, 210, 60))]),
        # A footnote after the first table does not interrupt it.
        ("footnote_after",
         [Block("table", table(["<tr><td>a</td><td>1</td></tr>"]), (10, 10, 210, 60)),
          Block("footnote", "1. A note.")],
         [Block("table", table(["<tr><td>b</td><td>2</td></tr>"]), (10, 10, 210, 60))]),
    ]


def main() -> None:
    upstream = load_upstream()
    records = []

    for name, prev_page, curr_page in cases():
        prev_block = next(b for b in reversed(prev_page) if b.label == "table")
        curr_block = next(b for b in curr_page if b.label == "table")

        can_merge, soup_prev, soup_curr = upstream.can_merge_tables(
            prev_page, prev_block, curr_page, curr_block)

        merged = (
            upstream.perform_table_merge(soup_prev, soup_curr) if can_merge else "")

        records.append({
            "name": name,
            "previous": [{"label": b.label, "content": b.content, "bbox": b.bbox}
                         for b in prev_page],
            "current": [{"label": b.label, "content": b.content, "bbox": b.bbox}
                        for b in curr_page],
            "can_merge": bool(can_merge),
            "merged": merged,
        })
        print(f"{name}: can_merge={can_merge}")

    payload = json.dumps(records, ensure_ascii=False, indent=1)
    save_exact("table_merge.npz",
               cases=np.frombuffer(payload.encode("utf-8"), dtype=np.uint8))


if __name__ == "__main__":
    main()
