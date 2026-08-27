"""Dumps tokenizer encodings for the C# parity tests.

The corpus mixes scripts, whitespace runs, digits, emoji (which exercise byte fallback) and the
prompt strings the pipeline actually sends, so a divergence in normalisation, merge ordering or
byte fallback shows up as a token-id mismatch.
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from _common import MODEL_DIR, fixture_path, save  # noqa: E402

CORPUS = [
    "OCR:",
    "Table Recognition:",
    "Formula Recognition:",
    "Chart Recognition:",
    "Seal Recognition:",
    "Spotting:",
    "Assistant:\n",
    "User: ",
    "<|begin_of_sentence|>User: <|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>OCR:\nAssistant:\n",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "  leading and trailing   spaces  ",
    "\n\ttabs\tand\nnewlines\n",
    "1234567890 3.14159 -42 1e-9",
    "PaddleOCR-VL 是百度开源的文档解析模型。",
    "日本語のテキストも処理できます。",
    "한국어 텍스트입니다.",
    "Тест на кириллице",
    "نص عربي للاختبار",
    "emoji: 🚀🔥✅ combining: éüñ",
    "<table><tr><td>a</td><td>b</td></tr></table>",
    r"$\frac{1}{2}\int_0^\infty e^{-x^2}dx$",
    "<fcel><ecel><nl><lcel><ucel><xcel>",
    "<|LOC_0|><|LOC_512|><|LOC_999|>",
    "mixed 中英文 mixed text 123 with punctuation, and: symbols!",
    "a" * 300,
    "ligature ﬁ and zero-width​space",
]


def main() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    encodings = []
    for text in CORPUS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        encodings.append(
            {
                "text": text,
                "ids": ids,
                "decoded": tokenizer.decode(ids, skip_special_tokens=False),
            }
        )

    path = fixture_path("tokenizer.json")
    path.write_text(json.dumps(encodings, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {path} ({len(encodings)} cases)")

    lengths = np.asarray([len(entry["ids"]) for entry in encodings], dtype=np.int64)
    save("tokenizer_lengths.npz", lengths=lengths)


if __name__ == "__main__":
    main()
