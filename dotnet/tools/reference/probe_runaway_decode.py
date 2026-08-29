"""Asks whether upstream's own decode terminates on a crop that makes the port run away.

This is not a fixture generator — it settles a behavioural question. `GenerationOptions
.StopOnRepetition` ends a block once its output has fallen into a verbatim cycle, which is a
deliberate divergence from upstream, and the case for it rests on upstream having no such stop.
This script checks that directly: it loads the real checkpoint through `transformers` and calls
`generate` with the arguments PaddleX's local backend passes, on a crop the caller supplies.

What PaddleX passes is worth restating, because it is the whole argument
(`paddlex/inference/models/doc_vlm/predictor.py`, the `generate_kwargs` block): `max_new_tokens`,
defaulting to `PADDLEOCR_VL_MAX_NEW_TOKENS = 8192`, and nothing else. `repetition_penalty`,
`temperature` and `top_p` are each warned about and dropped for the local backend; the server
backends send only a temperature and a token cap. The checkpoint's `generation_config.json` sets
no `no_repeat_ngram_size` and no stopping criteria, and the model is a stock `GenerationMixin`.
So greedy decoding here stops on the end-of-sequence token or on the budget, and on nothing else.

Usage:

    python3 probe_runaway_decode.py <image> [max_new_tokens] [--prompt "OCR:"] [--box l,t,r,b]
    python3 probe_runaway_decode.py <image> [max_new_tokens] --blocks page.json

It prints whether generation stopped on the stop token or ran to the cap, and writes the decoded
text beside the image. A run that ends "stopped on EOS: False" is the failure this is looking for.

`--blocks` takes the JSON `paddleocr-sharp parse --format json` writes and runs every block of the
page, each with the instruction its label earns, so upstream's whole recognition stage can be timed
against the port's. Both sides then see the same blocks; layout detection and markdown assembly are
excluded from the comparison, which is fair because neither is where the time goes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: E402

# `PADDLEOCR_VL_MAX_NEW_TOKENS` in paddlex/inference/models/doc_vlm/constants.py.
UPSTREAM_MAX_NEW_TOKENS = 8192

# pipeline.py maps a block label to its instruction; a `text` block gets this one.
DEFAULT_PROMPT = "OCR:"


def prompt_for(label: str) -> str:
    """The instruction a block's label earns, as `BlockPrompt.For` ports it from pipeline.py."""
    if label == "table":
        return "Table Recognition:"
    if "formula" in label and label != "formula_number":
        return "Formula Recognition:"
    return DEFAULT_PROMPT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("max_new_tokens", nargs="?", type=int, default=UPSTREAM_MAX_NEW_TOKENS)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--box", help="crop as left,top,right,bottom in page pixels")
    parser.add_argument("--blocks", help="a `parse --format json` result; runs every block of it")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    parser.add_argument(
        "--model",
        default=os.environ.get("PADDLEOCR_VL_DIR", "PaddlePaddle/PaddleOCR-VL-1.6"),
        help="checkpoint directory, or a Hugging Face repo id")
    args = parser.parse_args()

    page = Image.open(args.image).convert("RGB")

    if args.blocks:
        with open(args.blocks) as handle:
            pages = json.load(handle)
        crops = [(block["label"], prompt_for(block["label"]), page.crop(tuple(block["bbox"])))
                 for block in pages[0]["blocks"]]
        print(f"{len(crops)} blocks", flush=True)
    else:
        crop = page.crop(tuple(int(v) for v in args.box.split(","))) if args.box else page
        crops = [("", args.prompt, crop)]
        print(f"crop {crop.width}x{crop.height}", flush=True)

    torch.set_num_threads(args.threads)

    started = time.time()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32).eval()
    print(f"loaded in {time.time() - started:.0f}s (excluded from the totals below)", flush=True)

    eos = model.generation_config.eos_token_id
    results = []
    page_started = time.time()

    for index, (label, instruction, crop) in enumerate(crops):
        messages = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=[crop], text=[chat], return_tensors="pt")
        prompt_tokens = inputs["input_ids"].shape[1]

        started = time.time()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,   # the local backend is greedy and drops the sampling knobs
                use_cache=True,
            )
        elapsed = time.time() - started

        generated = output[0][prompt_tokens:]
        hit_eos = bool((generated == eos).any())
        decoded = processor.tokenizer.decode(generated, skip_special_tokens=True)

        results.append({"index": index, "label": label, "prompt_tokens": prompt_tokens,
                        "generated": len(generated), "stopped_on_eos": hit_eos,
                        "seconds": elapsed, "text": decoded})

        ran_on = "" if hit_eos else f"   !! ran to the {args.max_new_tokens}-token cap"
        print(f"[{index:2d}] {label or 'crop':16} prompt={prompt_tokens:4d} "
              f"gen={len(generated):5d} {elapsed:7.1f}s "
              f"({elapsed / max(1, len(generated)) * 1000:.0f} ms/token){ran_on}", flush=True)

    total = time.time() - page_started
    runaway = [r for r in results if not r["stopped_on_eos"]]

    print(f"\n{len(results)} block(s) in {total:.0f}s, {sum(r['generated'] for r in results)} tokens")
    if runaway:
        spent = sum(r["seconds"] for r in runaway)
        print(f"{len(runaway)} never stopped: {spent:.0f}s, {spent / total * 100:.0f}% of the total")
    else:
        print("every block stopped on the stop token")

    destination = os.path.splitext(args.image)[0] + ".upstream.json"
    with open(destination, "w") as handle:
        json.dump(results, handle, indent=1)
    print(f"-> {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
