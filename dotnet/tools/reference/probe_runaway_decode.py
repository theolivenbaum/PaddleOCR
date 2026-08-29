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

It prints whether generation stopped on the stop token or ran to the cap, and writes the decoded
text beside the image. A run that ends "stopped on EOS: False" is the failure this is looking for.
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("max_new_tokens", nargs="?", type=int, default=UPSTREAM_MAX_NEW_TOKENS)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--box", help="crop as left,top,right,bottom in page pixels")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    parser.add_argument(
        "--model",
        default=os.environ.get("PADDLEOCR_VL_DIR", "PaddlePaddle/PaddleOCR-VL-1.6"),
        help="checkpoint directory, or a Hugging Face repo id")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    if args.box:
        image = image.crop(tuple(int(v) for v in args.box.split(",")))
    print(f"crop {image.width}x{image.height}", flush=True)

    torch.set_num_threads(args.threads)

    started = time.time()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32).eval()
    print(f"loaded in {time.time() - started:.0f}s", flush=True)

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[image], text=[chat], return_tensors="pt")

    prompt_tokens = inputs["input_ids"].shape[1]
    eos = model.generation_config.eos_token_id
    print(f"prompt tokens: {prompt_tokens}, eos_token_id: {eos}", flush=True)

    started = time.time()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,       # the local backend is greedy and drops the sampling knobs
            use_cache=True,
        )
    elapsed = time.time() - started

    generated = output[0][prompt_tokens:]
    count = len(generated)
    hit_eos = bool((generated == eos).any())

    print(f"\ngenerated {count} tokens in {elapsed:.0f}s "
          f"({elapsed / max(1, count) * 1000:.0f} ms/token)")
    print(f"stopped on EOS: {hit_eos}   ran to the {args.max_new_tokens}-token cap: {not hit_eos}")

    decoded = processor.tokenizer.decode(generated, skip_special_tokens=True)
    destination = os.path.splitext(args.image)[0] + ".upstream.txt"
    with open(destination, "w") as handle:
        handle.write(decoded)

    print(f"{len(decoded)} chars -> {destination}")
    print(f"head: {decoded[:160]!r}")
    print(f"tail: {decoded[-160:]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
