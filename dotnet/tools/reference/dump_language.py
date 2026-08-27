"""Dumps decoder activations, prefill logits and greedy decode steps for the C# parity tests.

Runs the full `PaddleOCRVLForConditionalGeneration` on one rendered text image so the fixture
covers prompt construction, the 3-D rope index, the image-feature scatter and the decoder itself.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from _common import SAMPLE_LINES, load_model, load_processor, save, text_image  # noqa: E402

PROMPT = "OCR:"
GREEDY_STEPS = 24


@torch.no_grad()
def main() -> None:
    processor = load_processor()
    model = load_model("float32")

    image = text_image(SAMPLE_LINES)
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
        }
    ]
    text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(images=[image], text=[text], return_tensors="pt")

    input_ids = batch["input_ids"]
    pixel_values = batch["pixel_values"].to(torch.float32)
    grid = batch["image_grid_thw"]

    position_ids, rope_delta = model.get_rope_index(input_ids, grid, None, None, batch["attention_mask"])

    captured: dict[str, np.ndarray] = {}

    def record(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor.detach().reshape(-1, tensor.shape[-1]).float().numpy()

        return hook

    handles = [
        model.model.layers[index].register_forward_hook(record(f"lm_layer{index}"))
        for index in (0, 1, len(model.model.layers) // 2, len(model.model.layers) - 1)
    ]
    handles.append(model.model.norm.register_forward_hook(record("lm_norm")))

    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=batch["attention_mask"],
            pixel_values=pixel_values,
            image_grid_thw=grid,
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()

    logits = outputs.logits[0, -1].detach().float().numpy()

    generated = model.generate(
        **batch,
        max_new_tokens=GREEDY_STEPS,
        do_sample=False,
        use_cache=True,
    )
    new_tokens = generated[0, input_ids.shape[1]:].detach().cpu().numpy().astype(np.int64)
    decoded = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    print("greedy:", repr(decoded))

    payload = {
        "source": np.asarray(image, dtype=np.uint8),
        "input_ids": input_ids[0].numpy().astype(np.int64),
        "grid": grid[0].numpy().astype(np.int64),
        "position_ids": position_ids[:, 0, :].numpy().astype(np.int64),
        "rope_delta": np.asarray([int(rope_delta[0, 0])], dtype=np.int64),
        "prefill_logits": logits,
        "greedy_tokens": new_tokens,
        "prompt": np.frombuffer(PROMPT.encode("utf-8"), dtype=np.uint8),
        "decoded": np.frombuffer(decoded.encode("utf-8"), dtype=np.uint8),
    }
    payload.update(captured)

    save("language.npz", **payload)


if __name__ == "__main__":
    main()
