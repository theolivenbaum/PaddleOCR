"""Dumps vision-tower activations for the C# parity tests.

Runs `PaddleOCRVisionModel` exactly the way `PaddleOCRVLForConditionalGeneration.forward` does
(`interpolate_pos_encoding=True`, `use_rope=True`, `window_size=-1`, `return_pooler_output=False`)
and records the embedding output, every encoder layer's output, the final layer norm and the
`mlp_AR` projection.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from _common import load_model, load_processor, save, synthetic_image  # noqa: E402

# Small enough to keep the fixture and the CPU forward pass quick, but not square, so a
# transposed grid would be caught.
IMAGE_WIDTH = 196
IMAGE_HEIGHT = 154


@torch.no_grad()
def main() -> None:
    processor = load_processor()
    model = load_model("float32")

    image = synthetic_image(IMAGE_WIDTH, IMAGE_HEIGHT, seed=7)
    batch = processor.image_processor.preprocess(images=[image], return_tensors="pt")

    pixel_values = batch["pixel_values"].to(torch.float32)
    grid = batch["image_grid_thw"]
    thw = tuple(int(v) for v in grid[0])
    numel = thw[0] * thw[1] * thw[2]

    position_ids = torch.arange(numel) % (thw[1] * thw[2])
    cu_seqlens = torch.tensor([0, numel], dtype=torch.int32)
    sample_indices = torch.zeros(numel, dtype=torch.int64)

    captured: dict[str, np.ndarray] = {}

    def record(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor.detach().reshape(-1, tensor.shape[-1]).float().numpy()

        return hook

    vision = model.visual.vision_model
    handles = [vision.embeddings.register_forward_hook(record("embeddings"))]
    for index, layer in enumerate(vision.encoder.layers):
        handles.append(layer.register_forward_hook(record(f"layer{index}")))

    try:
        outputs = model.visual(
            pixel_values=pixel_values.unsqueeze(0),
            image_grid_thw=[thw],
            position_ids=position_ids,
            vision_return_embed_list=True,
            interpolate_pos_encoding=True,
            sample_indices=sample_indices,
            cu_seqlens=cu_seqlens,
            return_pooler_output=False,
            use_rope=True,
            window_size=-1,
        )
    finally:
        for handle in handles:
            handle.remove()

    post_layernorm = outputs.last_hidden_state[0]
    projected = model.mlp_AR([post_layernorm], grid)[0]

    payload = {
        "pixel_values": pixel_values.numpy(),
        "grid": np.asarray(thw, dtype=np.int64),
        "post_layernorm": post_layernorm.detach().float().numpy(),
        "projector": projected.detach().float().numpy(),
        "layer_count": np.asarray([len(vision.encoder.layers)], dtype=np.int64),
        "source": np.asarray(image, dtype=np.uint8),
    }
    payload.update(captured)

    save("vision.npz", **payload)


if __name__ == "__main__":
    main()
