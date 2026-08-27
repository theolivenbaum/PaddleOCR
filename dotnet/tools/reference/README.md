# Reference tooling

These scripts run the **upstream Python implementations** and dump their inputs and outputs as
`.npz` fixtures. The C# tests under `dotnet/tests/PaddleOcrSharp.Tests` read the same fixtures and
assert the port produces the same numbers.

Fixtures are large and are **not committed**. Tests that need one are skipped, not failed, when it
is missing, so `dotnet test` works on a clean clone.

## Environment

```bash
pip install "torch" "transformers==4.55.0" "torchvision==0.24.1" \
            pillow numpy safetensors sentencepiece protobuf einops \
            paddlepaddle opencv-python-headless
```

Notes on the pins:

- **`transformers` 4.55.x** is what the checkpoint's `config.json` declares. Newer releases moved
  `create_causal_mask`'s keyword and renamed `torch_dtype`, and the shipped remote code has not
  followed.
- **`torchvision`** must match the installed `torch` (0.24.1 for torch 2.9.x). The checkpoint's
  image processor imports it even though the code path we exercise does not use it.
- **`paddlepaddle`** is only needed for the layout fixtures; it runs the shipped inference graph so
  the C# graph interpreter can be compared against it.
- **`opencv-python-headless`** provides the `cv2.resize` the layout preprocessing uses.

### Patch needed in the downloaded checkpoint

`modeling_paddleocr_vl.py` as shipped calls `create_causal_mask(config=…, inputs_embeds=…)`, but
every `transformers` 4.55.x names that parameter `input_embeds`. Rename it in your local copy:

```bash
sed -i 's/inputs_embeds=inputs_embeds,/input_embeds=inputs_embeds,/' \
    "$PADDLEOCR_VL_DIR/modeling_paddleocr_vl.py"
rm -rf ~/.cache/huggingface/modules/transformers_modules
```

This is a keyword rename only; it changes no arithmetic.

## Paths

| Variable | Meaning | Default |
| --- | --- | --- |
| `PADDLEOCR_VL_DIR` | PaddleOCR-VL-1.6 checkpoint directory | `/home/user/ref/vl16` |
| `PP_DOCLAYOUT_V3_DIR` | PP-DocLayoutV3 inference program directory | `/home/user/ref/layout` |
| `PADDLEOCR_SHARP_FIXTURES` | Where fixtures are written | `dotnet/artifacts/fixtures` |

The C# tests read the same variables, so pointing all three at the same places makes the two sides
agree without further configuration.

## Scripts

| Script | Fixture | What it covers |
| --- | --- | --- |
| `dump_image_processing.py` | `image_processing.npz` | `smart_resize`, PIL bicubic, rescale/normalize, patchify, for six image shapes |
| `dump_vision.py` | `vision.npz` | vision-tower embeddings, all 27 encoder layers, the final norm and the `mlp_AR` projection |
| `dump_language.py` | `language.npz` | prompt ids, `get_rope_index`, decoder layers, prefill logits and 24 greedy steps |
| `dump_tokenizer.py` | `tokenizer.json` | encode and decode over a mixed-script corpus |
| `dump_layout.py` | `layout.npz` | `cv2.resize` output and every fetched tensor of the PP-DocLayoutV3 graph |
| `dump_end_to_end.py` | `end_to_end.npz` | layout boxes plus per-block recognised text for a whole page |

Run them from this directory:

```bash
for script in dump_*.py; do python3 "$script"; done
```

## Debugging a mismatch

Both model families can dump their intermediates:

- `PaddleOcrSharp.Models.Vision.VisionTrace` records the vision tower's per-layer hidden states and
  the projector's input and output. `VisionTower.RunEncoder` and `Project` take one.
- `PaddleOcrSharp.Models.Paddle.IPirTrace` receives every operator's results as the layout graph
  runs, together with the operator's `struct_name`, which is the module path Paddle recorded
  (`/PPHGNetV2/HG_Stage_2/…`). That makes it possible to bisect a divergence to a single Paddle
  module.

The Python side captures the equivalent states with `register_forward_hook`, as
`dump_vision.py` and `dump_language.py` do.
