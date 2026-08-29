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
| `dump_end_to_end.py` | `end_to_end.npz`, `end_to_end_table.npz` | layout boxes plus per-block recognised text for a whole page, after the pipeline's own repetition truncation and OTSL→HTML conversion |
| `dump_preprocessing.py` | `preprocessing.npz` | orientation-classifier inputs and logits for an upright and a rotated page, and UVDoc's input and flattened output |
| `dump_contours.py` | `contours.npz` | `findContours`, `contourArea`, `arcLength` and `approxPolyDP` over 60 masks |
| `dump_polygons.py` | `polygons.npz` | Shapely areas and overlap ratios, `minAreaRect`, and `fillPoly` rasters |
| `dump_layout_polygons.py` | `layout_polygons.npz` | `extract_polygon_points_by_masks` for all four shape modes |
| `dump_markdown.py` | `markdown.npz` | `MarkdownConverter` over every label, in six settings combinations |
| `dump_table_merge.py` | `table_merge.npz` | `merge_table.py`'s decisions and merged HTML for eleven page pairs |
| `dump_title_levels.py` | `title_levels.npz` | `title_level.py`'s numbering styles, text heights, clustering and final levels |
| `dump_block_merge.py` | `block_merge.npz` | `merge_blocks` grouping, ordering and `merge_images` pixels over ten layouts |
| `dump_pipeline_helpers.py` | `pipeline_helpers.npz` | `filter_overlap_boxes`, `convert_otsl_to_html`, `truncate_repetitive_content` and `crop_margin` over thirty-eight cases |

Run them from this directory:

```bash
for script in dump_*.py; do python3 "$script"; done
```

Several of these exec a slice of PaddleX rather than importing it: the checkout under
`/home/user/ref/PaddleX` is not an installed package, so importing it whole fails on its own
metadata lookup. Exec'ing the functions under test also pins each reference to the exact source
being ported, which is the point.

They need `shapely`, `beautifulsoup4`, `scikit-learn` and `colorlog` on top of the model
dependencies; `pip install` them if a dumper reports one missing.

## Running the whole Python pipeline beside the port

`compare_with_upstream.py` is not a fixture generator. It builds the genuine
`PaddleOCR-VL-1.6` pipeline — the shipped `paddlex/configs/pipelines/PaddleOCR-VL-1.6.yaml`,
rewritten only to point each sub-model at the local checkout — runs it over three rendered
pages, and writes each page's markdown and `parsing_res_list` next to the page itself:

```bash
pip install --no-compile --ignore-installed PyYAML -e "/home/user/ref/PaddleX[ocr]"
COMPARE_DIR=/tmp/compare python3 compare_with_upstream.py
paddleocr-sharp parse /tmp/compare/report.png --output /tmp/compare
```

That needs PaddleX importable rather than exec'd, since the whole pipeline is what is being run.
The output is what the side-by-side comparison of the two implementations is built from.

## Settling a behavioural question

`probe_runaway_decode.py` is the other non-fixture script. It loads the checkpoint through
`transformers` and calls `generate` with the arguments PaddleX's local backend passes — greedy, a
token cap, no penalties — so that a claim about what upstream *does* can be measured instead of
read off the source:

```bash
python3 probe_runaway_decode.py page.png 8192 --box 138,409,1008,545 --prompt "OCR:"
```

It prints whether generation stopped on the stop token or ran to the cap. It is what established
that upstream has no stop for a decode that has fallen into a cycle, which is the case for
`GenerationOptions.StopOnRepetition` — see the CLAUDE.md section of that name.

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
