# PaddleOCR-VL C# port — progress tracker

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked

See [`CLAUDE.md`](CLAUDE.md) for architecture notes and conventions.

---

## 0. Groundwork

- [x] Review upstream `modeling_paddleocr_vl.py` / `configuration_paddleocr_vl.py` / `image_processing_paddleocr_vl.py` / `processing_paddleocr_vl.py`
- [x] Review PaddleX `paddleocr_vl` pipeline (`pipeline.py`, `uilts.py`, `result.py`)
- [x] Identify layout model (`PP-DocLayoutV3`, RT-DETR + HGNetV2-L) and its config
- [x] Disable upstream CI workflows, pre-commit config and agent skill files (`*.disabled`)
- [x] `dotnet/` solution skeleton: library + CLI + test projects, builds and tests green
- [x] `CLAUDE.md` and `to-do.md`

## 1. Core tensor infrastructure — `src/PaddleOcrSharp/Core`

- [x] `Tensor` / `TensorView` over pooled `Memory<float>` with shape + stride
- [x] `TensorPool` — `ArrayPool`-backed rent/return, scoped lifetimes, zero steady-state alloc
- [x] dtype conversion: `bfloat16`/`float16` ↔ `float32` (vectorised, `Vector512` path)
- [x] SIMD element-wise kernels: add, mul, silu, gelu(erf), gelu(tanh), softmax (fp32 accum), rsqrt
- [x] `RmsNorm`, `LayerNorm` (fp32 accumulation, matching upstream precision)
- [x] GEMM: blocked, multi-threaded, bf16-weight × fp32-activation with on-the-fly tile conversion
- [x] Attention primitive: scaled dot-product with causal / block-diagonal masks, fp32 softmax
- [x] Unit tests for every kernel against a naive scalar reference

## 2. Weight formats — `src/PaddleOcrSharp/Formats`

- [x] `SafetensorsFile`: header parse, memory-mapped tensor views, lazy dtype-aware access
- [x] Paddle `inference.pdiparams` + `inference.json` reader (for `PP-DocLayoutV3`)
- [x] `.npz` reader/writer (test fixtures only)
- [x] Weight-name mapping tables (HF ⇄ internal module tree)

## 3. Model downloader — `src/PaddleOcrSharp/Download`

- [x] Hugging Face resolve-URL downloader: ranged resume, SHA-256/etag verify, parallel chunks
- [x] Local cache layout (`~/.cache/paddleocr-sharp/<repo>/<revision>/`) with lockfile
- [x] Mirror support (HF endpoint override, BOS/AIStudio)
- [x] Manifests for `PaddleOCR-VL-1.6`, `PP-DocLayoutV3`, `PP-LCNet_x1_0_doc_ori`, `UVDoc`
- [x] `paddleocr-sharp download` CLI verb + progress reporting
- [x] Tests: manifest resolution, cache hit/miss, resume, corrupt-file detection

## 4. Imaging — `src/PaddleOcrSharp/Imaging`

- [x] SkiaSharp decode (PNG/JPEG/BMP/WEBP) → RGB planar, EXIF orientation
- [x] PDF page rasterisation (`PaddleOcrSharp.Pdf`, PDFium via PDFtoImage)
- [x] `SmartResize` port (`factor`, `min_pixels`, `max_pixels`, aspect-ratio guard)
- [x] Bicubic resample matching PIL/`torchvision` (a = −0.5, antialias behaviour verified)
- [x] Rescale + normalize + HWC→CHW, fused and vectorised
- [x] Patchify to `(grid_h*grid_w, 3, 14, 14)` and grid-THW computation
- [x] `crop_margin`, seal/spotting pre-processing helpers
- [x] Parity tests vs. Python `image_processing_paddleocr_vl.PaddleOCRVLImageProcessor`

## 5. Vision tower — `src/PaddleOcrSharp/Models/Vision`

- [x] Patch embedding (Conv2d 14×14 stride 14 → GEMM)
- [x] Bilinear interpolation of the 27×27 position grid (`align_corners=false`) + LFU cache
- [x] 2-D RoPE (`SigLIPRotaryEmbedding`) and `rotate_half` application
- [x] Encoder layer (LN → MHA → LN → MLP) ×27, block-diagonal attention over packed images
- [x] `post_layernorm`, per-image split by `cu_seqlens`
- [x] Projector `mlp_AR` (pre-norm, 2×2 merge, 4608→4608→1024)
- [x] Parity tests: per-layer hidden states vs. Python dumps (rtol ≤ 2e-2 in bf16)

## 6. Language model — `src/PaddleOcrSharp/Models/Language`

- [x] Token embedding + `lm_head`
- [x] `Ernie4_5Attention` with GQA and 3-D M-RoPE (`mrope_section [16,24,24]`, θ 500 000)
- [x] `Ernie4_5MLP` (SwiGLU) and `RMSNorm`
- [x] Paged/contiguous KV cache with pooling; prefill + incremental decode
- [x] `get_rope_index` port (image grid ↔ text position ids)
- [x] Sampling: greedy, temperature, top-p, repetition penalty
- [x] Stop conditions (`</s>`), max-new-tokens, repetition-collapse guard
- [x] Parity tests: logits after prefill, then 16 greedy steps vs. Python

## 7. Tokenizer — `src/PaddleOcrSharp/Text`

- [x] `tokenizer.json` reader (vocab, merges, added tokens, normalizer, pre-tokenizer)
- [x] BPE encode/decode with added-token splitting (1019 special tokens incl. `<|LOC_n|>`)
- [x] Chat-template rendering for the fixed PaddleOCR-VL prompt shape (no Jinja engine)
- [x] Image-placeholder expansion (`grid.prod() / merge² ` placeholders)
- [x] Parity tests vs. `tokenizers` on a multilingual corpus

## 8. Layout detection — `src/PaddleOcrSharp/Models/Layout`

*Approach: PP-DocLayoutV3 ships only as a Paddle inference graph, so it runs through our own
PIR graph interpreter (`Models/Paddle`) rather than a hand-written RT-DETR. Every kernel is ours
and the result is exact by construction; the same interpreter also covers UVDoc and the
orientation classifier.*

- [x] Decide weight source: convert `inference.pdiparams` → safetensors, or read Paddle blob directly
- [x] HGNetV2-L backbone (stem, stages, LearnableAffineBlock)
- [x] Hybrid encoder (AIFI transformer level + CCFM/PAN fusion)
- [x] Deformable-DETR decoder (300 queries, 6 layers, 4 sample points)
- [x] Post-process: sigmoid scores, box decode, threshold 0.3, NMS, unclip, label map
- [x] Reading-order / mask heads as far as the pipeline needs them
- [x] Parity tests vs. Paddle reference detections on sample pages

## 9. Doc pre-processing (optional models)

- [x] `PP-LCNet_x1_0_doc_ori` orientation classifier (0/90/180/270)
- [x] `UVDoc` unwarping
- [x] Wire into pipeline behind `use_doc_orientation_classify` / `use_doc_unwarping`
- [x] Parity tests vs. Paddle inference for both graphs and both wrappers

## 10. Pipeline — `src/PaddleOcrSharp/Pipeline`

- [x] Layout box filtering (`filter_overlap_boxes`) and merge modes (`union` / `large`)
- [x] Block cropping, adjacent-block merging (`merge_blocks`)
- [x] Per-label prompts: `OCR:`, `Table Recognition:`, `Formula Recognition:`,
      `Chart Recognition:`, `Seal Recognition:`, `Spotting:` with per-label pixel budgets
- [x] Table figure tokenisation / untokenisation (`tokenize_figure_of_table`)
- [~] Batched VL recognition scheduling (blocks run sequentially or across `BlockConcurrency` threads; upstream batches by pixel budget)
- [x] OTSL → HTML table conversion
- [x] Repetition truncation (`truncate_repetitive_content`)
- [x] Spotting `<|LOC_n|>` post-processing
- [x] Markdown + JSON result assembly, `markdown_ignore_labels`, multi-page concatenation
- [x] End-to-end parity test on a sample document

## 11. CLI — `src/PaddleOcrSharp.Cli`

- [x] `download` — fetch models
- [x] `parse` — document → markdown / JSON, `--layout`, `--no-layout`, `--prompt-label`
- [x] `bench` — throughput and allocation report
- [x] `dump` — internal tensors via `VisionTrace` / `IPirTrace`, operator timings via `bench`
- [ ] Progress + structured logging
- [x] `bench --no-vl` / `--no-layout`; `--no-<flag>` now actually parses

## 12. Reference tooling — `dotnet/tools/reference`

- [x] `dump_image_processing.py`
- [x] `dump_vision.py` (per-layer hidden states)
- [x] `dump_language.py` (logits, KV cache, greedy steps)
- [x] `dump_tokenizer.py`
- [x] `dump_layout.py`
- [x] `dump_end_to_end.py`
- [x] `dump_preprocessing.py` (orientation classifier + UVDoc)
- [x] README describing fixture generation

## 13. Hardening

- [x] Multi-threading strategy + `ServerGC` tuning
- [~] Allocation audit (decode is ~10 MiB/step; graph tensors now skip zero-initialisation)
- [x] Benchmarks vs. Python reference (tokens/s, pages/min)
- [x] GEMM/conv/attention blocking pass — layout graph 7.9s -> 5.2s, vision tower 19s -> 16s
- [ ] AOT-compatibility check for the CLI
- [~] Public API review + XML docs (every public member documented; a shape review is still open)
