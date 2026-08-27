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
- [ ] Attention primitive: scaled dot-product with causal / block-diagonal masks, fp32 softmax
- [x] Unit tests for every kernel against a naive scalar reference

## 2. Weight formats — `src/PaddleOcrSharp/Formats`

- [x] `SafetensorsFile`: header parse, memory-mapped tensor views, lazy dtype-aware access
- [ ] Paddle `inference.pdiparams` + `inference.json` reader (for `PP-DocLayoutV3`)
- [x] `.npz` reader/writer (test fixtures only)
- [ ] Weight-name mapping tables (HF ⇄ internal module tree)

## 3. Model downloader — `src/PaddleOcrSharp/Download`

- [ ] Hugging Face resolve-URL downloader: ranged resume, SHA-256/etag verify, parallel chunks
- [ ] Local cache layout (`~/.cache/paddleocr-sharp/<repo>/<revision>/`) with lockfile
- [ ] Mirror support (HF endpoint override, BOS/AIStudio)
- [ ] Manifests for `PaddleOCR-VL-1.6`, `PP-DocLayoutV3`, `PP-LCNet_x1_0_doc_ori`, `UVDoc`
- [ ] `paddleocr-sharp download` CLI verb + progress reporting
- [ ] Tests: manifest resolution, cache hit/miss, resume, corrupt-file detection

## 4. Imaging — `src/PaddleOcrSharp/Imaging`

- [x] SkiaSharp decode (PNG/JPEG/BMP/WEBP) → RGB planar, EXIF orientation
- [ ] PDF page rasterisation (deferred — CLI accepts images first)
- [x] `SmartResize` port (`factor`, `min_pixels`, `max_pixels`, aspect-ratio guard)
- [x] Bicubic resample matching PIL/`torchvision` (a = −0.5, antialias behaviour verified)
- [x] Rescale + normalize + HWC→CHW, fused and vectorised
- [x] Patchify to `(grid_h*grid_w, 3, 14, 14)` and grid-THW computation
- [ ] `crop_margin`, seal/spotting pre-processing helpers
- [x] Parity tests vs. Python `image_processing_paddleocr_vl.PaddleOCRVLImageProcessor`

## 5. Vision tower — `src/PaddleOcrSharp/Models/Vision`

- [ ] Patch embedding (Conv2d 14×14 stride 14 → GEMM)
- [ ] Bilinear interpolation of the 27×27 position grid (`align_corners=false`) + LFU cache
- [ ] 2-D RoPE (`SigLIPRotaryEmbedding`) and `rotate_half` application
- [ ] Encoder layer (LN → MHA → LN → MLP) ×27, block-diagonal attention over packed images
- [ ] `post_layernorm`, per-image split by `cu_seqlens`
- [ ] Projector `mlp_AR` (pre-norm, 2×2 merge, 4608→4608→1024)
- [ ] Parity tests: per-layer hidden states vs. Python dumps (rtol ≤ 2e-2 in bf16)

## 6. Language model — `src/PaddleOcrSharp/Models/Language`

- [ ] Token embedding + `lm_head`
- [ ] `Ernie4_5Attention` with GQA and 3-D M-RoPE (`mrope_section [16,24,24]`, θ 500 000)
- [ ] `Ernie4_5MLP` (SwiGLU) and `RMSNorm`
- [ ] Paged/contiguous KV cache with pooling; prefill + incremental decode
- [ ] `get_rope_index` port (image grid ↔ text position ids)
- [ ] Sampling: greedy, temperature, top-p, repetition penalty
- [ ] Stop conditions (`</s>`), max-new-tokens, repetition-collapse guard
- [ ] Parity tests: logits after prefill, then 16 greedy steps vs. Python

## 7. Tokenizer — `src/PaddleOcrSharp/Text`

- [ ] `tokenizer.json` reader (vocab, merges, added tokens, normalizer, pre-tokenizer)
- [ ] BPE encode/decode with added-token splitting (1019 special tokens incl. `<|LOC_n|>`)
- [ ] Chat-template rendering for the fixed PaddleOCR-VL prompt shape (no Jinja engine)
- [ ] Image-placeholder expansion (`grid.prod() / merge² ` placeholders)
- [ ] Parity tests vs. `tokenizers` on a multilingual corpus

## 8. Layout detection — `src/PaddleOcrSharp/Models/Layout`

- [ ] Decide weight source: convert `inference.pdiparams` → safetensors, or read Paddle blob directly
- [ ] HGNetV2-L backbone (stem, stages, LearnableAffineBlock)
- [ ] Hybrid encoder (AIFI transformer level + CCFM/PAN fusion)
- [ ] Deformable-DETR decoder (300 queries, 6 layers, 4 sample points)
- [ ] Post-process: sigmoid scores, box decode, threshold 0.3, NMS, unclip, label map
- [ ] Reading-order / mask heads as far as the pipeline needs them
- [ ] Parity tests vs. Paddle reference detections on sample pages

## 9. Doc pre-processing (optional models)

- [ ] `PP-LCNet_x1_0_doc_ori` orientation classifier (0/90/180/270)
- [ ] `UVDoc` unwarping
- [ ] Wire into pipeline behind `use_doc_orientation_classify` / `use_doc_unwarping`

## 10. Pipeline — `src/PaddleOcrSharp/Pipeline`

- [ ] Layout box filtering (`filter_overlap_boxes`) and merge modes (`union` / `large`)
- [ ] Block cropping, adjacent-block merging (`merge_blocks`)
- [ ] Per-label prompts: `OCR:`, `Table Recognition:`, `Formula Recognition:`,
      `Chart Recognition:`, `Seal Recognition:`, `Spotting:` with per-label pixel budgets
- [ ] Table figure tokenisation / untokenisation
- [ ] Batched VL recognition scheduling
- [ ] OTSL → HTML table conversion
- [ ] Repetition truncation (`truncate_repetitive_content`)
- [ ] Spotting `<|LOC_n|>` post-processing
- [ ] Markdown + JSON result assembly, `markdown_ignore_labels`, multi-page concatenation
- [ ] End-to-end parity test on a sample document

## 11. CLI — `src/PaddleOcrSharp.Cli`

- [ ] `download` — fetch models
- [ ] `parse` — document → markdown / JSON, `--layout`, `--no-layout`, `--prompt-label`
- [ ] `bench` — throughput and allocation report
- [ ] `dump` — emit internal tensors for parity debugging
- [ ] Progress + structured logging

## 12. Reference tooling — `dotnet/tools/reference`

- [x] `dump_image_processing.py`
- [ ] `dump_vision.py` (per-layer hidden states)
- [ ] `dump_language.py` (logits, KV cache, greedy steps)
- [ ] `dump_tokenizer.py`
- [ ] `dump_layout.py`
- [ ] `dump_end_to_end.py`
- [ ] README describing fixture generation

## 13. Hardening

- [ ] Multi-threading strategy + `ServerGC` tuning
- [ ] Allocation audit (target: zero steady-state LOH traffic during decode)
- [ ] Benchmarks vs. Python reference (tokens/s, pages/min)
- [ ] AOT-compatibility check for the CLI
- [ ] Public API review + XML docs
