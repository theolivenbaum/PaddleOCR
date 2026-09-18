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

- [x] Static-mirror downloader: ranged resume, length verify, parallel chunks
- [x] Local cache layout (`~/.cache/paddleocr-sharp/<model>/`) with lockfile
- [x] Mirror support (`PADDLEOCR_SHARP_MODELS_URL` endpoint override, bearer token)
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
- [x] `crop_margin` — the contrast stretch before the threshold, and OpenCV's grey weights
      landing on an RGB buffer, both checked against upstream over seven crops — plus the
      seal and spotting pre-processing helpers
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
- [x] Post-process: integer rounding, sigmoid scores, box decode, threshold 0.3 (shared or
      per class), NMS, unclip (shared or per class), label map
- [x] Reading-order head
- [x] Mask head: region polygons (`layout_shape_mode` rect/quad/poly/auto), used for overlap
      filtering and for masking each block's crop
- [x] Parity tests vs. Paddle reference detections on sample pages

## 9. Doc pre-processing (optional models)

- [x] `PP-LCNet_x1_0_doc_ori` orientation classifier (0/90/180/270)
- [x] `UVDoc` unwarping
- [x] Wire into pipeline behind `use_doc_orientation_classify` / `use_doc_unwarping`
- [x] Parity tests vs. Paddle inference for both graphs and both wrappers

## 9b. Performance

- [x] Machine calibration ahead of every benchmark run (`MachineProfile`): FMA rate at 256 and 512
      bits, single- and all-thread, cache and DRAM read bandwidth, and the sample spread that says
      how far to trust the run. Stage figures are reported against it.
- [x] Per-shape GEMM sweep (`--gemm true`) at the shapes the model issues, as a fraction of the
      measured ceiling
- [x] Vectorised the Paddle interpreter's cast and boolean-normalise paths, which the mask head's
      twelve-million-element tensors run through: layout graph 9.3 s to 7.6 s, `conv2d` flat as a
      control
- [x] Vector width is a measured decision, not an assumed one (`Core/Simd.cs`), overridable with
      `PADDLEOCR_SHARP_VECTOR_BITS`
- [ ] Store boolean tensors in a byte-wide buffer. `PaddleTensor` backs them with `long[]`, so a
      `[1, 300, 200, 200]` mask costs 96 MB instead of 12 and its first touch is page-fault bound.
      The remaining elementwise time in the layout graph is mostly this.
- [x] Per-stage profile for the hand-written towers (`StageProfile`), the counterpart of
      `PirProfile` for the layout graph
- [x] Verified against the original: the same three pages, both pipelines back to back on one
      machine at their own defaults, 131.9 s Python against 68.3 s here with byte-identical
      markdown. Recorded in CLAUDE.md; the published comparison is corrected.
- [ ] Attention is 44% of the vision tower and reaches about a third of the matrix products'
      throughput. Its score product reduces along an inner dimension of 72, which is too short
      for the kernel; a flash-attention-shaped rewrite that tiles over keys with a running
      maximum is the way in. Key and value bandwidth is *not* the limit — halving those reads
      changed nothing.
- [ ] Convolution is now a third of the layout graph and is the next thing worth attacking

## 10. Pipeline — `src/PaddleOcrSharp/Pipeline`

- [x] Layout box filtering (`filter_overlap_boxes`) and merge modes (`union` / `large`),
      checked against the upstream function over thirteen layouts
- [x] Block cropping, adjacent-block merging and the reordering it implies (`merge_blocks`,
      `merge_images`), checked against the upstream functions over ten layouts
- [x] Per-label prompts: `OCR:`, `Table Recognition:`, `Formula Recognition:`,
      `Chart Recognition:`, `Seal Recognition:`, `Spotting:` with per-label pixel budgets
- [x] Table figure tokenisation / untokenisation (`tokenize_figure_of_table`)
- [x] Per-label pixel budgets, configurable the way upstream's `vlm_kwargs` are (`BlockPixelBudgets`)
- [x] Block batching, matching the backend being ported. The pipeline's VLM worker gathers boxes
      up to `vl_rec_model.batch_sampler.batch_size`, but the *local* predictor pins that to
      `PADDLEOCR_VL_LOCAL_BATCH_SIZE = 1` and warns if asked for more
      (`inference/models/doc_vlm/{constants,predictor}.py`) — batching above one belongs to the
      vLLM/SGLang back-ends, which are out of scope. One block at a time is therefore what the
      ported backend does, and it costs nothing: vision attention is block-diagonal per image, so
      a batch is arithmetically a sequence.
- [x] OTSL → HTML table conversion, checked against `convert_otsl_to_html` over ten tables
- [x] Repetition truncation, checked against `truncate_repetitive_content` over eight outputs
- [x] Spotting `<|LOC_n|>` post-processing
- [x] Markdown + JSON result assembly, `markdown_ignore_labels`, multi-page concatenation
- [x] JSON carries `block_order`, `group_id`, `title_level` and `block_polygon_points`;
      `format_block_content` renders each block instead of emitting its raw text
- [x] The pipeline's HTML-decorated ("pretty") markdown, byte-for-byte against `MarkdownConverter`
- [x] Cross-page table merging (`restructure_pages(merge_tables=True)`)
- [x] Title re-levelling (`assign_levels_to_parsing_res`); its height clustering is an exact
      one-dimensional k-means rather than scikit-learn's seeded local search, and agrees with it
      on the reference document
- [x] Reading-flow block numbering (`update_order_index`), emitted as `block_order`
- [x] End-to-end parity test on a sample document

## 11. CLI — `src/PaddleOcrSharp.Cli`

- [x] `download` — fetch models
- [x] `parse` — document → markdown / JSON, `--layout`, `--no-layout`, `--prompt-label`
- [x] `bench` — throughput and allocation report
- [x] `dump` — internal tensors via `VisionTrace` / `IPirTrace`, operator timings via `bench`
- [x] Progress reporting (synchronous, on stderr, so `--format json` pipes cleanly)
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
- [x] Allocation audit: every tensor in the hot loops is pooled. A vision pass allocates 1.2 MiB
      per page (was 15) after removing display-class allocations from the GEMM entry points; a
      decode step allocates ~310 KiB, which is `Parallel.For` task machinery across its 144
      projections and would need a custom scheduler to remove.
- [x] Benchmarks vs. Python reference (tokens/s, pages/min)
- [x] GEMM/conv/attention blocking pass — layout graph 7.9s -> 5.2s, vision tower 19s -> 16s
- [x] AOT-compatibility check for the CLI (publishes clean; `IsAotCompatible` guards regressions)
- [x] Public API review. `GenerateDocumentationFile` turns on CS1591, so an undocumented public
      member is now a warning rather than something nobody notices; the three gaps it found are
      closed. The Paddle operator kernels are `internal` — nothing outside the interpreter used
      them. The rest of the surface stays public on purpose: this port is meant to be inspectable
      stage by stage, which is also how the parity tests reach it.

## Performance — measured on the scanned corpus

- [x] `PageProfile`: per-stage wall time and allocations for everything outside the model call,
      printed by `parse --profile`; `--layout-profile` adds the layout graph's operators
- [x] Attention records its own parts from inside, as thread-ticks, since it threads over heads
- [x] Cap `Parallel.For` at the core count everywhere (`Core/Parallelism.cs`)
- [x] Attention score product: transposed keys, four rows by two column vectors, token block
      outermost
- [x] Attention value product: chunked reduction with the output tile in registers
- [x] Baseline and after, six pages of `pdf_scanned` + `images/ocr_*`, one page each
- [x] The layout graph allocated 4.5 GiB per detection. `TensorArena` pools the interpreter's
      intermediates against an exact per-array count of the value slots referencing them:
      **4,458 MiB and 7.5 s a detection become 145 MiB and 4.7 s**. The allocation was the
      binding constraint, which is why vectorising the broadcast fallback and tuning the GC had
      both been neutral.
- [x] `TensorPool` on `ArrayPool<T>.Shared`. Its own pool was created on a stale premise (the
      1 MiB bucket cap) and dropped every buffer past 16 of a size.
- [x] The arena's residual, which was not scaffolding. `PirProfile` reports allocation per
      operator and `PADDLEOCR_SHARP_ARENA_STATS=1` reports the arena's own accounting; between
      them, 128 of the 145 MiB was the fetches leaving the pool a bucket short every run, and 8
      was the input tensor built before the arena existed. `RunPooled` hands the fetch buffers
      back on dispose and the input is rented by hand: **145 MiB -> 9 MiB**, of which 2.5 is
      genuinely per-operator. End to end, three pages in one process allocate 1.6 GiB of layout
      against 13.4 GiB before the arena; the gap to the bench's 9 MiB is the graph's
      content-dependent shapes asking for buckets the pool has not got yet.
- [~] `Bool` and `Int32` tensors are still `long[]`. **Measured and not worth doing**: the arena's
      by-storage line puts integral storage at 928 MiB of the 6,359 MiB a detection moves (15%),
      and post-arena the operators that touch those tensors are 11.7% of the graph's time. Upper
      bound on the whole exercise is ~6% of a stage that is 8% of a page. The blast radius is
      small (9 files, ~85 call sites, most reads already funnelled through `GetLong`) — it is the
      payoff that is missing, not the feasibility.
- [ ] `conv2d` is 47.9% of a post-arena detection, and deformable attention's `matmul` +
      `transpose` + `add` another 17.5%. That is where this graph's remaining time is. Both are
      im2col-plus-GEMM shapes, so the levers are the ones that worked on the vision tower.
- [ ] Re-profile after every change to this graph. Three times now a fix has moved the bottleneck
      somewhere the previous profile could not see: element-wise kernels, then allocation, now the
      convolutions.
- [ ] `batch_norm_` allocates two per-channel arrays per call, 1 MiB over a detection and the
      largest operator-level line left. `LinearOps` could take them from the pool.
- [ ] Only ~10 of the layout graph's 300 query masks survive the score threshold, and the graph
      reduces all 300. Pruning would be a semantic change to a fetched tensor, so it needs care.
- [ ] Decode is 16% of the corpus at ~721 MB of bf16 weights per token, and the profile says it is
      bandwidth-bound on streaming them. Nothing here changed it; the lever is reading fewer bytes,
      not a faster kernel.

## Against a genuine PaddleX v1.6 install

- [x] Build upstream's real pipeline in a venv and benchmark it page for page against the port,
      one page per process pair, model loading excluded on both sides. Four pages: **1.16x, 2.12x,
      4.55x**, and one page upstream did not finish inside 90 minutes twice while the port parsed
      it in 74 s. Output identical apart from the trailing newline the port adds.
- [x] Split each side's page between its two stages, so the end-to-end ratio stops hiding that the
      halves go opposite ways: **the port's VL half is 1.5-2.2x faster and its layout half about
      3x slower** (~6 s against ~2 s). The layout deficit is near-constant, so it is what makes a
      one-block page 1.16x.
- [x] The port rendered PDFs at 200 dpi where upstream renders at 144 (`PDFReader(zoom=2.0)` over
      the natural 72) — 1.93x the pixels and so 1.93x the patches, on every PDF in every
      comparison. Default is now 144: `nougat_004_scanned.pdf` 50.6 s -> 39.6 s on that alone,
      matching upstream's output at the matched resolution.
- [~] Do **not** move the port to fp32 to "match" upstream. Upstream is fp32 on CPU only because
      `is_bfloat16_available` excludes CPU; the port's bf16 weight reads are half the bytes per
      token and are most of the decode win. Matching the dtype would cost the win and change no
      number that is compared.
- [ ] Fold `batch_norm_` into the preceding convolution's weights at graph load. Both are affine
      constants at inference, so the fold is exact; it removes an operator and a full pass over
      the activation feeding it. **Unmeasured** — a candidate, not a result.
- [ ] A direct convolution kernel for the backbone's small stride-1 shapes, where the im2col
      column buffer plausibly costs more than the multiply-adds it feeds. **Unmeasured.**

## Ternary weight quantization

Design: [`dotnet/docs/ternary.md`](dotnet/docs/ternary.md). Encoding investigated in the
`PrismML-Eng/llama.cpp` fork and the Bonsai whitepapers; scope is the VL model only.

- [x] `Formats/Gguf`: GGUF v3 reader and streaming writer, the fork's type ids (`PQ2_0` 142,
      `PTQ1_0` 143), block geometry and row sizes.
- [x] `TernaryBlocks`: both codecs, **byte-identical** to `quantize_row_*_ref` and
      `dequantize_row_*` in the fork — checked against the compiled C and against an independent
      Python transcription (`tools/reference/dump_ternary_blocks.py`), which agree on every byte.
- [x] `TernaryKernels`: the vectorised decoders, asserted equal to the scalar reference rather than
      close to it. Both layouts decode into contiguous runs, which is what makes that possible —
      for `PTQ1_0` a fixed digit index over `c` consecutive bytes is the `c` consecutive weights at
      `n·c`, which is not obvious from the encoder's scatter.
- [x] `HadamardRotation`: the normalized Sylvester–Walsh transform and the ±1 diagonal, sign first
      then rotation as the fork does it; fold and apply are one implementation so they cannot drift.
- [x] `WeightMatrix` gains quantized storage and `Gemm` applies a weight's rotation to the
      activation, so the tower, the decoder and the pipeline are unchanged.
- [x] `WeightQuantizer`: per-group scale search, GPTQ error feedback with a Cholesky of the damped
      Hessian, per-tensor error report.
- [x] `ActivationRecorder`: `H = Σ xᵀx` per weight while the bf16 model runs, as an `AsyncLocal`
      scope, split into passes that fit a memory budget.
- [x] `QuantizationPolicy`: per-tensor precision as JSON globs, with the four tensors that should
      not be quantized at all spelled out and why.
- [x] `tools/PaddleOcrSharp.Quantize`: `convert`, `validate`, `inspect`, `policy`.
- [x] Tests: golden vectors, container round-trips, rotation against the explicit matrix, the
      quantized kernels against their dequantized reference, GPTQ against round-to-nearest.
- [x] **Converted the real checkpoint and measured it.** 1.92 GB -> 0.72 GB (2.68x) in 51.5 s,
      70.3% of parameters at 1.75 bpw. It does not work: 0.00% character accuracy, vision tower
      cosine 0.330, and the boarding pass comes back as `POOOOO / IOOOOEEEE...`. With the rotation
      (block 128, the only one this model's widths admit) the tensor numbers improve a lot — worst
      error 0.53 -> 0.43, lowest row cosine 0.18 -> 0.87, no collapsed rows — and the tower reaches
      only 0.472, the text is still 0.00%, and it is three times slower again (0.09x) because every
      `Gemm.Linear` call redoes the transform. `docs/ternary.md` §8 is the full account.
- [x] Five defects the real conversion found and synthetic tensors could not: an `f32` exemption
      upcasting a bfloat16 checkpoint (75 -> 151 MB on a tensor the port never reads), rank-1
      tensors counted as `n x n` (1884 M against a real 959 M), an fp16 scale fallback that
      quantized against a scale the file cannot hold, a worst-row-cosine metric dominated by rows
      whose weights are 1e-8, and a validator comparing rotated weights against an unrotated
      source. All fixed; the last two were found only because two implementations disagreed about
      the same file.
- [x] **Integer bands, and zero loss.** `Q4_0`, `Q4_1`, `Q5_1` and `Q8_0` over the same container,
      policy and runner, each byte-identical to the compiled C reference. Measured against bf16:
      ptq1_0 0.72 GB at 0.00% character accuracy, q4_0 0.74 GB at 94.86%, q4_1 0.79 GB at 93.57%,
      q5_1 0.90 GB at 98.71%, **q8_0 1.15 GB with the text byte-identical on all four corpus
      images**. `q8_0` is now the default. The two four-bit bands are tied rather than ordered on
      this evidence.
- [x] Freed the two remaining big exemptions — the token embedding (106 M) and
      `packing_position_embedding` (37.7 M, never read by this port) — taking the quantized share
      from 70.3% to 85.3%.
- [x] **A corpus that discriminates.** `curiosity-ai/test_documents` and `PP-DocLayoutV3` are now
      in place, so the whole pipeline runs: layout, cropping, per-block recognition, markdown.
      Against bf16 on this repository's own benchmark pages the markdown is **byte-identical** on
      all four — `ocr_test_original.png`, `nougat_004_scanned.pdf`, `ocr_image.jpg` (a table, 13 KB)
      and `balance_sheet_1.png` (decode-heavy, 16 KB) — 30 KB of markdown without a differing byte.
- [ ] The same run over the remaining bands. `q5_1` and `q4_1` have only ever been measured through
      whole-image recognition, where three of four crops could not tell any band apart. Their real
      standing is unknown.
- [ ] **Measure a split policy** if a band between 1.67x and 2.14x is wanted. The vision tower is
      48.6% of the model, so `q8_0` there with `q4_1` elsewhere lands near 0.93 GB — barely better
      than `q5_1` everywhere at 0.89 GB, which is why uniform bands are what got measured.
- [ ] Run GPTQ against the real model. Implemented and tested on synthetic tensors, never run on
      the checkpoint: it needs a calibration corpus, which this environment did not have.
- [ ] **Vectorise the integer-band decoders.** Through the real pipeline the cost runs 0.91x, 0.79x,
      0.29x and 0.38x across the four pages — it tracks *output length*, so it is the decode step
      and not the vision tower, which an earlier whole-image comparison had wrongly implicated.
      `TernaryKernels` routes every integer band to the scalar reference, so `RunNarrow` decodes a
      full weight row per dot product one element at a time. That is the first thing to fix.
- [ ] **The 4304-wide vision projection is now the binding limit on size**: 4304 = 16 x 269 divides
      by neither the ternary group of 128 nor the integer group of 32, so 139 M parameters — 278 MB
      of the 1.15 GB `q8_0` file — stay bfloat16. Its other dimension is 1152 and divides both, so
      storing it transposed and reducing along rows reaches it; `Gemm.MatMul` already has a kernel
      in that shape. Worth about 130 MB, and it is a runner change, so it belongs with the speed
      work.
- [ ] Hoist the activation rotation: `q`/`k`/`v` each rotate the same normed activation, which the
      fork memoizes. The rotated model was markedly slower still, so this now has a reason.

## Closing the layout gap against upstream

- [x] `conv2d` reached `Gemm.MatMul`'s `k x n` form, which accumulates output rows in memory at one
      load and one store per multiply-add. Posed instead as the shape `Gemm.Linear` wants — filters
      as the activation rows, im2col columns as the weight panel, over a block of output rows —
      **81 -> ~180 GFLOP/s**, and the product lands in `[channel, pixel]` order so it copies out
      rather than transposing. **-10% of the stage.**
- [x] Nothing outside `conv2d` was threaded. `Parallelism.Chunked` is the one place that split now
      lives, and `Broadcast.Seed` lets a walk start somewhere other than zero so the counter-based
      operators can split too. **-31% of the stage**, the largest single item.
- [x] `batch_norm_` scalar and unpooled (156 -> 34 ms), `any` over a contiguous suffix through the
      generic reduction (68 ms -> off the profile), `depthwise_conv2d` bounds-checking 25 taps per
      output pixel at 3.7 GFLOP/s (149 -> 41 ms), `einsum` rebuilding both operands' offsets for
      each of 23 million terms (88 -> 45 ms).
- [x] `Gemm.MatMul`'s direct tile register-blocked, four rows by two vectors held across the
      reduction — the vision tower's store-port bound again. matmul 221 -> 178 ms. `Gemm.Linear`
      no longer copies a float32 panel it has nothing to widen.
- [x] Tiered compilation off for the CLI. Every run of the tool is a cold process, so nothing ever
      reaches the steady state tiering is for: page 10712/10036 -> 8149/8352 ms. The library is
      untouched. ReadyToRun measured worse and is not taken.
- [x] End to end, both sides in one shell run: **1.89x, 1.98x and 7.55x**, with the layout stage
      at 3.4-3.5 s per cold process against upstream's 3.0 s cold and 2.2 s warm — from 3x behind
      to parity cold and ~1.6x warm. Byte-identical output on all four test pages.
- [ ] What is left in `conv2d` is ~65% GEMM at ~180 GFLOP/s, ~30% im2col fill, ~5% copy-out.
      Closing it needs a GEMM that blocks the reduction as well as the output: `Linear` sweeps its
      panel once per four activation rows, which is 0.5 bytes per flop from L2 whatever the panel
      width. That is a change to the kernel both model halves depend on, so it wants its own
      measurement pass with the vision tower as a control.
