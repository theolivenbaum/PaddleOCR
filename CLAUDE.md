# PaddleOCR-VL — Pure C# Port (`dotnet/`)

This repository hosts, in addition to the upstream Python PaddleOCR sources, a **pure C#
re-implementation of the PaddleOCR-VL-1.6 document-parsing pipeline** under [`dotnet/`](dotnet/).

The C# port is self-contained: it owns the tensor math, the model graph, the tokenizer, the
image pipeline and the weight loading. **It does not use ONNX Runtime, Paddle Inference,
libtorch, or any other third-party inference engine.**

---

## Scope

| In scope | Out of scope |
| --- | --- |
| PaddleOCR-VL **1.6** pipeline (`PaddleOCR-VL-1.6-0.9B` + `PP-DocLayoutV3`) | PP-OCRv3/v4/v5 detector+recognizer (legacy models) |
| Doc pre-processing (orientation classify, unwarping) | Training / fine-tuning |
| Layout detection, block cropping, block-level VL recognition | PP-StructureV3, PP-ChatOCR, table-cell detectors |
| Markdown / JSON assembly, OTSL→HTML table conversion | Serving (HPS/Triton), distributed inference |
| CPU inference (SIMD-accelerated); GPU is a possible future backend | vLLM / SGLang / FastDeploy back-ends |

## Upstream references used

The port is validated line-by-line against these upstream sources. They are the **normative
specification** — when C# and Python disagree, Python is right.

| Component | Upstream reference |
| --- | --- |
| VL model graph | `PaddlePaddle/PaddleOCR-VL-1.6` on Hugging Face — `modeling_paddleocr_vl.py`, `configuration_paddleocr_vl.py` |
| Image pre-processing | same repo — `image_processing_paddleocr_vl.py` (`smart_resize`, patchify) |
| Prompt assembly | same repo — `processing_paddleocr_vl.py`, `chat_template.jinja` |
| Pipeline orchestration | `PaddlePaddle/PaddleX` — `paddlex/inference/pipelines/paddleocr_vl/{pipeline,uilts,result}.py` |
| Layout model | `PaddlePaddle/PP-DocLayoutV3` on Hugging Face; config in `paddlex/modules/base/utils/pdparams2safetensors/model_config.py` |
| Pipeline defaults | `deploy/paddleocr_vl_docker/pipeline_config_vllm.yaml` in this repo |

A read-only checkout of PaddleX and the Hugging Face model metadata is expected under
`/home/user/ref/` during development (see `dotnet/tools/reference/README.md`).

---

## Model architecture (as ported)

**`PaddleOCR-VL-1.6-0.9B` = NaViT-style SigLIP vision tower → 2×2 patch-merging projector → ERNIE-4.5 decoder.**

### Vision tower (`PaddleOCRVisionModel`, 27 layers)
- Native-resolution patchify: `smart_resize` to a multiple of `patch*merge = 28`,
  `min_pixels = 112_896`, `max_pixels = 1_003_520`, bicubic, `mean = std = 0.5`.
- `Conv2d(3, 1152, k=14, s=14)` per patch (equivalent to a `588 → 1152` GEMM).
- Absolute position embedding: the pretrained `27×27` grid is **bilinearly interpolated**
  (`align_corners=false`) to the image's `(grid_h, grid_w)` — `interpolate_pos_encoding=True`.
- 27 × pre-norm encoder layers: `LayerNorm(eps=1e-6)` → MHA(16 heads, head_dim 72) → `LayerNorm` → MLP(1152→4304→1152, `gelu_tanh`).
- 2-D RoPE inside attention: `SigLIPRotaryEmbedding(dim = head_dim/2 = 36, theta = 10000)`,
  frequencies indexed by `(row, col)`, concatenated then duplicated to 72 and applied
  `rotate_half`-style. Window attention is **disabled** (`window_size = -1`).
- Attention is **block-diagonal per image** (matches `flash_attn_varlen` with `cu_seqlens`).
- `post_layernorm`; the pooling head is **not** used at inference (`return_pooler_output=False`).

### Projector (`mlp_AR`)
`LayerNorm(1152, eps=1e-5)` → merge 2×2 spatial patches into one token (4608 dims) →
`Linear(4608, 4608)` → exact-erf `GELU` → `Linear(4608, 1024)`.

### Language model (`Ernie4_5Model`, 18 layers)
- `hidden 1024`, `heads 16`, `head_dim 128`, `kv_heads 2` (GQA ×8), `intermediate 3072`, SwiGLU.
- `RMSNorm(eps=1e-5)`, no biases, untied `lm_head`.
- **3-D M-RoPE**: `theta = 500_000`, `mrope_section = [16, 24, 24]`; `t/h/w` position ids come
  from `get_rope_index` — image tokens get a `(t, h, w)` grid, text tokens get `max+1` onward.
- Causal attention with a KV cache; greedy decoding by default.

### Layout (`PP-DocLayoutV3`)
An RT-DETR–style detector: HGNetV2-L backbone → hybrid encoder (`d_model 256`, 3 levels,
strides 8/16/32) → 6-layer deformable decoder, 300 queries, 25 classes, plus mask and
reading-order heads. Input `800×800`, `1/255` rescale, no mean/std, `NCHW`.

The graph has three fetches: boxes `[N, 7]`, a box count, and a `[N, 200, 200]` mask per query.
The masks are not decoration — upstream's default `layout_shape_mode = "auto"` reduces each to a
polygon and uses it both to decide whether two overlapping boxes are really the same region and
to white out everything outside the region in the block's crop. A slanted scan or an L-shaped
column is where that matters. Reproducing it needs `findContours`, `approxPolyDP`, `minAreaRect`,
`fillPoly` and Shapely's polygon intersection, all of which are ported in `Imaging/Contours.cs`
and `Imaging/Polygons.cs` and checked against the originals.

**This model is not hand-ported.** It ships only as a Paddle inference graph — there is no
upstream PyTorch module tree to port from — so `Models/Paddle` interprets the exported graph
directly. Every operator in that interpreter is our own kernel; what we take from Paddle is the
graph topology and the weights, both of which we would have to take anyway. The result is exact
by construction instead of by inspection, and the same interpreter also runs UVDoc and the
orientation classifier.

The interpreter covers ~65 operators: `conv2d` / `depthwise_conv2d` (im2col + GEMM, with a
direct path for depthwise), pooling, batch norm, layer norm, softmax, `matmul` / `bmm` / `einsum`,
`grid_sample` (which is what deformable attention is built on), bilinear and nearest
interpolation, `pad3d`, broadcasting element-wise ops and activations (`relu`, `hardswish`,
`hardsigmoid`, `prelu`, …), reductions, `top_k`, `argsort`, `gather_nd`, `index_put`,
`set_value` and the shape algebra.

Two of Paddle's inference-time conventions are easy to get wrong and are worth naming, because
both produce plausible-looking output rather than an error. `dropout` is not the identity at
inference when its `mode` is `downgrade_in_infer` — it scales by `1 - p`, and `p` arrives as an
input rather than an attribute. And the interpolation operators take their target size three
ways: an `OutSize` tensor, a `SizeTensor` *list* of rank-0 tensors (which is how a size read
from another tensor's shape at run time arrives), or a scale; UVDoc uses the list form.

---

## Project layout

```
dotnet/
  PaddleOcrSharp.slnx
  Directory.Build.props        # net10.0, unsafe, nullable, preview lang, package metadata
  Directory.Build.targets      # attaches the repository README to the packable projects
  src/
    PaddleOcrSharp/            # the library — everything below is here
      Core/                    # Tensor<T>, pooled buffers, SIMD kernels, GEMM
      Formats/                 # safetensors + paddle .pdiparams readers, bf16/f16
      Imaging/                 # SkiaSharp decode, smart_resize, normalize, patchify
      Text/                    # tokenizer (tokenizer.json BPE), chat template
      Models/Vision/           # SigLIP/NaViT encoder + projector
      Models/Language/         # ERNIE-4.5 decoder, KV cache, sampling
      Models/Layout/           # PP-DocLayoutV3 wrapper and detection post-processing
      Models/Paddle/           # Paddle PIR graph interpreter and its operator kernels
      Models/Preprocessing/    # orientation classifier, UVDoc unwarping
      Pipeline/                # orchestration, block prompts, markdown/OTSL assembly
      Download/                # model downloader (static HTTP mirror)
    PaddleOcrSharp.Pdf/        # PDF page rasterisation (the one native dependency)
    PaddleOcrSharp.Cli/        # `paddleocr-sharp` command-line front-end
  tests/
    PaddleOcrSharp.Tests/      # unit + numerical-parity tests
  tools/
    reference/                 # Python scripts that dump upstream reference tensors
```

## Two bicubic resizes, deliberately

The two model families reach C# through different Python image stacks, and their bicubic
resamplers disagree:

| | `PilResize` | `OpenCvResize` |
| --- | --- | --- |
| Used by | the VL model's `smart_resize` | the layout detector's 800×800 input |
| Mirrors | `PIL.Image.resize(BICUBIC)` | `cv2.resize(INTER_CUBIC)` |
| Kernel `a` | −0.5 | −0.75 |
| Downscale | support scaled by `in/out` (antialiased) | fixed support (aliased) |
| Border | kernel truncated | replicated |
| Arithmetic | Q22 fixed point, two uint8 passes | float32 |
| Parity | byte-exact | ≤1 level on ~0.02% of bytes |

SkiaSharp's resampler matches neither and is used only for decoding and encoding.

## Engineering conventions

- **.NET 10 / C# preview.** Use `System.Numerics.Tensors`, `Vector<T>` / `Vector512<T>`,
  `TensorPrimitives`, `ArrayPool<T>` and `MemoryPool<T>`. Hot loops must be allocation-free —
  and "allocation-free" includes the compiler's own: a method that contains a lambda *anywhere*
  allocates its display class on entry, whichever branch runs, so a hot method with a parallel
  and a serial path must keep the lambda in a separate method. That one detail was 437 KiB per
  vision attention layer.
- **Pool through `ArrayPool<T>.Shared`.** It pools buffers up to a gigabyte — the widely repeated
  1 MiB bucket cap is .NET Framework folklore — and it keeps every buffer of a size where
  `ArrayPool.Create` drops everything past `maxArraysPerBucket`. `TensorPool` is the one place that
  policy is stated. Measure the shared pool before adding another.
- **Weights stay in their on-disk dtype** (bf16) so a 0.9B model does not need a 4 GB float32
  shadow copy. The GEMM widens one column panel at a time and reuses it across every activation
  row; widening inside the inner loop instead costs more than the multiply-adds it feeds.
- **Every numerical stage is testable in isolation.** Each module exposes a deterministic entry
  point that the parity tests feed with `.npz` fixtures dumped from the Python reference. That is
  why the public surface is wider than a black-box library's would be; what is genuinely internal
  to one component — the Paddle operator kernels behind the graph interpreter — is `internal`.
  Public members are documented, and `GenerateDocumentationFile` makes a gap a build warning.
- **No `float` accumulation shortcuts** where upstream forces `float32` (softmax, RoPE,
  RMSNorm variance) — match upstream precision decisions exactly.
- **Stay native-AOT publishable.** Both libraries set `IsAotCompatible`, so the trim and AOT
  analysers run on every build; JSON goes through a source-generated context, not reflection.

## Validating against upstream

`dotnet/tools/reference/` holds Python scripts that load the real Hugging Face model, run a
stage, and dump inputs/outputs as `.npz`. `PaddleOcrSharp.Tests` loads the same `.npz` and
asserts the C# output matches within a stage-appropriate tolerance. Fixtures are generated on
demand (they are not committed) — see `dotnet/tools/reference/README.md`.

Tests that need fixtures are skipped, not failed, when the fixture directory is absent, so
`dotnet test` works on a clean clone.

## Where the time goes

`paddleocr-sharp bench` starts by measuring the machine, before it loads anything: the FMA rate
the hardware sustains at each vector width, at one thread and at every thread, and the read
bandwidth at each level of the hierarchy. A shared virtual machine is not a constant — the core
count is whatever the hypervisor schedules, the clock moves with the host's AVX-512 licence state,
and a noisy neighbour halves the memory bandwidth between one run and the next. Absolute
milliseconds from different days are not comparable, so the run prints its own ceilings and the
per-sample spread beside every figure, and `--gemm true` reports each GEMM shape as a fraction of
the ceiling just measured. On the reference machine that spread runs from 10% to 40%, which is
wider than most differences worth acting on: nothing below was accepted without an A/B in one
sitting, with an untouched operator alongside it as a control.

Measured on 4 cores at ~3 GHz, a 980x392 page (1960 patches).

| Stage | Cost |
| --- | --- |
| Vision tower (1960 patches) | ~16 s |
| Decoder (503-token prefill + 32 tokens) | ~3.6 s |
| Layout graph | ~7.6 s |

### A whole page, including the stages no profile covered

`PageProfile` (printed by `parse --profile` beside the block table) names the rest: rasterise,
decode, layout, crop, stack, table-figures, prepare, recognize, encode-figure, otsl-to-html,
assemble, render. It exists because those three profiles between them left a hole — on a short
scanned page `parse --profile` reported 6.2 s of recognition inside a 20.7 s page and said nothing
about the other 14.5 s. `--layout-profile` adds the layout graph's own operator breakdown.

Over six pages of `test_documents/pdf_scanned` and `images/ocr_*`, one page each:

| Stage | Share of the corpus |
| --- | --- |
| vision tower | 67.5% |
| decode | 16.0% |
| layout graph | 8.3% |
| prefill | 8.1% |
| everything else | 0.2% |

**Scanned documents invert the picture the rest of this section was written against.** There, two
runaway decodes were 95% of a page; here the tower is two thirds of the corpus and decode is a
sixth. Dense scanned text gives every block a near-maximal patch count and almost nothing to say:
one block reached 4,840 patches and generated 258 tokens. The consequence for tuning is that
attention's share is not a constant — it is quadratic in patches where every other stage is
linear, so 36% at 1960 patches is 65% at 4900.

The same table settles two smaller questions. The pipeline's own image work — cropping, masking,
stacking, the markup conversions — is 0.2% of a page and is not worth attention. And one layout
detection allocated 4.5 GiB, which turned out to be the largest single thing wrong with the port;
see below.

### Against the original

The port is faster than the pipeline it was ported from. Running upstream's own
`PaddleOCRVL16Pipeline` and this library over the same three pages, back to back on the same
machine, at each side's defaults:

| Page | Python | C# | |
| --- | --- | --- | --- |
| report (4 blocks) | 62.1 s | 33.6 s | 1.85x |
| benchmark (3 blocks, one table) | 49.7 s | 23.2 s | 2.14x |
| lines (1 block) | 20.1 s | 11.5 s | 1.75x |
| **total** | **131.9 s** | **68.3 s** | **1.93x** |

The markdown is byte-identical on all three, so this is the same work done in less time rather
than less work. Both sides use the whole machine: upstream threads inside oneDNN and Paddle, and
this port inside its own GEMM. `--block-concurrency 4` takes the total to about 59 s by
recognising blocks in parallel as well, at the cost of holding several blocks' activations at
once; it is not the default.

Two caveats on that table. Loading the checkpoint is excluded on both sides — cold, the first
memory-mapped read of the 0.9B weights costs around 97 s here and is entirely page-cache traffic,
which is what an earlier and much less flattering comparison had accidentally folded into the C#
column. And the machine moves: the same Python run measured 123.5 s a few days earlier against
131.9 s here, which is why both sides are always measured in one sitting.

Both model halves are GEMM-bound, and the shape of the win is the same in each: give the inner
loop enough reuse that it is compute-bound rather than load-bound. `Gemm.Linear` widens a bf16
column panel once and reuses it across every activation row; `Gemm.MatMul` tiles the output and
picks its kernel from the operand layout; attention blocks 16 query rows at a time so the keys and
values stay in cache across the block; the convolution treats an output row as one GEMM against
its im2col columns rather than a dot product per output pixel.

The layout graph is not GEMM-bound, and for a while it was not bound by anything defensible. The
mask head works on `[1, 300, 200, 200]` tensors — twelve million elements — and its casts,
comparisons and fills ran element at a time. Vectorising them took the stage from 9.3 s to 7.6 s
with `conv2d` flat as a control, and moved the convolutions from a fifth of the graph to a third.
What was actually left, though, was neither: the graph was allocation-bound, and pooling its
intermediates took it from 7.5 s to 4.7 s — see "Where the allocations go".

`PirProfile` (printed by `bench`) reports each operator's total, its slowest single call, and that
call's result shape and Paddle module path. The shape column is what makes the layout graph's cost
legible.

### A page's cost is not the sum of its blocks

`RecognitionProfile` (printed by `parse --profile`) is the VL half's answer to those two: one row
per block, giving the patches encoded, the prompt length, the **tokens generated**, the split
between vision, prefill and decode, the share of decode spent in the output head, and the bytes
allocated. The generated-token count is the column that matters, because a block's cost is set by
how much it says rather than by how big it is.

Decoding is bandwidth-bound on weight streaming, which is what makes that so. Every generated token
re-reads the 18 decoder layers (255M parameters) and the untied `lm_head` (106M), so **721 MB of
bf16 weights per token** before the key/value cache is counted — and the cache adds another 302 MB
per token once the context reaches 8k. The profile shows both halves of that directly: a block
decoding a few dozen tokens runs at ~40 ms/token with the output head taking 19-20% of decode,
while a block that reached 8k runs at ~85 ms/token with the head down to 9% — same weights, same
head, twice the cost, because the cache it re-reads has grown past the weights themselves. The head
is not worth attacking either way: greedy selection needs every logit, so its 212 MB is
irreducible.

The consequence is that a decoder which stops converging is not a quality problem with a
performance footnote, it is the performance problem. Measured on `equations.docx` page 1, fifteen
blocks, before the early stop below:

| | tokens | decode | allocated | share of page |
| --- | --- | --- | --- | --- |
| two runaway `text` blocks | 16,384 | 1,379 s | 5.1 GiB | 94.9% |
| the other thirteen blocks | 332 | 13 s | 0.26 GiB | 5.1% |

Both runaway blocks stopped only on the 8192-token budget, and every token past the first few
hundred was discarded by `RepetitionTruncator` immediately afterwards.

### Stopping a decode that has fallen into a cycle

`GenerationOptions.StopOnRepetition` (on by default; `--stop-on-repetition false` to disable) ends a
block once its token stream is `RepetitionRepeats` verbatim copies of one period of at most
`RepetitionMaximumPeriod` tokens, and only after `RepetitionMinimumTokens` have been generated.
Greedy decoding makes the test exact rather than statistical: there is no sampling noise to see
through, so a tail that has repeated six times is a cycle, not a coincidence.

The thresholds sit well past what `truncate_repetitive_content` needs to fire — it acts on five
repeats of an eight-character unit — so what is skipped would have been cut from the string anyway.
The 384-token floor is what keeps ordinary blocks out of the check: their whole output is shorter
than that, so they never reach it.

Measured over the nine-page benchmark corpus, at defaults:

| | before | after | |
| --- | --- | --- | --- |
| `equations.docx` p1 | 1,472 s | 123 s | 12.0x |
| that page's tokens | 16,716 | 1,100 | 15.2x |
| that page's allocations | 5.3 GiB | 445 MiB | 12.2x |
| whole corpus (9 pages) | 2,123 s | 595 s | 3.6x |

The page's three figures are one A/B in a single sitting, `--stop-on-repetition false` against the
default; the corpus row is the earlier nine-page sweep, so its page-1 number (1,615 s) is a
different day's measurement of the same run.

Once the runaway blocks are gone that page is fifteen small blocks and 48% vision, which is the
shape `--block-concurrency` was meant for: 123 s at the default 1, 95 s at 2, 84 s at 4, with
byte-identical output. It stays at 1 by default because the win is a property of this shape — a
page of few large blocks has no idle cores to fill — and the cost is holding several blocks'
activations at once.

#### Upstream has the same runaway, and no stop for it

The early stop is a deliberate divergence, so it is worth being precise that what it skips is not
something upstream keeps. Upstream cannot stop either, by construction:

- PaddleX's local backend builds `generate_kwargs` out of **`max_new_tokens` and nothing else**
  (`doc_vlm/predictor.py`); `repetition_penalty`, `temperature` and `top_p` are each warned about
  and dropped. The server backends send a temperature and a token cap. The default budget is
  `PADDLEOCR_VL_MAX_NEW_TOKENS = 8192`, which is where our own default comes from.
- The checkpoint's `generation_config.json` sets only `eos_token_id`, `pad_token_id` and
  `use_cache` — no `no_repeat_ngram_size`, no stopping criteria — and the model is a stock
  `GenerationMixin`. Greedy decoding there ends on the stop token or on the budget.
- `truncate_repetitive_content` exists precisely because of that: upstream generates the runaway
  tail and then throws it away.

Measured rather than inferred, with `tools/reference/probe_runaway_decode.py`, on the crop of the
block that runs away — upstream's own checkpoint through `transformers`, greedy, no penalties:

| | tokens | stopped on EOS | time |
| --- | --- | --- | --- |
| upstream, capped at 400 | 400 | no | 19 s |
| upstream, at its own 8192 default | 8,192 | no | **525 s** |

One block, nearly nine minutes. Running upstream over **every** block of that page the same way
(`--blocks`, the port's layout boxes so both sides see the same fifteen blocks) puts a number on
the whole stage:

| | blocks | tokens | recognition |
| --- | --- | --- | --- |
| upstream, through `transformers` | 15 | 16,741 | **1,479 s** |
| this port, before the stop | 15 | 16,716 | 1,462 s |
| this port, after the stop | 15 | 1,100 | 118 s |

Two of upstream's fifteen blocks never stopped and took 1,428 s of that 1,479 s — 97%. The two
sides agree to within 1.2% on time and 25 tokens before the stop, which is the fidelity check that
makes the third row meaningful: after it, the port does the same page's recognition **12x faster
than the original**, and the difference is entirely tokens not generated.

Layout detection and markdown assembly are outside that comparison; on the port's side they are
about 10 s, so including them would not move it. The port's rows are its profile's vision + prefill
+ decode; upstream's is wall time around `generate` with the model already loaded. The output is
`This is text.` repeated until the budget runs out, which is the same text this port produced
before the stop.

The two implementations then agree on what survives, which is what pins the divergence down to
string length rather than to the port. Upstream's own `truncate_repetitive_content`, at the
`min_count` its pipeline passes for a non-table block (50, and 5000 for a table — the values
`RepetitionTruncator` uses), returns:

| given | returns |
| --- | --- |
| upstream's 8,192-token output (28,671 chars) | `This is text.` ×3 |
| this port's stopped 384-token output (1,343 chars) | `This is text.` ×1 |

Both are upstream's function on the respective strings; the shorter one takes the
shortest-repeating-unit branch instead of the suffix branch. So the port's pre-stop output matched
upstream exactly, and its post-stop output is what upstream's own truncator makes of a shorter
decode of the same cycle. Neither is the page's text — the source repeats that sentence
thirty-one times and no decode of it terminated.

**Eight of the nine pages come out byte-identical**, and the detector fired on exactly two blocks in
the whole corpus — the two runaway ones. The ninth page differs only in how many copies of a
sentence the source repeats thirty-one times survive: three before, one after. Both are the
truncator's arbitrary reduction of a decode that never terminated, so neither is the page's text;
what changed is which arbitrary reduction, because the truncator takes a different branch on a
shorter string. On the corpus's character-accuracy metric that costs 0.4 points of mean, all of it
on that one page.

### Where the allocations go

A stage that allocates a lot is not automatically slow, so this is worth separating from the
timings above. Two of the three big allocators turned out to be fine and the third was the
layout graph's whole problem.

| Stage | Per call, before | After |
| --- | --- | --- |
| layout detection | 4,458 MiB, every time | 1,038 MiB cold, then **145 MiB** |
| vision tower | 411 MiB cold | 4 MiB per steady pass |
| decoder prefill | 154 MiB cold | 3 MiB per steady pass |

The tower and the decoder were already pooled; their cold figures are the pool filling, which is
why the per-block column of `RecognitionProfile` shows one large block at 672 MiB and the next
identical one at 67. Reading those as a leak is a mistake worth naming, because it sends you
looking in the wrong half of the pipeline.

**The layout graph was allocation-bound.** `TensorArena` gives a run's intermediate tensors pooled
storage, and what makes that safe is knowing when a buffer is really dead — a tensor's array is
not its own, since `reshape`, `cast` between same-width dtypes and `share_data_` all return a
tensor over an existing array. The arena counts, per array, how many of the interpreter's value
slots reference it, and returns the array at zero. The interpreter is the only place a run's
tensors live and every operator's results land in slots it names, so the count is exact.
Interleaved against the control at constant load, four detections in one process: **4,458 MiB and
7.5 s per detection become 145 MiB and 4.7 s**.

That result is also the answer to two experiments in the list below that came out neutral. Neither
vectorising the graph's element-wise fallback nor taking the collector off a core did anything,
because neither addressed the allocation.

Pooling a buffer that a fresh allocation would have zeroed is a real hazard:
`GC.AllocateUninitializedArray` of this size arrives zeroed in practice, so an operator that read
its own output before writing it would have been quietly correct and would now produce garbage.
`PADDLEOCR_SHARP_POISON_ARENA=1` fills every rented buffer with `NaN` and `long.MinValue` first;
the corpus is byte-identical with it on, which is what says no operator does that.

The 145 MiB that remained was not scaffolding, which is what it looked like. `PirProfile` now
reports allocation per operator beside the timings, and it put 128 of the 145 against a single
operator — so the guess that it was a shape array and a results array per operator, several
thousand times over, was wrong twice: wrong about the shape, and describing something that adds up
to 2.5 MiB.

`TensorArena` reports its own accounting under `PADDLEOCR_SHARP_ARENA_STATS=1`, which is what
identified it. Per detection, steady state:

```
arena: 1611 rents of 6359 MiB, 3 missed the pool (0 MiB allocated); returned 6231 MiB at
last use and 0 MiB at the end; 128 MiB kept as fetches
```

**The 128 MiB was the fetches.** A run's intermediates all go back at their last use, but its
fetches cannot — they are the caller's. So each run took a bucket's worth of buffers out of
circulation and the next run allocated the shortfall again, and for a `[1, 300, 200, 200]` mask
that is 128 MiB a page in perpetuity. `RunPooled` returns a `PirRunResult` that gives those
buffers back when disposed; all three callers copy what they need out of the fetches and drop
them, so all three use it.

A further 8 MiB was the graph's *input* tensor, built before the run and so never seen by the
arena inside it. Rented and returned by hand.

| | per detection |
| --- | --- |
| before the arena | 4,458 MiB |
| arena | 145 MiB |
| fetches returned to the pool | 17 MiB |
| input tensor rented | **9 MiB** |

What is left is the scaffolding, and it is small: `batch_norm_`'s two per-channel arrays at 12 KiB
a call are the largest line at 1 MiB, and every operator together is 2.5 MiB. The rest is the
`object?[]` of value slots, the gathered mask bytes the polygon extractor copies out, and the
boxes themselves.

Those figures are the bench's, which feeds the same input every iteration. A real document is less
tidy, because the graph has content-dependent shapes — what `top_k` selects and what the gathers
index are functions of the page — so a new page can ask for a bucket the pool has not got yet.
Three pages of `multi_page_scanned.pdf` in one process allocate **1.6 GiB across three
detections**, against 13.4 GiB for the same three before the arena. The effect is self-limiting:
the pool accumulates the buckets a corpus needs and stops growing. A page parsed in a process of
its own still pays the cold ~1 GiB and nothing else, which is what the per-page corpus runs show.

The number worth keeping an eye on is the first one in that stats line: **1,611 rents of 6,359
MiB**. That is the graph's intermediate volume, and it is now pooled rather than allocated, but it
is still 6.4 GiB of buffer traffic per detection. Most of it is width: the interpreter holds every
integral dtype as `long`, so a twelve-million-element boolean mask is 96 MB where it could be 12.

#### `ArrayPool<T>.Shared` is not the pool its reputation says

`TensorPool` existed because "the default shared pool caps buckets at 1 MiB (2^20 bytes)". That
was true of .NET Framework and has not been true for years. Measured, the shared pool round-trips
a **1 GiB** float array with zero allocation, at every size from 1 MiB up.

Believing it made things worse rather than neutral, because `ArrayPool.Create(_,
maxArraysPerBucket)` keeps only that many buffers of a size and drops the rest, where the shared
pool keeps them. Over a rent-and-return round of N live 4 MiB buffers:

| N | shared | created(16) |
| --- | --- | --- |
| 16 | 0 MiB | 0 MiB |
| 36 | 0 MiB | 16 MiB |
| 64 | 0 MiB | 128 MiB |

Thirty-six of one size is exactly what `KvCache` holds, two per layer, and one growth at the
decoder's geometry is 576 MiB — so a cache on a created pool would churn 66 MiB per block where
the shared pool churns nothing. `KvCacheTests` pins it. **Before adding a pool of your own here,
measure the shared one.**

### Where the vision tower's time goes

`bench` prints a stage profile for the tower (`StageProfile`, the hand-written towers' answer to
`PirProfile`), and attention now records its own parts from inside — it runs a thread per head, so
nothing outside it can attribute its time. Each thread accumulates its own and they are summed, so
the figures are thread-seconds; over a stage that keeps every core busy that is what makes the
parts comparable with each other.

At the 4900-patch page a scanned block actually produces, before the work below:

| Stage | Share | GFLOP/s |
| --- | --- | --- |
| attention | 65.5% | 73 |
| MLP matrix products | 19.6% | 215 |
| QKV projections | 7.9% | |
| output projection | 2.7% | |
| rotary + head shuffles, GELU, norms, residuals | 4.3% | |

The rate column is the whole argument: attention carried more arithmetic than the MLP products and
ran at a third of their rate, on the same machine, against a measured all-thread FMA ceiling of
285 GFLOP/s.

Four changes took it from 40.8 s to 25.1 s, each A/B'd in one sitting with the untouched stages as
controls:

| | attention | scores | values |
| --- | --- | --- | --- |
| before | 40.8 s | | |
| parallelism capped at the core count | 36.6 s | 150.2 ms | 169.4 ms |
| transposed keys | 36.6 s | 128.4 ms | 169.4 ms |
| chunked value reduction | 29.1 s | 128.4 ms | 96.1 ms |
| token block outermost | **25.1 s** | **98.2 ms** | **96.1 ms** |

- **The degree of parallelism was unbounded.** `Parallel.For`'s default lets the pool inject
  workers while a queue stays non-empty, which is a policy for work that blocks. None of this
  blocks, and attention holds megabytes per worker at this page size, so the extra threads divided
  the cache and added switches. The instrumentation said so directly: the thread-time sum was 6.8x
  the stage's wall time on a four-core machine, and capping it took that to 3.89x. `Parallelism`
  now holds one shared `ParallelOptions` for every kernel that spreads work.
- **The score product reduces over 72 columns**, so the general kernel finishes every output with
  a lane reduction, and its four accumulators cannot fill two FMA ports at four cycles of latency.
  Transposing each head's keys to `[headDim][tokens]` makes the lanes output columns: the tile
  becomes four query rows by two column vectors — eight chains, six loads per eight multiply-adds.
  One transpose per head per layer against the `tokens / 16` passes the product makes over them.
- **That is the experiment recorded below as neutral**, and it was: with six or seven threads on
  four cores the stage was contention-bound and no kernel change could show through it. Both
  measurements were right, which is worth remembering before dismissing a mechanism twice.
- **The score kernel's loop order.** A tile of transposed keys is 4.6 KB and the block's row groups
  sweep it from L1; row group outermost instead read the head's whole 1.4 MB key matrix once per
  group.
- **The value product was store-bound.** It kept its output rows in memory and streamed the values
  past them, so the loop-carried dependency was a store and a reload of every output element on
  every one of 4900 reduction steps — one store per multiply-add, against a single store port —
  and it swept the values once per group of four output rows. Chunking the reduction at 80 tokens
  puts the output tile in registers for the chunk and keeps the chunk's values in L1 across the
  tiles that sweep it. The chunk is sized for L1 deliberately: sized for L2 the kernel simply
  becomes L2-bound and gains nothing.

End to end on `code_and_formula_scanned.pdf` page 1 — blocks up to 4,840 patches — bracketed by two
control runs in the same sitting: **vision 200.6 s -> 160.0 s (-20%), the page 270.0 s -> 230.1 s
(-15%)**, with layout, prefill and decode flat and the markdown byte-identical.

The win scales with block size, because attention's share does. On pages whose blocks are 600-800
patches the same build measures within noise of the control, and the corpus splits accordingly:
-20% of vision where blocks approach the pixel budget, ~0% where they are a seventh of it.

Two earlier notes in this section are superseded rather than wrong. The isolated kernel rates
(18.4 and 17.8 GFLOP/s) were measured under the uncapped pool, and the reasoning that the score
product's lane reduction is "the cost whatever the tile" holds only while the tile keeps reducing
along the lanes — the point of transposing is to stop.

### Things that looked like wins and were not

Each of these is a plausible optimisation that the benchmark rejected. They are recorded because
the argument for them is still convincing on paper, and someone will otherwise try them again.

- **Banding `Gemm.Linear` over activation rows**, so a band stays cached across all the column
  panels. Consistently slower; the panel loop's traffic is evidently already absorbed by the
  shared cache.
- **Interleaving the weight panel** so the kernel's four weight loads are consecutive rather than
  four streams a row apart. Neutral at best and a 40% loss if the row-major copy is kept alive
  beside it, which doubles what each row-block sweeps through L2. The strides involved are not
  powers of two, so the prefetcher was already handling them.
- **Sizing attention's row-block from the token count**, so a page's keys and values are read
  half as many times. No measurable change, which is what rules key and value bandwidth out as
  attention's limit.
- **Giving attention's score product the four-by-four tile `Gemm.Linear` uses.** The argument is
  that `Dot4` issues five loads per four multiply-adds where `Dot4x4` issues eight per sixteen, so
  the wider tile should be load-bound where the narrow one is not. Exactly neutral, and the reason
  is worth keeping: both finish every output with a lane reduction, and over a 72-long inner
  dimension that reduction is the cost, at an identical 0.111 per multiply-add in both shapes.
- **Hoisting attention's value reduction outermost**, so the head's value matrix is read once per
  block instead of once per group of four rows — a fourfold cut in what looked like 20 GB/s of
  traffic. Neutral, because that traffic is served from L2 and L3, which are nowhere near their
  measured 51 and 22 GB/s.
- **Holding the value product's output tile in registers**, four rows by sixteen columns. A 15%
  loss: covering 64 columns then takes four passes over the value matrix, each touching a quarter
  of every row, which trades one streaming pass for four strided ones.
- **Transposing each head's keys so the score product never reduces along the vector lanes.**
  Measured neutral in the tower — 7,955 ms against 7,948 ms — and reverted. **This one was later
  re-measured and kept**: see the section above. The stage was contention-bound at the time, with
  six or seven pool threads on four cores, and no kernel change could show through that. The
  measurement was correct and the conclusion drawn from it was not, which is the trap worth
  keeping: a stage that is bound by something other than its kernel will report every kernel
  change as neutral, so establish what a stage is bound by before A/B-ing its inner loop.
- **Vectorising the layout graph's broadcast fallback.** Each element-wise kernel has fast paths
  for the contiguous shapes and a per-element delegate through `double` for everything else, and
  the mask head lives in that everything else: `[1, 300, 200, 200]` against `[1, 300, 1, 1]`
  broadcasts over the *leading* dimensions, so twelve million elements each paid two delegate calls
  and two conversions, and the operator moved 410 MB/s where the machine reads at 10 GB/s. A
  general plan — take the longest suffix of dimensions over which every operand is contiguous or
  constant, hand that suffix to the same vector kernels, walk the rest with a counter — covers
  every shape and removes the delegate entirely. Neutral: five warm layout runs measured 6.7-7.7 s
  against the control's 6.3-8.7 s, and a page measured 10.5-11.7 s against 11.7-12.3 s. Per-operator
  timings swung ±25-120% on operators the change never touched, which is what says the graph is not
  where its own profile appears to put it. Reverted, with the vectorised `Widen`, `Narrow` and
  `Truncate` conversions that went with it. **The reason is now known**: the graph was
  allocation-bound, and pooling its intermediates was worth 37% where the arithmetic was worth
  nothing.
- **Turning off background GC**, on the argument that a collector thread is a fourth claimant on
  four cores while the layout graph churns 4.5 GiB of large tensors. First measured as 8.3-9.0 s
  against 5.6-6.1 s over five warm layout runs, which is a large clean-looking separation and was
  entirely an ordering artefact: the configurations ran in sequence, and the whole sequence drifts
  downward as the 125 MB of layout weights settle into the page cache. Interleaved — on, off, on,
  off — the warm medians are 5.12, 5.27, 5.19, 5.32 s. **Interleave the configurations, or the
  first one measured pays for the page cache the others inherit.** The premise was half right and
  aimed at the wrong half: the churn was the problem, but the answer was to stop churning rather
  than to collect it faster.
- **Pooling `RgbImage`'s buffers through a pool sized past the shared one's supposed cap.** The
  premise was the stale cap above, so the change was pointless; worse, routing `Clone` through it
  rounds each image up to a power of two, and on a page's handful of differently-sized crops that
  cold-pool slack costs more than it saves — `stack` measured 35 MiB against the control's 20.
  Reverted. Images are a few tens of MiB a page against a gigabyte for the rest, so there was
  little to win either way.
- **Moving `KvCache` off `ArrayPool<T>.Shared`.** Same stale premise, and the measurement is the
  other way round: 0 MiB a round on the shared pool against 66-130 MiB on a created one. Reverted,
  and a test now pins it. The one thing worth keeping from the exercise is the shape of the mistake
  — a diagnosis inherited from a comment, acted on without measuring the thing the comment claimed.
- **Using AVX-512 where the runtime's preferred width says 256.** A dependency-free FMA loop is
  60% faster at 512 bits, and the ISA is reachable regardless of the policy — but every 512-bit
  GEMM variant measured slower than the 256-bit kernel, including a narrowed tile chosen to fit
  the register file. See `Core/Simd.cs`; the override is left to whoever measures their own
  machine.

### Two traps that make a SIMD microbenchmark lie

Both were hit while building the calibration above, and both report roughly a fifth of the truth
while looking entirely reasonable.

- **Accumulators that start equal.** Eight chains seeded to zero and updated identically are
  provably the same value, so the JIT emits one `vfmadd231ps` reusing a single register. The loop
  then measures FMA latency rather than throughput. Seed them differently.
- **Consuming the result through a `volatile` field.** The release barrier makes the JIT keep
  every SIMD local in the frame across the loop, so the body becomes load-operate-store and
  reports store-forwarding throughput. `GC.KeepAlive` prevents dead-code elimination without
  touching register allocation.

A third, related: an indexed accumulator — an array or a `stackalloc` span — is never
enregistered, so the chains have to be named locals.

## What the pipeline does beyond the models

Roughly half the port is not model code. These are the stages that decide what the models are
asked and what becomes of their answers, all of them checked against the upstream functions
named beside them:

| Stage | Upstream |
| --- | --- |
| Drop overlapping regions, consulting their outlines | `filter_overlap_boxes` |
| Stack a paragraph split across columns into one image, and reorder around it | `merge_blocks`, `merge_images` |
| Stretch a formula crop's contrast and trim its margins, upscale a small spotting crop | `crop_margin`, `pre_process_for_spotting` |
| Cover figures inside a table with `[Fn]` placeholders and put them back afterwards | `tokenize_figure_of_table`, `untokenize_figure_of_table` |
| Cut runaway repetition out of a block's output | `truncate_repetitive_content` |
| OTSL markup to HTML | `convert_otsl_to_html` |
| Spotting's `<|LOC_n|>` coordinates to polygons | `post_process_for_spotting` |
| Number the blocks that belong to the reading flow | `update_order_index` |
| Render the page, HTML decoration and all | `MarkdownConverter`, `build_handle_funcs_dict` |
| Rejoin a table split by a page break | `merge_tables_across_pages` |
| Decide how deep each heading sits | `assign_levels_to_parsing_res` |

Three places diverge from upstream on purpose, and each says so where it is implemented: the
token glyphs painted over a table's figures (SkiaSharp, not OpenCV's Hershey font), the
clustering behind heading levels (an exact one-dimensional k-means, not scikit-learn's seeded
local search), and the shuffle that assigns those token numbers (a stable bijection, not
Python's Mersenne Twister). Two upstream quirks are reproduced rather than repaired, because both
decide what a consumer actually gets: the doubled quote in `untokenize_figure_of_table`'s `alt`
attribute, and `crop_margin` asking OpenCV for a BGR-to-grey conversion of a buffer that is RGB,
which swaps the red and blue weights and so decides which pixels of a formula survive the trim.

## Working agreements for this port

1. Read the upstream Python for a stage **before** writing the C# for it. Quote the file and
   line range in the C# doc-comment so the mapping is auditable.
2. Land work in vertical slices that build and test green. Track progress in
   [`to-do.md`](to-do.md).
3. Where a stage cannot be reproduced exactly — OpenCV's SIMD rounding, OpenCV's Hershey font
   for the table-figure placeholders — say so in the doc comment and pin down what *is*
   guaranteed instead.
4. Upstream CI, pre-commit hooks and agent skill files in this repository are **disabled**
   (renamed to `*.disabled`) for the duration of the port; do not re-enable them.
5. Do not modify the upstream Python packages (`paddleocr/`, `ppocr/`, `ppstructure/`, …)
   — the port is additive and lives entirely under `dotnet/`.

## Publishing

`.devops/build-nuget.yml` builds, tests and publishes on `main`: `PaddleOCR`, `PaddleOCR.Pdf`,
and `PaddleOCR.Cli` — the last as a .NET tool, so `dotnet tool install -g PaddleOCR.Cli` puts
`paddleocr-sharp` on the PATH. The package ids drop the `Sharp`; the assemblies and namespaces
keep it, as `HNSW` and `HNSW.Net` do in the sibling repository. Versions are
CalVer (`yy.M.<build id mod 65536>`), stamped by the pipeline; `Directory.Build.props` carries
0.1.0 for local packs. The tool package is the project's publish output — that is how the
SkiaSharp and PDFium native assets get in — so it is the one project `dotnet pack --no-build`
cannot pack.
