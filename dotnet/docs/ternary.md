# Ternary weights for PaddleOCR-VL

This document specifies the ternary quantization path: what Bonsai's encoding is, which parts of
it we adopt, the file the converter writes, the runner that reads it, and how a converted model is
proved equivalent enough to ship.

It is a design, not a report. Where a number appears without a measurement beside it, it is an
expectation derived from the figures already in [`CLAUDE.md`](../../CLAUDE.md), and it is labelled
as such.

---

## 1. What Bonsai actually does

Sources, all read rather than inferred: `theolivenbaum/Bonsai-demo` (`MODEL-FORMATS.md`, `AGENTS.md`,
`README.md`, `ternary-bonsai-8b-whitepaper.pdf` §2.1, `bonsai-2-27b-whitepaper.pdf` §2.1-2.4), and
the `PrismML-Eng/llama.cpp` fork, which is where the bit layouts live:
`ggml/src/ggml-common.h`, `ggml/src/ggml-quants.c`, `src/llama-model.cpp`.

The approach separates into four pieces, and it is worth keeping them separate because we adopt
three of them and cannot adopt the fourth.

### 1.1 The representation

A weight is `t · s` with `t ∈ {−1, 0, +1}` and one FP16 scale `s` shared by a group of **128**
consecutive weights **along the input dimension**. A trit carries `log₂3 ≈ 1.585` bits, so

```
b_eff = log₂3 + 16/128 ≈ 1.71 bits/weight
```

which is the whitepapers' headline 9.3-9.4× against FP16.

### 1.2 The two packings

Both are group-128 containers over the same codec: the scale is `d = amax(group)` and the code is
`round(w/d)`. Neither *performs* ternarization — see §1.5.

`PQ2_0`, ggml type **142** (`block_pq2_0`):

```c
typedef struct {
    ggml_half d;               // 2 B
    uint8_t   qs[128 / 4];     // 32 B, 2 bits per weight
} block_pq2_0;                 // 34 B per 128 weights = 2.125 bpw
```

Weight `j` sits at bit offset `(j % 4) * 2` of byte `j / 4`, and the stored code is `t + 1`, so the
four codes mean `{−1, 0, +1, +2}`. The `+2` is reachable by the container and unreachable by a
ternary checkpoint; upstream's `Q2_0` (type 42) is the identical codec at group 64, 18 B, 2.25 bpw.

`PTQ1_0`, ggml type **143** (`block_ptq1_0`):

```c
typedef struct {
    uint8_t   qs[24];          // 5 trits per byte -> 120 weights
    uint8_t   qh[2];           // 4 trits per byte ->   8 weights
    ggml_half d;               // 2 B
} block_ptq1_0;                // 28 B per 128 weights = 1.75 bpw exactly
```

Five base-3 digits fit a byte because `3⁵ = 243 ≤ 256`. The packed byte is not the base-3 value but
`ceil(q · 256 / 243)`, which lets the decoder recover digit `n` as `((q · 3ⁿ) * 3) >> 8` — a
multiply and a shift, no division. Weights are **not** stored in order: `qs` is filled in stages of
`{32, 16, 8}` bytes (at 24 bytes: one 16-byte stage then one 8-byte stage), and within a stage of
`c` bytes, byte `m` holds weights `m, m+c, m+2c, m+3c, m+4c` most-significant digit first. `qh`
holds its four digits shifted up by one power of three so the decoder's `pow3[n]` table is shared.
This is upstream's `TQ1_0` layout at block 128 instead of 256.

### 1.3 The rotated basis

Bonsai 2 stores weights in a fixed rotated basis and transforms the activation at run time:

```
f(x) = W(Rx),   R = (1/√n) Hₙ S,   n = 1024
```

`Hₙ` is the Sylvester–Walsh–Hadamard matrix — `H[r][c] = −1` iff `popcount(r & c)` is odd, `+1`
otherwise — and `S` is a fixed diagonal of ±1. In the fork the rotation is materialised as an
`n × n` F32 tensor built exactly that way (`llama-model.cpp`, ~line 2011) and applied by an ordinary
`mul_mat` carrying a `GGML_HINT_SRC0_IS_HADAMARD` so a backend can substitute a fast Walsh–Hadamard
transform. The transform is `O(n log n)`; it costs about 10 k flops per 1024-wide activation row
against millions in the projection it precedes.

The point of it is not compression. A rotation spreads each coordinate's magnitude over the block,
so the outliers that make low-bit PTQ fail are flattened before the grouping sees them. It is the
single most load-bearing thing we can adopt, because unlike §1.5 it needs no training.

The file carries the rotation as metadata, and we copy the convention verbatim:

| key | meaning |
| --- | --- |
| `prism.hadamard.version` | `1` |
| `prism.hadamard.block_size` | power of two, 1024 for Bonsai 2 |
| `prism.hadamard.transform` | `normalized-sylvester-walsh-hadamard` |
| `prism.hadamard.axis` | `input-last-dimension` |
| `prism.hadamard.sign_mode` | `identity` or `explicit` |
| `prism.hadamard.sign_widths` / `sign_values` | the ±1 diagonals, one per input width |
| `prism.hadamard.weight_names` | **the explicit list of tensors the rotation applies to** |
| `prism.hadamard.inverse_weight_names` | tables whose rows are rotated and must be un-rotated after a lookup |

The name list matters more than it looks. It is what lets a loader refuse a file whose rotation it
would not apply everywhere it must — the fork throws rather than run half-rotated math, and so do we.

### 1.4 A named keep-list at full precision

Bonsai 2 leaves a short, explicitly named set of tensors unquantized and unrotated: the recurrent
state path, every normalization, and the q/k norms. That is 26.2 M parameters — 0.0976% of the
model, 52 MB — and it moves the checkpoint from 1.71 to 1.72 bits per weight. The lesson is the
shape of the decision, not the list: **precision is a per-tensor policy carried in the file**, and
the cost of exempting the sensitive few is a rounding error.

### 1.5 The part we cannot copy

`quantize_row_ptq1_0_ref` and `quantize_row_pq2_0_ref` are **container writers**. Each takes
`d = amax` over the group and rounds; for weights that are already ternary at that group size the
round-trip is exact, which is precisely the case the fork's own comment claims ("lossless for
checkpoints that are already ternary at group 128"). `quantize_ptq1_0` even discards the imatrix
argument, with the comment *"ternary codes come from the weights themselves; an imatrix has no
role"*.

So Bonsai's quality does not come from the file format. It comes from a quantization-aware training
and distillation pipeline that produces ternary weights in the first place, which we do not have and
are not going to acquire by reading a GGUF.

**The consequence for this project, stated once and plainly:** post-training ternarization of
PaddleOCR-VL-1.6-0.9B will not land where a QAT'd 27B lands. This was written as a prediction and
is now a measurement — see §8, where the first real conversion reads a boarding pass as
`POOOOO / IOOOOOEEEE…` at 0.00% character accuracy. The model is thirty times smaller, so
each weight carries more of the function; and OCR fails visibly — a wrong glyph is a wrong glyph,
where a language model's slightly worse paraphrase is not obviously anything. What we can bring
without training is the rotation (§1.3), GPTQ-style error compensation against real activations,
and a per-tensor policy that keeps the tensors validation says are sensitive at a higher width.
The design is therefore built so that **the bit width is a decision per tensor, taken by
measurement**, and "ternary everywhere" is one point in that space rather than the premise.

---

## 2. Scope

In: `PaddleOCR-VL-1.6` — the 27-layer SigLIP/NaViT vision tower, the `mlp_AR` projector, and the
18-layer ERNIE-4.5 decoder including `lm_head`. That is ~1.8 GB of the pipeline's footprint and
91.6% of a page's time (`CLAUDE.md`, "Where the time goes").

Out, for now: `PP-DocLayoutV3`, `UVDoc` and the orientation classifier. They reach the machine
through `Models/Paddle`'s graph interpreter rather than through `Gemm.Linear`, so quantizing them
means quantized `conv2d` kernels inside that interpreter — a much larger job, for 125 MB and 8% of
a page, on a stage that is compute-bound rather than bandwidth-bound. §7 says what it would take.

---

## 3. The file

A quantized model directory is the unquantized one with `model.safetensors` replaced by
`model.gguf`:

```
PaddleOCR-VL-1.6-ptq1_0/
  model.gguf            # weights, quantized and not
  config.json           # unchanged
  tokenizer.json        # unchanged
  preprocessor_config.json, generation_config.json, ...   # unchanged
```

GGUF v3, little-endian, `general.alignment = 32`, with the fork's type ids so a tensor written here
is byte-identical to one the fork would write. The tokenizer is **not** re-encoded into GGUF KV:
`tokenizer.json` already loads, re-encoding a BPE buys nothing, and the port's `BpeTokenizer` reads
the original. Likewise `config.json` stays authoritative for hyper-parameters; the GGUF carries a
mirror of them under `paddleocr.*` only so the file is self-describing to a human with a GGUF
viewer.

Tensor names are the checkpoint's, verbatim — `model.layers.7.mlp.up_proj.weight`. This is not a
llama.cpp architecture and will not load in llama.cpp; what "bit-compatible with the fork" buys is
that `gguf_dump.py` and the fork's own tooling can read the tensors, and that a block written here
decodes identically under the fork's kernels.

### 3.1 Quantization metadata

Beyond the `prism.hadamard.*` keys of §1.3:

| key | type | meaning |
| --- | --- | --- |
| `general.architecture` | string | `paddleocr-vl` |
| `general.file_type` | u32 | the dominant ggml ftype |
| `paddleocr.quantization.version` | u32 | `1` |
| `paddleocr.quantization.method` | string | `rtn`, `rtn+hadamard`, `gptq+hadamard` |
| `paddleocr.quantization.group_size` | u32 | 128 |
| `paddleocr.quantization.calibration` | string | what the Hessians were collected over, or empty |
| `paddleocr.quantization.source` | string | the checkpoint this was converted from |

Per-tensor precision needs no key: a tensor's ggml type is in its own tensor info, which is where a
reader looks anyway.

---

## 4. The runner

The port has one choke point for every matrix in both hand-ported halves:

```csharp
Gemm.Linear(x, rows, inner, WeightMatrix weight, bias, y, cols);
```

and `WeightMatrix` already carries its storage in the on-disk dtype and widens it inside the
kernel. The quantized runner is therefore not a second model implementation — it is a third storage
kind on `WeightMatrix`, and every call site above it is unchanged.

Two paths, and the split already exists:

- **Prefill and the vision tower** take `Gemm.RunPanel`, which widens a column panel into scratch
  once and reuses it across every activation row. A quantized panel decodes into that same scratch.
  The decode cost is amortized over the rows, so the arithmetic is unchanged and the win is
  footprint, not time.
- **Decode** takes `Gemm.RunNarrow` → `WeightMatrix.Dot4`, which reads every weight exactly once
  and is bound by line arrival. Here the packed weights are read *as packed*: four rows of trits
  decoded into vectors and multiplied. `CLAUDE.md` measures 721 MB of bf16 weights per token; at
  1.75 bpw that is 79 MB. **Expected**, not measured: this is where the change pays.

A weight that is stored rotated carries its `HadamardRotation`, and `Gemm.Linear` transforms the
activation into pooled scratch before the product. The transform is per call, so `q_proj`, `k_proj`
and `v_proj` each pay for the same rotation of the same activation; the fork memoizes that and we
can too, once a profile says it is worth the plumbing.

`WeightStore` gains a GGUF-backed source beside its safetensors one, so
`PaddleOcrVLModel.Load(directory)` picks whichever of `model.gguf` / `model.safetensors` is present
and nothing downstream knows the difference.

### 4.1 What stays float

By default, and overridable per tensor:

| tensor | why |
| --- | --- |
| every `*_layernorm.weight`, `*norm.weight`, `layer_norm{1,2}.*` | Bonsai keeps these; they are kilobytes |
| every bias | same |
| `embed_tokens.weight` | a gather, not a product; quantizing it costs a decode per token row |
| `vision_model.embeddings.position_embedding.weight` | bilinearly interpolated at run time |
| `patch_embedding.weight` | 588×1152, one product per page |

`lm_head` is 106 M parameters and re-read on every generated token, so it is the single most
valuable tensor to quantize — and the one whose error lands directly on the argmax. It is quantized
by default and is the first thing §6 looks at.

---

## 5. The converter

`dotnet/tools/PaddleOcrSharp.Quantize` — a console project, not packaged, since models are
regenerated rarely and the shipped CLI should stay lean.

```
paddleocr-quantize convert  --in <dir> --out <file.gguf> [--scheme ptq1_0|pq2_0|bf16]
                            [--policy policy.json] [--rotate 1024] [--calibration <dir>]
paddleocr-quantize validate --reference <dir> --quantized <file.gguf> [--corpus <dir>]
paddleocr-quantize inspect  <file.gguf>
```

Three stages, each usable on its own:

1. **Calibrate** (optional, needed for GPTQ). Run the bf16 model over a calibration corpus with an
   activation recorder attached, accumulating `H = Σ xᵀx` per linear — `[k, k]` float64, at most
   4304² = 148 MB for the widest, one tensor at a time. The recorder is a nullable hook on the two
   model classes; one null check per `Gemm.Linear` call, which is nothing against the product.
2. **Quantize**, per tensor, in this order:
   - rotate the weight's input axis by `R` if the policy says so (`W ← W Rᵀ`, so that `W(Rx)` is
     the original product);
   - for each group of 128, choose the scale. RTN takes `d = amax`, which is what the fork does and
     is right for already-ternary input; for real-valued input the scale that minimizes squared
     error over `t = clamp(round(w/d))` is not `amax`, so the default is a search over a small
     grid of `d = α · amax`, `α ∈ [0.4, 1.0]`, keeping the best — the standard trick, and cheap;
   - with a Hessian, run GPTQ: quantize column by column, and after each column push its error
     into the not-yet-quantized columns via the Cholesky factor of `H + λI`. Without one, this
     step is skipped and the result is plain RTN.
3. **Pack and write** into `PQ2_0` or `PTQ1_0` blocks and emit the GGUF.

The policy file is what makes §1.4's lesson operational:

```json
{
  "default": "ptq1_0",
  "rules": [
    { "match": "*norm*",             "scheme": "f32" },
    { "match": "*.bias",             "scheme": "f32" },
    { "match": "*embed_tokens*",     "scheme": "bf16" },
    { "match": "*lm_head*",          "scheme": "pq2_0" },
    { "match": "*patch_embedding*",  "scheme": "bf16" }
  ],
  "rotation": { "block_size": 1024, "sign_mode": "explicit", "seed": 1 }
}
```

---

## 6. Validation

Four levels, cheapest first, each with a gate. A conversion that fails a gate is reported, not
silently shipped.

**L0 — codec.** Round-trip every block layout over random and adversarial inputs, and check the
packed bytes against a Python transcription of `quantize_row_*_ref` in
`tools/reference/dump_ternary_blocks.py`. Gate: byte-identical. This is the level that says our
file is the fork's file.

**L1 — per tensor.** For each quantized tensor, relative Frobenius error `‖W − Ŵ‖/‖W‖`, cosine
similarity per output row (the minimum over rows, not the mean — a single destroyed row is a
destroyed head), and the fraction of groups whose scale saturated. Cheap enough to run on every
conversion, and the report is what a policy is tuned against.

**L2 — per stage.** Reuse the existing parity harness. The vision tower's output for a fixture
image, cosine against the bf16 tower. The decoder's prefill logits for a fixture prompt: top-1
agreement rate and the KL divergence of the last-token distribution. Gate: top-1 agreement above a
threshold the first real conversion sets, since there is no upstream number to inherit.

**L3 — per page.** The corpus in `test_documents/`, one page each, bf16 against quantized:
byte-identical markdown where it happens, and character accuracy where it does not — the metric
`CLAUDE.md` already uses for the repetition-stop change. This is the number that decides whether a
policy ships.

Plus **L4 — cost**: `bench` and `parse --profile` before and after, interleaved in one sitting with
an untouched stage as a control, per this repository's standing rule about measurement.

Tests that need a real checkpoint skip rather than fail when it is absent, as the rest of the suite
does. L0 and the kernel tests run on synthetic tensors and always run.

---

## 7. What is deliberately not here

- **The layout graph.** Quantizing `PP-DocLayoutV3` means a `conv2d` that consumes packed weights
  inside `PirInterpreter`. `conv2d` is 60% of a 2.69 s detection and is im2col-plus-GEMM against a
  filter matrix that is reused across every output row — the same shape as prefill, where
  quantization buys footprint and not time. 125 MB and 8% of a page is not where this starts.
- **Quantized activations.** Everything here quantizes weights only; activations stay float32, as
  they do in Bonsai. An int8 activation path is a different project with a different error budget.
- **Training.** See §1.5. If ternary PTQ turns out not to be shippable at 0.9B, the honest next
  step is a distillation run against the bf16 model's own outputs, not a cleverer rounding rule.

## 8. What the first real conversion measured

Everything above §7 was written before the converter had seen the 0.9B checkpoint. This section is
what happened when it did: `PaddleOCR-VL-1.6` from the model mirror, converted and then compared
against itself on `general_ocr_002.jpg` — a photographed boarding pass with Chinese and English
text. One machine, one sitting, model loading excluded.

### The file

| | |
| --- | --- |
| source | 1.92 GB, 620 tensors, 958.6 M parameters, bfloat16 throughout |
| converted | **0.72 GB (2.68x)** in 51.5 s |
| quantized | 268 tensors, 674.1 M parameters (70.3%) at 1.75 bpw |
| left alone | 284.5 M parameters (29.7%) — 139 M of 4304-wide vision projections the group does not divide, 106 M of token embedding, 37.7 M of a position table |

2.68x rather than the 9x the bit rate suggests, because the exemptions are most of what is left:
the 4304-wide projections alone are 14.5% of the model and cannot be packed at all.

### Round-to-nearest, no rotation, no calibration

| level | measurement |
| --- | --- |
| L1 per tensor | mean relative error **0.4477**, worst 0.5321 on `lm_head` |
| L2 vision tower | cosine **0.330**, relative error 1.06 |
| L3 text | **0.00% character accuracy** |
| L4 cost | recognition 16.7 s reference, **61.8 s quantized (0.27x)** |

The text is the whole answer:

```
reference:  www.997788.com 中国收藏热线 ⏎ 登机牌 BOARDING PASS ⏎ 航班 FLIGHT 日期 DATE 舱位 CLASS …
quantized:  POOOOO ⏎ IOOOOOOOOOOOOOOOOOOOOEEEEOOOEEEOOOOEEEEEEEEEEEEEEEEEEE …
```

### With the rotation

The only rotation block this model admits is **128**, not Bonsai's 1024: the vision tower is 1152
and 4608 wide and neither is a multiple of 1024, while every quantized width is a multiple of 128.
That happens to make the rotation exactly group-wide, which is the case that flattens a group's own
outlier ratio.

| | RTN | RTN + Hadamard(128) |
| --- | --- | --- |
| worst tensor relative error | 0.5321 | **0.4339** |
| lowest row cosine | 0.1758 | **0.8716** |
| rows collapsed to zero | many | **0** |
| vision tower cosine | 0.330 | **0.472** |

A real improvement at every level, and nowhere near enough: a tower output at cosine 0.47 is not a
slightly degraded tower, and the text does not recover.

### The verdict, and it is not close

Ternary post-training quantization of this checkpoint does not work. That is the outcome §1.5
predicted for the stated reason — Bonsai's ternary weights come out of quantization-aware training,
and the file format is a container for weights that are already ternary. Rotation and error
feedback narrow the gap; they do not cross it at 0.9B on a task where one wrong glyph is a visible
error.

What the numbers say about where to go next, in order:

1. **Stop at ternary for the whole model.** The per-tensor errors cluster at 0.44, which is what
   `log₂3` bits buys on weights this dense. A 4-bit band over the same container machinery would
   land near 0.1 and is the obvious next measurement, since every part of this project except the
   choice of block layout is indifferent to the bit width.
2. **Quantize selectively.** The policy mechanism exists for exactly this: the decoder's MLP is
   where the bandwidth is, and the vision tower is where the error hurts most. A file that is
   ternary in the decoder and bfloat16 in the tower is a five-minute experiment.
3. **GPTQ.** Implemented and tested on synthetic tensors, but not yet run against the real model —
   it needs a calibration corpus, which this environment did not have. It is worth measuring and
   it will not rescue a 0.44 relative error on its own.
4. **Distillation**, if ternary at this size is actually the goal. That is a training project, not
   a quantization one.

### Five defects the real conversion found that the synthetic tests could not

Worth listing, because every one of them was invisible on generated tensors and three of them were
producing plausible-looking wrong numbers rather than errors:

- **An `f32` exemption upcasts a bfloat16 checkpoint.** The starting policy named `f32` for norms,
  biases and position embeddings; the checkpoint is bfloat16, so the glob turned 75 MB of
  `packing_position_embedding` into 151 MB — a fifth of the output file spent widening a tensor
  this port never reads. Exemptions are now `source`, which keeps whatever the checkpoint holds.
- **Rank-1 tensors were counted as `n × n`.** The plan reported 1884.3 M parameters against a real
  958.6 M, and "35.8% quantized" where the truth is 70.3%. Invisible in the file, which takes its
  shapes from the tensor.
- **The fp16 scale fallback quantized against a scale the file cannot hold.** When the scale search
  underflowed fp16, the quantizer fell back to the raw `amax`, so the converter's own report
  described numbers it had not written. It now rounds the fallback through fp16 and zeroes a group
  whose scale cannot be represented, which is what the decoder produces anyway.
- **The worst-row-cosine metric was measuring dead rows.** One vision projection has 135 rows whose
  largest weight is about 1e-8 against a tensor maximum of 0.33. Ternary cannot preserve a direction
  there and nothing needs it to, but those rows were setting the headline figure. They are counted
  separately now.
- **The validator compared rotated weights against an unrotated source.** A rotated file holds
  `W Rᵀ`, which has no elementwise relationship to `W`; the comparison reported a relative error of
  1.35 and negative cosines, which looks exactly like a catastrophic conversion rather than like a
  comparison made in the wrong basis. It now folds the source the same way.

The first two were found by reading the converter's own output and disbelieving it. The third and
fifth were found because the converter and the validator disagreed about the same file, which is
the entire reason to measure the same quantity twice in two places.

### And the performance result is the wrong way round

61.8 s against 16.7 s is not a small miss, and the first cause is identified: `ChoosePanelWidth`
sized the quantized column panel by its *packed* stride — 0.22 bytes a weight — where what has to
stay in cache is the *decoded* float32 panel. That made a quantized panel eighteen times wider than
a bfloat16 one and put its scratch far outside L2. Fixed, and not yet re-measured.

That is unlikely to be the whole of it. This workload is one image: the vision tower dominates, and
the tower runs `Gemm.RunPanel`, where a decoded panel is reused across every activation row and the
best quantization can do is break even. The place the bit rate is supposed to pay — a decode step
reading 721 MB of weights per token — barely features here. Measuring it properly needs
`parse --profile` on a decode-heavy page, and it needs a model whose decode terminates.
