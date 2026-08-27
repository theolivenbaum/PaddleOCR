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
  Directory.Build.props        # net10.0, unsafe, nullable, preview lang
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
      Download/                # model downloader (Hugging Face / BOS mirrors)
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
  `TensorPrimitives`, `ArrayPool<T>` and `MemoryPool<T>`. Hot loops must be allocation-free.
- **Weights stay in their on-disk dtype** (bf16) so a 0.9B model does not need a 4 GB float32
  shadow copy. The GEMM widens one column panel at a time and reuses it across every activation
  row; widening inside the inner loop instead costs more than the multiply-adds it feeds.
- **Every numerical stage is testable in isolation.** Each module exposes a deterministic entry
  point that the parity tests feed with `.npz` fixtures dumped from the Python reference.
- **No `float` accumulation shortcuts** where upstream forces `float32` (softmax, RoPE,
  RMSNorm variance) — match upstream precision decisions exactly.

## Validating against upstream

`dotnet/tools/reference/` holds Python scripts that load the real Hugging Face model, run a
stage, and dump inputs/outputs as `.npz`. `PaddleOcrSharp.Tests` loads the same `.npz` and
asserts the C# output matches within a stage-appropriate tolerance. Fixtures are generated on
demand (they are not committed) — see `dotnet/tools/reference/README.md`.

Tests that need fixtures are skipped, not failed, when the fixture directory is absent, so
`dotnet test` works on a clean clone.

## Where the time goes

Measured with `paddleocr-sharp bench` on 4 cores (AVX-512), a 980x392 page (1960 patches).
`--no-vl` and `--no-layout` time the halves separately, which is what the layout figure needs —
run together, the two compete for the same cache.

| Stage | Cost |
| --- | --- |
| Vision tower (1960 patches) | ~16 s |
| Decoder (503-token prefill + 32 tokens) | ~3.6 s |
| Layout graph | ~5.2 s |

Both halves are GEMM-bound, and the shape of the win is the same in each: give the inner loop
enough reuse that it is compute-bound rather than load-bound. `Gemm.Linear` widens a bf16 column
panel once and reuses it across every activation row; `Gemm.MatMul` tiles the output and picks
its kernel from the operand layout; attention blocks 16 query rows at a time so the keys and
values stay in cache across the block; the convolution treats an output row as one GEMM against
its im2col columns rather than a dot product per output pixel.

`PirProfile` (printed by `bench`) reports each operator's total, its slowest single call, and
that call's result shape and Paddle module path. The shape column is what makes the layout
graph's cost legible — nearly a fifth of it is element-wise work on the mask head's
`[1, 300, 200, 200]` tensors, not the convolutions.

Not every plausible idea survives measurement: banding `Gemm.Linear` over activation rows, so a
band stays cached across all the column panels, is a clear win on paper and was consistently
slower in practice. The panel loop's traffic is evidently already absorbed by the shared cache.

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
