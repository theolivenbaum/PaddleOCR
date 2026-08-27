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
RT-DETR–style detector: HGNetV2-L backbone → hybrid encoder (`d_model 256`, 3 levels,
strides 8/16/32) → 6-layer deformable decoder, 300 queries, 25 classes, plus mask and
reading-order heads. Input `800×800`, no normalization (`mean 0`, `std 1`), `NCHW`.

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
      Models/Layout/           # PP-DocLayoutV3
      Pipeline/                # orchestration, block prompts, markdown/OTSL assembly
      Download/                # model downloader (Hugging Face / BOS mirrors)
    PaddleOcrSharp.Cli/        # `paddleocr-sharp` command-line front-end
  tests/
    PaddleOcrSharp.Tests/      # unit + numerical-parity tests
  tools/
    reference/                 # Python scripts that dump upstream reference tensors
```

## Engineering conventions

- **.NET 10 / C# preview.** Use `System.Numerics.Tensors`, `Vector<T>` / `Vector512<T>`,
  `TensorPrimitives`, `ArrayPool<T>` and `MemoryPool<T>`. Hot loops must be allocation-free.
- **Weights stay in their on-disk dtype where possible** (bf16) and are converted lazily per
  tile inside the GEMM, so a 0.9B model does not need a 4 GB float32 shadow copy.
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

## Working agreements for this port

1. Read the upstream Python for a stage **before** writing the C# for it. Quote the file and
   line range in the C# doc-comment so the mapping is auditable.
2. Land work in vertical slices that build and test green. Track progress in
   [`to-do.md`](to-do.md).
3. Upstream CI, pre-commit hooks and agent skill files in this repository are **disabled**
   (renamed to `*.disabled`) for the duration of the port; do not re-enable them.
4. Do not modify the upstream Python packages (`paddleocr/`, `ppocr/`, `ppstructure/`, …)
   — the port is additive and lives entirely under `dotnet/`.
