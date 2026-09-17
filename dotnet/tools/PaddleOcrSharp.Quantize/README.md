# `paddleocr-quantize`

Converts a published PaddleOCR-VL checkpoint into a ternary GGUF the port can run, and says what
the conversion cost. The design — where the encoding comes from, what it can and cannot buy, and
what each level of validation is for — is in [`dotnet/docs/ternary.md`](../../docs/ternary.md).

It is a tool, not a package. Models are regenerated rarely and the shipped `paddleocr-sharp` CLI
should stay lean, so this project is run from the repository:

```bash
dotnet run --project dotnet/tools/PaddleOcrSharp.Quantize -- <verb> …
```

## Converting

```bash
# Round-to-nearest with the scale search, no rotation, the starting policy.
dotnet run --project dotnet/tools/PaddleOcrSharp.Quantize -- convert \
    --in  ~/.cache/paddleocr-sharp/PaddleOCR-VL-1.6 \
    --out ~/models/PaddleOCR-VL-1.6-ptq1_0/model.gguf

# With the rotation, which is the single most useful thing that transfers from Bonsai.
… convert --rotate 1024 --sign explicit

# With error feedback against real activations. Each pass re-runs the calibration images, so a
# tighter budget means more passes rather than less memory per Hessian.
… convert --rotate 1024 --calibration ./calibration-pages --calibration-budget-gb 4
```

The output directory gets the checkpoint's `config.json`, `tokenizer.json` and preprocessor files
copied beside `model.gguf`, so it loads as a model directory with no further steps:
`PaddleOcrVLModel.Load` prefers `model.gguf` over `model.safetensors` when both are present.

## Policy

Precision is per tensor. `policy` writes the starting point, which is then edited and passed back:

```bash
dotnet run --project dotnet/tools/PaddleOcrSharp.Quantize -- policy --out policy.json
… convert --policy policy.json
```

```json
{
  "defaultScheme": "ptq1_0",
  "rules": [
    { "match": "*norm*",        "scheme": "f32" },
    { "match": "*.bias",        "scheme": "f32" },
    { "match": "*embed_tokens*","scheme": "bf16" },
    { "match": "*lm_head*",     "scheme": "pq2_0" }
  ],
  "rotation": { "blockSize": 1024, "signMode": "explicit", "seed": 1 }
}
```

A tensor whose input width is not a multiple of 128 cannot be packed at all, and the converter says
so rather than failing: the vision MLP's second projection is 4304 wide, which is 134 M parameters
over 27 layers that stay bfloat16.

## Validating and inspecting

```bash
… validate --reference ~/.cache/paddleocr-sharp/PaddleOCR-VL-1.6 \
           --quantized ~/models/PaddleOCR-VL-1.6-ptq1_0 \
           page.png

… inspect ~/models/PaddleOCR-VL-1.6-ptq1_0/model.gguf
```

`validate` runs the three levels the design calls L1 to L3 — per-tensor error, the vision tower's
output, and the recognised text with its character accuracy — because none of them predicts the
next one.
