# PaddleOcrSharp

A pure C# implementation of the **PaddleOCR-VL-1.6** document parsing pipeline.

No ONNX Runtime, no Paddle Inference, no libtorch. The tensor math, the model graphs, the
tokenizer, the image pipeline and the weight loading are all in this repository.

```bash
dotnet run --project src/PaddleOcrSharp.Cli -c Release -- download
dotnet run --project src/PaddleOcrSharp.Cli -c Release -- parse page.pdf --output-dir out/
```

## What it does

`parse` runs the shipped pipeline end to end:

1. **Layout detection** — PP-DocLayoutV3 finds the regions of a page and orders them.
2. **Block preparation** — overlapping detections are filtered, paragraphs split across columns
   or around figures are stacked back into one image, and figures are extracted as images.
3. **Recognition** — each region goes to the 0.9B vision-language model with the instruction its
   label calls for (`OCR:`, `Table Recognition:`, `Formula Recognition:`, …).
4. **Assembly** — OTSL tables become HTML, LaTeX delimiters are normalised, runaway repetition is
   trimmed, and the page is rendered as markdown and JSON.

## Projects

| Project | Contents |
| --- | --- |
| `src/PaddleOcrSharp` | Everything: tensors and SIMD kernels, safetensors and Paddle weight readers, the SigLIP vision tower, the ERNIE-4.5 decoder, the BPE tokenizer, the Paddle graph interpreter, the pipeline |
| `src/PaddleOcrSharp.Pdf` | PDF page rasterisation (the only native dependency: PDFium) |
| `src/PaddleOcrSharp.Cli` | `paddleocr-sharp`: `download`, `parse`, `recognize`, `bench` |
| `tests/PaddleOcrSharp.Tests` | Unit tests plus numerical parity tests against the Python reference |
| `tools/reference` | Python scripts that dump reference tensors — see [its README](tools/reference/README.md) |

## Using it as a library

```csharp
using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Pipeline;

using var parser = DocumentParser.Load(visionLanguageDirectory, layoutDirectory);
using RgbImage page = ImageIO.Load("page.png");

ParsedPage parsed = parser.Parse(page);
Console.WriteLine(parsed.ToMarkdown());

foreach (ParsedBlock block in parsed.Blocks)
{
    Console.WriteLine($"{block.Label} {block.Box.Left},{block.Box.Top} -> {block.Content}");
}
```

To recognise a single already-cropped region, skip the pipeline:

```csharp
using var model = PaddleOcrVLModel.Load(visionLanguageDirectory);
string text = model.Recognize(crop, BlockPrompt.Table);
```

Models are fetched on demand into `~/.cache/paddleocr-sharp`, or point the loaders at a directory
you manage yourself. `HF_ENDPOINT` redirects the downloader at a mirror.

## Correctness

Every stage is checked against the upstream Python implementation rather than against
expectations:

| Stage | How it is checked |
| --- | --- |
| Image pre-processing | byte-identical resize and `pixel_values` matching to 1e-6, over six image shapes |
| Vision tower | all 27 encoder layers plus the projector, against per-layer dumps |
| Tokenizer | encode and decode over 27 mixed-script cases, including byte fallback and the `<\|LOC_n\|>` tokens |
| Decoder | prompt ids, 3-D rope index, prefill logits and 24 greedy steps, token for token |
| Layout graph | every fetched tensor against Paddle's own inference run |
| Contours and polygons | the OpenCV and Shapely calls behind the mask head, against cv2 and Shapely themselves |
| Region polygons | `extract_polygon_points_by_masks` for all four `layout_shape_mode` values |
| Orientation classifier, UVDoc | both graphs against Paddle's own run, and both wrappers end to end |
| Markdown | the rendered page against PaddleX's own `MarkdownConverter`, over every label and six settings combinations |
| Whole pipeline | block labels, boxes and recognised text for a rendered page |

Run them with `dotnet test`. Tests that need the checkpoints or the generated fixtures skip
themselves when those are absent, so a clean clone still goes green.

## Performance

`paddleocr-sharp bench` reports per-stage timings, allocation, and a per-operator breakdown of
the layout graph — including each operator's slowest single call, with its result shape and the
Paddle module it came from. `--no-vl` and `--no-layout` time the halves separately.

On four cores with AVX-512, a 980×392 page takes roughly 16 s in the vision tower, 3.6 s to
decode, and 5.2 s in the layout graph. CPU only for now.

Every tensor in the hot loops is pooled: a whole vision pass allocates about 1.2 MiB, and a
decode step about 310 KiB — all of it framework bookkeeping for the parallel loops, none of it
tensors.

## Native AOT

The CLI publishes as a self-contained native binary, no runtime installed:

```bash
dotnet publish src/PaddleOcrSharp.Cli -c Release -r linux-x64 -p:PublishAot=true
```

Both libraries build with `IsAotCompatible`, so the trim and AOT analysers run on every build
and a reflection-based API cannot creep in unnoticed.

## Licence

Apache-2.0, matching the upstream project and the model weights.
