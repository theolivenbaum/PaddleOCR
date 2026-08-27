# PaddleOcrSharp — working on the port

The overview, the packages, the CLI and the library usage are in the
[repository README](../README.md). This file is about building and shipping what is in here.

## Projects

| Project | Contents | Ships as |
| --- | --- | --- |
| `src/PaddleOcrSharp` | Tensors and SIMD kernels, safetensors and Paddle weight readers, the SigLIP vision tower, the ERNIE-4.5 decoder, the BPE tokenizer, the Paddle graph interpreter, the pipeline | `PaddleOCR` |
| `src/PaddleOcrSharp.Pdf` | PDF page rasterisation (the only native dependency: PDFium) | `PaddleOCR.Pdf` |
| `src/PaddleOcrSharp.Cli` | `paddleocr-sharp`: `download`, `parse`, `recognize`, `bench` | `PaddleOCR.Cli`, a .NET tool |
| `tests/PaddleOcrSharp.Tests` | Unit tests plus numerical parity tests against the Python reference | — |
| `tools/reference` | Python scripts that dump reference tensors — see [its README](tools/reference/README.md) | — |

## Build and test

```bash
dotnet build PaddleOcrSharp.slnx -c Release
dotnet test tests/PaddleOcrSharp.Tests -c Release
dotnet run --project src/PaddleOcrSharp.Cli -c Release -- parse page.pdf --output-dir out/
```

Parity tests skip themselves when the checkpoints and the generated `.npz` fixtures are absent,
so a clean clone goes green; `tools/reference/README.md` explains how to produce them.

## Packing

The package ids drop the `Sharp` the assemblies and namespaces keep — `PaddleOCR`,
`PaddleOCR.Pdf`, `PaddleOCR.Cli` — the same split as `HNSW` and `HNSW.Net` in the sibling
repository. Package metadata lives in [`Directory.Build.props`](Directory.Build.props); the repository README
is attached to every packable project by [`Directory.Build.targets`](Directory.Build.targets),
which runs after the project file has decided whether it is packable.

```bash
dotnet pack src/PaddleOcrSharp     -c Release /p:Version=26.8.1
dotnet pack src/PaddleOcrSharp.Pdf -c Release /p:Version=26.8.1
dotnet pack src/PaddleOcrSharp.Cli -c Release /p:Version=26.8.1   # the tool
```

The tool package is the project's *publish* output, so the native assets SkiaSharp and PDFium
bring along travel with it, and `dotnet pack --no-build` does not work for it — that project is
packed from source. Between them those two ship natives for two dozen runtime identifiers, and a
RID-agnostic publish takes all of them: 549 MiB, of which 255 MiB is Windows debug symbols for
libSkiaSharp. `TrimToolRuntimeAssets` in the CLI project drops the native symbols and every RID
outside `ToolRuntimeIdentifiers` — win-x64/arm64, linux-x64/arm64, the musl pair, and the three
macOS ones — which takes the package from 197 MB to 71 MB. Add a RID there to ship it. A
RID-specific publish (`-r linux-x64`, the native-AOT path) resolves one RID and is left alone. Try it end to end without touching the machine's global tools:

```bash
dotnet tool install PaddleOCR.Cli --version 26.8.1 \
  --add-source src/PaddleOcrSharp.Cli/bin/Release --tool-path /tmp/tool
/tmp/tool/paddleocr-sharp
```

[`../.devops/build-nuget.yml`](../.devops/build-nuget.yml) is the Azure Pipelines definition that
does all of this on `main`, stamping a CalVer version and pushing the three packages.

## Native AOT

```bash
dotnet publish src/PaddleOcrSharp.Cli -c Release -r linux-x64 -p:PublishAot=true
```

Both libraries set `IsAotCompatible`, so the trim and AOT analysers run on every build.

## Conventions

`../CLAUDE.md` is the working agreement for the port: read the upstream Python for a stage before
writing the C# for it and quote the file and line range in the doc comment, land vertical slices
that build and test green, and say so in the doc comment wherever a stage cannot be reproduced
exactly. Progress is tracked in [`../to-do.md`](../to-do.md).
