using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;
using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models.Layout;
using PaddleOcrSharp.Pdf;
using PaddleOcrSharp.Pipeline;

namespace PaddleOcrSharp.Cli;

/// <summary>Runs the full document parsing pipeline on one or more page images.</summary>
public static class ParseCommand
{
    /// <summary>Runs the <c>parse</c> verb.</summary>
    public static async Task<int> RunAsync(CommandLine command)
    {
        if (command.Positional.Count == 0)
        {
            Console.Error.WriteLine("parse needs at least one image path.");
            return 2;
        }

        foreach (string path in command.Positional)
        {
            if (!File.Exists(path))
            {
                Console.Error.WriteLine($"'{path}' does not exist.");
                return 2;
            }
        }

        using var cancellation = new CancellationTokenSource();
        Console.CancelKeyPress += (_, eventArgs) =>
        {
            eventArgs.Cancel = true;
            cancellation.Cancel();
        };

        bool useLayout = command.GetBool("layout", true);

        string modelDirectory = await ModelLocator
            .ResolveVLAsync(command, allowDownload: true, cancellation.Token)
            .ConfigureAwait(false);

        string? layoutDirectory = useLayout
            ? await ModelLocator.ResolveLayoutAsync(command, allowDownload: true, cancellation.Token)
                .ConfigureAwait(false)
            : null;

        bool useOrientation = command.GetBool("doc-orientation", false);
        bool useUnwarping = command.GetBool("doc-unwarping", false);

        string? orientationDirectory = useOrientation
            ? await ModelLocator.ResolveOptionalAsync(
                command,
                PaddleOcrSharp.Download.ModelCatalog.DocOrientationClassifier,
                "orientation-dir",
                "PP_LCNET_DOC_ORI_DIR",
                allowDownload: true,
                cancellation.Token).ConfigureAwait(false)
            : null;

        string? unwarpingDirectory = useUnwarping
            ? await ModelLocator.ResolveOptionalAsync(
                command,
                PaddleOcrSharp.Download.ModelCatalog.DocUnwarping,
                "unwarping-dir",
                "UVDOC_DIR",
                allowDownload: true,
                cancellation.Token).ConfigureAwait(false)
            : null;

        var options = DocumentParserOptions.Default with
        {
            UseLayoutDetection = useLayout,
            UseDocOrientationClassify = useOrientation && orientationDirectory is not null,
            UseDocUnwarping = useUnwarping && unwarpingDirectory is not null,
            UseChartRecognition = command.GetBool("chart", false),
            UseSealRecognition = command.GetBool("seal", false),
            UseOcrForImageBlocks = command.GetBool("ocr-images", false),
            MergeLayoutBlocks = command.GetBool("merge-blocks", true),
            Layout = LayoutOptions.Default with
            {
                Threshold = command.GetFloat("layout-threshold", LayoutOptions.Default.Threshold),
                Nms = command.GetBool("layout-nms", LayoutOptions.Default.Nms),
            },
            BlockConcurrency = command.GetInt("block-concurrency", 1),
            PixelBudgets = new BlockPixelBudgets
            {
                MinPixels = command.GetInt("min-pixels", BlockPrompt.DefaultMinPixels),
                MaxPixels = command.GetInt("max-pixels", BlockPrompt.DefaultMaxPixels),
                OcrMaxPixels = OptionalPixels(command, "ocr-max-pixels"),
                TableMaxPixels = OptionalPixels(command, "table-max-pixels"),
                ChartMaxPixels = OptionalPixels(command, "chart-max-pixels"),
                FormulaMaxPixels = OptionalPixels(command, "formula-max-pixels"),
                SealMaxPixels = OptionalPixels(command, "seal-max-pixels"),
            },
        };

        var clock = Stopwatch.StartNew();
        using DocumentParser parser = DocumentParser.Load(
            modelDirectory, layoutDirectory, orientationDirectory, unwarpingDirectory);
        Console.Error.WriteLine($"Models loaded in {clock.Elapsed.TotalSeconds:F1}s");

        string? outputDirectory = command.Get("output-dir");
        if (outputDirectory is not null)
        {
            Directory.CreateDirectory(outputDirectory);
        }

        var pages = new List<ParsedPage>();
        int dpi = command.GetInt("dpi", PdfRasterizer.DefaultDpi);
        int maxPages = command.GetInt("max-pages", 0);
        int pageIndex = 0;

        foreach (string path in command.Positional)
        {
            foreach ((RgbImage image, string label) in LoadPages(path, dpi, maxPages, command.Get("password")))
            {
                using (image)
                {
                    clock.Restart();
                    var progress = new ConsoleBlockProgress();

                    ParsedPage page = parser.Parse(image, options, pageIndex, progress, cancellation.Token);
                    Console.Error.WriteLine(
                        $"\r{label}: {page.Blocks.Count} blocks in {clock.Elapsed.TotalSeconds:F1}s");
                    pages.Add(page);

                    if (outputDirectory is not null)
                    {
                        await WritePageAsync(page, label, outputDirectory, options, cancellation.Token)
                            .ConfigureAwait(false);
                    }
                }

                pageIndex++;
            }
        }

        if (outputDirectory is null)
        {
            string format = command.Get("format", "markdown")!;
            var document = new ParsedDocument(pages);
            Console.WriteLine(format.Equals("json", StringComparison.OrdinalIgnoreCase)
                ? ToJson(pages)
                : document.ToMarkdown(options.MarkdownSettings, command.Get("page-separator", "\n\n")!));
        }

        return 0;
    }

    /// <summary>
    /// Yields every page of an input: one for an image file, one per rendered page for a PDF.
    /// </summary>
    private static IEnumerable<(RgbImage Image, string Label)> LoadPages(
        string path,
        int dpi,
        int maxPages,
        string? password)
    {
        string stem = Path.GetFileNameWithoutExtension(path);

        if (!Path.GetExtension(path).Equals(".pdf", StringComparison.OrdinalIgnoreCase))
        {
            yield return (ImageIO.Load(path), stem);
            yield break;
        }

        int index = 0;
        foreach (RgbImage page in PdfRasterizer.Render(path, dpi, password, maxPages))
        {
            yield return (page, $"{stem}_page{index + 1:D3}");
            index++;
        }
    }

    private static async Task WritePageAsync(
        ParsedPage page,
        string stem,
        string outputDirectory,
        DocumentParserOptions options,
        CancellationToken cancellationToken)
    {

        await File.WriteAllTextAsync(
            Path.Combine(outputDirectory, $"{stem}.md"),
            page.ToMarkdown(options.MarkdownSettings),
            cancellationToken).ConfigureAwait(false);

        await File.WriteAllTextAsync(
            Path.Combine(outputDirectory, $"{stem}.json"),
            ToJson([page]),
            cancellationToken).ConfigureAwait(false);

        foreach (ParsedBlock block in page.Blocks)
        {
            if (block.Image is null || block.ImagePath is null)
            {
                continue;
            }

            string imageDirectory = Path.Combine(outputDirectory, options.Markdown.ImageDirectory);
            Directory.CreateDirectory(imageDirectory);
            await File.WriteAllBytesAsync(
                Path.Combine(imageDirectory, block.ImagePath), block.Image, cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>Reads a per-label pixel ceiling, leaving it unset when the flag is absent.</summary>
    private static int? OptionalPixels(CommandLine command, string name) =>
        command.Has(name) ? command.GetInt(name, BlockPrompt.DefaultMaxPixels) : null;

    private static string ToJson(IReadOnlyList<ParsedPage> pages) => JsonSerializer.Serialize(
        pages.Select(page => new JsonPage(
            page.Index,
            page.Width,
            page.Height,
            [.. page.Blocks.Select(block => new JsonBlock(
                block.Label,
                block.ReadingOrder,
                block.Order,
                block.Box.Score,
                [block.Box.Left, block.Box.Top, block.Box.Right, block.Box.Bottom],
                block.Content,
                block.ImagePath))]))
            .ToArray(),
        ResultJson.Default.JsonPageArray);
}

/// <summary>Writes block progress to standard error, on the calling thread.</summary>
/// <remarks>
/// Not <see cref="Progress{T}"/>: that posts each callback to the thread pool, so a report can
/// land after the line that is meant to replace it and leave a half-written progress line in the
/// output. Reporting happens once per block and costs nothing worth deferring.
/// </remarks>
internal sealed class ConsoleBlockProgress : IProgress<BlockProgress>
{
    private readonly Lock _gate = new();

    /// <inheritdoc />
    public void Report(BlockProgress value)
    {
        lock (_gate)
        {
            Console.Error.Write($"\r  block {value.BlockIndex + 1}/{value.BlockCount} ({value.Label})      ");
        }
    }
}

/// <summary>One page of the JSON result.</summary>
/// <remarks>
/// Declared rather than anonymous, and serialised through a generated context, so the CLI stays
/// publishable with native AOT — reflection-based serialisation is the one thing in the tree the
/// trimmer cannot see through.
/// </remarks>
internal sealed record JsonPage(
    [property: JsonPropertyName("page_index")] int PageIndex,
    [property: JsonPropertyName("width")] int Width,
    [property: JsonPropertyName("height")] int Height,
    [property: JsonPropertyName("blocks")] JsonBlock[] Blocks);

/// <summary>One recognised block of a page.</summary>
internal sealed record JsonBlock(
    [property: JsonPropertyName("label")] string Label,
    [property: JsonPropertyName("reading_order")] int ReadingOrder,
    [property: JsonPropertyName("block_order")] int? Order,
    [property: JsonPropertyName("score")] float Score,
    [property: JsonPropertyName("bbox")] float[] BoundingBox,
    [property: JsonPropertyName("content")] string Content,
    [property: JsonPropertyName("image")] string? Image);

/// <summary>Serialisation context for the CLI's JSON output.</summary>
[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(JsonPage[]))]
internal sealed partial class ResultJson : JsonSerializerContext;
