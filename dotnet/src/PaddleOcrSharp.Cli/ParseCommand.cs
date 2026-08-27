using System.Diagnostics;
using System.Text.Json;
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

        bool useLayout = command.GetBool("layout", !command.Has("no-layout"));

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
                    var progress = new Progress<BlockProgress>(value =>
                        Console.Error.Write(
                            $"\r  block {value.BlockIndex + 1}/{value.BlockCount} ({value.Label})      "));

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
            Console.WriteLine(format.Equals("json", StringComparison.OrdinalIgnoreCase)
                ? ToJson(pages)
                : string.Join("\n\n---\n\n", pages.Select(page => page.ToMarkdown(options.Markdown))));
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
            page.ToMarkdown(options.Markdown),
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

    private static string ToJson(IReadOnlyList<ParsedPage> pages) => JsonSerializer.Serialize(
        pages.Select(page => new
        {
            page_index = page.Index,
            width = page.Width,
            height = page.Height,
            blocks = page.Blocks.Select(block => new
            {
                label = block.Label,
                reading_order = block.ReadingOrder,
                score = block.Box.Score,
                bbox = new[] { block.Box.Left, block.Box.Top, block.Box.Right, block.Box.Bottom },
                content = block.Content,
                image = block.ImagePath,
            }),
        }),
        new JsonSerializerOptions { WriteIndented = true });
}
