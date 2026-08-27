using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models;
using PaddleOcrSharp.Models.Layout;
using PaddleOcrSharp.Models.Preprocessing;

namespace PaddleOcrSharp.Pipeline;

/// <summary>
/// The PaddleOCR-VL-1.6 document parsing pipeline: detect layout, crop each region, recognise it
/// with the VL model, then assemble the page.
/// </summary>
/// <remarks>
/// Port of <c>_PaddleOCRVLPipeline</c> in PaddleX. The block dispatch (which instruction and which
/// pixel budget each label gets), the table figure handling, the repetition guard and the
/// OTSL→HTML conversion follow <c>pipeline.py</c> and <c>uilts.py</c>.
/// </remarks>
public sealed class DocumentParser : IDisposable
{
    private readonly PaddleOcrVLModel _model;
    private readonly LayoutDetector? _layout;
    private readonly bool _ownsModel;

    private readonly DocOrientationClassifier? _orientation;
    private readonly DocumentUnwarper? _unwarper;

    /// <summary>Creates a parser over an already-loaded model and detector.</summary>
    /// <param name="model">The vision-language model.</param>
    /// <param name="layout">The layout detector, or <see langword="null"/> for whole-page mode.</param>
    /// <param name="ownsModel">Whether disposing the parser disposes the model.</param>
    /// <param name="orientation">Optional page-orientation classifier.</param>
    /// <param name="unwarper">Optional page-flattening model.</param>
    public DocumentParser(
        PaddleOcrVLModel model,
        LayoutDetector? layout,
        bool ownsModel = false,
        DocOrientationClassifier? orientation = null,
        DocumentUnwarper? unwarper = null)
    {
        _model = model;
        _layout = layout;
        _ownsModel = ownsModel;
        _orientation = orientation;
        _unwarper = unwarper;
    }

    /// <summary>Loads the models from their directories.</summary>
    /// <param name="visionLanguageDirectory">Directory holding the PaddleOCR-VL checkpoint.</param>
    /// <param name="layoutDirectory">Directory holding PP-DocLayoutV3, or <see langword="null"/>.</param>
    /// <param name="orientationDirectory">Directory holding PP-LCNet_x1_0_doc_ori, or <see langword="null"/>.</param>
    /// <param name="unwarpingDirectory">Directory holding UVDoc, or <see langword="null"/>.</param>
    public static DocumentParser Load(
        string visionLanguageDirectory,
        string? layoutDirectory,
        string? orientationDirectory = null,
        string? unwarpingDirectory = null)
    {
        PaddleOcrVLModel model = PaddleOcrVLModel.Load(visionLanguageDirectory);
        LayoutDetector? layout = layoutDirectory is null ? null : LayoutDetector.Load(layoutDirectory);
        DocOrientationClassifier? orientation = orientationDirectory is null
            ? null
            : DocOrientationClassifier.Load(orientationDirectory);
        DocumentUnwarper? unwarper = unwarpingDirectory is null
            ? null
            : DocumentUnwarper.Load(unwarpingDirectory);

        return new DocumentParser(model, layout, ownsModel: true, orientation, unwarper);
    }

    /// <summary>Parses one page.</summary>
    /// <param name="page">The page image.</param>
    /// <param name="options">Pipeline settings.</param>
    /// <param name="pageIndex">Index recorded on the result.</param>
    /// <param name="progress">Optional per-block progress callback.</param>
    /// <param name="cancellationToken">Cancels between blocks.</param>
    public ParsedPage Parse(
        RgbImage page,
        DocumentParserOptions? options = null,
        int pageIndex = 0,
        IProgress<BlockProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        DocumentParserOptions settings = options ?? DocumentParserOptions.Default;

        RgbImage? prepared = null;
        try
        {
            if (settings.UseDocOrientationClassify && _orientation is not null)
            {
                prepared = _orientation.Correct(page);
                page = prepared;
            }

            if (settings.UseDocUnwarping && _unwarper is not null)
            {
                RgbImage flattened = _unwarper.Unwarp(page);
                prepared?.Dispose();
                prepared = flattened;
                page = flattened;
            }

            return ParseCore(page, settings, pageIndex, progress, cancellationToken);
        }
        finally
        {
            prepared?.Dispose();
        }
    }

    private ParsedPage ParseCore(
        RgbImage page,
        DocumentParserOptions settings,
        int pageIndex,
        IProgress<BlockProgress>? progress,
        CancellationToken cancellationToken)
    {
        IReadOnlyList<LayoutBox> regions = settings.UseLayoutDetection && _layout is not null
            ? _layout.Detect(page, settings.Layout)
            : [WholePage(page, settings.WholePageLabel)];

        if (settings.MergeLayoutBlocks && regions.Count > 1)
        {
            regions = MergeAdjacent(regions);
        }

        var blocks = new ParsedBlock[regions.Count];

        void RecognizeAt(int index)
        {
            cancellationToken.ThrowIfCancellationRequested();
            LayoutBox region = regions[index].ClampTo(page.Width, page.Height);
            blocks[index] = Recognize(page, region, index, settings, cancellationToken);
            progress?.Report(new BlockProgress(index, regions.Count, region.Label));
        }

        if (settings.BlockConcurrency > 1)
        {
            Parallel.For(
                0,
                regions.Count,
                new ParallelOptions
                {
                    MaxDegreeOfParallelism = settings.BlockConcurrency,
                    CancellationToken = cancellationToken,
                },
                RecognizeAt);
        }
        else
        {
            for (int index = 0; index < regions.Count; index++)
            {
                RecognizeAt(index);
            }
        }

        return new ParsedPage(pageIndex, page.Width, page.Height, blocks);
    }

    private ParsedBlock Recognize(
        RgbImage page,
        LayoutBox region,
        int index,
        DocumentParserOptions options,
        CancellationToken cancellationToken)
    {
        bool keepAsImage = IsImageBlock(region.Label, options);
        using RgbImage crop = page.Crop(
            (int)MathF.Floor(region.Left),
            (int)MathF.Floor(region.Top),
            (int)MathF.Ceiling(region.Right),
            (int)MathF.Ceiling(region.Bottom));

        if (keepAsImage)
        {
            return new ParsedBlock(region.Label, region, string.Empty, region.ReadingOrder)
            {
                Image = ImageIO.EncodePng(crop),
                ImagePath = $"{region.Label}_{index}_{(int)region.Left}_{(int)region.Top}.png",
            };
        }

        string instruction = region.Label == "spotting"
            ? BlockPrompt.Spotting
            : BlockPrompt.For(region.Label, options.UseChartRecognition, options.UseSealRecognition);

        using RgbImage prepared = Prepare(crop, region.Label, instruction);

        // Spotting encodes coordinates as `<|LOC_n|>` tokens, which are special tokens: dropping
        // them during decoding would erase the geometry the mode exists to produce.
        GenerationOptions generation = region.Label == "spotting"
            ? options.Generation with { SkipSpecialTokens = false }
            : options.Generation;

        VisionPreprocessorOptions preprocessing = BlockPrompt.Options(region.Label);
        string raw = _model.Recognize(
            prepared, instruction, preprocessing, generation, cancellationToken);

        string content = RepetitionTruncator.Truncate(
            raw,
            region.Label == "table"
                ? RepetitionTruncator.TableMinimumLength
                : RepetitionTruncator.BlockMinimumLength);

        content = NormalizeMathDelimiters(content, region.Label);

        if (region.Label == "table")
        {
            string html = OtslTable.ToHtml(content);
            if (html.Length > 0)
            {
                content = html;
            }
        }

        IReadOnlyList<SpottedText> spotted = [];
        if (region.Label == "spotting")
        {
            (content, spotted) = Spotting.Parse(content, crop.Width, crop.Height);
        }

        return new ParsedBlock(region.Label, region, content, region.ReadingOrder)
        {
            SpottedText = spotted,
        };
    }

    /// <summary>
    /// Applies the per-instruction crop preparation: formulas get their margins trimmed, and a
    /// small spotting crop is doubled with Lanczos so the coordinate grid has room to resolve.
    /// </summary>
    private static RgbImage Prepare(RgbImage crop, string label, string instruction)
    {
        if (instruction == BlockPrompt.Formula)
        {
            return CropMargin(crop);
        }

        if (label == "spotting"
            && crop.Width < Spotting.UpscaleBelow
            && crop.Height < Spotting.UpscaleBelow)
        {
            return PilResize.ResizeLanczos(crop, crop.Width * 2, crop.Height * 2);
        }

        return crop.Clone();
    }

    private static bool IsImageBlock(string label, DocumentParserOptions options)
    {
        if (options.UseOcrForImageBlocks)
        {
            return false;
        }

        if (BlockLabels.ImageLabels.Contains(label))
        {
            return true;
        }

        return (label == "chart" && !options.UseChartRecognition)
            || (label == "seal" && !options.UseSealRecognition);
    }

    /// <summary>
    /// Rewrites LaTeX delimiters the model emits into the <c>$</c> forms markdown renders.
    /// </summary>
    /// <remarks>
    /// Port of the delimiter fix-up in <c>_paddleocr_vl_assemble_parsing_results</c>: when the
    /// output uses <c>\(…\)</c> or <c>\[…\]</c>, existing <c>$</c> signs are currency and are
    /// stripped before the delimiters are converted.
    /// </remarks>
    private static string NormalizeMathDelimiters(string content, string label)
    {
        bool hasInline = content.Contains("\\(", StringComparison.Ordinal)
            && content.Contains("\\)", StringComparison.Ordinal);
        bool hasDisplay = content.Contains("\\[", StringComparison.Ordinal)
            && content.Contains("\\]", StringComparison.Ordinal);

        if (!hasInline && !hasDisplay)
        {
            return content;
        }

        string result = content
            .Replace("$", string.Empty, StringComparison.Ordinal)
            .Replace("\\(", " $ ", StringComparison.Ordinal)
            .Replace("\\)", " $", StringComparison.Ordinal)
            .Replace("\\[\\[", "\\[", StringComparison.Ordinal)
            .Replace("\\]\\]", "\\]", StringComparison.Ordinal)
            .Replace("\\[", " $$ ", StringComparison.Ordinal)
            .Replace("\\]", " $$ ", StringComparison.Ordinal);

        return label == "formula_number"
            ? result.Replace("$", string.Empty, StringComparison.Ordinal)
            : result;
    }

    /// <summary>
    /// Trims uniform margins from a formula crop so the model sees the glyphs at a useful scale.
    /// </summary>
    /// <remarks>Port of <c>crop_margin</c>; the crop is skipped when it would leave nothing.</remarks>
    private static RgbImage CropMargin(RgbImage image)
    {
        int left = image.Width;
        int right = -1;
        int top = image.Height;
        int bottom = -1;

        for (int y = 0; y < image.Height; y++)
        {
            ReadOnlySpan<byte> row = image.Row(y);
            for (int x = 0; x < image.Width; x++)
            {
                int offset = x * 3;
                int luminance = (row[offset] + row[offset + 1] + row[offset + 2]) / 3;
                if (luminance >= 200)
                {
                    continue;
                }

                left = Math.Min(left, x);
                right = Math.Max(right, x);
                top = Math.Min(top, y);
                bottom = Math.Max(bottom, y);
            }
        }

        if (right < 0 || right - left <= 2 || bottom - top <= 2)
        {
            return image.Clone();
        }

        return image.Crop(left, top, right + 1, bottom + 1);
    }

    /// <summary>
    /// Merges vertically adjacent regions that share a label, so a paragraph split across two
    /// detections is recognised as one block.
    /// </summary>
    private static IReadOnlyList<LayoutBox> MergeAdjacent(IReadOnlyList<LayoutBox> regions)
    {
        string[] neverMerge = [.. BlockLabels.ImageLabels, "table", "chart", "seal"];

        var merged = new List<LayoutBox>(regions.Count);
        foreach (LayoutBox region in regions)
        {
            if (merged.Count == 0 || neverMerge.Contains(region.Label))
            {
                merged.Add(region);
                continue;
            }

            LayoutBox previous = merged[^1];
            bool sameLabel = previous.Label == region.Label && !neverMerge.Contains(previous.Label);
            bool horizontallyAligned =
                Math.Abs(previous.Left - region.Left) < 0.1f * Math.Max(previous.Width, region.Width) &&
                Math.Abs(previous.Right - region.Right) < 0.1f * Math.Max(previous.Width, region.Width);
            bool verticallyAdjacent = region.Top - previous.Bottom is >= -2f and < 20f;

            if (sameLabel && horizontallyAligned && verticallyAdjacent)
            {
                merged[^1] = previous with
                {
                    Left = Math.Min(previous.Left, region.Left),
                    Top = Math.Min(previous.Top, region.Top),
                    Right = Math.Max(previous.Right, region.Right),
                    Bottom = Math.Max(previous.Bottom, region.Bottom),
                    Score = Math.Max(previous.Score, region.Score),
                };
            }
            else
            {
                merged.Add(region);
            }
        }

        return merged;
    }

    private static LayoutBox WholePage(RgbImage page, string label) =>
        new(-1, label, 1f, 0f, 0f, page.Width, page.Height, 0);

    /// <inheritdoc />
    public void Dispose()
    {
        if (_ownsModel)
        {
            _model.Dispose();
        }

        _layout?.Dispose();
        _orientation?.Dispose();
        _unwarper?.Dispose();
    }
}

/// <summary>Progress of a page parse.</summary>
/// <param name="BlockIndex">Index of the block just recognised.</param>
/// <param name="BlockCount">Total number of blocks on the page.</param>
/// <param name="Label">Layout label of the block.</param>
public readonly record struct BlockProgress(int BlockIndex, int BlockCount, string Label);
