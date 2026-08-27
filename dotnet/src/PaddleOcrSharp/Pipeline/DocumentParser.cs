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
        IReadOnlyList<LayoutBox> detected = settings.UseLayoutDetection && _layout is not null
            ? _layout.Detect(page, settings.Layout)
            : [WholePage(page, settings.WholePageLabel)];

        // The page's pictures are gathered before the overlap filter runs, which is what lets a
        // figure a table swallowed still be identified after the filter has dropped it.
        IReadOnlyList<DocumentFigure> figures = GatherFigures(detected, page.Width, page.Height);

        IReadOnlyList<LayoutBox> regions = settings.UseLayoutDetection && _layout is not null
            ? OverlapFilter.Apply(detected)
            : detected;

        var crops = new RgbImage[regions.Count];
        var sizes = new (int Width, int Height)[regions.Count];

        try
        {
            for (int i = 0; i < regions.Count; i++)
            {
                LayoutBox region = regions[i].ClampTo(page.Width, page.Height);

                // Truncated, not rounded outward: `restructured_boxes` clamps each edge into the
                // page and then takes `int(...)` of all four, so the right and bottom edges fall
                // back to the pixel they are inside rather than the next one out.
                int left = (int)region.Left;
                int top = (int)region.Top;
                crops[i] = page.Crop(left, top, (int)region.Right, (int)region.Bottom);

                // A region with an outline keeps only what the outline covers; the rest of the
                // crop goes white so the model does not read a neighbour's text out of the
                // corners of a slanted or L-shaped block.
                if (region.Polygon is { Length: > 2 } outline)
                {
                    MaskToPolygon(crops[i], outline, left, top);
                }

                sizes[i] = (crops[i].Width, crops[i].Height);
            }

            IReadOnlyCollection<string> nonMergeLabels = NonMergeLabels(settings);
            List<BlockGroup> groups = settings.MergeLayoutBlocks && regions.Count > 1
                ? BlockMerger.Group(regions, sizes, nonMergeLabels)
                : [.. Enumerable.Range(0, regions.Count).Select(i => new BlockGroup([i], []))];

            var blocks = new ParsedBlock[regions.Count];
            var tokenizedByBlock = new IReadOnlyList<TokenizedFigure>?[regions.Count];
            var absorbed = new HashSet<string>(StringComparer.Ordinal);
            int completed = 0;

            void RecognizeGroup(int groupIndex)
            {
                cancellationToken.ThrowIfCancellationRequested();

                BlockGroup group = groups[groupIndex];
                int primary = group.Indices[0];
                LayoutBox region = regions[primary].ClampTo(page.Width, page.Height);

                using RgbImage merged = group.Indices.Count == 1
                    ? crops[primary].Clone()
                    : ImageStacker.Stack(
                        [.. group.Indices.Select(index => crops[index])], group.Alignments);

                IReadOnlyList<TokenizedFigure> tokenized = [];
                RgbImage prepared = merged;

                if (region.Label == "table" && settings.TokenizeTableFigures)
                {
                    (prepared, tokenized, IReadOnlyList<string> swallowed) =
                        TableFigureTokenizer.Tokenize(merged, region, figures);

                    lock (absorbed)
                    {
                        absorbed.UnionWith(swallowed);
                    }
                }

                try
                {
                    blocks[primary] = Recognize(prepared, region, settings, cancellationToken);
                    if (group.Indices.Count > 1)
                    {
                        blocks[primary] = blocks[primary] with { GroupId = primary };
                    }
                }
                finally
                {
                    if (!ReferenceEquals(prepared, merged))
                    {
                        prepared.Dispose();
                    }
                }

                if (tokenized.Count > 0)
                {
                    // Substituting the placeholders has to wait until every block is recognised,
                    // because a figure's own text goes in beside its image.
                    tokenizedByBlock[primary] = tokenized;
                }

                // Every region of a merged group keeps its box so the JSON still describes the
                // page, but only the first carries the recognised text.
                for (int i = 1; i < group.Indices.Count; i++)
                {
                    int index = group.Indices[i];
                    LayoutBox other = regions[index].ClampTo(page.Width, page.Height);
                    blocks[index] = new ParsedBlock(other.Label, other, string.Empty, other.ReadingOrder)
                    {
                        GroupId = primary,
                    };
                }

                progress?.Report(new BlockProgress(
                    Interlocked.Increment(ref completed) - 1, groups.Count, region.Label));
            }

            if (settings.BlockConcurrency > 1)
            {
                Parallel.For(
                    0,
                    groups.Count,
                    new ParallelOptions
                    {
                        MaxDegreeOfParallelism = settings.BlockConcurrency,
                        CancellationToken = cancellationToken,
                    },
                    RecognizeGroup);
            }
            else
            {
                for (int groupIndex = 0; groupIndex < groups.Count; groupIndex++)
                {
                    RecognizeGroup(groupIndex);
                }
            }

            for (int i = 0; i < blocks.Length; i++)
            {
                if (tokenizedByBlock[i] is { Count: > 0 } tokenized)
                {
                    blocks[i] = blocks[i] with
                    {
                        Content = TableFigureTokenizer.Untokenize(
                            blocks[i].Content,
                            tokenized,
                            settings.Markdown.ImageDirectory,
                            path => blocks.FirstOrDefault(
                                block => block?.ImagePath == path)?.Content),
                    };
                }
            }

            // Figures that a table absorbed are described inside its HTML, so they no longer
            // belong to the page as separate blocks.
            IReadOnlyList<ParsedBlock> retained = absorbed.Count == 0
                ? blocks
                : [.. blocks.Where(block => block.ImagePath is null || !absorbed.Contains(block.ImagePath))];

            return new ParsedPage(pageIndex, page.Width, page.Height, BlockOrder.Assign(retained, settings.Markdown.IgnoredLabels));
        }
        finally
        {
            foreach (RgbImage? crop in crops)
            {
                crop?.Dispose();
            }
        }
    }

    /// <summary>
    /// Labels that are never merged with a neighbour: figures, tables, and whichever of charts and
    /// seals are being kept as images rather than recognised.
    /// </summary>
    private static string[] NonMergeLabels(DocumentParserOptions options) =>
        [.. SkippedByModel(options), "table"];

    /// <summary>
    /// Labels whose blocks are not sent to the model at all, because they are kept as pictures.
    /// </summary>
    /// <remarks><c>image_labels</c>.</remarks>
    private static List<string> SkippedByModel(DocumentParserOptions options)
    {
        var labels = options.UseOcrForImageBlocks ? [] : new List<string>(BlockLabels.ImageLabels);

        if (!options.UseChartRecognition)
        {
            labels.Add("chart");
        }

        if (!options.UseSealRecognition)
        {
            labels.Add("seal");
        }

        return labels;
    }

    /// <summary>
    /// Labels whose blocks get their crop attached as a picture.
    /// </summary>
    /// <remarks>
    /// <c>vis_image_labels</c>, and deliberately not the same set as
    /// <see cref="SkippedByModel"/>: the two decisions are independent upstream. A seal is always
    /// kept as a picture even when it is also recognised, and asking for OCR on figures gives a
    /// figure both a picture and recognised text rather than swapping one for the other.
    /// </remarks>
    private static bool KeptAsPicture(string label, DocumentParserOptions options) =>
        BlockLabels.ImageLabels.Contains(label)
        || label == "seal"
        || (label == "chart" && !options.UseChartRecognition);

    private ParsedBlock Recognize(
        RgbImage crop,
        LayoutBox region,
        DocumentParserOptions options,
        CancellationToken cancellationToken)
    {
        byte[]? picture = null;
        string? picturePath = null;

        if (KeptAsPicture(region.Label, options))
        {
            picture = ImageIO.EncodeJpeg(crop);
            picturePath = FigurePath(region);
        }

        if (SkippedByModel(options).Contains(region.Label))
        {
            return new ParsedBlock(region.Label, region, string.Empty, region.ReadingOrder)
            {
                Image = picture,
                ImagePath = picturePath,
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

        VisionPreprocessorOptions preprocessing = BlockPrompt.Options(region.Label, options.PixelBudgets);
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
            Image = picture,
            ImagePath = picturePath,
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

    /// <summary>Whitens every pixel of <paramref name="crop"/> outside <paramref name="outline"/>.</summary>
    private static void MaskToPolygon(RgbImage crop, (float X, float Y)[] outline, int left, int top)
    {
        var shifted = new (int X, int Y)[outline.Length];
        for (int i = 0; i < outline.Length; i++)
        {
            shifted[i] = ((int)outline[i].X - left, (int)outline[i].Y - top);
        }

        bool[] inside = Polygons.Fill(shifted, crop.Width, crop.Height);

        for (int y = 0; y < crop.Height; y++)
        {
            Span<byte> row = crop.Row(y);
            int rowBase = y * crop.Width;
            for (int x = 0; x < crop.Width; x++)
            {
                if (inside[rowBase + x])
                {
                    continue;
                }

                row[x * 3] = 255;
                row[(x * 3) + 1] = 255;
                row[(x * 3) + 2] = 255;
            }
        }
    }

    /// <summary>
    /// The pictures a page contains, as <c>gather_imgs</c> collects them.
    /// </summary>
    /// <remarks>
    /// A narrower set of labels than the blocks that keep a picture: running-head and running-foot
    /// images are not gathered, and a seal is. This is the list a table's figure tokenisation
    /// works from, so the difference decides which pictures a table can swallow.
    /// </remarks>
    private static List<DocumentFigure> GatherFigures(
        IReadOnlyList<LayoutBox> detected,
        int pageWidth,
        int pageHeight)
    {
        var figures = new List<DocumentFigure>();

        foreach (LayoutBox box in detected)
        {
            if (box.Label is not ("image" or "figure" or "seal"))
            {
                continue;
            }

            LayoutBox clamped = box.ClampTo(pageWidth, pageHeight);
            if (clamped.Right <= clamped.Left || clamped.Bottom <= clamped.Top)
            {
                continue;
            }

            figures.Add(new DocumentFigure(FigurePath(box), box));
        }

        return figures;
    }

    /// <summary>
    /// File name a figure block is written to, as <c>construct_img_path</c>.
    /// </summary>
    /// <remarks>
    /// The box makes the name unique within a page, which is what lets a table's
    /// <c>[Fn]</c> placeholder be resolved back to the figure it covered. JPEG, because that is
    /// the extension upstream writes and the format its saver then infers from it.
    /// </remarks>
    private static string FigurePath(LayoutBox region) =>
        $"img_in_{region.Label}_box_{(int)region.Left}_{(int)region.Top}_{(int)region.Right}_{(int)region.Bottom}.jpg";

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
