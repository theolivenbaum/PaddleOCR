using PaddleOcrSharp.Models;
using PaddleOcrSharp.Models.Layout;

namespace PaddleOcrSharp.Pipeline;

/// <summary>Settings for a document parse.</summary>
public sealed record DocumentParserOptions
{
    /// <summary>Defaults matching <c>pipeline_config_vllm.yaml</c> for PaddleOCR-VL-1.6.</summary>
    public static DocumentParserOptions Default { get; } = new();

    /// <summary>Whether layout detection runs; when off the whole page goes to the VL model.</summary>
    public bool UseLayoutDetection { get; init; } = true;

    /// <summary>
    /// Whether the page is rotated upright by the orientation classifier before parsing.
    /// </summary>
    /// <remarks>Off by default, matching the shipped 1.6 pipeline's <c>use_doc_preprocessor</c>.</remarks>
    public bool UseDocOrientationClassify { get; init; }

    /// <summary>Whether the page is flattened by UVDoc before parsing.</summary>
    public bool UseDocUnwarping { get; init; }

    /// <summary>Whether chart blocks are recognised rather than kept as images.</summary>
    public bool UseChartRecognition { get; init; }

    /// <summary>Whether seal blocks are recognised rather than kept as images.</summary>
    public bool UseSealRecognition { get; init; }

    /// <summary>Whether image blocks are sent through OCR rather than kept as images.</summary>
    public bool UseOcrForImageBlocks { get; init; }

    /// <summary>Whether adjacent blocks of the same label are merged before recognition.</summary>
    public bool MergeLayoutBlocks { get; init; } = true;

    /// <summary>
    /// Whether figures inside a table are replaced by placeholder tokens before recognition and
    /// turned back into image references afterwards.
    /// </summary>
    public bool TokenizeTableFigures { get; init; } = true;

    /// <summary>Layout detection settings.</summary>
    public LayoutOptions Layout { get; init; } = LayoutOptions.Default;

    /// <summary>Decoding settings for every block.</summary>
    public GenerationOptions Generation { get; init; } = GenerationOptions.Default;

    /// <summary>Per-label pixel budgets for the VL model's input resize.</summary>
    public BlockPixelBudgets PixelBudgets { get; init; } = BlockPixelBudgets.Default;

    /// <summary>Markdown rendering settings.</summary>
    public MarkdownOptions Markdown { get; init; } = MarkdownOptions.Default;

    /// <summary>
    /// <see cref="Markdown"/> with the settings the renderer shares with the pipeline filled in.
    /// </summary>
    /// <remarks>
    /// Whether charts became tables, whether a seal or figure has recognised text to show, and
    /// whether layout detection ran are decisions made here, and upstream's renderer reads them
    /// straight out of the pipeline's own settings rather than being told again.
    /// </remarks>
    public MarkdownOptions MarkdownSettings => Markdown with
    {
        ShowImageText = UseOcrForImageBlocks,
        ShowSealText = UseSealRecognition,
        ChartsAsTables = UseChartRecognition,
        UseLayoutDetection = UseLayoutDetection,
    };

    /// <summary>
    /// The instruction used when layout detection is off, i.e. the whole page is one block.
    /// </summary>
    public string WholePagePrompt { get; init; } = BlockPrompt.Ocr;

    /// <summary>
    /// The label given to the whole-page block when layout detection is off.
    /// </summary>
    /// <remarks>
    /// Upstream selects this mode with <c>prompt_label</c>; the label matters because it decides
    /// the instruction, the pixel budget and the post-processing. Use <c>"spotting"</c> to get
    /// text-with-coordinates output over the whole page.
    /// </remarks>
    public string WholePageLabel { get; init; } = "text";

    /// <summary>Optional collector for what each block's recognition cost.</summary>
    /// <remarks>
    /// Off by default. When set, every block adds a <see cref="RecognitionRecord"/> saying how many
    /// tokens it generated and where its time went, which is the only way to see that one runaway
    /// block outweighs the rest of the page.
    /// </remarks>
    public RecognitionProfile? Profile { get; init; }

    /// <summary>Optional collector for what each pipeline stage outside the model call cost.</summary>
    /// <remarks>
    /// Off by default. Complements <see cref="Profile"/>: that one accounts for the tower and the
    /// token loop, this one for layout detection, cropping, masking, crop preparation and the
    /// markup conversions, which together are a fifth of a scanned page's run.
    /// </remarks>
    public PageProfile? StageProfile { get; init; }

    /// <summary>Optional collector for the layout graph's per-operator cost.</summary>
    /// <remarks>
    /// Off by default. Layout detection is a single stage from the pipeline's point of view, and
    /// on a scanned page it is a large one; this is what says which of the graph's operators it
    /// went to.
    /// </remarks>
    public Models.Paddle.PirProfile? LayoutProfile { get; init; }

    /// <summary>
    /// How the kernels spread work across threads, or <see langword="null"/> for
    /// <see cref="Core.Parallelism.Default"/> — one worker per core.
    /// </summary>
    /// <remarks>
    /// <para>
    /// One worker per core is the right default and not always the right answer: a host sharing
    /// the machine with its own work, or running under a CPU quota that
    /// <see cref="Environment.ProcessorCount"/> cannot see, wants fewer. The
    /// <see cref="ParallelOptions.CancellationToken"/> on it is honoured by every kernel region
    /// too, alongside the token passed to <see cref="DocumentParser.Parse"/>.
    /// </para>
    /// <para>
    /// The instance is read, never copied, so do not mutate it while a parse is running. This
    /// applies for the duration of <see cref="DocumentParser.Parse"/>; for the model and detector
    /// entry points used directly, wrap the call in <see cref="Core.Parallelism.Use"/>.
    /// </para>
    /// <para>
    /// Note that this is not <see cref="BlockConcurrency"/>. That decides how many blocks are
    /// recognised at once; this decides how many threads the kernels inside one of them use, and
    /// the two multiply.
    /// </para>
    /// </remarks>
    public ParallelOptions? Parallelism { get; init; }

    /// <summary>Number of blocks whose vision towers run concurrently.</summary>
    /// <remarks>
    /// The model's own kernels already use every core, so blocks are encoded one at a time by
    /// default; raising this only helps when many blocks are small enough to leave cores idle,
    /// and it costs holding several blocks' activations at once. When the blocks are decoded as a
    /// batch this applies to the vision tower alone — the token loop is already shared — and the
    /// kernels inside a block are narrowed to match, so the two settings do not multiply into more
    /// threads than cores.
    /// </remarks>
    public int BlockConcurrency { get; init; } = 1;

    /// <summary>Number of blocks whose token loops run as one batch.</summary>
    /// <remarks>
    /// <para>
    /// A decode step reads all 255M decoder parameters and the 106M-parameter output head to
    /// produce a single token — 646 MB of bfloat16 — so it is bound by how fast the weights arrive
    /// and not by the arithmetic. Blocks on a page are independent, so stepping several together
    /// reads those weights once for the whole batch. A page of forty small blocks is where that
    /// matters: its decode costs as many weight sweeps as the longest block in each batch needs,
    /// rather than as many as every block put together needs.
    /// </para>
    /// <para>
    /// The vision tower and the prefill still run per block, because both already reach the matrix
    /// products with hundreds of rows and have nothing to amortise. The cost of a batch is holding
    /// its key/value caches at once — tens of megabytes per block — which is why it is sized.
    /// </para>
    /// <para>
    /// Zero, the default, batches as many blocks as <see cref="DecodeBatchBytes"/> allows. A
    /// positive value caps the count as well. Set to 1 to decode a block at a time, which is what
    /// the port did before and what <see cref="BlockConcurrency"/> applies to.
    /// </para>
    /// </remarks>
    public int DecodeBatch { get; init; }

    /// <summary>
    /// Key/value cache a decode batch may hold at once, in bytes.
    /// </summary>
    /// <remarks>
    /// What bounds a batch is memory, not a count: every member holds its own cache for as long as
    /// the batch runs, and a block's cache is proportional to its prompt — tens of megabytes for a
    /// full-budget block and a few for a one-line heading. Sizing the batch by the cache it will
    /// need lets a page of forty short blocks decode in one pass while a page of large ones still
    /// splits, without either being told a block count that only suits the other.
    /// </remarks>
    public long DecodeBatchBytes { get; init; } = 768L * 1024 * 1024;
}
