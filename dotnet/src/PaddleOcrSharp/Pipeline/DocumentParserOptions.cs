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

    /// <summary>Layout detection settings.</summary>
    public LayoutOptions Layout { get; init; } = LayoutOptions.Default;

    /// <summary>Decoding settings for every block.</summary>
    public GenerationOptions Generation { get; init; } = GenerationOptions.Default;

    /// <summary>Markdown rendering settings.</summary>
    public MarkdownOptions Markdown { get; init; } = MarkdownOptions.Default;

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

    /// <summary>Number of blocks recognised concurrently.</summary>
    /// <remarks>
    /// The model's own kernels already use every core, so blocks are recognised one at a time by
    /// default; raising this only helps when many blocks are small enough to leave cores idle.
    /// </remarks>
    public int BlockConcurrency { get; init; } = 1;
}
