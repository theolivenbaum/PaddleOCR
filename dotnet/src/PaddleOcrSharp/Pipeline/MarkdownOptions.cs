namespace PaddleOcrSharp.Pipeline;

/// <summary>Markdown rendering settings.</summary>
public sealed record MarkdownOptions
{
    /// <summary>Defaults matching the shipped pipeline configuration.</summary>
    public static MarkdownOptions Default { get; } = new();

    /// <summary>Labels whose blocks are omitted from the markdown.</summary>
    public IReadOnlyCollection<string> IgnoredLabels { get; init; } = BlockLabels.MarkdownIgnored;

    /// <summary>Whether a formula's number is appended to the formula block.</summary>
    public bool ShowFormulaNumber { get; init; }

    /// <summary>Directory prefix written into image links.</summary>
    public string ImageDirectory { get; init; } = "imgs";

    /// <summary>
    /// Whether to render the HTML-decorated markdown the pipeline emits by default.
    /// </summary>
    /// <remarks>
    /// <c>pretty</c>. Captions and figures are wrapped in a centred <c>div</c>, images become
    /// width-scaled <c>img</c> tags, and tables carry border and alignment styling. Turning it off
    /// gives the plain variant: markdown image links and unstyled table HTML.
    /// </remarks>
    public bool Pretty { get; init; } = true;

    /// <summary>Whether a figure block's recognised text is written beneath the image.</summary>
    public bool ShowImageText { get; init; }

    /// <summary>Whether a seal block's recognised text is written beneath the image.</summary>
    public bool ShowSealText { get; init; }

    /// <summary>Whether chart blocks are rendered as HTML tables rather than as images.</summary>
    public bool ChartsAsTables { get; init; }

    /// <summary>
    /// Whether layout detection ran. With no layout, a seal block has no image to show and is
    /// rendered as text, which is what upstream falls back to.
    /// </summary>
    public bool UseLayoutDetection { get; init; } = true;
}
