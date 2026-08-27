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
}
