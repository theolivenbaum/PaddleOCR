using PaddleOcrSharp.Models.Layout;

namespace PaddleOcrSharp.Pipeline;

/// <summary>One recognised region of a page.</summary>
/// <param name="Label">Layout label of the region.</param>
/// <param name="Box">The region's bounding box.</param>
/// <param name="Content">Recognised content: text, LaTeX, or an HTML table.</param>
/// <param name="ReadingOrder">Position in the page's reading order.</param>
public sealed record ParsedBlock(string Label, LayoutBox Box, string Content, int ReadingOrder)
{
    /// <summary>
    /// The block's position in the page's reading flow, counting from one, or
    /// <see langword="null"/> when the label is one that does not take part.
    /// </summary>
    /// <remarks>
    /// <c>update_order_index</c>. Distinct from <see cref="ReadingOrder"/>, which is the raw
    /// value the detector's ordering head predicted and is only meaningful as a sort key.
    /// </remarks>
    public int? Order { get; init; }

    /// <summary>
    /// Identifies the blocks that were stacked into one image before recognition, or
    /// <see langword="null"/> for a block that stands alone.
    /// </summary>
    /// <remarks>
    /// <c>group_id</c>: the index of the group's first block, carried by every member, so a
    /// consumer can tell that a paragraph split across two columns is one paragraph.
    /// </remarks>
    public int? GroupId { get; init; }

    /// <summary>Encoded image bytes when the block is a figure kept as an image.</summary>
    public byte[]? Image { get; init; }

    /// <summary>Suggested file name for <see cref="Image"/>.</summary>
    public string? ImagePath { get; init; }

    /// <summary>
    /// Text runs with their quadrilaterals, populated only for blocks recognised in spotting mode.
    /// </summary>
    public IReadOnlyList<SpottedText> SpottedText { get; init; } = [];
}

/// <summary>The parse of a single page.</summary>
/// <param name="Index">Zero-based page index within the document.</param>
/// <param name="Width">Page width in pixels.</param>
/// <param name="Height">Page height in pixels.</param>
/// <param name="Blocks">Recognised regions, in reading order.</param>
public sealed record ParsedPage(int Index, int Width, int Height, IReadOnlyList<ParsedBlock> Blocks)
{
    /// <summary>Renders the page as markdown.</summary>
    /// <param name="options">Formatting options; defaults to the shipped pipeline's.</param>
    public string ToMarkdown(MarkdownOptions? options = null) =>
        MarkdownWriter.Write(Blocks, options ?? MarkdownOptions.Default, Width);
}

/// <summary>A parsed document: one or more pages.</summary>
/// <param name="Pages">The pages, in order.</param>
public sealed record ParsedDocument(IReadOnlyList<ParsedPage> Pages)
{
    /// <summary>
    /// Renders the whole document as markdown.
    /// </summary>
    /// <remarks>
    /// Pages are joined by a blank line, as <c>concatenate_markdown_pages</c> does. Pass a
    /// <paramref name="separator"/> to put a rule or heading between them instead.
    /// </remarks>
    public string ToMarkdown(MarkdownOptions? options = null, string separator = "\n\n") =>
        string.Join(separator, Pages.Select(page => page.ToMarkdown(options)));
}
