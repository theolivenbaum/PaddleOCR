namespace PaddleOcrSharp.Pipeline;

/// <summary>
/// The layout labels PP-DocLayoutV3 emits, in class-index order.
/// </summary>
/// <remarks>
/// The list is the <c>label_list</c> of the model's <c>inference.yml</c>. Several labels collapse
/// onto the same semantic class in the pipeline's own mapping (for example <c>footer</c> and
/// <c>footer_image</c>), which is why the raw names are kept verbatim here.
/// </remarks>
public static class BlockLabels
{
    /// <summary>Class names indexed by the detector's class id.</summary>
    public static readonly string[] All =
    [
        "abstract",
        "algorithm",
        "aside_text",
        "chart",
        "content",
        "display_formula",
        "doc_title",
        "figure_title",
        "footer",
        "footer_image",
        "footnote",
        "formula_number",
        "header",
        "header_image",
        "image",
        "inline_formula",
        "number",
        "paragraph_title",
        "reference",
        "reference_content",
        "seal",
        "table",
        "text",
        "vertical_text",
        "vision_footnote",
    ];

    /// <summary>Labels whose content is an image rather than text.</summary>
    public static readonly string[] ImageLabels = ["image", "header_image", "footer_image"];

    /// <summary>
    /// Labels excluded from markdown by default, from the shipped pipeline configuration.
    /// </summary>
    public static readonly string[] MarkdownIgnored =
    [
        "number", "footnote", "header", "header_image", "footer", "footer_image", "aside_text",
    ];

    /// <summary>
    /// Labels excluded from the page's block ordering.
    /// </summary>
    /// <remarks>
    /// <c>SKIP_ORDER_LABELS</c>. These are the regions that do not belong to the page's reading
    /// flow — running heads, captions, figures, tables, marginalia — so numbering them alongside
    /// the prose would make the order say something it does not mean. They keep their position in
    /// the block list; they just have no order number.
    /// </remarks>
    public static readonly string[] SkipOrder =
    [
        "figure_title", "vision_footnote", "image", "chart", "table", "header", "header_image",
        "footer", "footer_image", "footnote", "aside_text",
    ];

    /// <summary>Name of class <paramref name="id"/>, or <c>"unknown"</c> when out of range.</summary>
    public static string Name(int id) => (uint)id < (uint)All.Length ? All[id] : "unknown";
}
