using System.Text;
using System.Text.RegularExpressions;

namespace PaddleOcrSharp.Pipeline;

/// <summary>
/// Renders recognised blocks as markdown.
/// </summary>
/// <remarks>
/// <para>
/// Port of PaddleX's <c>MarkdownConverter.convert</c> together with the label-to-formatter map in
/// <c>markdown_format_funcs.build_handle_funcs_dict</c> and the wiring
/// <c>PaddleOCRVLResult._build_handle_funcs_dict</c> puts around it. Blocks are joined by a blank
/// line and each label gets its own transformation.
/// </para>
/// <para>
/// The default is the pipeline's own "pretty" rendering, which leans on HTML for the things
/// markdown cannot express: captions and figures centred in a <c>div</c>, images scaled to their
/// share of the page width, tables given borders and centred cells.
/// <see cref="MarkdownOptions.Pretty"/> turns that off for the plain variant.
/// </para>
/// </remarks>
public static partial class MarkdownWriter
{
    [GeneratedRegex(
        @"^\s*((?:[1-9][0-9]*(?:\.[1-9][0-9]*)*[\.、]?|[\(\（](?:[1-9][0-9]*|[一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+)[\)\）]|[一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+[、\.]?|(?:I|II|III|IV|V|VI|VII|VIII|IX|X)(?:\.|\s)))(\s*)(.*)$")]
    private static partial Regex TitleNumbering();

    /// <summary>
    /// The labels the renderer knows how to format.
    /// </summary>
    /// <remarks>
    /// A block whose label is absent is dropped entirely, separator and all. That is how
    /// <c>formula_number</c> disappears — it has no handler upstream, so an equation number is
    /// only ever seen through <see cref="MarkdownOptions.ShowFormulaNumber"/>, merged into the
    /// formula it belongs to.
    /// </remarks>
    private static readonly HashSet<string> Handled = new(StringComparer.Ordinal)
    {
        "paragraph_title", "abstract_title", "reference_title", "content_title", "doc_title",
        "table_title", "figure_title", "chart_title", "vision_footnote", "text", "ocr",
        "vertical_text", "reference_content", "abstract", "content", "image", "chart", "formula",
        "display_formula", "inline_formula", "table", "reference", "algorithm", "seal",
        "spotting", "number", "footnote", "header", "header_image", "footer", "footer_image",
        "aside_text",
    };

    /// <summary>Renders <paramref name="blocks"/> as markdown.</summary>
    /// <param name="blocks">The page's blocks, in reading order.</param>
    /// <param name="options">Formatting options.</param>
    /// <param name="pageWidth">
    /// Page width in pixels, which sets each figure's width as a percentage of it. Pass 0 to
    /// leave images unscaled.
    /// </param>
    public static string Write(IReadOnlyList<ParsedBlock> blocks, MarkdownOptions options, int pageWidth = 0)
    {
        var builder = new StringBuilder();

        for (int i = 0; i < blocks.Count; i++)
        {
            ParsedBlock block = blocks[i];

            if (!Handled.Contains(block.Label) || options.IgnoredLabels.Contains(block.Label))
            {
                continue;
            }

            string content = block.Content;

            if (options.ShowFormulaNumber
                && block.Label is "display_formula" or "formula"
                && i + 1 < blocks.Count
                && blocks[i + 1].Label == "formula_number")
            {
                content = MergeFormulaAndNumber(content, blocks[i + 1].Content);
            }

            // A handler that produces nothing still separates its neighbours, which is what
            // upstream's unconditional append does.
            if (builder.Length > 0)
            {
                builder.Append("\n\n");
            }

            builder.Append(Format(block, content, options, pageWidth));
        }

        return builder.ToString();
    }

    private static string Format(ParsedBlock block, string content, MarkdownOptions options, int pageWidth) =>
        block.Label switch
        {
            "doc_title" => CollapseSoftNewlines($"# {content}"),
            "paragraph_title" => block.TitleLevel is int level
                ? CollapseSoftNewlines($"{new string('#', level + 1)} {content}")
                : FormatTitle(content),
            "abstract_title" or "reference_title" or "content_title" => FormatPlainTitle(content),
            "table_title" or "figure_title" or "chart_title" => FormatText(content, options),
            "table" => FormatTable(content, options),
            "display_formula" or "formula" or "inline_formula" => content,
            "image" or "header_image" or "footer_image" =>
                FormatImage(block, content, options, pageWidth, options.ShowImageText),
            "chart" => options.ChartsAsTables
                ? ChartToHtmlTable(content)
                : FormatImage(block, content, options, pageWidth, options.ShowImageText),
            "seal" => options.UseLayoutDetection
                ? FormatImage(block, content, options, pageWidth, options.ShowSealText)
                : FormatText(content, options),
            "content" => content.Replace("-\n", "  \n", StringComparison.Ordinal)
                .Replace("\n", "  \n", StringComparison.Ordinal),
            "algorithm" => content.Trim('\n'),
            "abstract" => FormatFirstLine(content, ["摘要", "abstract"], line => $"## {line}\n", ' '),
            "reference" => FormatFirstLine(content, ["参考文献", "references"], line => $"## {line}", '\n'),
            "number" or "footnote" or "header" or "footer" or "aside_text" or "spotting" => content,
            _ => NormalizeNewlines(content),
        };

    /// <summary>Text that upstream centres when rendering prettily and passes through otherwise.</summary>
    private static string FormatText(string content, MarkdownOptions options) =>
        options.Pretty ? Centered(CollapseSoftNewlines(content)) : content;

    /// <summary>Wraps content in the centring <c>div</c> upstream uses.</summary>
    private static string Centered(string content) =>
        $"<div style=\"text-align: center;\">{content}</div>\n";

    /// <summary>
    /// Turns a numbered heading into a markdown heading whose level follows the numbering depth.
    /// </summary>
    private static string FormatTitle(string content)
    {
        string title = content;
        Match match = TitleNumbering().Match(title);
        if (match.Success)
        {
            title = match.Groups[1].Value.Trim() + " " + match.Groups[3].Value.TrimStart();
        }

        title = title.TrimEnd('.');
        int level = title.Contains('.', StringComparison.Ordinal)
            ? title.Count(c => c == '.') + 1
            : 1;

        return CollapseSoftNewlines($"{new string('#', level + 1)} {title}");
    }

    /// <summary>A section heading that carries no numbering.</summary>
    private static string FormatPlainTitle(string content) => CollapseSoftNewlines($"## {content}");

    /// <summary>Renders a figure, either as a scaled HTML tag or as a markdown image link.</summary>
    private static string FormatImage(
        ParsedBlock block,
        string content,
        MarkdownOptions options,
        int pageWidth,
        bool showText)
    {
        if (block.ImagePath is null)
        {
            return string.Empty;
        }

        string path = CollapseSoftNewlines($"{options.ImageDirectory}/{block.ImagePath}");

        if (!options.Pretty)
        {
            string plain = $"![]({path})";
            return showText ? $"{plain}\n\n{content}\n\n" : plain;
        }

        int scale = pageWidth > 0 ? (int)(block.Box.Width / pageWidth * 100) : 100;
        string tag = $"<img src=\"{path}\" alt=\"Image\" width=\"{scale}%\" />";

        if (!showText)
        {
            return Centered(CollapseSoftNewlines(tag));
        }

        // The wrapper keeps the newlines when text follows, because the text is a second line.
        return Centered($"{tag}\n\n{content}\n\n");
    }

    /// <summary>Adds the border and alignment styling the pipeline's tables carry.</summary>
    private static string FormatTable(string content, MarkdownOptions options)
    {
        if (!options.Pretty)
        {
            return SimplifyTable("\n" + content);
        }

        return "\n" + content
            .Replace("<table>", "<table border=1 style='margin: auto; word-wrap: break-word;'>", StringComparison.Ordinal)
            .Replace("<th>", "<th style='text-align: center; word-wrap: break-word;'>", StringComparison.Ordinal)
            .Replace("<td>", "<td style='text-align: center; word-wrap: break-word;'>", StringComparison.Ordinal);
    }

    /// <summary>
    /// Strips the document wrapper tags a table may arrive inside, and prepends a newline.
    /// </summary>
    /// <remarks>
    /// The newline is upstream's, and its caller adds one of its own, so a plain-rendered table
    /// starts two lines down rather than one.
    /// </remarks>
    private static string SimplifyTable(string content) => "\n" + content
        .Replace("<html>", string.Empty, StringComparison.Ordinal)
        .Replace("</html>", string.Empty, StringComparison.Ordinal)
        .Replace("<body>", string.Empty, StringComparison.Ordinal)
        .Replace("</body>", string.Empty, StringComparison.Ordinal);

    /// <summary>Renders a recognised chart's pipe-separated rows as an HTML table.</summary>
    private static string ChartToHtmlTable(string content)
    {
        string[] lines = content.Split('\n');
        var builder = new StringBuilder("<table border=1 style='margin: auto; width: max-content;'>\n");

        builder.Append("  <thead><tr>");
        foreach (string cell in lines[0].Split('|'))
        {
            builder.Append($"<th style='text-align: center;'>{cell.Trim()}</th>");
        }

        builder.Append("</tr></thead>\n  <tbody>\n");

        for (int i = 1; i < lines.Length; i++)
        {
            builder.Append("    <tr>");
            foreach (string cell in lines[i].Split('|'))
            {
                builder.Append($"<td style='text-align: center;'>{cell.Trim()}</td>");
            }

            builder.Append("</tr>\n");
        }

        builder.Append("  </tbody>\n</table>");
        return builder.ToString();
    }

    /// <summary>
    /// Promotes a leading keyword such as <c>Abstract</c> to a heading and leaves the rest alone.
    /// </summary>
    /// <remarks>
    /// Port of <c>format_first_line</c>: split on the delimiter, find the first non-blank piece,
    /// and reformat it only when it equals one of the templates. Anything else — including a piece
    /// that merely starts with the keyword — is left as it was, and the pieces are re-joined with
    /// the same delimiter.
    /// </remarks>
    private static string FormatFirstLine(
        string content,
        string[] templates,
        Func<string, string> format,
        char splitter)
    {
        string[] pieces = content.Split(splitter);

        for (int i = 0; i < pieces.Length; i++)
        {
            if (pieces[i].Trim().Length == 0)
            {
                continue;
            }

            if (templates.Contains(pieces[i].ToLowerInvariant()))
            {
                pieces[i] = format(pieces[i]);
            }

            break;
        }

        return string.Join(splitter, pieces);
    }

    /// <summary>Joins a formula with its equation number, as <c>merge_formula_and_number</c>.</summary>
    private static string MergeFormulaAndNumber(string formula, string number) =>
        $"$${formula.Replace("$$", string.Empty, StringComparison.Ordinal)} \\tag*{{{number}}}$$";

    private static string CollapseSoftNewlines(string value) =>
        value.Replace("-\n", string.Empty, StringComparison.Ordinal)
            .Replace("\n", " ", StringComparison.Ordinal);

    private static string NormalizeNewlines(string value) =>
        value.Replace("\n\n", "\n", StringComparison.Ordinal)
            .Replace("\n", "\n\n", StringComparison.Ordinal);
}
