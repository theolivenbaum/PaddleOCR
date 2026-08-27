using System.Text;
using System.Text.RegularExpressions;

namespace PaddleOcrSharp.Pipeline;

/// <summary>
/// Renders recognised blocks as markdown.
/// </summary>
/// <remarks>
/// Port of PaddleX's <c>MarkdownConverter.convert</c> together with the label-to-formatter map in
/// <c>markdown_format_funcs.build_handle_funcs_dict</c>: blocks are joined by a blank line, and
/// each label gets its own transformation — headings for titles, <c>$$</c> for display formulas,
/// raw HTML for tables, and image links for figures.
/// </remarks>
public static partial class MarkdownWriter
{
    [GeneratedRegex(
        @"^\s*((?:[1-9][0-9]*(?:\.[1-9][0-9]*)*[\.、]?|[\(\（](?:[1-9][0-9]*|[一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+)[\)\）]|[一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+[、\.]?|(?:I|II|III|IV|V|VI|VII|VIII|IX|X)(?:\.|\s)))(\s*)(.*)$")]
    private static partial Regex TitleNumbering();

    /// <summary>Renders <paramref name="blocks"/> as markdown.</summary>
    public static string Write(IReadOnlyList<ParsedBlock> blocks, MarkdownOptions options)
    {
        var builder = new StringBuilder();

        for (int i = 0; i < blocks.Count; i++)
        {
            ParsedBlock block = blocks[i];
            if (options.IgnoredLabels.Contains(block.Label))
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

            string formatted = Format(block, content, options);
            if (formatted.Length == 0)
            {
                continue;
            }

            if (builder.Length > 0)
            {
                builder.Append("\n\n");
            }

            builder.Append(formatted);
        }

        return builder.ToString();
    }

    private static string Format(ParsedBlock block, string content, MarkdownOptions options) => block.Label switch
    {
        "doc_title" => CollapseSoftNewlines($"# {content}"),
        "paragraph_title" or "abstract_title" or "reference_title" or "content_title" => FormatTitle(content),
        "table" => content,
        "display_formula" or "formula" or "inline_formula" => FormatFormula(content),
        "formula_number" => options.ShowFormulaNumber ? string.Empty : content,
        "image" or "chart" or "header_image" or "footer_image" => FormatImage(block, options),
        "seal" => content,
        "content" => content.Replace("-\n", "  \n", StringComparison.Ordinal)
            .Replace("\n", "  \n", StringComparison.Ordinal),
        "algorithm" => content.Trim('\n'),
        "abstract" => FormatFirstLine(content, ["摘要", "abstract"], line => $"## {line}\n", ' '),
        "reference" => FormatFirstLine(content, ["参考文献", "references"], line => $"## {line}", '\n'),
        "number" or "footnote" or "header" or "footer" or "aside_text" or "spotting" => content,
        _ => NormalizeNewlines(content),
    };

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

    private static string FormatFormula(string content)
    {
        string trimmed = content.Trim();
        if (trimmed.Length == 0)
        {
            return string.Empty;
        }

        return trimmed.StartsWith("$$", StringComparison.Ordinal)
            ? trimmed
            : $"$${trimmed}$$";
    }

    private static string FormatImage(ParsedBlock block, MarkdownOptions options)
    {
        if (block.ImagePath is null)
        {
            return block.Content;
        }

        string path = $"{options.ImageDirectory}/{block.ImagePath}";
        string tag = $"<div style=\"text-align: center;\"><img src=\"{path}\" alt=\"Image\" /></div>";
        return block.Content.Length > 0 ? $"{tag}\n\n{block.Content}" : tag;
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

    /// <summary>Joins a formula with its equation number, as upstream's helper does.</summary>
    private static string MergeFormulaAndNumber(string formula, string number)
    {
        string trimmed = number.Trim();
        return trimmed.Length == 0 ? formula : $"{formula.TrimEnd()} \\tag{{{trimmed.Trim('(', ')')}}}";
    }

    private static string CollapseSoftNewlines(string value) =>
        value.Replace("-\n", string.Empty, StringComparison.Ordinal)
            .Replace("\n", " ", StringComparison.Ordinal);

    private static string NormalizeNewlines(string value) =>
        value.Replace("\n\n", "\n", StringComparison.Ordinal)
            .Replace("\n", "\n\n", StringComparison.Ordinal);
}
