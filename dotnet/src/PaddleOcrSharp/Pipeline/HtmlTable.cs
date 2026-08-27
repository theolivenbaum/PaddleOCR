using System.Text;
using System.Text.RegularExpressions;

namespace PaddleOcrSharp.Pipeline;

/// <summary>One cell of a parsed table.</summary>
/// <param name="ColumnSpan">Columns the cell covers.</param>
/// <param name="RowSpan">Rows the cell covers.</param>
/// <param name="Text">The cell's text with markup stripped.</param>
public readonly record struct HtmlTableCell(int ColumnSpan, int RowSpan, string Text);

/// <summary>One row of a parsed table.</summary>
/// <param name="Markup">The row's original markup, <c>&lt;tr&gt;</c> included.</param>
/// <param name="Cells">The row's cells, in order.</param>
public readonly record struct HtmlTableRow(string Markup, IReadOnlyList<HtmlTableCell> Cells);

/// <summary>
/// Just enough of an HTML table reader to compare and splice the tables the pipeline produces.
/// </summary>
/// <remarks>
/// Rows keep their original markup, so joining two tables is a splice rather than a
/// re-serialisation — which is also what BeautifulSoup's round trip amounts to for markup this
/// simple, and avoids inventing a normal form the rest of the pipeline would then disagree with.
/// </remarks>
public static partial class HtmlTable
{
    [GeneratedRegex(@"<tr\b[^>]*>.*?</tr\s*>", RegexOptions.IgnoreCase | RegexOptions.Singleline)]
    private static partial Regex RowPattern();

    [GeneratedRegex(@"<(td|th)\b([^>]*)>(.*?)</\1\s*>", RegexOptions.IgnoreCase | RegexOptions.Singleline)]
    private static partial Regex CellPattern();

    [GeneratedRegex(@"\b(colspan|rowspan)\s*=\s*""?(\d+)""?", RegexOptions.IgnoreCase)]
    private static partial Regex SpanPattern();

    [GeneratedRegex("<[^>]*>", RegexOptions.Singleline)]
    private static partial Regex TagPattern();

    /// <summary>Reads a table's rows and cells.</summary>
    public static IReadOnlyList<HtmlTableRow> Parse(string html)
    {
        var rows = new List<HtmlTableRow>();

        foreach (Match row in RowPattern().Matches(html))
        {
            var cells = new List<HtmlTableCell>();

            foreach (Match cell in CellPattern().Matches(row.Value))
            {
                int columnSpan = 1;
                int rowSpan = 1;

                foreach (Match span in SpanPattern().Matches(cell.Groups[2].Value))
                {
                    int value = int.Parse(span.Groups[2].ValueSpan);
                    if (span.Groups[1].Value.Equals("colspan", StringComparison.OrdinalIgnoreCase))
                    {
                        columnSpan = value;
                    }
                    else
                    {
                        rowSpan = value;
                    }
                }

                cells.Add(new HtmlTableCell(columnSpan, rowSpan, TagPattern().Replace(cell.Groups[3].Value, string.Empty)));
            }

            rows.Add(new HtmlTableRow(row.Value, cells));
        }

        return rows;
    }

    /// <summary>
    /// The table's width in columns, counting the space spanning cells occupy.
    /// </summary>
    /// <remarks>
    /// A cell with a row span reserves its columns on the rows below, so the count has to walk an
    /// occupancy grid rather than add up one row's spans.
    /// </remarks>
    public static int TotalColumns(IReadOnlyList<HtmlTableRow> rows)
    {
        if (rows.Count == 0)
        {
            return 0;
        }

        var occupied = new HashSet<(int Row, int Column)>();
        int maximum = 0;

        for (int r = 0; r < rows.Count; r++)
        {
            int column = 0;

            foreach (HtmlTableCell cell in rows[r].Cells)
            {
                while (occupied.Contains((r, column)))
                {
                    column++;
                }

                for (int y = r; y < r + cell.RowSpan; y++)
                {
                    for (int x = column; x < column + cell.ColumnSpan; x++)
                    {
                        occupied.Add((y, x));
                    }
                }

                column += cell.ColumnSpan;
                maximum = Math.Max(maximum, column);
            }
        }

        return maximum;
    }

    /// <summary>The columns a row covers, counting spans.</summary>
    public static int RowColumns(HtmlTableRow row) => row.Cells.Sum(cell => cell.ColumnSpan);

    /// <summary>The cells a row has, regardless of how far each spans.</summary>
    public static int VisualColumns(HtmlTableRow row) => row.Cells.Count;

    /// <summary>
    /// How many rows at the top of both tables are the same, and whether they count as a header.
    /// </summary>
    /// <param name="first">The earlier table's rows.</param>
    /// <param name="second">The later table's rows.</param>
    /// <param name="maximumHeaderRows">How far down to look.</param>
    public static (int Rows, bool Match) DetectHeaders(
        IReadOnlyList<HtmlTableRow> first,
        IReadOnlyList<HtmlTableRow> second,
        int maximumHeaderRows = 5)
    {
        int limit = Math.Min(Math.Min(first.Count, second.Count), maximumHeaderRows);
        int headerRows = 0;
        bool match = true;

        for (int i = 0; i < limit; i++)
        {
            IReadOnlyList<HtmlTableCell> a = first[i].Cells;
            IReadOnlyList<HtmlTableCell> b = second[i].Cells;

            if (a.Count != b.Count)
            {
                match = headerRows > 0;
                break;
            }

            bool same = true;
            for (int c = 0; c < a.Count; c++)
            {
                if (Normalize(a[c].Text) != Normalize(b[c].Text) || a[c].ColumnSpan != b[c].ColumnSpan)
                {
                    same = false;
                    break;
                }
            }

            if (!same)
            {
                match = headerRows > 0;
                break;
            }

            headerRows++;
        }

        return (headerRows, headerRows != 0 && match);
    }

    /// <summary>
    /// Whether the first table's last row lines up with the second's first row of data.
    /// </summary>
    public static bool RowsLineUp(
        IReadOnlyList<HtmlTableRow> first,
        IReadOnlyList<HtmlTableRow> second)
    {
        if (first.Count == 0 || second.Count == 0)
        {
            return false;
        }

        (int headerRows, _) = DetectHeaders(first, second);
        if (second.Count <= headerRows)
        {
            return false;
        }

        HtmlTableRow last = first[^1];
        HtmlTableRow firstData = second[headerRows];

        return RowColumns(last) == RowColumns(firstData)
            || VisualColumns(last) == VisualColumns(firstData);
    }

    /// <summary>Appends <paramref name="second"/>'s data rows to <paramref name="first"/>.</summary>
    /// <param name="first">The earlier table's HTML.</param>
    /// <param name="second">The later table's HTML.</param>
    /// <param name="skipRows">Rows at the top of the later table to drop, its repeated header.</param>
    public static string Append(string first, string second, int skipRows)
    {
        IReadOnlyList<HtmlTableRow> rows = Parse(second);
        if (rows.Count <= skipRows)
        {
            return first;
        }

        int close = first.LastIndexOf("</table", StringComparison.OrdinalIgnoreCase);
        if (close < 0)
        {
            close = first.Length;
        }

        var builder = new StringBuilder(first[..close]);
        for (int i = skipRows; i < rows.Count; i++)
        {
            builder.Append(rows[i].Markup);
        }

        return builder.Append(first[close..]).ToString();
    }

    /// <summary>
    /// Strips whitespace and folds full-width characters onto their ASCII counterparts.
    /// </summary>
    /// <remarks>
    /// <c>full_to_half</c>: a header typed in full-width Latin — common in Chinese documents —
    /// should still match the same header typed in ASCII on the following page.
    /// </remarks>
    public static string Normalize(string text)
    {
        var builder = new StringBuilder(text.Length);

        foreach (char c in text)
        {
            if (char.IsWhiteSpace(c))
            {
                continue;
            }

            builder.Append(c is >= '！' and <= '～' ? (char)(c - 0xFEE0) : c);
        }

        return builder.ToString();
    }
}
