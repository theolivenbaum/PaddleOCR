namespace PaddleOcrSharp.Pipeline;

/// <summary>
/// Rejoins a table that a page break split in two.
/// </summary>
/// <remarks>
/// Port of <c>merge_tables_across_pages</c> in
/// <c>paddlex/inference/pipelines/layout_parsing/merge_table.py</c>, which
/// <c>restructure_pages(merge_tables=True)</c> applies over a whole document. The last table on
/// one page and the first on the next are treated as one when nothing but page furniture stands
/// between them and their shapes agree; the second table's rows move into the first and its own
/// block is left empty, keeping the page's block list intact.
/// </remarks>
public static class TableMerger
{
    /// <summary>Labels allowed to follow the earlier table without ending it.</summary>
    private static readonly string[] AllowedAfter =
        ["footer", "vision_footnote", "number", "footnote", "footer_image", "seal"];

    /// <summary>Labels allowed to precede the later table without separating it.</summary>
    private static readonly string[] AllowedBefore = ["header", "header_image", "number", "seal"];

    /// <summary>
    /// Merges tables split across consecutive pages.
    /// </summary>
    /// <param name="pages">The document's pages, in order.</param>
    /// <returns>The pages, with any split table rejoined into its first half.</returns>
    public static IReadOnlyList<ParsedPage> Apply(IReadOnlyList<ParsedPage> pages)
    {
        if (pages.Count < 2)
        {
            return pages;
        }

        var blocks = pages.Select(page => page.Blocks.ToArray()).ToArray();

        // Backwards, so a table running over three pages collapses into the first: the third is
        // folded into the second before the second is folded into the first.
        for (int i = pages.Count - 1; i > 0; i--)
        {
            ParsedBlock[] previous = blocks[i - 1];
            ParsedBlock[] current = blocks[i];

            int previousIndex = Array.FindLastIndex(previous, block => block.Label == "table");
            int currentIndex = Array.FindIndex(current, block => block.Label == "table");

            if (previousIndex < 0 || currentIndex < 0)
            {
                continue;
            }

            if (!CanMerge(previous, previousIndex, current, currentIndex))
            {
                continue;
            }

            IReadOnlyList<HtmlTableRow> before = HtmlTable.Parse(previous[previousIndex].Content);
            IReadOnlyList<HtmlTableRow> after = HtmlTable.Parse(current[currentIndex].Content);
            (int headerRows, _) = HtmlTable.DetectHeaders(before, after);

            previous[previousIndex] = previous[previousIndex] with
            {
                Content = HtmlTable.Append(
                    previous[previousIndex].Content, current[currentIndex].Content, headerRows),
            };

            current[currentIndex] = current[currentIndex] with { Content = string.Empty };
        }

        return [.. pages.Select((page, i) => page with { Blocks = blocks[i] })];
    }

    /// <summary>Whether two tables either side of a page break are one table.</summary>
    private static bool CanMerge(
        IReadOnlyList<ParsedBlock> previousPage,
        int previousIndex,
        IReadOnlyList<ParsedBlock> currentPage,
        int currentIndex)
    {
        ParsedBlock before = previousPage[previousIndex];
        ParsedBlock after = currentPage[currentIndex];

        // Boxes are compared as integers, which is how the pipeline stores them.
        int beforeWidth = (int)before.Box.Right - (int)before.Box.Left;
        int afterWidth = (int)after.Box.Right - (int)after.Box.Left;

        if (beforeWidth == 0 || afterWidth == 0)
        {
            return false;
        }

        if (Math.Abs(afterWidth - beforeWidth) / (double)Math.Min(afterWidth, beforeWidth) >= 0.1)
        {
            return false;
        }

        for (int i = previousIndex + 1; i < previousPage.Count; i++)
        {
            if (!AllowedAfter.Contains(previousPage[i].Label))
            {
                return false;
            }
        }

        for (int i = 0; i < currentIndex; i++)
        {
            if (!AllowedBefore.Contains(currentPage[i].Label))
            {
                return false;
            }
        }

        if (before.Content.Length == 0 || after.Content.Length == 0)
        {
            return false;
        }

        IReadOnlyList<HtmlTableRow> beforeRows = HtmlTable.Parse(before.Content);
        IReadOnlyList<HtmlTableRow> afterRows = HtmlTable.Parse(after.Content);

        return HtmlTable.TotalColumns(beforeRows) == HtmlTable.TotalColumns(afterRows)
            || HtmlTable.RowsLineUp(beforeRows, afterRows);
    }
}
