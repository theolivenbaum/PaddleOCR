using System.Text.RegularExpressions;

namespace PaddleOcrSharp.Pipeline;

/// <summary>
/// Works out how deep each heading sits in a document's structure.
/// </summary>
/// <remarks>
/// <para>
/// Port of <c>assign_levels_to_parsing_res</c> in
/// <c>paddlex/inference/pipelines/layout_parsing/title_level.py</c>, applied by
/// <c>restructure_pages(relevel_titles=True)</c>. A numbered heading gets its depth from three
/// signals and takes whichever two agree: the depth its numbering implies (<c>1.2.1</c> is three
/// deep), the order in which its numbering style first appeared in the document, and how its
/// text height compares with every other heading's. A heading that is a recognised section name
/// — Abstract, References, 参考文献 — is level one whatever it looks like, and anything left over
/// is decided by height alone.
/// </para>
/// <para>
/// The height comparison is where this departs from upstream, which reaches for scikit-learn's
/// KMeans. That is a seeded local search, so reproducing it would mean reproducing NumPy's
/// random stream and scikit-learn's seeding — brittle, and different between their own versions.
/// One-dimensional k-means has an exact optimum reachable by dynamic programming, so that is
/// what runs here: deterministic, and no worse than the local optimum it stands in for. On the
/// usual document, where headings come in a handful of distinct sizes, both simply rank the
/// sizes and agree.
/// </para>
/// </remarks>
public static partial class TitleLevels
{
    // Roman numerals: I, II, V, X, i., iv), V.
    [GeneratedRegex(@"^\s*([IVX]+)(?:[\.．\)\s]|$)", RegexOptions.IgnoreCase)]
    private static partial Regex Roman();

    // A single letter: A., B), c., D
    [GeneratedRegex(@"^\s*([A-Z])(?:[\.．\)\s])", RegexOptions.IgnoreCase)]
    private static partial Regex Letter();

    // Multi-level numeric numbering: 1, 1.1, 1.2.3, 2.
    [GeneratedRegex(@"^\s*(\d+(?:\.\d+)*)(?![）)])(?:[\.]?\s*|(?=[A-Z]))")]
    private static partial Regex NumberList();

    // Numeric numbering in parentheses: (1), (1.1), （2）, 1)
    [GeneratedRegex(@"^\s*(?:[\(（])?(\d+(?:\.\d+)*)[\)）]")]
    private static partial Regex BracketedNumberList();

    // Chinese numerals: 一, 二, 第一, 十三
    [GeneratedRegex(
        @"^\s*(?:第|[（\(])?([一二三四五六七八九十]{1,2})(?:[章节篇卷部条题讲课回）\)]|(?![a-zA-Z一-龥]))",
        RegexOptions.IgnoreCase)]
    private static partial Regex ChineseNumber();

    /// <summary>Section names that are level one however they are set.</summary>
    private static readonly HashSet<string> SpecialKeywords = new(StringComparer.Ordinal)
    {
        "ABSTRACT", "SUMMARY", "RESUME", "绪论", "引言", "CONTENTS", "REFERENCES", "REFERENCE",
        "参考文献", "APPENDIX", "APPENDICES", "附录", "ACKNOWLEDGMENTS", "INTRODUCTION",
        "BACKGROUNDANDRELATEDWORK", "BACKGROUND", "RELATEDWORK", "THEORETICALMODELS", "DATA",
        "METHOD", "METHODS", "METHODOLOGY", "TOPICANALYSIS", "RESULT", "RESULTS", "DISCUSSION",
        "CONCLUSIONS", "CONCLUSION", "LIMITATIONS", "研究背景", "相关工作", "研究方法", "实验结果",
        "讨论", "结论", "致谢", "目录",
    };

    /// <summary>Assigns a heading level to every <c>paragraph_title</c> in the document.</summary>
    /// <param name="pages">The document's pages, in order.</param>
    public static IReadOnlyList<ParsedPage> Apply(IReadOnlyList<ParsedPage> pages)
    {
        var blocks = pages.Select(page => page.Blocks.ToArray()).ToArray();

        var entries = new List<(int Page, int Index, string Content, int Height)>();

        for (int p = 0; p < blocks.Length; p++)
        {
            for (int b = 0; b < blocks[p].Length; b++)
            {
                if (blocks[p][b].Label == "paragraph_title")
                {
                    entries.Add((p, b, blocks[p][b].Content, Height(blocks[p][b])));
                }
            }
        }

        if (entries.Count == 0)
        {
            return pages;
        }

        int[] levels = Levels(
            [.. entries.Select(entry => entry.Content)],
            [.. entries.Select(entry => entry.Height)]);

        for (int i = 0; i < entries.Count; i++)
        {
            (int page, int index, _, _) = entries[i];
            blocks[page][index] = blocks[page][index] with { TitleLevel = levels[i] };
        }

        return [.. pages.Select((page, i) => page with { Blocks = blocks[i] })];
    }

    /// <summary>
    /// The heading depth for each entry, given its text and its text height.
    /// </summary>
    /// <param name="contents">Each heading's text, in document order.</param>
    /// <param name="heights">Each heading's text height, indexed alike.</param>
    /// <param name="clusters">
    /// A height-to-level map to use instead of clustering the heights, which the parity tests
    /// supply so the rest of the decision can be compared against upstream exactly.
    /// </param>
    public static int[] Levels(
        IReadOnlyList<string> contents,
        IReadOnlyList<int> heights,
        IReadOnlyDictionary<int, int>? clusters = null)
    {
        int count = contents.Count;
        var symbols = new (string? Symbol, int Level)[count];

        for (int i = 0; i < count; i++)
        {
            symbols[i] = SymbolAndLevel(contents[i]);
        }

        IReadOnlyDictionary<int, int> clusterMap = clusters ?? ClusterHeights(heights);

        // Numbering styles are ranked by where each first appears, so the style a document opens
        // with outranks one it only reaches later.
        var order = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach ((string? symbol, int level) in symbols)
        {
            if (level > 0 && symbol is not null && !order.ContainsKey(symbol))
            {
                order[symbol] = order.Count + 1;
            }
        }

        int[] levels = new int[count];
        int firstNumberLevel = 0;

        for (int i = 0; i < count; i++)
        {
            (string? symbol, int level) = symbols[i];
            int clusterLevel = clusterMap[heights[i]];
            string keyword = Keyword(contents[i]);

            if (level > 0 && symbol is not null)
            {
                int relative;

                if (symbol == "NUM_LIST")
                {
                    if (firstNumberLevel != 0)
                    {
                        relative = order[symbol] + (level - firstNumberLevel);
                    }
                    else
                    {
                        firstNumberLevel = level;
                        relative = order[symbol];
                    }
                }
                else
                {
                    relative = order[symbol];
                }

                // Two of the three signals agreeing settles it; otherwise the numbering style's
                // own rank does, since it is the one signal that looks at the whole document.
                levels[i] = Vote(level, relative, clusterLevel) ?? relative;
            }
            else if (SpecialKeywords.Contains(keyword))
            {
                levels[i] = 1;
            }
            else
            {
                levels[i] = clusterLevel;
            }
        }

        return levels;
    }

    /// <summary>The value at least two of the three votes share, if any.</summary>
    private static int? Vote(int first, int second, int third)
    {
        if (first == second || first == third)
        {
            return first;
        }

        return second == third ? second : null;
    }

    /// <summary>
    /// The numbering style a heading opens with and the depth it implies.
    /// </summary>
    /// <remarks>
    /// The order of the tests is upstream's and matters: a bracketed number is checked before a
    /// bare one, and a Roman numeral before a single letter, or <c>I.</c> would read as a letter.
    /// </remarks>
    /// <param name="content">The heading's text.</param>
    public static (string? Symbol, int Level) SymbolAndLevel(string content)
    {
        string text = content.Trim();

        if (BracketedNumberList().IsMatch(text))
        {
            return ("NUM_LIST_BRACKET", 4);
        }

        if (Roman().IsMatch(text))
        {
            return ("ROMAN", 1);
        }

        if (ChineseNumber().IsMatch(text))
        {
            return ("CHINESE_NUM", 1);
        }

        if (Letter().IsMatch(text))
        {
            return ("LETTER", 2);
        }

        Match numbers = NumberList().Match(text);
        if (numbers.Success)
        {
            return ("NUM_LIST", numbers.Groups[1].Value.Count(c => c == '.') + 1);
        }

        return (null, -1);
    }

    /// <summary>
    /// The height of one line of the heading's text.
    /// </summary>
    /// <remarks>
    /// The block's height divided by how many lines its content has, or its width when the block
    /// is taller than it is wide and so reads vertically.
    /// </remarks>
    public static int Height(ParsedBlock block)
    {
        if (block.Label == "doc_title")
        {
            return 0;
        }

        int left = (int)block.Box.Left;
        int top = (int)block.Box.Top;
        int right = (int)Math.Ceiling(block.Box.Right);
        int bottom = (int)Math.Ceiling(block.Box.Bottom);

        int height = bottom - top;
        int width = right - left;
        int lines = block.Content.Trim().Count(c => c == '\n') + 1;

        if (height == 0)
        {
            return 0;
        }

        return (double)width / height >= 1.0 ? height / lines : width / lines;
    }

    /// <summary>The form of a heading's text that is looked up among the section names.</summary>
    private static string Keyword(string content) => content
        .ToUpperInvariant()
        .Trim()
        .TrimEnd('：', ':', ' ')
        .Replace(" ", string.Empty, StringComparison.Ordinal);

    /// <summary>
    /// Groups heading heights into at most <paramref name="clusters"/> levels, largest first.
    /// </summary>
    /// <remarks>
    /// An exact one-dimensional k-means by dynamic programming: with the values sorted, every
    /// cluster of an optimal solution is a contiguous run, so the best split of the first
    /// <c>i</c> values into <c>j</c> runs can be built from the best split of a shorter prefix.
    /// </remarks>
    /// <param name="heights">Every heading's height, repeats included.</param>
    /// <param name="clusters">The most levels to use.</param>
    public static IReadOnlyDictionary<int, int> ClusterHeights(IReadOnlyList<int> heights, int clusters = 4)
    {
        int[] unique = [.. heights.Distinct().Order()];
        if (unique.Length == 0)
        {
            return new Dictionary<int, int>();
        }

        int k = Math.Min(clusters, unique.Length);
        double[] values = [.. heights.Order().Select(h => (double)h)];
        int n = values.Length;

        // Prefix sums make the cost of any contiguous run a constant-time lookup.
        double[] sum = new double[n + 1];
        double[] squares = new double[n + 1];
        for (int i = 0; i < n; i++)
        {
            sum[i + 1] = sum[i] + values[i];
            squares[i + 1] = squares[i] + (values[i] * values[i]);
        }

        double Cost(int from, int to)
        {
            int length = to - from;
            if (length <= 0)
            {
                return 0;
            }

            double total = sum[to] - sum[from];
            return squares[to] - squares[from] - (total * total / length);
        }

        double[,] best = new double[k + 1, n + 1];
        int[,] split = new int[k + 1, n + 1];

        for (int i = 1; i <= n; i++)
        {
            best[0, i] = double.PositiveInfinity;
        }

        for (int j = 1; j <= k; j++)
        {
            for (int i = j; i <= n; i++)
            {
                best[j, i] = double.PositiveInfinity;
                for (int m = j - 1; m < i; m++)
                {
                    double candidate = best[j - 1, m] + Cost(m, i);
                    if (candidate < best[j, i])
                    {
                        best[j, i] = candidate;
                        split[j, i] = m;
                    }
                }
            }
        }

        // Walk the splits back to get each cluster's centre, then rank them: the tallest text is
        // the shallowest heading.
        var centres = new List<double>(k);
        int end = n;
        for (int j = k; j >= 1; j--)
        {
            int start = split[j, end];
            centres.Add((sum[end] - sum[start]) / (end - start));
            end = start;
        }

        centres.Reverse();

        var map = new Dictionary<int, int>(unique.Length);
        foreach (int height in unique)
        {
            int nearest = 0;
            double closest = Math.Abs(height - centres[0]);

            for (int c = 1; c < centres.Count; c++)
            {
                double distance = Math.Abs(height - centres[c]);
                if (distance < closest)
                {
                    closest = distance;
                    nearest = c;
                }
            }

            map[height] = centres.Count - nearest;
        }

        return map;
    }
}
