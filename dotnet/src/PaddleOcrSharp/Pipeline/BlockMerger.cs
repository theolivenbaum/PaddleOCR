using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models.Layout;

namespace PaddleOcrSharp.Pipeline;

/// <summary>A run of layout regions that are recognised as one image.</summary>
/// <param name="Indices">Indices into the detection list, in order.</param>
/// <param name="Alignments">One alignment per join; empty for a single-region group.</param>
public readonly record struct BlockGroup(IReadOnlyList<int> Indices, IReadOnlyList<StackAlignment> Alignments)
{
    /// <summary>
    /// The identifier every member of the group carries, or <see langword="null"/> when the
    /// regions were never candidates for merging.
    /// </summary>
    /// <remarks>
    /// <c>group_id</c>, which upstream sets to the index of the group's first block. A text
    /// region that ends up alone still gets one — it went through the merger and came out a
    /// group of one — whereas a figure, a table, or a run the aspect-ratio guard abandoned does
    /// not.
    /// </remarks>
    public int? GroupId { get; init; }
}

/// <summary>
/// Decides which adjacent text regions should be recognised together.
/// </summary>
/// <remarks>
/// <para>
/// Port of <c>merge_blocks</c>. Two situations qualify, both only between consecutive <c>text</c>
/// regions:
/// </para>
/// <list type="bullet">
///   <item>
///     <b>cross-column continuation</b> — the regions do not overlap horizontally, the second
///     starts to the right of the first and above its bottom, and the gap is under 30% of the
///     wider region;
///   </item>
///   <item>
///     <b>vertical continuation</b> — the regions overlap horizontally, are within half a region
///     height of each other, share exactly one aligned edge, and the box that would contain both
///     touches a region that must not be merged (a figure or table), which is what indicates the
///     paragraph was split around it.
///   </item>
/// </list>
/// <para>
/// A group whose stacked image would be more than three times taller than wide is abandoned:
/// the native-resolution encoder would shrink such a strip until the text was unreadable.
/// </para>
/// <para>
/// Grouping also settles the page's block order. The groups come back in the order upstream
/// emits them, which is not detection order: a figure that sits between the two halves of a
/// paragraph is emitted after both halves rather than between them.
/// </para>
/// </remarks>
public static class BlockMerger
{
    private const float AlignmentTolerance = 5f;
    private const float MaximumAspectRatio = 3f;

    /// <summary>
    /// Groups <paramref name="regions"/> for recognition.
    /// </summary>
    /// <param name="regions">Detected regions, in reading order.</param>
    /// <param name="sizes">Crop size of each region, used for the aspect-ratio guard.</param>
    /// <param name="nonMergeLabels">Labels that are never merged.</param>
    public static List<BlockGroup> Group(
        IReadOnlyList<LayoutBox> regions,
        IReadOnlyList<(int Width, int Height)> sizes,
        IReadOnlyCollection<string> nonMergeLabels)
    {
        var mergeable = new List<int>();
        for (int i = 0; i < regions.Count; i++)
        {
            if (!nonMergeLabels.Contains(regions[i].Label))
            {
                mergeable.Add(i);
            }
        }

        var runs = new List<(List<int> Indices, List<StackAlignment> Alignments)>();
        List<int> current = [];
        List<StackAlignment> alignments = [];

        for (int position = 0; position < mergeable.Count; position++)
        {
            int index = mergeable[position];

            if (current.Count == 0)
            {
                current = [index];
                alignments = [];
                continue;
            }

            int previousIndex = mergeable[position - 1];
            LayoutBox previous = regions[previousIndex];
            LayoutBox box = regions[index];

            StackAlignment? alignment = Classify(regions, previousIndex, index, previous, box, nonMergeLabels);
            if (alignment is { } value)
            {
                current.Add(index);
                alignments.Add(value);
            }
            else
            {
                runs.Add((current, alignments));
                current = [index];
                alignments = [];
            }
        }

        if (current.Count > 0)
        {
            runs.Add((current, alignments));
        }

        // A run's members are consecutive among the mergeable regions but not among all of them:
        // the figure a paragraph wraps around sits between them. Upstream walks the detection
        // list, and on reaching a run's first region emits the whole run, then whatever
        // unmergeable regions fell inside the run's span, then continues past the run's end. So a
        // figure caught between the two halves of one paragraph comes out *after* both halves
        // rather than between them, and that reordering reaches the markdown.
        var runByStart = new Dictionary<int, (int End, List<int> Indices, List<StackAlignment> Alignments)>();
        foreach ((List<int> indices, List<StackAlignment> runAlignments) in runs)
        {
            runByStart[indices[0]] = (indices[^1], indices, runAlignments);
        }

        var groups = new List<BlockGroup>();
        var claimed = new HashSet<int>();

        int cursor = 0;
        while (cursor < regions.Count)
        {
            if (!runByStart.TryGetValue(cursor, out (int End, List<int> Indices, List<StackAlignment> Alignments) run)
                || run.Indices.Any(claimed.Contains))
            {
                if (claimed.Add(cursor))
                {
                    groups.Add(new BlockGroup([cursor], []));
                }

                cursor++;
                continue;
            }

            // The guard applies to every run, single-region ones included: a lone column narrow
            // enough to trip it is abandoned the same way, which is what leaves it without a
            // group identifier.
            if (IsTooTall(run.Indices, sizes))
            {
                foreach (int index in run.Indices)
                {
                    groups.Add(new BlockGroup([index], []));
                    claimed.Add(index);
                }
            }
            else
            {
                groups.Add(new BlockGroup(run.Indices, run.Alignments) { GroupId = run.Indices[0] });
                foreach (int index in run.Indices)
                {
                    claimed.Add(index);
                }
            }

            for (int inner = cursor + 1; inner < run.End; inner++)
            {
                if (claimed.Add(inner))
                {
                    groups.Add(new BlockGroup([inner], []));
                }
            }

            cursor = run.End + 1;
        }

        return groups;
    }

    private static bool IsTooTall(List<int> indices, IReadOnlyList<(int Width, int Height)> sizes)
    {
        int width = indices.Max(index => sizes[index].Width);
        int height = indices.Sum(index => sizes[index].Height);
        return width == 0 || (float)height / width >= MaximumAspectRatio;
    }

    private static StackAlignment? Classify(
        IReadOnlyList<LayoutBox> regions,
        int previousIndex,
        int index,
        LayoutBox previous,
        LayoutBox box,
        IReadOnlyCollection<string> nonMergeLabels)
    {
        if (box.Label != "text" || previous.Label != "text")
        {
            return null;
        }

        float horizontalOverlap = BoxGeometry.ProjectionOverlapRatio(box, previous, horizontal: true);

        bool isCross = horizontalOverlap == 0f
            && box.Left > previous.Right
            && box.Top < previous.Bottom
            && box.Left - previous.Right < Math.Max(previous.Width, box.Width) * 0.3f;

        if (isCross)
        {
            return StackAlignment.Center;
        }

        bool leftAligned = IsAligned(box.Left, previous.Left);
        bool rightAligned = IsAligned(box.Right, previous.Right);

        bool isVertical = horizontalOverlap > 0f
            && box.Bottom >= previous.Top
            && Math.Abs(box.Top - previous.Bottom) < Math.Max(previous.Height, box.Height) * 0.5f
            && (leftAligned ^ rightAligned)
            && TouchesUnmergeable(regions, previousIndex, index, nonMergeLabels);

        if (!isVertical)
        {
            return null;
        }

        return leftAligned ? StackAlignment.Left : StackAlignment.Right;
    }

    private static bool IsAligned(float a, float b) => Math.Abs(a - b) <= AlignmentTolerance;

    /// <summary>
    /// Whether the box spanning both regions overlaps a region that must not be merged.
    /// </summary>
    private static bool TouchesUnmergeable(
        IReadOnlyList<LayoutBox> regions,
        int previousIndex,
        int index,
        IReadOnlyCollection<string> nonMergeLabels)
    {
        LayoutBox span = BoxGeometry.Union(regions[previousIndex], regions[index]);

        for (int i = 0; i < regions.Count; i++)
        {
            if (i == previousIndex || i == index || !nonMergeLabels.Contains(regions[i].Label))
            {
                continue;
            }

            if (BoxGeometry.OverlapRatio(span, regions[i]) > 0f)
            {
                return true;
            }
        }

        return false;
    }
}
