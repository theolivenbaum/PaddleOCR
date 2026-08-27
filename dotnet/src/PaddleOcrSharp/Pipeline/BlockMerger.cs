using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models.Layout;

namespace PaddleOcrSharp.Pipeline;

/// <summary>A run of layout regions that are recognised as one image.</summary>
/// <param name="Indices">Indices into the detection list, in order.</param>
/// <param name="Alignments">One alignment per join; empty for a single-region group.</param>
public readonly record struct BlockGroup(IReadOnlyList<int> Indices, IReadOnlyList<StackAlignment> Alignments);

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

        var groups = new List<BlockGroup>();
        var claimed = new HashSet<int>();

        foreach ((List<int> indices, List<StackAlignment> runAlignments) in runs)
        {
            if (indices.Count > 1 && IsTooTall(indices, sizes))
            {
                foreach (int index in indices)
                {
                    groups.Add(new BlockGroup([index], []));
                    claimed.Add(index);
                }

                continue;
            }

            groups.Add(new BlockGroup(indices, runAlignments));
            foreach (int index in indices)
            {
                claimed.Add(index);
            }
        }

        // Regions that were never candidates keep their own group, and the whole list is put back
        // into detection order so reading order survives the grouping.
        for (int i = 0; i < regions.Count; i++)
        {
            if (!claimed.Contains(i))
            {
                groups.Add(new BlockGroup([i], []));
            }
        }

        groups.Sort(static (left, right) => left.Indices[0].CompareTo(right.Indices[0]));
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
