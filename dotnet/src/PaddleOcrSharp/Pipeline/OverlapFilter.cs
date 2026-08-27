using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models.Layout;

namespace PaddleOcrSharp.Pipeline;

/// <summary>
/// Drops layout regions that sit almost entirely inside another one.
/// </summary>
/// <remarks>
/// <para>
/// Port of <c>filter_overlap_boxes</c>. Three rules apply, in upstream's order:
/// </para>
/// <list type="bullet">
///   <item><c>reference</c> regions are removed outright;</item>
///   <item>degenerate boxes (under 6 px on a side) are removed;</item>
///   <item>
///     a pair overlapping by more than 0.7 of the smaller box loses the smaller one — except when
///     an <c>inline_formula</c> is involved, where the formula goes at 0.5, and except between
///     unlike figure-ish labels, which are allowed to nest.
///   </item>
/// </list>
/// <para>
/// The pairwise loop keeps upstream's sequential ordering: a box dropped early stops
/// participating, and that changes which of a later pair survives.
/// </para>
/// </remarks>
public static class OverlapFilter
{
    private const float SmallSideThreshold = 6f;
    private const float InlineFormulaThreshold = 0.5f;
    private const float ContainmentThreshold = 0.7f;

    private static readonly string[] FigureLabels = ["image", "table", "seal", "chart"];

    /// <summary>Filters <paramref name="boxes"/>, preserving order.</summary>
    public static List<LayoutBox> Apply(IReadOnlyList<LayoutBox> boxes)
    {
        List<LayoutBox> candidates = [.. boxes.Where(box => box.Label != "reference")];
        if (candidates.Count == 0)
        {
            return candidates;
        }

        bool[] dropped = new bool[candidates.Count];

        for (int i = 0; i < candidates.Count; i++)
        {
            if (candidates[i].Width < SmallSideThreshold || candidates[i].Height < SmallSideThreshold)
            {
                dropped[i] = true;
            }

            for (int j = i + 1; j < candidates.Count; j++)
            {
                if (dropped[i] || dropped[j])
                {
                    continue;
                }

                float overlap = BoxGeometry.OverlapRatio(candidates[i], candidates[j], OverlapReference.Small);

                if (candidates[i].Label == "inline_formula" || candidates[j].Label == "inline_formula")
                {
                    if (overlap > InlineFormulaThreshold)
                    {
                        if (candidates[i].Label == "inline_formula")
                        {
                            dropped[i] = true;
                        }

                        if (candidates[j].Label == "inline_formula")
                        {
                            dropped[j] = true;
                        }
                    }

                    continue;
                }

                if (overlap <= ContainmentThreshold)
                {
                    continue;
                }

                // Boxes can overlap heavily while the regions themselves barely touch — two
                // columns of a slanted scan, a caption beside a figure. When both regions have an
                // outline, that is what decides, not the rectangles around them.
                if (candidates[i].Polygon is { Length: > 2 } first
                    && candidates[j].Polygon is { Length: > 2 } second
                    && Polygons.OverlapRatio(first, second, Polygons.OverlapMode.Small)
                        < ContainmentThreshold)
                {
                    continue;
                }

                // A figure inside a table (or vice versa) is a real nesting, not a duplicate.
                bool iIsFigure = FigureLabels.Contains(candidates[i].Label);
                bool jIsFigure = FigureLabels.Contains(candidates[j].Label);
                if ((iIsFigure || jIsFigure) && candidates[i].Label != candidates[j].Label)
                {
                    bool involvesTable = candidates[i].Label == "table" || candidates[j].Label == "table";
                    if (!involvesTable || (iIsFigure && jIsFigure))
                    {
                        continue;
                    }
                }

                float areaI = Math.Abs(candidates[i].Width * candidates[i].Height);
                float areaJ = Math.Abs(candidates[j].Width * candidates[j].Height);
                if (areaI >= areaJ)
                {
                    dropped[j] = true;
                }
                else
                {
                    dropped[i] = true;
                }
            }
        }

        var result = new List<LayoutBox>(candidates.Count);
        for (int i = 0; i < candidates.Count; i++)
        {
            if (!dropped[i])
            {
                result.Add(candidates[i]);
            }
        }

        return result;
    }
}
