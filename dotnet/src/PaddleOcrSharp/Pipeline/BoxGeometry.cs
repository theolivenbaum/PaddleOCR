using PaddleOcrSharp.Models.Layout;

namespace PaddleOcrSharp.Pipeline;

/// <summary>What an overlap ratio is measured against.</summary>
public enum OverlapReference
{
    /// <summary>The union of both boxes.</summary>
    Union,

    /// <summary>The smaller of the two boxes.</summary>
    Small,

    /// <summary>The larger of the two boxes.</summary>
    Large,
}

/// <summary>
/// Box overlap measures shared by the layout post-processing and the block merger.
/// </summary>
/// <remarks>Port of the helpers in PaddleX's <c>layout_parsing/utils.py</c>.</remarks>
public static class BoxGeometry
{
    /// <summary>Area overlap between two boxes, relative to <paramref name="reference"/>.</summary>
    public static float OverlapRatio(LayoutBox a, LayoutBox b, OverlapReference reference = OverlapReference.Union)
    {
        float intersection = a.IntersectionWith(b);
        float areaA = Math.Abs(a.Width * a.Height);
        float areaB = Math.Abs(b.Width * b.Height);

        float referenceArea = reference switch
        {
            OverlapReference.Small => Math.Min(areaA, areaB),
            OverlapReference.Large => Math.Max(areaA, areaB),
            _ => areaA + areaB - intersection,
        };

        return referenceArea <= 0f ? 0f : intersection / referenceArea;
    }

    /// <summary>
    /// Overlap of the boxes' projections onto one axis, relative to <paramref name="reference"/>.
    /// </summary>
    /// <param name="a">First box.</param>
    /// <param name="b">Second box.</param>
    /// <param name="horizontal">Whether to project onto the x axis.</param>
    /// <param name="reference">What the overlap is measured against.</param>
    public static float ProjectionOverlapRatio(
        LayoutBox a,
        LayoutBox b,
        bool horizontal,
        OverlapReference reference = OverlapReference.Union)
    {
        (float startA, float endA) = horizontal ? (a.Left, a.Right) : (a.Top, a.Bottom);
        (float startB, float endB) = horizontal ? (b.Left, b.Right) : (b.Top, b.Bottom);

        float overlap = Math.Min(endA, endB) - Math.Max(startA, startB);
        if (overlap <= 0f)
        {
            return 0f;
        }

        float extent = reference switch
        {
            OverlapReference.Small => Math.Min(endA - startA, endB - startB),
            OverlapReference.Large => Math.Max(endA - startA, endB - startB),
            _ => Math.Max(endA, endB) - Math.Min(startA, startB),
        };

        return extent <= 0f ? 0f : overlap / extent;
    }

    /// <summary>The smallest box containing both inputs.</summary>
    public static LayoutBox Union(LayoutBox a, LayoutBox b) => a with
    {
        Left = Math.Min(a.Left, b.Left),
        Top = Math.Min(a.Top, b.Top),
        Right = Math.Max(a.Right, b.Right),
        Bottom = Math.Max(a.Bottom, b.Bottom),
    };
}
