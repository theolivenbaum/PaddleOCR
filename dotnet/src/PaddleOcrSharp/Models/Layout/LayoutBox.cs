namespace PaddleOcrSharp.Models.Layout;

/// <summary>One detected layout region, in original-image pixel coordinates.</summary>
/// <param name="ClassId">Detector class index.</param>
/// <param name="Label">Class name.</param>
/// <param name="Score">Confidence in <c>[0, 1]</c>.</param>
/// <param name="Left">Left edge.</param>
/// <param name="Top">Top edge.</param>
/// <param name="Right">Right edge.</param>
/// <param name="Bottom">Bottom edge.</param>
/// <param name="ReadingOrder">
/// Position in the page's reading order, as predicted by the model's ordering head.
/// </param>
public readonly record struct LayoutBox(
    int ClassId,
    string Label,
    float Score,
    float Left,
    float Top,
    float Right,
    float Bottom,
    int ReadingOrder)
{
    /// <summary>
    /// The region's outline, when the detector's mask head produced one.
    /// </summary>
    /// <remarks>
    /// Present whenever <see cref="LayoutOptions.ShapeMode"/> is not
    /// <see cref="LayoutShapeMode.Rect"/>. Its corners can sit outside the box: a rotated region's
    /// quad is the minimum-area rectangle around the mask, not a subset of the axis-aligned one.
    /// </remarks>
    public (float X, float Y)[]? Polygon { get; init; }

    /// <summary>Index of the detector query this box came from, which is also its mask's index.</summary>
    internal int QueryIndex { get; init; }

    /// <summary>Width of the region.</summary>
    public float Width => Right - Left;

    /// <summary>Height of the region.</summary>
    public float Height => Bottom - Top;

    /// <summary>Area of the region.</summary>
    public float Area => Math.Max(0f, Width) * Math.Max(0f, Height);

    /// <summary>Intersection area with <paramref name="other"/>.</summary>
    public float IntersectionWith(LayoutBox other)
    {
        float width = Math.Min(Right, other.Right) - Math.Max(Left, other.Left);
        float height = Math.Min(Bottom, other.Bottom) - Math.Max(Top, other.Top);
        return width <= 0 || height <= 0 ? 0f : width * height;
    }

    /// <summary>Intersection over union with <paramref name="other"/>.</summary>
    public float IntersectionOverUnion(LayoutBox other)
    {
        float intersection = IntersectionWith(other);
        float union = Area + other.Area - intersection;
        return union <= 0f ? 0f : intersection / union;
    }

    /// <summary>Returns a copy clamped to a <paramref name="width"/> × <paramref name="height"/> page.</summary>
    public LayoutBox ClampTo(int width, int height) => this with
    {
        Left = Math.Clamp(Left, 0, width),
        Top = Math.Clamp(Top, 0, height),
        Right = Math.Clamp(Right, 0, width),
        Bottom = Math.Clamp(Bottom, 0, height),
    };
}
