using PaddleOcrSharp.Imaging;

namespace PaddleOcrSharp.Models.Layout;

/// <summary>
/// Turns the layout detector's per-detection masks into region polygons.
/// </summary>
/// <remarks>
/// Port of <c>extract_polygon_points_by_masks</c>, <c>mask2polygon</c>,
/// <c>extract_custom_vertices</c> and <c>_normalize_layout_polygon</c> in
/// <c>paddlex/inference/models/layout_analysis/processors.py</c>. The detector emits a 200x200
/// mask per query alongside the boxes; upstream's default shape mode turns each into a polygon
/// and uses it for both overlap filtering and cropping, so a slanted or L-shaped region no longer
/// drags its neighbours in with it.
/// </remarks>
public static class LayoutPolygons
{
    /// <summary>Simplification tolerance, as a fraction of the contour's perimeter.</summary>
    private const double EpsilonRatio = 0.004;

    /// <summary>Below this angle a convex vertex is treated as a spike and pushed outward.</summary>
    private const double SharpAngleThreshold = 45;

    /// <summary>Fraction of the caller's distance limit that actually bounds an edge.</summary>
    private const double MaxDistanceRatio = 0.3;

    /// <summary>
    /// Extracts one polygon per detection.
    /// </summary>
    /// <param name="boxes">Detections, already filtered and ordered.</param>
    /// <param name="masks">One <paramref name="maskWidth"/> x <paramref name="maskHeight"/> mask per detection.</param>
    /// <param name="maskWidth">Mask width.</param>
    /// <param name="maskHeight">Mask height.</param>
    /// <param name="scaleX">Model input width over page width.</param>
    /// <param name="scaleY">Model input height over page height.</param>
    /// <param name="mode">Which shape to reduce each polygon to.</param>
    /// <returns>One polygon per detection, in page coordinates.</returns>
    public static (float X, float Y)[][] Extract(
        IReadOnlyList<LayoutBox> boxes,
        ReadOnlySpan<byte> masks,
        int maskWidth,
        int maskHeight,
        double scaleX,
        double scaleY,
        LayoutShapeMode mode)
    {
        var polygons = new (float X, float Y)[boxes.Count][];
        if (boxes.Count == 0)
        {
            return polygons;
        }

        // The mask is a quarter of the model's input resolution on each axis.
        double maskScaleX = scaleX / 4.0;
        double maskScaleY = scaleY / 4.0;

        // Upstream measures this as `max(boxes[:, 4] - boxes[:, 3])`, which is each box's right
        // edge minus its *top* edge rather than its left. Reproduced deliberately: it feeds the
        // edge-length limit below, so changing it would move vertices.
        double widestBox = double.NegativeInfinity;
        foreach (LayoutBox box in boxes)
        {
            widestBox = Math.Max(widestBox, box.Right - box.Top);
        }

        int plane = maskWidth * maskHeight;

        for (int i = 0; i < boxes.Count; i++)
        {
            LayoutBox box = boxes[i];
            int left = (int)box.Left;
            int top = (int)box.Top;
            int right = (int)box.Right;
            int bottom = (int)box.Bottom;

            (float X, float Y)[] rectangle = Rectangle(left, top, right, bottom);

            int width = right - left;
            int height = bottom - top;

            if (width <= 0 || height <= 0)
            {
                polygons[i] = rectangle;
                continue;
            }

            int x0 = Math.Clamp(RoundHalfToEven(left * maskScaleX), 0, maskWidth);
            int x1 = Math.Clamp(RoundHalfToEven(right * maskScaleX), 0, maskWidth);
            int y0 = Math.Clamp(RoundHalfToEven(top * maskScaleY), 0, maskHeight);
            int y1 = Math.Clamp(RoundHalfToEven(bottom * maskScaleY), 0, maskHeight);

            int cropWidth = x1 - x0;
            int cropHeight = y1 - y0;

            if (cropWidth <= 0 || cropHeight <= 0)
            {
                polygons[i] = rectangle;
                continue;
            }

            ReadOnlySpan<byte> mask = masks.Slice(i * plane, plane);
            byte[] cropped = new byte[cropWidth * cropHeight];
            bool any = false;

            for (int y = 0; y < cropHeight; y++)
            {
                for (int x = 0; x < cropWidth; x++)
                {
                    byte value = mask[((y0 + y) * maskWidth) + x0 + x];
                    cropped[(y * cropWidth) + x] = value;
                    any |= value != 0;
                }
            }

            if (!any)
            {
                polygons[i] = rectangle;
                continue;
            }

            byte[] resized = ResizeNearest(cropped, cropWidth, cropHeight, width, height);
            double distanceLimit = width > widestBox * 0.6 ? width : widestBox;

            (float X, float Y)[]? polygon = MaskToPolygon(resized, width, height, distanceLimit);

            if (polygon is { Length: > 0 })
            {
                for (int p = 0; p < polygon.Length; p++)
                {
                    polygon[p] = (polygon[p].X + left, polygon[p].Y + top);
                }
            }

            polygons[i] = Normalize(rectangle, polygon, mode, i > 0 ? polygons[i - 1] : null);
        }

        return polygons;
    }

    /// <summary>
    /// Reduces a binary mask to a polygon, as <c>mask2polygon</c>.
    /// </summary>
    /// <param name="mask">The mask.</param>
    /// <param name="width">Mask width.</param>
    /// <param name="height">Mask height.</param>
    /// <param name="distanceLimit">Longest edge allowed before intermediate vertices are re-added.</param>
    /// <returns>The polygon, or <see langword="null"/> when the mask is empty.</returns>
    public static (float X, float Y)[]? MaskToPolygon(
        ReadOnlySpan<byte> mask,
        int width,
        int height,
        double distanceLimit)
    {
        List<PixelPoint[]> contours = Contours.FindExternal(mask, width, height);
        if (contours.Count == 0)
        {
            return null;
        }

        // Ties on area are broken by discovery order, which the two sides do not share; a tie
        // between two contours of exactly equal area is the one case that can pick differently.
        PixelPoint[] largest = contours[0];
        double largestArea = Contours.Area(largest);

        for (int i = 1; i < contours.Count; i++)
        {
            double area = Contours.Area(contours[i]);
            if (area > largestArea)
            {
                largestArea = area;
                largest = contours[i];
            }
        }

        double epsilon = EpsilonRatio * Contours.ArcLength(largest, closed: true);
        PixelPoint[] simplified = Contours.ApproxPolyDp(largest, epsilon, closed: true);

        return ExtractCustomVertices(simplified, distanceLimit);
    }

    /// <summary>
    /// Drops the vertices a region outline does not need, as <c>extract_custom_vertices</c>.
    /// </summary>
    /// <remarks>
    /// Convex vertices are always kept. A concave one survives only inside a run of at least two
    /// consecutive concave vertices and only if its interior angle is at least 120 degrees, which
    /// keeps a genuine notch — an L-shaped column, a figure cut out of a paragraph — while
    /// discarding the single-pixel wobble a mask boundary is full of. Long edges then get
    /// intermediate vertices back, and a convex spike at about 45 degrees is pushed outward along
    /// its bisector so the region encloses what the spike points at.
    /// </remarks>
    /// <param name="polygon">The simplified contour.</param>
    /// <param name="distanceLimit">Edge length above which intermediate vertices are re-added.</param>
    public static (float X, float Y)[] ExtractCustomVertices(
        ReadOnlySpan<PixelPoint> polygon,
        double distanceLimit)
    {
        int n = polygon.Length;
        if (n == 0)
        {
            return [];
        }

        distanceLimit *= MaxDistanceRatio;

        bool[] convex = new bool[n];
        double[] angles = new double[n];
        (double X, double Y)[] toPrevious = new (double X, double Y)[n];
        (double X, double Y)[] toNext = new (double X, double Y)[n];

        for (int i = 0; i < n; i++)
        {
            PixelPoint previous = polygon[((i - 1) % n + n) % n];
            PixelPoint current = polygon[i];
            PixelPoint next = polygon[(i + 1) % n];

            toPrevious[i] = (previous.X - current.X, previous.Y - current.Y);
            toNext[i] = (next.X - current.X, next.Y - current.Y);

            double cross = ((double)(current.X - previous.X) * (next.Y - current.Y))
                - ((double)(current.Y - previous.Y) * (next.X - current.X));
            convex[i] = cross < 0;
            angles[i] = AngleBetween(toPrevious[i], toNext[i]);
        }

        var preserved = new HashSet<int>();
        var concave = new List<int>();

        for (int i = 0; i < n; i++)
        {
            if (!convex[i])
            {
                concave.Add(i);
            }
        }

        if (concave.Count > 0)
        {
            var groups = new List<int>();
            var current = new List<int> { concave[0] };

            for (int i = 1; i < concave.Count; i++)
            {
                bool adjacent = concave[i] - concave[i - 1] == 1
                    || (concave[i - 1] == n - 1 && concave[i] == 0);

                if (adjacent)
                {
                    current.Add(concave[i]);
                }
                else
                {
                    if (current.Count >= 2)
                    {
                        groups.AddRange(current);
                    }

                    current = [concave[i]];
                }
            }

            if (current.Count >= 2)
            {
                groups.AddRange(current);
            }

            // A run that wraps the seam is only preserved when both ends of it survived grouping.
            if (concave.Count >= 2 && concave[0] == 0 && concave[^1] == n - 1)
            {
                if (groups.Contains(0) && groups.Contains(n - 1))
                {
                    preserved.UnionWith(groups);
                }
            }
            else
            {
                preserved.UnionWith(groups);
            }
        }

        var kept = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (convex[i] || (preserved.Contains(i) && angles[i] >= 120))
            {
                kept.Add(i);
            }
        }

        var final = new List<int>();

        for (int index = 0; index < kept.Count; index++)
        {
            int current = kept[index];
            int next = kept[(index + 1) % kept.Count];
            final.Add(current);

            double dx = polygon[current].X - polygon[next].X;
            double dy = polygon[current].Y - polygon[next].Y;
            double distance = Math.Sqrt((dx * dx) + (dy * dy));

            if (distance <= distanceLimit)
            {
                continue;
            }

            var intermediate = new List<int>();
            if (next > current)
            {
                for (int i = current + 1; i < next; i++)
                {
                    intermediate.Add(i);
                }
            }
            else
            {
                for (int i = current + 1; i < n; i++)
                {
                    intermediate.Add(i);
                }

                for (int i = 0; i < next; i++)
                {
                    intermediate.Add(i);
                }
            }

            if (intermediate.Count == 0)
            {
                continue;
            }

            int needed = (int)Math.Ceiling(distance / distanceLimit) - 1;
            if (intermediate.Count <= needed)
            {
                final.AddRange(intermediate);
            }
            else if (needed > 0)
            {
                double step = (double)intermediate.Count / needed;
                for (int i = 0; i < needed; i++)
                {
                    final.Add(intermediate[(int)(i * step)]);
                }
            }
        }

        var result = new List<(float X, float Y)>(final.Count);

        foreach (int i in final.Distinct().Order())
        {
            PixelPoint point = polygon[i];

            if (convex[i] && Math.Abs(angles[i] - SharpAngleThreshold) < 1)
            {
                double n1 = Length(toPrevious[i]);
                double n2 = Length(toNext[i]);
                double dx = (toPrevious[i].X / n1) + (toNext[i].X / n2);
                double dy = (toPrevious[i].Y / n1) + (toNext[i].Y / n2);
                double norm = Math.Sqrt((dx * dx) + (dy * dy));
                double distance = (n1 + n2) / 2.0;

                result.Add((
                    (float)(point.X + (dx / norm * distance)),
                    (float)(point.Y + (dy / norm * distance))));
            }
            else
            {
                result.Add((point.X, point.Y));
            }
        }

        return [.. result];
    }

    /// <summary>
    /// Reduces a polygon to the shape <paramref name="mode"/> asks for, as
    /// <c>_normalize_layout_polygon</c>.
    /// </summary>
    /// <remarks>
    /// <see cref="LayoutShapeMode.Auto"/> is the interesting one: it keeps the axis-aligned
    /// rectangle when the region is essentially rectangular anyway, takes the rotated quad when
    /// the polygon is close enough to one and no neighbour is already sitting inside this box,
    /// and otherwise keeps the polygon.
    /// </remarks>
    /// <param name="rectangle">The detection's axis-aligned rectangle.</param>
    /// <param name="polygon">The extracted polygon, if any.</param>
    /// <param name="mode">The shape to reduce to.</param>
    /// <param name="previous">The previous detection's normalised polygon, if any.</param>
    public static (float X, float Y)[] Normalize(
        (float X, float Y)[] rectangle,
        (float X, float Y)[]? polygon,
        LayoutShapeMode mode,
        (float X, float Y)[]? previous)
    {
        if (polygon is null || polygon.Length < 4 || mode == LayoutShapeMode.Rect)
        {
            return rectangle;
        }

        if (mode == LayoutShapeMode.Poly)
        {
            return polygon;
        }

        (float X, float Y)[]? quad = Polygons.MinAreaQuad(polygon);

        if (mode == LayoutShapeMode.Quad)
        {
            return quad ?? rectangle;
        }

        if (quad is not null)
        {
            if (Polygons.OverlapRatio(rectangle, quad, Polygons.OverlapMode.Union) >= 0.95)
            {
                return rectangle;
            }

            double polygonToQuad = Polygons.OverlapRatio(polygon, quad, Polygons.OverlapMode.Union);
            double toPrevious = previous is null
                ? 0
                : Polygons.OverlapRatio(previous, rectangle, Polygons.OverlapMode.Small);

            if (polygonToQuad >= 0.8 && toPrevious < 0.01)
            {
                return quad;
            }
        }

        return polygon;
    }

    /// <summary>The four corners of an axis-aligned box, as <c>_rect_from_box</c>.</summary>
    public static (float X, float Y)[] Rectangle(int left, int top, int right, int bottom) =>
        [(left, top), (right, top), (right, bottom), (left, bottom)];

    /// <summary>Nearest-neighbour resize, matching <c>cv2.resize(..., INTER_NEAREST)</c>.</summary>
    private static byte[] ResizeNearest(byte[] source, int width, int height, int newWidth, int newHeight)
    {
        byte[] result = new byte[newWidth * newHeight];
        double scaleX = (double)width / newWidth;
        double scaleY = (double)height / newHeight;

        for (int y = 0; y < newHeight; y++)
        {
            int sourceY = Math.Min((int)(y * scaleY), height - 1);
            for (int x = 0; x < newWidth; x++)
            {
                int sourceX = Math.Min((int)(x * scaleX), width - 1);
                result[(y * newWidth) + x] = source[(sourceY * width) + sourceX];
            }
        }

        return result;
    }

    private static double Length((double X, double Y) vector) =>
        Math.Sqrt((vector.X * vector.X) + (vector.Y * vector.Y));

    /// <summary>Angle between two vectors, in degrees.</summary>
    private static double AngleBetween((double X, double Y) first, (double X, double Y) second)
    {
        double n1 = Length(first);
        double n2 = Length(second);
        if (n1 == 0 || n2 == 0)
        {
            return 0;
        }

        double dot = Math.Clamp((((first.X / n1) * (second.X / n2)) + ((first.Y / n1) * (second.Y / n2))), -1.0, 1.0);
        return Math.Acos(dot) * 180.0 / Math.PI;
    }

    /// <summary>Python's <c>round</c>, then truncation, as the upstream expression does.</summary>
    private static int RoundHalfToEven(double value) => (int)Math.Round(value, MidpointRounding.ToEven);
}
