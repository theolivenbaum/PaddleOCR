namespace PaddleOcrSharp.Imaging;

/// <summary>
/// Planar geometry for the layout model's polygon regions: minimum-area rectangles, areas, and
/// the overlap ratios <c>calculate_polygon_overlap_ratio</c> reports.
/// </summary>
public static class Polygons
{
    /// <summary>How an overlap is expressed as a ratio.</summary>
    public enum OverlapMode
    {
        /// <summary>Intersection over union.</summary>
        Union,

        /// <summary>Intersection over the smaller polygon's area.</summary>
        Small,

        /// <summary>Intersection over the larger polygon's area.</summary>
        Large,
    }

    /// <summary>Signed area of a polygon; positive when the vertices wind counter-clockwise.</summary>
    public static double SignedArea(ReadOnlySpan<(float X, float Y)> polygon)
    {
        if (polygon.Length < 3)
        {
            return 0;
        }

        double sum = 0;
        (float X, float Y) previous = polygon[^1];

        foreach ((float X, float Y) point in polygon)
        {
            sum += ((double)previous.X * point.Y) - ((double)point.X * previous.Y);
            previous = point;
        }

        return sum / 2.0;
    }

    /// <summary>Absolute area of a polygon.</summary>
    public static double Area(ReadOnlySpan<(float X, float Y)> polygon) => Math.Abs(SignedArea(polygon));

    /// <summary>
    /// Area shared by two polygons.
    /// </summary>
    /// <remarks>
    /// Both are decomposed into signed triangles fanned from the origin and the triangles are
    /// intersected pairwise, which needs no clipping of one concave outline against another: a
    /// triangle is convex, so Sutherland-Hodgman applies to every pair, and the signs make the
    /// fan's cancellation come out right. Exact for simple polygons, which is what a traced
    /// contour is.
    /// </remarks>
    public static double IntersectionArea(
        ReadOnlySpan<(float X, float Y)> first,
        ReadOnlySpan<(float X, float Y)> second)
    {
        if (first.Length < 3 || second.Length < 3)
        {
            return 0;
        }

        double total = 0;
        Span<(double X, double Y)> clipped = stackalloc (double X, double Y)[8];
        Span<(double X, double Y)> scratch = stackalloc (double X, double Y)[8];

        for (int i = 0; i < first.Length; i++)
        {
            (double X, double Y) a1 = (first[i].X, first[i].Y);
            (double X, double Y) a2 = (first[(i + 1) % first.Length].X, first[(i + 1) % first.Length].Y);

            for (int j = 0; j < second.Length; j++)
            {
                (double X, double Y) b1 = (second[j].X, second[j].Y);
                (double X, double Y) b2 = (second[(j + 1) % second.Length].X, second[(j + 1) % second.Length].Y);

                total += TriangleOverlap(a1, a2, b1, b2, clipped, scratch);
            }
        }

        return Math.Abs(total);
    }

    /// <summary>The overlap ratio between two polygons, as <c>calculate_polygon_overlap_ratio</c>.</summary>
    public static double OverlapRatio(
        ReadOnlySpan<(float X, float Y)> first,
        ReadOnlySpan<(float X, float Y)> second,
        OverlapMode mode)
    {
        double firstArea = Area(first);
        double secondArea = Area(second);
        double intersection = IntersectionArea(first, second);

        double divisor = mode switch
        {
            OverlapMode.Union => firstArea + secondArea - intersection,
            OverlapMode.Small => Math.Min(firstArea, secondArea),
            _ => Math.Max(firstArea, secondArea),
        };

        return divisor > 0 ? intersection / divisor : 0;
    }

    /// <summary>
    /// The four corners of the minimum-area enclosing rectangle, matching
    /// <c>cv2.boxPoints(cv2.minAreaRect(...))</c> followed by the caller's reordering.
    /// </summary>
    /// <remarks>
    /// Rotating calipers over the convex hull: the minimum-area rectangle always has a side
    /// flush with a hull edge, so trying each edge in turn is exhaustive. Upstream then sorts the
    /// corners by angle about the centre and rolls the one nearest the origin to the front.
    /// </remarks>
    /// <param name="polygon">The points to enclose.</param>
    /// <returns>Four corners, or <see langword="null"/> when there are fewer than three points.</returns>
    public static (float X, float Y)[]? MinAreaQuad(ReadOnlySpan<(float X, float Y)> polygon)
    {
        if (polygon.Length < 3)
        {
            return null;
        }

        (double X, double Y)[] hull = ConvexHull(polygon);
        if (hull.Length == 0)
        {
            return null;
        }

        if (hull.Length < 3)
        {
            // A degenerate hull (all points collinear) still has a rectangle: the segment itself.
            (double X, double Y) p = hull[0];
            (double X, double Y) q = hull[^1];
            return Order([
                ((float)p.X, (float)p.Y), ((float)q.X, (float)q.Y),
                ((float)q.X, (float)q.Y), ((float)p.X, (float)p.Y)]);
        }

        double bestArea = double.PositiveInfinity;
        (double X, double Y)[] best = new (double X, double Y)[4];

        for (int i = 0; i < hull.Length; i++)
        {
            (double X, double Y) a = hull[i];
            (double X, double Y) b = hull[(i + 1) % hull.Length];

            double edgeX = b.X - a.X;
            double edgeY = b.Y - a.Y;
            double length = Math.Sqrt((edgeX * edgeX) + (edgeY * edgeY));
            if (length == 0)
            {
                continue;
            }

            double ux = edgeX / length;
            double uy = edgeY / length;

            double minU = double.PositiveInfinity;
            double maxU = double.NegativeInfinity;
            double minV = double.PositiveInfinity;
            double maxV = double.NegativeInfinity;

            foreach ((double X, double Y) point in hull)
            {
                double u = (point.X * ux) + (point.Y * uy);
                double v = (-point.X * uy) + (point.Y * ux);
                minU = Math.Min(minU, u);
                maxU = Math.Max(maxU, u);
                minV = Math.Min(minV, v);
                maxV = Math.Max(maxV, v);
            }

            double area = (maxU - minU) * (maxV - minV);
            if (area >= bestArea)
            {
                continue;
            }

            bestArea = area;
            best[0] = Rotate(minU, minV, ux, uy);
            best[1] = Rotate(maxU, minV, ux, uy);
            best[2] = Rotate(maxU, maxV, ux, uy);
            best[3] = Rotate(minU, maxV, ux, uy);
        }

        if (double.IsPositiveInfinity(bestArea))
        {
            return null;
        }

        return Order([.. best.Select(p => ((float)p.X, (float)p.Y))]);
    }

    /// <summary>Sorts corners by angle about the centre, starting at the one nearest the origin.</summary>
    private static (float X, float Y)[] Order((float X, float Y)[] quad)
    {
        float centreX = 0;
        float centreY = 0;
        foreach ((float X, float Y) point in quad)
        {
            centreX += point.X;
            centreY += point.Y;
        }

        centreX /= quad.Length;
        centreY /= quad.Length;

        (float X, float Y)[] sorted = [.. quad.OrderBy(p => Math.Atan2(p.Y - centreY, p.X - centreX))];

        int topLeft = 0;
        for (int i = 1; i < sorted.Length; i++)
        {
            if (sorted[i].X + sorted[i].Y < sorted[topLeft].X + sorted[topLeft].Y)
            {
                topLeft = i;
            }
        }

        return [.. Enumerable.Range(0, sorted.Length).Select(i => sorted[(i + topLeft) % sorted.Length])];
    }

    private static (double X, double Y) Rotate(double u, double v, double ux, double uy) =>
        ((u * ux) - (v * uy), (u * uy) + (v * ux));

    /// <summary>Andrew's monotone chain, returning the hull counter-clockwise.</summary>
    private static (double X, double Y)[] ConvexHull(ReadOnlySpan<(float X, float Y)> polygon)
    {
        var points = new List<(double X, double Y)>(polygon.Length);
        foreach ((float X, float Y) point in polygon)
        {
            points.Add((point.X, point.Y));
        }

        points.Sort((a, b) => a.X != b.X ? a.X.CompareTo(b.X) : a.Y.CompareTo(b.Y));

        // Deduplicate, so collinear runs do not confuse the cross-product test.
        var unique = new List<(double X, double Y)>(points.Count);
        foreach ((double X, double Y) point in points)
        {
            if (unique.Count == 0 || unique[^1] != point)
            {
                unique.Add(point);
            }
        }

        if (unique.Count < 3)
        {
            return [.. unique];
        }

        // Andrew's monotone chain: the lower hull left to right, then the upper hull right to
        // left, each dropping any vertex that a new point makes non-convex.
        var hull = new List<(double X, double Y)>(unique.Count + 1);

        foreach ((double X, double Y) point in unique)
        {
            while (hull.Count >= 2 && Cross(hull[^2], hull[^1], point) <= 0)
            {
                hull.RemoveAt(hull.Count - 1);
            }

            hull.Add(point);
        }

        int lower = hull.Count + 1;

        for (int i = unique.Count - 2; i >= 0; i--)
        {
            (double X, double Y) point = unique[i];
            while (hull.Count >= lower && Cross(hull[^2], hull[^1], point) <= 0)
            {
                hull.RemoveAt(hull.Count - 1);
            }

            hull.Add(point);
        }

        // The last point closes the loop onto the first.
        hull.RemoveAt(hull.Count - 1);
        return [.. hull];
    }

    private static double Cross((double X, double Y) o, (double X, double Y) a, (double X, double Y) b) =>
        ((a.X - o.X) * (b.Y - o.Y)) - ((a.Y - o.Y) * (b.X - o.X));

    /// <summary>Signed area shared by two origin-anchored triangles.</summary>
    private static double TriangleOverlap(
        (double X, double Y) a1,
        (double X, double Y) a2,
        (double X, double Y) b1,
        (double X, double Y) b2,
        Span<(double X, double Y)> clipped,
        Span<(double X, double Y)> scratch)
    {
        double signA = Math.Sign(Cross((0, 0), a1, a2));
        double signB = Math.Sign(Cross((0, 0), b1, b2));

        if (signA == 0 || signB == 0)
        {
            return 0;
        }

        // Orient both triangles counter-clockwise so the clipper's half-plane test is uniform.
        (double X, double Y) p1 = signA > 0 ? a1 : a2;
        (double X, double Y) p2 = signA > 0 ? a2 : a1;
        (double X, double Y) q1 = signB > 0 ? b1 : b2;
        (double X, double Y) q2 = signB > 0 ? b2 : b1;

        clipped[0] = (0, 0);
        clipped[1] = p1;
        clipped[2] = p2;
        int count = 3;

        count = Clip(clipped, count, (0, 0), q1, scratch);
        count = Clip(clipped, count, q1, q2, scratch);
        count = Clip(clipped, count, q2, (0, 0), scratch);

        if (count < 3)
        {
            return 0;
        }

        double area = 0;
        for (int i = 1; i + 1 < count; i++)
        {
            area += Cross(clipped[0], clipped[i], clipped[i + 1]) / 2.0;
        }

        return area * signA * signB;
    }

    /// <summary>Clips a convex polygon to the left of the directed line <c>a → b</c>.</summary>
    private static int Clip(
        Span<(double X, double Y)> polygon,
        int count,
        (double X, double Y) a,
        (double X, double Y) b,
        Span<(double X, double Y)> scratch)
    {
        int written = 0;

        for (int i = 0; i < count; i++)
        {
            (double X, double Y) current = polygon[i];
            (double X, double Y) next = polygon[(i + 1) % count];

            double side = Cross(a, b, current);
            double nextSide = Cross(a, b, next);

            if (side >= 0)
            {
                scratch[written++] = current;
            }

            if ((side > 0 && nextSide < 0) || (side < 0 && nextSide > 0))
            {
                double t = side / (side - nextSide);
                scratch[written++] = (
                    current.X + ((next.X - current.X) * t),
                    current.Y + ((next.Y - current.Y) * t));
            }
        }

        scratch[..written].CopyTo(polygon);
        return written;
    }
}
