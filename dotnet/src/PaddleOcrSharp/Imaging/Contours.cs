namespace PaddleOcrSharp.Imaging;

/// <summary>An integer pixel coordinate, matching OpenCV's <c>Point</c> ordering.</summary>
/// <param name="X">Column.</param>
/// <param name="Y">Row.</param>
public readonly record struct PixelPoint(int X, int Y);

/// <summary>
/// Contour extraction and simplification, reproducing the OpenCV calls the layout model's mask
/// head goes through: <c>findContours</c>, <c>contourArea</c>, <c>arcLength</c> and
/// <c>approxPolyDP</c>.
/// </summary>
/// <remarks>
/// <para>
/// Only <c>RETR_EXTERNAL</c> with <c>CHAIN_APPROX_SIMPLE</c> is implemented, which is what
/// <c>mask2polygon</c> asks for. An external border is the outer boundary of an 8-connected
/// foreground component, so the components are found first and each one's border is then traced
/// with OpenCV's neighbour ordering; holes never produce an external border and are ignored,
/// which is what lets the two-pass structure stand in for Suzuki-Abe's single pass.
/// </para>
/// <para>
/// Contours are traced in raster order of their first pixel. OpenCV's own ordering differs, but
/// the one caller takes the largest contour by area, so only the set and each contour's point
/// sequence are observable.
/// </para>
/// </remarks>
public static class Contours
{
    // OpenCV's 3x3 neighbour order, starting at "right" and turning counter-clockwise on screen
    // (the image's y axis points down). icvCodeDeltas in the OpenCV source.
    private static ReadOnlySpan<sbyte> DeltaX => [1, 1, 0, -1, -1, -1, 0, 1];

    private static ReadOnlySpan<sbyte> DeltaY => [0, -1, -1, -1, 0, 1, 1, 1];

    /// <summary>
    /// Finds the outer border of every 8-connected foreground component.
    /// </summary>
    /// <param name="mask">Binary image; any non-zero byte is foreground.</param>
    /// <param name="width">Image width.</param>
    /// <param name="height">Image height.</param>
    /// <returns>One contour per component, as <c>CHAIN_APPROX_SIMPLE</c> point lists.</returns>
    public static List<PixelPoint[]> FindExternal(ReadOnlySpan<byte> mask, int width, int height)
    {
        var contours = new List<PixelPoint[]>();
        if (width <= 0 || height <= 0)
        {
            return contours;
        }

        // Component id per pixel, so a component's border is traced exactly once. 0 = background,
        // otherwise the 1-based index of the component.
        int[] labels = new int[width * height];
        int[] queue = new int[width * height];
        int components = 0;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int index = (y * width) + x;
                if (mask[index] == 0 || labels[index] != 0)
                {
                    continue;
                }

                components++;
                Flood(mask, labels, queue, width, height, index, components);
                contours.Add(Trace(labels, width, height, x, y, components));
            }
        }

        return contours;
    }

    /// <summary>Absolute area enclosed by a contour, matching <c>cv2.contourArea</c>.</summary>
    public static double Area(ReadOnlySpan<PixelPoint> contour)
    {
        if (contour.Length < 3)
        {
            return 0;
        }

        double sum = 0;
        PixelPoint previous = contour[^1];

        foreach (PixelPoint point in contour)
        {
            sum += ((double)previous.X * point.Y) - ((double)point.X * previous.Y);
            previous = point;
        }

        return Math.Abs(sum) / 2.0;
    }

    /// <summary>Perimeter of a contour, matching <c>cv2.arcLength</c>.</summary>
    /// <remarks>
    /// Each segment is measured in float32 and the total accumulated in double, which is what
    /// OpenCV does and is not interchangeable with measuring in double throughout: the two differ
    /// around the seventh digit, and the caller turns this number into
    /// <c>approxPolyDP</c>'s epsilon, where that is enough to keep or drop a vertex.
    /// </remarks>
    /// <param name="contour">The contour.</param>
    /// <param name="closed">Whether the last point joins back to the first.</param>
    public static double ArcLength(ReadOnlySpan<PixelPoint> contour, bool closed)
    {
        if (contour.Length < 2)
        {
            return 0;
        }

        double length = 0;
        PixelPoint previous = closed ? contour[^1] : contour[0];

        for (int i = closed ? 0 : 1; i < contour.Length; i++)
        {
            float dx = contour[i].X - previous.X;
            float dy = contour[i].Y - previous.Y;
            length += MathF.Sqrt((dx * dx) + (dy * dy));
            previous = contour[i];
        }

        return length;
    }

    /// <summary>
    /// Douglas-Peucker simplification, matching <c>cv2.approxPolyDP</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Two details separate this from the textbook algorithm, and both change which vertices
    /// survive. The distance being maximised is to the <em>segment</em>, not to the infinite line
    /// through it: a point that projects past either end is measured to that endpoint instead.
    /// And a closed curve has no endpoints to anchor on, so it is split at the point farthest
    /// from an arbitrary start and then at the point farthest from that, settled over three
    /// passes, and the two arcs are simplified independently.
    /// </para>
    /// <para>
    /// A final pass then drops any vertex that sits almost on the line between its neighbours,
    /// provided the corner does not double back. Skipping it leaves vertices the recursion had
    /// to keep as arc boundaries but that the finished outline does not need.
    /// </para>
    /// </remarks>
    /// <param name="contour">The contour to simplify.</param>
    /// <param name="epsilon">Maximum distance from the simplified curve to the original.</param>
    /// <param name="closed">Whether the curve is closed.</param>
    public static PixelPoint[] ApproxPolyDp(ReadOnlySpan<PixelPoint> contour, double epsilon, bool closed)
    {
        int count = contour.Length;
        if (count == 0)
        {
            return [];
        }

        double eps = epsilon * epsilon;
        var simplified = new List<PixelPoint>(count);
        var stack = new Stack<(int Start, int End)>();

        bool isClosed = closed;
        int iterations = 3;
        int position = 0;
        int splitStart = 0;
        bool withinEpsilon = false;
        PixelPoint startPoint = default;
        PixelPoint endPoint;
        PixelPoint point;

        if (!closed)
        {
            if (contour[count - 1] != contour[0])
            {
                stack.Push((0, count - 1));
            }
            else
            {
                isClosed = true;
                iterations = 1;
            }
        }

        if (isClosed)
        {
            for (int iteration = 0; iteration < iterations; iteration++)
            {
                double maximum = 0;
                position = (position + splitStart) % count;
                startPoint = Read(contour, ref position, count);

                for (int j = 1; j < count; j++)
                {
                    point = Read(contour, ref position, count);
                    double dx = point.X - startPoint.X;
                    double dy = point.Y - startPoint.Y;
                    double distance = (dx * dx) + (dy * dy);

                    if (distance > maximum)
                    {
                        maximum = distance;
                        splitStart = j;
                    }
                }

                withinEpsilon = maximum <= eps;
            }

            if (withinEpsilon)
            {
                simplified.Add(startPoint);
            }
            else
            {
                int sliceStart = position % count;
                int rightEnd = sliceStart;
                splitStart = (splitStart + sliceStart) % count;

                stack.Push((splitStart, rightEnd));
                stack.Push((sliceStart, splitStart));
            }
        }

        while (stack.Count > 0)
        {
            (int start, int end) = stack.Pop();
            endPoint = contour[end];
            position = start;
            startPoint = Read(contour, ref position, count);

            if (position != end)
            {
                double dx = endPoint.X - startPoint.X;
                double dy = endPoint.Y - startPoint.Y;
                double segment = (dx * dx) + (dy * dy);
                double maximum = 0;

                while (position != end)
                {
                    point = Read(contour, ref position, count);
                    double offsetX = point.X - startPoint.X;
                    double offsetY = point.Y - startPoint.Y;
                    double projection = (offsetX * dx) + (offsetY * dy);
                    double scaled;

                    if (projection < 0)
                    {
                        scaled = ((offsetX * offsetX) + (offsetY * offsetY)) * segment;
                    }
                    else if (projection > segment)
                    {
                        double toEndX = point.X - endPoint.X;
                        double toEndY = point.Y - endPoint.Y;
                        scaled = ((toEndX * toEndX) + (toEndY * toEndY)) * segment;
                    }
                    else
                    {
                        double cross = (offsetY * dx) - (offsetX * dy);
                        scaled = cross * cross;
                    }

                    if (scaled > maximum)
                    {
                        maximum = scaled;
                        splitStart = (position + count - 1) % count;
                    }
                }

                withinEpsilon = maximum <= eps * segment;
            }
            else
            {
                withinEpsilon = true;
                startPoint = contour[start];
            }

            if (withinEpsilon)
            {
                simplified.Add(startPoint);
            }
            else
            {
                stack.Push((splitStart, end));
                stack.Push((start, splitStart));
            }
        }

        if (!closed)
        {
            simplified.Add(contour[count - 1]);
        }

        return Straighten([.. simplified], eps, closed);
    }

    /// <summary>Drops vertices that lie almost on the line between their neighbours.</summary>
    private static PixelPoint[] Straighten(PixelPoint[] points, double eps, bool closed)
    {
        int count = points.Length;
        if (count == 0)
        {
            return points;
        }

        int remaining = count;
        int readPosition = closed ? count - 1 : 0;
        PixelPoint start = Read(points, ref readPosition, count);
        int writePosition = readPosition;
        PixelPoint middle = Read(points, ref readPosition, count);

        int edge = closed ? 0 : 1;

        for (int i = edge; i < count - edge && remaining > 2; i++)
        {
            PixelPoint end = Read(points, ref readPosition, count);

            double dx = end.X - start.X;
            double dy = end.Y - start.Y;
            double distance = Math.Abs(((middle.X - start.X) * dy) - ((middle.Y - start.Y) * dx));
            double inner = ((double)(middle.X - start.X) * (end.X - middle.X))
                + ((double)(middle.Y - start.Y) * (end.Y - middle.Y));

            if (distance * distance <= 0.5 * eps * ((dx * dx) + (dy * dy))
                && dx != 0
                && dy != 0
                && inner >= 0)
            {
                remaining--;
                points[writePosition] = start = end;
                if (++writePosition >= count)
                {
                    writePosition = 0;
                }

                middle = Read(points, ref readPosition, count);
                i++;
                continue;
            }

            points[writePosition] = start = middle;
            if (++writePosition >= count)
            {
                writePosition = 0;
            }

            middle = end;
        }

        if (!closed)
        {
            points[writePosition] = middle;
        }

        return points[..remaining];
    }

    /// <summary>Reads the point at <paramref name="position"/> and advances it, wrapping.</summary>
    private static PixelPoint Read(ReadOnlySpan<PixelPoint> points, ref int position, int count)
    {
        PixelPoint point = points[position];
        if (++position >= count)
        {
            position = 0;
        }

        return point;
    }

    /// <summary>Labels the 8-connected component containing <paramref name="seed"/>.</summary>
    private static void Flood(
        ReadOnlySpan<byte> mask,
        int[] labels,
        int[] queue,
        int width,
        int height,
        int seed,
        int label)
    {
        int head = 0;
        int tail = 0;
        queue[tail++] = seed;
        labels[seed] = label;

        while (head < tail)
        {
            int index = queue[head++];
            int x = index % width;
            int y = index / width;

            for (int direction = 0; direction < 8; direction++)
            {
                int nx = x + DeltaX[direction];
                int ny = y + DeltaY[direction];

                if (nx < 0 || ny < 0 || nx >= width || ny >= height)
                {
                    continue;
                }

                int neighbour = (ny * width) + nx;
                if (mask[neighbour] == 0 || labels[neighbour] != 0)
                {
                    continue;
                }

                labels[neighbour] = label;
                queue[tail++] = neighbour;
            }
        }
    }

    /// <summary>
    /// Traces one component's outer border from its topmost-leftmost pixel.
    /// </summary>
    /// <remarks>
    /// This is OpenCV's <c>icvFetchContour</c>: from each border pixel, scan the neighbourhood
    /// anticlockwise starting one step past where the previous step arrived from, and take the
    /// first foreground neighbour. <c>CHAIN_APPROX_SIMPLE</c> records a point only when the
    /// direction changes, which is what collapses straight and diagonal runs to their endpoints.
    /// </remarks>
    private static PixelPoint[] Trace(int[] labels, int width, int height, int startX, int startY, int label)
    {
        bool Foreground(int x, int y) =>
            x >= 0 && y >= 0 && x < width && y < height && labels[(y * width) + x] == label;

        // The scan reaches the component from the left, so the search starts there.
        int direction = 4;
        int first = -1;

        for (int step = 0; step < 8; step++)
        {
            direction = (direction - 1) & 7;
            if (Foreground(startX + DeltaX[direction], startY + DeltaY[direction]))
            {
                first = direction;
                break;
            }
        }

        if (first < 0)
        {
            return [new PixelPoint(startX, startY)];
        }

        var points = new List<PixelPoint>();
        int currentX = startX;
        int currentY = startY;
        int firstX = startX + DeltaX[first];
        int firstY = startY + DeltaY[first];
        int previous = first ^ 4;
        int search = first;

        while (true)
        {
            int next;
            int nextX;
            int nextY;

            do
            {
                search = (search + 1) & 7;
                nextX = currentX + DeltaX[search];
                nextY = currentY + DeltaY[search];
            }
            while (!Foreground(nextX, nextY));

            next = search;

            if (next != previous)
            {
                points.Add(new PixelPoint(currentX, currentY));
                previous = next;
            }

            if (nextX == startX && nextY == startY && currentX == firstX && currentY == firstY)
            {
                break;
            }

            currentX = nextX;
            currentY = nextY;
            search = (next + 4) & 7;
        }

        return points.Count == 0 ? [new PixelPoint(startX, startY)] : [.. points];
    }
}
