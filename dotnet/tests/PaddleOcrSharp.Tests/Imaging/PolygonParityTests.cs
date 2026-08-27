using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Tests.Fixtures;

namespace PaddleOcrSharp.Tests.Imaging;

/// <summary>
/// Polygon areas, intersections and minimum-area quads, against Shapely and OpenCV. Fixtures
/// come from <c>dotnet/tools/reference/dump_polygons.py</c>.
/// </summary>
public class PolygonParityTests
{
    private const string FixtureName = "polygons.npz";

    [Fact]
    public void AreasMatchShapely()
    {
        Fixture.RequireOrSkip(FixtureName);
        var fixtures = Fixture.Load(FixtureName);

        for (int index = 0; index < Count(fixtures); index++)
        {
            (float X, float Y)[] polygon = Polygon(fixtures, index);
            double expected = fixtures[$"area_{index}"].ToDoubles()[0];
            Assert.Equal(expected, Polygons.Area(polygon), 1e-9);
        }
    }

    [Fact]
    public void IntersectionAreasMatchShapely()
    {
        Fixture.RequireOrSkip(FixtureName);
        var fixtures = Fixture.Load(FixtureName);
        int count = Count(fixtures);
        double[] expected = fixtures["intersections"].ToDoubles();

        var failures = new List<string>();

        for (int i = 0; i < count; i++)
        {
            (float X, float Y)[] first = Polygon(fixtures, i);
            for (int j = 0; j < count; j++)
            {
                (float X, float Y)[] second = Polygon(fixtures, j);
                double actual = Polygons.IntersectionArea(first, second);
                double reference = expected[(i * count) + j];

                if (Math.Abs(actual - reference) > 1e-6)
                {
                    failures.Add($"({i},{j}): {actual} != {reference}");
                }
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures.Take(10)));
    }

    [Fact]
    public void OverlapRatiosMatchShapely()
    {
        Fixture.RequireOrSkip(FixtureName);
        var fixtures = Fixture.Load(FixtureName);
        int count = Count(fixtures);
        double[] expected = fixtures["ratios"].ToDoubles();

        Polygons.OverlapMode[] modes =
            [Polygons.OverlapMode.Union, Polygons.OverlapMode.Small, Polygons.OverlapMode.Large];

        var failures = new List<string>();

        for (int i = 0; i < count; i++)
        {
            (float X, float Y)[] first = Polygon(fixtures, i);
            for (int j = 0; j < count; j++)
            {
                (float X, float Y)[] second = Polygon(fixtures, j);
                for (int m = 0; m < modes.Length; m++)
                {
                    double actual = Polygons.OverlapRatio(first, second, modes[m]);
                    double reference = expected[(((i * count) + j) * 3) + m];

                    if (Math.Abs(actual - reference) > 1e-9)
                    {
                        failures.Add($"({i},{j},{modes[m]}): {actual} != {reference}");
                    }
                }
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures.Take(10)));
    }

    [Fact]
    public void MinimumAreaQuadsMatchOpenCv()
    {
        Fixture.RequireOrSkip(FixtureName);
        var fixtures = Fixture.Load(FixtureName);

        var failures = new List<string>();

        for (int index = 0; index < Count(fixtures); index++)
        {
            (float X, float Y)[] polygon = Polygon(fixtures, index);
            (float X, float Y)[]? actual = Polygons.MinAreaQuad(polygon);
            double[] expected = fixtures[$"quad_{index}"].ToDoubles();

            Assert.NotNull(actual);
            Assert.Equal(4, actual.Length);

            for (int corner = 0; corner < 4; corner++)
            {
                // The rectangle is found by a different search, so corners agree geometrically
                // rather than bit for bit.
                if (Math.Abs(actual[corner].X - expected[corner * 2]) > 1e-3
                    || Math.Abs(actual[corner].Y - expected[(corner * 2) + 1]) > 1e-3)
                {
                    failures.Add(
                        $"polygon {index} corner {corner}: ({actual[corner].X}, {actual[corner].Y}) " +
                        $"!= ({expected[corner * 2]}, {expected[(corner * 2) + 1]})");
                }
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures.Take(10)));
    }

    [Fact]
    public void FilledPolygonsMatchOpenCv()
    {
        Fixture.RequireOrSkip(FixtureName);
        var fixtures = Fixture.Load(FixtureName);
        long[] size = fixtures["fill_size"].ToInt64();
        int width = (int)size[0];
        int height = (int)size[1];

        var failures = new List<string>();

        for (int index = 0; index < Count(fixtures); index++)
        {
            long[] flat = fixtures[$"fillpts_{index}"].ToInt64();
            var points = new (int X, int Y)[flat.Length / 2];
            for (int i = 0; i < points.Length; i++)
            {
                points[i] = ((int)flat[i * 2], (int)flat[(i * 2) + 1]);
            }

            bool[] actual = Polygons.Fill(points, width, height);
            byte[] expected = fixtures[$"fill_{index}"].ToBytes();

            int differing = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                if (actual[i] != (expected[i] != 0))
                {
                    differing++;
                }
            }

            if (differing > 0)
            {
                failures.Add($"polygon {index}: {differing} of {expected.Length} pixels differ");
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    private static int Count(Dictionary<string, PaddleOcrSharp.Formats.NpyArray> fixtures) =>
        (int)fixtures["count"].ToInt64()[0];

    private static (float X, float Y)[] Polygon(
        Dictionary<string, PaddleOcrSharp.Formats.NpyArray> fixtures, int index)
    {
        double[] flat = fixtures[$"poly_{index}"].ToDoubles();
        var points = new (float X, float Y)[flat.Length / 2];
        for (int i = 0; i < points.Length; i++)
        {
            points[i] = ((float)flat[i * 2], (float)flat[(i * 2) + 1]);
        }

        return points;
    }
}
