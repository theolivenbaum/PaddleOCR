using PaddleOcrSharp.Formats;
using PaddleOcrSharp.Models.Layout;
using PaddleOcrSharp.Tests.Fixtures;

namespace PaddleOcrSharp.Tests.Models;

/// <summary>
/// Region polygons from the layout detector's mask head, against
/// <c>extract_polygon_points_by_masks</c> itself. Fixtures come from
/// <c>dotnet/tools/reference/dump_layout_polygons.py</c>.
/// </summary>
public class LayoutPolygonParityTests
{
    private const string FixtureName = "layout_polygons.npz";

    [Theory]
    [InlineData(LayoutShapeMode.Rect, "rect")]
    [InlineData(LayoutShapeMode.Quad, "quad")]
    [InlineData(LayoutShapeMode.Poly, "poly")]
    [InlineData(LayoutShapeMode.Auto, "auto")]
    public void PolygonsMatchUpstream(LayoutShapeMode mode, string name)
    {
        Fixture.RequireOrSkip(FixtureName);
        var fixtures = Fixture.Load(FixtureName);
        int cases = (int)fixtures["count"].ToInt64()[0];

        var failures = new List<string>();

        for (int index = 0; index < cases; index++)
        {
            NpyArray boxArray = fixtures[$"boxes_{index}"];
            float[] boxValues = boxArray.ToFloats();
            int boxCount = boxArray.Shape[0];

            var boxes = new List<LayoutBox>(boxCount);
            for (int b = 0; b < boxCount; b++)
            {
                boxes.Add(new LayoutBox(
                    (int)boxValues[b * 6],
                    "region",
                    boxValues[(b * 6) + 1],
                    boxValues[(b * 6) + 2],
                    boxValues[(b * 6) + 3],
                    boxValues[(b * 6) + 4],
                    boxValues[(b * 6) + 5],
                    b));
            }

            NpyArray maskArray = fixtures[$"masks_{index}"];
            int maskHeight = maskArray.Shape[1];
            int maskWidth = maskArray.Shape[2];
            long[] maskValues = maskArray.ToInt64();
            byte[] masks = new byte[maskValues.Length];
            for (int i = 0; i < masks.Length; i++)
            {
                masks[i] = (byte)(maskValues[i] != 0 ? 1 : 0);
            }

            long[] page = fixtures[$"page_{index}"].ToInt64();

            (float X, float Y)[][] actual = LayoutPolygons.Extract(
                boxes, masks, maskWidth, maskHeight, 800.0 / page[0], 800.0 / page[1], mode);

            long[] lengths = fixtures[$"poly_{index}_{name}_lengths"].ToInt64();
            double[] flat = fixtures[$"poly_{index}_{name}"].ToDoubles();

            int offset = 0;
            for (int b = 0; b < boxCount; b++)
            {
                int expectedLength = (int)lengths[b];

                if (actual[b].Length != expectedLength)
                {
                    failures.Add(
                        $"case {index} box {b}: {actual[b].Length} points, expected {expectedLength}");
                    offset += expectedLength;
                    continue;
                }

                for (int p = 0; p < expectedLength; p++)
                {
                    double expectedX = flat[(offset + p) * 2];
                    double expectedY = flat[((offset + p) * 2) + 1];

                    if (Math.Abs(actual[b][p].X - expectedX) > 1e-3
                        || Math.Abs(actual[b][p].Y - expectedY) > 1e-3)
                    {
                        failures.Add(
                            $"case {index} box {b} point {p}: ({actual[b][p].X}, {actual[b][p].Y}) " +
                            $"!= ({expectedX}, {expectedY})");
                    }
                }

                offset += expectedLength;
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures.Take(10)));
    }
}
