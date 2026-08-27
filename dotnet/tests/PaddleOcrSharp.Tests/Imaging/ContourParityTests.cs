using PaddleOcrSharp.Formats;
using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Tests.Fixtures;

namespace PaddleOcrSharp.Tests.Imaging;

/// <summary>
/// The four OpenCV calls behind <c>mask2polygon</c>, against <c>cv2</c> itself. Fixtures come
/// from <c>dotnet/tools/reference/dump_contours.py</c>.
/// </summary>
public class ContourParityTests
{
    private const string FixtureName = "contours.npz";

    [Fact]
    public void ContoursMatchOpenCv()
    {
        Fixture.RequireOrSkip(FixtureName);
        var fixtures = Fixture.Load(FixtureName);
        int count = (int)fixtures["count"].ToInt64()[0];

        var failures = new List<string>();

        for (int index = 0; index < count; index++)
        {
            NpyArray maskArray = fixtures[$"mask_{index}"];
            int height = maskArray.Shape[0];
            int width = maskArray.Shape[1];
            byte[] mask = maskArray.ToBytes();

            List<PixelPoint[]> actual = Contours.FindExternal(mask, width, height);

            long[] lengths = fixtures[$"lengths_{index}"].ToInt64();
            long[] flat = fixtures[$"points_{index}"].ToInt64();
            double[] areas = fixtures[$"areas_{index}"].ToDoubles();
            double[] arcs = fixtures[$"arcs_{index}"].ToDoubles();
            long[] approximationLengths = fixtures[$"approx_lengths_{index}"].ToInt64();
            long[] approximationFlat = fixtures[$"approx_{index}"].ToInt64();

            if (actual.Count != lengths.Length)
            {
                failures.Add($"mask {index}: expected {lengths.Length} contours, got {actual.Count}");
                continue;
            }

            // The two sides discover components in different orders, so contours are matched by
            // their starting point, which is unique per component.
            var expected = new List<(PixelPoint[] Points, double Area, double Arc, PixelPoint[] Approximate)>();
            int offset = 0;
            int approximationOffset = 0;

            for (int c = 0; c < lengths.Length; c++)
            {
                var points = new PixelPoint[lengths[c]];
                for (int p = 0; p < points.Length; p++)
                {
                    points[p] = new PixelPoint((int)flat[(offset + p) * 2], (int)flat[((offset + p) * 2) + 1]);
                }

                offset += points.Length;

                var approximate = new PixelPoint[approximationLengths[c]];
                for (int p = 0; p < approximate.Length; p++)
                {
                    approximate[p] = new PixelPoint(
                        (int)approximationFlat[(approximationOffset + p) * 2],
                        (int)approximationFlat[((approximationOffset + p) * 2) + 1]);
                }

                approximationOffset += approximate.Length;
                expected.Add((points, areas[c], arcs[c], approximate));
            }

            foreach (PixelPoint[] contour in actual)
            {
                int match = expected.FindIndex(e => e.Points[0] == contour[0]);
                if (match < 0)
                {
                    failures.Add($"mask {index}: no reference contour starts at {contour[0]}");
                    continue;
                }

                var reference = expected[match];
                expected.RemoveAt(match);

                if (!contour.SequenceEqual(reference.Points))
                {
                    failures.Add(
                        $"mask {index}: contour at {contour[0]} traced as " +
                        $"[{string.Join(" ", contour)}], expected [{string.Join(" ", reference.Points)}]");
                    continue;
                }

                double area = Contours.Area(contour);
                if (Math.Abs(area - reference.Area) > 1e-9)
                {
                    failures.Add($"mask {index}: area {area} != {reference.Area}");
                }

                double arc = Contours.ArcLength(contour, closed: true);
                if (Math.Abs(arc - reference.Arc) > 1e-6)
                {
                    failures.Add($"mask {index}: arc length {arc} != {reference.Arc}");
                }

                PixelPoint[] approximate = Contours.ApproxPolyDp(contour, 0.004 * reference.Arc, closed: true);
                if (!approximate.SequenceEqual(reference.Approximate))
                {
                    failures.Add(
                        $"mask {index}: approxPolyDP gave [{string.Join(" ", approximate)}], " +
                        $"expected [{string.Join(" ", reference.Approximate)}]");
                }
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures.Take(8)));
    }
}
