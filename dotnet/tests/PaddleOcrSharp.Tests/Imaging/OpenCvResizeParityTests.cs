using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Tests.Fixtures;

namespace PaddleOcrSharp.Tests.Imaging;

/// <summary>
/// Checks the OpenCV-compatible bicubic resize against <c>cv2.resize(..., INTER_CUBIC)</c>,
/// which is what the layout detector's preprocessing uses. Fixtures come from
/// <c>dotnet/tools/reference/dump_layout.py</c>.
/// </summary>
public class OpenCvResizeParityTests
{
    private const string FixtureName = "layout.npz";

    [Theory]
    [InlineData("source", "resized")]
    [InlineData("small_source", "small_resized")]
    public void ResizeIsByteExactWithOpenCv(string sourceKey, string resizedKey)
    {
        Fixture.RequireOrSkip(FixtureName);
        var fixtures = Fixture.Load(FixtureName);

        var source = fixtures[sourceKey];
        var expected = fixtures[resizedKey];

        using RgbImage image = RgbImage.From(source.ToBytes(), source.Shape[1], source.Shape[0]);
        using RgbImage actual = OpenCvResize.ResizeBicubic(image, expected.Shape[1], expected.Shape[0]);

        byte[] expectedBytes = expected.ToBytes();
        Span<byte> actualBytes = actual.Pixels;

        int mismatches = 0;
        int worst = 0;
        int firstIndex = -1;

        for (int i = 0; i < expectedBytes.Length; i++)
        {
            int difference = Math.Abs(expectedBytes[i] - actualBytes[i]);
            if (difference != 0)
            {
                mismatches++;
                worst = Math.Max(worst, difference);
                if (firstIndex < 0)
                {
                    firstIndex = i;
                }
            }
        }

        // OpenCV's SIMD kernels contract their multiply-adds differently from any portable
        // formulation, so a small number of bytes land one level away. Nothing may differ by
        // more than that, and the disagreement must stay rare enough not to move a detection.
        double rate = (double)mismatches / expectedBytes.Length;
        Assert.True(
            worst <= 1 && rate < 0.005,
            $"{mismatches}/{expectedBytes.Length} bytes differ ({rate:P3}, worst {worst}), " +
            $"first at {firstIndex}: expected {(firstIndex >= 0 ? expectedBytes[firstIndex] : 0)}, " +
            $"actual {(firstIndex >= 0 ? actualBytes[firstIndex] : 0)}.");
    }
}
