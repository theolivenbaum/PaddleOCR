using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Tests.Fixtures;

namespace PaddleOcrSharp.Tests.Imaging;

/// <summary>
/// Compares the C# image pipeline against `PaddleOCRVLImageProcessor` from the checkpoint.
/// Fixtures come from <c>dotnet/tools/reference/dump_image_processing.py</c>.
/// </summary>
public class ImagePreprocessingParityTests
{
    private const string FixtureName = "image_processing.npz";

    [Theory]
    [InlineData(457, 613, 448, 616)]
    [InlineData(1080, 1920, 728, 1316)]
    [InlineData(96, 128, 308, 392)]
    [InlineData(137, 2048, 140, 2044)]
    [InlineData(280, 280, 336, 336)]
    [InlineData(941, 57, 1372, 84)]
    public void SmartResizeMatchesUpstream(int height, int width, int expectedHeight, int expectedWidth)
    {
        (int h, int w) = SmartResize.Compute(
            height,
            width,
            VisionPreprocessorOptions.Default.Factor,
            VisionPreprocessorOptions.Default.MinPixels,
            VisionPreprocessorOptions.Default.MaxPixels);

        Assert.Equal(expectedHeight, h);
        Assert.Equal(expectedWidth, w);
    }

    [Fact]
    public void BicubicResizeIsByteExactWithPillow()
    {
        Fixture.RequireOrSkip(FixtureName);
        var fixtures = Fixture.Load(FixtureName);
        int cases = (int)fixtures["case_count"].ToInt64()[0];

        for (int index = 0; index < cases; index++)
        {
            var source = fixtures[$"case{index}_source"];
            var expected = fixtures[$"case{index}_resized"];

            using RgbImage image = RgbImage.From(source.ToBytes(), source.Shape[1], source.Shape[0]);
            using RgbImage actual = PilResize.ResizeBicubic(image, expected.Shape[1], expected.Shape[0]);

            byte[] expectedBytes = expected.ToBytes();
            Span<byte> actualBytes = actual.Pixels;

            int mismatches = 0;
            int firstMismatch = -1;
            for (int i = 0; i < expectedBytes.Length; i++)
            {
                if (expectedBytes[i] != actualBytes[i])
                {
                    mismatches++;
                    if (firstMismatch < 0)
                    {
                        firstMismatch = i;
                    }
                }
            }

            Assert.True(
                mismatches == 0,
                $"case{index}: {mismatches}/{expectedBytes.Length} bytes differ, first at {firstMismatch} " +
                $"(expected {(firstMismatch >= 0 ? expectedBytes[firstMismatch] : 0)}, " +
                $"actual {(firstMismatch >= 0 ? actualBytes[firstMismatch] : 0)}).");
        }
    }

    [Fact]
    public void PreprocessMatchesUpstreamPixelValues()
    {
        Fixture.RequireOrSkip(FixtureName);
        var fixtures = Fixture.Load(FixtureName);
        int cases = (int)fixtures["case_count"].ToInt64()[0];

        for (int index = 0; index < cases; index++)
        {
            var source = fixtures[$"case{index}_source"];
            var expected = fixtures[$"case{index}_pixel_values"];
            long[] grid = fixtures[$"case{index}_grid"].ToInt64();

            using RgbImage image = RgbImage.From(source.ToBytes(), source.Shape[1], source.Shape[0]);
            using PreprocessedImage actual = VisionPreprocessor.Preprocess(
                image, VisionPreprocessorOptions.Default);

            Assert.Equal((int)grid[0], actual.Grid.Temporal);
            Assert.Equal((int)grid[1], actual.Grid.Height);
            Assert.Equal((int)grid[2], actual.Grid.Width);

            TensorAssert.Close(
                expected.ToFloats(),
                actual.PixelValues.Span,
                absoluteTolerance: 1e-6,
                because: $"case{index}");
        }
    }
}
