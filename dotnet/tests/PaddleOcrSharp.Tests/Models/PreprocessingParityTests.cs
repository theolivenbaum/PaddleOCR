using PaddleOcrSharp.Formats;
using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models.Paddle;
using PaddleOcrSharp.Models.Preprocessing;
using PaddleOcrSharp.Tests.Fixtures;

namespace PaddleOcrSharp.Tests.Models;

/// <summary>
/// Parity tests for the two document pre-processing models. Both ship only as Paddle inference
/// graphs, so each is checked twice: the graph alone against Paddle's own fetched outputs, and
/// then the public wrapper end to end, which also exercises our pre- and post-processing.
/// Fixtures come from <c>dotnet/tools/reference/dump_preprocessing.py</c>.
/// </summary>
public class PreprocessingParityTests
{
    private const string FixtureName = "preprocessing.npz";

    [Theory]
    [InlineData("upright", 0)]
    [InlineData("rotated", 90)]
    public void OrientationGraphMatchesPaddleInference(string page, int expectedAngle)
    {
        Fixture.RequireOrSkip(FixtureName);
        PreprocessingModelFixture.RequireOrientationOrSkip();

        var fixtures = Fixture.Load(FixtureName);
        var input = fixtures[$"ori_{page}_input"];
        float[] expected = fixtures[$"ori_{page}_logits"].ToFloats();

        using var interpreter = PirInterpreter.Load(PreprocessingModelFixture.OrientationDirectory!);

        Dictionary<string, PaddleTensor> outputs = interpreter.Run(
            new Dictionary<string, PaddleTensor>(StringComparer.Ordinal)
            {
                ["x"] = PaddleTensor.FromFloats(input.ToFloats(), input.Shape),
            });

        Span<float> actual = outputs.Values.First().FloatSpan;

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], 0.002f);
        }

        Assert.Equal(expectedAngle, 90 * Argmax(expected));
    }

    [Theory]
    [InlineData("upright", 0)]
    [InlineData("rotated", 90)]
    public void OrientationClassifierPredictsTheReferenceAngle(string page, int expectedAngle)
    {
        Fixture.RequireOrSkip(FixtureName);
        PreprocessingModelFixture.RequireOrientationOrSkip();

        var fixtures = Fixture.Load(FixtureName);
        var source = fixtures[$"ori_{page}_source"];
        float[] expected = fixtures[$"ori_{page}_logits"].ToFloats();

        using RgbImage image = RgbImage.From(source.ToBytes(), source.Shape[1], source.Shape[0]);
        using var classifier = DocOrientationClassifier.Load(PreprocessingModelFixture.OrientationDirectory!);

        (int angle, float score) = classifier.Predict(image);

        Assert.Equal(expectedAngle, angle);

        // Our resize is OpenCV-compatible to within a level, so the score can drift a little
        // further than the graph-only comparison above allows.
        Assert.Equal(expected[Argmax(expected)], score, 0.01f);
    }

    [Fact]
    public void CorrectingARotatedPageRestoresIt()
    {
        Fixture.RequireOrSkip(FixtureName);
        PreprocessingModelFixture.RequireOrientationOrSkip();

        var fixtures = Fixture.Load(FixtureName);
        NpyArray upright = fixtures["ori_upright_source"];
        NpyArray rotated = fixtures["ori_rotated_source"];

        using RgbImage page = RgbImage.From(rotated.ToBytes(), rotated.Shape[1], rotated.Shape[0]);
        using var classifier = DocOrientationClassifier.Load(
            PreprocessingModelFixture.OrientationDirectory!);
        using RgbImage corrected = classifier.Correct(page);

        // The fixture's rotated page is the upright one turned a quarter clockwise, so correcting
        // it has to turn it back — the direction the rotation goes is the whole point.
        Assert.Equal(upright.Shape[1], corrected.Width);
        Assert.Equal(upright.Shape[0], corrected.Height);
        Assert.Equal(upright.ToBytes(), corrected.Pixels.ToArray());
    }

    [Fact]
    public void UnwarpGraphMatchesPaddleInference()
    {
        Fixture.RequireOrSkip(FixtureName);
        PreprocessingModelFixture.RequireUnwarpOrSkip();

        var fixtures = Fixture.Load(FixtureName);
        var input = fixtures["uvdoc_input"];
        var expectedArray = fixtures["uvdoc_output"];
        float[] expected = expectedArray.ToFloats();

        using var interpreter = PirInterpreter.Load(PreprocessingModelFixture.UnwarpDirectory!);

        Dictionary<string, PaddleTensor> outputs = interpreter.Run(
            new Dictionary<string, PaddleTensor>(StringComparer.Ordinal)
            {
                ["image"] = PaddleTensor.FromFloats(input.ToFloats(), input.Shape),
            });

        PaddleTensor warped = outputs.Values.First();

        Assert.Equal(expectedArray.Shape, warped.Shape);
        TensorAssert.Close(expected, warped.FloatSpan, 0.002f);
    }

    [Fact]
    public void UnwarperReproducesTheReferenceImage()
    {
        Fixture.RequireOrSkip(FixtureName);
        PreprocessingModelFixture.RequireUnwarpOrSkip();

        var fixtures = Fixture.Load(FixtureName);
        var source = fixtures["uvdoc_source"];
        float[] expected = fixtures["uvdoc_output"].ToFloats();

        using RgbImage image = RgbImage.From(source.ToBytes(), source.Shape[1], source.Shape[0]);
        using var unwarper = DocumentUnwarper.Load(PreprocessingModelFixture.UnwarpDirectory!);
        using RgbImage flattened = unwarper.Unwarp(image);

        Assert.Equal(source.Shape[1], flattened.Width);
        Assert.Equal(source.Shape[0], flattened.Height);

        int plane = flattened.Width * flattened.Height;
        int worst = 0;

        for (int y = 0; y < flattened.Height; y++)
        {
            ReadOnlySpan<byte> row = flattened.Row(y);
            for (int x = 0; x < flattened.Width; x++)
            {
                for (int c = 0; c < 3; c++)
                {
                    int reference = (int)MathF.Round(
                        Math.Clamp(expected[(c * plane) + (y * flattened.Width) + x], 0f, 1f) * 255f);
                    worst = Math.Max(worst, Math.Abs(row[(x * 3) + c] - reference));
                }
            }
        }

        Assert.True(worst <= 1, $"Unwarped pixels differ from the reference by up to {worst} levels.");
    }

    private static int Argmax(ReadOnlySpan<float> values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }

        return best;
    }
}
