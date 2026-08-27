using PaddleOcrSharp.Models.Paddle;
using PaddleOcrSharp.Tests.Fixtures;

namespace PaddleOcrSharp.Tests.Models;

/// <summary>
/// Runs the PP-DocLayoutV3 inference graph through the C# Paddle interpreter and compares its
/// fetched outputs against Paddle's own. Fixtures come from
/// <c>dotnet/tools/reference/dump_layout.py</c>.
/// </summary>
public class LayoutGraphParityTests
{
    private const string FixtureName = "layout.npz";

    [Fact]
    public void DetectionsMatchPaddleInference()
    {
        Fixture.RequireOrSkip(FixtureName);
        LayoutModelFixture.RequireOrSkip();

        var fixtures = Fixture.Load(FixtureName);

        using var interpreter = PirInterpreter.Load(LayoutModelFixture.Directory!);

        var image = fixtures["image"];
        var scaleFactor = fixtures["scale_factor"];
        var imageShape = fixtures["im_shape"];

        var inputs = new Dictionary<string, PaddleTensor>(StringComparer.Ordinal)
        {
            ["image"] = PaddleTensor.FromFloats(image.ToFloats(), image.Shape),
            ["scale_factor"] = PaddleTensor.FromFloats(scaleFactor.ToFloats(), scaleFactor.Shape),
            ["im_shape"] = PaddleTensor.FromFloats(imageShape.ToFloats(), imageShape.Shape),
        };

        Dictionary<string, PaddleTensor> outputs = interpreter.Run(inputs);

        PaddleTensor detections = outputs["fetch_name_0"];
        var expected = fixtures["output0"];

        Assert.Equal(expected.Shape[0], detections.Shape[0]);
        Assert.Equal(expected.Shape[1], detections.Shape[1]);

        float[] expectedValues = expected.ToFloats();
        Span<float> actualValues = detections.FloatSpan;

        // Compare only rows the reference keeps above the pipeline's threshold: the tail of the
        // 300 queries is noise whose ordering is not meaningful.
        int columns = expected.Shape[1];
        int compared = 0;

        for (int row = 0; row < expected.Shape[0]; row++)
        {
            if (expectedValues[(row * columns) + 1] < 0.05f)
            {
                continue;
            }

            compared++;
            Assert.Equal(expectedValues[row * columns], actualValues[row * columns], 0.001f);
            Assert.Equal(expectedValues[(row * columns) + 1], actualValues[(row * columns) + 1], 0.002f);

            for (int column = 2; column < 6; column++)
            {
                Assert.Equal(
                    expectedValues[(row * columns) + column],
                    actualValues[(row * columns) + column],
                    0.5f);
            }
        }

        Assert.True(compared > 0, "The reference fixture contains no detections to compare.");
    }
}

/// <summary>Locates the PP-DocLayoutV3 inference program.</summary>
public static class LayoutModelFixture
{
    /// <summary>Directory holding <c>inference.json</c>, or <see langword="null"/>.</summary>
    public static string? Directory
    {
        get
        {
            string? configured = Environment.GetEnvironmentVariable("PP_DOCLAYOUT_V3_DIR");
            if (!string.IsNullOrEmpty(configured) && File.Exists(Path.Combine(configured, "inference.json")))
            {
                return configured;
            }

            const string Fallback = "/home/user/ref/layout";
            return File.Exists(Path.Combine(Fallback, "inference.json")) ? Fallback : null;
        }
    }

    /// <summary>Skips the calling test when the model is not present.</summary>
    public static void RequireOrSkip()
    {
        if (Directory is null)
        {
            Assert.Skip("PP-DocLayoutV3 model not found; set PP_DOCLAYOUT_V3_DIR to a download.");
        }
    }
}
