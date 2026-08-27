using PaddleOcrSharp.Core;
using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models;
using PaddleOcrSharp.Models.Vision;
using PaddleOcrSharp.Tests.Fixtures;

namespace PaddleOcrSharp.Tests.Models;

/// <summary>
/// Compares the C# vision tower against the Hugging Face reference, layer by layer.
/// Fixtures come from <c>dotnet/tools/reference/dump_vision.py</c>.
/// </summary>
[Collection(CheckpointCollection.Name)]
public class VisionTowerParityTests(CheckpointFixture checkpoint)
{
    private const string FixtureName = "vision.npz";

    [Fact]
    public void EncoderMatchesUpstreamLayerByLayer()
    {
        Fixture.RequireOrSkip(FixtureName);
        checkpoint.RequireOrSkip();

        var fixtures = Fixture.Load(FixtureName);
        long[] grid = fixtures["grid"].ToInt64();
        var imageGrid = new ImageGrid((int)grid[0], (int)grid[1], (int)grid[2]);
        int layers = (int)fixtures["layer_count"].ToInt64()[0];

        float[] pixelValues = fixtures["pixel_values"].ToFloats();
        using Tensor pixels = Tensor.From(pixelValues, imageGrid.PatchCount, pixelValues.Length / imageGrid.PatchCount);
        using var image = new PreprocessedImage(pixels, imageGrid);

        var tower = new VisionTower(checkpoint.Weights, VisionConfig.Default, languageHiddenSize: 1024);
        var trace = new VisionTrace();
        using Tensor hidden = tower.RunEncoder(image, trace);

        // Tolerances widen with depth because float32 reassociation compounds; the reference is
        // itself float32, so anything beyond a few 1e-3 at layer 27 would signal a real bug.
        AssertStage(fixtures, trace, "embeddings", 2e-4);
        for (int i = 0; i < layers; i++)
        {
            AssertStage(fixtures, trace, $"layer{i}", 2e-4 + (i * 6e-5));
        }

        AssertStage(fixtures, trace, "post_layernorm", 3e-3);

        using Tensor projected = tower.Project(hidden, imageGrid, trace);
        var expectedProjection = fixtures["projector"];
        Assert.Equal(expectedProjection.Shape[0], projected.Shape[0]);
        Assert.Equal(expectedProjection.Shape[1], projected.Shape[1]);
        TensorAssert.Close(
            expectedProjection.ToFloats(),
            projected.Span,
            absoluteTolerance: 4e-3,
            relativeTolerance: 4e-3,
            because: "projector");
    }

    [Fact]
    public void PreprocessingReproducesReferencePixelValues()
    {
        Fixture.RequireOrSkip(FixtureName);

        var fixtures = Fixture.Load(FixtureName);
        var source = fixtures["source"];

        using RgbImage input = RgbImage.From(source.ToBytes(), source.Shape[1], source.Shape[0]);
        using PreprocessedImage actual = VisionPreprocessor.Preprocess(input, VisionPreprocessorOptions.Default);

        TensorAssert.Close(fixtures["pixel_values"].ToFloats(), actual.PixelValues.Span, absoluteTolerance: 1e-6);
    }

    private static void AssertStage(
        Dictionary<string, PaddleOcrSharp.Formats.NpyArray> fixtures,
        VisionTrace trace,
        string stage,
        double tolerance)
    {
        if (!fixtures.TryGetValue(stage, out var expected))
        {
            return;
        }

        (float[] values, int rows, int cols) = trace.Get(stage);
        Assert.Equal(expected.Shape[0], rows);
        Assert.Equal(expected.Shape[1], cols);

        TensorAssert.Close(
            expected.ToFloats(),
            values,
            absoluteTolerance: tolerance,
            relativeTolerance: tolerance,
            because: stage);
    }
}
