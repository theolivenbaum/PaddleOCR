using System.Text;
using PaddleOcrSharp.Core;
using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models;
using PaddleOcrSharp.Models.Language;
using PaddleOcrSharp.Models.Vision;
using PaddleOcrSharp.Tests.Fixtures;
using PaddleOcrSharp.Text;

namespace PaddleOcrSharp.Tests.Models;

/// <summary>
/// End-to-end parity for the decoder: prompt construction, 3-D rope index, image-feature scatter,
/// prefill logits and greedy decoding. Fixtures come from
/// <c>dotnet/tools/reference/dump_language.py</c>.
/// </summary>
[Collection(CheckpointCollection.Name)]
public class LanguageModelParityTests(CheckpointFixture checkpoint)
{
    private const string FixtureName = "language.npz";

    [Fact]
    public void PromptMatchesTheProcessorOutput()
    {
        Fixture.RequireOrSkip(FixtureName);
        CheckpointFixture.RequireTokenizerOrSkip();

        var fixtures = Fixture.Load(FixtureName);
        long[] grid = fixtures["grid"].ToInt64();
        long[] expected = fixtures["input_ids"].ToInt64();
        string prompt = Encoding.UTF8.GetString(fixtures["prompt"].ToBytes());

        var tokenizer = BpeTokenizer.FromFile(Path.Combine(CheckpointFixture.Directory!, "tokenizer.json"));
        using WeightStore? _ = null;

        int imageTokens = (int)(grid[0] * grid[1] * grid[2]) / 4;
        var ids = new List<int>();
        tokenizer.EncodeInto("<|begin_of_sentence|>User: <|IMAGE_START|>", ids);
        for (int i = 0; i < imageTokens; i++)
        {
            ids.Add(100_295);
        }

        tokenizer.EncodeInto($"<|IMAGE_END|>{prompt}\nAssistant:\n", ids);

        TensorAssert.Equal(expected, [.. ids.Select(value => (long)value)]);
    }

    [Fact]
    public void RopeIndexMatchesGetRopeIndex()
    {
        Fixture.RequireOrSkip(FixtureName);

        var fixtures = Fixture.Load(FixtureName);
        long[] grid = fixtures["grid"].ToInt64();
        int[] tokens = [.. fixtures["input_ids"].ToInt64().Select(value => (int)value)];
        var expected = fixtures["position_ids"];
        long[] expectedValues = expected.ToInt64();
        int length = expected.Shape[1];

        (PositionIds actual, int delta) = RopeIndex.Compute(
            tokens,
            [((int)grid[0], (int)grid[1], (int)grid[2])],
            LanguageConfig.Default,
            spatialMergeSize: 2);

        Assert.Equal((int)fixtures["rope_delta"].ToInt64()[0], delta);

        for (int i = 0; i < length; i++)
        {
            Assert.Equal(expectedValues[i], actual.Temporal[i]);
            Assert.Equal(expectedValues[length + i], actual.Height[i]);
            Assert.Equal(expectedValues[(2 * length) + i], actual.Width[i]);
        }
    }

    [Fact]
    public void PrefillAndGreedyDecodeMatchUpstream()
    {
        Fixture.RequireOrSkip(FixtureName);
        checkpoint.RequireOrSkip();
        CheckpointFixture.RequireTokenizerOrSkip();

        var fixtures = Fixture.Load(FixtureName);
        var source = fixtures["source"];
        long[] grid = fixtures["grid"].ToInt64();
        var imageGrid = new ImageGrid((int)grid[0], (int)grid[1], (int)grid[2]);
        string prompt = Encoding.UTF8.GetString(fixtures["prompt"].ToBytes());

        var tokenizer = BpeTokenizer.FromFile(Path.Combine(CheckpointFixture.Directory!, "tokenizer.json"));
        using PaddleOcrVLModel model = PaddleOcrVLModel.FromWeights(
            checkpoint.Weights, ModelConfiguration.Default, tokenizer);

        using RgbImage image = RgbImage.From(source.ToBytes(), source.Shape[1], source.Shape[0]);
        using PreprocessedImage preprocessed = VisionPreprocessor.Preprocess(
            image, VisionPreprocessorOptions.Default);

        Assert.Equal(imageGrid, preprocessed.Grid);

        int[] tokens = model.BuildPrompt(preprocessed.Grid, prompt);
        TensorAssert.Equal(fixtures["input_ids"].ToInt64(), [.. tokens.Select(value => (long)value)]);

        using Tensor imageEmbeddings = model.Vision.Encode(preprocessed);

        var options = GenerationOptions.Default with { MaxNewTokens = fixtures["greedy_tokens"].Shape[0] };
        List<int> generated = model.Generate(
            tokens, imageEmbeddings, preprocessed.Grid, options, TestContext.Current.CancellationToken);

        long[] expectedTokens = fixtures["greedy_tokens"].ToInt64();
        string expectedText = Encoding.UTF8.GetString(fixtures["decoded"].ToBytes());
        string actualText = tokenizer.Decode(generated);

        Assert.Equal(expectedText, actualText);
        TensorAssert.Equal(expectedTokens, [.. generated.Select(value => (long)value)]);
    }
}
