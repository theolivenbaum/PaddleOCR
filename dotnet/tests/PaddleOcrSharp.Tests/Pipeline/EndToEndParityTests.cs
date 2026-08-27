using System.Text;
using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models;
using PaddleOcrSharp.Models.Layout;
using PaddleOcrSharp.Pipeline;
using PaddleOcrSharp.Tests.Fixtures;
using PaddleOcrSharp.Text;

namespace PaddleOcrSharp.Tests.Pipeline;

/// <summary>
/// Runs the whole pipeline on a page and compares its blocks against the upstream Python run.
/// Fixtures come from <c>dotnet/tools/reference/dump_end_to_end.py</c>.
/// </summary>
[Collection(CheckpointCollection.Name)]
public class EndToEndParityTests(CheckpointFixture checkpoint)
{
    private const string FixtureName = "end_to_end.npz";

    [Fact]
    public void PageParseMatchesUpstream()
    {
        Fixture.RequireOrSkip(FixtureName);
        checkpoint.RequireOrSkip();
        CheckpointFixture.RequireTokenizerOrSkip();
        LayoutModelFixture.RequireOrSkip();

        var fixtures = Fixture.Load(FixtureName);
        var source = fixtures["source"];
        float[] boxes = fixtures["boxes"].ToFloats();
        int blockCount = (int)fixtures["block_count"].ToInt64()[0];

        var tokenizer = BpeTokenizer.FromFile(Path.Combine(CheckpointFixture.Directory!, "tokenizer.json"));
        using PaddleOcrVLModel model = PaddleOcrVLModel.FromWeights(
            checkpoint.Weights, ModelConfiguration.Default, tokenizer);
        using LayoutDetector layout = LayoutDetector.Load(LayoutModelFixture.Directory!);
        using var parser = new DocumentParser(model, layout);

        using RgbImage page = RgbImage.From(source.ToBytes(), source.Shape[1], source.Shape[0]);
        ParsedPage parsed = parser.Parse(
            page,
            DocumentParserOptions.Default,
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal(blockCount, parsed.Blocks.Count);

        for (int i = 0; i < blockCount; i++)
        {
            ParsedBlock block = parsed.Blocks[i];
            int expectedClass = (int)boxes[i * 7];
            string expectedContent = Encoding.UTF8.GetString(fixtures[$"content{i}"].ToBytes());

            Assert.Equal(expectedClass, block.Box.ClassId);

            // Boxes come from the detector; a pixel of drift is the resize's one-level difference
            // propagating, not a pipeline disagreement.
            Assert.Equal(boxes[(i * 7) + 2], block.Box.Left, 2f);
            Assert.Equal(boxes[(i * 7) + 3], block.Box.Top, 2f);
            Assert.Equal(boxes[(i * 7) + 4], block.Box.Right, 2f);
            Assert.Equal(boxes[(i * 7) + 5], block.Box.Bottom, 2f);

            Assert.Equal(expectedContent, block.Content);
        }
    }
}
