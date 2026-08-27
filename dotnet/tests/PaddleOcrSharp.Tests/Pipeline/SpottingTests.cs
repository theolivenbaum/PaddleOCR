using PaddleOcrSharp.Pipeline;

namespace PaddleOcrSharp.Tests.Pipeline;

/// <summary>Checks the spotting output parser against both of upstream's formats.</summary>
public class SpottingTests
{
    [Fact]
    public void DelimitedRunsAreParsed()
    {
        const string Output =
            "<|TEXT_START|>Hello<|TEXT_END|>" +
            "<|LOC_BEGIN|><|LOC_100|><|LOC_200|><|LOC_300|><|LOC_200|>" +
            "<|LOC_300|><|LOC_400|><|LOC_100|><|LOC_400|><|LOC_END|>";

        (string text, IReadOnlyList<SpottedText> runs) = Spotting.Parse(Output, 1000, 1000);

        Assert.Equal("Hello", text);
        SpottedText run = Assert.Single(runs);
        Assert.Equal("Hello", run.Text);
        Assert.Equal(4, run.Polygon.Count);
        Assert.Equal((100f, 200f), run.Polygon[0]);
        Assert.Equal((300f, 400f), run.Polygon[2]);
    }

    [Fact]
    public void CoordinatesScaleToTheImageSize()
    {
        const string Output =
            "<|TEXT_START|>x<|TEXT_END|><|LOC_BEGIN|>" +
            "<|LOC_500|><|LOC_500|><|LOC_500|><|LOC_500|>" +
            "<|LOC_500|><|LOC_500|><|LOC_500|><|LOC_500|><|LOC_END|>";

        (_, IReadOnlyList<SpottedText> runs) = Spotting.Parse(Output, 800, 400);

        Assert.Equal((400f, 200f), runs[0].Polygon[0]);
    }

    [Fact]
    public void BareLocationTokensFallBackToPositionalParsing()
    {
        const string Output =
            "first line<|LOC_0|><|LOC_0|><|LOC_100|><|LOC_0|><|LOC_100|><|LOC_50|><|LOC_0|><|LOC_50|>" +
            "second line<|LOC_0|><|LOC_60|><|LOC_100|><|LOC_60|><|LOC_100|><|LOC_110|><|LOC_0|><|LOC_110|>";

        (string text, IReadOnlyList<SpottedText> runs) = Spotting.Parse(Output, 1000, 1000);

        Assert.Equal(2, runs.Count);
        Assert.Equal("first line", runs[0].Text);
        Assert.Equal("second line", runs[1].Text);
        Assert.Equal("first line\n\nsecond line", text);
    }

    [Fact]
    public void RunsWithTooFewCoordinatesAreSkipped()
    {
        const string Output = "<|TEXT_START|>x<|TEXT_END|><|LOC_BEGIN|><|LOC_1|><|LOC_2|><|LOC_END|>";

        (string text, IReadOnlyList<SpottedText> runs) = Spotting.Parse(Output, 100, 100);

        Assert.Empty(runs);
        Assert.Equal(string.Empty, text);
    }

    [Fact]
    public void PlainTextProducesNoRuns()
    {
        (string text, IReadOnlyList<SpottedText> runs) = Spotting.Parse("just some text", 100, 100);

        Assert.Empty(runs);
        Assert.Equal(string.Empty, text);
    }
}
