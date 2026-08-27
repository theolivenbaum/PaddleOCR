using PaddleOcrSharp.Models.Layout;
using PaddleOcrSharp.Pipeline;

namespace PaddleOcrSharp.Tests.Pipeline;

/// <summary>Checks the per-label markdown formatting.</summary>
public class MarkdownWriterTests
{
    private static ParsedBlock Block(string label, string content, int order = 0) =>
        new(label, new LayoutBox(0, label, 1f, 0, 0, 10, 10, order), content, order);

    [Fact]
    public void DocTitleBecomesLevelOneHeading()
    {
        string markdown = MarkdownWriter.Write([Block("doc_title", "A Paper")], MarkdownOptions.Default);
        Assert.Equal("# A Paper", markdown);
    }

    [Fact]
    public void NumberedParagraphTitleGetsADepthMatchedHeading()
    {
        string markdown = MarkdownWriter.Write(
            [Block("paragraph_title", "2.1 Method")], MarkdownOptions.Default);

        Assert.Equal("### 2.1 Method", markdown);
    }

    [Fact]
    public void UnnumberedParagraphTitleGetsLevelTwo()
    {
        string markdown = MarkdownWriter.Write(
            [Block("paragraph_title", "Method")], MarkdownOptions.Default);

        Assert.Equal("## Method", markdown);
    }

    [Fact]
    public void TextBlocksAreSeparatedByABlankLine()
    {
        string markdown = MarkdownWriter.Write(
            [Block("text", "first"), Block("text", "second", 1)], MarkdownOptions.Default);

        Assert.Equal("first\n\nsecond", markdown);
    }

    [Fact]
    public void IgnoredLabelsAreDropped()
    {
        string markdown = MarkdownWriter.Write(
            [Block("header", "page header"), Block("text", "body", 1)], MarkdownOptions.Default);

        Assert.Equal("body", markdown);
    }

    [Fact]
    public void TableContentIsEmittedVerbatim()
    {
        const string Html = "<table><tr><td>a</td></tr></table>";
        string markdown = MarkdownWriter.Write([Block("table", Html)], MarkdownOptions.Default);
        Assert.Equal(Html, markdown);
    }

    [Fact]
    public void FormulasAreWrappedInDisplayMath()
    {
        string markdown = MarkdownWriter.Write(
            [Block("display_formula", "E = mc^2")], MarkdownOptions.Default);

        Assert.Equal("$$E = mc^2$$", markdown);
    }

    [Fact]
    public void AlreadyWrappedFormulasAreNotDoubleWrapped()
    {
        string markdown = MarkdownWriter.Write(
            [Block("display_formula", "$$E = mc^2$$")], MarkdownOptions.Default);

        Assert.Equal("$$E = mc^2$$", markdown);
    }

    [Fact]
    public void ImageBlocksRenderAsCentredImgTags()
    {
        var block = Block("image", string.Empty) with { ImagePath = "image_0_10_20.png" };
        string markdown = MarkdownWriter.Write([block], MarkdownOptions.Default);

        Assert.Contains("<img src=\"imgs/image_0_10_20.png\"", markdown);
        Assert.Contains("text-align: center", markdown);
    }
}
