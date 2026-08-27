using PaddleOcrSharp.Models.Layout;
using PaddleOcrSharp.Pipeline;

namespace PaddleOcrSharp.Tests.Pipeline;

/// <summary>
/// The page's reading-flow numbering, against <c>update_order_index</c>.
/// </summary>
public class BlockOrderTests
{
    [Fact]
    public void SkipLabelsAreThoseUpstreamSkips()
    {
        Assert.Equal(
            [
                "figure_title", "vision_footnote", "image", "chart", "table", "header",
                "header_image", "footer", "footer_image", "footnote", "aside_text",
            ],
            BlockLabels.SkipOrder);
    }

    [Fact]
    public void OnlyFlowingBlocksAreNumbered()
    {
        ParsedPage page = Parse([
            "doc_title", "text", "image", "text", "table", "paragraph_title", "figure_title",
        ]);

        Assert.Equal(
            [1, 2, null, 3, null, 4, null],
            page.Blocks.Select(block => block.Order));
    }

    [Fact]
    public void MarkdownExclusionsAlsoSitOut()
    {
        // Upstream unions SKIP_ORDER_LABELS with markdown_ignore_labels, so a label the markdown
        // drops does not take a number either. "header" and "footer" are in both lists; "number"
        // is only in the markdown one, which is what makes the union observable.
        ParsedPage page = Parse(["text", "number", "text"]);

        Assert.Equal([1, null, 2], page.Blocks.Select(block => block.Order));
    }

    [Fact]
    public void NumberingIsPerPage()
    {
        Assert.Equal([1, 2], Parse(["text", "text"]).Blocks.Select(block => block.Order));
        Assert.Equal([1, 2], Parse(["text", "text"]).Blocks.Select(block => block.Order));
    }

    private static ParsedPage Parse(string[] labels)
    {
        ParsedBlock[] blocks = [.. labels.Select((label, index) => new ParsedBlock(
            label,
            new LayoutBox(0, label, 1f, 0, index * 10, 100, (index * 10) + 8, index),
            "content",
            index))];

        return new ParsedPage(
            0,
            100,
            labels.Length * 10,
            BlockOrder.Assign(blocks, MarkdownOptions.Default.IgnoredLabels));
    }
}
