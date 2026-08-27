using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models.Layout;
using PaddleOcrSharp.Pipeline;

namespace PaddleOcrSharp.Tests.Pipeline;

/// <summary>Checks region grouping and the overlap filter against hand-built layouts.</summary>
public class BlockMergerTests
{
    private static readonly string[] NonMerge = ["image", "header_image", "footer_image", "table", "chart", "seal"];

    private static LayoutBox Box(string label, float left, float top, float right, float bottom, int order = 0) =>
        new(0, label, 0.9f, left, top, right, bottom, order);

    private static (int Width, int Height)[] Sizes(IEnumerable<LayoutBox> boxes) =>
        [.. boxes.Select(box => ((int)box.Width, (int)box.Height))];

    [Fact]
    public void UnrelatedRegionsEachFormTheirOwnGroup()
    {
        LayoutBox[] regions =
        [
            Box("doc_title", 0, 0, 400, 40),
            Box("text", 0, 60, 400, 120, 1),
            Box("table", 0, 140, 400, 300, 2),
        ];

        List<BlockGroup> groups = BlockMerger.Group(regions, Sizes(regions), NonMerge);

        Assert.Equal(3, groups.Count);
        Assert.All(groups, group => Assert.Single(group.Indices));
    }

    [Fact]
    public void ColumnsThatContinueSidewaysAreGrouped()
    {
        // A left column and a right column that starts before the left one ends: the classic
        // two-column continuation.
        LayoutBox[] regions =
        [
            Box("text", 0, 0, 200, 400),
            Box("text", 240, 20, 440, 420, 1),
        ];

        List<BlockGroup> groups = BlockMerger.Group(regions, [(200, 100), (200, 100)], NonMerge);

        BlockGroup group = Assert.Single(groups);
        Assert.Equal([0, 1], group.Indices);
        Assert.Equal([StackAlignment.Center], group.Alignments);
    }

    [Fact]
    public void ATallStackIsLeftUnmerged()
    {
        LayoutBox[] regions =
        [
            Box("text", 0, 0, 200, 400),
            Box("text", 240, 20, 440, 420, 1),
        ];

        // Stacked, these two crops would be 200 wide and 800 tall: past the aspect-ratio guard.
        List<BlockGroup> groups = BlockMerger.Group(regions, [(200, 400), (200, 400)], NonMerge);

        Assert.Equal(2, groups.Count);
        Assert.All(groups, group => Assert.Single(group.Indices));
    }

    [Fact]
    public void ParagraphSplitAroundAFigureIsGrouped()
    {
        // Two text runs sharing a left edge but not a right one — exactly one aligned edge is
        // what upstream requires — with a figure straddling the gap between them. The figure is
        // what tells the merger the paragraph was interrupted rather than ended.
        LayoutBox[] regions =
        [
            Box("text", 0, 0, 200, 100),
            Box("text", 0, 130, 170, 240, 1),
            Box("image", 190, 90, 400, 150, 2),
        ];

        List<BlockGroup> groups = BlockMerger.Group(regions, [(200, 100), (170, 110), (210, 60)], NonMerge);

        Assert.Equal(2, groups.Count);
        Assert.Equal([0, 1], groups[0].Indices);
        Assert.Equal([StackAlignment.Left], groups[0].Alignments);
        Assert.Equal([2], groups[1].Indices);
    }

    [Fact]
    public void ReferenceRegionsAreFilteredOut()
    {
        LayoutBox[] regions = [Box("text", 0, 0, 100, 50), Box("reference", 0, 60, 100, 90)];

        List<LayoutBox> filtered = OverlapFilter.Apply(regions);

        Assert.Equal("text", Assert.Single(filtered).Label);
    }

    [Fact]
    public void TinyRegionsAreFilteredOut()
    {
        LayoutBox[] regions = [Box("text", 0, 0, 100, 50), Box("text", 10, 10, 13, 13, 1)];

        List<LayoutBox> filtered = OverlapFilter.Apply(regions);

        Assert.Single(filtered);
    }

    [Fact]
    public void ContainedRegionLosesToItsContainer()
    {
        LayoutBox[] regions = [Box("text", 0, 0, 200, 200), Box("text", 10, 10, 100, 100, 1)];

        List<LayoutBox> filtered = OverlapFilter.Apply(regions);

        LayoutBox kept = Assert.Single(filtered);
        Assert.Equal(200f, kept.Right);
    }

    [Fact]
    public void AFigureInsideATableSurvives()
    {
        LayoutBox[] regions = [Box("table", 0, 0, 200, 200), Box("image", 10, 10, 100, 100, 1)];

        List<LayoutBox> filtered = OverlapFilter.Apply(regions);

        Assert.Equal(2, filtered.Count);
    }

    [Fact]
    public void InlineFormulaInsideTextIsDropped()
    {
        LayoutBox[] regions = [Box("text", 0, 0, 200, 200), Box("inline_formula", 20, 20, 80, 60, 1)];

        List<LayoutBox> filtered = OverlapFilter.Apply(regions);

        Assert.Equal("text", Assert.Single(filtered).Label);
    }

    [Fact]
    public void StackingPlacesImagesAtTheExpectedOffsets()
    {
        using RgbImage wide = RgbImage.From(new byte[60 * 2 * 3], 60, 2);
        using RgbImage narrow = RgbImage.From(Enumerable.Repeat((byte)200, 20 * 2 * 3).ToArray(), 20, 2);

        using RgbImage stacked = ImageStacker.Stack([wide, narrow], [StackAlignment.Center]);

        Assert.Equal(60, stacked.Width);
        Assert.Equal(4, stacked.Height);

        // The narrow image is centred: 20 px of white, then its 20 px, then white again.
        Span<byte> row = stacked.Row(2);
        Assert.Equal(255, row[0]);
        Assert.Equal(200, row[20 * 3]);
        Assert.Equal(255, row[(59 * 3)]);
    }

    [Fact]
    public void StackingRightAlignsWhenAsked()
    {
        using RgbImage wide = RgbImage.From(new byte[60 * 2 * 3], 60, 2);
        using RgbImage narrow = RgbImage.From(Enumerable.Repeat((byte)200, 20 * 2 * 3).ToArray(), 20, 2);

        using RgbImage stacked = ImageStacker.Stack([wide, narrow], [StackAlignment.Right]);

        Span<byte> row = stacked.Row(2);
        Assert.Equal(255, row[0]);
        Assert.Equal(200, row[(40 * 3)]);
    }
}
