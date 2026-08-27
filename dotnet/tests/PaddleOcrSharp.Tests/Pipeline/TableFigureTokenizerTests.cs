using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models.Layout;
using PaddleOcrSharp.Pipeline;

namespace PaddleOcrSharp.Tests.Pipeline;

/// <summary>Checks the table-figure placeholder round trip.</summary>
public class TableFigureTokenizerTests
{
    private static LayoutBox Box(string label, float left, float top, float right, float bottom) =>
        new(0, label, 0.9f, left, top, right, bottom, 0);

    private static RgbImage Grey(int width, int height) =>
        RgbImage.From(Enumerable.Repeat((byte)128, width * height * 3).ToArray(), width, height);

    [Fact]
    public void FiguresInsideTheTableAreTokenized()
    {
        LayoutBox table = Box("table", 0, 0, 400, 300);
        LayoutBox[] regions = [table, Box("image", 50, 50, 150, 150)];
        string?[] paths = [null, "image_1_50_50.png"];

        using RgbImage crop = Grey(400, 300);
        (RgbImage painted, IReadOnlyList<TokenizedFigure> figures) =
            TableFigureTokenizer.Tokenize(crop, table, regions, paths);

        using (painted)
        {
            TokenizedFigure figure = Assert.Single(figures);
            Assert.Equal(1, figure.RegionIndex);
            Assert.StartsWith("[F", figure.Token);
            Assert.Equal("image_1_50_50.png", figure.Path);

            // The figure's area was painted over: its corner is no longer the original grey.
            Assert.NotEqual(128, painted.Row(55)[52 * 3]);
        }
    }

    [Fact]
    public void FiguresOutsideTheTableAreLeftAlone()
    {
        LayoutBox table = Box("table", 0, 0, 200, 200);
        LayoutBox[] regions = [table, Box("image", 300, 300, 400, 400)];
        string?[] paths = [null, "image_1_300_300.png"];

        using RgbImage crop = Grey(200, 200);
        (RgbImage painted, IReadOnlyList<TokenizedFigure> figures) =
            TableFigureTokenizer.Tokenize(crop, table, regions, paths);

        using (painted)
        {
            Assert.Empty(figures);
            Assert.Equal(128, painted.Row(10)[10 * 3]);
        }
    }

    [Fact]
    public void TinyFiguresAreBlankedButNotTokenized()
    {
        LayoutBox table = Box("table", 0, 0, 200, 200);
        LayoutBox[] regions = [table, Box("image", 20, 20, 35, 35)];
        string?[] paths = [null, "image_1_20_20.png"];

        using RgbImage crop = Grey(200, 200);
        (RgbImage painted, IReadOnlyList<TokenizedFigure> figures) =
            TableFigureTokenizer.Tokenize(crop, table, regions, paths);

        using (painted)
        {
            Assert.Empty(figures);
            Assert.Equal(255, painted.Row(25)[25 * 3]);
        }
    }

    [Fact]
    public void TokenNumbersAvoidConfusableDigits()
    {
        LayoutBox table = Box("table", 0, 0, 1000, 1000);
        var regions = new List<LayoutBox> { table };
        var paths = new List<string?> { null };

        for (int i = 0; i < 12; i++)
        {
            regions.Add(Box("image", 10 + (i * 60), 10, 60 + (i * 60), 60));
            paths.Add($"image_{i}.png");
        }

        using RgbImage crop = Grey(1000, 1000);
        (RgbImage painted, IReadOnlyList<TokenizedFigure> figures) =
            TableFigureTokenizer.Tokenize(crop, table, regions, paths);

        using (painted)
        {
            Assert.Equal(12, figures.Count);
            foreach (TokenizedFigure figure in figures)
            {
                string digits = figure.Token[2..^1];
                Assert.DoesNotContain('0', digits);
                Assert.DoesNotContain('1', digits);
                Assert.DoesNotContain('9', digits);
            }

            Assert.Equal(figures.Count, figures.Select(figure => figure.Token).Distinct().Count());
        }
    }

    [Fact]
    public void PlaceholdersBecomeImageReferences()
    {
        IReadOnlyList<TokenizedFigure> figures = [new TokenizedFigure("[F23]", 4, "image_4_10_20.png")];

        string html = TableFigureTokenizer.Untokenize(
            "<table><tr><td>[F23]</td><td>x</td></tr></table>", figures, "imgs");

        Assert.Contains("<img src=\"imgs/image_4_10_20.png\" alt=\"Image\" />", html);
        Assert.DoesNotContain("[F23]", html);
    }

    [Fact]
    public void UnknownPlaceholdersAreLeftInPlace()
    {
        IReadOnlyList<TokenizedFigure> figures = [new TokenizedFigure("[F23]", 4, "a.png")];

        string html = TableFigureTokenizer.Untokenize("<td>[F44]</td>", figures, "imgs");

        Assert.Equal("<td>[F44]</td>", html);
    }
}
