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
        DocumentFigure[] figuresOnPage = [new("a.jpg", Box("image", 50, 50, 150, 150))];

        using RgbImage crop = Grey(400, 300);
        (RgbImage painted, IReadOnlyList<TokenizedFigure> figures, IReadOnlyList<string> absorbed) =
            TableFigureTokenizer.Tokenize(crop, table, figuresOnPage);

        using (painted)
        {
            TokenizedFigure figure = Assert.Single(figures);
            Assert.StartsWith("[F", figure.Token);
            Assert.Equal("a.jpg", figure.Path);
            Assert.Equal(["a.jpg"], absorbed);

            // The figure's area was painted over: its corner is no longer the original grey.
            Assert.NotEqual(128, painted.Row(55)[52 * 3]);
        }
    }

    [Fact]
    public void FiguresOutsideTheTableAreLeftAlone()
    {
        LayoutBox table = Box("table", 0, 0, 200, 200);
        DocumentFigure[] figuresOnPage = [new("a.jpg", Box("image", 300, 300, 400, 400))];

        using RgbImage crop = Grey(200, 200);
        (RgbImage painted, IReadOnlyList<TokenizedFigure> figures, IReadOnlyList<string> absorbed) =
            TableFigureTokenizer.Tokenize(crop, table, figuresOnPage);

        using (painted)
        {
            Assert.Empty(figures);
            Assert.Empty(absorbed);
            Assert.Equal(128, painted.Row(10)[10 * 3]);
        }
    }

    [Fact]
    public void TinyFiguresAreBlankedButNotTokenized()
    {
        LayoutBox table = Box("table", 0, 0, 200, 200);
        DocumentFigure[] figuresOnPage = [new("a.jpg", Box("image", 20, 20, 35, 35))];

        using RgbImage crop = Grey(200, 200);
        (RgbImage painted, IReadOnlyList<TokenizedFigure> figures, IReadOnlyList<string> absorbed) =
            TableFigureTokenizer.Tokenize(crop, table, figuresOnPage);

        using (painted)
        {
            Assert.Empty(figures);

            // Covered over, and still removed from the page: a picture the table swallowed must
            // not also stand on its own.
            Assert.Equal(["a.jpg"], absorbed);
            Assert.Equal(255, painted.Row(25)[25 * 3]);
        }
    }

    [Fact]
    public void TokenNumbersAvoidConfusableDigits()
    {
        LayoutBox table = Box("table", 0, 0, 1000, 1000);
        var figuresOnPage = new List<DocumentFigure>();

        for (int i = 0; i < 12; i++)
        {
            figuresOnPage.Add(new DocumentFigure(
                $"img_{i}.jpg", Box("image", 10 + (i * 60), 10, 60 + (i * 60), 60)));
        }

        using RgbImage crop = Grey(1000, 1000);
        (RgbImage painted, IReadOnlyList<TokenizedFigure> figures, _) =
            TableFigureTokenizer.Tokenize(crop, table, figuresOnPage);

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
        IReadOnlyList<TokenizedFigure> figures =
            [new TokenizedFigure("[F23]", "img_in_image_box_10_20_60_80.jpg")];

        string html = TableFigureTokenizer.Untokenize(
            "<table><tr><td>[F23]</td><td>x</td></tr></table>", figures, "imgs");

        // The doubled quote after `Image` is upstream's own, and the port keeps it so the emitted
        // HTML matches what a consumer would get from PaddleX.
        Assert.Contains(
            "<img src=\"imgs/img_in_image_box_10_20_60_80.jpg\" alt=\"Image\"\" />", html);
        Assert.DoesNotContain("[F23]", html);
    }

    [Fact]
    public void AFiguresOwnTextFollowsItsImage()
    {
        IReadOnlyList<TokenizedFigure> figures =
            [new TokenizedFigure("[F23]", "img_in_image_box_10_20_60_80.jpg")];

        string html = TableFigureTokenizer.Untokenize(
            "<td>[F23]</td>", figures, "imgs", _ => "A caption");

        Assert.EndsWith("\" />\n\nA caption\n\n</td>", html);
    }

    [Fact]
    public void UnknownPlaceholdersAreLeftInPlace()
    {
        IReadOnlyList<TokenizedFigure> figures = [new TokenizedFigure("[F23]", "a.jpg")];

        string html = TableFigureTokenizer.Untokenize("<td>[F44]</td>", figures, "imgs");

        Assert.Equal("<td>[F44]</td>", html);
    }
}
