using PaddleOcrSharp.Formats;
using PaddleOcrSharp.Models.Layout;
using PaddleOcrSharp.Pipeline;
using PaddleOcrSharp.Tests.Fixtures;

namespace PaddleOcrSharp.Tests.Pipeline;

/// <summary>
/// The rendered markdown, against PaddleX's own <c>MarkdownConverter</c>. Fixtures come from
/// <c>dotnet/tools/reference/dump_markdown.py</c>, which drives the upstream renderer over the
/// same block list.
/// </summary>
public class MarkdownParityTests
{
    private const string FixtureName = "markdown.npz";

    [Theory]
    [InlineData("pretty")]
    [InlineData("plain")]
    [InlineData("pretty_formula_number")]
    [InlineData("pretty_ignore")]
    [InlineData("pretty_chart")]
    [InlineData("pretty_ocr_images")]
    public void RenderingMatchesUpstream(string name)
    {
        Fixture.RequireOrSkip(FixtureName);
        var fixtures = Fixture.Load(FixtureName);

        int pageWidth = (int)fixtures["page_width"].ToInt64()[0];
        string expected = ReadString(fixtures[name]);

        MarkdownOptions options = name switch
        {
            "plain" => MarkdownOptions.Default with { Pretty = false, IgnoredLabels = [] },
            "pretty_formula_number" => MarkdownOptions.Default with
            {
                ShowFormulaNumber = true, IgnoredLabels = [],
            },
            "pretty_ignore" => MarkdownOptions.Default with
            {
                IgnoredLabels = BlockLabels.MarkdownIgnored,
            },
            "pretty_chart" => MarkdownOptions.Default with { ChartsAsTables = true, IgnoredLabels = [] },
            "pretty_ocr_images" => MarkdownOptions.Default with
            {
                ShowImageText = true, ShowSealText = true, IgnoredLabels = [],
            },
            _ => MarkdownOptions.Default with { IgnoredLabels = [] },
        };

        string actual = MarkdownWriter.Write(Blocks(options), options, pageWidth);

        Assert.Equal(expected, actual);
    }

    /// <summary>The same blocks the reference dumper renders.</summary>
    private static ParsedBlock[] Blocks(MarkdownOptions options)
    {
        ParsedBlock Block(string label, string content, float x1, float y1, float x2, float y2,
            string? image = null) =>
            new(label, new LayoutBox(0, label, 1f, x1, y1, x2, y2, 0), content, 0)
            {
                ImagePath = image,
            };

        return
        [
            Block("doc_title", "A Document\nTitle", 40, 20, 760, 70),
            Block("paragraph_title", "2.1 Method", 40, 90, 400, 120),
            Block("text", "First line.\nSecond line.\n\nA new paragraph.", 40, 130, 760, 260),
            Block("abstract", "Abstract This paper describes a thing.", 40, 270, 760, 330),
            Block("content", "Chapter one-\nis here.\nChapter two.", 40, 340, 760, 400),
            Block("figure_title", "Figure 1. A caption\nthat wraps.", 200, 410, 600, 440),
            Block("image", string.Empty, 150, 450, 650, 700, "img_0.png"),
            Block("table", "<table><tr><th>A</th><td>1</td></tr></table>", 40, 710, 760, 800),
            Block("display_formula", "$$x = y + 1$$", 200, 810, 600, 850),
            Block("formula_number", "(1)", 700, 810, 760, 850),
            Block("reference", "References\n[1] Someone.", 40, 860, 760, 940),
            Block("algorithm", "\nAlgorithm 1\nstep\n", 40, 950, 760, 1010),
            Block("chart", "A|B\n1|2\n3|4", 150, 1020, 650, 1200, "chart_0.png"),
            Block("seal", "A seal", 600, 1210, 760, 1300, "seal_0.png"),
            Block("vertical_text", "Side\nnote", 10, 400, 35, 700),
            Block("header", "Running head", 40, 0, 760, 18),
            Block("footer", "Page 3", 40, 1310, 760, 1330),
            Block("number", "3", 740, 1310, 760, 1330),
            Block("aside_text", "Margin note", 770, 400, 800, 700),
            Block("footnote", "1. A footnote.", 40, 1290, 760, 1310),
            Block("vision_footnote", "Source: somewhere", 150, 700, 650, 720),
            Block("reference_content", "[2] Another.", 40, 940, 760, 960),
            Block("inline_formula", "$a$", 300, 130, 320, 150),
            Block("spotting", "Spotted text", 40, 130, 760, 260),
        ];
    }

    private static string ReadString(NpyArray array) =>
        System.Text.Encoding.UTF8.GetString(array.ToBytes());
}
