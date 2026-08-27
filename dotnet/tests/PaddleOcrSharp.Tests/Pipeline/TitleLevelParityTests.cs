using System.Text.Json;
using PaddleOcrSharp.Models.Layout;
using PaddleOcrSharp.Pipeline;
using PaddleOcrSharp.Tests.Fixtures;

namespace PaddleOcrSharp.Tests.Pipeline;

/// <summary>
/// Heading levels, against <c>title_level.py</c> itself. Fixtures come from
/// <c>dotnet/tools/reference/dump_title_levels.py</c>.
/// </summary>
public class TitleLevelParityTests
{
    private const string FixtureName = "title_levels.npz";

    private sealed record Reference(Entry[] Entries, Dictionary<string, int> ClusterMap, Block[][] Pages);

    private sealed record Entry(string Content, int Height, string? Symbol, int SymbolLevel);

    private sealed record Block(string Label, string Content, float[] Bbox, int? TitleLevel);

    private static Reference Load()
    {
        Fixture.RequireOrSkip(FixtureName);
        var fixtures = Fixture.Load(FixtureName);

        return JsonSerializer.Deserialize<Reference>(
            System.Text.Encoding.UTF8.GetString(fixtures["data"].ToBytes()),
            new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower })!;
    }

    [Fact]
    public void NumberingStylesMatchUpstream()
    {
        Reference reference = Load();
        var failures = new List<string>();

        foreach (Entry entry in reference.Entries)
        {
            (string? symbol, int level) = TitleLevels.SymbolAndLevel(entry.Content);

            if (symbol != entry.Symbol || level != entry.SymbolLevel)
            {
                failures.Add($"'{entry.Content}': ({symbol}, {level}) != ({entry.Symbol}, {entry.SymbolLevel})");
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public void TextHeightsMatchUpstream()
    {
        Reference reference = Load();
        Block[] titles = [.. reference.Pages.SelectMany(page => page)
            .Where(block => block.Label == "paragraph_title")];

        Assert.Equal(reference.Entries.Length, titles.Length);

        for (int i = 0; i < titles.Length; i++)
        {
            ParsedBlock block = ToBlock(titles[i], i);
            Assert.Equal(reference.Entries[i].Height, TitleLevels.Height(block));
        }
    }

    [Fact]
    public void LevelsMatchUpstreamGivenItsOwnClustering()
    {
        // The clustering is the one part that cannot be reproduced exactly — upstream reaches for
        // scikit-learn's seeded KMeans — so it is supplied here and everything the rest of the
        // decision does is compared exactly.
        Reference reference = Load();
        Dictionary<int, int> clusters = reference.ClusterMap.ToDictionary(
            pair => int.Parse(pair.Key), pair => pair.Value);

        int[] levels = TitleLevels.Levels(
            [.. reference.Entries.Select(entry => entry.Content)],
            [.. reference.Entries.Select(entry => entry.Height)],
            clusters);

        int[] expected = [.. reference.Pages.SelectMany(page => page)
            .Where(block => block.Label == "paragraph_title")
            .Select(block => block.TitleLevel!.Value)];

        Assert.Equal(expected, levels);
    }

    [Fact]
    public void ClusteringAgreesWithScikitLearnOnTheReference()
    {
        // Not guaranteed in general, but the exact one-dimensional optimum and scikit-learn's
        // local search do agree on a document whose headings come in a few distinct sizes, which
        // is the shape a real document has.
        Reference reference = Load();
        IReadOnlyDictionary<int, int> clusters = TitleLevels.ClusterHeights(
            [.. reference.Entries.Select(entry => entry.Height)]);

        foreach ((string height, int level) in reference.ClusterMap)
        {
            Assert.Equal(level, clusters[int.Parse(height)]);
        }
    }

    [Fact]
    public void AssignedLevelsReachTheBlocks()
    {
        Reference reference = Load();

        var document = new ParsedDocument([.. reference.Pages.Select((page, index) =>
            new ParsedPage(index, 800, 1000, [.. page.Select(ToBlock)]))]);

        ParsedDocument levelled = document.AssignTitleLevels();

        int[] expected = [.. reference.Pages.SelectMany(page => page)
            .Where(block => block.Label == "paragraph_title")
            .Select(block => block.TitleLevel!.Value)];

        int[] actual = [.. levelled.Pages.SelectMany(page => page.Blocks)
            .Where(block => block.Label == "paragraph_title")
            .Select(block => block.TitleLevel!.Value)];

        Assert.Equal(expected, actual);

        // Everything that is not a heading is left without one.
        Assert.All(
            levelled.Pages.SelectMany(page => page.Blocks).Where(b => b.Label != "paragraph_title"),
            block => Assert.Null(block.TitleLevel));
    }

    private static ParsedBlock ToBlock(Block block, int index) =>
        new(block.Label,
            new LayoutBox(0, block.Label, 1f, block.Bbox[0], block.Bbox[1], block.Bbox[2], block.Bbox[3], index),
            block.Content,
            index);
}
