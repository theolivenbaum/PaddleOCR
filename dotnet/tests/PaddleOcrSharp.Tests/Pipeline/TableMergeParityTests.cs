using System.Text.Json;
using PaddleOcrSharp.Models.Layout;
using PaddleOcrSharp.Pipeline;
using PaddleOcrSharp.Tests.Fixtures;

namespace PaddleOcrSharp.Tests.Pipeline;

/// <summary>
/// Rejoining a table split across a page break, against <c>merge_table.py</c> itself. Fixtures
/// come from <c>dotnet/tools/reference/dump_table_merge.py</c>.
/// </summary>
public class TableMergeParityTests
{
    private const string FixtureName = "table_merge.npz";

    private sealed record Case(
        string Name, Block[] Previous, Block[] Current, bool CanMerge, string Merged);

    private sealed record Block(string Label, string Content, float[] Bbox);

    [Fact]
    public void MergeDecisionsAndOutputMatchUpstream()
    {
        Fixture.RequireOrSkip(FixtureName);
        var fixtures = Fixture.Load(FixtureName);

        Case[] cases = JsonSerializer.Deserialize<Case[]>(
            System.Text.Encoding.UTF8.GetString(fixtures["cases"].ToBytes()),
            new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower })!;

        Assert.NotEmpty(cases);
        var failures = new List<string>();

        foreach (Case testCase in cases)
        {
            var document = new ParsedDocument([
                Page(0, testCase.Previous),
                Page(1, testCase.Current),
            ]);

            ParsedDocument merged = document.MergeTablesAcrossPages();

            string firstTable = merged.Pages[0].Blocks.First(b => b.Label == "table").Content;
            string secondTable = merged.Pages[1].Blocks.First(b => b.Label == "table").Content;

            if (testCase.CanMerge)
            {
                if (firstTable != testCase.Merged)
                {
                    failures.Add($"{testCase.Name}: merged to\n  {firstTable}\nexpected\n  {testCase.Merged}");
                }

                if (secondTable.Length != 0)
                {
                    failures.Add($"{testCase.Name}: the absorbed table should be emptied");
                }
            }
            else
            {
                string original = testCase.Previous.First(b => b.Label == "table").Content;
                if (firstTable != original)
                {
                    failures.Add($"{testCase.Name}: merged when upstream would not");
                }
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    private static ParsedPage Page(int index, Block[] blocks) =>
        new(index, 800, 1000, [.. blocks.Select((block, i) => new ParsedBlock(
            block.Label,
            new LayoutBox(0, block.Label, 1f, block.Bbox[0], block.Bbox[1], block.Bbox[2], block.Bbox[3], i),
            block.Content,
            i))]);
}
