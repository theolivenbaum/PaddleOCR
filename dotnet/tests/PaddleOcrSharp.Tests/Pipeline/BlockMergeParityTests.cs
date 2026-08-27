using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using PaddleOcrSharp.Formats;
using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models.Layout;
using PaddleOcrSharp.Pipeline;
using PaddleOcrSharp.Tests.Fixtures;

namespace PaddleOcrSharp.Tests.Pipeline;

/// <summary>
/// Block grouping and image stacking against <c>merge_blocks</c> and <c>merge_images</c>
/// themselves. Fixtures come from <c>dotnet/tools/reference/dump_block_merge.py</c>, which drives
/// the upstream functions over ten layouts chosen to reach every branch: both alignments of the
/// paragraph-split-around-a-figure rule, the same geometry without the figure, both edges
/// aligned, a gap that is too large, the two-column rule and its gutter limit, a three-region
/// run, a run the aspect-ratio guard abandons, and a heading that is not text.
/// </summary>
/// <remarks>
/// Three things are compared, because all three reach the output: which regions were joined and
/// with what alignment, the order the regions come back in — a figure caught inside a run's span
/// is emitted after the run, not in its original place — and the stacked pixels.
/// </remarks>
public class BlockMergeParityTests
{
    private const string FixtureName = "block_merge.npz";

    private static readonly string[] NonMergeLabels =
        ["image", "header_image", "footer_image", "chart", "seal", "table"];

    private sealed record Case(string Name, InputBlock[] Blocks, ResultBlock[] Result);

    private sealed record InputBlock(string Label, int[] Box, int Seed, string ImageKey);

    private sealed record ResultBlock(
        string Label,
        int[] Box,
        int Seed,
        int? GroupId,
        [property: JsonPropertyName("merge_aligns")] string[]? MergeAligns,
        string? ImageKey);

    [Fact]
    public void GroupingOrderAndStackedPixelsMatchUpstream()
    {
        Fixture.RequireOrSkip(FixtureName);
        Dictionary<string, NpyArray> fixtures = Fixture.Load(FixtureName);

        Case[] cases = JsonSerializer.Deserialize<Case[]>(
            Encoding.UTF8.GetString(fixtures["cases"].ToBytes()),
            new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower })!;

        Assert.NotEmpty(cases);
        var failures = new List<string>();

        foreach (Case testCase in cases)
        {
            LayoutBox[] regions =
            [
                .. testCase.Blocks.Select((block, index) => new LayoutBox(
                    index,
                    block.Label,
                    0.9f,
                    block.Box[0],
                    block.Box[1],
                    block.Box[2],
                    block.Box[3],
                    index)),
            ];

            RgbImage[] crops = [.. testCase.Blocks.Select(block => Decode(fixtures[block.ImageKey]))];

            try
            {
                (int Width, int Height)[] sizes = [.. crops.Select(crop => (crop.Width, crop.Height))];
                List<BlockGroup> groups = BlockMerger.Group(regions, sizes, NonMergeLabels);

                Check(testCase, groups, crops, fixtures, failures);
            }
            finally
            {
                foreach (RgbImage crop in crops)
                {
                    crop.Dispose();
                }
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    private static void Check(
        Case testCase,
        List<BlockGroup> groups,
        RgbImage[] crops,
        Dictionary<string, NpyArray> fixtures,
        List<string> failures)
    {
        // Upstream returns one entry per region, groups first; ours returns one entry per group,
        // so flattening it has to reproduce the same sequence.
        int[] order = [.. groups.SelectMany(group => group.Indices)];
        int[] expectedOrder = [.. testCase.Result.Select(entry => entry.Seed)];
        int[] actualOrder = [.. order.Select(index => testCase.Blocks[index].Seed)];

        if (!expectedOrder.SequenceEqual(actualOrder))
        {
            failures.Add($"{testCase.Name}: order [{string.Join(", ", actualOrder)}], "
                + $"expected [{string.Join(", ", expectedOrder)}]");
            return;
        }

        int position = 0;
        foreach (BlockGroup group in groups)
        {
            ResultBlock head = testCase.Result[position];

            if (group.GroupId != head.GroupId)
            {
                failures.Add($"{testCase.Name}: group id {group.GroupId} for '{head.Label}', "
                    + $"expected {head.GroupId?.ToString() ?? "none"}");
            }

            string[] expectedAligns = head.MergeAligns ?? [];
            string[] actualAligns = [.. group.Alignments.Select(alignment => alignment switch
            {
                StackAlignment.Left => "left",
                StackAlignment.Right => "right",
                _ => "center",
            })];

            if (!expectedAligns.SequenceEqual(actualAligns))
            {
                failures.Add($"{testCase.Name}: alignments [{string.Join(", ", actualAligns)}], "
                    + $"expected [{string.Join(", ", expectedAligns)}]");
            }

            // Only the group's first region carries an image upstream; the rest are None.
            if (head.ImageKey is { } key)
            {
                using RgbImage stacked = ImageStacker.Stack(
                    [.. group.Indices.Select(index => crops[index])], group.Alignments);
                ComparePixels(testCase.Name, head.Label, stacked, fixtures[key], failures);
            }

            for (int i = 1; i < group.Indices.Count; i++)
            {
                if (testCase.Result[position + i].ImageKey is not null)
                {
                    failures.Add($"{testCase.Name}: upstream kept an image on a group's "
                        + $"{i + 1} member, so the group boundaries disagree");
                }
            }

            position += group.Indices.Count;
        }
    }

    private static void ComparePixels(
        string name, string label, RgbImage actual, NpyArray expected, List<string> failures)
    {
        if (expected.Shape[0] != actual.Height || expected.Shape[1] != actual.Width)
        {
            failures.Add($"{name}: '{label}' stacked to {actual.Width}x{actual.Height}, expected "
                + $"{expected.Shape[1]}x{expected.Shape[0]}");
            return;
        }

        byte[] pixels = expected.ToBytes();
        for (int y = 0; y < actual.Height; y++)
        {
            if (!actual.Row(y).SequenceEqual(pixels.AsSpan(y * actual.Width * 3, actual.Width * 3)))
            {
                failures.Add($"{name}: '{label}' stacked pixels differ on row {y}");
                return;
            }
        }
    }

    private static RgbImage Decode(NpyArray array)
    {
        RgbImage image = RgbImage.Rent(array.Shape[1], array.Shape[0]);
        array.ToBytes().AsSpan(0, array.Shape[0] * array.Shape[1] * 3).CopyTo(image.Pixels);
        return image;
    }
}
