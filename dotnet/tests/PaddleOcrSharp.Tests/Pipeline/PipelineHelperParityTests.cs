using System.Text;
using System.Text.Json;
using PaddleOcrSharp.Formats;
using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models.Layout;
using PaddleOcrSharp.Pipeline;
using PaddleOcrSharp.Tests.Fixtures;

namespace PaddleOcrSharp.Tests.Pipeline;

/// <summary>
/// The four pure helpers on the default path — <c>filter_overlap_boxes</c>,
/// <c>convert_otsl_to_html</c>, <c>truncate_repetitive_content</c> and <c>crop_margin</c> —
/// against the upstream functions themselves. Fixtures come from
/// <c>dotnet/tools/reference/dump_pipeline_helpers.py</c>.
/// </summary>
public class PipelineHelperParityTests
{
    private const string FixtureName = "pipeline_helpers.npz";

    private sealed record Cases(
        OverlapCase[] Overlaps,
        TableCase[] Tables,
        TruncationCase[] Truncations,
        MarginCase[] Margins);

    private sealed record MarginCase(string Name, string InputKey, string OutputKey);

    private sealed record OverlapCase(
        string Name, string ShapeMode, FixtureBox[] Boxes, float[][] Kept, string[] KeptLabels);

    private sealed record FixtureBox(string Label, float[] Coordinate, float[][]? PolygonPoints);

    private sealed record TableCase(string Name, string Otsl, string Html);

    private sealed record TruncationCase(string Name, string Content, string Truncated);

    private static (Cases Cases, Dictionary<string, NpyArray> Arrays) Load()
    {
        Fixture.RequireOrSkip(FixtureName);
        Dictionary<string, NpyArray> arrays = Fixture.Load(FixtureName);
        return (JsonSerializer.Deserialize<Cases>(
            Encoding.UTF8.GetString(arrays["cases"].ToBytes()),
            new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower })!,
            arrays);
    }

    [Fact]
    public void OverlapFilterKeepsWhatUpstreamKeeps()
    {
        Cases cases = Load().Cases;
        Assert.NotEmpty(cases.Overlaps);
        var failures = new List<string>();

        foreach (OverlapCase testCase in cases.Overlaps)
        {
            LayoutBox[] boxes =
            [
                .. testCase.Boxes.Select((box, index) => new LayoutBox(
                    0,
                    box.Label,
                    0.9f,
                    box.Coordinate[0],
                    box.Coordinate[1],
                    box.Coordinate[2],
                    box.Coordinate[3],
                    index)
                {
                    // `rect` mode never attaches outlines, so the polygon arm of the rule cannot
                    // fire there; the detector reproduces that by leaving the property null.
                    Polygon = testCase.ShapeMode == "rect" || box.PolygonPoints is null
                        ? null
                        : [.. box.PolygonPoints.Select(point => (point[0], point[1]))],
                }),
            ];

            List<LayoutBox> kept = OverlapFilter.Apply(boxes);

            string[] actualLabels = [.. kept.Select(box => box.Label)];
            if (!actualLabels.SequenceEqual(testCase.KeptLabels))
            {
                failures.Add($"{testCase.Name}: kept [{string.Join(", ", actualLabels)}], "
                    + $"expected [{string.Join(", ", testCase.KeptLabels)}]");
                continue;
            }

            for (int i = 0; i < kept.Count; i++)
            {
                float[] expected = testCase.Kept[i];
                if (kept[i].Left != expected[0] || kept[i].Top != expected[1]
                    || kept[i].Right != expected[2] || kept[i].Bottom != expected[3])
                {
                    failures.Add($"{testCase.Name}: box {i} is "
                        + $"({kept[i].Left}, {kept[i].Top}, {kept[i].Right}, {kept[i].Bottom}), "
                        + $"expected ({string.Join(", ", expected)})");
                }
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public void OtslConversionMatchesUpstream()
    {
        Cases cases = Load().Cases;
        Assert.NotEmpty(cases.Tables);
        var failures = new List<string>();

        foreach (TableCase testCase in cases.Tables)
        {
            string html = OtslTable.ToHtml(testCase.Otsl);
            if (html != testCase.Html)
            {
                failures.Add($"{testCase.Name}:\n  got      {html}\n  expected {testCase.Html}");
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public void RepetitionTruncationMatchesUpstream()
    {
        Cases cases = Load().Cases;
        Assert.NotEmpty(cases.Truncations);
        var failures = new List<string>();

        foreach (TruncationCase testCase in cases.Truncations)
        {
            // The dump calls upstream with its own defaults; the pipeline passes a lower
            // threshold, so the parity check uses the signature the dump exercised.
            string truncated = RepetitionTruncator.Truncate(testCase.Content, minimumLength: 3000);
            if (truncated != testCase.Truncated)
            {
                failures.Add($"{testCase.Name}: {testCase.Content.Length} characters became "
                    + $"{truncated.Length}, expected {testCase.Truncated.Length}"
                    + (truncated.Length < 200 && testCase.Truncated.Length < 200
                        ? $"\n  got      {Escape(truncated)}\n  expected {Escape(testCase.Truncated)}"
                        : string.Empty));
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public void FormulaMarginTrimmingMatchesUpstream()
    {
        (Cases cases, Dictionary<string, NpyArray> arrays) = Load();
        Assert.NotEmpty(cases.Margins);
        var failures = new List<string>();

        foreach (MarginCase testCase in cases.Margins)
        {
            NpyArray source = arrays[testCase.InputKey];
            NpyArray expected = arrays[testCase.OutputKey];

            // Our port folds in the caller's `w > 2 and h > 2` test, so a crop upstream would
            // hand back thinner than three pixels is one we decline to make at all.
            bool tooThin = expected.Shape[0] <= 2 || expected.Shape[1] <= 2;
            int expectedWidth = tooThin ? source.Shape[1] : expected.Shape[1];
            int expectedHeight = tooThin ? source.Shape[0] : expected.Shape[0];
            byte[] expectedPixels = (tooThin ? source : expected).ToBytes();

            using RgbImage image = Decode(source);
            using RgbImage cropped = DocumentParser.CropMargin(image);

            if (cropped.Width != expectedWidth || cropped.Height != expectedHeight)
            {
                failures.Add($"{testCase.Name}: cropped to {cropped.Width}x{cropped.Height}, "
                    + $"expected {expectedWidth}x{expectedHeight}");
                continue;
            }

            for (int y = 0; y < cropped.Height; y++)
            {
                if (!cropped.Row(y).SequenceEqual(
                    expectedPixels.AsSpan(y * cropped.Width * 3, cropped.Width * 3)))
                {
                    failures.Add($"{testCase.Name}: pixels differ on row {y}");
                    break;
                }
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    private static RgbImage Decode(NpyArray array)
    {
        RgbImage image = RgbImage.Rent(array.Shape[1], array.Shape[0]);
        array.ToBytes().AsSpan(0, array.Shape[0] * array.Shape[1] * 3).CopyTo(image.Pixels);
        return image;
    }

    private static string Escape(string value) => value.Replace("\n", "\\n");
}
