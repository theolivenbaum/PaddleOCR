using System.Text.RegularExpressions;

namespace PaddleOcrSharp.Pipeline;

/// <summary>One spotted text run and the quadrilateral it occupies.</summary>
/// <param name="Text">The recognised text.</param>
/// <param name="Polygon">Four corner points, in original-image pixel coordinates.</param>
public readonly record struct SpottedText(string Text, IReadOnlyList<(float X, float Y)> Polygon);

/// <summary>
/// Parses the model's spotting output, which interleaves text with quantised corner coordinates.
/// </summary>
/// <remarks>
/// <para>
/// The <c>Spotting:</c> instruction produces runs shaped as
/// <c>&lt;|TEXT_START|&gt;…&lt;|TEXT_END|&gt;&lt;|LOC_BEGIN|&gt;&lt;|LOC_x|&gt;…&lt;|LOC_END|&gt;</c>,
/// with eight <c>LOC</c> tokens per run giving four <c>(x, y)</c> corners on a 0–1000 grid. When the
/// delimiters are missing, upstream falls back to slicing the string on runs of eight bare
/// <c>LOC</c> tokens, taking the text between them.
/// </para>
/// <para>Port of <c>post_process_for_spotting</c>.</para>
/// </remarks>
public static partial class Spotting
{
    /// <summary>Below this size the crop is doubled before recognition, as upstream does.</summary>
    public const int UpscaleBelow = 1500;

    [GeneratedRegex(@"<\|TEXT_START\|>(.*?)<\|TEXT_END\|>", RegexOptions.Singleline)]
    private static partial Regex TextBlock();

    [GeneratedRegex(@"<\|LOC_BEGIN\|>(.*?)<\|LOC_END\|>", RegexOptions.Singleline)]
    private static partial Regex LocationBlock();

    [GeneratedRegex(@"<\|LOC_(\d+)\|>")]
    private static partial Regex LocationToken();

    /// <summary>
    /// Extracts the spotted runs and the plain-text rendering of the result.
    /// </summary>
    /// <param name="output">Raw model output.</param>
    /// <param name="width">Width of the image the model saw, in pixels.</param>
    /// <param name="height">Height of the image the model saw, in pixels.</param>
    public static (string Text, IReadOnlyList<SpottedText> Runs) Parse(string output, int width, int height)
    {
        var runs = new List<SpottedText>();

        MatchCollection texts = TextBlock().Matches(output);
        MatchCollection locations = LocationBlock().Matches(output);

        int paired = Math.Min(texts.Count, locations.Count);
        for (int i = 0; i < paired; i++)
        {
            MatchCollection items = LocationToken().Matches(locations[i].Groups[1].Value);
            if (items.Count < 8)
            {
                continue;
            }

            runs.Add(new SpottedText(
                texts[i].Groups[1].Value.Trim(),
                ToPolygon(items.Take(8).Select(match => int.Parse(match.Groups[1].Value)), width, height)));
        }

        if (runs.Count == 0)
        {
            Match[] tokens = [.. LocationToken().Matches(output).Cast<Match>()];
            int cursor = 0;
            for (int i = 0; i + 7 < tokens.Length; i += 8)
            {
                Match[] group = tokens[i..(i + 8)];
                string text = output[cursor..group[0].Index].Trim();
                runs.Add(new SpottedText(
                    text,
                    ToPolygon(group.Select(match => int.Parse(match.Groups[1].Value)), width, height)));
                cursor = group[^1].Index + group[^1].Length;
            }
        }

        return (string.Join("\n\n", runs.Select(run => run.Text)), runs);
    }

    private static (float X, float Y)[] ToPolygon(IEnumerable<int> values, int width, int height)
    {
        int[] coordinates = [.. values];
        var points = new (float X, float Y)[4];
        for (int i = 0; i < 4; i++)
        {
            points[i] = (
                coordinates[i * 2] / 1000f * width,
                coordinates[(i * 2) + 1] / 1000f * height);
        }

        return points;
    }
}
