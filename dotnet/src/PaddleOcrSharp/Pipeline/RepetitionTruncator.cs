namespace PaddleOcrSharp.Pipeline;

/// <summary>
/// Trims the runaway repetition a VL model occasionally falls into.
/// </summary>
/// <remarks>
/// Port of <c>truncate_repetitive_content</c>. Three patterns are recognised, in the order
/// upstream checks them: a repeating phrase at the end of one long line, a whole line that is one
/// short unit repeated, and a single line repeated across most of a multi-line result.
/// </remarks>
public static class RepetitionTruncator
{
    /// <summary>Lower bound on output length before any check runs.</summary>
    public const int DefaultMinimumLength = 3000;

    /// <summary>The bound the pipeline uses for table blocks, whose output is legitimately long.</summary>
    public const int TableMinimumLength = 5000;

    /// <summary>The bound the pipeline uses for every other block.</summary>
    public const int BlockMinimumLength = 50;

    /// <summary>
    /// Returns <paramref name="content"/> with detected repetition removed.
    /// </summary>
    /// <param name="content">Model output.</param>
    /// <param name="minimumLength">Length below which the content is returned untouched.</param>
    /// <param name="lineThreshold">Minimum line count before the line-level check applies.</param>
    /// <param name="characterThreshold">Minimum repeat count for the whole-line check.</param>
    /// <param name="minimumUnitLength">Minimum length for the whole-line check.</param>
    public static string Truncate(
        string content,
        int minimumLength = DefaultMinimumLength,
        int lineThreshold = 10,
        int characterThreshold = 10,
        int minimumUnitLength = 10)
    {
        if (content.Length < minimumLength)
        {
            return content;
        }

        string stripped = content.Trim();
        if (stripped.Length == 0)
        {
            return content;
        }

        bool singleLine = !stripped.Contains('\n');

        if (singleLine && stripped.Length > 100
            && FindRepeatingSuffix(stripped, minimumUnit: 8, minimumRepeats: 5) is var suffix
            && suffix is not null)
        {
            (string prefix, string unit, int count) = suffix.Value;
            if ((long)unit.Length * count > stripped.Length * 0.5)
            {
                return prefix;
            }
        }

        if (singleLine && stripped.Length > minimumUnitLength
            && FindShortestRepeatingUnit(stripped) is { } repeatingUnit
            && stripped.Length / repeatingUnit.Length >= characterThreshold)
        {
            return repeatingUnit;
        }

        string[] lines = [.. content
            .Split('\n')
            .Select(line => line.Trim())
            .Where(line => line.Length > 0)];

        if (lines.Length < lineThreshold)
        {
            return content;
        }

        (string line, int count) mostCommon = lines
            .GroupBy(line => line, StringComparer.Ordinal)
            .Select(group => (line: group.Key, count: group.Count()))
            .MaxBy(entry => entry.count);

        return mostCommon.count >= lineThreshold && (double)mostCommon.count / lines.Length >= 0.8
            ? mostCommon.line
            : content;
    }

    /// <summary>The shortest unit whose repetition reproduces <paramref name="value"/> exactly.</summary>
    public static string? FindShortestRepeatingUnit(string value)
    {
        for (int length = 1; length <= value.Length / 2; length++)
        {
            if (value.Length % length != 0)
            {
                continue;
            }

            ReadOnlySpan<char> unit = value.AsSpan(0, length);
            bool matches = true;
            for (int offset = length; offset < value.Length && matches; offset += length)
            {
                matches = value.AsSpan(offset, length).SequenceEqual(unit);
            }

            if (matches)
            {
                return value[..length];
            }
        }

        return null;
    }

    /// <summary>
    /// Finds a phrase repeated at the end of <paramref name="value"/>.
    /// </summary>
    /// <returns>The prefix before the repetition, the repeated unit and its count.</returns>
    public static (string Prefix, string Unit, int Count)? FindRepeatingSuffix(
        string value,
        int minimumUnit = 8,
        int minimumRepeats = 5)
    {
        for (int length = value.Length / minimumRepeats; length >= minimumUnit; length--)
        {
            string unit = value[^length..];
            if (!EndsWithRepeats(value, unit, minimumRepeats))
            {
                continue;
            }

            int count = 0;
            int end = value.Length;
            while (end >= length && value.AsSpan(end - length, length).SequenceEqual(unit))
            {
                end -= length;
                count++;
            }

            return (value[..(value.Length - (count * length))], unit, count);
        }

        return null;
    }

    private static bool EndsWithRepeats(string value, string unit, int repeats)
    {
        if ((long)unit.Length * repeats > value.Length)
        {
            return false;
        }

        for (int i = 0; i < repeats; i++)
        {
            int start = value.Length - ((i + 1) * unit.Length);
            if (!value.AsSpan(start, unit.Length).SequenceEqual(unit))
            {
                return false;
            }
        }

        return true;
    }
}
