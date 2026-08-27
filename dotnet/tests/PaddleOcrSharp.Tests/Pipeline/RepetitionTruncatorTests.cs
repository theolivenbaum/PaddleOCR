using PaddleOcrSharp.Pipeline;

namespace PaddleOcrSharp.Tests.Pipeline;

/// <summary>Checks the repetition guard against the three patterns upstream recognises.</summary>
public class RepetitionTruncatorTests
{
    [Fact]
    public void ShortContentIsUntouched()
    {
        const string Content = "a short line";
        Assert.Equal(Content, RepetitionTruncator.Truncate(Content, minimumLength: 50));
    }

    [Fact]
    public void RepeatingSuffixIsTrimmed()
    {
        string content = "The real content ends here. " + string.Concat(Enumerable.Repeat("repeat me now ", 60));

        string result = RepetitionTruncator.Truncate(content, minimumLength: 50);

        Assert.StartsWith("The real content ends here.", result);
        Assert.True(result.Length < content.Length / 2);
    }

    [Fact]
    public void WholeLineRepetitionCollapsesToItsUnit()
    {
        // Short enough to skip the suffix rule (which needs more than 100 characters) so the
        // whole-string repetition rule is the one that fires.
        string content = string.Concat(Enumerable.Repeat("abcde", 12));

        string result = RepetitionTruncator.Truncate(content, minimumLength: 50);

        Assert.Equal("abcde", result);
    }

    [Fact]
    public void ALineThatIsNothingButRepetitionCollapsesToEmpty()
    {
        // Upstream's suffix rule runs first on long single lines: the whole string is the repeated
        // unit, so the prefix before the repetition is empty and that is what it returns.
        string content = string.Concat(Enumerable.Repeat("abcdefghij", 40));

        Assert.Equal(string.Empty, RepetitionTruncator.Truncate(content, minimumLength: 50));
    }

    [Fact]
    public void DominantRepeatedLineCollapses()
    {
        string content = string.Join('\n', Enumerable.Repeat("the same line again", 40));

        string result = RepetitionTruncator.Truncate(content, minimumLength: 50);

        Assert.Equal("the same line again", result);
    }

    [Fact]
    public void MixedContentSurvives()
    {
        string content = string.Join('\n', Enumerable.Range(0, 40).Select(i => $"line number {i}"));

        string result = RepetitionTruncator.Truncate(content, minimumLength: 50);

        Assert.Equal(content, result);
    }

    [Theory]
    [InlineData("abab", "ab")]
    [InlineData("xyzxyzxyz", "xyz")]
    [InlineData("abcd", null)]
    public void ShortestRepeatingUnitIsFound(string input, string? expected) =>
        Assert.Equal(expected, RepetitionTruncator.FindShortestRepeatingUnit(input));
}
