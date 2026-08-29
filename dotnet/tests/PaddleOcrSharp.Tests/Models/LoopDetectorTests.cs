using PaddleOcrSharp.Models;

namespace PaddleOcrSharp.Tests.Models;

/// <summary>
/// The early-stop check that keeps a runaway decode from generating tokens
/// <c>RepetitionTruncator</c> would only throw away again.
/// </summary>
/// <remarks>
/// Two properties matter and are tested separately: it must fire on a cycle, and it must not fire
/// on anything an ordinary block produces. The second is the one with teeth — stopping a block that
/// was still saying something new would silently shorten a real answer.
/// </remarks>
public class LoopDetectorTests
{
    private static readonly GenerationOptions Options = GenerationOptions.Default;

    private static List<int> Tokens(int count, Func<int, int> value)
    {
        var tokens = new List<int>(count);
        for (int i = 0; i < count; i++)
        {
            tokens.Add(value(i));
        }

        return tokens;
    }

    [Fact]
    public void DoesNotFireBelowTheMinimumTokenCount()
    {
        var detector = new LoopDetector(Options);

        // Pathologically repetitive, but shorter than any block worth checking.
        List<int> tokens = Tokens(Options.RepetitionMinimumTokens - 1, _ => 7);

        Assert.False(detector.IsLooping(tokens));
    }

    [Fact]
    public void FiresOnASingleTokenRepeatedPastTheMinimum()
    {
        var detector = new LoopDetector(Options);
        List<int> tokens = Tokens(Options.RepetitionMinimumTokens + 1, _ => 7);

        Assert.True(detector.IsLooping(tokens));
    }

    [Fact]
    public void FiresOnAMultiTokenPhraseRepeatedVerbatim()
    {
        var detector = new LoopDetector(Options);
        int[] phrase = [11, 22, 33, 44, 55];

        // A prefix of real content, then the same phrase over and over.
        List<int> tokens = Tokens(Options.RepetitionMinimumTokens, i => i);
        for (int i = 0; i < phrase.Length * Options.RepetitionRepeats; i++)
        {
            tokens.Add(phrase[i % phrase.Length]);
        }

        Assert.True(detector.IsLooping(tokens));
    }

    [Fact]
    public void DoesNotFireOnFewerRepeatsThanRequired()
    {
        var detector = new LoopDetector(Options);
        int[] phrase = [11, 22, 33, 44, 55];

        List<int> tokens = Tokens(Options.RepetitionMinimumTokens, i => i);
        for (int i = 0; i < phrase.Length * (Options.RepetitionRepeats - 1); i++)
        {
            tokens.Add(phrase[i % phrase.Length]);
        }

        Assert.False(detector.IsLooping(tokens));
    }

    [Fact]
    public void DoesNotFireOnLongNonRepeatingOutput()
    {
        var detector = new LoopDetector(Options);

        // Four thousand distinct tokens: a long table, not a loop.
        List<int> tokens = Tokens(4000, i => i);

        Assert.False(detector.IsLooping(tokens));
    }

    [Fact]
    public void DoesNotFireOnAPeriodLongerThanTheBound()
    {
        var detector = new LoopDetector(Options);
        int period = Options.RepetitionMaximumPeriod + 1;

        List<int> tokens = Tokens(Options.RepetitionMinimumTokens, i => i);
        for (int i = 0; i < period * Options.RepetitionRepeats; i++)
        {
            tokens.Add(1000 + (i % period));
        }

        Assert.False(detector.IsLooping(tokens));
    }

    [Fact]
    public void IsDisabledWhenTheOptionIsOff()
    {
        var detector = new LoopDetector(Options with { StopOnRepetition = false });
        List<int> tokens = Tokens(Options.RepetitionMinimumTokens * 2, _ => 7);

        Assert.False(detector.IsLooping(tokens));
    }

    [Fact]
    public void RepeatedRowsOfAWideTableDoNotCountAsACycle()
    {
        // An OTSL table whose rows differ only in their last cell: the shape repeats, the content
        // does not, so the token stream never repeats verbatim.
        var detector = new LoopDetector(Options);
        var tokens = new List<int>();

        for (int row = 0; row < 400; row++)
        {
            tokens.AddRange([501, 502, 503, 504, 600 + row]);
        }

        Assert.False(detector.IsLooping(tokens));
    }

    [Fact]
    public void EndsWithRepeatsMatchesTheTailOnly()
    {
        ReadOnlySpan<int> tokens = [9, 9, 9, 1, 2, 1, 2, 1, 2];

        Assert.True(LoopDetector.EndsWithRepeats(tokens, period: 2, repeats: 3));
        Assert.False(LoopDetector.EndsWithRepeats(tokens, period: 2, repeats: 4));
        Assert.False(LoopDetector.EndsWithRepeats(tokens, period: 3, repeats: 2));
    }
}
