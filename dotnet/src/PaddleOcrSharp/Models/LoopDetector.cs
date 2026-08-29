namespace PaddleOcrSharp.Models;

/// <summary>
/// Watches a greedy decode for the point at which it has provably fallen into a cycle.
/// </summary>
/// <remarks>
/// <para>
/// A VL model asked to read a crop it cannot parse - a display formula rendered small, a figure
/// caught by the layout model as text - sometimes stops converging and emits the same phrase until
/// the token budget runs out. <c>RepetitionTruncator</c> already removes that tail from the string,
/// so every token after the cycle established itself is thrown away; the only thing generating them
/// buys is time, and because attention re-reads the whole key/value cache each step, the tokens that
/// are certain to be discarded are also the most expensive ones on the page.
/// </para>
/// <para>
/// Detection is exact rather than statistical: decoding is greedy, so the check is simply whether
/// the tail is <see cref="Repeats"/> verbatim copies of one period. The thresholds are deliberately
/// well past what upstream's truncator needs to fire - it acts on five repeats of an eight-character
/// unit - so anything stopped here would have been cut from the string anyway. The minimum token
/// count keeps every ordinary block, whose output is a few hundred tokens, out of the check
/// entirely.
/// </para>
/// </remarks>
public sealed class LoopDetector
{
    private readonly bool _enabled;
    private readonly int _minimumTokens;
    private readonly int _maximumPeriod;
    private readonly int _repeats;
    private int _nextCheck;

    /// <summary>Creates a detector from a call's decoding settings.</summary>
    /// <param name="options">Settings whose repetition knobs configure the detector.</param>
    public LoopDetector(GenerationOptions options)
    {
        _enabled = options.StopOnRepetition;
        _minimumTokens = options.RepetitionMinimumTokens;
        _maximumPeriod = options.RepetitionMaximumPeriod;
        _repeats = options.RepetitionRepeats;
        _nextCheck = _minimumTokens;
    }

    /// <summary>Number of verbatim repetitions of one period that count as a cycle.</summary>
    public int Repeats => _repeats;

    /// <summary>
    /// Whether <paramref name="generated"/> now ends in a cycle, and so nothing further is worth
    /// decoding.
    /// </summary>
    /// <param name="generated">Tokens produced so far, most recent last.</param>
    public bool IsLooping(List<int> generated)
    {
        if (!_enabled || generated.Count < _nextCheck)
        {
            return false;
        }

        // Scanning every period on every token would cost more than the check saves on the blocks
        // that never loop, and a cycle that is real stays real: re-checking every 32 tokens finds it
        // within 32 tokens of the earliest possible detection.
        _nextCheck = generated.Count + CheckInterval;

        ReadOnlySpan<int> tokens = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(generated);

        for (int period = 1; period <= _maximumPeriod; period++)
        {
            if (period * _repeats > tokens.Length)
            {
                break;
            }

            if (EndsWithRepeats(tokens, period, _repeats))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>Tokens between successive scans once the minimum length is reached.</summary>
    public const int CheckInterval = 32;

    /// <summary>
    /// Whether the last <paramref name="repeats"/> windows of <paramref name="period"/> tokens are
    /// all identical.
    /// </summary>
    /// <param name="tokens">Token stream, most recent last.</param>
    /// <param name="period">Candidate cycle length.</param>
    /// <param name="repeats">Number of copies required.</param>
    public static bool EndsWithRepeats(ReadOnlySpan<int> tokens, int period, int repeats)
    {
        ReadOnlySpan<int> unit = tokens[^period..];

        for (int i = 1; i < repeats; i++)
        {
            int start = tokens.Length - ((i + 1) * period);
            if (!tokens.Slice(start, period).SequenceEqual(unit))
            {
                return false;
            }
        }

        return true;
    }
}
