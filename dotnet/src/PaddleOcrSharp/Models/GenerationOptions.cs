namespace PaddleOcrSharp.Models;

/// <summary>Decoding settings for one VL recognition call.</summary>
/// <remarks>
/// The local PaddleX backend generates greedily and ignores sampling parameters entirely
/// (<c>doc_vlm/predictor.py</c> warns and drops <c>temperature</c>, <c>top_p</c> and
/// <c>repetition_penalty</c>). The knobs exist here so the server-side behaviour can be matched
/// when a caller asks for it, but the defaults reproduce the local pipeline.
/// </remarks>
public sealed record GenerationOptions
{
    /// <summary>Defaults matching the local PaddleX backend: greedy, 8192 new tokens.</summary>
    public static GenerationOptions Default { get; } = new();

    /// <summary>Maximum tokens to generate.</summary>
    public int MaxNewTokens { get; init; } = 8192;

    /// <summary>Softmax temperature; <c>0</c> selects greedy decoding.</summary>
    public float Temperature { get; init; }

    /// <summary>Nucleus-sampling mass; <c>0</c> or <c>1</c> disables the filter.</summary>
    public float TopP { get; init; }

    /// <summary>Penalty applied to already-generated tokens; <c>1</c> disables it.</summary>
    public float RepetitionPenalty { get; init; } = 1f;

    /// <summary>Seed for sampling; ignored when decoding greedily.</summary>
    public int Seed { get; init; } = 0;

    /// <summary>Whether special tokens are removed from the decoded string.</summary>
    public bool SkipSpecialTokens { get; init; } = true;

    /// <summary>
    /// Whether decoding stops once the output has fallen into a verbatim cycle.
    /// </summary>
    /// <remarks>
    /// On by default. The tokens this skips are the ones <c>RepetitionTruncator</c> removes from the
    /// string afterwards, so the text a caller gets is the same; what changes is that a block whose
    /// decoder ran away no longer costs the whole token budget. Turn it off to reproduce upstream's
    /// generate-everything-then-trim behaviour exactly.
    /// </remarks>
    public bool StopOnRepetition { get; init; } = true;

    /// <summary>Tokens that must be generated before <see cref="StopOnRepetition"/> starts looking.</summary>
    /// <remarks>
    /// An ordinary block's whole output is shorter than this, so the check never sees it. The bound
    /// is what keeps legitimately repetitive content - a table column of equal values, a row of
    /// leader dots - out of the detector until the output is already longer than such content is.
    /// </remarks>
    public int RepetitionMinimumTokens { get; init; } = 384;

    /// <summary>Longest cycle <see cref="StopOnRepetition"/> looks for, in tokens.</summary>
    public int RepetitionMaximumPeriod { get; init; } = 64;

    /// <summary>Verbatim repetitions of one period that count as a cycle.</summary>
    public int RepetitionRepeats { get; init; } = 6;

    /// <summary>Whether decoding is greedy.</summary>
    public bool IsGreedy => Temperature <= 0f;
}
