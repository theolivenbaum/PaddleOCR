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

    /// <summary>Whether decoding is greedy.</summary>
    public bool IsGreedy => Temperature <= 0f;
}
