namespace PaddleOcrSharp.Models;

/// <summary>What one call to <see cref="PaddleOcrVLModel.Generate(int[], Core.Tensor, Imaging.ImageGrid, GenerationOptions, out GenerationStats, System.Threading.CancellationToken)"/> cost.</summary>
/// <param name="PromptTokens">Prompt length, image placeholders included.</param>
/// <param name="GeneratedTokens">Tokens produced, the stop token excluded.</param>
/// <param name="HitTokenBudget">Whether the loop ended on <see cref="GenerationOptions.MaxNewTokens"/>.</param>
/// <param name="StoppedEarly">Whether the loop ended because the output had fallen into a cycle.</param>
/// <param name="Prefill">Time in the prompt pass.</param>
/// <param name="Decode">Time in the token loop.</param>
/// <param name="DecodeLogits">Part of <paramref name="Decode"/> spent projecting the hidden state onto the vocabulary.</param>
public readonly record struct GenerationStats(
    int PromptTokens,
    int GeneratedTokens,
    bool HitTokenBudget,
    bool StoppedEarly,
    TimeSpan Prefill,
    TimeSpan Decode,
    TimeSpan DecodeLogits)
{
    /// <summary>
    /// Part of <see cref="Decode"/> spent in the 18 decoder layers rather than the output head.
    /// </summary>
    public TimeSpan DecodeLayers => Decode - DecodeLogits;
}
