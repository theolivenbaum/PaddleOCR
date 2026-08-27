using System.Numerics.Tensors;
using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Models;

/// <summary>
/// Token selection: greedy by default, with optional repetition penalty, temperature and
/// nucleus filtering for callers that want to mirror the server-side backends.
/// </summary>
public sealed class Sampler
{
    private readonly GenerationOptions _options;
    private readonly Random _random;

    /// <summary>Creates a sampler for one generation.</summary>
    public Sampler(GenerationOptions options)
    {
        _options = options;
        _random = new Random(options.Seed);
    }

    /// <summary>
    /// Picks the next token from <paramref name="logits"/>.
    /// </summary>
    /// <param name="logits">Vocabulary logits; modified in place when penalties apply.</param>
    /// <param name="history">Tokens generated so far, for the repetition penalty.</param>
    public int Select(Span<float> logits, IReadOnlyList<int> history)
    {
        if (_options.RepetitionPenalty != 1f && history.Count > 0)
        {
            ApplyRepetitionPenalty(logits, history, _options.RepetitionPenalty);
        }

        if (_options.IsGreedy)
        {
            return TensorPrimitives.IndexOfMax(logits);
        }

        if (_options.Temperature != 1f)
        {
            TensorPrimitives.Divide(logits, _options.Temperature, logits);
        }

        Kernels.Softmax(logits);
        return _options.TopP is > 0f and < 1f ? SampleNucleus(logits, _options.TopP) : SampleFull(logits);
    }

    /// <summary>
    /// Divides positive logits by the penalty and multiplies negative ones, matching
    /// <c>transformers</c>' <c>RepetitionPenaltyLogitsProcessor</c>.
    /// </summary>
    private static void ApplyRepetitionPenalty(Span<float> logits, IReadOnlyList<int> history, float penalty)
    {
        foreach (int token in history)
        {
            if ((uint)token >= (uint)logits.Length)
            {
                continue;
            }

            float value = logits[token];
            logits[token] = value > 0f ? value / penalty : value * penalty;
        }
    }

    private int SampleFull(ReadOnlySpan<float> probabilities)
    {
        float target = (float)_random.NextDouble();
        float cumulative = 0f;
        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (cumulative >= target)
            {
                return i;
            }
        }

        return probabilities.Length - 1;
    }

    private int SampleNucleus(ReadOnlySpan<float> probabilities, float topP)
    {
        int[] order = new int[probabilities.Length];
        for (int i = 0; i < order.Length; i++)
        {
            order[i] = i;
        }

        float[] copy = probabilities.ToArray();
        Array.Sort(copy, order);
        Array.Reverse(copy);
        Array.Reverse(order);

        float mass = 0f;
        int cutoff = 0;
        while (cutoff < copy.Length)
        {
            mass += copy[cutoff];
            cutoff++;
            if (mass >= topP)
            {
                break;
            }
        }

        float target = (float)(_random.NextDouble() * mass);
        float cumulative = 0f;
        for (int i = 0; i < cutoff; i++)
        {
            cumulative += copy[i];
            if (cumulative >= target)
            {
                return order[i];
            }
        }

        return order[cutoff - 1];
    }
}
