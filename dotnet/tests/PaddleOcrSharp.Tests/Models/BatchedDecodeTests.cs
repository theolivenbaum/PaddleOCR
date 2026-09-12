using PaddleOcrSharp.Core;
using PaddleOcrSharp.Models;
using PaddleOcrSharp.Models.Language;
using PaddleOcrSharp.Tests.Fixtures;

namespace PaddleOcrSharp.Tests.Models;

/// <summary>
/// Pins the batched decode step against the single-sequence one.
/// </summary>
/// <remarks>
/// <c>ForwardBatch</c> steps several independent sequences through the decoder at once so their
/// shared weights are read once rather than once each. The sequences must not see each other: what
/// this checks is that a row of the batch is what that sequence would have got on its own, to the
/// precision the two paths can agree to. They are not bit-identical, because a batch of four or
/// more rows reaches <c>Gemm.Linear</c>'s widened-panel kernel where a single row reaches its
/// fused one, and the two group the same products differently.
/// </remarks>
[Collection(CheckpointCollection.Name)]
public class BatchedDecodeTests(CheckpointFixture checkpoint)
{
    [Fact]
    public void BatchedStepMatchesEachSequenceOnItsOwn()
    {
        checkpoint.RequireOrSkip();

        ModelConfiguration configuration = ModelConfiguration.Load(CheckpointFixture.Directory!);
        LanguageConfig language = configuration.Language;
        var decoder = new ErnieDecoder(checkpoint.Weights, language);

        int width = language.HiddenSize;
        int[][] prompts = [[1, 2, 3, 4, 5], [7, 8, 9], [11, 12, 13, 14, 15, 16]];
        var separate = new float[prompts.Length][];
        var caches = new KvCache[prompts.Length];
        var positions = new int[prompts.Length];

        try
        {
            for (int i = 0; i < prompts.Length; i++)
            {
                caches[i] = decoder.CreateCache(prompts[i].Length + 4);

                using Tensor prompt = Tensor.Rent(prompts[i].Length, width);
                decoder.Embed(prompts[i], prompt.Span);
                decoder.Forward(
                    prompt.Memory, prompts[i].Length, PositionIds.Sequential(prompts[i].Length), caches[i]);

                positions[i] = prompts[i].Length;
            }

            // Each sequence's next step, computed on its own against a clone of its cache.
            for (int i = 0; i < prompts.Length; i++)
            {
                using KvCache alone = CloneStateOf(caches[i], decoder, language);
                using Tensor step = Tensor.Rent(1, width);
                decoder.Embed([42], step.Span);
                decoder.Forward(step.Memory, 1, PositionIds.Sequential(1, positions[i]), alone);
                separate[i] = step.Span.ToArray();
            }

            // The same step for all three at once.
            using Tensor batch = Tensor.Rent(prompts.Length, width);
            for (int i = 0; i < prompts.Length; i++)
            {
                decoder.Embed([42], batch.Span.Slice(i * width, width));
            }

            decoder.ForwardBatch(batch.Memory, prompts.Length, positions, caches);

            for (int i = 0; i < prompts.Length; i++)
            {
                TensorAssert.Close(
                    separate[i],
                    batch.Span.Slice(i * width, width).ToArray(),
                    absoluteTolerance: 2e-3,
                    relativeTolerance: 2e-3,
                    because: $"row {i} of the batch");
            }
        }
        finally
        {
            foreach (KvCache cache in caches)
            {
                cache?.Dispose();
            }
        }
    }

    /// <summary>A cache holding the same positions as <paramref name="source"/>.</summary>
    private static KvCache CloneStateOf(KvCache source, ErnieDecoder decoder, LanguageConfig language)
    {
        var clone = new KvCache(language.NumHiddenLayers, language.KeyValueWidth, source.Length + 4);
        for (int layer = 0; layer < language.NumHiddenLayers; layer++)
        {
            source.KeyWindow(layer, source.Length).Span.CopyTo(clone.KeySlot(layer, 0, source.Length));
            source.ValueWindow(layer, source.Length).Span.CopyTo(clone.ValueSlot(layer, 0, source.Length));
        }

        clone.Advance(source.Length);
        return clone;
    }
}
