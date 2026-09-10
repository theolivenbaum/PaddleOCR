using System.Diagnostics;
using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Models.Vision;

/// <summary>
/// Scaled dot-product attention over a single packed sequence, laid out per head.
/// </summary>
/// <remarks>
/// Scores are materialised a block of query rows at a time, so a long sequence never needs a
/// full <c>[tokens, tokens]</c> matrix — a 5120-patch page would be 100 MB per head — while the
/// block still gives the keys and values enough reuse to stay in cache across it. A row at a
/// time streamed the whole key matrix past for every single query. Work is split across heads.
/// </remarks>
public static class Attention
{
    /// <summary>
    /// Computes <c>softmax(Q · Kᵀ · scale) · V</c> for every head.
    /// </summary>
    /// <param name="queries">Queries as <c>[heads][tokens][headDim]</c>.</param>
    /// <param name="keys">Keys, same layout.</param>
    /// <param name="values">Values, same layout.</param>
    /// <param name="output">Receives the result in the same layout.</param>
    /// <param name="heads">Number of heads.</param>
    /// <param name="tokens">Sequence length.</param>
    /// <param name="headDim">Width of one head.</param>
    /// <param name="scale">Softmax scale, normally <c>1 / sqrt(headDim)</c>.</param>
    /// <param name="profile">
    /// Receives the parts' cost when supplied. Attention runs a thread per head, so nothing
    /// outside it can attribute its time; each thread accumulates its own and they are summed.
    /// </param>
    public static void Bidirectional(
        ReadOnlyMemory<float> queries,
        ReadOnlyMemory<float> keys,
        ReadOnlyMemory<float> values,
        Memory<float> output,
        int heads,
        int tokens,
        int headDim,
        float scale,
        StageProfile? profile = null)
    {
        // Wide enough that a block amortises the key and value traffic, narrow enough that the
        // scores stay well inside L2 (16 x 5120 floats is 320 KB at the largest page we accept).
        const int Block = 16;

        long transposeTicks = 0, scoreTicks = 0, softmaxTicks = 0, valueTicks = 0;

        Parallel.For(0, heads, Parallelism.Options, head =>
        {
            int headOffset = head * tokens * headDim;

            ReadOnlyMemory<float> k = keys.Slice(headOffset, tokens * headDim);
            ReadOnlyMemory<float> v = values.Slice(headOffset, tokens * headDim);

            using PooledBuffer buffer = TensorPool.Rent(Block * tokens);
            Memory<float> scores = buffer.Memory;

            // One transpose of this head's keys serves every row-block below, of which there are
            // `tokens / Block`. See AttentionKernels for why the score product wants them this
            // way round.
            using PooledBuffer keysT = TensorPool.Rent(tokens * headDim);
            long mark = Stopwatch.GetTimestamp();
            AttentionKernels.TransposeHead(k.Span, tokens, headDim, keysT.Span);
            long transpose = Stopwatch.GetTimestamp() - mark;
            long score = 0, soft = 0, value = 0;

            for (int start = 0; start < tokens; start += Block)
            {
                int rows = Math.Min(Block, tokens - start);
                Memory<float> block = scores[..(rows * tokens)];

                mark = Stopwatch.GetTimestamp();
                AttentionKernels.Scores(
                    queries.Span.Slice(headOffset + (start * headDim), rows * headDim),
                    rows,
                    headDim,
                    keysT.Span,
                    tokens,
                    block.Span);
                score += Stopwatch.GetTimestamp() - mark;

                mark = Stopwatch.GetTimestamp();
                Span<float> rowsSpan = block.Span;
                for (int row = 0; row < rows; row++)
                {
                    Span<float> scoreRow = rowsSpan.Slice(row * tokens, tokens);
                    Kernels.Scale(scoreRow, scale);
                    Kernels.Softmax(scoreRow);
                }

                soft += Stopwatch.GetTimestamp() - mark;
                mark = Stopwatch.GetTimestamp();

                AttentionKernels.Values(
                    block.Span,
                    rows,
                    tokens,
                    v.Span,
                    headDim,
                    output.Span.Slice(headOffset + (start * headDim), rows * headDim));
                value += Stopwatch.GetTimestamp() - mark;
            }

            if (profile is not null)
            {
                Interlocked.Add(ref transposeTicks, transpose);
                Interlocked.Add(ref scoreTicks, score);
                Interlocked.Add(ref softmaxTicks, soft);
                Interlocked.Add(ref valueTicks, value);
            }
        });

        if (profile is not null)
        {
            profile.AddThreadTicks("attention transpose", transposeTicks, heads);
            profile.AddThreadTicks("attention scores", scoreTicks, heads);
            profile.AddThreadTicks("attention softmax", softmaxTicks, heads);
            profile.AddThreadTicks("attention values", valueTicks, heads);
        }
    }
}
