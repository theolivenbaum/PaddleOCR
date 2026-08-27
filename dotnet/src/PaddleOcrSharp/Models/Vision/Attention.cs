using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Models.Vision;

/// <summary>
/// Scaled dot-product attention over a single packed sequence, laid out per head.
/// </summary>
/// <remarks>
/// One score row is materialised at a time, so a long sequence never needs a full
/// <c>[tokens, tokens]</c> matrix — a 5120-patch page would be 100 MB per head. Work is split
/// across heads, and each head transposes its values once so the weighted sum is a long dot
/// product per output channel rather than a short scaled add per key.
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
    public static void Bidirectional(
        ReadOnlyMemory<float> queries,
        ReadOnlyMemory<float> keys,
        ReadOnlyMemory<float> values,
        Memory<float> output,
        int heads,
        int tokens,
        int headDim,
        float scale)
    {
        Parallel.For(0, heads, head =>
        {
            int headOffset = head * tokens * headDim;

            // Values are transposed once per head so the weighted sum becomes one long dot
            // product per output channel instead of a short scaled add per key.
            using PooledBuffer transposed = TensorPool.Rent(headDim * tokens);
            Span<float> valueColumns = transposed.Span;
            ReadOnlySpan<float> v = values.Span.Slice(headOffset, tokens * headDim);

            for (int token = 0; token < tokens; token++)
            {
                for (int d = 0; d < headDim; d++)
                {
                    valueColumns[(d * tokens) + token] = v[(token * headDim) + d];
                }
            }

            using PooledBuffer scores = TensorPool.Rent(tokens);
            Span<float> scoreRow = scores.Span;

            ReadOnlySpan<float> q = queries.Span.Slice(headOffset, tokens * headDim);
            ReadOnlySpan<float> k = keys.Span.Slice(headOffset, tokens * headDim);
            Span<float> o = output.Span.Slice(headOffset, tokens * headDim);

            for (int i = 0; i < tokens; i++)
            {
                ReadOnlySpan<float> queryRow = q.Slice(i * headDim, headDim);

                for (int j = 0; j < tokens; j++)
                {
                    scoreRow[j] = Gemm.Dot(queryRow, k.Slice(j * headDim, headDim)) * scale;
                }

                Kernels.Softmax(scoreRow);

                Span<float> outputRow = o.Slice(i * headDim, headDim);
                for (int d = 0; d < headDim; d++)
                {
                    outputRow[d] = Gemm.Dot(scoreRow, valueColumns.Slice(d * tokens, tokens));
                }
            }
        });
    }
}
