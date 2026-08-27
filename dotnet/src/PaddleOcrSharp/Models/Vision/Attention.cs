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
        // Wide enough that a block amortises the key and value traffic, narrow enough that the
        // scores stay well inside L2 (16 x 5120 floats is 320 KB at the largest page we accept).
        const int Block = 16;

        Parallel.For(0, heads, head =>
        {
            int headOffset = head * tokens * headDim;

            ReadOnlyMemory<float> k = keys.Slice(headOffset, tokens * headDim);
            ReadOnlyMemory<float> v = values.Slice(headOffset, tokens * headDim);

            using PooledBuffer buffer = TensorPool.Rent(Block * tokens);
            Memory<float> scores = buffer.Memory;

            for (int start = 0; start < tokens; start += Block)
            {
                int rows = Math.Min(Block, tokens - start);
                Memory<float> block = scores[..(rows * tokens)];

                // Keys are the reduction vectors, so the score block is a product against a
                // transposed right-hand operand.
                Gemm.MatMul(
                    queries.Slice(headOffset + (start * headDim), rows * headDim),
                    rows,
                    headDim,
                    transposeA: false,
                    k,
                    tokens,
                    transposeB: true,
                    block,
                    allowParallel: false);

                Span<float> rowsSpan = block.Span;
                for (int row = 0; row < rows; row++)
                {
                    Span<float> scoreRow = rowsSpan.Slice(row * tokens, tokens);
                    Kernels.Scale(scoreRow, scale);
                    Kernels.Softmax(scoreRow);
                }

                // Values are indexed by key, which is the reduction axis here, so this one is a
                // direct product — no transpose of the values needed.
                Gemm.MatMul(
                    block,
                    rows,
                    tokens,
                    transposeA: false,
                    v,
                    headDim,
                    transposeB: false,
                    output.Slice(headOffset + (start * headDim), rows * headDim),
                    allowParallel: false);
            }
        });
    }
}
