using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Models.Vision;

/// <summary>
/// Scaled dot-product attention over a single packed sequence, laid out per head.
/// </summary>
/// <remarks>
/// Query rows are processed in tiles so a long sequence never materialises a full
/// <c>[tokens, tokens]</c> score matrix: a 5120-patch page would need 100 MB per head otherwise.
/// </remarks>
public static class Attention
{
    /// <summary>Query rows handled per tile.</summary>
    private const int QueryTile = 64;

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
        int tiles = (tokens + QueryTile - 1) / QueryTile;
        int totalTiles = heads * tiles;

        Parallel.For(
            0,
            totalTiles,
            () => TensorPool.Rent(QueryTile * tokens),
            (index, _, scores) =>
            {
                int head = index / tiles;
                int tile = index % tiles;
                int queryStart = tile * QueryTile;
                int queryCount = Math.Min(QueryTile, tokens - queryStart);

                int headOffset = head * tokens * headDim;
                ReadOnlySpan<float> q = queries.Span.Slice(headOffset, tokens * headDim);
                ReadOnlySpan<float> k = keys.Span.Slice(headOffset, tokens * headDim);
                ReadOnlySpan<float> v = values.Span.Slice(headOffset, tokens * headDim);
                Span<float> o = output.Span.Slice(headOffset, tokens * headDim);
                Span<float> scoreBuffer = scores.Span;

                for (int i = 0; i < queryCount; i++)
                {
                    ReadOnlySpan<float> queryRow = q.Slice((queryStart + i) * headDim, headDim);
                    Span<float> scoreRow = scoreBuffer.Slice(i * tokens, tokens);

                    for (int j = 0; j < tokens; j++)
                    {
                        scoreRow[j] = Gemm.Dot(queryRow, k.Slice(j * headDim, headDim)) * scale;
                    }

                    Kernels.Softmax(scoreRow);

                    Span<float> outputRow = o.Slice((queryStart + i) * headDim, headDim);
                    outputRow.Clear();
                    for (int j = 0; j < tokens; j++)
                    {
                        float weight = scoreRow[j];
                        if (weight != 0f)
                        {
                            Kernels.AddScaled(outputRow, v.Slice(j * headDim, headDim), weight);
                        }
                    }
                }

                return scores;
            },
            scores => scores.Dispose());
    }
}
