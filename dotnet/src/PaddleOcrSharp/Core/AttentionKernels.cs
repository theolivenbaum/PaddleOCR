using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace PaddleOcrSharp.Core;

/// <summary>
/// The two matrix products inside attention, in the shapes attention issues them.
/// </summary>
/// <remarks>
/// <para>
/// These are not <see cref="Gemm.MatMul"/> shapes. The score product reduces over one head's
/// width — 72 for this tower — which is nine vector steps, and the general kernel finishes every
/// output element with a lane reduction over that. Nine multiply-adds per reduction is the ratio
/// whatever the tile: <c>Dot4</c> pays four reductions per 36 multiply-adds and <c>Dot4x4</c>
/// sixteen per 144.
/// </para>
/// <para>
/// The cost that ratio hides is the dependency chain. <c>Dot4</c> keeps four accumulators, one
/// per output, and each is a loop-carried chain of fused multiply-adds. Two FMA ports at four
/// cycles of latency need eight chains in flight to stay busy, so four chains cap the loop at
/// about one multiply-add per cycle where the machine will do two — and the chains are only four
/// because the reduction lives in the vector lanes, which is also what forces the reductions.
/// </para>
/// <para>
/// Holding the keys transposed as <c>[headDim][tokens]</c> removes both at once: the lanes become
/// output columns, so a step over the head's width is an ordinary accumulation, and the tile can
/// be four query rows by two column vectors — eight independent chains, exactly enough to fill
/// the ports, with six loads per eight multiply-adds. The transpose costs one pass over the head's
/// keys per layer against the <c>tokens / 16</c> passes the product itself makes over them.
/// </para>
/// </remarks>
internal static class AttentionKernels
{
    /// <summary>Query rows one pass of the score kernel computes.</summary>
    private const int ScoreRows = 4;

    /// <summary>
    /// Keys one pass of the score kernel covers, so its slice of the transposed keys stays in L1
    /// while the block's row groups sweep it.
    /// </summary>
    private const int ScoreColumns = 128;

    /// <summary>
    /// Computes <c>scores[r, t] = Σ_d queries[r, d] · keysT[d, t]</c>.
    /// </summary>
    /// <param name="queries">Queries, <c>[rows, headDim]</c> row-major.</param>
    /// <param name="rows">Query rows in this block.</param>
    /// <param name="headDim">Width of one head — the reduction length.</param>
    /// <param name="keysT">Keys transposed to <c>[headDim, tokens]</c>.</param>
    /// <param name="tokens">Sequence length, and the score matrix's row length.</param>
    /// <param name="scores">Receives <c>[rows, tokens]</c>, written in full.</param>
    internal static void Scores(
        ReadOnlySpan<float> queries,
        int rows,
        int headDim,
        ReadOnlySpan<float> keysT,
        int tokens,
        Span<float> scores)
    {
        // Token block outermost, row group innermost. A tile of the transposed keys is
        // `headDim` lines of `ScoreColumns` floats — 4.6 KB at a head width of 72 — and the row
        // groups inside it each sweep that tile once, from L1. Row group outermost instead reads
        // the head's whole key matrix once per group, which at 1.4 MB is four passes through L2
        // per block for the same arithmetic.
        int token = 0;
        for (; token <= tokens - ScoreColumns; token += ScoreColumns)
        {
            int row = 0;
            for (; row <= rows - ScoreRows; row += ScoreRows)
            {
                ScoreQuad(queries, row, headDim, keysT, tokens, scores, token, ScoreColumns);
            }

            for (; row < rows; row++)
            {
                ScoreSingle(queries, row, headDim, keysT, tokens, scores, token, ScoreColumns);
            }
        }

        if (token < tokens)
        {
            int remaining = tokens - token;
            int row = 0;
            for (; row <= rows - ScoreRows; row += ScoreRows)
            {
                ScoreQuad(queries, row, headDim, keysT, tokens, scores, token, remaining);
            }

            for (; row < rows; row++)
            {
                ScoreSingle(queries, row, headDim, keysT, tokens, scores, token, remaining);
            }
        }
    }

    /// <summary>Four query rows against every key, two column vectors at a time.</summary>
    private static void ScoreQuad(
        ReadOnlySpan<float> queries,
        int row,
        int headDim,
        ReadOnlySpan<float> keysT,
        int tokens,
        Span<float> scores,
        int first,
        int count)
    {
        ref float q = ref MemoryMarshal.GetReference(queries);
        ref float k = ref MemoryMarshal.GetReference(keysT);
        ref float s = ref MemoryMarshal.GetReference(scores);

        int q0 = row * headDim;
        int s0 = row * tokens;
        int width = Vector256<float>.Count;
        int token = first;
        int end = first + count;

        if (Simd.Use512)
        {
            int wide = Vector512<float>.Count;
            for (; token <= end - (2 * wide); token += 2 * wide)
            {
                Vector512<float> z0 = Vector512<float>.Zero, y0 = Vector512<float>.Zero;
                Vector512<float> z1 = Vector512<float>.Zero, y1 = Vector512<float>.Zero;
                Vector512<float> z2 = Vector512<float>.Zero, y2 = Vector512<float>.Zero;
                Vector512<float> z3 = Vector512<float>.Zero, y3 = Vector512<float>.Zero;

                for (int d = 0; d < headDim; d++)
                {
                    int column = (d * tokens) + token;
                    Vector512<float> kLo = Vector512.LoadUnsafe(ref k, (nuint)column);
                    Vector512<float> kHi = Vector512.LoadUnsafe(ref k, (nuint)(column + wide));

                    Vector512<float> v = Vector512.Create(Unsafe.Add(ref q, q0 + d));
                    z0 = Vector512.FusedMultiplyAdd(v, kLo, z0);
                    y0 = Vector512.FusedMultiplyAdd(v, kHi, y0);

                    v = Vector512.Create(Unsafe.Add(ref q, q0 + headDim + d));
                    z1 = Vector512.FusedMultiplyAdd(v, kLo, z1);
                    y1 = Vector512.FusedMultiplyAdd(v, kHi, y1);

                    v = Vector512.Create(Unsafe.Add(ref q, q0 + (2 * headDim) + d));
                    z2 = Vector512.FusedMultiplyAdd(v, kLo, z2);
                    y2 = Vector512.FusedMultiplyAdd(v, kHi, y2);

                    v = Vector512.Create(Unsafe.Add(ref q, q0 + (3 * headDim) + d));
                    z3 = Vector512.FusedMultiplyAdd(v, kLo, z3);
                    y3 = Vector512.FusedMultiplyAdd(v, kHi, y3);
                }

                z0.StoreUnsafe(ref s, (nuint)(s0 + token));
                y0.StoreUnsafe(ref s, (nuint)(s0 + token + wide));
                z1.StoreUnsafe(ref s, (nuint)(s0 + tokens + token));
                y1.StoreUnsafe(ref s, (nuint)(s0 + tokens + token + wide));
                z2.StoreUnsafe(ref s, (nuint)(s0 + (2 * tokens) + token));
                y2.StoreUnsafe(ref s, (nuint)(s0 + (2 * tokens) + token + wide));
                z3.StoreUnsafe(ref s, (nuint)(s0 + (3 * tokens) + token));
                y3.StoreUnsafe(ref s, (nuint)(s0 + (3 * tokens) + token + wide));
            }
        }

        if (Simd.Use256)
        {
            for (; token <= end - (2 * width); token += 2 * width)
            {
                Vector256<float> a0 = Vector256<float>.Zero, b0 = Vector256<float>.Zero;
                Vector256<float> a1 = Vector256<float>.Zero, b1 = Vector256<float>.Zero;
                Vector256<float> a2 = Vector256<float>.Zero, b2 = Vector256<float>.Zero;
                Vector256<float> a3 = Vector256<float>.Zero, b3 = Vector256<float>.Zero;

                for (int d = 0; d < headDim; d++)
                {
                    int column = (d * tokens) + token;
                    Vector256<float> kLo = Vector256.LoadUnsafe(ref k, (nuint)column);
                    Vector256<float> kHi = Vector256.LoadUnsafe(ref k, (nuint)(column + width));

                    Vector256<float> v = Vector256.Create(Unsafe.Add(ref q, q0 + d));
                    a0 = Vector256.FusedMultiplyAdd(v, kLo, a0);
                    b0 = Vector256.FusedMultiplyAdd(v, kHi, b0);

                    v = Vector256.Create(Unsafe.Add(ref q, q0 + headDim + d));
                    a1 = Vector256.FusedMultiplyAdd(v, kLo, a1);
                    b1 = Vector256.FusedMultiplyAdd(v, kHi, b1);

                    v = Vector256.Create(Unsafe.Add(ref q, q0 + (2 * headDim) + d));
                    a2 = Vector256.FusedMultiplyAdd(v, kLo, a2);
                    b2 = Vector256.FusedMultiplyAdd(v, kHi, b2);

                    v = Vector256.Create(Unsafe.Add(ref q, q0 + (3 * headDim) + d));
                    a3 = Vector256.FusedMultiplyAdd(v, kLo, a3);
                    b3 = Vector256.FusedMultiplyAdd(v, kHi, b3);
                }

                a0.StoreUnsafe(ref s, (nuint)(s0 + token));
                b0.StoreUnsafe(ref s, (nuint)(s0 + token + width));
                a1.StoreUnsafe(ref s, (nuint)(s0 + tokens + token));
                b1.StoreUnsafe(ref s, (nuint)(s0 + tokens + token + width));
                a2.StoreUnsafe(ref s, (nuint)(s0 + (2 * tokens) + token));
                b2.StoreUnsafe(ref s, (nuint)(s0 + (2 * tokens) + token + width));
                a3.StoreUnsafe(ref s, (nuint)(s0 + (3 * tokens) + token));
                b3.StoreUnsafe(ref s, (nuint)(s0 + (3 * tokens) + token + width));
            }
        }

        for (; token < end; token++)
        {
            float t0 = 0f, t1 = 0f, t2 = 0f, t3 = 0f;
            for (int d = 0; d < headDim; d++)
            {
                float key = Unsafe.Add(ref k, (d * tokens) + token);
                t0 += Unsafe.Add(ref q, q0 + d) * key;
                t1 += Unsafe.Add(ref q, q0 + headDim + d) * key;
                t2 += Unsafe.Add(ref q, q0 + (2 * headDim) + d) * key;
                t3 += Unsafe.Add(ref q, q0 + (3 * headDim) + d) * key;
            }

            Unsafe.Add(ref s, s0 + token) = t0;
            Unsafe.Add(ref s, s0 + tokens + token) = t1;
            Unsafe.Add(ref s, s0 + (2 * tokens) + token) = t2;
            Unsafe.Add(ref s, s0 + (3 * tokens) + token) = t3;
        }
    }

    /// <summary>One query row against every key, for a block whose height is not a multiple of four.</summary>
    private static void ScoreSingle(
        ReadOnlySpan<float> queries,
        int row,
        int headDim,
        ReadOnlySpan<float> keysT,
        int tokens,
        Span<float> scores,
        int first,
        int count)
    {
        ref float q = ref MemoryMarshal.GetReference(queries);
        ref float k = ref MemoryMarshal.GetReference(keysT);
        ref float s = ref MemoryMarshal.GetReference(scores);

        int q0 = row * headDim;
        int s0 = row * tokens;
        int width = Vector256<float>.Count;
        int token = first;
        int end = first + count;

        if (Simd.Use512)
        {
            int wide = Vector512<float>.Count;
            for (; token <= end - (4 * wide); token += 4 * wide)
            {
                Vector512<float> z0 = Vector512<float>.Zero, z1 = Vector512<float>.Zero;
                Vector512<float> z2 = Vector512<float>.Zero, z3 = Vector512<float>.Zero;

                for (int d = 0; d < headDim; d++)
                {
                    int column = (d * tokens) + token;
                    Vector512<float> v = Vector512.Create(Unsafe.Add(ref q, q0 + d));
                    z0 = Vector512.FusedMultiplyAdd(v, Vector512.LoadUnsafe(ref k, (nuint)column), z0);
                    z1 = Vector512.FusedMultiplyAdd(
                        v, Vector512.LoadUnsafe(ref k, (nuint)(column + wide)), z1);
                    z2 = Vector512.FusedMultiplyAdd(
                        v, Vector512.LoadUnsafe(ref k, (nuint)(column + (2 * wide))), z2);
                    z3 = Vector512.FusedMultiplyAdd(
                        v, Vector512.LoadUnsafe(ref k, (nuint)(column + (3 * wide))), z3);
                }

                z0.StoreUnsafe(ref s, (nuint)(s0 + token));
                z1.StoreUnsafe(ref s, (nuint)(s0 + token + wide));
                z2.StoreUnsafe(ref s, (nuint)(s0 + token + (2 * wide)));
                z3.StoreUnsafe(ref s, (nuint)(s0 + token + (3 * wide)));
            }
        }

        if (Simd.Use256)
        {
            for (; token <= end - (4 * width); token += 4 * width)
            {
                Vector256<float> a0 = Vector256<float>.Zero, a1 = Vector256<float>.Zero;
                Vector256<float> a2 = Vector256<float>.Zero, a3 = Vector256<float>.Zero;

                for (int d = 0; d < headDim; d++)
                {
                    int column = (d * tokens) + token;
                    Vector256<float> v = Vector256.Create(Unsafe.Add(ref q, q0 + d));
                    a0 = Vector256.FusedMultiplyAdd(v, Vector256.LoadUnsafe(ref k, (nuint)column), a0);
                    a1 = Vector256.FusedMultiplyAdd(
                        v, Vector256.LoadUnsafe(ref k, (nuint)(column + width)), a1);
                    a2 = Vector256.FusedMultiplyAdd(
                        v, Vector256.LoadUnsafe(ref k, (nuint)(column + (2 * width))), a2);
                    a3 = Vector256.FusedMultiplyAdd(
                        v, Vector256.LoadUnsafe(ref k, (nuint)(column + (3 * width))), a3);
                }

                a0.StoreUnsafe(ref s, (nuint)(s0 + token));
                a1.StoreUnsafe(ref s, (nuint)(s0 + token + width));
                a2.StoreUnsafe(ref s, (nuint)(s0 + token + (2 * width)));
                a3.StoreUnsafe(ref s, (nuint)(s0 + token + (3 * width)));
            }
        }

        for (; token < end; token++)
        {
            float sum = 0f;
            for (int d = 0; d < headDim; d++)
            {
                sum += Unsafe.Add(ref q, q0 + d) * Unsafe.Add(ref k, (d * tokens) + token);
            }

            Unsafe.Add(ref s, s0 + token) = sum;
        }
    }

    /// <summary>
    /// Transposes one head's <c>[tokens, headDim]</c> matrix into <c>[headDim, tokens]</c>.
    /// </summary>
    /// <param name="source">The head's rows, <c>[tokens, headDim]</c>.</param>
    /// <param name="tokens">Number of rows in <paramref name="source"/>.</param>
    /// <param name="headDim">Number of columns in <paramref name="source"/>.</param>
    /// <param name="destination">Receives <c>[headDim, tokens]</c>.</param>
    internal static void TransposeHead(
        ReadOnlySpan<float> source,
        int tokens,
        int headDim,
        Span<float> destination)
    {
        // Blocked so neither side walks the whole matrix with a long stride. A head is 72 columns,
        // so a 32-row block of the source is 9 KB and its image in the destination touches 72
        // rows of 128 bytes — both inside L1 for the duration of the block.
        const int Block = 32;

        for (int t0 = 0; t0 < tokens; t0 += Block)
        {
            int tEnd = Math.Min(t0 + Block, tokens);
            for (int d = 0; d < headDim; d++)
            {
                int column = d * tokens;
                for (int t = t0; t < tEnd; t++)
                {
                    destination[column + t] = source[(t * headDim) + d];
                }
            }
        }
    }

    /// <summary>Tokens one pass of the value kernel reduces over.</summary>
    /// <remarks>
    /// A chunk's slice of the values — <c>ChunkTokens x headDim</c> floats, 23 KB at a head width
    /// of 72 — has to stay in L1 while the twenty output tiles each sweep it once. That is what
    /// makes the sweeps free; sized for L2 instead, the kernel simply becomes L2-bound and gains
    /// nothing.
    /// </remarks>
    private const int ChunkTokens = 80;

    /// <summary>Query rows one pass of the value kernel accumulates.</summary>
    private const int ValueRows = 4;

    /// <summary>
    /// Computes <c>result[r, d] = Σ_t weights[r, t] · values[t, d]</c>.
    /// </summary>
    /// <param name="weights">Attention weights, <c>[rows, tokens]</c> row-major.</param>
    /// <param name="rows">Query rows in this block.</param>
    /// <param name="tokens">Sequence length — the reduction length.</param>
    /// <param name="values">Values, <c>[tokens, headDim]</c> row-major.</param>
    /// <param name="headDim">Width of one head.</param>
    /// <param name="result">Receives <c>[rows, headDim]</c>, written in full.</param>
    /// <remarks>
    /// <para>
    /// The general kernel keeps this product's output rows in memory and streams the values past
    /// them, which makes the loop-carried dependency a store and a reload of every output element
    /// on every one of the thousands of reduction steps — one store per multiply-add, and the
    /// store port is single. It also sweeps the values once per group of four output rows, so a
    /// sixteen-row block reads them four times.
    /// </para>
    /// <para>
    /// Chunking the reduction fixes both. Inside a chunk the output tile — four rows by sixteen
    /// columns, eight accumulators — lives in registers, so the chunk costs two stores per output
    /// element instead of two per element per token; and the chunk's values stay in L1 across the
    /// tiles that sweep it, so the level that sees them once per block is L2. Six loads per eight
    /// multiply-adds leaves the loop bound by the FMA ports, which is the point.
    /// </para>
    /// </remarks>
    internal static void Values(
        ReadOnlySpan<float> weights,
        int rows,
        int tokens,
        ReadOnlySpan<float> values,
        int headDim,
        Span<float> result)
    {
        result[..(rows * headDim)].Clear();

        if (!Simd.Use256)
        {
            Fallback(weights, rows, tokens, values, headDim, result);
            return;
        }

        int width = Vector256<float>.Count;

        ref float w = ref MemoryMarshal.GetReference(weights);
        ref float v = ref MemoryMarshal.GetReference(values);
        ref float y = ref MemoryMarshal.GetReference(result);

        for (int chunk = 0; chunk < tokens; chunk += ChunkTokens)
        {
            int steps = Math.Min(ChunkTokens, tokens - chunk);

            int row = 0;
            for (; row <= rows - ValueRows; row += ValueRows)
            {
                int column = 0;
                if (Simd.Use512)
                {
                    int wide = Vector512<float>.Count;
                    for (; column <= headDim - (2 * wide); column += 2 * wide)
                    {
                        ValueTileWide512(ref w, ref v, ref y, row, column, chunk, steps, tokens, headDim, wide);
                    }
                }

                for (; column <= headDim - (2 * width); column += 2 * width)
                {
                    ValueTileWide(ref w, ref v, ref y, row, column, chunk, steps, tokens, headDim, width);
                }

                for (; column <= headDim - width; column += width)
                {
                    ValueTileNarrow(ref w, ref v, ref y, row, column, chunk, steps, tokens, headDim);
                }

                for (; column < headDim; column++)
                {
                    ValueColumnTail(ref w, ref v, ref y, row, ValueRows, column, chunk, steps, tokens, headDim);
                }
            }

            for (; row < rows; row++)
            {
                for (int column = 0; column < headDim; column++)
                {
                    ValueColumnTail(ref w, ref v, ref y, row, 1, column, chunk, steps, tokens, headDim);
                }
            }
        }
    }

    /// <summary>Four output rows by two column vectors, accumulated in eight registers.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ValueTileWide(
        ref float w,
        ref float v,
        ref float y,
        int row,
        int column,
        int chunk,
        int steps,
        int tokens,
        int headDim,
        int width)
    {
        Vector256<float> a0 = Vector256<float>.Zero, b0 = Vector256<float>.Zero;
        Vector256<float> a1 = Vector256<float>.Zero, b1 = Vector256<float>.Zero;
        Vector256<float> a2 = Vector256<float>.Zero, b2 = Vector256<float>.Zero;
        Vector256<float> a3 = Vector256<float>.Zero, b3 = Vector256<float>.Zero;

        int w0 = (row * tokens) + chunk;
        int vBase = (chunk * headDim) + column;

        for (int step = 0; step < steps; step++)
        {
            int offset = vBase + (step * headDim);
            Vector256<float> lo = Vector256.LoadUnsafe(ref v, (nuint)offset);
            Vector256<float> hi = Vector256.LoadUnsafe(ref v, (nuint)(offset + width));

            Vector256<float> s = Vector256.Create(Unsafe.Add(ref w, w0 + step));
            a0 = Vector256.FusedMultiplyAdd(s, lo, a0);
            b0 = Vector256.FusedMultiplyAdd(s, hi, b0);

            s = Vector256.Create(Unsafe.Add(ref w, w0 + tokens + step));
            a1 = Vector256.FusedMultiplyAdd(s, lo, a1);
            b1 = Vector256.FusedMultiplyAdd(s, hi, b1);

            s = Vector256.Create(Unsafe.Add(ref w, w0 + (2 * tokens) + step));
            a2 = Vector256.FusedMultiplyAdd(s, lo, a2);
            b2 = Vector256.FusedMultiplyAdd(s, hi, b2);

            s = Vector256.Create(Unsafe.Add(ref w, w0 + (3 * tokens) + step));
            a3 = Vector256.FusedMultiplyAdd(s, lo, a3);
            b3 = Vector256.FusedMultiplyAdd(s, hi, b3);
        }

        Accumulate(ref y, (row * headDim) + column, a0, b0, width);
        Accumulate(ref y, ((row + 1) * headDim) + column, a1, b1, width);
        Accumulate(ref y, ((row + 2) * headDim) + column, a2, b2, width);
        Accumulate(ref y, ((row + 3) * headDim) + column, a3, b3, width);
    }

    /// <summary>
    /// Four output rows by two 512-bit column vectors, accumulated in eight <c>zmm</c> registers.
    /// </summary>
    /// <remarks>
    /// Widening the lanes does not reorder anything: the lanes are output columns and the
    /// reduction still steps over the chunk's tokens one at a time, so every output element
    /// accumulates its terms in exactly the order the 256-bit tile accumulates them. The two
    /// kernels are bit-identical, which is what lets the wide one be chosen on vector width alone.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ValueTileWide512(
        ref float w,
        ref float v,
        ref float y,
        int row,
        int column,
        int chunk,
        int steps,
        int tokens,
        int headDim,
        int width)
    {
        Vector512<float> a0 = Vector512<float>.Zero, b0 = Vector512<float>.Zero;
        Vector512<float> a1 = Vector512<float>.Zero, b1 = Vector512<float>.Zero;
        Vector512<float> a2 = Vector512<float>.Zero, b2 = Vector512<float>.Zero;
        Vector512<float> a3 = Vector512<float>.Zero, b3 = Vector512<float>.Zero;

        int w0 = (row * tokens) + chunk;
        int vBase = (chunk * headDim) + column;

        for (int step = 0; step < steps; step++)
        {
            int offset = vBase + (step * headDim);
            Vector512<float> lo = Vector512.LoadUnsafe(ref v, (nuint)offset);
            Vector512<float> hi = Vector512.LoadUnsafe(ref v, (nuint)(offset + width));

            Vector512<float> s = Vector512.Create(Unsafe.Add(ref w, w0 + step));
            a0 = Vector512.FusedMultiplyAdd(s, lo, a0);
            b0 = Vector512.FusedMultiplyAdd(s, hi, b0);

            s = Vector512.Create(Unsafe.Add(ref w, w0 + tokens + step));
            a1 = Vector512.FusedMultiplyAdd(s, lo, a1);
            b1 = Vector512.FusedMultiplyAdd(s, hi, b1);

            s = Vector512.Create(Unsafe.Add(ref w, w0 + (2 * tokens) + step));
            a2 = Vector512.FusedMultiplyAdd(s, lo, a2);
            b2 = Vector512.FusedMultiplyAdd(s, hi, b2);

            s = Vector512.Create(Unsafe.Add(ref w, w0 + (3 * tokens) + step));
            a3 = Vector512.FusedMultiplyAdd(s, lo, a3);
            b3 = Vector512.FusedMultiplyAdd(s, hi, b3);
        }

        Accumulate512(ref y, (row * headDim) + column, a0, b0, width);
        Accumulate512(ref y, ((row + 1) * headDim) + column, a1, b1, width);
        Accumulate512(ref y, ((row + 2) * headDim) + column, a2, b2, width);
        Accumulate512(ref y, ((row + 3) * headDim) + column, a3, b3, width);
    }

    /// <summary>Four output rows by one column vector, for the columns a wide tile does not cover.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ValueTileNarrow(
        ref float w,
        ref float v,
        ref float y,
        int row,
        int column,
        int chunk,
        int steps,
        int tokens,
        int headDim)
    {
        Vector256<float> a0 = Vector256<float>.Zero, a1 = Vector256<float>.Zero;
        Vector256<float> a2 = Vector256<float>.Zero, a3 = Vector256<float>.Zero;

        int w0 = (row * tokens) + chunk;
        int vBase = (chunk * headDim) + column;

        for (int step = 0; step < steps; step++)
        {
            Vector256<float> value = Vector256.LoadUnsafe(ref v, (nuint)(vBase + (step * headDim)));
            a0 = Vector256.FusedMultiplyAdd(
                Vector256.Create(Unsafe.Add(ref w, w0 + step)), value, a0);
            a1 = Vector256.FusedMultiplyAdd(
                Vector256.Create(Unsafe.Add(ref w, w0 + tokens + step)), value, a1);
            a2 = Vector256.FusedMultiplyAdd(
                Vector256.Create(Unsafe.Add(ref w, w0 + (2 * tokens) + step)), value, a2);
            a3 = Vector256.FusedMultiplyAdd(
                Vector256.Create(Unsafe.Add(ref w, w0 + (3 * tokens) + step)), value, a3);
        }

        AccumulateOne(ref y, (row * headDim) + column, a0);
        AccumulateOne(ref y, ((row + 1) * headDim) + column, a1);
        AccumulateOne(ref y, ((row + 2) * headDim) + column, a2);
        AccumulateOne(ref y, ((row + 3) * headDim) + column, a3);
    }

    /// <summary>One output column for up to four rows, scalar.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ValueColumnTail(
        ref float w,
        ref float v,
        ref float y,
        int row,
        int rowCount,
        int column,
        int chunk,
        int steps,
        int tokens,
        int headDim)
    {
        for (int r = 0; r < rowCount; r++)
        {
            float sum = 0f;
            int w0 = ((row + r) * tokens) + chunk;
            int vBase = (chunk * headDim) + column;

            for (int step = 0; step < steps; step++)
            {
                sum += Unsafe.Add(ref w, w0 + step) * Unsafe.Add(ref v, vBase + (step * headDim));
            }

            Unsafe.Add(ref y, ((row + r) * headDim) + column) += sum;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Accumulate(ref float y, int offset, Vector256<float> lo, Vector256<float> hi, int width)
    {
        (Vector256.LoadUnsafe(ref y, (nuint)offset) + lo).StoreUnsafe(ref y, (nuint)offset);
        (Vector256.LoadUnsafe(ref y, (nuint)(offset + width)) + hi).StoreUnsafe(ref y, (nuint)(offset + width));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Accumulate512(ref float y, int offset, Vector512<float> lo, Vector512<float> hi, int width)
    {
        (Vector512.LoadUnsafe(ref y, (nuint)offset) + lo).StoreUnsafe(ref y, (nuint)offset);
        (Vector512.LoadUnsafe(ref y, (nuint)(offset + width)) + hi).StoreUnsafe(ref y, (nuint)(offset + width));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateOne(ref float y, int offset, Vector256<float> value) =>
        (Vector256.LoadUnsafe(ref y, (nuint)offset) + value).StoreUnsafe(ref y, (nuint)offset);

    /// <summary>Scalar value product, for a machine with no 256-bit vectors.</summary>
    private static void Fallback(
        ReadOnlySpan<float> weights,
        int rows,
        int tokens,
        ReadOnlySpan<float> values,
        int headDim,
        Span<float> result)
    {
        for (int row = 0; row < rows; row++)
        {
            for (int token = 0; token < tokens; token++)
            {
                float weight = weights[(row * tokens) + token];
                for (int d = 0; d < headDim; d++)
                {
                    result[(row * headDim) + d] += weight * values[(token * headDim) + d];
                }
            }
        }
    }
}
