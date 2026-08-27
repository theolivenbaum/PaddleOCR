using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace PaddleOcrSharp.Core;

/// <summary>
/// Matrix products in the layout PyTorch's <c>nn.Linear</c> uses: the weight is stored as
/// <c>[out, in]</c> and the product is <c>y = x · Wᵀ + b</c>.
/// </summary>
/// <remarks>
/// <para>
/// Weights are kept in their on-disk dtype (bfloat16 for PaddleOCR-VL) and widened inside the
/// inner loop. Halving the weight bytes matters more than the widening costs: single-token
/// decoding is bound by weight bandwidth, not by arithmetic.
/// </para>
/// <para>
/// Accumulation is always float32, matching PyTorch, which accumulates bf16 matmuls in fp32.
/// </para>
/// </remarks>
public static class Gemm
{
    /// <summary>Rows of <c>x</c> handled by one pass of the inner kernel.</summary>
    private const int RowBlock = 4;

    /// <summary>Output columns handled by one pass of the inner kernel.</summary>
    private const int ColBlock = 4;

    /// <summary>Minimum output elements before the work is spread across threads.</summary>
    private const int ParallelThreshold = 8192;

    /// <summary>
    /// Computes <c>y[m, n] = Σ_k x[m, k] · w[n, k] + bias[n]</c>.
    /// </summary>
    /// <param name="x">Activations, <c>[m, k]</c>, row-major.</param>
    /// <param name="rows">Number of activation rows (<c>m</c>).</param>
    /// <param name="inner">Shared inner dimension (<c>k</c>).</param>
    /// <param name="weight">Weight, <c>[n, k]</c>, row-major — the <c>nn.Linear</c> layout.</param>
    /// <param name="bias">Optional bias of length <c>n</c>; pass an empty span for no bias.</param>
    /// <param name="y">Output, <c>[m, n]</c>, row-major.</param>
    /// <param name="cols">Number of output columns (<c>n</c>).</param>
    public static void Linear(
        ReadOnlyMemory<float> x,
        int rows,
        int inner,
        WeightMatrix weight,
        ReadOnlyMemory<float> bias,
        Memory<float> y,
        int cols)
    {
        if (weight.Rows != cols || weight.Cols != inner)
        {
            throw new ArgumentException(
                $"Weight is [{weight.Rows}, {weight.Cols}] but the call needs [{cols}, {inner}].",
                nameof(weight));
        }

        if (x.Length < (long)rows * inner)
        {
            throw new ArgumentException("Activation buffer is too small.", nameof(x));
        }

        if (y.Length < (long)rows * cols)
        {
            throw new ArgumentException("Output buffer is too small.", nameof(y));
        }

        if (!bias.IsEmpty && bias.Length < cols)
        {
            throw new ArgumentException("Bias buffer is too small.", nameof(bias));
        }

        if ((long)rows * cols * inner < ParallelThreshold || Environment.ProcessorCount <= 1)
        {
            RunPanel(x.Span, rows, inner, weight, bias.Span, y.Span, cols, 0, cols);
            return;
        }

        // Split along the output columns, not the rows: a thread that owns a column panel walks
        // the same slice of the weight matrix for every activation row, so the panel stays in
        // cache. Splitting along rows would make every thread stream the whole weight matrix once
        // per row, which for a 1152x4304 projection is 10 MB of traffic per token.
        int panel = ChoosePanelWidth(inner, cols, weight.Dtype);
        int panels = (cols + panel - 1) / panel;

        Parallel.For(0, panels, index =>
        {
            int start = index * panel;
            RunPanel(
                x.Span, rows, inner, weight, bias.Span, y.Span, cols, start, Math.Min(panel, cols - start));
        });
    }

    /// <summary>
    /// Chooses a column-panel width whose slice of the weight matrix stays inside L2.
    /// </summary>
    private static int ChoosePanelWidth(int inner, int cols, DType dtype)
    {
        const int TargetPanelBytes = 256 * 1024;

        int rowBytes = Math.Max(1, inner * dtype.ByteSize());
        int panel = Math.Max(ColBlock, TargetPanelBytes / rowBytes);
        panel = (panel + ColBlock - 1) / ColBlock * ColBlock;

        // Never leave threads idle: cap the panel so there is at least one per processor.
        int maximum = Math.Max(ColBlock, (cols + Environment.ProcessorCount - 1) / Environment.ProcessorCount);
        maximum = (maximum + ColBlock - 1) / ColBlock * ColBlock;

        return Math.Min(panel, Math.Max(ColBlock, maximum));
    }

    /// <summary>
    /// Computes one column panel for every activation row.
    /// </summary>
    /// <remarks>
    /// The panel's weights are widened to float32 once and reused across every activation row.
    /// Widening inside the inner loop instead would cost one shift and one interleave for every
    /// four multiply-adds, which caps the kernel at a fraction of the machine's FMA throughput.
    /// </remarks>
    private static void RunPanel(
        ReadOnlySpan<float> x,
        int rows,
        int inner,
        WeightMatrix weight,
        ReadOnlySpan<float> bias,
        Span<float> y,
        int cols,
        int colStart,
        int colCount)
    {
        using PooledBuffer panel = TensorPool.Rent(colCount * inner);
        Span<float> w = panel.Span;
        weight.CopyRows(colStart, colCount, w);

        int colEnd = colStart + colCount;
        Span<float> tile = stackalloc float[16];

        int m = 0;
        for (; m <= rows - RowBlock; m += RowBlock)
        {
            ReadOnlySpan<float> x0 = x.Slice(m * inner, inner);
            ReadOnlySpan<float> x1 = x.Slice((m + 1) * inner, inner);
            ReadOnlySpan<float> x2 = x.Slice((m + 2) * inner, inner);
            ReadOnlySpan<float> x3 = x.Slice((m + 3) * inner, inner);

            int local = 0;
            for (; local <= colCount - ColBlock; local += ColBlock)
            {
                Dot4x4(x0, x1, x2, x3, w, local * inner, inner, tile);
                for (int r = 0; r < RowBlock; r++)
                {
                    Span<float> row = y.Slice(((m + r) * cols) + colStart + local, ColBlock);
                    row[0] = tile[r * 4];
                    row[1] = tile[(r * 4) + 1];
                    row[2] = tile[(r * 4) + 2];
                    row[3] = tile[(r * 4) + 3];
                }
            }

            for (; local < colCount; local++)
            {
                ReadOnlySpan<float> wr = w.Slice(local * inner, inner);
                y[(m * cols) + colStart + local] = Dot(x0, wr);
                y[((m + 1) * cols) + colStart + local] = Dot(x1, wr);
                y[((m + 2) * cols) + colStart + local] = Dot(x2, wr);
                y[((m + 3) * cols) + colStart + local] = Dot(x3, wr);
            }

            if (!bias.IsEmpty)
            {
                for (int r = 0; r < RowBlock; r++)
                {
                    Kernels.AddInPlace(y.Slice(((m + r) * cols) + colStart, colCount), bias[colStart..colEnd]);
                }
            }
        }

        for (; m < rows; m++)
        {
            ReadOnlySpan<float> xr = x.Slice(m * inner, inner);
            Span<float> yr = y.Slice((m * cols) + colStart, colCount);

            for (int local = 0; local < colCount; local++)
            {
                yr[local] = Dot(xr, w.Slice(local * inner, inner));
            }

            if (!bias.IsEmpty)
            {
                Kernels.AddInPlace(yr, bias[colStart..colEnd]);
            }
        }
    }

    /// <summary>
    /// Sixteen dot products: four activation rows against four consecutive float32 weight rows.
    /// </summary>
    /// <param name="x0">First activation row.</param>
    /// <param name="x1">Second activation row.</param>
    /// <param name="x2">Third activation row.</param>
    /// <param name="x3">Fourth activation row.</param>
    /// <param name="w">Widened weight panel.</param>
    /// <param name="offset">Index of the first weight row inside <paramref name="w"/>.</param>
    /// <param name="stride">Distance between consecutive weight rows.</param>
    /// <param name="results">Receives 16 results, activation-major: <c>results[i * 4 + j]</c>.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Dot4x4(
        ReadOnlySpan<float> x0,
        ReadOnlySpan<float> x1,
        ReadOnlySpan<float> x2,
        ReadOnlySpan<float> x3,
        ReadOnlySpan<float> w,
        int offset,
        int stride,
        Span<float> results)
    {
        int length = x0.Length;
        int i = 0;

        if (Vector512.IsHardwareAccelerated && length >= Vector512<float>.Count)
        {
            Vector512<float> a00 = Vector512<float>.Zero, a01 = Vector512<float>.Zero;
            Vector512<float> a02 = Vector512<float>.Zero, a03 = Vector512<float>.Zero;
            Vector512<float> a10 = Vector512<float>.Zero, a11 = Vector512<float>.Zero;
            Vector512<float> a12 = Vector512<float>.Zero, a13 = Vector512<float>.Zero;
            Vector512<float> a20 = Vector512<float>.Zero, a21 = Vector512<float>.Zero;
            Vector512<float> a22 = Vector512<float>.Zero, a23 = Vector512<float>.Zero;
            Vector512<float> a30 = Vector512<float>.Zero, a31 = Vector512<float>.Zero;
            Vector512<float> a32 = Vector512<float>.Zero, a33 = Vector512<float>.Zero;

            for (; i <= length - Vector512<float>.Count; i += Vector512<float>.Count)
            {
                Vector512<float> w0 = Vector512.LoadUnsafe(in w[offset + i]);
                Vector512<float> w1 = Vector512.LoadUnsafe(in w[offset + stride + i]);
                Vector512<float> w2 = Vector512.LoadUnsafe(in w[offset + (2 * stride) + i]);
                Vector512<float> w3 = Vector512.LoadUnsafe(in w[offset + (3 * stride) + i]);

                Vector512<float> v = Vector512.LoadUnsafe(in x0[i]);
                a00 = Vector512.FusedMultiplyAdd(v, w0, a00);
                a01 = Vector512.FusedMultiplyAdd(v, w1, a01);
                a02 = Vector512.FusedMultiplyAdd(v, w2, a02);
                a03 = Vector512.FusedMultiplyAdd(v, w3, a03);

                v = Vector512.LoadUnsafe(in x1[i]);
                a10 = Vector512.FusedMultiplyAdd(v, w0, a10);
                a11 = Vector512.FusedMultiplyAdd(v, w1, a11);
                a12 = Vector512.FusedMultiplyAdd(v, w2, a12);
                a13 = Vector512.FusedMultiplyAdd(v, w3, a13);

                v = Vector512.LoadUnsafe(in x2[i]);
                a20 = Vector512.FusedMultiplyAdd(v, w0, a20);
                a21 = Vector512.FusedMultiplyAdd(v, w1, a21);
                a22 = Vector512.FusedMultiplyAdd(v, w2, a22);
                a23 = Vector512.FusedMultiplyAdd(v, w3, a23);

                v = Vector512.LoadUnsafe(in x3[i]);
                a30 = Vector512.FusedMultiplyAdd(v, w0, a30);
                a31 = Vector512.FusedMultiplyAdd(v, w1, a31);
                a32 = Vector512.FusedMultiplyAdd(v, w2, a32);
                a33 = Vector512.FusedMultiplyAdd(v, w3, a33);
            }

            results[0] = Vector512.Sum(a00);
            results[1] = Vector512.Sum(a01);
            results[2] = Vector512.Sum(a02);
            results[3] = Vector512.Sum(a03);
            results[4] = Vector512.Sum(a10);
            results[5] = Vector512.Sum(a11);
            results[6] = Vector512.Sum(a12);
            results[7] = Vector512.Sum(a13);
            results[8] = Vector512.Sum(a20);
            results[9] = Vector512.Sum(a21);
            results[10] = Vector512.Sum(a22);
            results[11] = Vector512.Sum(a23);
            results[12] = Vector512.Sum(a30);
            results[13] = Vector512.Sum(a31);
            results[14] = Vector512.Sum(a32);
            results[15] = Vector512.Sum(a33);
        }
        else if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            Vector256<float> a00 = Vector256<float>.Zero, a01 = Vector256<float>.Zero;
            Vector256<float> a02 = Vector256<float>.Zero, a03 = Vector256<float>.Zero;
            Vector256<float> a10 = Vector256<float>.Zero, a11 = Vector256<float>.Zero;
            Vector256<float> a12 = Vector256<float>.Zero, a13 = Vector256<float>.Zero;
            Vector256<float> a20 = Vector256<float>.Zero, a21 = Vector256<float>.Zero;
            Vector256<float> a22 = Vector256<float>.Zero, a23 = Vector256<float>.Zero;
            Vector256<float> a30 = Vector256<float>.Zero, a31 = Vector256<float>.Zero;
            Vector256<float> a32 = Vector256<float>.Zero, a33 = Vector256<float>.Zero;

            for (; i <= length - Vector256<float>.Count; i += Vector256<float>.Count)
            {
                Vector256<float> w0 = Vector256.LoadUnsafe(in w[offset + i]);
                Vector256<float> w1 = Vector256.LoadUnsafe(in w[offset + stride + i]);
                Vector256<float> w2 = Vector256.LoadUnsafe(in w[offset + (2 * stride) + i]);
                Vector256<float> w3 = Vector256.LoadUnsafe(in w[offset + (3 * stride) + i]);

                Vector256<float> v = Vector256.LoadUnsafe(in x0[i]);
                a00 = Vector256.FusedMultiplyAdd(v, w0, a00);
                a01 = Vector256.FusedMultiplyAdd(v, w1, a01);
                a02 = Vector256.FusedMultiplyAdd(v, w2, a02);
                a03 = Vector256.FusedMultiplyAdd(v, w3, a03);

                v = Vector256.LoadUnsafe(in x1[i]);
                a10 = Vector256.FusedMultiplyAdd(v, w0, a10);
                a11 = Vector256.FusedMultiplyAdd(v, w1, a11);
                a12 = Vector256.FusedMultiplyAdd(v, w2, a12);
                a13 = Vector256.FusedMultiplyAdd(v, w3, a13);

                v = Vector256.LoadUnsafe(in x2[i]);
                a20 = Vector256.FusedMultiplyAdd(v, w0, a20);
                a21 = Vector256.FusedMultiplyAdd(v, w1, a21);
                a22 = Vector256.FusedMultiplyAdd(v, w2, a22);
                a23 = Vector256.FusedMultiplyAdd(v, w3, a23);

                v = Vector256.LoadUnsafe(in x3[i]);
                a30 = Vector256.FusedMultiplyAdd(v, w0, a30);
                a31 = Vector256.FusedMultiplyAdd(v, w1, a31);
                a32 = Vector256.FusedMultiplyAdd(v, w2, a32);
                a33 = Vector256.FusedMultiplyAdd(v, w3, a33);
            }

            results[0] = Vector256.Sum(a00);
            results[1] = Vector256.Sum(a01);
            results[2] = Vector256.Sum(a02);
            results[3] = Vector256.Sum(a03);
            results[4] = Vector256.Sum(a10);
            results[5] = Vector256.Sum(a11);
            results[6] = Vector256.Sum(a12);
            results[7] = Vector256.Sum(a13);
            results[8] = Vector256.Sum(a20);
            results[9] = Vector256.Sum(a21);
            results[10] = Vector256.Sum(a22);
            results[11] = Vector256.Sum(a23);
            results[12] = Vector256.Sum(a30);
            results[13] = Vector256.Sum(a31);
            results[14] = Vector256.Sum(a32);
            results[15] = Vector256.Sum(a33);
        }
        else
        {
            results[..16].Clear();
        }

        for (; i < length; i++)
        {
            float w0 = w[offset + i];
            float w1 = w[offset + stride + i];
            float w2 = w[offset + (2 * stride) + i];
            float w3 = w[offset + (3 * stride) + i];

            results[0] += x0[i] * w0;
            results[1] += x0[i] * w1;
            results[2] += x0[i] * w2;
            results[3] += x0[i] * w3;
            results[4] += x1[i] * w0;
            results[5] += x1[i] * w1;
            results[6] += x1[i] * w2;
            results[7] += x1[i] * w3;
            results[8] += x2[i] * w0;
            results[9] += x2[i] * w1;
            results[10] += x2[i] * w2;
            results[11] += x2[i] * w3;
            results[12] += x3[i] * w0;
            results[13] += x3[i] * w1;
            results[14] += x3[i] * w2;
            results[15] += x3[i] * w3;
        }
    }

    /// <summary>
    /// Computes <c>y[m, n] = Σ_k a[m, k] · b[k, n]</c> for two dense float32 matrices, i.e. the
    /// non-transposed product used by attention (<c>probs · V</c>).
    /// </summary>
    public static void MatMul(
        ReadOnlySpan<float> a,
        int rows,
        int inner,
        ReadOnlySpan<float> b,
        Span<float> y,
        int cols)
    {
        for (int m = 0; m < rows; m++)
        {
            Span<float> yr = y.Slice(m * cols, cols);
            yr.Clear();
            ReadOnlySpan<float> ar = a.Slice(m * inner, inner);
            for (int k = 0; k < inner; k++)
            {
                float scale = ar[k];
                if (scale != 0f)
                {
                    Kernels.AddScaled(yr, b.Slice(k * cols, cols), scale);
                }
            }
        }
    }

    /// <summary>
    /// Four dot products of <paramref name="x"/> against consecutive rows of <paramref name="w"/>.
    /// </summary>
    /// <remarks>
    /// Reading four weight rows against one activation row keeps the activation in registers
    /// across four streams, which is what makes the convolution inner loop compute-bound rather
    /// than load-bound.
    /// </remarks>
    /// <param name="x">The activation row.</param>
    /// <param name="w">Weights, four rows of <c>x.Length</c> starting at <paramref name="offset"/>.</param>
    /// <param name="offset">Index of the first weight row.</param>
    /// <param name="stride">Distance between consecutive weight rows.</param>
    /// <param name="a0">Dot product with the first row.</param>
    /// <param name="a1">Dot product with the second row.</param>
    /// <param name="a2">Dot product with the third row.</param>
    /// <param name="a3">Dot product with the fourth row.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Dot4(
        ReadOnlySpan<float> x,
        ReadOnlySpan<float> w,
        int offset,
        int stride,
        out float a0,
        out float a1,
        out float a2,
        out float a3)
    {
        int length = x.Length;
        int i = 0;
        float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;

        if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            Vector256<float> v0 = Vector256<float>.Zero;
            Vector256<float> v1 = Vector256<float>.Zero;
            Vector256<float> v2 = Vector256<float>.Zero;
            Vector256<float> v3 = Vector256<float>.Zero;

            for (; i <= length - Vector256<float>.Count; i += Vector256<float>.Count)
            {
                Vector256<float> xv = Vector256.LoadUnsafe(in x[i]);
                v0 = Vector256.FusedMultiplyAdd(xv, Vector256.LoadUnsafe(in w[offset + i]), v0);
                v1 = Vector256.FusedMultiplyAdd(xv, Vector256.LoadUnsafe(in w[offset + stride + i]), v1);
                v2 = Vector256.FusedMultiplyAdd(xv, Vector256.LoadUnsafe(in w[offset + (2 * stride) + i]), v2);
                v3 = Vector256.FusedMultiplyAdd(xv, Vector256.LoadUnsafe(in w[offset + (3 * stride) + i]), v3);
            }

            s0 = Vector256.Sum(v0);
            s1 = Vector256.Sum(v1);
            s2 = Vector256.Sum(v2);
            s3 = Vector256.Sum(v3);
        }

        for (; i < length; i++)
        {
            float xv = x[i];
            s0 += xv * w[offset + i];
            s1 += xv * w[offset + stride + i];
            s2 += xv * w[offset + (2 * stride) + i];
            s3 += xv * w[offset + (3 * stride) + i];
        }

        a0 = s0;
        a1 = s1;
        a2 = s2;
        a3 = s3;
    }

    /// <summary>Dot product of two float32 spans, accumulated in float32.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int length = Math.Min(a.Length, b.Length);
        int i = 0;
        float sum = 0f;

        if (Vector512.IsHardwareAccelerated && length >= Vector512<float>.Count)
        {
            Vector512<float> acc = Vector512<float>.Zero;
            for (; i <= length - Vector512<float>.Count; i += Vector512<float>.Count)
            {
                acc = Vector512.FusedMultiplyAdd(
                    Vector512.LoadUnsafe(in a[i]), Vector512.LoadUnsafe(in b[i]), acc);
            }

            sum = Vector512.Sum(acc);
        }
        else if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            Vector256<float> acc = Vector256<float>.Zero;
            for (; i <= length - Vector256<float>.Count; i += Vector256<float>.Count)
            {
                acc = Vector256.FusedMultiplyAdd(
                    Vector256.LoadUnsafe(in a[i]), Vector256.LoadUnsafe(in b[i]), acc);
            }

            sum = Vector256.Sum(acc);
        }
        else if (Vector128.IsHardwareAccelerated && length >= Vector128<float>.Count)
        {
            Vector128<float> acc = Vector128<float>.Zero;
            for (; i <= length - Vector128<float>.Count; i += Vector128<float>.Count)
            {
                acc = Vector128.FusedMultiplyAdd(
                    Vector128.LoadUnsafe(in a[i]), Vector128.LoadUnsafe(in b[i]), acc);
            }

            sum = Vector128.Sum(acc);
        }

        for (; i < length; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }
}
