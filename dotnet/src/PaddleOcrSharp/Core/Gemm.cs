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

        long work = (long)rows * cols * inner;
        int threads = Environment.ProcessorCount;

        if (work < ParallelThreshold || threads <= 1)
        {
            RunRowRange(x.Span, inner, weight, bias.Span, y.Span, cols, 0, rows);
            return;
        }

        // Split along m so every thread writes a disjoint, contiguous run of output rows.
        // When m is small (decoding a single token) split along n instead.
        if (rows >= threads)
        {
            int chunk = (rows + threads - 1) / threads;
            Parallel.For(0, (rows + chunk - 1) / chunk, block =>
            {
                int start = block * chunk;
                RunRowRange(
                    x.Span, inner, weight, bias.Span, y.Span, cols, start, Math.Min(chunk, rows - start));
            });
        }
        else
        {
            int chunk = Math.Max(ColBlock, (cols + threads - 1) / threads);
            chunk = (chunk + ColBlock - 1) / ColBlock * ColBlock;
            Parallel.For(0, (cols + chunk - 1) / chunk, block =>
            {
                int start = block * chunk;
                RunColRange(
                    x.Span, rows, inner, weight, bias.Span, y.Span, cols, start, Math.Min(chunk, cols - start));
            });
        }
    }

    private static void RunRowRange(
        ReadOnlySpan<float> x,
        int inner,
        WeightMatrix weight,
        ReadOnlySpan<float> bias,
        Span<float> y,
        int cols,
        int rowStart,
        int rowCount)
    {
        for (int m = rowStart; m < rowStart + rowCount; m++)
        {
            RunTile(x, inner, weight, bias, y, cols, m, 1, 0, cols);
        }
    }

    private static void RunColRange(
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
        int m = 0;
        for (; m <= rows - RowBlock; m += RowBlock)
        {
            RunTile(x, inner, weight, bias, y, cols, m, RowBlock, colStart, colCount);
        }

        for (; m < rows; m++)
        {
            RunTile(x, inner, weight, bias, y, cols, m, 1, colStart, colCount);
        }
    }

    private static void RunTile(
        ReadOnlySpan<float> x,
        int inner,
        WeightMatrix weight,
        ReadOnlySpan<float> bias,
        Span<float> y,
        int cols,
        int rowStart,
        int rowCount,
        int colStart,
        int colCount)
    {
        int colEnd = colStart + colCount;

        for (int m = rowStart; m < rowStart + rowCount; m++)
        {
            ReadOnlySpan<float> xr = x.Slice(m * inner, inner);
            Span<float> yr = y.Slice(m * cols, cols);

            int n = colStart;
            for (; n <= colEnd - ColBlock; n += ColBlock)
            {
                weight.Dot4(xr, n, out float a0, out float a1, out float a2, out float a3);
                yr[n] = a0;
                yr[n + 1] = a1;
                yr[n + 2] = a2;
                yr[n + 3] = a3;
            }

            for (; n < colEnd; n++)
            {
                yr[n] = weight.Dot(xr, n);
            }

            if (!bias.IsEmpty)
            {
                Kernels.AddInPlace(yr[colStart..colEnd], bias[colStart..colEnd]);
            }
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
