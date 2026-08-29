using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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

        RunPanelsInParallel(panels, panel, x, rows, inner, weight, bias, y, cols);
    }

    /// <summary>
    /// Runs the column panels concurrently, kept out of <see cref="Linear"/> for the same reason
    /// as <see cref="RunTilesInParallel"/>: a method holding a lambda pays for it on entry, and
    /// <see cref="Linear"/> has a serial branch that a decode step takes many times per token.
    /// </summary>
    private static void RunPanelsInParallel(
        int panels,
        int panel,
        ReadOnlyMemory<float> x,
        int rows,
        int inner,
        WeightMatrix weight,
        ReadOnlyMemory<float> bias,
        Memory<float> y,
        int cols) =>
        Parallel.For(0, panels, index =>
        {
            int start = index * panel;
            RunPanel(
                x.Span, rows, inner, weight, bias.Span, y.Span, cols, start, Math.Min(panel, cols - start));
        });

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
        int colEnd = colStart + colCount;

        // Widening only pays when the panel is reused. Decoding a single token reads every weight
        // exactly once, so there the fused path that widens inside the dot product is strictly
        // cheaper — and the output projection is a 103424-row matrix, where the difference is the
        // whole step.
        if (rows < RowBlock)
        {
            RunNarrow(x, rows, inner, weight, bias, y, cols, colStart, colCount);
            return;
        }

        using PooledBuffer panel = TensorPool.Rent(colCount * inner);
        Span<float> w = panel.Span;
        weight.CopyRows(colStart, colCount, w);

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

        if (m < rows)
        {
            RunNarrow(x[(m * inner)..], rows - m, inner, weight, bias, y[(m * cols)..], cols, colStart, colCount);
        }
    }

    /// <summary>
    /// Computes one column panel for fewer rows than the register block, reading the weights in
    /// their stored dtype.
    /// </summary>
    private static void RunNarrow(
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
        int colEnd = colStart + colCount;

        for (int m = 0; m < rows; m++)
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

        if (Simd.Use512 && length >= Vector512<float>.Count)
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
        else if (Simd.Use256 && length >= Vector256<float>.Count)
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
    /// Computes <c>y[m, n] = Σ_k a[m, k] · b[k, n]</c> for two dense float32 matrices, with
    /// either operand optionally stored transposed.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The layout decides the kernel. When the right-hand matrix is stored transposed its rows
    /// are the reduction vectors, so each output element is a dot product of two contiguous runs
    /// and <see cref="Dot4"/> applies; otherwise the reduction walks down its columns, so the
    /// loop instead accumulates whole output rows with <see cref="Kernels.AddScaled4"/>. Both
    /// keep four streams in flight, which is what makes the inner loop compute-bound.
    /// </para>
    /// <para>
    /// The output is tiled and the tiles run in parallel above a work threshold. Tiling over
    /// both axes rather than rows alone matters because the shapes here are often lopsided — a
    /// deformable-attention product is 13 125 rows by 256 columns, a mask product 300 by 40 000.
    /// </para>
    /// </remarks>
    /// <param name="a">The left matrix, <c>m × k</c> row-major, or <c>k × m</c> when transposed.</param>
    /// <param name="m">Rows of the product.</param>
    /// <param name="k">The reduction length.</param>
    /// <param name="transposeA">Whether <paramref name="a"/> is stored transposed.</param>
    /// <param name="b">The right matrix, <c>k × n</c> row-major, or <c>n × k</c> when transposed.</param>
    /// <param name="n">Columns of the product.</param>
    /// <param name="transposeB">Whether <paramref name="b"/> is stored transposed.</param>
    /// <param name="y">The <c>m × n</c> product, written in full.</param>
    /// <param name="allowParallel">
    /// Whether the tiles may run in parallel. Callers that are already inside a parallel loop —
    /// the convolution, which threads over output rows — pass <see langword="false"/> rather than
    /// nesting a second level of parallelism inside each of their own work items.
    /// </param>
    public static void MatMul(
        ReadOnlyMemory<float> a,
        int m,
        int k,
        bool transposeA,
        ReadOnlyMemory<float> b,
        int n,
        bool transposeB,
        Memory<float> y,
        bool allowParallel = true)
    {
        if (m == 0 || n == 0)
        {
            return;
        }

        if (k == 0)
        {
            y.Span[..(m * n)].Clear();
            return;
        }

        // Every kernel below reads the left matrix a row at a time, so a transposed one is
        // materialised once rather than gathered on every pass.
        using PooledBuffer transposed = transposeA ? TensorPool.Rent(m * k) : default;
        ReadOnlyMemory<float> left = a;

        if (transposeA)
        {
            Transpose(a.Span, k, m, transposed.Span);
            left = transposed.Memory;
        }

        // A column tile of the right-hand matrix should stay resident across the reduction.
        int columnBlock = transposeB ? 64 : Math.Min(n, Math.Max(64, (1 << 15) / Math.Max(1, k)));
        int rowBlock = 64;

        int rowTiles = ((m - 1) / rowBlock) + 1;
        int columnTiles = ((n - 1) / columnBlock) + 1;
        int tiles = rowTiles * columnTiles;

        // Neither branch may put a lambda in this method's body: attention calls it thousands of
        // times per layer on the serial path, and a method containing a lambda allocates its
        // display class on entry regardless of which branch runs. Hence the static helpers.
        if (allowParallel && tiles > 1 && (long)m * k * n >= ParallelThreshold)
        {
            RunTilesInParallel(tiles, left, b, y, m, k, n, transposeB, rowBlock, columnBlock, columnTiles);
        }
        else
        {
            for (int tile = 0; tile < tiles; tile++)
            {
                RunTile(tile, left, b, y, m, k, n, transposeB, rowBlock, columnBlock, columnTiles);
            }
        }
    }

    /// <summary>
    /// Runs the tiles concurrently. Separate from <see cref="MatMul"/> because a method that
    /// contains a lambda allocates its display class on entry whether or not the branch holding
    /// the lambda is taken, and the serial path here is the hot one.
    /// </summary>
    private static void RunTilesInParallel(
        int tiles,
        ReadOnlyMemory<float> a,
        ReadOnlyMemory<float> b,
        Memory<float> y,
        int m,
        int k,
        int n,
        bool transposeB,
        int rowBlock,
        int columnBlock,
        int columnTiles) =>
        Parallel.For(0, tiles, tile =>
            RunTile(tile, a, b, y, m, k, n, transposeB, rowBlock, columnBlock, columnTiles));

    /// <summary>Computes one tile of the output.</summary>
    private static void RunTile(
        int tile,
        ReadOnlyMemory<float> a,
        ReadOnlyMemory<float> b,
        Memory<float> y,
        int m,
        int k,
        int n,
        bool transposeB,
        int rowBlock,
        int columnBlock,
        int columnTiles)
    {
        int row = (tile / columnTiles) * rowBlock;
        int column = (tile % columnTiles) * columnBlock;
        int rows = Math.Min(rowBlock, m - row);
        int columns = Math.Min(columnBlock, n - column);

        if (transposeB)
        {
            TransposedTile(a.Span, b.Span, y.Span, k, n, row, rows, column, columns);
        }
        else
        {
            DirectTile(a.Span, b.Span, y.Span, k, n, row, rows, column, columns);
        }
    }

    /// <summary>One output tile when the right-hand matrix is stored <c>n × k</c>.</summary>
    private static void TransposedTile(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> y,
        int k,
        int n,
        int row,
        int rows,
        int column,
        int columns)
    {
        for (int i = 0; i < rows; i++)
        {
            ReadOnlySpan<float> left = a.Slice((row + i) * k, k);
            Span<float> destination = y.Slice(((row + i) * n) + column, columns);

            int j = 0;
            for (; j <= columns - 4; j += 4)
            {
                Dot4(left, b, ((column + j) * k) + 0, k, out float a0, out float a1, out float a2, out float a3);
                destination[j] = a0;
                destination[j + 1] = a1;
                destination[j + 2] = a2;
                destination[j + 3] = a3;
            }

            for (; j < columns; j++)
            {
                destination[j] = Dot(left, b.Slice((column + j) * k, k));
            }
        }
    }

    /// <summary>One output tile when the right-hand matrix is stored <c>k × n</c>.</summary>
    private static void DirectTile(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> y,
        int k,
        int n,
        int row,
        int rows,
        int column,
        int columns)
    {
        int i = 0;
        for (; i <= rows - 4; i += 4)
        {
            Span<float> d0 = y.Slice(((row + i) * n) + column, columns);
            Span<float> d1 = y.Slice(((row + i + 1) * n) + column, columns);
            Span<float> d2 = y.Slice(((row + i + 2) * n) + column, columns);
            Span<float> d3 = y.Slice(((row + i + 3) * n) + column, columns);

            d0.Clear();
            d1.Clear();
            d2.Clear();
            d3.Clear();

            // References are taken once for the whole reduction. Re-slicing the right-hand row and
            // calling out per term costs a span construction, a broadcast quad and a call for four
            // multiply-adds of work, which is most of what this loop was spending: the reduction
            // axis here is the token count, so it runs thousands of times for one small tile.
            ref float aRef = ref MemoryMarshal.GetReference(a);
            ref float bRef = ref MemoryMarshal.GetReference(b);
            ref float dest0 = ref MemoryMarshal.GetReference(d0);
            ref float dest1 = ref MemoryMarshal.GetReference(d1);
            ref float dest2 = ref MemoryMarshal.GetReference(d2);
            ref float dest3 = ref MemoryMarshal.GetReference(d3);

            int base0 = (row + i) * k;

            for (int p = 0; p < k; p++)
            {
                float s0 = Unsafe.Add(ref aRef, base0 + p);
                float s1 = Unsafe.Add(ref aRef, base0 + k + p);
                float s2 = Unsafe.Add(ref aRef, base0 + (2 * k) + p);
                float s3 = Unsafe.Add(ref aRef, base0 + (3 * k) + p);

                // Masks and attention weights are genuinely sparse; skipping a whole quad of
                // zeros is worth the four compares. Written out rather than as a tuple comparison,
                // which does not reduce to four compares.
                if (s0 == 0f && s1 == 0f && s2 == 0f && s3 == 0f)
                {
                    continue;
                }

                AccumulateQuad(
                    ref Unsafe.Add(ref bRef, (p * n) + column),
                    ref dest0, ref dest1, ref dest2, ref dest3,
                    columns, s0, s1, s2, s3);
            }
        }

        for (; i < rows; i++)
        {
            Span<float> destination = y.Slice(((row + i) * n) + column, columns);
            destination.Clear();

            int offset = (row + i) * k;
            for (int p = 0; p < k; p++)
            {
                float scale = a[offset + p];
                if (scale != 0f)
                {
                    Kernels.AddScaled(destination, b.Slice((p * n) + column, columns), scale);
                }
            }
        }
    }

    /// <summary>
    /// Adds one right-hand row, scaled four ways, into four destination rows.
    /// </summary>
    /// <remarks>
    /// The reference-taking form of <see cref="Kernels.AddScaled4"/>, inlined into the reduction so
    /// the caller pays neither a span construction nor a call per term.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateQuad(
        ref float source,
        ref float d0,
        ref float d1,
        ref float d2,
        ref float d3,
        int length,
        float s0,
        float s1,
        float s2,
        float s3)
    {
        int i = 0;

        if (Simd.Use256 && length >= Vector256<float>.Count)
        {
            Vector256<float> v0 = Vector256.Create(s0);
            Vector256<float> v1 = Vector256.Create(s1);
            Vector256<float> v2 = Vector256.Create(s2);
            Vector256<float> v3 = Vector256.Create(s3);

            for (; i <= length - Vector256<float>.Count; i += Vector256<float>.Count)
            {
                var offset = (nuint)i;
                Vector256<float> x = Vector256.LoadUnsafe(ref source, offset);
                Vector256.FusedMultiplyAdd(x, v0, Vector256.LoadUnsafe(ref d0, offset))
                    .StoreUnsafe(ref d0, offset);
                Vector256.FusedMultiplyAdd(x, v1, Vector256.LoadUnsafe(ref d1, offset))
                    .StoreUnsafe(ref d1, offset);
                Vector256.FusedMultiplyAdd(x, v2, Vector256.LoadUnsafe(ref d2, offset))
                    .StoreUnsafe(ref d2, offset);
                Vector256.FusedMultiplyAdd(x, v3, Vector256.LoadUnsafe(ref d3, offset))
                    .StoreUnsafe(ref d3, offset);
            }
        }

        for (; i < length; i++)
        {
            float x = Unsafe.Add(ref source, i);
            Unsafe.Add(ref d0, i) += x * s0;
            Unsafe.Add(ref d1, i) += x * s1;
            Unsafe.Add(ref d2, i) += x * s2;
            Unsafe.Add(ref d3, i) += x * s3;
        }
    }

    /// <summary>Transposes a <paramref name="rows"/> × <paramref name="columns"/> matrix.</summary>
    private static void Transpose(ReadOnlySpan<float> source, int rows, int columns, Span<float> destination)
    {
        const int Block = 32;

        for (int i0 = 0; i0 < rows; i0 += Block)
        {
            int iEnd = Math.Min(i0 + Block, rows);
            for (int j0 = 0; j0 < columns; j0 += Block)
            {
                int jEnd = Math.Min(j0 + Block, columns);
                for (int i = i0; i < iEnd; i++)
                {
                    for (int j = j0; j < jEnd; j++)
                    {
                        destination[(j * rows) + i] = source[(i * columns) + j];
                    }
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

        if (Simd.Use256 && length >= Vector256<float>.Count)
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

        if (Simd.Use512 && length >= Vector512<float>.Count)
        {
            Vector512<float> acc = Vector512<float>.Zero;
            for (; i <= length - Vector512<float>.Count; i += Vector512<float>.Count)
            {
                acc = Vector512.FusedMultiplyAdd(
                    Vector512.LoadUnsafe(in a[i]), Vector512.LoadUnsafe(in b[i]), acc);
            }

            sum = Vector512.Sum(acc);
        }
        else if (Simd.Use256 && length >= Vector256<float>.Count)
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
