using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace PaddleOcrSharp.Core;

/// <summary>
/// A read-only <c>[rows, cols]</c> weight matrix that stays in its on-disk dtype and is widened
/// to float32 inside the inner loop.
/// </summary>
/// <remarks>
/// Only the two dtypes the shipped checkpoints use are given fast paths: bfloat16 (PaddleOCR-VL)
/// and float32 (PP-DocLayoutV3). Everything else must be converted at load time.
/// </remarks>
public readonly struct WeightMatrix
{
    private readonly ReadOnlyMemory<byte> _bytes;

    private WeightMatrix(ReadOnlyMemory<byte> bytes, DType dtype, int rows, int cols)
    {
        _bytes = bytes;
        Dtype = dtype;
        Rows = rows;
        Cols = cols;
    }

    /// <summary>Storage dtype.</summary>
    public DType Dtype { get; }

    /// <summary>Number of rows (the <c>out_features</c> of an <c>nn.Linear</c>).</summary>
    public int Rows { get; }

    /// <summary>Number of columns (the <c>in_features</c> of an <c>nn.Linear</c>).</summary>
    public int Cols { get; }

    /// <summary><see langword="true"/> when no storage is attached.</summary>
    public bool IsEmpty => _bytes.IsEmpty;

    /// <summary>Wraps raw bytes as a weight matrix.</summary>
    public static WeightMatrix Create(ReadOnlyMemory<byte> bytes, DType dtype, int rows, int cols)
    {
        long expected = (long)rows * cols * dtype.ByteSize();
        if (bytes.Length < expected)
        {
            throw new ArgumentException(
                $"Need {expected} bytes for a [{rows}, {cols}] {dtype} matrix but only {bytes.Length} are available.",
                nameof(bytes));
        }

        if (dtype is not (DType.Float32 or DType.BFloat16))
        {
            throw new NotSupportedException(
                $"{dtype} weights must be converted to float32 or bfloat16 before use.");
        }

        return new WeightMatrix(bytes, dtype, rows, cols);
    }

    /// <summary>Wraps a float32 array as a weight matrix.</summary>
    public static WeightMatrix FromFloats(float[] values, int rows, int cols) =>
        Create(MemoryMarshal.AsBytes<float>(values).ToArray(), DType.Float32, rows, cols);

    /// <summary>Dot product of <paramref name="x"/> with row <paramref name="row"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float Dot(ReadOnlySpan<float> x, int row)
    {
        if (Dtype == DType.Float32)
        {
            return Gemm.Dot(x, MemoryMarshal.Cast<byte, float>(_bytes.Span).Slice(row * Cols, Cols));
        }

        return DotBF16(x, MemoryMarshal.Cast<byte, ushort>(_bytes.Span).Slice(row * Cols, Cols));
    }

    /// <summary>
    /// Dot products of <paramref name="x"/> with rows <c>row</c>..<c>row + 3</c>.
    /// </summary>
    /// <remarks>
    /// Doing four rows at once keeps <paramref name="x"/> in registers across four weight streams,
    /// which is what lifts the kernel off the load ports during prefill.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Dot4(ReadOnlySpan<float> x, int row, out float a0, out float a1, out float a2, out float a3)
    {
        if (Dtype == DType.Float32)
        {
            ReadOnlySpan<float> w = MemoryMarshal.Cast<byte, float>(_bytes.Span);
            int offset = row * Cols;
            a0 = Gemm.Dot(x, w.Slice(offset, Cols));
            a1 = Gemm.Dot(x, w.Slice(offset + Cols, Cols));
            a2 = Gemm.Dot(x, w.Slice(offset + (2 * Cols), Cols));
            a3 = Gemm.Dot(x, w.Slice(offset + (3 * Cols), Cols));
            return;
        }

        ReadOnlySpan<ushort> raw = MemoryMarshal.Cast<byte, ushort>(_bytes.Span);
        int start = row * Cols;
        DotBF16x4(
            x,
            raw.Slice(start, Cols),
            raw.Slice(start + Cols, Cols),
            raw.Slice(start + (2 * Cols), Cols),
            raw.Slice(start + (3 * Cols), Cols),
            out a0,
            out a1,
            out a2,
            out a3);
    }

    /// <summary>
    /// Dot products of four activation rows against four weight rows, sixteen results in one pass
    /// over the shared inner dimension.
    /// </summary>
    /// <remarks>
    /// This is the micro-kernel the blocked GEMM is built on. Holding four activation rows and
    /// four weight rows in registers means each weight element is loaded once and used four times,
    /// which is what turns the projection from bandwidth-bound into compute-bound.
    /// </remarks>
    /// <param name="x0">First activation row.</param>
    /// <param name="x1">Second activation row.</param>
    /// <param name="x2">Third activation row.</param>
    /// <param name="x3">Fourth activation row.</param>
    /// <param name="row">Index of the first of four consecutive weight rows.</param>
    /// <param name="results">Receives 16 results, activation-major: <c>results[i * 4 + j]</c>.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Dot4x4(
        ReadOnlySpan<float> x0,
        ReadOnlySpan<float> x1,
        ReadOnlySpan<float> x2,
        ReadOnlySpan<float> x3,
        int row,
        Span<float> results)
    {
        int length = Cols;
        int offset = row * length;

        if (Dtype == DType.Float32)
        {
            ReadOnlySpan<float> w = MemoryMarshal.Cast<byte, float>(_bytes.Span);
            for (int j = 0; j < 4; j++)
            {
                ReadOnlySpan<float> wj = w.Slice(offset + (j * length), length);
                results[j] = Gemm.Dot(x0, wj);
                results[4 + j] = Gemm.Dot(x1, wj);
                results[8 + j] = Gemm.Dot(x2, wj);
                results[12 + j] = Gemm.Dot(x3, wj);
            }

            return;
        }

        ReadOnlySpan<ushort> raw = MemoryMarshal.Cast<byte, ushort>(_bytes.Span);
        int i = 0;
        Span<float> sums = results[..16];
        sums.Clear();

        if (Vector256.IsHardwareAccelerated && length >= 8)
        {
            Vector256<float> a00 = Vector256<float>.Zero, a01 = Vector256<float>.Zero;
            Vector256<float> a02 = Vector256<float>.Zero, a03 = Vector256<float>.Zero;
            Vector256<float> a10 = Vector256<float>.Zero, a11 = Vector256<float>.Zero;
            Vector256<float> a12 = Vector256<float>.Zero, a13 = Vector256<float>.Zero;
            Vector256<float> a20 = Vector256<float>.Zero, a21 = Vector256<float>.Zero;
            Vector256<float> a22 = Vector256<float>.Zero, a23 = Vector256<float>.Zero;
            Vector256<float> a30 = Vector256<float>.Zero, a31 = Vector256<float>.Zero;
            Vector256<float> a32 = Vector256<float>.Zero, a33 = Vector256<float>.Zero;

            for (; i <= length - 8; i += 8)
            {
                Vector256<float> w0 = WidenLow(Vector128.LoadUnsafe(in raw[offset + i]));
                Vector256<float> w1 = WidenLow(Vector128.LoadUnsafe(in raw[offset + length + i]));
                Vector256<float> w2 = WidenLow(Vector128.LoadUnsafe(in raw[offset + (2 * length) + i]));
                Vector256<float> w3 = WidenLow(Vector128.LoadUnsafe(in raw[offset + (3 * length) + i]));

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

            sums[0] = Vector256.Sum(a00);
            sums[1] = Vector256.Sum(a01);
            sums[2] = Vector256.Sum(a02);
            sums[3] = Vector256.Sum(a03);
            sums[4] = Vector256.Sum(a10);
            sums[5] = Vector256.Sum(a11);
            sums[6] = Vector256.Sum(a12);
            sums[7] = Vector256.Sum(a13);
            sums[8] = Vector256.Sum(a20);
            sums[9] = Vector256.Sum(a21);
            sums[10] = Vector256.Sum(a22);
            sums[11] = Vector256.Sum(a23);
            sums[12] = Vector256.Sum(a30);
            sums[13] = Vector256.Sum(a31);
            sums[14] = Vector256.Sum(a32);
            sums[15] = Vector256.Sum(a33);
        }

        for (; i < length; i++)
        {
            float w0 = FloatConversion.BF16ToFloat(raw[offset + i]);
            float w1 = FloatConversion.BF16ToFloat(raw[offset + length + i]);
            float w2 = FloatConversion.BF16ToFloat(raw[offset + (2 * length) + i]);
            float w3 = FloatConversion.BF16ToFloat(raw[offset + (3 * length) + i]);

            sums[0] += x0[i] * w0;
            sums[1] += x0[i] * w1;
            sums[2] += x0[i] * w2;
            sums[3] += x0[i] * w3;
            sums[4] += x1[i] * w0;
            sums[5] += x1[i] * w1;
            sums[6] += x1[i] * w2;
            sums[7] += x1[i] * w3;
            sums[8] += x2[i] * w0;
            sums[9] += x2[i] * w1;
            sums[10] += x2[i] * w2;
            sums[11] += x2[i] * w3;
            sums[12] += x3[i] * w0;
            sums[13] += x3[i] * w1;
            sums[14] += x3[i] * w2;
            sums[15] += x3[i] * w3;
        }
    }

    /// <summary>Widens 8 packed bfloat16 values into one float32 vector.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> WidenLow(Vector128<ushort> raw)
    {
        (Vector128<uint> lo, Vector128<uint> hi) = Vector128.Widen(raw);
        return Vector256.Create((lo << 16).AsSingle(), (hi << 16).AsSingle());
    }

    /// <summary>Widens row <paramref name="row"/> into <paramref name="destination"/>.</summary>
    public void CopyRow(int row, Span<float> destination)
    {
        if (Dtype == DType.Float32)
        {
            MemoryMarshal.Cast<byte, float>(_bytes.Span).Slice(row * Cols, Cols).CopyTo(destination);
        }
        else
        {
            FloatConversion.BF16ToFloat(
                MemoryMarshal.Cast<byte, ushort>(_bytes.Span).Slice(row * Cols, Cols), destination);
        }
    }

    /// <summary>
    /// Widens <paramref name="count"/> consecutive rows starting at <paramref name="row"/> into
    /// <paramref name="destination"/>, row-major.
    /// </summary>
    public void CopyRows(int row, int count, Span<float> destination)
    {
        int start = row * Cols;
        int length = count * Cols;

        if (Dtype == DType.Float32)
        {
            MemoryMarshal.Cast<byte, float>(_bytes.Span).Slice(start, length).CopyTo(destination);
        }
        else
        {
            FloatConversion.BF16ToFloat(
                MemoryMarshal.Cast<byte, ushort>(_bytes.Span).Slice(start, length), destination);
        }
    }

    /// <summary>Widens the whole matrix into <paramref name="destination"/>, row-major.</summary>
    public void CopyTo(Span<float> destination)
    {
        if (Dtype == DType.Float32)
        {
            MemoryMarshal.Cast<byte, float>(_bytes.Span)[..(Rows * Cols)].CopyTo(destination);
        }
        else
        {
            FloatConversion.BF16ToFloat(
                MemoryMarshal.Cast<byte, ushort>(_bytes.Span)[..(Rows * Cols)], destination);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotBF16(ReadOnlySpan<float> x, ReadOnlySpan<ushort> w)
    {
        int length = w.Length;
        int i = 0;
        float sum = 0f;

        if (Vector256.IsHardwareAccelerated && length >= 16)
        {
            Vector256<float> acc0 = Vector256<float>.Zero;
            Vector256<float> acc1 = Vector256<float>.Zero;
            for (; i <= length - 16; i += 16)
            {
                (Vector256<float> lo, Vector256<float> hi) = WidenBF16(Vector256.LoadUnsafe(in w[i]));
                acc0 = Vector256.FusedMultiplyAdd(Vector256.LoadUnsafe(in x[i]), lo, acc0);
                acc1 = Vector256.FusedMultiplyAdd(Vector256.LoadUnsafe(in x[i + 8]), hi, acc1);
            }

            sum = Vector256.Sum(acc0 + acc1);
        }
        else if (Vector128.IsHardwareAccelerated && length >= 8)
        {
            Vector128<float> acc0 = Vector128<float>.Zero;
            Vector128<float> acc1 = Vector128<float>.Zero;
            for (; i <= length - 8; i += 8)
            {
                (Vector128<float> lo, Vector128<float> hi) = WidenBF16(Vector128.LoadUnsafe(in w[i]));
                acc0 = Vector128.FusedMultiplyAdd(Vector128.LoadUnsafe(in x[i]), lo, acc0);
                acc1 = Vector128.FusedMultiplyAdd(Vector128.LoadUnsafe(in x[i + 4]), hi, acc1);
            }

            sum = Vector128.Sum(acc0 + acc1);
        }

        for (; i < length; i++)
        {
            sum += x[i] * FloatConversion.BF16ToFloat(w[i]);
        }

        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DotBF16x4(
        ReadOnlySpan<float> x,
        ReadOnlySpan<ushort> w0,
        ReadOnlySpan<ushort> w1,
        ReadOnlySpan<ushort> w2,
        ReadOnlySpan<ushort> w3,
        out float a0,
        out float a1,
        out float a2,
        out float a3)
    {
        int length = x.Length;
        int i = 0;
        float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;

        if (Vector256.IsHardwareAccelerated && length >= 16)
        {
            Vector256<float> v0 = Vector256<float>.Zero;
            Vector256<float> v1 = Vector256<float>.Zero;
            Vector256<float> v2 = Vector256<float>.Zero;
            Vector256<float> v3 = Vector256<float>.Zero;

            for (; i <= length - 16; i += 16)
            {
                Vector256<float> xLo = Vector256.LoadUnsafe(in x[i]);
                Vector256<float> xHi = Vector256.LoadUnsafe(in x[i + 8]);

                (Vector256<float> lo, Vector256<float> hi) = WidenBF16(Vector256.LoadUnsafe(in w0[i]));
                v0 = Vector256.FusedMultiplyAdd(xLo, lo, v0);
                v0 = Vector256.FusedMultiplyAdd(xHi, hi, v0);

                (lo, hi) = WidenBF16(Vector256.LoadUnsafe(in w1[i]));
                v1 = Vector256.FusedMultiplyAdd(xLo, lo, v1);
                v1 = Vector256.FusedMultiplyAdd(xHi, hi, v1);

                (lo, hi) = WidenBF16(Vector256.LoadUnsafe(in w2[i]));
                v2 = Vector256.FusedMultiplyAdd(xLo, lo, v2);
                v2 = Vector256.FusedMultiplyAdd(xHi, hi, v2);

                (lo, hi) = WidenBF16(Vector256.LoadUnsafe(in w3[i]));
                v3 = Vector256.FusedMultiplyAdd(xLo, lo, v3);
                v3 = Vector256.FusedMultiplyAdd(xHi, hi, v3);
            }

            s0 = Vector256.Sum(v0);
            s1 = Vector256.Sum(v1);
            s2 = Vector256.Sum(v2);
            s3 = Vector256.Sum(v3);
        }

        for (; i < length; i++)
        {
            float xv = x[i];
            s0 += xv * FloatConversion.BF16ToFloat(w0[i]);
            s1 += xv * FloatConversion.BF16ToFloat(w1[i]);
            s2 += xv * FloatConversion.BF16ToFloat(w2[i]);
            s3 += xv * FloatConversion.BF16ToFloat(w3[i]);
        }

        a0 = s0;
        a1 = s1;
        a2 = s2;
        a3 = s3;
    }

    /// <summary>Splits 16 packed bfloat16 values into two float32 vectors.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (Vector256<float> Low, Vector256<float> High) WidenBF16(Vector256<ushort> raw)
    {
        (Vector256<uint> lo, Vector256<uint> hi) = Vector256.Widen(raw);
        return ((lo << 16).AsSingle(), (hi << 16).AsSingle());
    }

    /// <summary>Splits 8 packed bfloat16 values into two float32 vectors.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (Vector128<float> Low, Vector128<float> High) WidenBF16(Vector128<ushort> raw)
    {
        (Vector128<uint> lo, Vector128<uint> hi) = Vector128.Widen(raw);
        return ((lo << 16).AsSingle(), (hi << 16).AsSingle());
    }
}
