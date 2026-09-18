using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using PaddleOcrSharp.Formats.Gguf;
using PaddleOcrSharp.Quantization;

namespace PaddleOcrSharp.Core;

/// <summary>
/// A read-only <c>[rows, cols]</c> weight matrix that stays in its on-disk dtype and is widened
/// to float32 inside the inner loop.
/// </summary>
/// <remarks>
/// <para>
/// Three storage kinds have fast paths: bfloat16 (PaddleOCR-VL), float32 (PP-DocLayoutV3) and the
/// ternary block layouts of a quantized checkpoint. Everything else must be converted at load time.
/// </para>
/// <para>
/// A quantized matrix behaves like any other here: it answers the same four questions — one dot,
/// four dots, one row widened, many rows widened — so every caller above this type, and the whole
/// of <see cref="Gemm"/>, is unchanged by quantization. What differs is only where the win lands.
/// <see cref="CopyRows"/> serves the panel kernel, which decodes a column panel once and reuses it
/// across every activation row, so prefill gains footprint and not time; <see cref="Dot4"/> serves
/// a decode step, which reads every weight exactly once and is bound by how fast the bytes arrive,
/// so there the win is the whole point.
/// </para>
/// </remarks>
public readonly struct WeightMatrix
{
    private readonly ReadOnlyMemory<byte> _bytes;
    private readonly int _rowBytes;

    private WeightMatrix(
        ReadOnlyMemory<byte> bytes,
        DType dtype,
        GgmlType quantization,
        int rows,
        int cols,
        int rowBytes,
        HadamardRotation? rotation,
        string? name)
    {
        _bytes = bytes;
        _rowBytes = rowBytes;
        Dtype = dtype;
        Quantization = quantization;
        Rows = rows;
        Cols = cols;
        Rotation = rotation;
        Name = name;
    }

    /// <summary>
    /// The checkpoint name this matrix was loaded under, when it came from a store that knows it.
    /// </summary>
    /// <remarks>
    /// It exists for one reason: <see cref="ActivationRecorder"/> accumulates a Hessian per
    /// <i>input site</i> while the unquantized model runs over a calibration set, and the only
    /// thing that names a site is the weight the activation is about to meet. Nothing on the
    /// inference path reads it.
    /// </remarks>
    public string? Name { get; }

    /// <summary>Storage dtype. Meaningful only when <see cref="IsQuantized"/> is false.</summary>
    public DType Dtype { get; }

    /// <summary>
    /// Block layout of a quantized matrix; <see cref="GgmlType.F32"/> when the matrix is not
    /// quantized.
    /// </summary>
    public GgmlType Quantization { get; }

    /// <summary>Whether the storage is packed ternary blocks rather than plain elements.</summary>
    public bool IsQuantized => Quantization.IsQuantized();

    /// <summary>
    /// The basis the weights are stored in, when they were folded by a rotation the activation
    /// must be transformed by before the product; <see langword="null"/> otherwise.
    /// </summary>
    public HadamardRotation? Rotation { get; }

    /// <summary>Bytes one row of the matrix occupies.</summary>
    public int RowByteLength => _rowBytes;

    /// <summary>Number of rows (the <c>out_features</c> of an <c>nn.Linear</c>).</summary>
    public int Rows { get; }

    /// <summary>Number of columns (the <c>in_features</c> of an <c>nn.Linear</c>).</summary>
    public int Cols { get; }

    /// <summary><see langword="true"/> when no storage is attached.</summary>
    public bool IsEmpty => _bytes.IsEmpty;

    /// <summary>
    /// The storage as float32, when it already is — so a caller that would otherwise widen a
    /// panel into scratch can read the rows where they lie.
    /// </summary>
    /// <returns><see langword="true"/> and the values when the dtype is float32.</returns>
    /// <param name="values">Receives the matrix, row-major, on success.</param>
    public bool TryGetFloats(out ReadOnlySpan<float> values)
    {
        if (IsQuantized || Dtype != DType.Float32)
        {
            values = default;
            return false;
        }

        values = MemoryMarshal.Cast<byte, float>(_bytes.Span);
        return true;
    }

    /// <summary>Wraps raw bytes as a weight matrix.</summary>
    /// <param name="bytes">Row-major storage.</param>
    /// <param name="dtype">Storage dtype; float32 or bfloat16.</param>
    /// <param name="rows">Output features.</param>
    /// <param name="cols">Input features.</param>
    /// <param name="name">Optional checkpoint name, for calibration.</param>
    public static WeightMatrix Create(
        ReadOnlyMemory<byte> bytes, DType dtype, int rows, int cols, string? name = null)
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

        return new WeightMatrix(bytes, dtype, GgmlType.F32, rows, cols, cols * dtype.ByteSize(), rotation: null, name);
    }

    /// <summary>Wraps packed ternary blocks as a weight matrix.</summary>
    /// <param name="bytes">The packed rows, each <c>RowSize(cols)</c> bytes long.</param>
    /// <param name="quantization">Block layout; must be a quantized ggml type.</param>
    /// <param name="rows">Output features.</param>
    /// <param name="cols">Input features; must be a multiple of the layout's group size.</param>
    /// <param name="rotation">
    /// The rotation the weights were folded by, or <see langword="null"/> when they were not.
    /// </param>
    /// <param name="name">Optional checkpoint name.</param>
    public static WeightMatrix CreateQuantized(
        ReadOnlyMemory<byte> bytes,
        GgmlType quantization,
        int rows,
        int cols,
        HadamardRotation? rotation = null,
        string? name = null)
    {
        if (!quantization.IsQuantized())
        {
            throw new ArgumentException($"{quantization} is not a quantized layout.", nameof(quantization));
        }

        int rowBytes = checked((int)quantization.RowSize(cols));
        long expected = (long)rows * rowBytes;
        if (bytes.Length < expected)
        {
            throw new ArgumentException(
                $"Need {expected} bytes for a [{rows}, {cols}] {quantization.TypeName()} matrix but only {bytes.Length} are available.",
                nameof(bytes));
        }

        return new WeightMatrix(bytes, DType.UInt8, quantization, rows, cols, rowBytes, rotation, name);
    }

    /// <summary>Wraps a float32 array as a weight matrix.</summary>
    public static WeightMatrix FromFloats(float[] values, int rows, int cols) =>
        Create(MemoryMarshal.AsBytes<float>(values).ToArray(), DType.Float32, rows, cols);

    /// <summary>Dot product of <paramref name="x"/> with row <paramref name="row"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float Dot(ReadOnlySpan<float> x, int row)
    {
        if (IsQuantized)
        {
            using PooledBuffer scratch = TensorPool.Rent(Cols);
            return DotQuantized(x, row, scratch.Span);
        }

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
        if (IsQuantized)
        {
            using PooledBuffer scratch = TensorPool.Rent(Cols);
            a0 = DotQuantized(x, row, scratch.Span);
            a1 = DotQuantized(x, row + 1, scratch.Span);
            a2 = DotQuantized(x, row + 2, scratch.Span);
            a3 = DotQuantized(x, row + 3, scratch.Span);
            return;
        }

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
            row + 8 <= Rows ? 4 * Cols : 0,
            out a0,
            out a1,
            out a2,
            out a3);
    }

    /// <summary>Widens row <paramref name="row"/> into <paramref name="destination"/>.</summary>
    public void CopyRow(int row, Span<float> destination)
    {
        if (IsQuantized)
        {
            TernaryKernels.DecodeRow(
                Quantization, _bytes.Span.Slice(row * _rowBytes, _rowBytes), destination[..Cols]);
            return;
        }

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
        if (IsQuantized)
        {
            TernaryKernels.DecodeRow(
                Quantization,
                _bytes.Span.Slice(row * _rowBytes, count * _rowBytes),
                destination[..(count * Cols)]);
            return;
        }

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
        if (IsQuantized)
        {
            CopyRows(0, Rows, destination);
            return;
        }

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

    /// <summary>
    /// Dot product of <paramref name="x"/> with a packed row, one block at a time.
    /// </summary>
    /// <param name="x">Activation row.</param>
    /// <param name="row">Weight row index.</param>
    /// <param name="scratch">A buffer of at least <see cref="Cols"/> floats, reused across rows.</param>
    /// <remarks>
    /// <para>
    /// The row is decoded into <paramref name="scratch"/> and consumed immediately, so it never
    /// leaves the cache; what crosses the memory system is the packed bytes, which is the traffic a
    /// decode step is bound by. At 1.75 bits a weight that is a ninth of what bfloat16 costs.
    /// </para>
    /// <para>
    /// A whole row at a time, not a block at a time. Decoding block by block put a call and a lane
    /// reduction between every 32 or 128 weights, and the reduction is the expensive half: over a
    /// 4608-long row it turns one horizontal sum into a hundred and forty-four.
    /// </para>
    /// </remarks>
    private float DotQuantized(ReadOnlySpan<float> x, int row, Span<float> scratch)
    {
        TernaryKernels.DecodeRow(
            Quantization, _bytes.Span.Slice(row * _rowBytes, _rowBytes), scratch[..Cols]);

        return Gemm.Dot(x, scratch[..Cols]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotBF16(ReadOnlySpan<float> x, ReadOnlySpan<ushort> w)
    {
        int length = w.Length;
        int i = 0;
        float sum = 0f;

        if (Simd.Use256 && length >= 16)
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

    /// <summary>Issues a prefetch for the line holding <c>row[offset]</c>.</summary>
    /// <remarks>
    /// <para>
    /// The single-row kernel below is what a decode step runs: it reads every weight exactly once
    /// and does one multiply-add with each, so nothing it touches is ever reused and the loop is
    /// bound by how fast the lines arrive. A token costs 646 MB of weights and they arrive as a
    /// hundred and forty short bursts, one per projection per layer.
    /// </para>
    /// <para>
    /// What has to be prefetched is the <i>next group of rows</i>, not further along the current
    /// ones: a row of this model is two to six kilobytes, which the loop crosses in well under the
    /// memory latency, so an offset that stays inside the row arrives too late to be worth
    /// anything. Prefetching at <c>i + 4·Cols</c> from each of the four rows in flight covers the
    /// four rows the next call will read, one line per row per two steps, which is exactly the
    /// rate the loop consumes them at.
    /// </para>
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Prefetch(ref ushort row, int offset)
    {
        if (Sse.IsSupported)
        {
            Sse.Prefetch0(Unsafe.AsPointer(ref Unsafe.Add(ref row, offset)));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DotBF16x4(
        ReadOnlySpan<float> x,
        ReadOnlySpan<ushort> w0,
        ReadOnlySpan<ushort> w1,
        ReadOnlySpan<ushort> w2,
        ReadOnlySpan<ushort> w3,
        int lookahead,
        out float a0,
        out float a1,
        out float a2,
        out float a3)
    {
        int length = x.Length;
        int i = 0;
        float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;

        if (Simd.Use256 && length >= 16)
        {
            Vector256<float> v0 = Vector256<float>.Zero;
            Vector256<float> v1 = Vector256<float>.Zero;
            Vector256<float> v2 = Vector256<float>.Zero;
            Vector256<float> v3 = Vector256<float>.Zero;

            ref ushort p0 = ref MemoryMarshal.GetReference(w0);
            ref ushort p1 = ref MemoryMarshal.GetReference(w1);
            ref ushort p2 = ref MemoryMarshal.GetReference(w2);
            ref ushort p3 = ref MemoryMarshal.GetReference(w3);

            for (; i <= length - 16; i += 16)
            {
                // One line per row every other step: the step consumes 32 bytes of each row, so
                // issuing on every one would ask for the same line twice.
                if (lookahead != 0 && (i & 31) == 0)
                {
                    Prefetch(ref p0, i + lookahead);
                    Prefetch(ref p1, i + lookahead);
                    Prefetch(ref p2, i + lookahead);
                    Prefetch(ref p3, i + lookahead);
                }

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
