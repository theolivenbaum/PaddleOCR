using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace PaddleOcrSharp.Core;

/// <summary>
/// Vectorised conversions between the storage dtypes used by the shipped weights
/// (bfloat16 / float16) and the float32 the kernels compute in.
/// </summary>
/// <remarks>
/// bfloat16 is the top 16 bits of a float32, so widening is a shift and narrowing is a
/// round-to-nearest-even truncation — the same rounding PyTorch uses.
/// </remarks>
public static class FloatConversion
{
    /// <summary>Widens a single bfloat16 bit pattern to float32.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float BF16ToFloat(ushort bits) => BitConverter.UInt32BitsToSingle((uint)bits << 16);

    /// <summary>Narrows a float32 to a bfloat16 bit pattern with round-to-nearest-even.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ushort FloatToBF16(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);

        // NaN must stay NaN; propagate a quiet NaN rather than rounding into infinity.
        if ((bits & 0x7F80_0000u) == 0x7F80_0000u && (bits & 0x007F_FFFFu) != 0)
        {
            return (ushort)((bits >> 16) | 0x0040u);
        }

        uint rounding = 0x7FFFu + ((bits >> 16) & 1u);
        return (ushort)((bits + rounding) >> 16);
    }

    /// <summary>
    /// Widens <paramref name="source"/> bfloat16 bit patterns into <paramref name="destination"/>.
    /// </summary>
    public static void BF16ToFloat(ReadOnlySpan<ushort> source, Span<float> destination)
    {
        if (destination.Length < source.Length)
        {
            throw new ArgumentException("Destination is shorter than source.", nameof(destination));
        }

        int i = 0;

        if (Vector256.IsHardwareAccelerated)
        {
            int width = Vector256<ushort>.Count; // 16 halves -> two 8-wide float vectors
            for (; i <= source.Length - width; i += width)
            {
                Vector256<ushort> raw = Vector256.LoadUnsafe(in source[i]);
                (Vector256<uint> lo, Vector256<uint> hi) = Vector256.Widen(raw);
                Vector256<float> loF = (lo << 16).AsSingle();
                Vector256<float> hiF = (hi << 16).AsSingle();
                loF.StoreUnsafe(ref destination[i]);
                hiF.StoreUnsafe(ref destination[i + Vector256<uint>.Count]);
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            int width = Vector128<ushort>.Count;
            for (; i <= source.Length - width; i += width)
            {
                Vector128<ushort> raw = Vector128.LoadUnsafe(in source[i]);
                (Vector128<uint> lo, Vector128<uint> hi) = Vector128.Widen(raw);
                (lo << 16).AsSingle().StoreUnsafe(ref destination[i]);
                (hi << 16).AsSingle().StoreUnsafe(ref destination[i + Vector128<uint>.Count]);
            }
        }

        for (; i < source.Length; i++)
        {
            destination[i] = BF16ToFloat(source[i]);
        }
    }

    /// <summary>Narrows <paramref name="source"/> float32 values into bfloat16 bit patterns.</summary>
    public static void FloatToBF16(ReadOnlySpan<float> source, Span<ushort> destination)
    {
        if (destination.Length < source.Length)
        {
            throw new ArgumentException("Destination is shorter than source.", nameof(destination));
        }

        for (int i = 0; i < source.Length; i++)
        {
            destination[i] = FloatToBF16(source[i]);
        }
    }

    /// <summary>Widens <paramref name="source"/> float16 values into <paramref name="destination"/>.</summary>
    public static void FP16ToFloat(ReadOnlySpan<Half> source, Span<float> destination)
    {
        if (destination.Length < source.Length)
        {
            throw new ArgumentException("Destination is shorter than source.", nameof(destination));
        }

        // float16 only shows up in third-party conversions of these checkpoints; a scalar loop
        // is fine because the hot paths (bfloat16 weights, float32 activations) never hit it.
        for (int i = 0; i < source.Length; i++)
        {
            destination[i] = (float)source[i];
        }
    }

    /// <summary>
    /// Reads <paramref name="source"/> as a packed array of <paramref name="dtype"/> elements and
    /// writes them as float32 into <paramref name="destination"/>.
    /// </summary>
    public static void ToFloat(ReadOnlySpan<byte> source, DType dtype, Span<float> destination)
    {
        switch (dtype)
        {
            case DType.Float32:
                MemoryMarshal.Cast<byte, float>(source).CopyTo(destination);
                break;
            case DType.BFloat16:
                BF16ToFloat(MemoryMarshal.Cast<byte, ushort>(source), destination);
                break;
            case DType.Float16:
                FP16ToFloat(MemoryMarshal.Cast<byte, Half>(source), destination);
                break;
            case DType.Int64:
            {
                ReadOnlySpan<long> values = MemoryMarshal.Cast<byte, long>(source);
                for (int i = 0; i < values.Length; i++)
                {
                    destination[i] = values[i];
                }

                break;
            }
            case DType.Int32:
            {
                ReadOnlySpan<int> values = MemoryMarshal.Cast<byte, int>(source);
                for (int i = 0; i < values.Length; i++)
                {
                    destination[i] = values[i];
                }

                break;
            }
            case DType.UInt8:
            {
                for (int i = 0; i < source.Length; i++)
                {
                    destination[i] = source[i];
                }

                break;
            }
            case DType.Int8:
            {
                ReadOnlySpan<sbyte> values = MemoryMarshal.Cast<byte, sbyte>(source);
                for (int i = 0; i < values.Length; i++)
                {
                    destination[i] = values[i];
                }

                break;
            }
            case DType.Bool:
            {
                for (int i = 0; i < source.Length; i++)
                {
                    destination[i] = source[i] != 0 ? 1f : 0f;
                }

                break;
            }
            default:
                throw new NotSupportedException($"Cannot convert {dtype} to float32.");
        }
    }
}
