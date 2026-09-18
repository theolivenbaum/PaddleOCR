using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace PaddleOcrSharp.Formats.Gguf;

/// <summary>Block geometry of the integer quantization types, all of which group 32 weights.</summary>
/// <remarks>
/// From <c>ggml/src/ggml-common.h</c>. Field order is the byte order, so it is part of the format:
/// <code>
/// typedef struct { ggml_half d;               uint8_t qs[16]; } block_q4_0;  // 18 B
/// typedef struct { ggml_half d, m;            uint8_t qs[16]; } block_q4_1;  // 20 B
/// typedef struct { ggml_half d, m; uint8_t qh[4]; uint8_t qs[16]; } block_q5_1;  // 24 B
/// typedef struct { ggml_half d;               int8_t  qs[32]; } block_q8_0;  // 34 B
/// </code>
/// </remarks>
public static class IntegerBlockGeometry
{
    /// <summary>Weights covered by one block of any of these types.</summary>
    public const int GroupSize = 32;

    /// <summary>Bytes in one <c>Q4_0</c> block — 4.5 bits a weight.</summary>
    public const int Q40BlockBytes = 2 + (GroupSize / 2);

    /// <summary>Bytes in one <c>Q4_1</c> block — 5.0 bits a weight.</summary>
    public const int Q41BlockBytes = 4 + (GroupSize / 2);

    /// <summary>Bytes in one <c>Q5_1</c> block — 6.0 bits a weight.</summary>
    public const int Q51BlockBytes = 4 + 4 + (GroupSize / 2);

    /// <summary>Bytes in one <c>Q8_0</c> block — 8.5 bits a weight.</summary>
    public const int Q80BlockBytes = 2 + GroupSize;
}

/// <summary>
/// Encoders and decoders for the integer block layouts, byte-identical to
/// <c>quantize_row_*_ref</c> and <c>dequantize_row_*</c> in <c>ggml/src/ggml-quants.c</c>.
/// </summary>
/// <remarks>
/// <para>
/// These are the bands that make a converted checkpoint usable. Ternary is a representation
/// problem — <c>log₂3</c> bits cannot carry a dense weight matrix without training that put it
/// there — where four to eight bits is an ordinary rounding problem, and the reference encoders
/// solve it well enough that the first thing to do is measure them rather than improve them.
/// </para>
/// <para>
/// Two layout details are easy to get wrong and silent when wrong. The nibble types pack weight
/// <c>j</c> and weight <c>j + 16</c> into byte <c>j</c> — low nibble and high nibble — rather than
/// adjacent weights into one byte. And <c>Q4_1</c>, <c>Q5_1</c> are <b>asymmetric</b>: they store a
/// minimum alongside the scale and quantize <c>x − m</c>, which is why they beat their symmetric
/// counterparts on weights whose distribution is not centred.
/// </para>
/// </remarks>
public static class IntegerBlocks
{
    private const int Group = IntegerBlockGeometry.GroupSize;
    private const int Half = Group / 2;

    /// <summary>Encodes into <c>Q8_0</c> blocks.</summary>
    /// <param name="source">Weights; length must be a multiple of 32.</param>
    /// <param name="destination">Receives <c>34 · length / 32</c> bytes.</param>
    public static void EncodeQ80(ReadOnlySpan<float> source, Span<byte> destination)
    {
        int blocks = Blocks(source.Length);
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<float> group = source.Slice(b * Group, Group);
            Span<byte> block = destination.Slice(b * IntegerBlockGeometry.Q80BlockBytes, IntegerBlockGeometry.Q80BlockBytes);

            float amax = 0f;
            for (int j = 0; j < Group; j++)
            {
                amax = MathF.Max(amax, MathF.Abs(group[j]));
            }

            float scale = amax / 127f;
            float inverse = scale != 0f ? 1f / scale : 0f;
            WriteHalf(block, (Half)scale);

            Span<byte> codes = block[2..];
            for (int j = 0; j < Group; j++)
            {
                codes[j] = unchecked((byte)(sbyte)RoundAwayFromZero(group[j] * inverse));
            }
        }
    }

    /// <summary>Decodes <c>Q8_0</c> blocks.</summary>
    /// <param name="source">Packed blocks.</param>
    /// <param name="destination">Receives 32 floats per block.</param>
    public static void DecodeQ80(ReadOnlySpan<byte> source, Span<float> destination)
    {
        int blocks = Blocks(destination.Length);
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<byte> block = source.Slice(b * IntegerBlockGeometry.Q80BlockBytes, IntegerBlockGeometry.Q80BlockBytes);
            float scale = (float)ReadHalf(block);
            ReadOnlySpan<sbyte> codes = MemoryMarshal.Cast<byte, sbyte>(block[2..]);
            Span<float> group = destination.Slice(b * Group, Group);

            for (int j = 0; j < Group; j++)
            {
                group[j] = codes[j] * scale;
            }
        }
    }

    /// <summary>Encodes into <c>Q4_0</c> blocks.</summary>
    /// <param name="source">Weights; length must be a multiple of 32.</param>
    /// <param name="destination">Receives <c>18 · length / 32</c> bytes.</param>
    public static void EncodeQ40(ReadOnlySpan<float> source, Span<byte> destination)
    {
        int blocks = Blocks(source.Length);
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<float> group = source.Slice(b * Group, Group);
            Span<byte> block = destination.Slice(b * IntegerBlockGeometry.Q40BlockBytes, IntegerBlockGeometry.Q40BlockBytes);

            // The scale is built from the signed extreme, not the magnitude, so that the extreme
            // lands on code 0 and the grid is {-8 … 7} · d.
            float amax = 0f;
            float extreme = 0f;
            for (int j = 0; j < Group; j++)
            {
                float value = group[j];
                if (amax < MathF.Abs(value))
                {
                    amax = MathF.Abs(value);
                    extreme = value;
                }
            }

            float scale = extreme / -8f;
            float inverse = scale != 0f ? 1f / scale : 0f;
            WriteHalf(block, (Half)scale);

            Span<byte> codes = block[2..];
            for (int j = 0; j < Half; j++)
            {
                int low = Nibble((group[j] * inverse) + 8.5f);
                int high = Nibble((group[j + Half] * inverse) + 8.5f);
                codes[j] = (byte)(low | (high << 4));
            }
        }
    }

    /// <summary>Decodes <c>Q4_0</c> blocks.</summary>
    /// <param name="source">Packed blocks.</param>
    /// <param name="destination">Receives 32 floats per block.</param>
    public static void DecodeQ40(ReadOnlySpan<byte> source, Span<float> destination)
    {
        int blocks = Blocks(destination.Length);
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<byte> block = source.Slice(b * IntegerBlockGeometry.Q40BlockBytes, IntegerBlockGeometry.Q40BlockBytes);
            float scale = (float)ReadHalf(block);
            ReadOnlySpan<byte> codes = block[2..];
            Span<float> group = destination.Slice(b * Group, Group);

            for (int j = 0; j < Half; j++)
            {
                group[j] = ((codes[j] & 0x0F) - 8) * scale;
                group[j + Half] = ((codes[j] >> 4) - 8) * scale;
            }
        }
    }

    /// <summary>Encodes into <c>Q4_1</c> blocks.</summary>
    /// <param name="source">Weights; length must be a multiple of 32.</param>
    /// <param name="destination">Receives <c>20 · length / 32</c> bytes.</param>
    public static void EncodeQ41(ReadOnlySpan<float> source, Span<byte> destination)
    {
        int blocks = Blocks(source.Length);
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<float> group = source.Slice(b * Group, Group);
            Span<byte> block = destination.Slice(b * IntegerBlockGeometry.Q41BlockBytes, IntegerBlockGeometry.Q41BlockBytes);

            (float minimum, float maximum) = Extremes(group);
            float scale = (maximum - minimum) / 15f;
            float inverse = scale != 0f ? 1f / scale : 0f;

            WriteHalf(block, (Half)scale);
            WriteHalf(block[2..], (Half)minimum);

            Span<byte> codes = block[4..];
            for (int j = 0; j < Half; j++)
            {
                int low = Nibble(((group[j] - minimum) * inverse) + 0.5f);
                int high = Nibble(((group[j + Half] - minimum) * inverse) + 0.5f);
                codes[j] = (byte)(low | (high << 4));
            }
        }
    }

    /// <summary>Decodes <c>Q4_1</c> blocks.</summary>
    /// <param name="source">Packed blocks.</param>
    /// <param name="destination">Receives 32 floats per block.</param>
    public static void DecodeQ41(ReadOnlySpan<byte> source, Span<float> destination)
    {
        int blocks = Blocks(destination.Length);
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<byte> block = source.Slice(b * IntegerBlockGeometry.Q41BlockBytes, IntegerBlockGeometry.Q41BlockBytes);
            float scale = (float)ReadHalf(block);
            float minimum = (float)ReadHalf(block[2..]);
            ReadOnlySpan<byte> codes = block[4..];
            Span<float> group = destination.Slice(b * Group, Group);

            for (int j = 0; j < Half; j++)
            {
                group[j] = ((codes[j] & 0x0F) * scale) + minimum;
                group[j + Half] = ((codes[j] >> 4) * scale) + minimum;
            }
        }
    }

    /// <summary>Encodes into <c>Q5_1</c> blocks.</summary>
    /// <param name="source">Weights; length must be a multiple of 32.</param>
    /// <param name="destination">Receives <c>24 · length / 32</c> bytes.</param>
    public static void EncodeQ51(ReadOnlySpan<float> source, Span<byte> destination)
    {
        int blocks = Blocks(source.Length);
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<float> group = source.Slice(b * Group, Group);
            Span<byte> block = destination.Slice(b * IntegerBlockGeometry.Q51BlockBytes, IntegerBlockGeometry.Q51BlockBytes);

            (float minimum, float maximum) = Extremes(group);
            float scale = (maximum - minimum) / 31f;
            float inverse = scale != 0f ? 1f / scale : 0f;

            WriteHalf(block, (Half)scale);
            WriteHalf(block[2..], (Half)minimum);

            uint high = 0;
            Span<byte> codes = block[8..];
            for (int j = 0; j < Half; j++)
            {
                int low0 = (int)(((group[j] - minimum) * inverse) + 0.5f);
                int low1 = (int)(((group[j + Half] - minimum) * inverse) + 0.5f);

                codes[j] = (byte)((low0 & 0x0F) | ((low1 & 0x0F) << 4));

                // The fifth bit of each code lives in its own plane, one bit per weight.
                high |= (uint)((low0 & 0x10) >> 4) << j;
                high |= (uint)((low1 & 0x10) >> 4) << (j + Half);
            }

            BitConverter.TryWriteBytes(block[4..8], high);
        }
    }

    /// <summary>Decodes <c>Q5_1</c> blocks.</summary>
    /// <param name="source">Packed blocks.</param>
    /// <param name="destination">Receives 32 floats per block.</param>
    public static void DecodeQ51(ReadOnlySpan<byte> source, Span<float> destination)
    {
        int blocks = Blocks(destination.Length);
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<byte> block = source.Slice(b * IntegerBlockGeometry.Q51BlockBytes, IntegerBlockGeometry.Q51BlockBytes);
            float scale = (float)ReadHalf(block);
            float minimum = (float)ReadHalf(block[2..]);
            uint high = BitConverter.ToUInt32(block[4..8]);
            ReadOnlySpan<byte> codes = block[8..];
            Span<float> group = destination.Slice(b * Group, Group);

            for (int j = 0; j < Half; j++)
            {
                int low = (codes[j] & 0x0F) | (int)((high >> j << 4) & 0x10);
                int top = (codes[j] >> 4) | (int)((high >> (j + 12)) & 0x10);
                group[j] = (low * scale) + minimum;
                group[j + Half] = (top * scale) + minimum;
            }
        }
    }

    private static (float Minimum, float Maximum) Extremes(ReadOnlySpan<float> group)
    {
        float minimum = float.MaxValue;
        float maximum = float.MinValue;
        for (int j = 0; j < group.Length; j++)
        {
            float value = group[j];
            if (value < minimum)
            {
                minimum = value;
            }

            if (value > maximum)
            {
                maximum = value;
            }
        }

        return (minimum, maximum);
    }

    /// <summary>C's <c>MIN(15, (int8_t)(v))</c>: truncate toward zero, then cap at fifteen.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Nibble(float value) => Math.Min(15, (int)value);

    /// <summary>C's <c>roundf</c>: halves go away from zero, not to even.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int RoundAwayFromZero(float value) =>
        (int)MathF.Round(value, MidpointRounding.AwayFromZero);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Half ReadHalf(ReadOnlySpan<byte> source) =>
        BitConverter.UInt16BitsToHalf(MemoryMarshal.Read<ushort>(source));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void WriteHalf(Span<byte> destination, Half value) =>
        MemoryMarshal.Write(destination, BitConverter.HalfToUInt16Bits(value));

    private static int Blocks(int elements)
    {
        if (elements % Group != 0)
        {
            throw new ArgumentException(
                $"An integer-quantized row must be a multiple of {Group} weights but has {elements}.",
                nameof(elements));
        }

        return elements / Group;
    }
}
