using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace PaddleOcrSharp.Formats.Gguf;

/// <summary>Block geometry of the two Prism ternary packings.</summary>
/// <remarks>
/// From <c>ggml/src/ggml-common.h</c> in the <c>PrismML-Eng/llama.cpp</c> fork:
/// <code>
/// #define QK_PQ2_0 128
/// typedef struct { ggml_half d; uint8_t qs[QK_PQ2_0/4]; } block_pq2_0;          // 34 B
///
/// #define QK_PTQ1_0 128
/// typedef struct {
///     uint8_t qs[(QK_PTQ1_0 - 4*QK_PTQ1_0/64)/5]; // 24 B, 5 trits per byte
///     uint8_t qh[QK_PTQ1_0/64];                   //  2 B, 4 trits per byte
///     ggml_half d;
/// } block_ptq1_0;                                                               // 28 B
/// </code>
/// </remarks>
public static class TernaryBlockGeometry
{
    /// <summary>Weights covered by one block of either packing.</summary>
    public const int GroupSize = 128;

    /// <summary>Bytes in one <c>PQ2_0</c> block: an fp16 scale then 2 bits per weight.</summary>
    public const int Pq20BlockBytes = 2 + (GroupSize / 4);

    /// <summary>Bytes of base-3 digits in one <c>PTQ1_0</c> block, five weights to a byte.</summary>
    public const int Ptq10LowBytes = (GroupSize - (4 * GroupSize / 64)) / 5;

    /// <summary>Bytes of the <c>PTQ1_0</c> tail, four weights to a byte.</summary>
    public const int Ptq10HighBytes = GroupSize / 64;

    /// <summary>Bytes in one <c>PTQ1_0</c> block: the digits, the tail, then an fp16 scale.</summary>
    public const int Ptq10BlockBytes = Ptq10LowBytes + Ptq10HighBytes + 2;

    /// <summary>
    /// Byte counts the <c>PTQ1_0</c> digit area is filled in, largest first.
    /// </summary>
    /// <remarks>
    /// Upstream <c>TQ1_0</c> stages a 48-byte area as 32 then 16. The fork generalises that to
    /// 32/16/8 so a 24-byte area is covered as 16 then 8; at 48 bytes the sequence reduces to
    /// upstream's exactly.
    /// </remarks>
    internal static ReadOnlySpan<int> Ptq10Stages => [32, 16, 8];
}

/// <summary>
/// Encoders and decoders for the <c>PQ2_0</c> and <c>PTQ1_0</c> block layouts, byte-identical to
/// <c>quantize_row_*_ref</c> and <c>dequantize_row_*</c> in the PrismML <c>llama.cpp</c> fork
/// (<c>ggml/src/ggml-quants.c</c>).
/// </summary>
/// <remarks>
/// <para>
/// These are container operations, not a quantizer. Both encoders take the group's scale as
/// <c>amax</c> and round, which is exact when the input is already ternary at this group size and
/// is merely round-to-nearest when it is not. Choosing the trits is
/// <see cref="Quantization.TernaryQuantizer"/>'s job; this type only lays them out.
/// </para>
/// <para>
/// Two details decide whether a transcription of the C is right. The C rounds with
/// <c>roundf</c>/<c>lroundf</c>, which is half-away-from-zero, where .NET's
/// <see cref="MathF.Round(float)"/> is half-to-even — a silent one-code difference on exact
/// halves. And the base-3 digits are not stored as the base-3 value but as
/// <c>ceil(v · 256 / 243)</c>, which is what lets the decoder recover digit <c>n</c> with a
/// <c>uint8</c> multiply by <c>3ⁿ</c> and a shift instead of a division.
/// </para>
/// </remarks>
public static class TernaryBlocks
{
    private const int Group = TernaryBlockGeometry.GroupSize;

    /// <summary>Powers of three that fit a byte, indexed by digit position.</summary>
    private static ReadOnlySpan<byte> Pow3 => [1, 3, 9, 27, 81, 243];

    /// <summary>Bytes needed to store <paramref name="elements"/> weights in <c>PQ2_0</c>.</summary>
    public static long Pq20Size(long elements) => Blocks(elements) * TernaryBlockGeometry.Pq20BlockBytes;

    /// <summary>Bytes needed to store <paramref name="elements"/> weights in <c>PTQ1_0</c>.</summary>
    public static long Ptq10Size(long elements) => Blocks(elements) * TernaryBlockGeometry.Ptq10BlockBytes;

    /// <summary>
    /// Encodes <paramref name="source"/> into <c>PQ2_0</c> blocks.
    /// </summary>
    /// <param name="source">Weights; the length must be a multiple of 128.</param>
    /// <param name="destination">Receives <c>34 · length / 128</c> bytes.</param>
    public static void EncodePq20(ReadOnlySpan<float> source, Span<byte> destination)
    {
        int blocks = (int)Blocks(source.Length);
        CheckDestination(destination.Length, blocks * TernaryBlockGeometry.Pq20BlockBytes);

        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<float> group = source.Slice(b * Group, Group);
            Span<byte> block = destination.Slice(b * TernaryBlockGeometry.Pq20BlockBytes, TernaryBlockGeometry.Pq20BlockBytes);

            float scale = AbsoluteMaximum(group);
            float inverse = scale > 0f ? 1f / scale : 0f;
            WriteHalf(block, (Half)scale);

            Span<byte> codes = block[2..];
            codes.Clear();

            for (int j = 0; j < Group; j++)
            {
                int q = RoundAwayFromZero(group[j] * inverse) + 1;
                q = Math.Clamp(q, 0, 3);
                codes[j / 4] |= (byte)(q << ((j % 4) * 2));
            }
        }
    }

    /// <summary>Decodes <c>PQ2_0</c> blocks into <paramref name="destination"/>.</summary>
    /// <param name="source">Packed blocks.</param>
    /// <param name="destination">Receives <c>128</c> floats per block.</param>
    public static void DecodePq20(ReadOnlySpan<byte> source, Span<float> destination)
    {
        int blocks = (int)Blocks(destination.Length);
        CheckSource(source.Length, blocks * TernaryBlockGeometry.Pq20BlockBytes);

        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<byte> block = source.Slice(b * TernaryBlockGeometry.Pq20BlockBytes, TernaryBlockGeometry.Pq20BlockBytes);
            float scale = (float)ReadHalf(block);
            ReadOnlySpan<byte> codes = block[2..];
            Span<float> group = destination.Slice(b * Group, Group);

            for (int j = 0; j < Group; j++)
            {
                int q = (codes[j / 4] >> ((j % 4) * 2)) & 0x03;
                group[j] = (q - 1) * scale;
            }
        }
    }

    /// <summary>
    /// Encodes <paramref name="source"/> into <c>PTQ1_0</c> blocks.
    /// </summary>
    /// <param name="source">Weights; the length must be a multiple of 128.</param>
    /// <param name="destination">Receives <c>28 · length / 128</c> bytes.</param>
    public static void EncodePtq10(ReadOnlySpan<float> source, Span<byte> destination)
    {
        int blocks = (int)Blocks(source.Length);
        CheckDestination(destination.Length, blocks * TernaryBlockGeometry.Ptq10BlockBytes);

        const int low = TernaryBlockGeometry.Ptq10LowBytes;
        const int high = TernaryBlockGeometry.Ptq10HighBytes;

        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<float> group = source.Slice(b * Group, Group);
            Span<byte> block = destination.Slice(b * TernaryBlockGeometry.Ptq10BlockBytes, TernaryBlockGeometry.Ptq10BlockBytes);

            float scale = AbsoluteMaximum(group);
            float inverse = scale > 0f ? 1f / scale : 0f;

            // The C walks a moving pointer through the group; `read` is that pointer's offset.
            int read = 0;
            int j = 0;
            foreach (int stage in TernaryBlockGeometry.Ptq10Stages)
            {
                while (j + stage <= low)
                {
                    for (int m = 0; m < stage; m++)
                    {
                        int value = 0;
                        for (int n = 0; n < 5; n++)
                        {
                            value = (value * 3) + Trit(group[read + m + (n * stage)], inverse);
                        }

                        block[j + m] = ScaleToByte(value);
                    }

                    read += 5 * stage;
                    j += stage;
                }
            }

            for (int h = 0; h < high; h++)
            {
                int value = 0;
                for (int m = 0; m < 4; m++)
                {
                    value = (value * 3) + Trit(group[read + h + (m * high)], inverse);
                }

                // Shift the first digit into the most significant trit so the decoder's power
                // table is the same one the 5-digit area uses.
                block[low + h] = ScaleToByte(value * 3);
            }

            WriteHalf(block[(low + high)..], (Half)scale);
        }
    }

    /// <summary>Decodes <c>PTQ1_0</c> blocks into <paramref name="destination"/>.</summary>
    /// <param name="source">Packed blocks.</param>
    /// <param name="destination">Receives <c>128</c> floats per block.</param>
    public static void DecodePtq10(ReadOnlySpan<byte> source, Span<float> destination)
    {
        int blocks = (int)Blocks(destination.Length);
        CheckSource(source.Length, blocks * TernaryBlockGeometry.Ptq10BlockBytes);

        const int low = TernaryBlockGeometry.Ptq10LowBytes;
        const int high = TernaryBlockGeometry.Ptq10HighBytes;

        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<byte> block = source.Slice(b * TernaryBlockGeometry.Ptq10BlockBytes, TernaryBlockGeometry.Ptq10BlockBytes);
            float scale = (float)ReadHalf(block[(low + high)..]);
            Span<float> group = destination.Slice(b * Group, Group);

            int write = 0;
            int j = 0;
            foreach (int stage in TernaryBlockGeometry.Ptq10Stages)
            {
                while (j + stage <= low)
                {
                    for (int n = 0; n < 5; n++)
                    {
                        for (int m = 0; m < stage; m++)
                        {
                            group[write++] = (Digit(block[j + m], n) - 1) * scale;
                        }
                    }

                    j += stage;
                }
            }

            for (int n = 0; n < 4; n++)
            {
                for (int h = 0; h < high; h++)
                {
                    group[write++] = (Digit(block[low + h], n) - 1) * scale;
                }
            }
        }
    }

    /// <summary>Decodes one row of <paramref name="type"/> into <paramref name="destination"/>.</summary>
    /// <param name="type">Block layout; must be a quantized type.</param>
    /// <param name="source">Packed blocks.</param>
    /// <param name="destination">Receives one float per weight.</param>
    public static void Decode(GgmlType type, ReadOnlySpan<byte> source, Span<float> destination)
    {
        switch (type)
        {
            case GgmlType.PQ2_0:
                DecodePq20(source, destination);
                break;
            case GgmlType.PTQ1_0:
                DecodePtq10(source, destination);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(type), type, "Not a ternary block type.");
        }
    }

    /// <summary>Encodes one row into <paramref name="type"/>'s block layout.</summary>
    /// <param name="type">Block layout; must be a quantized type.</param>
    /// <param name="source">Weights.</param>
    /// <param name="destination">Receives the packed blocks.</param>
    public static void Encode(GgmlType type, ReadOnlySpan<float> source, Span<byte> destination)
    {
        switch (type)
        {
            case GgmlType.PQ2_0:
                EncodePq20(source, destination);
                break;
            case GgmlType.PTQ1_0:
                EncodePtq10(source, destination);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(type), type, "Not a ternary block type.");
        }
    }

    /// <summary>
    /// Recovers base-3 digit <paramref name="digit"/> from a stored byte.
    /// </summary>
    /// <remarks>
    /// The byte holds <c>ceil(v · 256 / 243)</c>; multiplying by <c>3ⁿ</c> in eight bits discards
    /// the digits above <c>n</c>, and <c>(q · 3) &gt;&gt; 8</c> takes the one below them.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Digit(byte stored, int digit)
    {
        byte shifted = (byte)(stored * Pow3[digit]);
        return (shifted * 3) >> 8;
    }

    /// <summary>Maps a base-3 value onto the byte the decoder can take digits out of.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte ScaleToByte(int value) => (byte)(((value * 256) + 242) / 243);

    /// <summary>Rounds one weight to <c>{0, 1, 2}</c> — the trit plus one.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Trit(float value, float inverse) =>
        Math.Clamp(RoundAwayFromZero(value * inverse) + 1, 0, 2);

    /// <summary>C's <c>roundf</c>: halves go away from zero, not to even.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int RoundAwayFromZero(float value) =>
        (int)MathF.Round(value, MidpointRounding.AwayFromZero);

    private static float AbsoluteMaximum(ReadOnlySpan<float> values)
    {
        float maximum = 0f;
        for (int i = 0; i < values.Length; i++)
        {
            float magnitude = MathF.Abs(values[i]);
            if (magnitude > maximum)
            {
                maximum = magnitude;
            }
        }

        return maximum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Half ReadHalf(ReadOnlySpan<byte> source) =>
        BitConverter.UInt16BitsToHalf(MemoryMarshal.Read<ushort>(source));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void WriteHalf(Span<byte> destination, Half value) =>
        MemoryMarshal.Write(destination, BitConverter.HalfToUInt16Bits(value));

    private static long Blocks(long elements)
    {
        if (elements % Group != 0)
        {
            throw new ArgumentException(
                $"A ternary row must be a multiple of {Group} weights but has {elements}.", nameof(elements));
        }

        return elements / Group;
    }

    private static void CheckDestination(int actual, int needed)
    {
        if (actual < needed)
        {
            throw new ArgumentException($"Need {needed} bytes but only {actual} are available.", "destination");
        }
    }

    private static void CheckSource(int actual, int needed)
    {
        if (actual < needed)
        {
            throw new ArgumentException($"Need {needed} bytes but only {actual} are available.", "source");
        }
    }
}
