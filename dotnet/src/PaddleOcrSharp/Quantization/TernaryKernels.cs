using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using PaddleOcrSharp.Formats.Gguf;

namespace PaddleOcrSharp.Quantization;

/// <summary>
/// Vectorised decoders for the ternary block layouts — the hot-path counterpart to the scalar
/// reference in <see cref="TernaryBlocks"/>.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="TernaryBlocks"/> is the specification: a transcription of the fork's C, checked
/// against it byte for byte. This is the same function written for the machine, and
/// <c>TernaryCodecTests</c> asserts the two agree exactly on every block. Nothing here may be
/// "improved" without that test going along with it.
/// </para>
/// <para>
/// Both layouts turn out to decode into <i>contiguous</i> runs, which is the property that makes
/// vectorising them worth doing. For <c>PTQ1_0</c> that is not obvious from the encoder, which
/// scatters five weights across positions <c>m, m+c, … m+4c</c> of a stage: read the other way
/// round, a fixed digit index over <c>c</c> consecutive bytes produces exactly the <c>c</c>
/// consecutive weights at <c>n·c</c>. For <c>PQ2_0</c> a byte's four codes are four consecutive
/// weights already, and a shuffle that replicates each byte four times lines them up with a
/// per-lane multiplier.
/// </para>
/// <para>
/// A decoded block is 512 bytes and is consumed immediately, so the store and reload through the
/// caller's buffer stays in L1. Fusing the decode into the dot product would save that; it would
/// also mean two more kernels to keep in step with the reference, and the traffic that actually
/// bounds a decode step is the packed bytes, which are the same either way.
/// </para>
/// </remarks>
internal static class TernaryKernels
{
    private const int Group = TernaryBlockGeometry.GroupSize;

    private static ReadOnlySpan<ushort> Pow3 => [1, 3, 9, 27, 81];

    /// <summary>
    /// Byte indices that replicate each of four consecutive bytes over four lanes.
    /// </summary>
    private static ReadOnlySpan<byte> ReplicateFour =>
    [
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
    ];

    /// <summary>
    /// Per-lane multipliers that move code <c>k</c> of a byte into the top of a six-bit field.
    /// </summary>
    private static ReadOnlySpan<ushort> CodeShifts => [64, 16, 4, 1, 64, 16, 4, 1];

    /// <summary>Decodes one packed row into <paramref name="destination"/>.</summary>
    /// <param name="type">Block layout.</param>
    /// <param name="packed">The row's packed blocks.</param>
    /// <param name="destination">Receives one float per weight; length must be a multiple of 128.</param>
    public static void DecodeRow(GgmlType type, ReadOnlySpan<byte> packed, Span<float> destination)
    {
        int blocks = destination.Length / Group;
        int blockBytes = type.TypeSize();

        switch (type)
        {
            case GgmlType.PQ2_0:
                for (int b = 0; b < blocks; b++)
                {
                    DecodeBlockPq20(packed.Slice(b * blockBytes, blockBytes), destination.Slice(b * Group, Group));
                }

                break;

            case GgmlType.PTQ1_0:
                for (int b = 0; b < blocks; b++)
                {
                    DecodeBlockPtq10(packed.Slice(b * blockBytes, blockBytes), destination.Slice(b * Group, Group));
                }

                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(type), type, "Not a ternary block type.");
        }
    }

    /// <summary>Decodes one 34-byte <c>PQ2_0</c> block.</summary>
    public static void DecodeBlockPq20(ReadOnlySpan<byte> block, Span<float> destination)
    {
        float scale = (float)BitConverter.UInt16BitsToHalf(MemoryMarshal.Read<ushort>(block));
        ReadOnlySpan<byte> codes = block[2..];

        if (!Vector128.IsHardwareAccelerated)
        {
            TernaryBlocks.DecodePq20(block, destination);
            return;
        }

        Vector128<byte> replicate = Vector128.Create(ReplicateFour);
        Vector128<ushort> shifts = Vector128.Create(CodeShifts);
        Vector128<ushort> three = Vector128.Create((ushort)3);
        Vector128<float> one = Vector128.Create(1f);
        Vector128<float> factor = Vector128.Create(scale);

        // Each pass turns four packed bytes into sixteen consecutive weights.
        for (int group = 0; group < Group / 16; group++)
        {
            Vector128<byte> raw = Vector128.Create(
                MemoryMarshal.Read<uint>(codes[(group * 4)..])).AsByte();
            Vector128<byte> spread = Vector128.Shuffle(raw, replicate);

            (Vector128<ushort> low, Vector128<ushort> high) = Vector128.Widen(spread);
            EmitPq20(low, shifts, three, one, factor, destination[(group * 16)..]);
            EmitPq20(high, shifts, three, one, factor, destination[((group * 16) + 8)..]);
        }
    }

    /// <summary>Decodes one 28-byte <c>PTQ1_0</c> block.</summary>
    public static void DecodeBlockPtq10(ReadOnlySpan<byte> block, Span<float> destination)
    {
        const int low = TernaryBlockGeometry.Ptq10LowBytes;
        const int high = TernaryBlockGeometry.Ptq10HighBytes;

        if (!Vector128.IsHardwareAccelerated)
        {
            TernaryBlocks.DecodePtq10(block, destination);
            return;
        }

        float scale = (float)BitConverter.UInt16BitsToHalf(MemoryMarshal.Read<ushort>(block[(low + high)..]));
        Vector128<float> factor = Vector128.Create(scale);

        int write = 0;
        int j = 0;
        foreach (int stage in TernaryBlockGeometry.Ptq10Stages)
        {
            while (j + stage <= low)
            {
                for (int digit = 0; digit < 5; digit++)
                {
                    EmitPtq10(block.Slice(j, stage), Pow3[digit], factor, destination.Slice(write, stage));
                    write += stage;
                }

                j += stage;
            }
        }

        // Eight weights in two bytes: too short to pay for a vector, and the tail of every block.
        for (int digit = 0; digit < 4; digit++)
        {
            for (int h = 0; h < high; h++)
            {
                byte shifted = (byte)(block[low + h] * Pow3[digit]);
                destination[write++] = (((shifted * 3) >> 8) - 1) * scale;
            }
        }
    }

    /// <summary>Turns eight replicated-byte lanes into eight dequantized weights.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void EmitPq20(
        Vector128<ushort> spread,
        Vector128<ushort> shifts,
        Vector128<ushort> three,
        Vector128<float> one,
        Vector128<float> factor,
        Span<float> destination)
    {
        // (byte * 2^(6 - 2k)) >> 6, masked to two bits, is code k of that byte.
        Vector128<ushort> codes = ((spread * shifts) >>> 6) & three;
        (Vector128<uint> first, Vector128<uint> second) = Vector128.Widen(codes);

        Vector128<float> a = (Vector128.ConvertToSingle(first.AsInt32()) - one) * factor;
        Vector128<float> b = (Vector128.ConvertToSingle(second.AsInt32()) - one) * factor;
        a.CopyTo(destination);
        b.CopyTo(destination[4..]);
    }

    /// <summary>
    /// Writes the base-3 digit <c>log₃(power)</c> of each of <paramref name="bytes"/> as a
    /// dequantized weight.
    /// </summary>
    private static void EmitPtq10(
        ReadOnlySpan<byte> bytes,
        ushort power,
        Vector128<float> factor,
        Span<float> destination)
    {
        Vector128<ushort> multiplier = Vector128.Create(power);
        Vector128<ushort> mask = Vector128.Create((ushort)0xFF);
        Vector128<ushort> three = Vector128.Create((ushort)3);
        Vector128<float> one = Vector128.Create(1f);

        int i = 0;
        for (; i <= bytes.Length - 8; i += 8)
        {
            Vector128<ushort> raw = Vector128.WidenLower(
                Vector128.Create(MemoryMarshal.Read<ulong>(bytes[i..]), 0UL).AsByte());

            // The C multiplies in eight bits and lets it wrap, which is the mask below; the
            // product then loses the digits above this one, and the shift takes the one below.
            Vector128<ushort> codes = (((raw * multiplier) & mask) * three) >>> 8;
            (Vector128<uint> first, Vector128<uint> second) = Vector128.Widen(codes);

            ((Vector128.ConvertToSingle(first.AsInt32()) - one) * factor).CopyTo(destination[i..]);
            ((Vector128.ConvertToSingle(second.AsInt32()) - one) * factor).CopyTo(destination[(i + 4)..]);
        }

        for (; i < bytes.Length; i++)
        {
            byte shifted = (byte)(bytes[i] * power);
            destination[i] = (((shifted * 3) >> 8) - 1) * factor[0];
        }
    }
}
