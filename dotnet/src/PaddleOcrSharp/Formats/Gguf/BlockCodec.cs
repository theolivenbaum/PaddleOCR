namespace PaddleOcrSharp.Formats.Gguf;

/// <summary>
/// The one place a <see cref="GgmlType"/> is turned into the codec that reads and writes it.
/// </summary>
/// <remarks>
/// Every caller that packs or unpacks a quantized tensor goes through here, so adding a band is a
/// case in two switches rather than a change to the converter, the weight store, the kernels and
/// the validator — which is what it was before the integer bands arrived.
/// </remarks>
public static class BlockCodec
{
    /// <summary>Encodes one row into <paramref name="type"/>'s block layout.</summary>
    /// <param name="type">Block layout; must be a quantized type.</param>
    /// <param name="source">Weights, one row.</param>
    /// <param name="destination">Receives the packed blocks.</param>
    public static void Encode(GgmlType type, ReadOnlySpan<float> source, Span<byte> destination)
    {
        switch (type)
        {
            case GgmlType.PQ2_0:
                TernaryBlocks.EncodePq20(source, destination);
                break;
            case GgmlType.PTQ1_0:
                TernaryBlocks.EncodePtq10(source, destination);
                break;
            case GgmlType.Q4_0:
                IntegerBlocks.EncodeQ40(source, destination);
                break;
            case GgmlType.Q4_1:
                IntegerBlocks.EncodeQ41(source, destination);
                break;
            case GgmlType.Q5_1:
                IntegerBlocks.EncodeQ51(source, destination);
                break;
            case GgmlType.Q8_0:
                IntegerBlocks.EncodeQ80(source, destination);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(type), type, "Not a quantized block type.");
        }
    }

    /// <summary>Decodes packed blocks into <paramref name="destination"/>.</summary>
    /// <param name="type">Block layout; must be a quantized type.</param>
    /// <param name="source">Packed blocks.</param>
    /// <param name="destination">Receives one float per weight.</param>
    public static void Decode(GgmlType type, ReadOnlySpan<byte> source, Span<float> destination)
    {
        switch (type)
        {
            case GgmlType.PQ2_0:
                TernaryBlocks.DecodePq20(source, destination);
                break;
            case GgmlType.PTQ1_0:
                TernaryBlocks.DecodePtq10(source, destination);
                break;
            case GgmlType.Q4_0:
                IntegerBlocks.DecodeQ40(source, destination);
                break;
            case GgmlType.Q4_1:
                IntegerBlocks.DecodeQ41(source, destination);
                break;
            case GgmlType.Q5_1:
                IntegerBlocks.DecodeQ51(source, destination);
                break;
            case GgmlType.Q8_0:
                IntegerBlocks.DecodeQ80(source, destination);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(type), type, "Not a quantized block type.");
        }
    }
}
