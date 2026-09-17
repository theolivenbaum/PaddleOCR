using PaddleOcrSharp.Formats.Gguf;

namespace PaddleOcrSharp.Quantization;

/// <summary>
/// The quantization header of a converted checkpoint: how it was produced, and which tensors are
/// stored in a rotated basis.
/// </summary>
/// <remarks>
/// <para>
/// The rotation keys are the <c>PrismML-Eng/llama.cpp</c> fork's, verbatim
/// (<c>src/llama-model.cpp</c>), so a file this port writes describes itself the way a Bonsai file
/// does. The rest sit under <c>paddleocr.quantization.*</c> because they have no counterpart
/// there.
/// </para>
/// <para>
/// The <c>weight_names</c> list is the load-bearing one. A rotation applied to some of a layer's
/// matrices and not the others is not a worse model, it is a different function, and the failure
/// is silent — the output is plausible text that is simply wrong. The fork refuses to load such a
/// file rather than run it, and so does <see cref="Read"/>: a rotated tensor whose name is absent
/// from the list, or a width with no sign vector, throws.
/// </para>
/// </remarks>
public sealed class QuantizationMetadata
{
    /// <summary>Metadata key prefix for the rotation, shared with the Prism fork.</summary>
    public const string RotationPrefix = "prism.hadamard.";

    /// <summary>Metadata key prefix for this port's own quantization keys.</summary>
    public const string QuantizationPrefix = "paddleocr.quantization.";

    /// <summary>The transform name the rotation keys must carry.</summary>
    public const string TransformName = "normalized-sylvester-walsh-hadamard";

    /// <summary>The axis name the rotation keys must carry.</summary>
    public const string AxisName = "input-last-dimension";

    /// <summary>The rotation, or <see langword="null"/> when nothing is rotated.</summary>
    public HadamardRotation? Rotation { get; init; }

    /// <summary>Tensors stored in the rotated basis.</summary>
    public IReadOnlySet<string> RotatedWeights { get; init; } = new HashSet<string>(StringComparer.Ordinal);

    /// <summary>How the trits were chosen, e.g. <c>gptq+hadamard</c>.</summary>
    public string Method { get; init; } = "rtn";

    /// <summary>What the calibration Hessians were collected over, or an empty string.</summary>
    public string Calibration { get; init; } = string.Empty;

    /// <summary>The checkpoint this file was converted from.</summary>
    public string Source { get; init; } = string.Empty;

    /// <summary>Weights per shared scale.</summary>
    public int GroupSize { get; init; } = TernaryBlockGeometry.GroupSize;

    /// <summary>Reads the quantization header of <paramref name="file"/>.</summary>
    /// <exception cref="InvalidDataException">The header is inconsistent or uses an unknown mode.</exception>
    public static QuantizationMetadata Read(GgufFile file)
    {
        string method = file.Meta(QuantizationPrefix + "method")?.AsString() ?? "rtn";
        string calibration = file.Meta(QuantizationPrefix + "calibration")?.AsString() ?? string.Empty;
        string source = file.Meta(QuantizationPrefix + "source")?.AsString() ?? string.Empty;
        int groupSize = file.Meta(QuantizationPrefix + "group_size") is { } group
            ? checked((int)group.AsUInt32())
            : TernaryBlockGeometry.GroupSize;

        if (file.Meta(RotationPrefix + "version") is not { } version)
        {
            return new QuantizationMetadata
            {
                Method = method,
                Calibration = calibration,
                Source = source,
                GroupSize = groupSize,
            };
        }

        if (version.AsUInt32() != 1)
        {
            throw new InvalidDataException($"Unsupported {RotationPrefix}version: {version.AsUInt32()}.");
        }

        int blockSize = checked((int)Require(file, "block_size").AsUInt32());
        string transform = Require(file, "transform").AsString();
        string axis = Require(file, "axis").AsString();
        string signMode = Require(file, "sign_mode").AsString();
        string[] names = Require(file, "weight_names").AsStringArray();

        if (blockSize <= 0 || (blockSize & (blockSize - 1)) != 0)
        {
            throw new InvalidDataException($"Invalid {RotationPrefix}block_size: {blockSize}.");
        }

        if (transform != TransformName)
        {
            throw new InvalidDataException($"Unsupported {RotationPrefix}transform: {transform}.");
        }

        if (axis != AxisName)
        {
            throw new InvalidDataException($"Unsupported {RotationPrefix}axis: {axis}.");
        }

        if (names.Length == 0)
        {
            throw new InvalidDataException($"{RotationPrefix}weight_names is empty.");
        }

        Dictionary<int, float[]>? signs = null;
        switch (signMode)
        {
            case "identity":
                break;

            case "explicit":
            {
                int[] widths = Require(file, "sign_widths").AsInt32Array();
                int[] values = Require(file, "sign_values").AsInt32Array();
                if (widths.Length == 0)
                {
                    throw new InvalidDataException(
                        $"{RotationPrefix}sign_mode is explicit but sign_widths is empty; an empty table would read "
                        + "as identity and silently change the model function.");
                }

                signs = [];
                int offset = 0;
                foreach (int width in widths)
                {
                    if (width <= 0 || width % blockSize != 0 || offset + width > values.Length)
                    {
                        throw new InvalidDataException($"Invalid {RotationPrefix} sign width: {width}.");
                    }

                    float[] vector = new float[width];
                    for (int i = 0; i < width; i++)
                    {
                        int value = values[offset + i];
                        if (value is not (1 or -1))
                        {
                            throw new InvalidDataException($"{RotationPrefix}sign_values must be ±1.");
                        }

                        vector[i] = value;
                    }

                    signs[width] = vector;
                    offset += width;
                }

                if (offset != values.Length)
                {
                    throw new InvalidDataException($"{RotationPrefix}sign_values length mismatch.");
                }

                break;
            }

            default:
                throw new InvalidDataException($"Unsupported {RotationPrefix}sign_mode: {signMode}.");
        }

        return new QuantizationMetadata
        {
            Rotation = new HadamardRotation(blockSize, signs),
            RotatedWeights = new HashSet<string>(names, StringComparer.Ordinal),
            Method = method,
            Calibration = calibration,
            Source = source,
            GroupSize = groupSize,
        };
    }

    /// <summary>Writes this header into <paramref name="writer"/>.</summary>
    /// <param name="writer">The file being built; must still be accepting metadata.</param>
    public void Write(GgufWriter writer)
    {
        writer.Add(QuantizationPrefix + "version", 1u);
        writer.Add(QuantizationPrefix + "method", Method);
        writer.Add(QuantizationPrefix + "group_size", (uint)GroupSize);
        writer.Add(QuantizationPrefix + "calibration", Calibration);
        writer.Add(QuantizationPrefix + "source", Source);

        if (Rotation is null)
        {
            return;
        }

        if (RotatedWeights.Count == 0)
        {
            throw new InvalidOperationException("A rotation with no named weights would never be applied.");
        }

        writer.Add(RotationPrefix + "version", 1u);
        writer.Add(RotationPrefix + "block_size", (uint)Rotation.BlockSize);
        writer.Add(RotationPrefix + "transform", TransformName);
        writer.Add(RotationPrefix + "axis", AxisName);
        writer.Add(RotationPrefix + "sign_mode", Rotation.HasSigns ? "explicit" : "identity");
        writer.Add(RotationPrefix + "weight_names", GgufValue.From([.. RotatedWeights.Order(StringComparer.Ordinal)]));

        if (!Rotation.HasSigns)
        {
            return;
        }

        int[] widths = [.. Rotation.SignWidths.Order()];
        var values = new List<int>();
        foreach (int width in widths)
        {
            foreach (float sign in Rotation.Signs(width))
            {
                values.Add(sign < 0f ? -1 : 1);
            }
        }

        writer.Add(RotationPrefix + "sign_widths", GgufValue.From(widths));
        writer.Add(RotationPrefix + "sign_values", GgufValue.From(values.ToArray()));
    }

    /// <summary>The rotation that applies to <paramref name="name"/>, if any.</summary>
    public HadamardRotation? For(string name) =>
        Rotation is not null && RotatedWeights.Contains(name) ? Rotation : null;

    private static GgufValue Require(GgufFile file, string key) =>
        file.Meta(RotationPrefix + key)
        ?? throw new InvalidDataException($"The file declares a rotation but has no {RotationPrefix}{key}.");
}
