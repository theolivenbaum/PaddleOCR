using System.Text.Json;
using System.Text.Json.Serialization;
using PaddleOcrSharp.Formats.Gguf;

namespace PaddleOcrSharp.Quantization;

/// <summary>One rule of a <see cref="QuantizationPolicy"/>.</summary>
/// <param name="Match">
/// A glob over the tensor name; <c>*</c> matches any run of characters and <c>?</c> one character.
/// </param>
/// <param name="Scheme">
/// What to store a matching tensor as: <c>ptq1_0</c>, <c>pq2_0</c>, <c>bf16</c> or <c>f32</c>.
/// </param>
/// <param name="Rotate">
/// Whether a matching tensor is folded into the rotated basis; defaults to whether it is quantized.
/// </param>
public sealed record QuantizationRule(string Match, string Scheme, bool? Rotate = null);

/// <summary>The rotation settings a policy asks for.</summary>
/// <param name="BlockSize">Rotation block; <c>0</c> disables the rotation entirely.</param>
/// <param name="SignMode"><c>identity</c> or <c>explicit</c>.</param>
/// <param name="Seed">Seed for the explicit sign vectors.</param>
public sealed record RotationPolicy(int BlockSize = 0, string SignMode = "identity", int Seed = 1);

/// <summary>
/// Which precision each tensor of a checkpoint is converted to.
/// </summary>
/// <remarks>
/// <para>
/// Bonsai 2 keeps a short, explicitly named set of tensors at full precision — the recurrent state
/// path, every normalization, the q/k norms — at a cost of 0.0976% of the parameters and 0.01 bits
/// per weight overall. The lesson is not the list, which is that model's, but the shape of the
/// decision: precision is a property of a tensor, chosen by measurement, and exempting the
/// sensitive few costs almost nothing.
/// </para>
/// <para>
/// So the policy is data rather than code. <see cref="Recommended"/> is a starting point, not an
/// answer; what ships is whatever the validation ladder in <c>docs/ternary.md</c> says survives.
/// </para>
/// </remarks>
public sealed class QuantizationPolicy
{
    /// <summary>The scheme for tensors no rule matches.</summary>
    public string DefaultScheme { get; init; } = "ptq1_0";

    /// <summary>Rules, applied in order; the first match wins.</summary>
    public IReadOnlyList<QuantizationRule> Rules { get; init; } = [];

    /// <summary>Rotation settings.</summary>
    public RotationPolicy Rotation { get; init; } = new();

    /// <summary>
    /// The port's starting policy for PaddleOCR-VL: ternary weights everywhere the model spends
    /// its bandwidth, and float everywhere a matrix product is not what happens.
    /// </summary>
    /// <remarks>
    /// The four exemptions are not guesses about sensitivity, they are places where quantizing
    /// would be the wrong operation. Norms and biases are kilobytes read once per row. The token
    /// embedding is a gather, so a packed row would have to be decoded per token for no bandwidth
    /// saved. The position embedding is bilinearly interpolated at run time, on values whose
    /// differences are the signal. The patch embedding is one product per page. <c>lm_head</c> is
    /// none of those — 106 M parameters re-read on every generated token — so it is quantized, and
    /// it is the first tensor the per-stage validation should look at, because its error lands
    /// directly on an argmax.
    /// </remarks>
    public static QuantizationPolicy Recommended { get; } = new()
    {
        DefaultScheme = "ptq1_0",
        Rules =
        [
            new QuantizationRule("*norm*", "f32"),
            new QuantizationRule("*layer_norm*", "f32"),
            new QuantizationRule("*.bias", "f32"),
            new QuantizationRule("*embed_tokens*", "bf16"),
            new QuantizationRule("*position_embedding*", "f32"),
            new QuantizationRule("*patch_embedding*", "bf16"),
        ],
        Rotation = new RotationPolicy(BlockSize: 0),
    };

    /// <summary>Reads a policy from JSON.</summary>
    /// <param name="path">A JSON file in the shape of this type.</param>
    public static QuantizationPolicy Load(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return JsonSerializer.Deserialize(stream, QuantizationPolicyContext.Default.QuantizationPolicy)
            ?? throw new InvalidDataException($"'{path}' is not a quantization policy.");
    }

    /// <summary>Writes this policy as JSON.</summary>
    /// <param name="path">Destination file.</param>
    public void Save(string path)
    {
        using FileStream stream = File.Create(path);
        JsonSerializer.Serialize(stream, this, QuantizationPolicyContext.Default.QuantizationPolicy);
    }

    /// <summary>The stored type for <paramref name="name"/>.</summary>
    /// <param name="name">Tensor name.</param>
    public GgmlType SchemeFor(string name) => Parse(RuleFor(name)?.Scheme ?? DefaultScheme, name);

    /// <summary>Whether <paramref name="name"/> is folded into the rotated basis.</summary>
    /// <param name="name">Tensor name.</param>
    public bool RotatesFor(string name)
    {
        if (Rotation.BlockSize <= 0)
        {
            return false;
        }

        QuantizationRule? rule = RuleFor(name);
        return rule?.Rotate ?? SchemeFor(name).IsQuantized();
    }

    /// <summary>Builds the rotation this policy describes, or <see langword="null"/>.</summary>
    /// <param name="widths">The input widths that need a sign vector.</param>
    public HadamardRotation? BuildRotation(IEnumerable<int> widths)
    {
        if (Rotation.BlockSize <= 0)
        {
            return null;
        }

        if (!string.Equals(Rotation.SignMode, "explicit", StringComparison.Ordinal))
        {
            return new HadamardRotation(Rotation.BlockSize);
        }

        Dictionary<int, float[]> signs = [];
        foreach (int width in widths.Distinct().Order())
        {
            signs[width] = HadamardRotation.BuildSigns(width, Rotation.Seed + width);
        }

        return new HadamardRotation(Rotation.BlockSize, signs);
    }

    private QuantizationRule? RuleFor(string name)
    {
        foreach (QuantizationRule rule in Rules)
        {
            if (Matches(rule.Match, name))
            {
                return rule;
            }
        }

        return null;
    }

    private static GgmlType Parse(string scheme, string name) => scheme.ToLowerInvariant() switch
    {
        "ptq1_0" or "ptq1" => GgmlType.PTQ1_0,
        "pq2_0" or "pq2" => GgmlType.PQ2_0,
        "bf16" => GgmlType.BF16,
        "f32" or "fp32" => GgmlType.F32,
        _ => throw new InvalidDataException($"Unknown scheme '{scheme}' for '{name}'."),
    };

    /// <summary>Glob match: <c>*</c> for any run, <c>?</c> for one character.</summary>
    /// <param name="pattern">The glob.</param>
    /// <param name="value">The name to test.</param>
    public static bool Matches(string pattern, string value)
    {
        int p = 0;
        int v = 0;
        int star = -1;
        int mark = 0;

        while (v < value.Length)
        {
            if (p < pattern.Length && (pattern[p] == '?' || pattern[p] == value[v]))
            {
                p++;
                v++;
            }
            else if (p < pattern.Length && pattern[p] == '*')
            {
                star = p++;
                mark = v;
            }
            else if (star >= 0)
            {
                p = star + 1;
                v = ++mark;
            }
            else
            {
                return false;
            }
        }

        while (p < pattern.Length && pattern[p] == '*')
        {
            p++;
        }

        return p == pattern.Length;
    }
}

/// <summary>Source-generated JSON context, so the policy reader stays trim- and AOT-safe.</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    WriteIndented = true,
    ReadCommentHandling = JsonCommentHandling.Skip,
    AllowTrailingCommas = true)]
[JsonSerializable(typeof(QuantizationPolicy))]
public partial class QuantizationPolicyContext : JsonSerializerContext;
