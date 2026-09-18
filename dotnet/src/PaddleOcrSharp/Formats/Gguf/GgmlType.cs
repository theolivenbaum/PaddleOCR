namespace PaddleOcrSharp.Formats.Gguf;

/// <summary>
/// The ggml element types this port reads and writes, with the numeric values ggml assigns them.
/// </summary>
/// <remarks>
/// <para>
/// Only the types a PaddleOCR-VL checkpoint can hold are listed. The two Prism-private ternary
/// types come from the <c>PrismML-Eng/llama.cpp</c> fork (<c>ggml/include/ggml.h</c>): they sit at
/// 142 and 143, past upstream's <c>GGML_TYPE_COUNT</c>, which is what makes a file carrying them
/// fail safely on a stock build rather than decode as something else.
/// </para>
/// <para>
/// The numeric values are part of the file format. Do not renumber them.
/// </para>
/// </remarks>
public enum GgmlType
{
    /// <summary>IEEE-754 binary32. <c>GGML_TYPE_F32</c>.</summary>
    F32 = 0,

    /// <summary>IEEE-754 binary16. <c>GGML_TYPE_F16</c>.</summary>
    F16 = 1,

    /// <summary>
    /// Symmetric 4-bit, 32 weights per block, one fp16 scale. 18 bytes, 4.5 bpw.
    /// <c>GGML_TYPE_Q4_0</c>.
    /// </summary>
    Q4_0 = 2,

    /// <summary>
    /// Asymmetric 4-bit, 32 weights per block, an fp16 scale and an fp16 minimum. 20 bytes,
    /// 5.0 bpw. <c>GGML_TYPE_Q4_1</c>.
    /// </summary>
    Q4_1 = 3,

    /// <summary>
    /// Asymmetric 5-bit, 32 weights per block, scale, minimum and a plane of fifth bits.
    /// 24 bytes, 6.0 bpw. <c>GGML_TYPE_Q5_1</c>.
    /// </summary>
    Q5_1 = 7,

    /// <summary>
    /// Symmetric 8-bit, 32 weights per block, one fp16 scale. 34 bytes, 8.5 bpw.
    /// <c>GGML_TYPE_Q8_0</c>.
    /// </summary>
    Q8_0 = 8,

    /// <summary>Signed 32-bit integer. <c>GGML_TYPE_I32</c>.</summary>
    I32 = 26,

    /// <summary>Truncated binary32. <c>GGML_TYPE_BF16</c>.</summary>
    BF16 = 30,

    /// <summary>
    /// Prism group-128 two-bit codec, 34 bytes per 128 weights (2.125 bpw).
    /// <c>GGML_TYPE_PQ2_0</c>.
    /// </summary>
    PQ2_0 = 142,

    /// <summary>
    /// Prism group-128 base-3 ternary codec, 28 bytes per 128 weights (1.75 bpw).
    /// <c>GGML_TYPE_PTQ1_0</c>.
    /// </summary>
    PTQ1_0 = 143,
}

/// <summary>Block geometry and naming for <see cref="GgmlType"/>.</summary>
public static class GgmlTypeExtensions
{
    /// <summary>
    /// Number of elements one stored block covers. Unquantized types have a block of one.
    /// </summary>
    public static int BlockSize(this GgmlType type) => type switch
    {
        GgmlType.F32 or GgmlType.F16 or GgmlType.I32 or GgmlType.BF16 => 1,
        GgmlType.Q4_0 or GgmlType.Q4_1 or GgmlType.Q5_1 or GgmlType.Q8_0 => IntegerBlockGeometry.GroupSize,
        GgmlType.PQ2_0 => TernaryBlockGeometry.GroupSize,
        GgmlType.PTQ1_0 => TernaryBlockGeometry.GroupSize,
        _ => throw new ArgumentOutOfRangeException(nameof(type), type, "Unknown ggml type."),
    };

    /// <summary>Size in bytes of one stored block.</summary>
    public static int TypeSize(this GgmlType type) => type switch
    {
        GgmlType.F32 or GgmlType.I32 => 4,
        GgmlType.F16 or GgmlType.BF16 => 2,
        GgmlType.Q4_0 => IntegerBlockGeometry.Q40BlockBytes,
        GgmlType.Q4_1 => IntegerBlockGeometry.Q41BlockBytes,
        GgmlType.Q5_1 => IntegerBlockGeometry.Q51BlockBytes,
        GgmlType.Q8_0 => IntegerBlockGeometry.Q80BlockBytes,
        GgmlType.PQ2_0 => TernaryBlockGeometry.Pq20BlockBytes,
        GgmlType.PTQ1_0 => TernaryBlockGeometry.Ptq10BlockBytes,
        _ => throw new ArgumentOutOfRangeException(nameof(type), type, "Unknown ggml type."),
    };

    /// <summary>Whether the type stores blocks of quantized codes rather than plain elements.</summary>
    public static bool IsQuantized(this GgmlType type) =>
        type is GgmlType.PQ2_0 or GgmlType.PTQ1_0
            or GgmlType.Q4_0 or GgmlType.Q4_1 or GgmlType.Q5_1 or GgmlType.Q8_0;

    /// <summary>Whether the type's codes are ternary, as opposed to a wider integer grid.</summary>
    public static bool IsTernary(this GgmlType type) => type is GgmlType.PQ2_0 or GgmlType.PTQ1_0;

    /// <summary>The name ggml prints for the type, which is also what a GGUF dumper shows.</summary>
    public static string TypeName(this GgmlType type) => type switch
    {
        GgmlType.F32 => "f32",
        GgmlType.F16 => "f16",
        GgmlType.I32 => "i32",
        GgmlType.BF16 => "bf16",
        GgmlType.Q4_0 => "q4_0",
        GgmlType.Q4_1 => "q4_1",
        GgmlType.Q5_1 => "q5_1",
        GgmlType.Q8_0 => "q8_0",
        GgmlType.PQ2_0 => "pq2_0",
        GgmlType.PTQ1_0 => "ptq1_0",
        _ => throw new ArgumentOutOfRangeException(nameof(type), type, "Unknown ggml type."),
    };

    /// <summary>
    /// Bytes one row of <paramref name="elements"/> values occupies — ggml's <c>ggml_row_size</c>.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// The row length is not a multiple of the type's block size, which ggml requires of the
    /// fastest-varying dimension of a quantized tensor.
    /// </exception>
    public static long RowSize(this GgmlType type, long elements)
    {
        int block = type.BlockSize();
        if (elements % block != 0)
        {
            throw new ArgumentException(
                $"A {type.TypeName()} row must be a multiple of {block} elements but has {elements}.",
                nameof(elements));
        }

        return elements / block * type.TypeSize();
    }
}
