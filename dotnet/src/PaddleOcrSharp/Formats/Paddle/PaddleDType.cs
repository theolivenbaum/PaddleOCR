namespace PaddleOcrSharp.Formats.Paddle;

/// <summary>Element types that appear in a Paddle inference program.</summary>
public enum PaddleDType
{
    /// <summary>32-bit float.</summary>
    Float32,

    /// <summary>64-bit float.</summary>
    Float64,

    /// <summary>16-bit float.</summary>
    Float16,

    /// <summary>Truncated 32-bit float.</summary>
    BFloat16,

    /// <summary>32-bit signed integer.</summary>
    Int32,

    /// <summary>64-bit signed integer.</summary>
    Int64,

    /// <summary>16-bit signed integer.</summary>
    Int16,

    /// <summary>8-bit signed integer.</summary>
    Int8,

    /// <summary>8-bit unsigned integer.</summary>
    UInt8,

    /// <summary>Single-byte boolean.</summary>
    Bool,
}

/// <summary>Helpers for <see cref="PaddleDType"/>.</summary>
public static class PaddleDTypeExtensions
{
    /// <summary>Whether the type is a floating-point type.</summary>
    public static bool IsFloat(this PaddleDType dtype) =>
        dtype is PaddleDType.Float32 or PaddleDType.Float64 or PaddleDType.Float16 or PaddleDType.BFloat16;

    /// <summary>Bytes per element.</summary>
    public static int ByteSize(this PaddleDType dtype) => dtype switch
    {
        PaddleDType.Float32 or PaddleDType.Int32 => 4,
        PaddleDType.Float64 or PaddleDType.Int64 => 8,
        PaddleDType.Float16 or PaddleDType.BFloat16 or PaddleDType.Int16 => 2,
        PaddleDType.Int8 or PaddleDType.UInt8 or PaddleDType.Bool => 1,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, "Unknown dtype."),
    };

    /// <summary>Maps a PIR type tag such as <c>0.t_f32</c> onto a dtype.</summary>
    public static PaddleDType FromPirTag(string tag) => tag switch
    {
        "0.t_f32" => PaddleDType.Float32,
        "0.t_f64" => PaddleDType.Float64,
        "0.t_f16" => PaddleDType.Float16,
        "0.t_bf16" => PaddleDType.BFloat16,
        "0.t_i32" => PaddleDType.Int32,
        "0.t_i64" => PaddleDType.Int64,
        "0.t_i16" => PaddleDType.Int16,
        "0.t_i8" => PaddleDType.Int8,
        "0.t_ui8" or "0.t_u8" => PaddleDType.UInt8,
        "0.t_bool" => PaddleDType.Bool,
        _ => throw new NotSupportedException($"Unsupported PIR element type '{tag}'."),
    };

    /// <summary>Maps a dtype attribute string such as <c>float32</c> onto a dtype.</summary>
    public static PaddleDType FromName(string name) => name switch
    {
        "float32" or "float" => PaddleDType.Float32,
        "float64" or "double" => PaddleDType.Float64,
        "float16" => PaddleDType.Float16,
        "bfloat16" => PaddleDType.BFloat16,
        "int32" => PaddleDType.Int32,
        "int64" => PaddleDType.Int64,
        "int16" => PaddleDType.Int16,
        "int8" => PaddleDType.Int8,
        "uint8" => PaddleDType.UInt8,
        "bool" => PaddleDType.Bool,
        _ => throw new NotSupportedException($"Unsupported dtype name '{name}'."),
    };

    /// <summary>Maps the numeric <c>VarType</c> code used inside <c>.pdiparams</c> onto a dtype.</summary>
    public static PaddleDType FromVarTypeCode(int code) => code switch
    {
        0 => PaddleDType.Bool,
        1 => PaddleDType.Int16,
        2 => PaddleDType.Int32,
        3 => PaddleDType.Int64,
        4 => PaddleDType.Float16,
        5 => PaddleDType.Float32,
        6 => PaddleDType.Float64,
        20 => PaddleDType.UInt8,
        21 => PaddleDType.Int8,
        22 => PaddleDType.BFloat16,
        _ => throw new NotSupportedException($"Unsupported Paddle VarType code {code}."),
    };
}
