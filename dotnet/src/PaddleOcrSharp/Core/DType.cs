namespace PaddleOcrSharp.Core;

/// <summary>
/// Element types that can appear in a weight file we load (safetensors or Paddle
/// <c>.pdiparams</c>). Only the subset actually produced by the upstream exporters is listed.
/// </summary>
public enum DType
{
    /// <summary>IEEE-754 binary32.</summary>
    Float32,

    /// <summary>IEEE-754 binary64. Reference fixtures use it; no model weight does.</summary>
    Float64,

    /// <summary>IEEE-754 binary16.</summary>
    Float16,

    /// <summary>Truncated binary32 ("brain float"); the dtype PaddleOCR-VL ships in.</summary>
    BFloat16,

    /// <summary>Signed 64-bit integer.</summary>
    Int64,

    /// <summary>Signed 32-bit integer.</summary>
    Int32,

    /// <summary>Signed 8-bit integer.</summary>
    Int8,

    /// <summary>Unsigned 8-bit integer.</summary>
    UInt8,

    /// <summary>Single-byte boolean.</summary>
    Bool,
}

/// <summary>Helpers for <see cref="DType"/>.</summary>
public static class DTypeExtensions
{
    /// <summary>Size, in bytes, of a single element of <paramref name="dtype"/>.</summary>
    public static int ByteSize(this DType dtype) => dtype switch
    {
        DType.Float32 => 4,
        DType.Float64 => 8,
        DType.Float16 => 2,
        DType.BFloat16 => 2,
        DType.Int64 => 8,
        DType.Int32 => 4,
        DType.Int8 => 1,
        DType.UInt8 => 1,
        DType.Bool => 1,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, "Unknown dtype."),
    };

    /// <summary>Maps a safetensors dtype string (e.g. <c>"BF16"</c>) onto a <see cref="DType"/>.</summary>
    public static DType FromSafetensors(string name) => name switch
    {
        "F32" => DType.Float32,
        "F16" => DType.Float16,
        "BF16" => DType.BFloat16,
        "I64" => DType.Int64,
        "I32" => DType.Int32,
        "I8" => DType.Int8,
        "U8" => DType.UInt8,
        "BOOL" => DType.Bool,
        _ => throw new NotSupportedException($"Unsupported safetensors dtype '{name}'."),
    };
}
