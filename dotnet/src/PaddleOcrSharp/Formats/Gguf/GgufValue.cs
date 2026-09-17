using System.Globalization;

namespace PaddleOcrSharp.Formats.Gguf;

/// <summary>Types a GGUF key/value entry can hold, with GGUF's own numbering.</summary>
public enum GgufValueType
{
    /// <summary>Unsigned 8-bit integer.</summary>
    UInt8 = 0,

    /// <summary>Signed 8-bit integer.</summary>
    Int8 = 1,

    /// <summary>Unsigned 16-bit integer.</summary>
    UInt16 = 2,

    /// <summary>Signed 16-bit integer.</summary>
    Int16 = 3,

    /// <summary>Unsigned 32-bit integer.</summary>
    UInt32 = 4,

    /// <summary>Signed 32-bit integer.</summary>
    Int32 = 5,

    /// <summary>IEEE-754 binary32.</summary>
    Float32 = 6,

    /// <summary>Single-byte boolean.</summary>
    Bool = 7,

    /// <summary>Length-prefixed UTF-8, with no terminator.</summary>
    String = 8,

    /// <summary>An array of one of the other types.</summary>
    Array = 9,

    /// <summary>Unsigned 64-bit integer.</summary>
    UInt64 = 10,

    /// <summary>Signed 64-bit integer.</summary>
    Int64 = 11,

    /// <summary>IEEE-754 binary64.</summary>
    Float64 = 12,
}

/// <summary>
/// One GGUF metadata value: a scalar, or an array of one element type.
/// </summary>
/// <remarks>
/// Values are boxed. A checkpoint's metadata is a few dozen entries read once at load, so the
/// allocation is irrelevant and the alternative — a discriminated union over thirteen types — is
/// not worth the surface.
/// </remarks>
public sealed class GgufValue
{
    private readonly object _value;

    private GgufValue(GgufValueType type, bool isArray, object value)
    {
        Type = type;
        IsArray = isArray;
        _value = value;
    }

    /// <summary>Element type. For an array, the type of its elements.</summary>
    public GgufValueType Type { get; }

    /// <summary>Whether this entry is an array.</summary>
    public bool IsArray { get; }

    /// <summary>Number of elements; one for a scalar.</summary>
    public int Count => IsArray ? ((Array)_value).Length : 1;

    /// <summary>Wraps a string.</summary>
    public static GgufValue From(string value) => new(GgufValueType.String, false, value);

    /// <summary>Wraps an unsigned 32-bit integer.</summary>
    public static GgufValue From(uint value) => new(GgufValueType.UInt32, false, value);

    /// <summary>Wraps a signed 32-bit integer.</summary>
    public static GgufValue From(int value) => new(GgufValueType.Int32, false, value);

    /// <summary>Wraps a 32-bit float.</summary>
    public static GgufValue From(float value) => new(GgufValueType.Float32, false, value);

    /// <summary>Wraps a boolean.</summary>
    public static GgufValue From(bool value) => new(GgufValueType.Bool, false, value);

    /// <summary>Wraps a string array.</summary>
    public static GgufValue From(string[] values) => new(GgufValueType.String, true, values);

    /// <summary>Wraps a signed 32-bit integer array.</summary>
    public static GgufValue From(int[] values) => new(GgufValueType.Int32, true, values);

    /// <summary>Wraps a 32-bit float array.</summary>
    public static GgufValue From(float[] values) => new(GgufValueType.Float32, true, values);

    /// <summary>Builds a value from an already-typed payload, for the reader.</summary>
    internal static GgufValue Raw(GgufValueType type, bool isArray, object value) => new(type, isArray, value);

    /// <summary>The value as a string, formatting a scalar of any other type.</summary>
    public string AsString() => _value switch
    {
        string text => text,
        string[] texts => string.Join(", ", texts),
        IFormattable formattable => formattable.ToString(null, CultureInfo.InvariantCulture),
        _ => _value.ToString() ?? string.Empty,
    };

    /// <summary>The value as an unsigned 32-bit integer.</summary>
    /// <exception cref="InvalidOperationException">The entry is not an integer scalar.</exception>
    public uint AsUInt32() => _value switch
    {
        uint value => value,
        int value => (uint)value,
        ulong value => (uint)value,
        long value => (uint)value,
        ushort value => value,
        short value => (uint)value,
        byte value => value,
        sbyte value => (uint)value,
        _ => throw new InvalidOperationException($"A {Type} value is not an integer."),
    };

    /// <summary>The value as a 32-bit float.</summary>
    /// <exception cref="InvalidOperationException">The entry is not a floating-point scalar.</exception>
    public float AsSingle() => _value switch
    {
        float value => value,
        double value => (float)value,
        _ => throw new InvalidOperationException($"A {Type} value is not a float."),
    };

    /// <summary>The value as a boolean.</summary>
    /// <exception cref="InvalidOperationException">The entry is not a boolean scalar.</exception>
    public bool AsBoolean() => _value is bool value
        ? value
        : throw new InvalidOperationException($"A {Type} value is not a boolean.");

    /// <summary>The value as a string array.</summary>
    /// <exception cref="InvalidOperationException">The entry is not a string array.</exception>
    public string[] AsStringArray() => _value switch
    {
        string[] values => values,
        string value => [value],
        _ => throw new InvalidOperationException($"A {Type} value is not a string array."),
    };

    /// <summary>The value as a signed 32-bit integer array.</summary>
    /// <exception cref="InvalidOperationException">The entry is not an integer array.</exception>
    public int[] AsInt32Array() => _value switch
    {
        int[] values => values,
        int value => [value],
        _ => throw new InvalidOperationException($"A {Type} value is not an integer array."),
    };

    /// <summary>The value as a 32-bit float array.</summary>
    /// <exception cref="InvalidOperationException">The entry is not a float array.</exception>
    public float[] AsSingleArray() => _value switch
    {
        float[] values => values,
        float value => [value],
        _ => throw new InvalidOperationException($"A {Type} value is not a float array."),
    };

    /// <summary>The boxed payload, for the writer.</summary>
    internal object Payload => _value;

    /// <inheritdoc />
    public override string ToString() => IsArray ? $"[{Count} × {Type}]" : AsString();
}
