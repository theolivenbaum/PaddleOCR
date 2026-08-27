using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Formats;

/// <summary>A single array read out of a <c>.npy</c> stream.</summary>
/// <param name="Dtype">Element type.</param>
/// <param name="Shape">Dimensions, outermost first.</param>
/// <param name="Bytes">Raw little-endian payload in C order.</param>
public readonly record struct NpyArray(DType Dtype, int[] Shape, byte[] Bytes)
{
    /// <summary>Total number of elements.</summary>
    public int ElementCount => Tensor.ElementCount(Shape);

    /// <summary>Widens the array to float32.</summary>
    public float[] ToFloats()
    {
        float[] result = new float[ElementCount];
        FloatConversion.ToFloat(Bytes, Dtype, result);
        return result;
    }

    /// <summary>Reads the array as int64 values.</summary>
    public long[] ToInt64()
    {
        long[] result = new long[ElementCount];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = Dtype switch
            {
                DType.Int64 => BinaryPrimitives.ReadInt64LittleEndian(Bytes.AsSpan(i * 8, 8)),
                DType.Int32 => BinaryPrimitives.ReadInt32LittleEndian(Bytes.AsSpan(i * 4, 4)),
                _ => throw new InvalidOperationException($"Cannot read {Dtype} as int64."),
            };
        }

        return result;
    }

    /// <summary>Reads the array as raw bytes (only valid for uint8 arrays).</summary>
    public byte[] ToBytes() => Dtype == DType.UInt8
        ? Bytes
        : throw new InvalidOperationException($"Cannot read {Dtype} as bytes.");
}

/// <summary>
/// Minimal reader for NumPy's <c>.npy</c> and <c>.npz</c> containers, used to exchange
/// reference tensors with the Python dumpers under <c>dotnet/tools/reference</c>.
/// </summary>
public static class NpyFile
{
    /// <summary>The six magic bytes every .npy stream starts with: <c>0x93 "NUMPY"</c>.</summary>
    private static ReadOnlySpan<byte> Magic => [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y'];

    /// <summary>Reads a single <c>.npy</c> payload from <paramref name="stream"/>.</summary>
    public static NpyArray Read(Stream stream)
    {
        Span<byte> magic = stackalloc byte[6];
        stream.ReadExactly(magic);
        if (!magic.SequenceEqual(Magic))
        {
            throw new InvalidDataException("Stream does not start with the .npy magic bytes.");
        }

        int major = stream.ReadByte();
        _ = stream.ReadByte(); // minor version — irrelevant for parsing

        int headerLength;
        if (major == 1)
        {
            Span<byte> length = stackalloc byte[2];
            stream.ReadExactly(length);
            headerLength = BinaryPrimitives.ReadUInt16LittleEndian(length);
        }
        else
        {
            Span<byte> length = stackalloc byte[4];
            stream.ReadExactly(length);
            headerLength = (int)BinaryPrimitives.ReadUInt32LittleEndian(length);
        }

        byte[] headerBytes = new byte[headerLength];
        stream.ReadExactly(headerBytes);
        string header = Encoding.ASCII.GetString(headerBytes);

        string descriptor = ExtractValue(header, "descr").Trim('\'', '"');
        bool fortranOrder = ExtractValue(header, "fortran_order").Contains("True", StringComparison.Ordinal);
        if (fortranOrder)
        {
            throw new NotSupportedException("Fortran-ordered .npy arrays are not supported.");
        }

        int[] shape = ParseShape(ExtractValue(header, "shape"));
        DType dtype = ParseDescriptor(descriptor);

        int count = Tensor.ElementCount(shape);
        byte[] payload = new byte[(long)count * dtype.ByteSize()];
        stream.ReadExactly(payload);

        return new NpyArray(dtype, shape, payload);
    }

    /// <summary>Reads every array in a <c>.npz</c> archive, keyed by entry name without the extension.</summary>
    public static Dictionary<string, NpyArray> ReadArchive(string path)
    {
        using FileStream file = File.OpenRead(path);
        using var archive = new ZipArchive(file, ZipArchiveMode.Read);

        var result = new Dictionary<string, NpyArray>(StringComparer.Ordinal);
        foreach (ZipArchiveEntry entry in archive.Entries)
        {
            string name = entry.Name.EndsWith(".npy", StringComparison.Ordinal)
                ? entry.Name[..^4]
                : entry.Name;

            // ZipArchive entry streams are forward-only; copy so `Read` can seek-free parse.
            using Stream source = entry.Open();
            using var buffer = new MemoryStream((int)entry.Length);
            source.CopyTo(buffer);
            buffer.Position = 0;
            result[name] = Read(buffer);
        }

        return result;
    }

    private static string ExtractValue(string header, string key)
    {
        int keyIndex = header.IndexOf($"'{key}'", StringComparison.Ordinal);
        if (keyIndex < 0)
        {
            throw new InvalidDataException($".npy header has no '{key}' entry.");
        }

        int colon = header.IndexOf(':', keyIndex);
        int cursor = colon + 1;
        int depth = 0;
        var value = new StringBuilder();

        while (cursor < header.Length)
        {
            char c = header[cursor];
            if (c == '(')
            {
                depth++;
            }
            else if (c == ')')
            {
                depth--;
            }
            else if (c == ',' && depth == 0)
            {
                break;
            }
            else if (c == '}' && depth == 0)
            {
                break;
            }

            value.Append(c);
            cursor++;
        }

        return value.ToString().Trim();
    }

    private static int[] ParseShape(string value)
    {
        string inner = value.Trim().Trim('(', ')').Trim();
        if (inner.Length == 0)
        {
            return [];
        }

        string[] parts = inner.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        int[] shape = new int[parts.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            shape[i] = int.Parse(parts[i]);
        }

        return shape;
    }

    private static DType ParseDescriptor(string descriptor) => descriptor switch
    {
        "<f4" or "=f4" or "f4" => DType.Float32,
        "<f2" or "=f2" or "f2" => DType.Float16,
        "<i8" or "=i8" or "i8" => DType.Int64,
        "<i4" or "=i4" or "i4" => DType.Int32,
        "|i1" or "i1" => DType.Int8,
        "|u1" or "u1" => DType.UInt8,
        "|b1" or "b1" => DType.Bool,
        _ => throw new NotSupportedException($"Unsupported .npy dtype descriptor '{descriptor}'."),
    };
}
