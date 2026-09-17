using System.Buffers.Binary;
using System.Text;

namespace PaddleOcrSharp.Formats.Gguf;

/// <summary>
/// Writes a GGUF v3 container, streaming the tensor data rather than holding it.
/// </summary>
/// <remarks>
/// <para>
/// A GGUF header carries each tensor's offset, so the sizes have to be known before any data is
/// written. The writer therefore has two phases: declare every tensor, then
/// <see cref="BeginData"/>, then write each tensor's bytes in the order they were declared.
/// A 1.8 GB checkpoint converts one tensor at a time this way, never holding more than one.
/// </para>
/// <para>
/// The byte layout follows <c>gguf_write_out</c> in the <c>PrismML-Eng/llama.cpp</c> fork
/// (<c>ggml/src/gguf.cpp</c>), so a file written here is the file that fork would write.
/// </para>
/// </remarks>
public sealed class GgufWriter : IDisposable
{
    private readonly Stream _stream;
    private readonly bool _ownsStream;
    private readonly int _alignment;
    private readonly List<KeyValuePair<string, GgufValue>> _metadata = [];
    private readonly List<Declaration> _declarations = [];
    private readonly Dictionary<string, int> _byName = new(StringComparer.Ordinal);

    private long _dataStart = -1;
    private int _written;

    /// <summary>Creates a writer over <paramref name="path"/>.</summary>
    /// <param name="path">File to create, overwriting anything already there.</param>
    /// <param name="alignment">Data alignment; GGUF's default is 32.</param>
    public GgufWriter(string path, int alignment = GgufFile.DefaultAlignment)
        : this(new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20), true, alignment)
    {
    }

    /// <summary>Creates a writer over a caller-owned stream.</summary>
    /// <param name="stream">Seekable, writable destination.</param>
    /// <param name="ownsStream">Whether <see cref="Dispose"/> should close it.</param>
    /// <param name="alignment">Data alignment; GGUF's default is 32.</param>
    public GgufWriter(Stream stream, bool ownsStream, int alignment = GgufFile.DefaultAlignment)
    {
        if (alignment <= 0 || (alignment & (alignment - 1)) != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(alignment), alignment, "Alignment must be a power of two.");
        }

        _stream = stream;
        _ownsStream = ownsStream;
        _alignment = alignment;
        Add("general.alignment", GgufValue.From((uint)alignment));
    }

    /// <summary>Adds or replaces a metadata entry.</summary>
    /// <param name="key">Metadata key, e.g. <c>general.architecture</c>.</param>
    /// <param name="value">The value.</param>
    /// <exception cref="InvalidOperationException">Metadata is closed; <see cref="BeginData"/> ran.</exception>
    public void Add(string key, GgufValue value)
    {
        EnsureDeclaring();
        int existing = _metadata.FindIndex(entry => entry.Key == key);
        var pair = new KeyValuePair<string, GgufValue>(key, value);
        if (existing >= 0)
        {
            _metadata[existing] = pair;
        }
        else
        {
            _metadata.Add(pair);
        }
    }

    /// <summary>Adds a string metadata entry.</summary>
    public void Add(string key, string value) => Add(key, GgufValue.From(value));

    /// <summary>Adds an unsigned 32-bit metadata entry.</summary>
    public void Add(string key, uint value) => Add(key, GgufValue.From(value));

    /// <summary>
    /// Declares a tensor. Every tensor must be declared before <see cref="BeginData"/>.
    /// </summary>
    /// <param name="name">Tensor name.</param>
    /// <param name="type">Stored ggml type.</param>
    /// <param name="ne">
    /// Dimensions in GGUF order, fastest-varying first — <c>[in_features, out_features]</c> for an
    /// <c>nn.Linear</c> weight.
    /// </param>
    /// <returns>The number of bytes the tensor's data will occupy.</returns>
    public long Declare(string name, GgmlType type, ReadOnlySpan<long> ne)
    {
        EnsureDeclaring();

        if (ne.Length is 0 or > 4)
        {
            throw new ArgumentException($"Tensor '{name}' must have between 1 and 4 dimensions.", nameof(ne));
        }

        long elements = 1;
        foreach (long dimension in ne)
        {
            if (dimension <= 0)
            {
                throw new ArgumentException($"Tensor '{name}' has a non-positive dimension.", nameof(ne));
            }

            elements *= dimension;
        }

        long length = type.RowSize(ne[0]) * (elements / ne[0]);

        long offset = _declarations.Count == 0
            ? 0
            : GgufFile.Align(_declarations[^1].Offset + _declarations[^1].Length, _alignment);

        if (!_byName.TryAdd(name, _declarations.Count))
        {
            throw new ArgumentException($"Tensor '{name}' is declared twice.", nameof(name));
        }

        _declarations.Add(new Declaration(name, type, ne.ToArray(), offset, length));
        return length;
    }

    /// <summary>
    /// Writes the header and opens the data section. No further declarations are accepted.
    /// </summary>
    public void BeginData()
    {
        EnsureDeclaring();

        WriteUInt32(GgufFile.Magic);
        WriteUInt32(GgufFile.Version);
        WriteInt64(_declarations.Count);
        WriteInt64(_metadata.Count);

        foreach ((string key, GgufValue value) in _metadata)
        {
            WriteString(key);
            WriteValue(value);
        }

        foreach (Declaration declaration in _declarations)
        {
            WriteString(declaration.Name);
            WriteUInt32((uint)declaration.Ne.Length);
            foreach (long dimension in declaration.Ne)
            {
                WriteInt64(dimension);
            }

            WriteUInt32((uint)declaration.Type);
            WriteInt64(declaration.Offset);
        }

        Pad();
        _dataStart = _stream.Position;
    }

    /// <summary>
    /// Writes one tensor's data. Tensors must be written in the order they were declared.
    /// </summary>
    /// <param name="name">Tensor name; must match the next declaration.</param>
    /// <param name="data">Exactly the bytes <see cref="Declare"/> returned for it.</param>
    public void Write(string name, ReadOnlySpan<byte> data)
    {
        if (_dataStart < 0)
        {
            throw new InvalidOperationException("Call BeginData before writing tensor data.");
        }

        if (_written >= _declarations.Count)
        {
            throw new InvalidOperationException("Every declared tensor has already been written.");
        }

        Declaration declaration = _declarations[_written];
        if (declaration.Name != name)
        {
            throw new InvalidOperationException(
                $"Tensors must be written in declaration order; expected '{declaration.Name}' but got '{name}'.");
        }

        if (data.Length != declaration.Length)
        {
            throw new ArgumentException(
                $"Tensor '{name}' needs {declaration.Length} bytes but {data.Length} were given.", nameof(data));
        }

        if (_stream.Position != _dataStart + declaration.Offset)
        {
            throw new InvalidOperationException($"Stream is not positioned at '{name}'.");
        }

        _stream.Write(data);
        _written++;

        if (_written < _declarations.Count)
        {
            long target = _dataStart + _declarations[_written].Offset;
            while (_stream.Position < target)
            {
                _stream.WriteByte(0);
            }
        }
    }

    /// <summary>Checks that every declared tensor was written, and flushes.</summary>
    public void Complete()
    {
        if (_written != _declarations.Count)
        {
            throw new InvalidOperationException(
                $"{_declarations.Count - _written} declared tensors were never written.");
        }

        _stream.Flush();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_ownsStream)
        {
            _stream.Dispose();
        }
    }

    private void EnsureDeclaring()
    {
        if (_dataStart >= 0)
        {
            throw new InvalidOperationException("The header has already been written.");
        }
    }

    private void Pad()
    {
        while (_stream.Position % _alignment != 0)
        {
            _stream.WriteByte(0);
        }
    }

    private void WriteValue(GgufValue value)
    {
        if (value.IsArray)
        {
            WriteUInt32((uint)GgufValueType.Array);
            WriteUInt32((uint)value.Type);
            WriteInt64(value.Count);
            foreach (object element in (Array)value.Payload)
            {
                WriteScalar(value.Type, element);
            }

            return;
        }

        WriteUInt32((uint)value.Type);
        WriteScalar(value.Type, value.Payload);
    }

    private void WriteScalar(GgufValueType type, object value)
    {
        switch (type)
        {
            case GgufValueType.UInt8:
                _stream.WriteByte((byte)value);
                break;
            case GgufValueType.Int8:
                _stream.WriteByte((byte)(sbyte)value);
                break;
            case GgufValueType.Bool:
                _stream.WriteByte((bool)value ? (byte)1 : (byte)0);
                break;
            case GgufValueType.UInt16:
                WriteScalarBytes(2, (span) => BinaryPrimitives.WriteUInt16LittleEndian(span, (ushort)value));
                break;
            case GgufValueType.Int16:
                WriteScalarBytes(2, (span) => BinaryPrimitives.WriteInt16LittleEndian(span, (short)value));
                break;
            case GgufValueType.UInt32:
                WriteUInt32((uint)value);
                break;
            case GgufValueType.Int32:
                WriteScalarBytes(4, (span) => BinaryPrimitives.WriteInt32LittleEndian(span, (int)value));
                break;
            case GgufValueType.Float32:
                WriteScalarBytes(4, (span) => BinaryPrimitives.WriteSingleLittleEndian(span, (float)value));
                break;
            case GgufValueType.UInt64:
                WriteScalarBytes(8, (span) => BinaryPrimitives.WriteUInt64LittleEndian(span, (ulong)value));
                break;
            case GgufValueType.Int64:
                WriteInt64((long)value);
                break;
            case GgufValueType.Float64:
                WriteScalarBytes(8, (span) => BinaryPrimitives.WriteDoubleLittleEndian(span, (double)value));
                break;
            case GgufValueType.String:
                WriteString((string)value);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(type), type, "Unknown GGUF value type.");
        }
    }

    private void WriteScalarBytes(int size, SpanAction write)
    {
        Span<byte> buffer = stackalloc byte[8];
        Span<byte> slice = buffer[..size];
        write(slice);
        _stream.Write(slice);
    }

    private void WriteUInt32(uint value)
    {
        Span<byte> buffer = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(buffer, value);
        _stream.Write(buffer);
    }

    private void WriteInt64(long value)
    {
        Span<byte> buffer = stackalloc byte[8];
        BinaryPrimitives.WriteInt64LittleEndian(buffer, value);
        _stream.Write(buffer);
    }

    private void WriteString(string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        WriteInt64(bytes.Length);
        _stream.Write(bytes);
    }

    private delegate void SpanAction(Span<byte> destination);

    private readonly record struct Declaration(string Name, GgmlType Type, long[] Ne, long Offset, long Length);
}
