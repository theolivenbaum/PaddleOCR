using System.Buffers.Binary;
using System.IO.MemoryMappedFiles;
using System.Text;

namespace PaddleOcrSharp.Formats.Gguf;

/// <summary>One tensor inside a GGUF file, still in its stored ggml type.</summary>
/// <param name="Name">Tensor name, e.g. <c>model.layers.0.mlp.up_proj.weight</c>.</param>
/// <param name="Type">Stored ggml type.</param>
/// <param name="Ne">
/// Dimensions in GGUF order — fastest-varying first, so an <c>nn.Linear</c> weight is
/// <c>[in_features, out_features]</c>.
/// </param>
/// <param name="Bytes">The raw bytes, mapped straight from the file.</param>
public readonly record struct GgufTensor(string Name, GgmlType Type, long[] Ne, ReadOnlyMemory<byte> Bytes)
{
    /// <summary>Total number of elements.</summary>
    public long ElementCount
    {
        get
        {
            long count = 1;
            foreach (long dimension in Ne)
            {
                count *= dimension;
            }

            return count;
        }
    }

    /// <summary>Length of one row, in elements — GGUF's <c>ne[0]</c>.</summary>
    public long RowLength => Ne.Length > 0 ? Ne[0] : 0;

    /// <summary>Number of rows.</summary>
    public long RowCount => RowLength == 0 ? 0 : ElementCount / RowLength;

    /// <summary>
    /// Dimensions in the port's row-major order, outermost first — the reverse of
    /// <see cref="Ne"/>, so an <c>nn.Linear</c> weight is <c>[out_features, in_features]</c>.
    /// </summary>
    public int[] Shape
    {
        get
        {
            int[] shape = new int[Ne.Length];
            for (int i = 0; i < Ne.Length; i++)
            {
                shape[i] = checked((int)Ne[Ne.Length - 1 - i]);
            }

            return shape;
        }
    }
}

/// <summary>
/// Reader for the <see href="https://github.com/ggml-org/ggml/blob/master/docs/gguf.md">GGUF</see>
/// container, version 3.
/// </summary>
/// <remarks>
/// <para>
/// The layout is: the magic <c>GGUF</c>, a <c>uint32</c> version, an <c>int64</c> tensor count, an
/// <c>int64</c> metadata count, the metadata entries, one info record per tensor (name, rank,
/// <c>ne</c>, type, offset), padding to <c>general.alignment</c>, then the tensor data. Every
/// offset in a tensor info is relative to the start of that data section.
/// </para>
/// <para>
/// The file is memory-mapped, exactly as <see cref="SafetensorsFile"/> is, so a 1.8 GB checkpoint
/// costs no managed allocation and the OS pages weights in on demand.
/// </para>
/// </remarks>
public sealed class GgufFile : IDisposable
{
    /// <summary>The four bytes every GGUF file starts with.</summary>
    public const uint Magic = 0x46554747; // "GGUF", little-endian

    /// <summary>The only container version this reader accepts.</summary>
    public const uint Version = 3;

    /// <summary>The alignment used when a file does not say otherwise.</summary>
    public const int DefaultAlignment = 32;

    private readonly MemoryMappedFile _file;
    private readonly MemoryMappedViewAccessor _view;
    private readonly Dictionary<string, GgufTensor> _tensors;
    private unsafe byte* _base;

    private GgufFile(
        MemoryMappedFile file,
        MemoryMappedViewAccessor view,
        Dictionary<string, GgufTensor> tensors,
        IReadOnlyDictionary<string, GgufValue> metadata,
        int alignment)
    {
        _file = file;
        _view = view;
        _tensors = tensors;
        Metadata = metadata;
        Alignment = alignment;
    }

    /// <summary>Every key/value entry in the header.</summary>
    public IReadOnlyDictionary<string, GgufValue> Metadata { get; }

    /// <summary>The file's data alignment.</summary>
    public int Alignment { get; }

    /// <summary>Names of every tensor in the file.</summary>
    public IReadOnlyCollection<string> Names => _tensors.Keys;

    /// <summary>Number of tensors in the file.</summary>
    public int Count => _tensors.Count;

    /// <summary>Opens <paramref name="path"/> and parses its header.</summary>
    /// <exception cref="InvalidDataException">The file is not a GGUF v3 container.</exception>
    public static unsafe GgufFile Open(string path)
    {
        long fileLength = new FileInfo(path).Length;
        if (fileLength < 24)
        {
            throw new InvalidDataException($"'{path}' is too short to be a GGUF file.");
        }

        MemoryMappedFile file = MemoryMappedFile.CreateFromFile(
            path, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);
        MemoryMappedViewAccessor view = file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);

        byte* basePointer = null;
        view.SafeMemoryMappedViewHandle.AcquirePointer(ref basePointer);

        try
        {
            var cursor = new Cursor(new ReadOnlySpan<byte>(basePointer, checked((int)Math.Min(fileLength, int.MaxValue))));

            uint magic = cursor.ReadUInt32();
            if (magic != Magic)
            {
                throw new InvalidDataException($"'{path}' does not start with the GGUF magic.");
            }

            uint version = cursor.ReadUInt32();
            if (version != Version)
            {
                throw new InvalidDataException($"'{path}' is GGUF version {version}; only {Version} is supported.");
            }

            long tensorCount = cursor.ReadInt64();
            long metadataCount = cursor.ReadInt64();
            if (tensorCount < 0 || metadataCount < 0)
            {
                throw new InvalidDataException($"'{path}' declares a negative tensor or metadata count.");
            }

            var metadata = new Dictionary<string, GgufValue>(StringComparer.Ordinal);
            for (long i = 0; i < metadataCount; i++)
            {
                string key = cursor.ReadString();
                metadata[key] = cursor.ReadValue();
            }

            int alignment = metadata.TryGetValue("general.alignment", out GgufValue? value)
                ? checked((int)value.AsUInt32())
                : DefaultAlignment;

            if (alignment <= 0 || (alignment & (alignment - 1)) != 0)
            {
                throw new InvalidDataException($"'{path}' declares a non-power-of-two alignment of {alignment}.");
            }

            var infos = new List<(string Name, GgmlType Type, long[] Ne, long Offset)>(checked((int)tensorCount));
            for (long i = 0; i < tensorCount; i++)
            {
                string name = cursor.ReadString();
                uint rank = cursor.ReadUInt32();
                if (rank > 4)
                {
                    throw new InvalidDataException($"Tensor '{name}' has rank {rank}; GGUF allows at most 4.");
                }

                long[] ne = new long[rank];
                for (int d = 0; d < rank; d++)
                {
                    ne[d] = cursor.ReadInt64();
                }

                var type = (GgmlType)cursor.ReadUInt32();
                long offset = cursor.ReadInt64();
                infos.Add((name, type, ne, offset));
            }

            long dataStart = Align(cursor.Position, alignment);

            var tensors = new Dictionary<string, GgufTensor>(StringComparer.Ordinal);
            foreach ((string name, GgmlType type, long[] ne, long offset) in infos)
            {
                long elements = 1;
                foreach (long dimension in ne)
                {
                    if (dimension <= 0)
                    {
                        throw new InvalidDataException($"Tensor '{name}' has a non-positive dimension.");
                    }

                    elements *= dimension;
                }

                long rowLength = ne.Length > 0 ? ne[0] : 0;
                long length = rowLength == 0 ? 0 : type.RowSize(rowLength) * (elements / rowLength);

                if (offset < 0 || dataStart + offset + length > fileLength)
                {
                    throw new InvalidDataException($"Tensor '{name}' runs past the end of '{path}'.");
                }

                var manager = new UnmanagedMemoryManager<byte>(
                    basePointer + dataStart + offset, checked((int)length));
                tensors[name] = new GgufTensor(name, type, ne, manager.Memory);
            }

            return new GgufFile(file, view, tensors, metadata, alignment) { _base = basePointer };
        }
        catch
        {
            view.SafeMemoryMappedViewHandle.ReleasePointer();
            view.Dispose();
            file.Dispose();
            throw;
        }
    }

    /// <summary>Gets a tensor by name.</summary>
    /// <exception cref="KeyNotFoundException">The file has no such tensor.</exception>
    public GgufTensor this[string name] =>
        _tensors.TryGetValue(name, out GgufTensor tensor)
            ? tensor
            : throw new KeyNotFoundException($"Checkpoint has no tensor named '{name}'.");

    /// <summary>Tries to get a tensor by name.</summary>
    public bool TryGet(string name, out GgufTensor tensor) => _tensors.TryGetValue(name, out tensor);

    /// <summary>Whether the file contains <paramref name="name"/>.</summary>
    public bool Contains(string name) => _tensors.ContainsKey(name);

    /// <summary>Gets a metadata entry, or <see langword="null"/> when it is absent.</summary>
    public GgufValue? Meta(string key) => Metadata.GetValueOrDefault(key);

    /// <summary>Rounds <paramref name="offset"/> up to a multiple of <paramref name="alignment"/>.</summary>
    public static long Align(long offset, int alignment) => (offset + alignment - 1) / alignment * alignment;

    /// <inheritdoc />
    public unsafe void Dispose()
    {
        _tensors.Clear();
        if (_base is not null)
        {
            _view.SafeMemoryMappedViewHandle.ReleasePointer();
            _base = null;
        }

        _view.Dispose();
        _file.Dispose();
    }

    /// <summary>A forward-only cursor over the mapped header.</summary>
    private ref struct Cursor(ReadOnlySpan<byte> data)
    {
        private readonly ReadOnlySpan<byte> _data = data;
        private int _position;

        public readonly long Position => _position;

        public uint ReadUInt32()
        {
            uint value = BinaryPrimitives.ReadUInt32LittleEndian(Take(4));
            return value;
        }

        public long ReadInt64() => BinaryPrimitives.ReadInt64LittleEndian(Take(8));

        public string ReadString()
        {
            long length = ReadInt64();
            if (length < 0 || length > int.MaxValue)
            {
                throw new InvalidDataException($"GGUF string length {length} is out of range.");
            }

            return Encoding.UTF8.GetString(Take((int)length));
        }

        public GgufValue ReadValue()
        {
            var type = (GgufValueType)ReadUInt32();
            if (type != GgufValueType.Array)
            {
                return GgufValue.Raw(type, isArray: false, ReadScalar(type));
            }

            var elementType = (GgufValueType)ReadUInt32();
            long count = ReadInt64();
            if (count < 0 || count > int.MaxValue)
            {
                throw new InvalidDataException($"GGUF array length {count} is out of range.");
            }

            return GgufValue.Raw(elementType, isArray: true, ReadArray(elementType, (int)count));
        }

        private object ReadScalar(GgufValueType type) => type switch
        {
            GgufValueType.UInt8 => Take(1)[0],
            GgufValueType.Int8 => (sbyte)Take(1)[0],
            GgufValueType.UInt16 => BinaryPrimitives.ReadUInt16LittleEndian(Take(2)),
            GgufValueType.Int16 => BinaryPrimitives.ReadInt16LittleEndian(Take(2)),
            GgufValueType.UInt32 => BinaryPrimitives.ReadUInt32LittleEndian(Take(4)),
            GgufValueType.Int32 => BinaryPrimitives.ReadInt32LittleEndian(Take(4)),
            GgufValueType.Float32 => BinaryPrimitives.ReadSingleLittleEndian(Take(4)),
            GgufValueType.Bool => Take(1)[0] != 0,
            GgufValueType.String => ReadString(),
            GgufValueType.UInt64 => BinaryPrimitives.ReadUInt64LittleEndian(Take(8)),
            GgufValueType.Int64 => BinaryPrimitives.ReadInt64LittleEndian(Take(8)),
            GgufValueType.Float64 => BinaryPrimitives.ReadDoubleLittleEndian(Take(8)),
            _ => throw new InvalidDataException($"Unknown GGUF value type {(int)type}."),
        };

        private object ReadArray(GgufValueType type, int count)
        {
            switch (type)
            {
                case GgufValueType.String:
                {
                    string[] values = new string[count];
                    for (int i = 0; i < count; i++)
                    {
                        values[i] = ReadString();
                    }

                    return values;
                }

                case GgufValueType.Int32:
                {
                    int[] values = new int[count];
                    for (int i = 0; i < count; i++)
                    {
                        values[i] = BinaryPrimitives.ReadInt32LittleEndian(Take(4));
                    }

                    return values;
                }

                case GgufValueType.Float32:
                {
                    float[] values = new float[count];
                    for (int i = 0; i < count; i++)
                    {
                        values[i] = BinaryPrimitives.ReadSingleLittleEndian(Take(4));
                    }

                    return values;
                }

                default:
                {
                    object[] values = new object[count];
                    for (int i = 0; i < count; i++)
                    {
                        values[i] = ReadScalar(type);
                    }

                    return values;
                }
            }
        }

        private ReadOnlySpan<byte> Take(int count)
        {
            ReadOnlySpan<byte> slice = _data.Slice(_position, count);
            _position += count;
            return slice;
        }
    }
}
