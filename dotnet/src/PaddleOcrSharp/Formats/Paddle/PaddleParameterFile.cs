using System.Buffers.Binary;
using System.IO.MemoryMappedFiles;

namespace PaddleOcrSharp.Formats.Paddle;

/// <summary>One tensor inside a Paddle combined parameter file.</summary>
/// <param name="Dtype">Element type.</param>
/// <param name="Shape">Dimensions.</param>
/// <param name="Bytes">Raw little-endian payload.</param>
public readonly record struct PaddleParameter(PaddleDType Dtype, int[] Shape, ReadOnlyMemory<byte> Bytes)
{
    /// <summary>Total number of elements.</summary>
    public int Count
    {
        get
        {
            int count = 1;
            foreach (int dimension in Shape)
            {
                count *= dimension;
            }

            return count;
        }
    }
}

/// <summary>
/// Reader for Paddle's combined parameter file (<c>inference.pdiparams</c>).
/// </summary>
/// <remarks>
/// <para>
/// The file is a concatenation of serialised dense tensors, each laid out as:
/// </para>
/// <code>
/// uint32  version
/// uint64  lodLevel, then one (uint64 count + count × uint64) run per level
/// uint32  tensorDescVersion
/// int32   tensorDescLength
/// bytes   protobuf VarType.TensorDesc { data_type = 1, dims = 2 }
/// bytes   raw element data
/// </code>
/// <para>
/// The tensors carry no names. Paddle's PIR exporter writes them in the order produced by
/// sorting the parameter names as byte strings, which is what
/// <see cref="Read(string, IReadOnlyList{string})"/> relies on — the caller supplies the names
/// from the matching <c>inference.json</c> and the two are zipped back together. The shapes in
/// the file are checked against the program's declared shapes, so a wrong pairing is detected
/// rather than silently mis-loaded.
/// </para>
/// </remarks>
public sealed class PaddleParameterFile : IDisposable
{
    private readonly MemoryMappedFile _file;
    private readonly MemoryMappedViewAccessor _view;
    private readonly Dictionary<string, PaddleParameter> _parameters;
    private unsafe byte* _base;

    private PaddleParameterFile(
        MemoryMappedFile file,
        MemoryMappedViewAccessor view,
        Dictionary<string, PaddleParameter> parameters)
    {
        _file = file;
        _view = view;
        _parameters = parameters;
    }

    /// <summary>Number of parameters.</summary>
    public int Count => _parameters.Count;

    /// <summary>Parameter names.</summary>
    public IReadOnlyCollection<string> Names => _parameters.Keys;

    /// <summary>
    /// Reads <paramref name="path"/>, pairing the stored tensors with <paramref name="names"/>.
    /// </summary>
    /// <param name="path">Path to the <c>.pdiparams</c> file.</param>
    /// <param name="names">Parameter names taken from the program; order is irrelevant.</param>
    public static unsafe PaddleParameterFile Read(string path, IReadOnlyList<string> names)
    {
        MemoryMappedFile file = MemoryMappedFile.CreateFromFile(
            path, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);
        MemoryMappedViewAccessor view = file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);

        byte* basePointer = null;
        view.SafeMemoryMappedViewHandle.AcquirePointer(ref basePointer);

        try
        {
            long length = new FileInfo(path).Length;
            var span = new ReadOnlySpan<byte>(basePointer, checked((int)length));

            string[] sorted = [.. names];
            Array.Sort(sorted, CompareOrdinalBytes);

            var parameters = new Dictionary<string, PaddleParameter>(StringComparer.Ordinal);
            int offset = 0;
            int index = 0;

            while (offset < span.Length)
            {
                if (index >= sorted.Length)
                {
                    throw new InvalidDataException(
                        $"'{path}' holds more tensors than the {sorted.Length} parameters the program declares.");
                }

                offset += 4; // tensor version
                ulong lodLevel = BinaryPrimitives.ReadUInt64LittleEndian(span[offset..]);
                offset += 8;
                for (ulong level = 0; level < lodLevel; level++)
                {
                    ulong count = BinaryPrimitives.ReadUInt64LittleEndian(span[offset..]);
                    offset += 8 + (int)(count * 8);
                }

                offset += 4; // tensor-desc version
                int descriptorLength = BinaryPrimitives.ReadInt32LittleEndian(span[offset..]);
                offset += 4;

                (PaddleDType dtype, int[] shape) = ReadDescriptor(span.Slice(offset, descriptorLength));
                offset += descriptorLength;

                int elements = 1;
                foreach (int dimension in shape)
                {
                    elements *= dimension;
                }

                int byteCount = elements * dtype.ByteSize();
                var manager = new UnmanagedMemoryManager<byte>(basePointer + offset, byteCount);
                parameters[sorted[index]] = new PaddleParameter(dtype, shape, manager.Memory);

                offset += byteCount;
                index++;
            }

            if (index != sorted.Length)
            {
                throw new InvalidDataException(
                    $"'{path}' holds {index} tensors but the program declares {sorted.Length} parameters.");
            }

            return new PaddleParameterFile(file, view, parameters) { _base = basePointer };
        }
        catch
        {
            view.SafeMemoryMappedViewHandle.ReleasePointer();
            view.Dispose();
            file.Dispose();
            throw;
        }
    }

    /// <summary>Gets a parameter by name.</summary>
    public PaddleParameter this[string name] =>
        _parameters.TryGetValue(name, out PaddleParameter parameter)
            ? parameter
            : throw new KeyNotFoundException($"Parameter file has no tensor named '{name}'.");

    /// <summary>Tries to get a parameter by name.</summary>
    public bool TryGet(string name, out PaddleParameter parameter) =>
        _parameters.TryGetValue(name, out parameter);

    /// <summary>
    /// Byte-ordinal comparison, matching the <c>std::string</c> ordering Paddle sorts with.
    /// </summary>
    private static int CompareOrdinalBytes(string left, string right)
    {
        int length = Math.Min(left.Length, right.Length);
        for (int i = 0; i < length; i++)
        {
            if (left[i] != right[i])
            {
                return left[i] < right[i] ? -1 : 1;
            }
        }

        return left.Length.CompareTo(right.Length);
    }

    /// <summary>
    /// Reads the two fields of <c>VarType.TensorDesc</c> we need out of its protobuf encoding.
    /// </summary>
    private static (PaddleDType Dtype, int[] Shape) ReadDescriptor(ReadOnlySpan<byte> descriptor)
    {
        int dataType = -1;
        var dims = new List<int>();
        int i = 0;

        while (i < descriptor.Length)
        {
            byte key = descriptor[i++];
            int field = key >> 3;
            int wire = key & 7;

            switch (wire)
            {
                case 0:
                {
                    long value = ReadVarint(descriptor, ref i);
                    if (field == 1)
                    {
                        dataType = (int)value;
                    }
                    else if (field == 2)
                    {
                        dims.Add((int)value);
                    }

                    break;
                }

                case 2:
                {
                    int length = (int)ReadVarint(descriptor, ref i);
                    ReadOnlySpan<byte> payload = descriptor.Slice(i, length);
                    i += length;

                    if (field == 2)
                    {
                        int j = 0;
                        while (j < payload.Length)
                        {
                            dims.Add((int)ReadVarint(payload, ref j));
                        }
                    }

                    break;
                }

                case 5:
                    i += 4;
                    break;

                case 1:
                    i += 8;
                    break;

                default:
                    throw new InvalidDataException($"Unexpected protobuf wire type {wire} in a tensor descriptor.");
            }
        }

        if (dataType < 0)
        {
            throw new InvalidDataException("Tensor descriptor has no data type.");
        }

        return (PaddleDTypeExtensions.FromVarTypeCode(dataType), [.. dims]);
    }

    private static long ReadVarint(ReadOnlySpan<byte> buffer, ref int index)
    {
        long value = 0;
        int shift = 0;
        while (true)
        {
            byte b = buffer[index++];
            value |= (long)(b & 0x7F) << shift;
            if ((b & 0x80) == 0)
            {
                return value;
            }

            shift += 7;
        }
    }

    /// <inheritdoc />
    public unsafe void Dispose()
    {
        _parameters.Clear();
        if (_base is not null)
        {
            _view.SafeMemoryMappedViewHandle.ReleasePointer();
            _base = null;
        }

        _view.Dispose();
        _file.Dispose();
    }
}
