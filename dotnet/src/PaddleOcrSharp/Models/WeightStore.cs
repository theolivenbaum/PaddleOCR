using System.Collections.Concurrent;
using PaddleOcrSharp.Core;
using PaddleOcrSharp.Formats;
using PaddleOcrSharp.Formats.Gguf;
using PaddleOcrSharp.Quantization;

namespace PaddleOcrSharp.Models;

/// <summary>
/// Named access to a checkpoint's parameters, from either container the port reads.
/// </summary>
/// <remarks>
/// <para>
/// Matrices stay in their stored form — bfloat16, float32 or packed ternary blocks — and are read
/// straight from the memory map. Vectors (norm scales, biases) are small, are touched on every row
/// of every layer, and are therefore widened to float32 once and cached.
/// </para>
/// <para>
/// A <c>.safetensors</c> file is the unquantized checkpoint as published; a <c>.gguf</c> file is
/// what <c>PaddleOcrSharp.Quantize</c> writes, which may hold quantized tensors, unquantized ones,
/// or both. Everything above this type — the vision tower, the decoder, the pipeline — sees the
/// same <see cref="WeightMatrix"/> either way and does not know which it got.
/// </para>
/// </remarks>
public sealed class WeightStore : IDisposable
{
    private readonly SafetensorsFile? _safetensors;
    private readonly GgufFile? _gguf;
    private readonly QuantizationMetadata? _quantization;
    private readonly ConcurrentDictionary<string, float[]> _vectorCache = new(StringComparer.Ordinal);
    private readonly bool _ownsFile;

    private WeightStore(SafetensorsFile file, bool ownsFile)
    {
        _safetensors = file;
        _ownsFile = ownsFile;
    }

    private WeightStore(GgufFile file, bool ownsFile)
    {
        _gguf = file;
        _quantization = QuantizationMetadata.Read(file);
        _ownsFile = ownsFile;
    }

    /// <summary>
    /// Opens the checkpoint at <paramref name="path"/>, choosing the reader by extension.
    /// </summary>
    /// <param name="path">A <c>.safetensors</c> or <c>.gguf</c> file.</param>
    public static WeightStore Open(string path) =>
        Path.GetExtension(path).Equals(".gguf", StringComparison.OrdinalIgnoreCase)
            ? new WeightStore(GgufFile.Open(path), ownsFile: true)
            : new WeightStore(SafetensorsFile.Open(path), ownsFile: true);

    /// <summary>Wraps an already-open safetensors file without taking ownership of it.</summary>
    public static WeightStore Wrap(SafetensorsFile file) => new(file, ownsFile: false);

    /// <summary>Wraps an already-open GGUF file without taking ownership of it.</summary>
    public static WeightStore Wrap(GgufFile file) => new(file, ownsFile: false);

    /// <summary>Every parameter name in the checkpoint.</summary>
    public IReadOnlyCollection<string> Names => _safetensors?.Names ?? _gguf!.Names;

    /// <summary>The quantization header, when the checkpoint came from a GGUF file.</summary>
    public QuantizationMetadata? Quantization => _quantization;

    /// <summary>Whether any tensor in the checkpoint is stored as quantized blocks.</summary>
    public bool HasQuantizedWeights =>
        _gguf is not null && _gguf.Names.Any(name => _gguf[name].Type.IsQuantized());

    /// <summary>Whether <paramref name="name"/> exists.</summary>
    public bool Contains(string name) => _safetensors?.Contains(name) ?? _gguf!.Contains(name);

    /// <summary>Gets a 2-D parameter as a weight matrix in <c>nn.Linear</c> layout.</summary>
    public WeightMatrix Matrix(string name)
    {
        if (_safetensors is not null)
        {
            WeightTensor stored = _safetensors[name];
            if (stored.Shape.Length != 2)
            {
                throw new InvalidOperationException($"'{name}' has rank {stored.Shape.Length}; expected a matrix.");
            }

            return WeightMatrix.Create(stored.Bytes, stored.Dtype, stored.Shape[0], stored.Shape[1], name);
        }

        GgufTensor tensor = _gguf![name];
        if (tensor.Ne.Length != 2)
        {
            throw new InvalidOperationException($"'{name}' has rank {tensor.Ne.Length}; expected a matrix.");
        }

        return AsMatrix(tensor, checked((int)tensor.Ne[1]), checked((int)tensor.Ne[0]));
    }

    /// <summary>
    /// Gets a parameter of any rank as a <c>[rows, cols]</c> matrix, folding trailing dimensions
    /// into the columns. Used for convolution kernels stored as <c>[out, in, kh, kw]</c>.
    /// </summary>
    public WeightMatrix MatrixFlattened(string name)
    {
        if (_safetensors is not null)
        {
            WeightTensor tensor = _safetensors[name];
            int rows = tensor.Shape[0];
            int cols = tensor.ElementCount / rows;
            return WeightMatrix.Create(tensor.Bytes, tensor.Dtype, rows, cols, name);
        }

        GgufTensor stored = _gguf![name];

        // GGUF lists dimensions fastest-varying first, so the row count is the outermost, which is
        // the last of ne — and everything below it folds into the columns.
        int outer = checked((int)stored.Ne[^1]);
        int inner = checked((int)(stored.ElementCount / outer));
        return AsMatrix(stored, outer, inner);
    }

    /// <summary>Gets a parameter widened to float32 and cached.</summary>
    public float[] Vector(string name) => _vectorCache.GetOrAdd(name, static (key, self) =>
        self.ToFloats(key), this);

    /// <summary>Gets a parameter widened to float32, or an empty array if it is absent.</summary>
    public float[] OptionalVector(string name) => Contains(name) ? Vector(name) : [];

    /// <summary>Shape of <paramref name="name"/>, outermost dimension first.</summary>
    public int[] Shape(string name) => _safetensors?[name].Shape ?? _gguf![name].Shape;

    /// <summary>
    /// Raw safetensors tensor access, for callers that need the dtype or the bytes.
    /// </summary>
    /// <exception cref="NotSupportedException">The checkpoint is a GGUF file.</exception>
    public WeightTensor Tensor(string name) =>
        _safetensors is not null
            ? _safetensors[name]
            : throw new NotSupportedException(
                "Raw tensor access is safetensors-only; read a GGUF checkpoint through Matrix or Vector.");

    /// <summary>The GGUF tensor behind <paramref name="name"/>, when the checkpoint is one.</summary>
    /// <exception cref="NotSupportedException">The checkpoint is a safetensors file.</exception>
    public GgufTensor GgufTensor(string name) =>
        _gguf is not null
            ? _gguf[name]
            : throw new NotSupportedException("The checkpoint is not a GGUF file.");

    /// <inheritdoc />
    public void Dispose()
    {
        _vectorCache.Clear();
        if (_ownsFile)
        {
            _safetensors?.Dispose();
            _gguf?.Dispose();
        }
    }

    private WeightMatrix AsMatrix(GgufTensor tensor, int rows, int cols) =>
        tensor.Type.IsQuantized()
            ? WeightMatrix.CreateQuantized(
                tensor.Bytes, tensor.Type, rows, cols, _quantization?.For(tensor.Name), tensor.Name)
            : WeightMatrix.Create(tensor.Bytes, ToDType(tensor.Type), rows, cols, tensor.Name);

    private float[] ToFloats(string name)
    {
        if (_safetensors is not null)
        {
            return _safetensors[name].ToFloats();
        }

        Formats.Gguf.GgufTensor tensor = _gguf![name];
        float[] result = new float[tensor.ElementCount];

        if (tensor.Type.IsQuantized())
        {
            BlockCodec.Decode(tensor.Type, tensor.Bytes.Span, result);
            return result;
        }

        FloatConversion.ToFloat(tensor.Bytes.Span, ToDType(tensor.Type), result);
        return result;
    }

    private static DType ToDType(GgmlType type) => type switch
    {
        GgmlType.F32 => DType.Float32,
        GgmlType.F16 => DType.Float16,
        GgmlType.BF16 => DType.BFloat16,
        GgmlType.I32 => DType.Int32,
        _ => throw new NotSupportedException($"{type.TypeName()} tensors have no unpacked dtype."),
    };
}
