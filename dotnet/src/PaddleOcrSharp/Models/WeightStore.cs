using System.Collections.Concurrent;
using PaddleOcrSharp.Core;
using PaddleOcrSharp.Formats;

namespace PaddleOcrSharp.Models;

/// <summary>
/// Named access to a checkpoint's parameters.
/// </summary>
/// <remarks>
/// Matrices stay in their on-disk dtype and are read straight from the memory map. Vectors
/// (norm scales, biases) are small, are touched on every row of every layer, and are therefore
/// widened to float32 once and cached.
/// </remarks>
public sealed class WeightStore : IDisposable
{
    private readonly SafetensorsFile _file;
    private readonly ConcurrentDictionary<string, float[]> _vectorCache = new(StringComparer.Ordinal);
    private readonly bool _ownsFile;

    private WeightStore(SafetensorsFile file, bool ownsFile)
    {
        _file = file;
        _ownsFile = ownsFile;
    }

    /// <summary>Opens the safetensors file at <paramref name="path"/>.</summary>
    public static WeightStore Open(string path) => new(SafetensorsFile.Open(path), ownsFile: true);

    /// <summary>Wraps an already-open file without taking ownership of it.</summary>
    public static WeightStore Wrap(SafetensorsFile file) => new(file, ownsFile: false);

    /// <summary>Every parameter name in the checkpoint.</summary>
    public IReadOnlyCollection<string> Names => _file.Names;

    /// <summary>Whether <paramref name="name"/> exists.</summary>
    public bool Contains(string name) => _file.Contains(name);

    /// <summary>Gets a 2-D parameter as a weight matrix in <c>nn.Linear</c> layout.</summary>
    public WeightMatrix Matrix(string name) => _file[name].AsMatrix();

    /// <summary>
    /// Gets a parameter of any rank as a <c>[rows, cols]</c> matrix, folding trailing dimensions
    /// into the columns. Used for convolution kernels stored as <c>[out, in, kh, kw]</c>.
    /// </summary>
    public WeightMatrix MatrixFlattened(string name)
    {
        WeightTensor tensor = _file[name];
        int rows = tensor.Shape[0];
        int cols = tensor.ElementCount / rows;
        return tensor.AsMatrix(rows, cols);
    }

    /// <summary>Gets a parameter widened to float32 and cached.</summary>
    public float[] Vector(string name) => _vectorCache.GetOrAdd(name, static (key, self) =>
        self._file[key].ToFloats(), this);

    /// <summary>Gets a parameter widened to float32, or an empty array if it is absent.</summary>
    public float[] OptionalVector(string name) => _file.Contains(name) ? Vector(name) : [];

    /// <summary>Shape of <paramref name="name"/>.</summary>
    public int[] Shape(string name) => _file[name].Shape;

    /// <summary>Raw tensor access, for callers that need the dtype or the bytes.</summary>
    public WeightTensor Tensor(string name) => _file[name];

    /// <inheritdoc />
    public void Dispose()
    {
        _vectorCache.Clear();
        if (_ownsFile)
        {
            _file.Dispose();
        }
    }
}
