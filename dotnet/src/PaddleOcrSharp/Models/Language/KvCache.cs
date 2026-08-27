using System.Buffers;

namespace PaddleOcrSharp.Models.Language;

/// <summary>
/// Contiguous key/value cache for the decoder, one pair of buffers per layer.
/// </summary>
/// <remarks>
/// Keys and values are stored as <c>[position, kvHeads · headDim]</c> so appending a token is a
/// single copy and one head's history is a strided walk of contiguous 128-float rows. Capacity
/// grows geometrically, so a long generation does not re-allocate on every step.
/// </remarks>
public sealed class KvCache : IDisposable
{
    private readonly int _layers;
    private readonly int _width;
    private float[][] _keys;
    private float[][] _values;
    private int _capacity;

    /// <summary>Creates a cache sized for an initial <paramref name="capacity"/> positions.</summary>
    /// <param name="layers">Number of decoder layers.</param>
    /// <param name="keyValueWidth">Width of the packed key (or value) vector per position.</param>
    /// <param name="capacity">Initial capacity in positions.</param>
    public KvCache(int layers, int keyValueWidth, int capacity = 512)
    {
        _layers = layers;
        _width = keyValueWidth;
        _capacity = Math.Max(1, capacity);
        _keys = new float[layers][];
        _values = new float[layers][];

        for (int i = 0; i < layers; i++)
        {
            _keys[i] = ArrayPool<float>.Shared.Rent(_capacity * _width);
            _values[i] = ArrayPool<float>.Shared.Rent(_capacity * _width);
        }
    }

    /// <summary>Number of positions currently held.</summary>
    public int Length { get; private set; }

    /// <summary>Width of one cached key or value vector.</summary>
    public int Width => _width;

    /// <summary>Discards all cached positions without releasing the buffers.</summary>
    public void Clear() => Length = 0;

    /// <summary>Ensures room for <paramref name="positions"/> more entries.</summary>
    public void Reserve(int positions)
    {
        int required = Length + positions;
        if (required <= _capacity)
        {
            return;
        }

        int capacity = _capacity;
        while (capacity < required)
        {
            capacity *= 2;
        }

        for (int i = 0; i < _layers; i++)
        {
            float[] keys = ArrayPool<float>.Shared.Rent(capacity * _width);
            float[] values = ArrayPool<float>.Shared.Rent(capacity * _width);
            _keys[i].AsSpan(0, Length * _width).CopyTo(keys);
            _values[i].AsSpan(0, Length * _width).CopyTo(values);
            ArrayPool<float>.Shared.Return(_keys[i]);
            ArrayPool<float>.Shared.Return(_values[i]);
            _keys[i] = keys;
            _values[i] = values;
        }

        _capacity = capacity;
    }

    /// <summary>
    /// The destination slice for <paramref name="count"/> new key vectors of <paramref name="layer"/>.
    /// </summary>
    public Span<float> KeySlot(int layer, int start, int count) =>
        _keys[layer].AsSpan(start * _width, count * _width);

    /// <summary>
    /// The destination slice for <paramref name="count"/> new value vectors of <paramref name="layer"/>.
    /// </summary>
    public Span<float> ValueSlot(int layer, int start, int count) =>
        _values[layer].AsSpan(start * _width, count * _width);

    /// <summary>All cached keys of <paramref name="layer"/>, as <c>[Length, Width]</c>.</summary>
    public ReadOnlyMemory<float> Keys(int layer) => _keys[layer].AsMemory(0, Length * _width);

    /// <summary>All cached values of <paramref name="layer"/>, as <c>[Length, Width]</c>.</summary>
    public ReadOnlyMemory<float> Values(int layer) => _values[layer].AsMemory(0, Length * _width);

    /// <summary>
    /// The first <paramref name="positions"/> key vectors of <paramref name="layer"/>, including
    /// slots written by the current forward pass but not yet published by <see cref="Advance"/>.
    /// </summary>
    public ReadOnlyMemory<float> KeyWindow(int layer, int positions) =>
        _keys[layer].AsMemory(0, positions * _width);

    /// <summary>
    /// The first <paramref name="positions"/> value vectors of <paramref name="layer"/>, including
    /// slots written by the current forward pass but not yet published by <see cref="Advance"/>.
    /// </summary>
    public ReadOnlyMemory<float> ValueWindow(int layer, int positions) =>
        _values[layer].AsMemory(0, positions * _width);

    /// <summary>Marks <paramref name="count"/> positions as written.</summary>
    public void Advance(int count) => Length += count;

    /// <inheritdoc />
    public void Dispose()
    {
        for (int i = 0; i < _layers; i++)
        {
            if (_keys[i] is { Length: > 0 })
            {
                ArrayPool<float>.Shared.Return(_keys[i]);
            }

            if (_values[i] is { Length: > 0 })
            {
                ArrayPool<float>.Shared.Return(_values[i]);
            }
        }

        _keys = [];
        _values = [];
        Length = 0;
    }
}
