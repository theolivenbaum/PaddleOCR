using System.Buffers;

namespace PaddleOcrSharp.Core;

/// <summary>
/// A scoped rental of a pooled <see cref="float"/> buffer.
/// </summary>
/// <remarks>
/// The rented array is normally larger than <see cref="Length"/>; always go through
/// <see cref="Span"/> so kernels never see the slack at the end.
/// </remarks>
public readonly struct PooledBuffer : IDisposable
{
    private readonly float[] _array;

    internal PooledBuffer(float[] array, int length)
    {
        _array = array;
        Length = length;
    }

    /// <summary>Number of usable elements.</summary>
    public int Length { get; }

    /// <summary>The usable portion of the rented array.</summary>
    public Span<float> Span => _array.AsSpan(0, Length);

    /// <summary>The usable portion of the rented array as a <see cref="Memory{T}"/>.</summary>
    public Memory<float> Memory => _array.AsMemory(0, Length);

    /// <summary>Returns the buffer to the pool.</summary>
    public void Dispose()
    {
        if (_array is not null)
        {
            TensorPool.Return(_array);
        }
    }
}

/// <summary>
/// Central <see cref="ArrayPool{T}"/> wrapper for the transient activation buffers the model
/// allocates per layer. Keeping these pooled is what lets a decode step run allocation-free.
/// </summary>
public static class TensorPool
{
    // The default shared pool caps buckets at 1 MiB (2^20 bytes). Activations here reach tens of
    // megabytes (e.g. a 5000x4304 vision MLP intermediate), so we own a pool sized for them.
    private const int MaxArrayLength = 1 << 28; // 256M floats = 1 GiB
    private const int MaxArraysPerBucket = 16;

    private static readonly ArrayPool<float> Pool =
        ArrayPool<float>.Create(MaxArrayLength, MaxArraysPerBucket);

    private static readonly ArrayPool<int> IntPool =
        ArrayPool<int>.Create(1 << 24, MaxArraysPerBucket);

    /// <summary>Rents a buffer of at least <paramref name="length"/> floats.</summary>
    /// <param name="length">Number of usable elements.</param>
    /// <param name="clear">When <see langword="true"/>, zeroes the usable portion before returning.</param>
    public static PooledBuffer Rent(int length, bool clear = false)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);

        float[] array = length == 0 ? [] : Pool.Rent(length);
        if (clear && length > 0)
        {
            array.AsSpan(0, length).Clear();
        }

        return new PooledBuffer(array, length);
    }

    /// <summary>Rents a raw float array of at least <paramref name="length"/> elements.</summary>
    internal static float[] RentArray(int length) => length == 0 ? [] : Pool.Rent(length);

    /// <summary>Rents a buffer of at least <paramref name="length"/> ints.</summary>
    public static int[] RentInts(int length) => length == 0 ? [] : IntPool.Rent(length);

    /// <summary>Returns an int buffer previously obtained from <see cref="RentInts"/>.</summary>
    public static void ReturnInts(int[] array)
    {
        if (array.Length != 0)
        {
            IntPool.Return(array);
        }
    }

    /// <summary>Returns an array previously obtained from this pool.</summary>
    internal static void Return(float[] array)
    {
        if (array.Length != 0)
        {
            Pool.Return(array);
        }
    }
}
