using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Models.Paddle;

/// <summary>
/// Pooled storage for the intermediate tensors of one graph run.
/// </summary>
/// <remarks>
/// <para>
/// A layout detection allocated 4.5 GiB. Not held — the interpreter already drops a value at its
/// last use, so the live set is bounded — but churned, once per page, and the graph's shapes are
/// identical on every page it ever runs. Every one of those buffers can come from a pool.
/// </para>
/// <para>
/// What makes that safe is knowing when a buffer is really dead, and a tensor's array is not its
/// own: <c>reshape</c>, <c>cast</c> between same-width dtypes and <c>share_data_</c> all return a
/// tensor over an existing array. Returning a buffer at one alias's last use would hand a live
/// array to the next renter. So the arena counts, per array, how many of the interpreter's value
/// slots reference it, and returns the array when that reaches zero. The interpreter is the only
/// place a run's tensors live, and every operator's results land in slots it names, so the count
/// is exact rather than conservative.
/// </para>
/// <para>
/// Arrays reachable from the run's fetches are excluded when the arena is disposed: those are the
/// caller's now.
/// </para>
/// <para>
/// The arena is thread-static and set only around <see cref="PirInterpreter.Run"/>. A tensor
/// allocated on some other thread simply misses the pool rather than corrupting it, which is the
/// right way round for a kernel that decides to spread its own work.
/// </para>
/// </remarks>
internal sealed class TensorArena : IDisposable
{
    [ThreadStatic]
    private static TensorArena? _current;

    private readonly Dictionary<object, int> _slots = new(ReferenceEqualityComparer.Instance);
    private readonly HashSet<object> _rented = new(ReferenceEqualityComparer.Instance);
    private readonly TensorArena? _previous;

    private TensorArena()
    {
        _previous = _current;
        _current = this;
    }

    /// <summary>The arena serving allocations on this thread, if any.</summary>
    internal static TensorArena? Current => _current;

    /// <summary>
    /// Whether a rented buffer is filled with a value no operator can mistake for a plausible
    /// intermediate, so an operator that reads its own output before writing it fails loudly.
    /// </summary>
    /// <remarks>
    /// A fresh allocation this size arrives zeroed, so before pooling such an operator would have
    /// been quietly correct. <c>PADDLEOCR_SHARP_POISON_ARENA=1</c> turns this on; the graph's
    /// output is unchanged with it on, which is what says no operator does that.
    /// </remarks>
    private static readonly bool Poison =
        Environment.GetEnvironmentVariable("PADDLEOCR_SHARP_POISON_ARENA") == "1";

    /// <summary>Installs a new arena on this thread until it is disposed.</summary>
    internal static TensorArena Begin() => new();

    /// <summary>Rents float storage for at least <paramref name="count"/> elements.</summary>
    internal float[] RentFloats(int count)
    {
        float[] array = TensorPool.RentArray(count);
        if (array.Length != 0)
        {
            _rented.Add(array);
            if (Poison)
            {
                array.AsSpan(0, count).Fill(float.NaN);
            }
        }

        return array;
    }

    /// <summary>Rents long storage for at least <paramref name="count"/> elements.</summary>
    internal long[] RentLongs(int count)
    {
        long[] array = TensorPool.RentLongs(count);
        if (array.Length != 0)
        {
            _rented.Add(array);
            if (Poison)
            {
                array.AsSpan(0, count).Fill(long.MinValue);
            }
        }

        return array;
    }

    /// <summary>Records that one more value slot references <paramref name="value"/>'s storage.</summary>
    /// <param name="value">A tensor, a tensor list, or <see langword="null"/>.</param>
    internal void Retain(object? value)
    {
        switch (value)
        {
            case PaddleTensor tensor:
                Adjust(Storage(tensor), 1);
                break;
            case PaddleTensor[] list:
                foreach (PaddleTensor element in list)
                {
                    Adjust(Storage(element), 1);
                }

                break;
        }
    }

    /// <summary>
    /// Records that one fewer value slot references <paramref name="value"/>'s storage, returning
    /// it to the pool if that was the last.
    /// </summary>
    /// <param name="value">A tensor, a tensor list, or <see langword="null"/>.</param>
    internal void Release(object? value)
    {
        switch (value)
        {
            case PaddleTensor tensor:
                Adjust(Storage(tensor), -1);
                break;
            case PaddleTensor[] list:
                foreach (PaddleTensor element in list)
                {
                    Adjust(Storage(element), -1);
                }

                break;
        }
    }

    /// <summary>
    /// Returns every buffer still rented, except those reachable from <paramref name="keep"/>.
    /// </summary>
    /// <param name="keep">Tensors the caller owns after the run — the graph's fetches.</param>
    internal void ReleaseAllExcept(IEnumerable<PaddleTensor> keep)
    {
        foreach (PaddleTensor tensor in keep)
        {
            if (Storage(tensor) is { } storage)
            {
                _rented.Remove(storage);
            }
        }

        foreach (object array in _rented)
        {
            ReturnToPool(array);
        }

        _rented.Clear();
        _slots.Clear();
    }

    /// <inheritdoc />
    public void Dispose() => _current = _previous;

    private static object? Storage(PaddleTensor tensor) => (object?)tensor.Floats ?? tensor.Ints;

    private void Adjust(object? storage, int delta)
    {
        // Parameters and caller-supplied inputs are not the arena's to account for, and neither is
        // an empty tensor's zero-length array.
        if (storage is null || !_rented.Contains(storage))
        {
            return;
        }

        ref int count = ref System.Runtime.InteropServices.CollectionsMarshal
            .GetValueRefOrAddDefault(_slots, storage, out _);

        count += delta;
        if (count > 0)
        {
            return;
        }

        _slots.Remove(storage);
        _rented.Remove(storage);
        ReturnToPool(storage);
    }

    private static void ReturnToPool(object array)
    {
        switch (array)
        {
            case float[] floats:
                TensorPool.Return(floats);
                break;
            case long[] longs:
                TensorPool.ReturnLongs(longs);
                break;
        }
    }
}
