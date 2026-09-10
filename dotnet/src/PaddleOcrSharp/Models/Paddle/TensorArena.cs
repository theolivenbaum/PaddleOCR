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

    /// <summary>
    /// Whether the arena writes a one-line summary of what it served, and from where, when a run
    /// ends. <c>PADDLEOCR_SHARP_ARENA_STATS=1</c> turns it on.
    /// </summary>
    /// <remarks>
    /// A rent that the pool cannot serve allocates, and from outside there is no way to tell one
    /// from the other — the operator profile just shows bytes against whichever operator asked.
    /// Measuring the allocation across each rent separates them.
    /// </remarks>
    private static readonly bool Stats =
        Environment.GetEnvironmentVariable("PADDLEOCR_SHARP_ARENA_STATS") == "1";

    private int _rents;
    private int _misses;
    private long _rentedBytes;
    private long _missedBytes;
    private long _returnedEarly;
    private long _returnedAtEnd;
    private long _keptBytes;

    /// <summary>Installs a new arena on this thread until it is disposed.</summary>
    internal static TensorArena Begin() => new();

    /// <summary>Rents float storage for at least <paramref name="count"/> elements.</summary>
    internal float[] RentFloats(int count)
    {
        long before = Stats ? GC.GetAllocatedBytesForCurrentThread() : 0;
        float[] array = TensorPool.RentArray(count);
        if (array.Length != 0)
        {
            _rented.Add(array);
            Record(before, (long)array.Length * sizeof(float));
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
        long before = Stats ? GC.GetAllocatedBytesForCurrentThread() : 0;
        long[] array = TensorPool.RentLongs(count);
        if (array.Length != 0)
        {
            _rented.Add(array);
            Record(before, (long)array.Length * sizeof(long));
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
    /// Returns every buffer still rented, except those reachable from <paramref name="keep"/>,
    /// and hands those back so the caller can return them when it is finished.
    /// </summary>
    /// <param name="keep">Tensors the caller owns after the run — the graph's fetches.</param>
    /// <returns>The storage behind <paramref name="keep"/>, to pass to <see cref="Return"/>.</returns>
    internal object[] ReleaseAllExcept(IEnumerable<PaddleTensor> keep)
    {
        var kept = new List<object>();

        foreach (PaddleTensor tensor in keep)
        {
            // Two fetches can share one array; `Remove` reporting false is what keeps it out of
            // the list twice, which would return it to the pool twice.
            if (Storage(tensor) is { } storage && _rented.Remove(storage))
            {
                _keptBytes += Size(storage);
                kept.Add(storage);
            }
        }

        foreach (object array in _rented)
        {
            _returnedAtEnd += Size(array);
            ReturnToPool(array);
        }

        _rented.Clear();
        _slots.Clear();

        if (Stats)
        {
            Console.Error.WriteLine(
                $"arena: {_rents} rents of {_rentedBytes >> 20} MiB, {_misses} missed the pool "
                + $"({_missedBytes >> 20} MiB allocated); returned {_returnedEarly >> 20} MiB at "
                + $"last use and {_returnedAtEnd >> 20} MiB at the end; {_keptBytes >> 20} MiB "
                + "kept as fetches");
        }

        return [.. kept];
    }

    /// <summary>Returns buffers a previous run handed to its caller.</summary>
    /// <param name="arrays">Storage from <see cref="ReleaseAllExcept"/>.</param>
    /// <remarks>
    /// Without this the fetches leave the pool one bucket short on every run, and the next run
    /// allocates that shortfall again — 128 MiB a detection for the layout graph's mask fetch.
    /// </remarks>
    internal static void Return(object[] arrays)
    {
        foreach (object array in arrays)
        {
            ReturnToPool(array);
        }
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
        _returnedEarly += Size(storage);
        ReturnToPool(storage);
    }

    private void Record(long before, long bytes)
    {
        if (!Stats)
        {
            return;
        }

        _rents++;
        _rentedBytes += bytes;

        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        if (allocated <= 0)
        {
            return;
        }

        _misses++;
        _missedBytes += allocated;
    }

    private static long Size(object array) => array switch
    {
        float[] floats => (long)floats.Length * sizeof(float),
        long[] longs => (long)longs.Length * sizeof(long),
        _ => 0,
    };

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
