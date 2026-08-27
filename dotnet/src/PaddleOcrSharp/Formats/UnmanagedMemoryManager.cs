using System.Buffers;

namespace PaddleOcrSharp.Formats;

/// <summary>
/// Exposes a block of unmanaged memory (a memory-mapped weight file) as <see cref="Memory{T}"/>
/// so weights can be used without ever being copied onto the managed heap.
/// </summary>
/// <typeparam name="T">Element type.</typeparam>
internal sealed unsafe class UnmanagedMemoryManager<T> : MemoryManager<T>
    where T : unmanaged
{
    private readonly T* _pointer;
    private readonly int _length;

    public UnmanagedMemoryManager(T* pointer, int length)
    {
        _pointer = pointer;
        _length = length;
    }

    public override Span<T> GetSpan() => new(_pointer, _length);

    public override MemoryHandle Pin(int elementIndex = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(elementIndex);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(elementIndex, _length);
        return new MemoryHandle(_pointer + elementIndex);
    }

    public override void Unpin()
    {
        // The backing memory-mapped view is pinned for its whole lifetime.
    }

    protected override void Dispose(bool disposing)
    {
        // Ownership of the mapping stays with the file that created this manager.
    }
}
