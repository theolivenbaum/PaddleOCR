using PaddleOcrSharp.Models.Language;

namespace PaddleOcrSharp.Tests.Models;

/// <summary>Tests for the decoder's key/value cache.</summary>
public class KvCacheTests
{
    // The decoder's own geometry: 18 layers, 2 key/value heads of 128.
    private const int Layers = 18;
    private const int Width = 2 * 128;

    /// <summary>
    /// A cache grown to hold a long generation must come back out of a pool on the next block.
    /// </summary>
    /// <remarks>
    /// <para>
    /// One growth at the decoder's geometry is 576 MiB of key and value buffers, so whether they
    /// are pooled decides whether a page of long-generating blocks churns half a gigabyte per
    /// block or nothing. <see cref="System.Buffers.ArrayPool{T}.Shared"/> handles it: measured, it
    /// round-trips buffers of this size — and up to a gigabyte — with zero allocation.
    /// </para>
    /// <para>
    /// The reason to pin it is that the obvious-looking alternative does not. An
    /// <c>ArrayPool.Create(_, maxArraysPerBucket)</c> keeps only that many buffers of a size and
    /// drops the rest, and this cache holds thirty-six of one size — two per layer — so at the
    /// usual 16 it loses twenty every block. This test fails at 66 MiB a round against such a
    /// pool, which is how that was found.
    /// </para>
    /// </remarks>
    [Fact]
    public void GrowthPastTheSharedPoolsBucketCapIsStillPooled()
    {
        // Enough to force one growth to 16384, where each layer's buffer is four times the cap.
        const int Positions = 9000;
        const long Budget = 32L << 20;

        Grow(Positions);

        long first = Measure(Positions);
        long second = Measure(Positions);

        Assert.True(
            second < Budget,
            $"A repeat of the same growth allocated {second / (1L << 20)} MiB, so the buffers are "
            + $"not being pooled at this size. The first round allocated {first / (1L << 20)} MiB.");
    }

    private static long Measure(int positions)
    {
        long before = GC.GetAllocatedBytesForCurrentThread();
        Grow(positions);
        return GC.GetAllocatedBytesForCurrentThread() - before;
    }

    private static void Grow(int positions)
    {
        using var cache = new KvCache(Layers, Width);
        cache.Reserve(positions);
        cache.Advance(positions);
    }
}
