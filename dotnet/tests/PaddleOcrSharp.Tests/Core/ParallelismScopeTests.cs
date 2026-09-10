using System.Collections.Concurrent;
using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Tests.Core;

/// <summary>Tests for the ambient degree of parallelism the kernels use.</summary>
public class ParallelismScopeTests
{
    /// <summary>With nothing in scope the kernels get the process default.</summary>
    [Fact]
    public void WithoutAScopeTheDefaultApplies() =>
        Assert.Same(Parallelism.Default, Parallelism.Options);

    /// <summary>A scope applies until it is disposed, and then stops applying.</summary>
    [Fact]
    public void AScopeAppliesAndThenUnwinds()
    {
        var mine = new ParallelOptions { MaxDegreeOfParallelism = 2 };

        using (Parallelism.Use(mine))
        {
            Assert.Same(mine, Parallelism.Options);

            var inner = new ParallelOptions { MaxDegreeOfParallelism = 3 };
            using (Parallelism.Use(inner))
            {
                Assert.Same(inner, Parallelism.Options);
            }

            Assert.Same(mine, Parallelism.Options);
        }

        Assert.Same(Parallelism.Default, Parallelism.Options);
    }

    /// <summary>Passing no options leaves whatever is in scope alone.</summary>
    /// <remarks>
    /// This is what lets the pipeline hand its unset configuration value straight through instead
    /// of branching on it.
    /// </remarks>
    [Fact]
    public void NullLeavesTheScopeUntouched()
    {
        var mine = new ParallelOptions { MaxDegreeOfParallelism = 2 };

        using (Parallelism.Use(mine))
        using (Parallelism.Use(null))
        {
            Assert.Same(mine, Parallelism.Options);
        }
    }

    /// <summary>
    /// A scope reaches a kernel running on a worker thread of the caller's own
    /// <c>Parallel.For</c>.
    /// </summary>
    /// <remarks>
    /// This is the case that rules out a thread-static: with <c>BlockConcurrency</c> above one the
    /// pipeline recognises blocks on pool threads, and everything a block does — its tower, its
    /// decoder, every GEMM inside them — runs there. An <see cref="AsyncLocal{T}"/> flows into
    /// those bodies with the execution context; a thread-static would not.
    /// </remarks>
    [Fact]
    public void AScopeReachesKernelsOnPoolThreads()
    {
        var mine = new ParallelOptions { MaxDegreeOfParallelism = 2 };
        var seen = new ConcurrentBag<ParallelOptions>();

        using (Parallelism.Use(mine))
        {
            Parallel.For(0, 32, _ => seen.Add(Parallelism.Options));
        }

        Assert.NotEmpty(seen);
        Assert.All(seen, options => Assert.Same(mine, options));
    }
}
