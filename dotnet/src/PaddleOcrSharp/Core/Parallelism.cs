namespace PaddleOcrSharp.Core;

/// <summary>
/// How many threads the kernels spread work across.
/// </summary>
/// <remarks>
/// <para>
/// <c>Parallel.For</c>'s default degree is unbounded, which for compute-bound work is the wrong
/// default twice over. The thread pool starts at <see cref="Environment.ProcessorCount"/> workers
/// and its hill-climbing heuristic injects more while a queue stays non-empty — a policy meant for
/// work that blocks. None of the work here blocks: every extra worker is another claimant on the
/// same caches and the same load ports, and the switches come out of the same budget.
/// </para>
/// <para>
/// Measured on the vision tower at a 4900-patch page, capping attention's head loop at the core
/// count: <b>attention 42.8 s → 36.6 s</b> and the whole tower 65.4 s → 58.3 s, with the untouched
/// matrix products inside their noise as the control. The instrumentation says why directly —
/// attention's parts accumulate their own thread's time, and that sum went from 6.8x the stage's
/// wall time to 3.89x on a four-core machine. Six or seven threads had been sharing four cores.
/// </para>
/// <para>
/// One worker per core is the right default and not always the right answer: a host that shares
/// the machine with its own work, or runs under a CPU quota that
/// <see cref="Environment.ProcessorCount"/> does not see, wants fewer.
/// <c>DocumentParserOptions.Parallelism</c> carries a
/// <see cref="ParallelOptions"/> for a parse, and <see cref="Use"/> does the same for any other
/// entry point. Setting <see cref="Default"/> changes it for the process.
/// </para>
/// </remarks>
public static class Parallelism
{
    private static readonly AsyncLocal<ParallelOptions?> Scoped = new();

    private static ParallelOptions _default = new()
    {
        MaxDegreeOfParallelism = Environment.ProcessorCount,
    };

    /// <summary>
    /// The options every kernel uses when nothing narrower is in scope. One worker per core.
    /// </summary>
    /// <exception cref="ArgumentNullException">The value is <see langword="null"/>.</exception>
    public static ParallelOptions Default
    {
        get => _default;
        set
        {
            ArgumentNullException.ThrowIfNull(value);
            _default = value;
        }
    }

    /// <summary>What the kernels should use here — the innermost scope, or <see cref="Default"/>.</summary>
    internal static ParallelOptions Options => Scoped.Value ?? _default;

    /// <summary>
    /// Makes <paramref name="options"/> the kernels' options until the returned scope is disposed.
    /// </summary>
    /// <param name="options">
    /// The options to use, or <see langword="null"/> to leave the current ones in place — which is
    /// what lets a caller pass an unset configuration value straight through.
    /// </param>
    /// <returns>A scope that restores the previous options.</returns>
    /// <remarks>
    /// <para>
    /// The scope is ambient rather than a parameter because the kernels that spread work are
    /// static and are reached through a dozen layers of shape algebra; threading an argument to
    /// each would put a parameter on <c>Gemm.Linear</c> for the benefit of its callers' callers.
    /// It is an <see cref="AsyncLocal{T}"/> rather than a thread-static so that it survives the
    /// pipeline's own <c>Parallel.For</c>: with <c>--block-concurrency</c> above one, a block's
    /// tower and decoder run on pool threads, and a thread-static set on the calling thread would
    /// not be there. <c>ParallelismScopeTests</c> pins both halves of that.
    /// </para>
    /// <para>
    /// It is read once per parallel region — a few thousand times over a page, against regions
    /// that each cost microseconds — so the lookup does not show up. The tower measures within
    /// noise of the <c>static readonly</c> field this replaced.
    /// </para>
    /// </remarks>
    public static Scope Use(ParallelOptions? options) => new(options);

    /// <summary>A period during which the kernels use one caller's options.</summary>
    public readonly struct Scope : IDisposable
    {
        private readonly ParallelOptions? _previous;
        private readonly bool _entered;

        internal Scope(ParallelOptions? options)
        {
            if (options is null)
            {
                _previous = null;
                _entered = false;
                return;
            }

            _previous = Scoped.Value;
            _entered = true;
            Scoped.Value = options;
        }

        /// <summary>Restores the options that were in scope before.</summary>
        public void Dispose()
        {
            if (_entered)
            {
                Scoped.Value = _previous;
            }
        }
    }
}
