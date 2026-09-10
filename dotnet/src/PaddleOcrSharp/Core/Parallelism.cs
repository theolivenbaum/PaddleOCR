namespace PaddleOcrSharp.Core;

/// <summary>
/// The degree of parallelism the kernels use.
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
/// One instance is shared rather than constructed per call: a <c>ParallelOptions</c> is
/// configuration that <c>Parallel.For</c> only reads, and allocating one inside a loop that
/// attention enters thirty thousand times a layer would be its own regression.
/// </para>
/// </remarks>
internal static class Parallelism
{
    /// <summary>One worker per core, shared by every kernel that spreads work across threads.</summary>
    internal static readonly ParallelOptions Options = new()
    {
        MaxDegreeOfParallelism = Environment.ProcessorCount,
    };
}
