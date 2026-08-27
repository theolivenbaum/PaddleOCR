using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace PaddleOcrSharp.Cli;

/// <summary>
/// What this machine can actually do, measured before any model is loaded.
/// </summary>
/// <remarks>
/// <para>
/// A shared virtual machine's throughput is not a constant. The core count is whatever the
/// hypervisor is willing to schedule, the clock moves with the host's thermal and AVX-512 licence
/// state, and a noisy neighbour can halve the memory bandwidth between one run and the next. A
/// stage time in milliseconds is therefore not comparable across runs, and comparing an
/// optimisation's "before" against its "after" on different days is how a regression gets
/// recorded as a win.
/// </para>
/// <para>
/// So every benchmark run starts by measuring the machine itself: the FMA rate the hardware will
/// actually sustain, at both vector widths and at one and every thread, and the read bandwidth at
/// each level of the hierarchy. The model stages are then reported as a fraction of those
/// ceilings as well as in milliseconds, which is the figure that means the same thing tomorrow.
/// </para>
/// </remarks>
public sealed record MachineProfile
{
    /// <summary>Single-threaded 512-bit FMA rate, GFLOP/s; zero when the width is unavailable.</summary>
    public required double Vector512SingleThread { get; init; }

    /// <summary>Single-threaded 256-bit FMA rate, GFLOP/s.</summary>
    public required double Vector256SingleThread { get; init; }

    /// <summary>All-thread FMA rate at the widest available vector, GFLOP/s.</summary>
    public required double VectorAllThreads { get; init; }

    /// <summary>Spread of the FMA measurement, as a fraction of the best sample.</summary>
    /// <remarks>
    /// The honest error bar on everything else. Above a few percent the machine is busy with
    /// something else and the run's stage times should be read as an upper bound.
    /// </remarks>
    public required double FmaJitter { get; init; }

    /// <summary>Read bandwidth from a buffer that fits in L1, GB/s.</summary>
    public required double L1ReadBandwidth { get; init; }

    /// <summary>Read bandwidth from a buffer sized for L2, GB/s.</summary>
    public required double L2ReadBandwidth { get; init; }

    /// <summary>Read bandwidth from a buffer sized for the last-level cache, GB/s.</summary>
    public required double L3ReadBandwidth { get; init; }

    /// <summary>Single-threaded read bandwidth from a buffer far larger than any cache, GB/s.</summary>
    public required double DramReadBandwidth { get; init; }

    /// <summary>All-thread read bandwidth from memory, GB/s.</summary>
    public required double DramReadBandwidthAllThreads { get; init; }

    /// <summary>Threads the measurement used.</summary>
    public required int Threads { get; init; }

    /// <summary>Widest vector the runtime's own policy will use, in bits.</summary>
    public required int VectorWidth { get; init; }

    /// <summary>Widest vector the hardware will actually run, in bits.</summary>
    /// <remarks>
    /// Wider than <see cref="VectorWidth"/> whenever the ISA is present but the runtime's
    /// preferred width is narrower, which is the default on every AVX-512 machine. The kernels
    /// use this one.
    /// </remarks>
    public required int WidestUsableBits { get; init; }

    /// <summary>
    /// Estimated clock, in GHz, inferred from the FMA rate and the number of FMA ports the
    /// two vector widths imply.
    /// </summary>
    public double EstimatedClockGhz
    {
        get
        {
            // The lane count has to match whichever measurement won, not the runtime's
            // preferred width: on a machine where 512-bit wins while the policy says 256, using
            // the policy's width doubles the inferred clock.
            bool wider = Vector512SingleThread >= Vector256SingleThread && Vector512SingleThread > 0;
            double best = wider ? Vector512SingleThread : Vector256SingleThread;
            double flopsPerPort = (wider ? 16 : 8) * 2.0;
            return flopsPerPort > 0 ? best / (flopsPerPort * FmaPorts) : 0;
        }
    }

    /// <summary>
    /// How many vector FMAs the core retires per cycle, inferred from the two widths.
    /// </summary>
    /// <remarks>
    /// A core with one 512-bit unit does 512-bit FMAs at the same rate per instruction as
    /// 256-bit ones, so doubling the width doubles the FLOP rate. A core with two units gains
    /// nothing further from the wider vector, because the second unit is what the 256-bit path
    /// was already using. The ratio between the two measurements separates the cases.
    /// </remarks>
    public int FmaPorts =>
        Vector512SingleThread <= 0 || Vector256SingleThread <= 0 ? 2
        : Vector512SingleThread / Vector256SingleThread > 1.6 ? 1
        : 2;

    /// <summary>Thread scaling actually achieved on the FMA test.</summary>
    public double ThreadScaling =>
        Math.Max(Vector512SingleThread, Vector256SingleThread) is var single && single > 0
            ? VectorAllThreads / single
            : 0;

    /// <summary>Measures the machine.</summary>
    /// <param name="seconds">Roughly how long to spend; the default keeps it under two seconds.</param>
    public static MachineProfile Measure(double seconds = 4.0)
    {
        int threads = Environment.ProcessorCount;
        double slice = seconds / 6;

        // Gated on the ISA, not on Vector512.IsHardwareAccelerated: the runtime's preferred
        // width defaults to 256 even where 512 runs, and the point of measuring both is to find
        // out what that default is costing.
        bool has512 = Avx512F.IsSupported && Avx512BW.IsSupported;

        (double best512, double jitter512) = has512 ? FmaRate(Fma512, slice) : (0, 0);
        (double best256, double jitter256) = Vector256.IsHardwareAccelerated || Avx2.IsSupported
            ? FmaRate(Fma256, slice)
            : (0, 0);

        Func<int, double> widest = has512 ? Fma512 : Fma256;
        double all = FmaRateParallel(widest, slice, threads);

        return new MachineProfile
        {
            Vector512SingleThread = best512,
            Vector256SingleThread = best256,
            VectorAllThreads = all,
            FmaJitter = Math.Max(jitter512, jitter256),
            L1ReadBandwidth = ReadRate(16 * 1024, slice),
            L2ReadBandwidth = ReadRate(512 * 1024, slice),
            L3ReadBandwidth = ReadRate(8 * 1024 * 1024, slice),
            DramReadBandwidth = ReadRate(192 * 1024 * 1024, slice),
            DramReadBandwidthAllThreads = ReadRateParallel(192 * 1024 * 1024, slice, threads),
            Threads = threads,
            VectorWidth = Vector512.IsHardwareAccelerated ? 512 : Vector256.IsHardwareAccelerated ? 256 : 128,
            WidestUsableBits = has512 ? 512 : Vector256.IsHardwareAccelerated || Avx2.IsSupported ? 256 : 128,
        };
    }

    /// <summary>Renders the profile as the block that heads a benchmark run.</summary>
    public string Describe()
    {
        var text = new System.Text.StringBuilder();
        text.AppendLine("Machine (measured before loading anything)");
        text.AppendLine(
            $"  {Threads} threads, {WidestUsableBits}-bit vectors"
            + (VectorWidth < WidestUsableBits ? $" (runtime prefers {VectorWidth})" : string.Empty)
            + ", "
            + $"{FmaPorts} FMA port{(FmaPorts == 1 ? string.Empty : "s")}, "
            + $"~{EstimatedClockGhz:F2} GHz");
        text.AppendLine(
            $"  FMA        {Vector256SingleThread,7:F1} GF/s @256  {Vector512SingleThread,7:F1} GF/s @512  "
            + $"{VectorAllThreads,7:F1} GF/s all threads (x{ThreadScaling:F2})");
        text.AppendLine(
            $"  Read       L1 {L1ReadBandwidth,6:F1}  L2 {L2ReadBandwidth,6:F1}  L3 {L3ReadBandwidth,6:F1}  "
            + $"DRAM {DramReadBandwidth,6:F1} / {DramReadBandwidthAllThreads:F1} all  (GB/s)");
        text.Append($"  Jitter     {FmaJitter * 100:F1}% across samples");

        if (FmaJitter > 0.05)
        {
            text.Append("  — the machine is busy; treat stage times as an upper bound");
        }

        return text.ToString();
    }

    /// <summary>Best-of-five, plus the spread, so a scheduling hiccup does not become the answer.</summary>
    private static (double Best, double Jitter) FmaRate(Func<int, double> body, double seconds)
    {
        const int Samples = 5;
        double best = 0;
        double worst = double.MaxValue;

        int iterations = Calibrate(body, seconds / Samples);

        for (int i = 0; i < Samples; i++)
        {
            double rate = body(iterations);
            best = Math.Max(best, rate);
            worst = Math.Min(worst, rate);
        }

        return (best, best > 0 ? (best - worst) / best : 0);
    }

    /// <summary>Grows the iteration count until one sample takes about the time asked for.</summary>
    private static int Calibrate(Func<int, double> body, double seconds)
    {
        int iterations = 1 << 12;
        for (int attempt = 0; attempt < 24; attempt++)
        {
            var clock = Stopwatch.StartNew();
            body(iterations);
            double elapsed = clock.Elapsed.TotalSeconds;

            if (elapsed >= seconds)
            {
                return iterations;
            }

            // Aim straight at the target rather than doubling, but never shrink and never let a
            // sub-microsecond sample suggest a wild jump.
            double factor = elapsed > 1e-6 ? Math.Min(16, seconds / elapsed) : 16;
            iterations = (int)Math.Min(int.MaxValue / 2, iterations * Math.Max(2, factor));
        }

        return iterations;
    }

    private static double FmaRateParallel(Func<int, double> body, double seconds, int threads)
    {
        int iterations = Calibrate(body, seconds);
        var rates = new double[threads];

        var clock = Stopwatch.StartNew();
        Parallel.For(0, threads, index => rates[index] = body(iterations));
        double elapsed = clock.Elapsed.TotalSeconds;

        // Each thread did the same work, so the aggregate rate is that work over the wall clock
        // rather than the sum of the per-thread rates: a thread that was descheduled for half the
        // run would otherwise be reported as if it had been running the whole time.
        double flopsPerThread = rates[0] > 0 ? rates[0] * 1e9 * PerThreadSeconds(rates[0], iterations) : 0;
        return flopsPerThread > 0 ? threads * flopsPerThread / elapsed / 1e9 : 0;
    }

    private static double PerThreadSeconds(double rate, int iterations) =>
        FlopsPerIteration(iterations) / (rate * 1e9);

    private static double FlopsPerIteration(int iterations) => (double)iterations * ChainCount * Lanes * 2;

    /// <summary>
    /// Independent accumulator chains, enough to cover the FMA latency at two issues per cycle.
    /// </summary>
    /// <remarks>
    /// Eight, and they have to be eight named locals rather than an array or a
    /// <c>stackalloc</c> span. The JIT will not enregister an indexed accumulator, so the
    /// indexed form measures store-to-load forwarding and reports about a quarter of the
    /// machine's real FMA rate — which, taken as the ceiling, would make every kernel below look
    /// like it was exceeding the hardware.
    /// </remarks>
    private const int ChainCount = 8;

    private static int Lanes => Vector512.IsHardwareAccelerated ? 16 : 8;

    /// <summary>
    /// One 512-bit FMA chain per accumulator, all reading the same in-register operands, so the
    /// loop measures issue throughput rather than any part of the memory system.
    /// </summary>
    /// <remarks>
    /// The accumulators are seeded with different values, which is not cosmetic. Started at zero
    /// they would all hold the same number after every step, the JIT would prove the eight chains
    /// equal, and it emits one <c>vfmadd231ps</c> reusing a single register eight times. The loop
    /// then measures the FMA's four-cycle latency rather than its two-per-cycle throughput and
    /// reports about a fifth of the machine's real rate.
    /// </remarks>
    /// <remarks>
    /// <see cref="MethodImplOptions.AggressiveOptimization"/> is load-bearing. Without it the
    /// method is first jitted at tier 0 and the long loop is escaped through on-stack
    /// replacement, and OSR code keeps the frame's locals where tier 0 put them — on the stack.
    /// The loop head then reloads all eight accumulators from memory and spills them again on
    /// every iteration, so the measurement reports store-forwarding throughput and lands about
    /// six times under the machine's real FMA rate.
    /// </remarks>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static double Fma512(int iterations)
    {
        Vector512<float> x = Vector512.Create(1.0000001f);
        Vector512<float> y = Vector512.Create(0.9999999f);
        Vector512<float> a0 = Vector512.Create(1f), a1 = Vector512.Create(2f);
        Vector512<float> a2 = Vector512.Create(3f), a3 = Vector512.Create(4f);
        Vector512<float> a4 = Vector512.Create(5f), a5 = Vector512.Create(6f);
        Vector512<float> a6 = Vector512.Create(7f), a7 = Vector512.Create(8f);

        var clock = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            a0 = Vector512.FusedMultiplyAdd(x, y, a0);
            a1 = Vector512.FusedMultiplyAdd(x, y, a1);
            a2 = Vector512.FusedMultiplyAdd(x, y, a2);
            a3 = Vector512.FusedMultiplyAdd(x, y, a3);
            a4 = Vector512.FusedMultiplyAdd(x, y, a4);
            a5 = Vector512.FusedMultiplyAdd(x, y, a5);
            a6 = Vector512.FusedMultiplyAdd(x, y, a6);
            a7 = Vector512.FusedMultiplyAdd(x, y, a7);
        }

        double seconds = clock.Elapsed.TotalSeconds;
        Consume(Vector512.Sum(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7));
        return seconds > 0 ? (double)iterations * ChainCount * 16 * 2 / seconds / 1e9 : 0;
    }

    /// <summary>The same loop at 256 bits, which is what separates one FMA port from two.</summary>
    /// <inheritdoc cref="Fma512" />
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static double Fma256(int iterations)
    {
        Vector256<float> x = Vector256.Create(1.0000001f);
        Vector256<float> y = Vector256.Create(0.9999999f);
        Vector256<float> a0 = Vector256.Create(1f), a1 = Vector256.Create(2f);
        Vector256<float> a2 = Vector256.Create(3f), a3 = Vector256.Create(4f);
        Vector256<float> a4 = Vector256.Create(5f), a5 = Vector256.Create(6f);
        Vector256<float> a6 = Vector256.Create(7f), a7 = Vector256.Create(8f);

        var clock = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            a0 = Vector256.FusedMultiplyAdd(x, y, a0);
            a1 = Vector256.FusedMultiplyAdd(x, y, a1);
            a2 = Vector256.FusedMultiplyAdd(x, y, a2);
            a3 = Vector256.FusedMultiplyAdd(x, y, a3);
            a4 = Vector256.FusedMultiplyAdd(x, y, a4);
            a5 = Vector256.FusedMultiplyAdd(x, y, a5);
            a6 = Vector256.FusedMultiplyAdd(x, y, a6);
            a7 = Vector256.FusedMultiplyAdd(x, y, a7);
        }

        double seconds = clock.Elapsed.TotalSeconds;
        Consume(Vector256.Sum(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7));
        return seconds > 0 ? (double)iterations * ChainCount * 8 * 2 / seconds / 1e9 : 0;
    }

    /// <summary>Streaming read bandwidth from a buffer of the given size, GB/s.</summary>
    private static double ReadRate(int bytes, double seconds)
    {
        float[] buffer = new float[bytes / sizeof(float)];
        buffer.AsSpan().Fill(1f);

        int passes = Math.Max(1, (int)(seconds * 4e9 / bytes));
        Consume(Read(buffer, 1));

        var clock = Stopwatch.StartNew();
        Consume(Read(buffer, passes));
        double elapsed = clock.Elapsed.TotalSeconds;

        return elapsed > 0 ? (double)bytes * passes / elapsed / 1e9 : 0;
    }

    private static double ReadRateParallel(int bytes, double seconds, int threads)
    {
        var buffers = new float[threads][];
        for (int i = 0; i < threads; i++)
        {
            buffers[i] = new float[bytes / sizeof(float)];
            buffers[i].AsSpan().Fill(1f);
        }

        int passes = Math.Max(1, (int)(seconds * 4e9 / bytes));

        var clock = Stopwatch.StartNew();
        Parallel.For(0, threads, index => Consume(Read(buffers[index], passes)));
        double elapsed = clock.Elapsed.TotalSeconds;

        return elapsed > 0 ? (double)bytes * passes * threads / elapsed / 1e9 : 0;
    }

    /// <summary>Sums the buffer with enough accumulators that the adds are not the limit.</summary>
    /// <inheritdoc cref="Fma512" />
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static float Read(float[] buffer, int passes)
    {
        // Eight chains for the same reason the FMA loop needs eight: with four, the vector adds
        // form dependency chains shorter than their own latency and the loop reports the add
        // latency rather than the cache's bandwidth.
        Vector512<float> a0 = Vector512.Create(1f), a1 = Vector512.Create(2f);
        Vector512<float> a2 = Vector512.Create(3f), a3 = Vector512.Create(4f);
        Vector512<float> a4 = Vector512.Create(5f), a5 = Vector512.Create(6f);
        Vector512<float> a6 = Vector512.Create(7f), a7 = Vector512.Create(8f);
        int step = Vector512<float>.Count * 8;

        for (int pass = 0; pass < passes; pass++)
        {
            ReadOnlySpan<float> span = buffer;
            int i = 0;
            for (; i <= span.Length - step; i += step)
            {
                a0 += Vector512.LoadUnsafe(in span[i]);
                a1 += Vector512.LoadUnsafe(in span[i + 16]);
                a2 += Vector512.LoadUnsafe(in span[i + 32]);
                a3 += Vector512.LoadUnsafe(in span[i + 48]);
                a4 += Vector512.LoadUnsafe(in span[i + 64]);
                a5 += Vector512.LoadUnsafe(in span[i + 80]);
                a6 += Vector512.LoadUnsafe(in span[i + 96]);
                a7 += Vector512.LoadUnsafe(in span[i + 112]);
            }
        }

        return Vector512.Sum(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7);
    }

    /// <summary>
    /// Keeps a loop's result alive without letting the JIT see a reason to spill.
    /// </summary>
    /// <remarks>
    /// The obvious way to consume a benchmark's result — assigning it to a <c>volatile</c> static
    /// — quietly invalidates the measurement. A volatile store is a release barrier, and the JIT
    /// responds by keeping every SIMD local in the frame across the loop rather than in
    /// registers, so the loop body becomes load, operate, store on every iteration and reports
    /// store-forwarding throughput. On this port's benchmark machine that turned 159 GFLOP/s into
    /// 27. <see cref="GC.KeepAlive"/> has the same effect on dead-code elimination and none on
    /// register allocation.
    /// </remarks>
    private static void Consume(float value) => GC.KeepAlive(value);
}
