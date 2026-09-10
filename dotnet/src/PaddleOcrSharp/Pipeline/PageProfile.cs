using System.Diagnostics;
using System.Text;

namespace PaddleOcrSharp.Pipeline;

/// <summary>
/// Wall time and allocations per pipeline stage, for the half of a page's cost that sits outside
/// the model call.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="Models.RecognitionProfile"/> accounts for the vision tower, the prefill and the
/// token loop, and the layout graph reports its own operators. Between them sits everything else a
/// page pays for — rasterising a PDF page, decoding an image, cropping and masking the regions,
/// stacking a merged group, trimming a formula's margins, painting a table's figure placeholders,
/// converting OTSL to HTML, and rendering the markdown. On a scanned page that is a fifth of the
/// run, and until this existed none of it was attributable.
/// </para>
/// <para>
/// A stage costs two timestamps and one allocation read, so the profile can stay on for a real
/// parse rather than only a benchmark. Stages nest: <c>recognize</c> contains <c>prepare</c>, so
/// the shares are reported against the outermost stage rather than summed.
/// </para>
/// </remarks>
public sealed class PageProfile
{
    private readonly Dictionary<string, Entry> _stages = new(StringComparer.Ordinal);
    private readonly Lock _gate = new();

    /// <summary>What one stage cost across every call to it.</summary>
    /// <param name="Stage">The stage's name.</param>
    /// <param name="Calls">How many times it ran.</param>
    /// <param name="Elapsed">Total wall time inside it.</param>
    /// <param name="Slowest">Its slowest single call.</param>
    /// <param name="AllocatedBytes">Bytes allocated on the calling threads while inside it.</param>
    public readonly record struct StageRecord(
        string Stage,
        int Calls,
        TimeSpan Elapsed,
        TimeSpan Slowest,
        long AllocatedBytes);

    private sealed class Entry
    {
        public int Calls;
        public long Ticks;
        public long SlowestTicks;
        public long Bytes;
    }

    /// <summary>Every stage recorded, most expensive first.</summary>
    public IReadOnlyList<StageRecord> Stages
    {
        get
        {
            lock (_gate)
            {
                return
                [
                    .. _stages
                        .Select(pair => new StageRecord(
                            pair.Key,
                            pair.Value.Calls,
                            TimeSpan.FromTicks(pair.Value.Ticks),
                            TimeSpan.FromTicks(pair.Value.SlowestTicks),
                            pair.Value.Bytes))
                        .OrderByDescending(record => record.Elapsed),
                ];
            }
        }
    }

    /// <summary>
    /// Starts timing <paramref name="stage"/>; disposing the result records what it cost.
    /// </summary>
    /// <param name="stage">Stage name, used as the aggregation key.</param>
    /// <returns>A scope to dispose at the end of the stage.</returns>
    public Scope Measure(string stage) => new(this, stage);

    /// <summary>Records one completed stage directly, for a caller that has its own clock.</summary>
    /// <param name="stage">Stage name.</param>
    /// <param name="elapsed">Time the stage took.</param>
    /// <param name="allocatedBytes">Bytes it allocated.</param>
    public void Add(string stage, TimeSpan elapsed, long allocatedBytes)
    {
        lock (_gate)
        {
            if (!_stages.TryGetValue(stage, out Entry? entry))
            {
                entry = new Entry();
                _stages[stage] = entry;
            }

            entry.Calls++;
            entry.Ticks += elapsed.Ticks;
            entry.SlowestTicks = Math.Max(entry.SlowestTicks, elapsed.Ticks);
            entry.Bytes += Math.Max(0, allocatedBytes);
        }
    }

    /// <summary>A stage in progress. Records the stage when disposed.</summary>
    public readonly struct Scope : IDisposable
    {
        private readonly PageProfile? _profile;
        private readonly string _stage;
        private readonly long _start;
        private readonly long _bytes;

        internal Scope(PageProfile? profile, string stage)
        {
            _profile = profile;
            _stage = stage;
            _start = Stopwatch.GetTimestamp();
            _bytes = GC.GetAllocatedBytesForCurrentThread();
        }

        /// <summary>Records the stage.</summary>
        public void Dispose() => _profile?.Add(
            _stage,
            Stopwatch.GetElapsedTime(_start),
            GC.GetAllocatedBytesForCurrentThread() - _bytes);
    }

    /// <summary>Renders the stages, most expensive first.</summary>
    public override string ToString()
    {
        IReadOnlyList<StageRecord> stages = Stages;
        if (stages.Count == 0)
        {
            return "no stages recorded";
        }

        // `page` wraps every other stage, so it is the denominator rather than a row of the sum.
        double total = stages.FirstOrDefault(stage => stage.Stage == "page").Elapsed.TotalMilliseconds;
        if (total <= 0)
        {
            total = stages.Sum(stage => stage.Elapsed.TotalMilliseconds);
        }

        var text = new StringBuilder();
        text.AppendLine($"{"stage",-22}{"calls",7}{"total",11}{"slowest",11}{"alloc",11}{"share",8}");

        foreach (StageRecord stage in stages)
        {
            text.AppendLine(
                $"{stage.Stage,-22}{stage.Calls,7}"
                + $"{stage.Elapsed.TotalMilliseconds,9:F0}ms"
                + $"{stage.Slowest.TotalMilliseconds,9:F0}ms"
                + $"{Bytes(stage.AllocatedBytes),11}"
                + $"{stage.Elapsed.TotalMilliseconds / total * 100,7:F1}%");
        }

        text.Append("stages nest; shares are against `page`");
        return text.ToString();
    }

    private static string Bytes(long value) => value switch
    {
        >= 1L << 30 => $"{value / (double)(1L << 30):F1} GiB",
        >= 1L << 20 => $"{value / (double)(1L << 20):F0} MiB",
        >= 1L << 10 => $"{value / (double)(1L << 10):F0} KiB",
        _ => $"{value} B",
    };
}
