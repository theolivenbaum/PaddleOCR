using System.Diagnostics;

namespace PaddleOcrSharp.Models.Vision;

/// <summary>
/// Accumulates how long each stage of a model takes, so a run can say where its seconds went.
/// </summary>
/// <remarks>
/// The counterpart of the layout graph's <c>PirProfile</c>, for the hand-written towers, which
/// have no operator boundaries to hook. Passing one in costs a null check and a timestamp per
/// stage per layer — tens of microseconds over a whole page — and passing none costs a null
/// check. It exists because the shape of a hand-written encoder's cost is not obvious from the
/// FLOP count: the matrix products dominate the arithmetic but the norms, the rotary application
/// and the head shuffles all read and write the same activations again, and only measurement says
/// how much that is worth.
/// </remarks>
public sealed class StageProfile
{
    private readonly Dictionary<string, Entry> _entries = new(StringComparer.Ordinal);

    /// <summary>Starts timing a stage; dispose the result to record it.</summary>
    /// <param name="name">Stage name, aggregated across every call.</param>
    public Scope Measure(string name) => new(this, name);

    /// <summary>Records one stage's elapsed time.</summary>
    /// <param name="name">Stage name.</param>
    /// <param name="elapsed">How long it took.</param>
    public void Add(string name, TimeSpan elapsed)
    {
        ref Entry entry = ref System.Runtime.InteropServices.CollectionsMarshal
            .GetValueRefOrAddDefault(_entries, name, out _);

        entry.Elapsed += elapsed;
        entry.Count++;
    }

    /// <summary>Renders the stages, slowest first.</summary>
    public override string ToString()
    {
        double total = _entries.Values.Sum(entry => entry.Elapsed.TotalMilliseconds);
        var text = new System.Text.StringBuilder();
        text.AppendLine($"{"stage",-22}{"total",9}{"calls",8}{"share",9}{"per call",11}");

        foreach ((string name, Entry entry) in _entries.OrderByDescending(pair => pair.Value.Elapsed))
        {
            double milliseconds = entry.Elapsed.TotalMilliseconds;
            text.AppendLine(
                $"{name,-22}{milliseconds,7:F0}ms{entry.Count,8}{milliseconds / total * 100,8:F1} %"
                + $"{milliseconds / entry.Count,9:F2}ms");
        }

        text.Append($"{"total",-22}{total,7:F0}ms");
        return text.ToString();
    }

    private struct Entry
    {
        public TimeSpan Elapsed;
        public int Count;
    }

    /// <summary>Times a stage for the duration of a <c>using</c> block.</summary>
    public readonly struct Scope : IDisposable
    {
        private readonly StageProfile? _profile;
        private readonly string _name;
        private readonly long _started;

        internal Scope(StageProfile? profile, string name)
        {
            _profile = profile;
            _name = name;
            _started = Stopwatch.GetTimestamp();
        }

        /// <summary>Records the elapsed time.</summary>
        public void Dispose() => _profile?.Add(_name, Stopwatch.GetElapsedTime(_started));
    }
}
