using System.Text;

namespace PaddleOcrSharp.Models;

/// <summary>
/// What one call to <see cref="PaddleOcrVLModel"/>'s recognise entry point cost.
/// </summary>
/// <param name="Label">Block label the call was made for, or the empty string for a direct call.</param>
/// <param name="Patches">Patches the vision tower encoded.</param>
/// <param name="PromptTokens">Prompt length, image placeholders included.</param>
/// <param name="GeneratedTokens">Tokens the decoder produced, the stop token excluded.</param>
/// <param name="HitTokenBudget">Whether generation stopped on <see cref="GenerationOptions.MaxNewTokens"/> rather than on the stop token.</param>
/// <param name="StoppedEarly">Whether the repetition detector cut generation short.</param>
/// <param name="Vision">Time in the vision tower and projector.</param>
/// <param name="Prefill">Time in the decoder's prompt pass.</param>
/// <param name="Decode">Time in the token loop.</param>
/// <param name="DecodeLogits">Part of the token loop spent in the output head.</param>
/// <param name="ThreadAllocatedBytes">Bytes allocated on the calling thread. Exact and monotonic, and the figure the report shows: the buffers a recognition itself churns - the key/value cache above all - are allocated here rather than on the kernels' worker threads.</param>
/// <param name="TotalAllocatedBytes">Bytes allocated process-wide across the call, worker threads included. Approximate: the runtime sums per-thread allocation contexts, so a pool thread retiring mid-call subtracts its unused budget from the total. Clamped at zero for that reason.</param>
public readonly record struct RecognitionRecord(
    string Label,
    int Patches,
    int PromptTokens,
    int GeneratedTokens,
    bool HitTokenBudget,
    bool StoppedEarly,
    TimeSpan Vision,
    TimeSpan Prefill,
    TimeSpan Decode,
    TimeSpan DecodeLogits,
    long ThreadAllocatedBytes,
    long TotalAllocatedBytes)
{
    /// <summary>Wall time of the whole call.</summary>
    public TimeSpan Total => Vision + Prefill + Decode;

    /// <summary>Share of the token loop spent in the output head.</summary>
    public double LogitsShare =>
        Decode.TotalMilliseconds == 0 ? 0 : DecodeLogits.TotalMilliseconds / Decode.TotalMilliseconds;

    /// <summary>Mean time per generated token, the prefill excluded.</summary>
    public double MillisecondsPerToken =>
        GeneratedTokens == 0 ? 0 : Decode.TotalMilliseconds / GeneratedTokens;
}

/// <summary>
/// Collects a <see cref="RecognitionRecord"/> per block, so a page can say which block spent its
/// seconds and whether the decoder ran away.
/// </summary>
/// <remarks>
/// The VL half's answer to the vision tower's <c>StageProfile</c> and the layout graph's
/// <c>PirProfile</c>. Passing one costs a null check plus two timestamps and two allocation reads
/// per block; passing none costs a null check. It exists because a page's cost is not the sum of
/// its blocks' sizes — a single block whose decoder falls into repetition can outweigh every other
/// block on the page put together, and only the generated-token count makes that visible.
/// </remarks>
public sealed class RecognitionProfile
{
    private readonly List<RecognitionRecord> _records = [];
    private readonly Lock _gate = new();

    /// <summary>Every recorded call, in completion order.</summary>
    public IReadOnlyList<RecognitionRecord> Records
    {
        get
        {
            lock (_gate)
            {
                return [.. _records];
            }
        }
    }

    /// <summary>Records one call. Safe to call from several block threads at once.</summary>
    /// <param name="record">The call to record.</param>
    public void Add(RecognitionRecord record)
    {
        lock (_gate)
        {
            _records.Add(record);
        }
    }

    /// <summary>Renders the blocks, slowest first.</summary>
    public override string ToString()
    {
        IReadOnlyList<RecognitionRecord> records = Records;
        if (records.Count == 0)
        {
            return "no blocks recognised";
        }

        double total = records.Sum(record => record.Total.TotalMilliseconds);
        var text = new StringBuilder();
        text.AppendLine(
            $"{"block",-16}{"patches",9}{"prompt",8}{"gen",7}{"vision",9}{"prefill",9}{"decode",10}{"head",8}"
            + $"{"ms/tok",9}{"alloc",10}{"share",8}");

        foreach (RecognitionRecord record in records.OrderByDescending(record => record.Total))
        {
            string label = record.Label.Length == 0 ? "(page)" : record.Label;
            if (record.HitTokenBudget)
            {
                label += " !";
            }
            else if (record.StoppedEarly)
            {
                label += " ~";
            }

            text.AppendLine(
                $"{label,-16}{record.Patches,9}{record.PromptTokens,8}{record.GeneratedTokens,7}"
                + $"{record.Vision.TotalMilliseconds,7:F0}ms{record.Prefill.TotalMilliseconds,7:F0}ms"
                + $"{record.Decode.TotalMilliseconds,8:F0}ms{record.LogitsShare * 100,7:F0}%"
                + $"{record.MillisecondsPerToken,9:F1}"
                + $"{Bytes(record.ThreadAllocatedBytes),10}"
                + $"{record.Total.TotalMilliseconds / total * 100,7:F1} %");
        }

        text.AppendLine(
            $"{"total",-16}{records.Sum(r => r.Patches),9}{records.Sum(r => (long)r.PromptTokens),8}"
            + $"{records.Sum(r => (long)r.GeneratedTokens),7}"
            + $"{records.Sum(r => r.Vision.TotalMilliseconds),7:F0}ms"
            + $"{records.Sum(r => r.Prefill.TotalMilliseconds),7:F0}ms"
            + $"{records.Sum(r => r.Decode.TotalMilliseconds),8:F0}ms"
            + $"{records.Sum(r => r.DecodeLogits.TotalMilliseconds) / Math.Max(1, records.Sum(r => r.Decode.TotalMilliseconds)) * 100,7:F0}%"
            + $"{string.Empty,9}{Bytes(records.Sum(r => r.ThreadAllocatedBytes)),10}");

        text.Append("! stopped on the token budget   ~ stopped early on repetition");
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
