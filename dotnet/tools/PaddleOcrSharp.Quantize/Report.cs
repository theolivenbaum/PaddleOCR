using System.Text.Json;
using PaddleOcrSharp.Formats.Gguf;
using PaddleOcrSharp.Quantization;

namespace PaddleOcrSharp.Quantize;

/// <summary>Writes a conversion's per-tensor numbers as JSON, for tuning a policy against.</summary>
/// <remarks>
/// The console output is for watching a conversion; this is for comparing two of them. A policy is
/// tuned by moving one tensor at a time between schemes and reading which way the numbers went,
/// which needs the numbers in a file rather than in a scrollback.
/// </remarks>
internal static class Report
{
    public static void Write(
        string path,
        List<ConversionPlanEntry> plan,
        List<TensorQuantizationReport> reports)
    {
        Dictionary<string, TensorQuantizationReport> byName =
            reports.ToDictionary(report => report.Name, StringComparer.Ordinal);

        using FileStream stream = File.Create(path);
        using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true });

        writer.WriteStartObject();
        writer.WriteStartArray("tensors");

        foreach (ConversionPlanEntry entry in plan)
        {
            writer.WriteStartObject();
            writer.WriteString("name", entry.Name);
            writer.WriteString("type", entry.Type.TypeName());
            writer.WriteNumber("rows", entry.Rows);
            writer.WriteNumber("cols", entry.Cols);
            writer.WriteBoolean("rotated", entry.Rotate);

            if (entry.Note is not null)
            {
                writer.WriteString("note", entry.Note);
            }

            if (byName.TryGetValue(entry.Name, out TensorQuantizationReport summary))
            {
                writer.WriteNumber("relativeError", summary.RelativeError);
                writer.WriteNumber("worstRowCosine", summary.WorstRowCosine);
                writer.WriteNumber("zeroFraction", summary.ZeroFraction);
                writer.WriteNumber("bitsPerWeight", summary.BitsPerWeight);
            }

            writer.WriteEndObject();
        }

        writer.WriteEndArray();
        writer.WriteEndObject();
    }
}
