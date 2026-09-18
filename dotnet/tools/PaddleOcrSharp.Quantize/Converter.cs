using System.Diagnostics;
using PaddleOcrSharp.Core;
using PaddleOcrSharp.Formats;
using PaddleOcrSharp.Formats.Gguf;
using PaddleOcrSharp.Quantization;

namespace PaddleOcrSharp.Quantize;

/// <summary>What one tensor becomes, decided before anything is written.</summary>
internal sealed record ConversionPlanEntry(
    string Name,
    GgmlType Type,
    long[] Ne,
    int Rows,
    int Cols,
    long Parameters,
    bool Rotate,
    string? Note);

/// <summary>Settings for one conversion run.</summary>
internal sealed record ConversionSettings
{
    public required string SourceDirectory { get; init; }

    public required string Output { get; init; }

    public required QuantizationPolicy Policy { get; init; }

    public QuantizationOptions Options { get; init; } = new();

    public string? CalibrationDirectory { get; init; }

    public string CalibrationFilter { get; init; } = "*";

    public long CalibrationBudgetBytes { get; init; } = 2L << 30;

    public string? ReportPath { get; init; }
}

/// <summary>
/// Converts a published safetensors checkpoint into a quantized GGUF.
/// </summary>
/// <remarks>
/// The plan is built in full before a byte is written, because GGUF stores every tensor's offset
/// in its header: the sizes have to be known first. That turns out to be the right shape anyway —
/// the plan is also what gets printed, so a conversion can be inspected before it is run.
/// </remarks>
internal static class Converter
{
    public static int Run(ConversionSettings settings)
    {
        string weights = Path.Combine(settings.SourceDirectory, "model.safetensors");
        if (!File.Exists(weights))
        {
            Console.Error.WriteLine($"No model.safetensors in '{settings.SourceDirectory}'.");
            return 1;
        }

        using SafetensorsFile source = SafetensorsFile.Open(weights);

        List<ConversionPlanEntry> plan = BuildPlan(source, settings.Policy);
        HadamardRotation? rotation = settings.Policy.BuildRotation(
            plan.Where(entry => entry.Rotate).Select(entry => entry.Cols));

        var rotated = new HashSet<string>(
            plan.Where(entry => entry.Rotate).Select(entry => entry.Name), StringComparer.Ordinal);

        PrintPlan(plan);

        Dictionary<string, double[]> hessians = settings.CalibrationDirectory is null
            ? []
            : Calibration.Collect(settings, plan, rotation, rotated);

        var metadata = new QuantizationMetadata
        {
            Rotation = rotated.Count > 0 ? rotation : null,
            RotatedWeights = rotated,
            Method = (hessians.Count > 0 ? "gptq" : "rtn") + (rotated.Count > 0 ? "+hadamard" : string.Empty),
            Calibration = settings.CalibrationDirectory is null
                ? string.Empty
                : $"{Path.GetFileName(settings.CalibrationDirectory.TrimEnd(Path.DirectorySeparatorChar))} "
                  + $"({hessians.Count} tensors)",
            Source = Path.GetFileName(settings.SourceDirectory.TrimEnd(Path.DirectorySeparatorChar)),
            GroupSize = TernaryBlockGeometry.GroupSize,
        };

        var reports = new List<TensorQuantizationReport>();
        var stopwatch = Stopwatch.StartNew();

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(settings.Output))!);
        using (var writer = new GgufWriter(settings.Output))
        {
            writer.Add("general.architecture", "paddleocr-vl");
            writer.Add("general.name", metadata.Source);
            writer.Add("general.file_type", (uint)DominantType(plan));
            metadata.Write(writer);

            foreach (ConversionPlanEntry entry in plan)
            {
                writer.Declare(entry.Name, entry.Type, entry.Ne);
            }

            writer.BeginData();

            foreach (ConversionPlanEntry entry in plan)
            {
                WeightTensor tensor = source[entry.Name];

                if (!entry.Type.IsQuantized())
                {
                    writer.Write(entry.Name, Unquantized(tensor, entry.Type));
                    continue;
                }

                float[] values = tensor.ToFloats();
                byte[] packed = new byte[entry.Type.RowSize(entry.Cols) * entry.Rows];
                hessians.TryGetValue(entry.Name, out double[]? hessian);

                TensorQuantizationReport report = WeightQuantizer.Quantize(
                    entry.Name,
                    values,
                    entry.Rows,
                    entry.Cols,
                    settings.Options with { Scheme = entry.Type },
                    entry.Rotate ? rotation : null,
                    hessian,
                    packed);

                reports.Add(report);
                writer.Write(entry.Name, packed);

                Console.WriteLine(
                    $"  {entry.Name,-58} {report.Scheme.TypeName(),-7} "
                    + $"err {report.RelativeError,6:F4}  worst cos {report.WorstRowCosine,6:F4}  "
                    + $"zeros {report.ZeroFraction,5:P0}"
                    + $"{(report.DegenerateRows > 0 ? $"  {report.DegenerateRows} degenerate" : string.Empty)}"
                    + $"{(hessian is null ? string.Empty : "  gptq")}");
            }

            writer.Complete();
        }

        long bytes = new FileInfo(settings.Output).Length;
        long sourceBytes = new FileInfo(weights).Length;

        Console.WriteLine();
        Console.WriteLine(
            $"{settings.Output}: {bytes / 1e9:F2} GB from {sourceBytes / 1e9:F2} GB "
            + $"({(double)sourceBytes / bytes:F2}x) in {stopwatch.Elapsed.TotalSeconds:F1} s");

        if (reports.Count > 0)
        {
            Console.WriteLine(
                $"quantized {reports.Count} tensors; worst relative error "
                + $"{reports.Max(report => report.RelativeError):F4} on "
                + $"{reports.MaxBy(report => report.RelativeError).Name}");
            Console.WriteLine(
                $"lowest row cosine {reports.Min(report => report.WorstRowCosine):F4} on "
                + $"{reports.MinBy(report => report.WorstRowCosine).Name}; "
                + $"{reports.Sum(report => report.DegenerateRows)} degenerate rows excluded");
        }

        if (settings.ReportPath is not null)
        {
            Report.Write(settings.ReportPath, plan, reports);
            Console.WriteLine($"report: {settings.ReportPath}");
        }

        CopyCompanionFiles(settings.SourceDirectory, settings.Output);
        return 0;
    }

    /// <summary>
    /// Decides every tensor's stored type, falling back where the layout cannot represent it.
    /// </summary>
    /// <remarks>
    /// The fallback is not a detail. A block covers 128 weights along the input axis, so a tensor
    /// whose input width is not a multiple of 128 cannot be packed at all — and the vision MLP's
    /// second projection is 4304 wide, which 128 does not divide. That is 134 M parameters over 27
    /// layers, so the conversion says so out loud rather than leaving it to be noticed in the
    /// file size.
    /// </remarks>
    private static List<ConversionPlanEntry> BuildPlan(SafetensorsFile source, QuantizationPolicy policy)
    {
        var plan = new List<ConversionPlanEntry>();

        foreach (string name in source.Names.Order(StringComparer.Ordinal))
        {
            WeightTensor tensor = source[name];
            GgmlType requested = policy.SchemeFor(name, SourceType(tensor.Dtype));

            // A rank-1 tensor is one column, not an n x n matrix. Reading its element count as
            // both dimensions squares it, which is invisible in the file — `ne` comes from the
            // shape — and silently wrong in every count that uses rows x cols.
            int rows = tensor.Shape.Length > 1 ? tensor.Shape[0] : tensor.ElementCount;
            int cols = tensor.Shape.Length > 1 ? tensor.ElementCount / rows : 1;
            string? note = null;
            GgmlType type = requested;

            if (requested.IsQuantized())
            {
                if (tensor.Shape.Length < 2)
                {
                    type = GgmlType.F32;
                    note = "rank 1";
                }
                else if (cols % requested.BlockSize() != 0)
                {
                    type = GgmlType.BF16;
                    note = $"{cols} inputs is not a multiple of {requested.TypeName()}'s "
                        + $"{requested.BlockSize()}-weight group";
                }
            }

            long[] ne = type.IsQuantized()
                ? [cols, rows]
                : [.. tensor.Shape.Reverse().Select(dimension => (long)dimension)];

            bool rotate = type.IsQuantized()
                && policy.RotatesFor(name)
                && cols % policy.Rotation.BlockSize == 0;

            if (policy.RotatesFor(name) && !rotate && type.IsQuantized())
            {
                note = $"{cols} inputs is not a multiple of the {policy.Rotation.BlockSize}-wide rotation";
            }

            plan.Add(new ConversionPlanEntry(name, type, ne, rows, cols, tensor.ElementCount, rotate, note));
        }

        return plan;
    }

    private static void PrintPlan(List<ConversionPlanEntry> plan)
    {
        long quantizedParameters = 0;
        long totalParameters = 0;
        var byType = new Dictionary<GgmlType, long>();

        foreach (ConversionPlanEntry entry in plan)
        {
            long parameters = entry.Parameters;
            totalParameters += parameters;
            byType[entry.Type] = byType.GetValueOrDefault(entry.Type) + parameters;
            if (entry.Type.IsQuantized())
            {
                quantizedParameters += parameters;
            }

            if (entry.Note is not null)
            {
                Console.WriteLine($"  note: {entry.Name} stays {entry.Type.TypeName()} — {entry.Note}");
            }
        }

        Console.WriteLine();
        Console.WriteLine($"{plan.Count} tensors, {totalParameters / 1e6:F1} M parameters:");
        foreach ((GgmlType type, long parameters) in byType.OrderByDescending(pair => pair.Value))
        {
            Console.WriteLine(
                $"  {type.TypeName(),-7} {parameters / 1e6,8:F1} M  {(double)parameters / totalParameters,6:P1}");
        }

        Console.WriteLine($"  quantized share: {(double)quantizedParameters / totalParameters:P1}");
        Console.WriteLine();
    }

    private static GgmlType DominantType(List<ConversionPlanEntry> plan) =>
        plan.GroupBy(entry => entry.Type)
            .OrderByDescending(group => group.Sum(entry => entry.Parameters))
            .First()
            .Key;

    /// <summary>The ggml type a source dtype maps onto, for the <c>source</c> scheme.</summary>
    private static GgmlType SourceType(DType dtype) => dtype switch
    {
        DType.Float32 => GgmlType.F32,
        DType.Float16 => GgmlType.F32,
        DType.BFloat16 => GgmlType.BF16,
        _ => GgmlType.F32,
    };

    private static byte[] Unquantized(WeightTensor tensor, GgmlType type)
    {
        if (type == GgmlType.BF16 && tensor.Dtype == DType.BFloat16)
        {
            return tensor.Bytes.ToArray();
        }

        float[] values = tensor.ToFloats();

        if (type == GgmlType.F32)
        {
            byte[] bytes = new byte[values.Length * 4];
            Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            return bytes;
        }

        if (type == GgmlType.BF16)
        {
            ushort[] packed = new ushort[values.Length];
            FloatConversion.FloatToBF16(values, packed);
            byte[] bytes = new byte[values.Length * 2];
            Buffer.BlockCopy(packed, 0, bytes, 0, bytes.Length);
            return bytes;
        }

        throw new NotSupportedException($"Cannot write an unquantized tensor as {type.TypeName()}.");
    }

    /// <summary>
    /// Copies the config, tokenizer and preprocessor files beside the converted weights, so the
    /// output directory loads as a model directory.
    /// </summary>
    private static void CopyCompanionFiles(string sourceDirectory, string output)
    {
        string destination = Path.GetDirectoryName(Path.GetFullPath(output))!;
        if (Path.GetFullPath(sourceDirectory) == destination)
        {
            return;
        }

        foreach (string file in Directory.EnumerateFiles(sourceDirectory))
        {
            string name = Path.GetFileName(file);
            if (name is "model.safetensors" || name.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            File.Copy(file, Path.Combine(destination, name), overwrite: true);
        }

        Console.WriteLine($"copied the checkpoint's config and tokenizer into {destination}");
    }
}
