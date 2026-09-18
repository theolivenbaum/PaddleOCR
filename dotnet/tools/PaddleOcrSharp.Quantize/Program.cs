using PaddleOcrSharp.Formats.Gguf;
using PaddleOcrSharp.Quantization;
using PaddleOcrSharp.Quantize;

// A converter, not a library entry point: the verbs are few and the argument shapes are simple,
// so the parsing is inline rather than a dependency.
if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
{
    Usage();
    return 0;
}

string verb = args[0];
Dictionary<string, string> options = new(StringComparer.OrdinalIgnoreCase);
List<string> positional = [];

for (int i = 1; i < args.Length; i++)
{
    if (!args[i].StartsWith("--", StringComparison.Ordinal))
    {
        positional.Add(args[i]);
        continue;
    }

    string name = args[i][2..];
    int equals = name.IndexOf('=');
    if (equals >= 0)
    {
        options[name[..equals]] = name[(equals + 1)..];
        continue;
    }

    bool hasValue = i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal);
    options[name] = hasValue ? args[++i] : "true";
}

string? Option(string name) => options.TryGetValue(name, out string? value) ? value : null;

switch (verb)
{
    case "convert":
    {
        string? input = Option("in") ?? (positional.Count > 0 ? positional[0] : null);
        if (input is null)
        {
            Console.Error.WriteLine("convert needs --in <model directory>.");
            return 1;
        }

        string output = Option("out")
            ?? Path.Combine(input.TrimEnd(Path.DirectorySeparatorChar) + "-ternary", "model.gguf");

        QuantizationPolicy policy = Option("policy") is { } policyPath
            ? QuantizationPolicy.Load(policyPath)
            : BuildPolicy(Option("scheme"), Option("rotate"), Option("sign"), Option("seed"));

        var settings = new ConversionSettings
        {
            SourceDirectory = input,
            Output = output,
            Policy = policy,
            Options = new QuantizationOptions(
                ScaleSearchSteps: int.Parse(Option("scale-search") ?? "24"),
                ScaleSearchFloor: float.Parse(Option("scale-floor") ?? "0.35"),
                Damping: double.Parse(Option("damping") ?? "0.01"),
                BlockColumns: int.Parse(Option("gptq-block") ?? "128")),
            CalibrationDirectory = Option("calibration"),
            CalibrationFilter = Option("gptq-filter") ?? "*",
            CalibrationBudgetBytes = (long)(double.Parse(Option("calibration-budget-gb") ?? "2") * (1L << 30)),
            ReportPath = Option("report"),
        };

        return Converter.Run(settings);
    }

    case "validate":
    {
        string? reference = Option("reference");
        string? quantized = Option("quantized");
        if (reference is null || quantized is null)
        {
            Console.Error.WriteLine("validate needs --reference <dir> and --quantized <dir>.");
            return 1;
        }

        bool compareTensors = !string.Equals(Option("tensors"), "false", StringComparison.OrdinalIgnoreCase);
        return Validator.Run(reference, quantized, [.. positional], compareTensors);
    }

    case "inspect":
    {
        string? path = Option("in") ?? (positional.Count > 0 ? positional[0] : null);
        if (path is null)
        {
            Console.Error.WriteLine("inspect needs a .gguf path.");
            return 1;
        }

        Inspect(path);
        return 0;
    }

    case "policy":
    {
        string path = Option("out") ?? "policy.json";
        QuantizationPolicy.Recommended.Save(path);
        Console.WriteLine($"wrote the starting policy to {path}");
        return 0;
    }

    default:
        Console.Error.WriteLine($"Unknown verb '{verb}'.");
        Usage();
        return 1;
}

static QuantizationPolicy BuildPolicy(string? scheme, string? rotate, string? signMode, string? seed)
{
    int blockSize = rotate is null ? 0 : int.Parse(rotate);
    return new QuantizationPolicy
    {
        DefaultScheme = scheme ?? QuantizationPolicy.Recommended.DefaultScheme,
        Rules = QuantizationPolicy.Recommended.Rules,
        Rotation = new RotationPolicy(blockSize, signMode ?? "explicit", int.Parse(seed ?? "1")),
    };
}

static void Inspect(string path)
{
    using GgufFile file = GgufFile.Open(path);

    Console.WriteLine($"{path}");
    Console.WriteLine($"  {file.Count} tensors, alignment {file.Alignment}");
    Console.WriteLine();

    foreach ((string key, GgufValue value) in file.Metadata.OrderBy(pair => pair.Key, StringComparer.Ordinal))
    {
        string rendered = value.IsArray && value.Count > 8 ? $"[{value.Count} × {value.Type}]" : value.ToString();
        Console.WriteLine($"  {key,-42} {rendered}");
    }

    Console.WriteLine();

    var byType = new Dictionary<GgmlType, (long Tensors, long Parameters, long Bytes)>();
    foreach (string name in file.Names)
    {
        GgufTensor tensor = file[name];
        (long tensors, long parameters, long bytes) = byType.GetValueOrDefault(tensor.Type);
        byType[tensor.Type] = (tensors + 1, parameters + tensor.ElementCount, bytes + tensor.Bytes.Length);
    }

    long totalParameters = byType.Values.Sum(entry => entry.Parameters);
    long totalBytes = byType.Values.Sum(entry => entry.Bytes);

    Console.WriteLine("  type      tensors    parameters       bytes    bpw");
    foreach ((GgmlType type, (long tensors, long parameters, long bytes)) in
             byType.OrderByDescending(pair => pair.Value.Parameters))
    {
        Console.WriteLine(
            $"  {type.TypeName(),-8} {tensors,8} {parameters / 1e6,10:F1} M {bytes / 1e6,10:F1} MB "
            + $"{bytes * 8.0 / parameters,6:F2}");
    }

    Console.WriteLine(
        $"  {"total",-8} {file.Count,8} {totalParameters / 1e6,10:F1} M {totalBytes / 1e6,10:F1} MB "
        + $"{totalBytes * 8.0 / totalParameters,6:F2}");
}

static void Usage()
{
    Console.WriteLine(
        """
        paddleocr-quantize — convert and validate ternary PaddleOCR-VL checkpoints

          convert  --in <model dir> [--out <file.gguf>]
                   [--policy <policy.json> | --scheme ptq1_0|pq2_0]
                   [--rotate <block size>] [--sign identity|explicit] [--seed <n>]
                   [--calibration <image dir>] [--gptq-filter <glob>]
                   [--calibration-budget-gb <n>] [--damping <d>] [--gptq-block <n>]
                   [--scale-search <steps>] [--scale-floor <fraction>] [--report <file.json>]

          validate --reference <model dir> --quantized <model dir> [--tensors false] [image …]

          inspect  <file.gguf>

          policy   [--out policy.json]

        The design, and what each level of validation is for, is in dotnet/docs/ternary.md.
        """);
}
