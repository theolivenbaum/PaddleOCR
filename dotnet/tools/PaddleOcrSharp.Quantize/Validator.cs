using System.Diagnostics;
using PaddleOcrSharp.Formats.Gguf;
using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models;
using PaddleOcrSharp.Models.Vision;
using PaddleOcrSharp.Quantization;

namespace PaddleOcrSharp.Quantize;

/// <summary>
/// Compares a converted checkpoint against the one it came from, at the levels
/// <c>docs/ternary.md</c> calls L1 to L3.
/// </summary>
/// <remarks>
/// The ladder exists because the cheap measurements do not predict the expensive one. A tensor's
/// relative error says nothing about whether a page still reads correctly — the model has
/// eighteen layers of opportunity to absorb or amplify it — and a page that reads correctly says
/// nothing about which tensor to fix when one does not. Each level answers a different question,
/// so a conversion runs all three.
/// </remarks>
internal static class Validator
{
    public static int Run(string referenceDirectory, string quantizedDirectory, string[] images)
    {
        using PaddleOcrVLModel reference = PaddleOcrVLModel.Load(referenceDirectory);
        using PaddleOcrVLModel quantized = PaddleOcrVLModel.Load(quantizedDirectory);

        CompareTensors(referenceDirectory, quantizedDirectory);

        if (images.Length == 0)
        {
            Console.WriteLine();
            Console.WriteLine("No images given, so the stage and page levels were skipped.");
            return 0;
        }

        foreach (string path in images)
        {
            Console.WriteLine();
            Console.WriteLine($"=== {Path.GetFileName(path)} ===");
            using RgbImage image = ImageIO.Load(path);
            CompareStages(reference, quantized, image);
            ComparePages(reference, quantized, image);
        }

        return 0;
    }

    /// <summary>L1: what the packing cost each tensor.</summary>
    private static void CompareTensors(string referenceDirectory, string quantizedDirectory)
    {
        using var source = WeightStore.Open(PaddleOcrVLModel.WeightsPath(referenceDirectory));
        using GgufFile target = GgufFile.Open(PaddleOcrVLModel.WeightsPath(quantizedDirectory));
        QuantizationMetadata metadata = QuantizationMetadata.Read(target);

        Console.WriteLine("tensor                                                     type      rel.err  worst cos");

        var rows = new List<(string Name, string Type, double Error, double Cosine, long Collapsed)>();

        foreach (string name in target.Names.Order(StringComparer.Ordinal))
        {
            GgufTensor stored = target[name];
            if (!stored.Type.IsQuantized() || !source.Contains(name))
            {
                continue;
            }

            // Not source.Vector: that caches, and caching every widened tensor of a 0.9B
            // checkpoint is 3.8 GB of float32 held for the length of the comparison.
            float[] original = source.Tensor(name).ToFloats();
            float[] decoded = new float[stored.ElementCount];
            BlockCodec.Decode(stored.Type, stored.Bytes.Span, decoded);

            int cols = checked((int)stored.RowLength);
            int count = checked((int)stored.RowCount);

            // A rotated tensor holds W Rᵀ, which has no elementwise relationship to W at all.
            // Comparing the two directly reports a relative error above one and negative cosines —
            // which is what this did before, and it looked exactly like a catastrophic conversion
            // rather than like a comparison made in the wrong basis. The source is folded the same
            // way so both sides are in the basis the file is written in.
            if (metadata.For(name) is { } rotation)
            {
                rotation.Fold(original, count, cols);
            }
            (double error, double cosine, long collapsed) = RowStatistics(original, decoded, count, cols);
            rows.Add((name, stored.Type.TypeName(), error, cosine, collapsed));
        }

        foreach ((string name, string type, double error, double cosine, long collapsed) in
                 rows.OrderByDescending(row => row.Error).Take(20))
        {
            Console.WriteLine(
                $"{name,-58} {type,-8} {error,8:F4}  {cosine,9:F4}{(collapsed > 0 ? $"  {collapsed} collapsed" : string.Empty)}");
        }

        if (rows.Count > 20)
        {
            Console.WriteLine($"… and {rows.Count - 20} more");
        }

        if (rows.Count > 0)
        {
            Console.WriteLine(
                $"mean relative error {rows.Average(row => row.Error):F4}, "
                + $"lowest row cosine {rows.Min(row => row.Cosine):F4}, "
                + $"{rows.Sum(row => row.Collapsed)} rows collapsed to zero");
        }
    }

    /// <summary>
    /// L2: the vision tower's output and the decoder's first logits, which is where a tensor-level
    /// error either disappears or turns into a different token.
    /// </summary>
    private static void CompareStages(PaddleOcrVLModel reference, PaddleOcrVLModel quantized, RgbImage image)
    {
        using PreprocessedImage preprocessed = VisionPreprocessor.Preprocess(image, VisionPreprocessorOptions.Default);

        using PaddleOcrSharp.Core.Tensor expected = reference.Vision.Encode(preprocessed);
        using PaddleOcrSharp.Core.Tensor actual = quantized.Vision.Encode(preprocessed);

        Console.WriteLine(
            $"vision tower: {expected.Shape[0]} tokens, cosine {Cosine(expected.Span, actual.Span):F6}, "
            + $"relative error {RelativeError(expected.Span, actual.Span):F6}");

        int[] prompt = reference.BuildPrompt(preprocessed.Grid, "OCR:");
        Console.WriteLine($"prompt: {prompt.Length} tokens");
    }

    /// <summary>L3: the text itself, which is the only level a user ever sees.</summary>
    private static void ComparePages(PaddleOcrVLModel reference, PaddleOcrVLModel quantized, RgbImage image)
    {
        var stopwatch = Stopwatch.StartNew();
        string expected = reference.Recognize(image, "OCR:");
        TimeSpan referenceTime = stopwatch.Elapsed;

        stopwatch.Restart();
        string actual = quantized.Recognize(image, "OCR:");
        TimeSpan quantizedTime = stopwatch.Elapsed;

        Console.WriteLine(
            $"recognition: {referenceTime.TotalSeconds:F1} s reference, {quantizedTime.TotalSeconds:F1} s quantized "
            + $"({referenceTime.TotalSeconds / Math.Max(quantizedTime.TotalSeconds, 1e-9):F2}x)");

        if (string.Equals(expected, actual, StringComparison.Ordinal))
        {
            Console.WriteLine("text: identical");
            return;
        }

        Console.WriteLine($"text: character accuracy {CharacterAccuracy(expected, actual):P2}");
        Console.WriteLine($"  reference: {Preview(expected)}");
        Console.WriteLine($"  quantized: {Preview(actual)}");
    }

    private static (double Error, double Cosine, long Collapsed) RowStatistics(
        ReadOnlySpan<float> original, ReadOnlySpan<float> decoded, int rows, int cols)
    {
        double total = 0;
        double energy = 0;

        double[] dots = new double[rows];
        double[] sourceNorms = new double[rows];
        double[] targetNorms = new double[rows];

        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> a = original.Slice(r * cols, cols);
            ReadOnlySpan<float> b = decoded.Slice(r * cols, cols);
            double dot = 0;
            double normA = 0;
            double normB = 0;

            for (int i = 0; i < cols; i++)
            {
                double difference = a[i] - b[i];
                total += difference * difference;
                energy += (double)a[i] * a[i];
                dot += (double)a[i] * b[i];
                normA += (double)a[i] * a[i];
                normB += (double)b[i] * b[i];
            }

            dots[r] = dot;
            sourceNorms[r] = normA;
            targetNorms[r] = normB;
        }

        // The same rule the converter reports under, so the two cannot disagree about the same
        // file — which they did, and resolving it is what found the fp16 scale fallback.
        (double worst, long collapsed) = WeightQuantizer.WorstCosine(dots, sourceNorms, targetNorms);
        return (energy > 0 ? Math.Sqrt(total / energy) : 0, worst, collapsed);
    }

    private static double Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double dot = 0;
        double normA = 0;
        double normB = 0;
        int length = Math.Min(a.Length, b.Length);

        for (int i = 0; i < length; i++)
        {
            dot += (double)a[i] * b[i];
            normA += (double)a[i] * a[i];
            normB += (double)b[i] * b[i];
        }

        return normA > 0 && normB > 0 ? dot / Math.Sqrt(normA * normB) : 0;
    }

    private static double RelativeError(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double total = 0;
        double energy = 0;
        int length = Math.Min(a.Length, b.Length);

        for (int i = 0; i < length; i++)
        {
            double difference = a[i] - b[i];
            total += difference * difference;
            energy += (double)a[i] * a[i];
        }

        return energy > 0 ? Math.Sqrt(total / energy) : 0;
    }

    /// <summary>
    /// One minus the normalised edit distance, which is the corpus metric this repository already
    /// reports for the repetition-stop change.
    /// </summary>
    private static double CharacterAccuracy(string expected, string actual)
    {
        if (expected.Length == 0)
        {
            return actual.Length == 0 ? 1 : 0;
        }

        int[] previous = new int[actual.Length + 1];
        int[] current = new int[actual.Length + 1];

        for (int j = 0; j <= actual.Length; j++)
        {
            previous[j] = j;
        }

        for (int i = 1; i <= expected.Length; i++)
        {
            current[0] = i;
            for (int j = 1; j <= actual.Length; j++)
            {
                int cost = expected[i - 1] == actual[j - 1] ? 0 : 1;
                current[j] = Math.Min(Math.Min(current[j - 1] + 1, previous[j] + 1), previous[j - 1] + cost);
            }

            (previous, current) = (current, previous);
        }

        return Math.Max(0, 1.0 - ((double)previous[actual.Length] / expected.Length));
    }

    private static string Preview(string text)
    {
        string collapsed = text.ReplaceLineEndings(" ⏎ ");
        return collapsed.Length <= 120 ? collapsed : collapsed[..120] + " …";
    }
}
