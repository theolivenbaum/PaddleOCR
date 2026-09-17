using PaddleOcrSharp.Formats.Gguf;
using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models;
using PaddleOcrSharp.Quantization;

namespace PaddleOcrSharp.Quantize;

/// <summary>
/// Runs the unquantized model over a calibration corpus and returns one Hessian per tensor.
/// </summary>
/// <remarks>
/// <para>
/// A Hessian is <c>cols²</c> doubles — 8 MB at the decoder's width, 170 MB at the projector's —
/// and holding every tensor's at once would be tens of gigabytes. So the eligible tensors are
/// split into passes that fit a budget, and the corpus is run once per pass. That is the trade
/// this makes: wall-clock time, which is an offline cost paid once per released checkpoint,
/// against memory, which is a hard limit.
/// </para>
/// <para>
/// The Hessian is collected in the <i>unrotated</i> basis, because that is the basis the model
/// runs in. A rotated tensor needs <c>R H Rᵀ</c>, which is applied here rather than in the
/// quantizer, so the quantizer only ever sees a Hessian that matches the weights it was handed.
/// </para>
/// </remarks>
internal static class Calibration
{
    public static Dictionary<string, double[]> Collect(
        ConversionSettings settings,
        List<ConversionPlanEntry> plan,
        HadamardRotation? rotation,
        HashSet<string> rotated)
    {
        string[] images = Directory
            .EnumerateFiles(settings.CalibrationDirectory!)
            .Where(IsImage)
            .Order(StringComparer.Ordinal)
            .ToArray();

        if (images.Length == 0)
        {
            Console.Error.WriteLine($"No images in '{settings.CalibrationDirectory}'; falling back to round-to-nearest.");
            return [];
        }

        List<ConversionPlanEntry> eligible =
        [
            .. plan.Where(entry =>
                entry.Type.IsQuantized() && QuantizationPolicy.Matches(settings.CalibrationFilter, entry.Name)),
        ];

        if (eligible.Count == 0)
        {
            return [];
        }

        List<List<ConversionPlanEntry>> passes = Split(eligible, settings.CalibrationBudgetBytes);
        Console.WriteLine(
            $"calibrating over {images.Length} images, {eligible.Count} tensors, {passes.Count} pass(es)");

        var result = new Dictionary<string, double[]>(StringComparer.Ordinal);

        for (int pass = 0; pass < passes.Count; pass++)
        {
            List<ConversionPlanEntry> batch = passes[pass];
            Console.WriteLine(
                $"  pass {pass + 1}/{passes.Count}: {batch.Count} tensors, "
                + $"{batch.Sum(entry => (long)entry.Cols * entry.Cols * 8) / 1e9:F2} GB of Hessians");

            using PaddleOcrVLModel model = PaddleOcrVLModel.Load(settings.SourceDirectory);
            (ActivationRecorder recorder, IDisposable scope) =
                ActivationRecorder.Use(batch.Select(entry => entry.Name));

            using (scope)
            {
                foreach (string path in images)
                {
                    using RgbImage image = ImageIO.Load(path);
                    model.Recognize(image, "OCR:");
                    Console.WriteLine($"    {Path.GetFileName(path)}");
                }
            }

            foreach (ConversionPlanEntry entry in batch)
            {
                double[]? hessian = recorder.Hessian(entry.Name);
                if (hessian is null)
                {
                    Console.Error.WriteLine($"    {entry.Name}: nothing reached it; it will use round-to-nearest");
                    continue;
                }

                if (rotation is not null && rotated.Contains(entry.Name))
                {
                    RotateHessian(hessian, entry.Cols, rotation);
                }

                result[entry.Name] = hessian;
            }
        }

        return result;
    }

    /// <summary>Rewrites <c>H</c> as <c>R H Rᵀ</c>, in place.</summary>
    /// <remarks>
    /// The rotation is orthogonal, so this is a change of basis and not an approximation: the
    /// quantizer then minimises the same quantity — the error the activations actually see — in
    /// the basis the weights are stored in.
    /// </remarks>
    private static void RotateHessian(double[] hessian, int cols, HadamardRotation rotation)
    {
        float[] buffer = new float[cols];

        for (int r = 0; r < cols; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                buffer[c] = (float)hessian[(r * cols) + c];
            }

            rotation.Apply(buffer, 1, cols);

            for (int c = 0; c < cols; c++)
            {
                hessian[(r * cols) + c] = buffer[c];
            }
        }

        for (int c = 0; c < cols; c++)
        {
            for (int r = 0; r < cols; r++)
            {
                buffer[r] = (float)hessian[(r * cols) + c];
            }

            rotation.Apply(buffer, 1, cols);

            for (int r = 0; r < cols; r++)
            {
                hessian[(r * cols) + c] = buffer[r];
            }
        }
    }

    private static List<List<ConversionPlanEntry>> Split(List<ConversionPlanEntry> tensors, long budget)
    {
        var passes = new List<List<ConversionPlanEntry>>();
        var current = new List<ConversionPlanEntry>();
        long used = 0;

        foreach (ConversionPlanEntry entry in tensors.OrderByDescending(entry => entry.Cols))
        {
            long cost = (long)entry.Cols * entry.Cols * 8;
            if (current.Count > 0 && used + cost > budget)
            {
                passes.Add(current);
                current = [];
                used = 0;
            }

            current.Add(entry);
            used += cost;
        }

        if (current.Count > 0)
        {
            passes.Add(current);
        }

        return passes;
    }

    private static bool IsImage(string path) =>
        Path.GetExtension(path).ToLowerInvariant() is ".png" or ".jpg" or ".jpeg" or ".bmp" or ".webp";
}
