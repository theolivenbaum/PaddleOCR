using System.Diagnostics;
using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Cli;

/// <summary>
/// Times <see cref="Gemm.Linear"/> at the shapes the model actually asks for.
/// </summary>
/// <remarks>
/// The stage timings say where the seconds go; this says how much of each second is the machine's
/// fault. Every shape below is one the vision tower or the decoder issues, so the efficiency
/// column — the fraction of the FMA ceiling <see cref="MachineProfile"/> just measured — is a
/// direct read on how much is left on the table.
/// </remarks>
public static class GemmBenchmark
{
    private sealed record Shape(string Name, int Rows, int Inner, int Cols);

    /// <summary>Runs the shape sweep and prints a table.</summary>
    /// <param name="machine">Calibration to normalise against; efficiency is omitted without it.</param>
    public static void Run(MachineProfile? machine)
    {
        Shape[] shapes =
        [
            // Vision tower, at the patch count a 980x392 page produces.
            new("vision qkv", 1960, 1152, 3456),
            new("vision proj", 1960, 1152, 1152),
            new("vision mlp up", 1960, 1152, 4304),
            new("vision mlp down", 1960, 4304, 1152),
            new("projector 1", 490, 4608, 4608),
            new("projector 2", 490, 4608, 1024),

            // Decoder prefill, then a single decode step, where the weight traffic dominates.
            new("decode prefill qkv", 503, 1024, 2560),
            new("decode prefill mlp", 503, 1024, 6144),
            new("decode step qkv", 1, 1024, 2560),
            new("decode step mlp", 1, 1024, 6144),
            new("decode step head", 1, 1024, 103424),
        ];

        double ceiling = machine?.VectorAllThreads ?? 0;
        Console.WriteLine("GEMM (bf16 weights, float32 accumulate)");
        Console.WriteLine(
            ceiling > 0
                ? "  shape                    m     k      n     ms     GF/s   of peak  spread"
                : "  shape                    m     k      n     ms     GF/s    spread");

        bool over = false;

        foreach (Shape shape in shapes)
        {
            (double milliseconds, double gflops, double spread) = Time(shape);
            string line =
                $"  {shape.Name,-18} {shape.Rows,6} {shape.Inner,5} {shape.Cols,6} "
                + $"{milliseconds,7:F1} {gflops,8:F1}";

            Console.WriteLine(
                ceiling > 0
                    ? $"{line} {gflops / ceiling * 100,8:F0}% {spread * 100,6:F1}%"
                    : $"{line} {spread * 100,6:F1}%");

            over |= ceiling > 0 && gflops > ceiling;
        }

        if (over)
        {
            // The ceiling is measured once, at the start. A shape beating it does not mean the
            // kernel exceeded the hardware; it means the machine was busier when the ceiling was
            // taken than when the shape ran, so the whole column is a lower bound.
            Console.WriteLine(
                "  a shape above 100% means the machine was busier during calibration than during"
                + " the sweep;\n  the ceiling is a lower bound, so read the column as ordering"
                + " rather than as absolute efficiency.");
        }

        Console.WriteLine();
    }

    private static (double Milliseconds, double GFlops, double Spread) Time(Shape shape)
    {
        float[] x = new float[(long)shape.Rows * shape.Inner];
        var random = new Random(17);
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = (float)(random.NextDouble() - 0.5);
        }

        WeightMatrix weight = SyntheticWeight(shape.Cols, shape.Inner, random);
        float[] y = new float[(long)shape.Rows * shape.Cols];

        // One untimed pass so the panel buffers are rented and the code is jitted.
        Gemm.Linear(x, shape.Rows, shape.Inner, weight, default, y, shape.Cols);

        double flops = 2.0 * shape.Rows * shape.Inner * shape.Cols;
        int repeats = Math.Max(1, (int)(4e9 / flops));

        // Best of nine, and the spread reported beside it. On a shared host a single sample moves
        // by 20% between runs, which is wider than most of the differences worth acting on; the
        // best sample is the one least contaminated by whatever else the host was doing, and the
        // spread says whether to believe the comparison at all.
        const int Samples = 9;
        double best = double.MaxValue;
        double worst = 0;

        for (int sample = 0; sample < Samples; sample++)
        {
            var clock = Stopwatch.StartNew();
            for (int i = 0; i < repeats; i++)
            {
                Gemm.Linear(x, shape.Rows, shape.Inner, weight, default, y, shape.Cols);
            }

            double seconds = clock.Elapsed.TotalSeconds / repeats;
            best = Math.Min(best, seconds);
            worst = Math.Max(worst, seconds);
        }

        return (best * 1000, flops / best / 1e9, worst > 0 ? (worst - best) / worst : 0);
    }

    /// <summary>A bf16 weight matrix of the given shape, filled with plausible magnitudes.</summary>
    private static WeightMatrix SyntheticWeight(int rows, int cols, Random random)
    {
        byte[] bytes = new byte[(long)rows * cols * sizeof(ushort)];
        Span<ushort> values = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, ushort>(bytes.AsSpan());

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = FloatConversion.FloatToBF16((float)(random.NextDouble() - 0.5) * 0.1f);
        }

        return WeightMatrix.Create(bytes, DType.BFloat16, rows, cols);
    }
}
