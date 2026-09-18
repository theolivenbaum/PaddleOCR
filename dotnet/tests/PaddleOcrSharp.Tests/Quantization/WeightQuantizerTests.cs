using PaddleOcrSharp.Formats.Gguf;
using PaddleOcrSharp.Quantization;

namespace PaddleOcrSharp.Tests.Quantization;

/// <summary>The part of the pipeline that decides the trits, rather than the one that packs them.</summary>
public class WeightQuantizerTests
{
    [Theory]
    [InlineData(GgmlType.PQ2_0)]
    [InlineData(GgmlType.PTQ1_0)]
    public void AnAlreadyTernaryTensorIsCarriedThroughUnchanged(GgmlType type)
    {
        // The case the fork's encoder assumes and the case a QAT checkpoint would arrive in: the
        // quantizer must not "improve" weights that are already on the grid.
        const int Rows = 4;
        const int Cols = 256;
        var random = new Random(31);

        float[] weights = new float[Rows * Cols];
        for (int row = 0; row < Rows; row++)
        {
            float scale = (float)(Half)(0.02f + (row * 0.01f));
            for (int i = 0; i < Cols; i++)
            {
                weights[(row * Cols) + i] = (random.Next(3) - 1) * scale;
            }
        }

        float[] original = (float[])weights.Clone();
        byte[] packed = new byte[type.RowSize(Cols) * Rows];

        TensorQuantizationReport report = WeightQuantizer.Quantize(
            "test", weights, Rows, Cols, new QuantizationOptions(type), rotation: null, hessian: null, packed);

        float[] decoded = new float[weights.Length];
        TernaryBlocks.Decode(type, packed, decoded);

        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], decoded[i], 1e-4f);
        }

        Assert.True(report.RelativeError < 1e-3, $"relative error was {report.RelativeError}");
    }

    [Fact]
    public void SearchingForTheScaleBeatsTakingTheAbsoluteMaximum()
    {
        // amax is optimal only for an already-ternary group. On real-valued weights it spends the
        // whole range on the single largest one and rounds most of the rest to zero, which is the
        // reason this quantizer is more than a call to the fork's encoder.
        const int Rows = 8;
        const int Cols = 512;
        float[] source = Gaussian(Rows * Cols, seed: 37);

        double searched = Convert(source, Rows, Cols, new QuantizationOptions(ScaleSearchSteps: 24));
        double amaxOnly = Convert(source, Rows, Cols, new QuantizationOptions(ScaleSearchSteps: 0));

        Assert.True(
            searched < amaxOnly * 0.95,
            $"the search gave {searched:F4} against amax's {amaxOnly:F4}, which is not an improvement");
    }

    [Fact]
    public void AnIdentityHessianReproducesRoundToNearest()
    {
        // With H = I there is no direction to move error into, so GPTQ has to reduce to the
        // row-wise path. If it does not, the feedback maths is wrong rather than merely unhelpful.
        const int Rows = 4;
        const int Cols = 256;
        float[] source = Gaussian(Rows * Cols, seed: 41);

        double plain = Convert(source, Rows, Cols, new QuantizationOptions());

        double[] identity = new double[Cols * Cols];
        for (int i = 0; i < Cols; i++)
        {
            identity[(i * Cols) + i] = 1;
        }

        double feedback = Convert(source, Rows, Cols, new QuantizationOptions(), identity);

        Assert.Equal(plain, feedback, 0.02);
    }

    [Fact]
    public void ErrorFeedbackImprovesWhatTheActivationsActuallySee()
    {
        // GPTQ does not reduce ‖W − Ŵ‖; it moves the error into directions the calibration says
        // the activations do not visit. So the thing to measure is ‖(W − Ŵ)x‖ over those
        // activations, not the weight error.
        const int Rows = 4;
        const int Cols = 256;
        const int Samples = 64;

        float[] source = Gaussian(Rows * Cols, seed: 43);
        float[][] activations = new float[Samples][];
        var random = new Random(47);

        // A strongly anisotropic input: a handful of directions carry most of the energy, which is
        // what a real layer's activations look like and what gives the feedback something to use.
        float[] gains = new float[Cols];
        for (int i = 0; i < Cols; i++)
        {
            gains[i] = i < 16 ? 4f : 0.1f;
        }

        double[] hessian = new double[Cols * Cols];
        for (int s = 0; s < Samples; s++)
        {
            float[] sample = new float[Cols];
            for (int i = 0; i < Cols; i++)
            {
                sample[i] = (float)Gauss(random) * gains[i];
            }

            activations[s] = sample;
            for (int i = 0; i < Cols; i++)
            {
                for (int j = 0; j < Cols; j++)
                {
                    hessian[(i * Cols) + j] += (double)sample[i] * sample[j];
                }
            }
        }

        for (int i = 0; i < hessian.Length; i++)
        {
            hessian[i] /= Samples;
        }

        double plain = OutputError(source, Rows, Cols, activations, hessian: null);
        double feedback = OutputError(source, Rows, Cols, activations, hessian);

        Assert.True(
            feedback < plain,
            $"error feedback gave {feedback:F4} against round-to-nearest's {plain:F4}");
    }

    [Fact]
    public void AnInputWidthTheGroupDoesNotDivideIsRefused() =>
        Assert.Throws<ArgumentException>(() => WeightQuantizer.Quantize(
            "bad", new float[200], 1, 200, new QuantizationOptions(), null, null, new byte[256]));

    private static double Convert(
        float[] source, int rows, int cols, QuantizationOptions options, double[]? hessian = null)
    {
        float[] weights = (float[])source.Clone();
        byte[] packed = new byte[options.Scheme.RowSize(cols) * rows];
        return WeightQuantizer
            .Quantize("test", weights, rows, cols, options, rotation: null, hessian, packed)
            .RelativeError;
    }

    /// <summary>Relative error of the products, which is what GPTQ optimises.</summary>
    private static double OutputError(
        float[] source, int rows, int cols, float[][] activations, double[]? hessian)
    {
        float[] weights = (float[])source.Clone();
        var options = new QuantizationOptions();
        byte[] packed = new byte[options.Scheme.RowSize(cols) * rows];
        WeightQuantizer.Quantize("test", weights, rows, cols, options, rotation: null, hessian, packed);

        float[] decoded = new float[source.Length];
        TernaryBlocks.Decode(options.Scheme, packed, decoded);

        double total = 0;
        double energy = 0;

        foreach (float[] sample in activations)
        {
            for (int r = 0; r < rows; r++)
            {
                double expected = 0;
                double actual = 0;
                for (int i = 0; i < cols; i++)
                {
                    expected += source[(r * cols) + i] * sample[i];
                    actual += decoded[(r * cols) + i] * sample[i];
                }

                total += (expected - actual) * (expected - actual);
                energy += expected * expected;
            }
        }

        return Math.Sqrt(total / energy);
    }

    private static float[] Gaussian(int count, int seed)
    {
        var random = new Random(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = (float)Gauss(random) * 0.02f;
        }

        return values;
    }

    private static double Gauss(Random random) =>
        Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble());
}
