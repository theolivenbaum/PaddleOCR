using PaddleOcrSharp.Core;
using PaddleOcrSharp.Tests.Fixtures;

namespace PaddleOcrSharp.Tests.Core;

/// <summary>Checks the SIMD kernels against straightforward scalar references.</summary>
public class KernelTests
{
    private static float[] Random(int count, int seed, float scale = 1f)
    {
        var random = new Random(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = (float)((random.NextDouble() * 2 - 1) * scale);
        }

        return values;
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(64)]
    [InlineData(1153)]
    public void SiluMatchesScalarReference(int count)
    {
        float[] values = Random(count, seed: count);
        float[] expected = values.Select(x => x / (1f + MathF.Exp(-x))).ToArray();

        Kernels.Silu(values);

        TensorAssert.Close(expected, values, absoluteTolerance: 1e-6, relativeTolerance: 1e-6);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(9)]
    [InlineData(4304)]
    public void GeluTanhMatchesScalarReference(int count)
    {
        float[] values = Random(count, seed: count + 1, scale: 6f);
        float[] expected = values
            .Select(x => 0.5f * x * (1f + MathF.Tanh(0.7978845608028654f * (x + (0.044715f * x * x * x)))))
            .ToArray();

        Kernels.GeluTanh(values);

        TensorAssert.Close(expected, values, absoluteTolerance: 2e-6, relativeTolerance: 1e-5);
    }

    [Fact]
    public void GeluErfMatchesKnownValues()
    {
        // Reference values from torch.nn.functional.gelu(x) with the exact (erf) formulation.
        float[] input = [-3f, -1f, -0.5f, 0f, 0.5f, 1f, 3f];
        float[] expected =
        [
            -0.00404969f, -0.15865526f, -0.15426877f, 0f, 0.34573123f, 0.84134471f, 2.99595031f,
        ];

        float[] actual = (float[])input.Clone();
        Kernels.GeluErf(actual);

        TensorAssert.Close(expected, actual, absoluteTolerance: 3e-6);
    }

    [Fact]
    public void SoftmaxSumsToOneAndIsShiftInvariant()
    {
        float[] values = Random(1024, seed: 42, scale: 20f);
        float[] shifted = values.Select(x => x + 100f).ToArray();

        Kernels.Softmax(values);
        Kernels.Softmax(shifted);

        Assert.Equal(1.0, values.Sum(), 4);
        // Shifting the inputs by 100 costs a few float32 ulps in the exponent, so compare
        // relatively rather than demanding bit equality.
        TensorAssert.Close(values, shifted, absoluteTolerance: 1e-9, relativeTolerance: 1e-5);
    }

    [Fact]
    public void SoftmaxHandlesMaskedRow()
    {
        float[] values = [0f, float.NegativeInfinity, 0f, float.NegativeInfinity];
        Kernels.Softmax(values);

        Assert.Equal(0.5f, values[0], 6);
        Assert.Equal(0f, values[1]);
        Assert.Equal(0.5f, values[2], 6);
    }

    [Fact]
    public void AddScaledMatchesScalarReference()
    {
        float[] destination = Random(333, seed: 7);
        float[] source = Random(333, seed: 8);
        float[] expected = destination.Zip(source, (d, s) => d + (s * 0.75f)).ToArray();

        Kernels.AddScaled(destination, source, 0.75f);

        TensorAssert.Close(expected, destination, absoluteTolerance: 1e-6);
    }

    [Theory]
    [InlineData(1f)]
    [InlineData(-2.5f)]
    [InlineData(1e-8f)]
    [InlineData(1.7e38f)]
    public void BFloat16RoundTripKeepsTopSixteenBits(float value)
    {
        ushort bits = FloatConversion.FloatToBF16(value);
        float restored = FloatConversion.BF16ToFloat(bits);

        Assert.Equal(value, restored, Math.Abs(value) * 0.01f);
    }

    [Fact]
    public void BFloat16SaturatesAndPreservesSpecialValues()
    {
        // Rounding a float32 just under the float32 maximum overflows the bfloat16 exponent, which
        // is exactly what `torch.Tensor.bfloat16()` does.
        Assert.True(float.IsPositiveInfinity(FloatConversion.BF16ToFloat(FloatConversion.FloatToBF16(3.4e38f))));
        Assert.True(float.IsNegativeInfinity(FloatConversion.BF16ToFloat(FloatConversion.FloatToBF16(-3.4e38f))));
        Assert.True(float.IsNaN(FloatConversion.BF16ToFloat(FloatConversion.FloatToBF16(float.NaN))));
        Assert.True(float.IsPositiveInfinity(
            FloatConversion.BF16ToFloat(FloatConversion.FloatToBF16(float.PositiveInfinity))));
        Assert.Equal(0f, FloatConversion.BF16ToFloat(FloatConversion.FloatToBF16(0f)));
    }

    [Fact]
    public void BFloat16WideningMatchesScalarPath()
    {
        var random = new Random(11);
        ushort[] bits = new ushort[1000];
        for (int i = 0; i < bits.Length; i++)
        {
            bits[i] = FloatConversion.FloatToBF16((float)((random.NextDouble() * 2 - 1) * 10));
        }

        float[] expected = bits.Select(FloatConversion.BF16ToFloat).ToArray();
        float[] actual = new float[bits.Length];
        FloatConversion.BF16ToFloat(bits, actual);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void RmsNormMatchesScalarReference()
    {
        const int Width = 1024;
        float[] values = Random(Width * 3, seed: 5);
        float[] weight = Random(Width, seed: 6, scale: 2f);
        float[] expected = new float[values.Length];

        for (int row = 0; row < 3; row++)
        {
            double sum = 0;
            for (int i = 0; i < Width; i++)
            {
                sum += (double)values[(row * Width) + i] * values[(row * Width) + i];
            }

            double scale = 1.0 / Math.Sqrt((sum / Width) + 1e-5);
            for (int i = 0; i < Width; i++)
            {
                expected[(row * Width) + i] = (float)(values[(row * Width) + i] * scale * weight[i]);
            }
        }

        Norms.RmsNorm(values, Width, weight, 1e-5f);

        TensorAssert.Close(expected, values, absoluteTolerance: 1e-5, relativeTolerance: 1e-5);
    }

    [Fact]
    public void LayerNormMatchesScalarReference()
    {
        const int Width = 1152;
        float[] values = Random(Width * 2, seed: 15);
        float[] weight = Random(Width, seed: 16, scale: 2f);
        float[] bias = Random(Width, seed: 17);
        float[] expected = new float[values.Length];

        for (int row = 0; row < 2; row++)
        {
            double mean = 0;
            for (int i = 0; i < Width; i++)
            {
                mean += values[(row * Width) + i];
            }

            mean /= Width;

            double variance = 0;
            for (int i = 0; i < Width; i++)
            {
                double d = values[(row * Width) + i] - mean;
                variance += d * d;
            }

            variance /= Width;
            double scale = 1.0 / Math.Sqrt(variance + 1e-6);

            for (int i = 0; i < Width; i++)
            {
                expected[(row * Width) + i] =
                    (float)(((values[(row * Width) + i] - mean) * scale * weight[i]) + bias[i]);
            }
        }

        Norms.LayerNorm(values, Width, weight, bias, 1e-6f);

        TensorAssert.Close(expected, values, absoluteTolerance: 1e-5, relativeTolerance: 1e-5);
    }
}
