using PaddleOcrSharp.Quantization;

namespace PaddleOcrSharp.Tests.Quantization;

/// <summary>The rotation, against the matrix it is a fast transform of.</summary>
public class HadamardRotationTests
{
    [Fact]
    public void TheTransformIsTheNormalizedSylvesterMatrix()
    {
        // The fork builds an explicit n×n rotation whose sign is the parity of popcount(row & col)
        // (llama-model.cpp), then multiplies. The fast transform has to be that same matrix.
        const int N = 16;
        var rotation = new HadamardRotation(N);
        var random = new Random(3);

        float[] input = new float[N];
        for (int i = 0; i < N; i++)
        {
            input[i] = (float)((random.NextDouble() * 2) - 1);
        }

        float[] expected = Reference(input, N);
        float[] actual = (float[])input.Clone();
        rotation.Apply(actual, 1, N);

        for (int i = 0; i < N; i++)
        {
            Assert.Equal(expected[i], actual[i], 1e-5f);
        }
    }

    [Fact]
    public void TheRotationPreservesTheDotProductItWraps()
    {
        // This is the whole contract: (W Rᵀ)(R x) = W x. If it did not hold, a converted model
        // would be a different function rather than a compressed one.
        const int Width = 64;
        const int Rows = 3;
        var random = new Random(5);
        var rotation = new HadamardRotation(
            Width, new Dictionary<int, float[]> { [Width] = HadamardRotation.BuildSigns(Width, 7) });

        float[] weights = new float[Rows * Width];
        float[] activations = new float[Width];
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = (float)((random.NextDouble() * 2) - 1);
        }

        for (int i = 0; i < Width; i++)
        {
            activations[i] = (float)((random.NextDouble() * 2) - 1);
        }

        double[] expected = new double[Rows];
        for (int r = 0; r < Rows; r++)
        {
            for (int i = 0; i < Width; i++)
            {
                expected[r] += weights[(r * Width) + i] * activations[i];
            }
        }

        float[] folded = (float[])weights.Clone();
        rotation.Fold(folded, Rows, Width);
        float[] rotated = (float[])activations.Clone();
        rotation.Apply(rotated, 1, Width);

        for (int r = 0; r < Rows; r++)
        {
            double actual = 0;
            for (int i = 0; i < Width; i++)
            {
                actual += folded[(r * Width) + i] * rotated[i];
            }

            Assert.Equal(expected[r], actual, 1e-4);
        }
    }

    [Fact]
    public void ARotationFlattensWhatAQuantizerWouldOtherwiseSpendItsRangeOn()
    {
        // The reason to rotate at all. A group whose largest weight dwarfs the rest forces amax up
        // and rounds most of the group to zero; spreading the coordinate over the block does not.
        const int Width = 1024;
        float[] spiky = new float[Width];
        var random = new Random(9);
        for (int i = 0; i < Width; i++)
        {
            spiky[i] = (float)(random.NextDouble() - 0.5) * 0.01f;
        }

        spiky[17] = 1f;

        float before = Peakiness(spiky);
        var rotation = new HadamardRotation(Width);
        rotation.Apply(spiky, 1, Width);
        float after = Peakiness(spiky);

        Assert.True(after < before / 4f, $"peakiness went {before:F1} -> {after:F1}, which is not a flattening");
    }

    [Fact]
    public void AWidthThatTheBlockDoesNotDivideIsRefused()
    {
        var rotation = new HadamardRotation(1024);
        float[] values = new float[1536];

        Assert.Throws<ArgumentException>(() => rotation.Apply(values, 1, 1536));
    }

    [Fact]
    public void ARotationWithExplicitSignsRefusesAWidthItHasNoVectorFor()
    {
        var rotation = new HadamardRotation(
            16, new Dictionary<int, float[]> { [32] = HadamardRotation.BuildSigns(32, 1) });

        Assert.Throws<InvalidOperationException>(() => rotation.Apply(new float[64], 1, 64));
    }

    /// <summary>Ratio of the largest magnitude to the root-mean-square one.</summary>
    private static float Peakiness(ReadOnlySpan<float> values)
    {
        float maximum = 0;
        double energy = 0;
        foreach (float value in values)
        {
            maximum = MathF.Max(maximum, MathF.Abs(value));
            energy += (double)value * value;
        }

        return maximum / MathF.Sqrt((float)(energy / values.Length));
    }

    /// <summary>The explicit matrix product the fast transform stands in for.</summary>
    private static float[] Reference(ReadOnlySpan<float> input, int n)
    {
        float scale = 1f / MathF.Sqrt(n);
        float[] output = new float[n];

        for (int row = 0; row < n; row++)
        {
            double sum = 0;
            for (int col = 0; col < n; col++)
            {
                bool negative = int.PopCount(row & col) % 2 == 1;
                sum += (negative ? -scale : scale) * input[col];
            }

            output[row] = (float)sum;
        }

        return output;
    }
}
