using PaddleOcrSharp.Core;
using PaddleOcrSharp.Tests.Fixtures;

namespace PaddleOcrSharp.Tests.Core;

/// <summary>Checks <see cref="Gemm"/> against a naive triple loop, for both weight dtypes.</summary>
public class GemmTests
{
    private static float[] Random(int count, int seed)
    {
        var random = new Random(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = (float)((random.NextDouble() * 2 - 1) * 0.5);
        }

        return values;
    }

    private static float[] Reference(float[] x, int rows, int inner, float[] w, float[]? bias, int cols)
    {
        float[] y = new float[rows * cols];
        for (int m = 0; m < rows; m++)
        {
            for (int n = 0; n < cols; n++)
            {
                double sum = bias?[n] ?? 0;
                for (int k = 0; k < inner; k++)
                {
                    sum += (double)x[(m * inner) + k] * w[(n * inner) + k];
                }

                y[(m * cols) + n] = (float)sum;
            }
        }

        return y;
    }

    [Theory]
    [InlineData(1, 1024, 3072)]
    [InlineData(4, 1152, 1152)]
    [InlineData(37, 71, 13)]
    [InlineData(129, 4608, 1024)]
    public void LinearWithFloat32WeightsMatchesReference(int rows, int inner, int cols)
    {
        float[] x = Random(rows * inner, seed: rows);
        float[] w = Random(cols * inner, seed: cols);
        float[] bias = Random(cols, seed: 99);

        float[] expected = Reference(x, rows, inner, w, bias, cols);
        float[] actual = new float[rows * cols];

        Gemm.Linear(x, rows, inner, WeightMatrix.FromFloats(w, cols, inner), bias, actual, cols);

        TensorAssert.Close(expected, actual, absoluteTolerance: 1e-4, relativeTolerance: 1e-4);
    }

    [Theory]
    [InlineData(1, 1024, 2048)]
    [InlineData(8, 1152, 4304)]
    [InlineData(53, 97, 31)]
    public void LinearWithBFloat16WeightsMatchesWidenedReference(int rows, int inner, int cols)
    {
        float[] x = Random(rows * inner, seed: rows + 3);
        float[] w = Random(cols * inner, seed: cols + 3);

        // Round-trip the weights so the reference sees exactly the values the kernel will read.
        ushort[] packed = new ushort[w.Length];
        FloatConversion.FloatToBF16(w, packed);
        float[] widened = new float[w.Length];
        FloatConversion.BF16ToFloat(packed, widened);

        float[] expected = Reference(x, rows, inner, widened, bias: null, cols);
        float[] actual = new float[rows * cols];

        var weight = WeightMatrix.Create(
            System.Runtime.InteropServices.MemoryMarshal.AsBytes<ushort>(packed).ToArray(),
            DType.BFloat16,
            cols,
            inner);

        Gemm.Linear(x, rows, inner, weight, ReadOnlyMemory<float>.Empty, actual, cols);

        TensorAssert.Close(expected, actual, absoluteTolerance: 1e-4, relativeTolerance: 1e-4);
    }

    // Small shapes exercise the tile remainders; the large one crosses the parallel threshold and
    // spans several tiles on both axes.
    [Theory]
    [InlineData(13, 29, 17, false, false)]
    [InlineData(13, 29, 17, true, false)]
    [InlineData(13, 29, 17, false, true)]
    [InlineData(13, 29, 17, true, true)]
    [InlineData(1, 64, 96, false, false)]
    [InlineData(96, 1, 3, false, true)]
    [InlineData(200, 137, 208, false, false)]
    [InlineData(200, 137, 208, false, true)]
    public void MatMulMatchesReference(int rows, int inner, int cols, bool transposeA, bool transposeB)
    {
        float[] a = Random(rows * inner, seed: 1);
        float[] b = Random(inner * cols, seed: 2);

        float[] expected = new float[rows * cols];
        for (int m = 0; m < rows; m++)
        {
            for (int n = 0; n < cols; n++)
            {
                double sum = 0;
                for (int k = 0; k < inner; k++)
                {
                    sum += (double)a[(m * inner) + k] * b[(k * cols) + n];
                }

                expected[(m * cols) + n] = (float)sum;
            }
        }

        float[] left = transposeA ? Transposed(a, rows, inner) : a;
        float[] right = transposeB ? Transposed(b, inner, cols) : b;

        float[] actual = new float[rows * cols];
        Gemm.MatMul(left, rows, inner, transposeA, right, cols, transposeB, actual);

        TensorAssert.Close(expected, actual, absoluteTolerance: 1e-5, relativeTolerance: 1e-5);
    }

    private static float[] Transposed(float[] values, int rows, int columns)
    {
        float[] result = new float[values.Length];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                result[(j * rows) + i] = values[(i * columns) + j];
            }
        }

        return result;
    }
}
