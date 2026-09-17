using PaddleOcrSharp.Core;
using PaddleOcrSharp.Formats.Gguf;
using PaddleOcrSharp.Quantization;

namespace PaddleOcrSharp.Tests.Quantization;

/// <summary>
/// A quantized <see cref="WeightMatrix"/> against the float matrix it decodes to, through both of
/// <see cref="Gemm"/>'s paths.
/// </summary>
/// <remarks>
/// The two paths are not an implementation detail here. One row takes the fused kernel a decode
/// step uses, four or more take the panel kernel prefill uses, and they group the same products
/// differently — so both are checked against the same dequantized reference rather than against
/// each other.
/// </remarks>
public class QuantizedKernelTests
{
    [Theory]
    [InlineData(GgmlType.PQ2_0, 1)]
    [InlineData(GgmlType.PQ2_0, 7)]
    [InlineData(GgmlType.PTQ1_0, 1)]
    [InlineData(GgmlType.PTQ1_0, 7)]
    public void TheProductMatchesTheDequantizedWeight(GgmlType type, int rows)
    {
        const int Cols = 256;
        const int Inner = 384;

        (byte[] packed, float[] dequantized) = Weights(type, Cols, Inner, seed: 13);
        float[] activations = Activations(rows * Inner, seed: 17);

        float[] expected = new float[rows * Cols];
        Gemm.Linear(
            activations, rows, Inner, WeightMatrix.FromFloats(dequantized, Cols, Inner), default, expected, Cols);

        float[] actual = new float[rows * Cols];
        Gemm.Linear(
            activations,
            rows,
            Inner,
            WeightMatrix.CreateQuantized(packed, type, Cols, Inner),
            default,
            actual,
            Cols);

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], 2e-3f);
        }
    }

    [Theory]
    [InlineData(GgmlType.PQ2_0)]
    [InlineData(GgmlType.PTQ1_0)]
    public void WideningARowGivesTheSameValuesAsWideningThePanel(GgmlType type)
    {
        const int Cols = 64;
        const int Inner = 256;

        (byte[] packed, float[] dequantized) = Weights(type, Cols, Inner, seed: 19);
        var matrix = WeightMatrix.CreateQuantized(packed, type, Cols, Inner);

        float[] single = new float[Inner];
        float[] many = new float[Cols * Inner];
        matrix.CopyRow(5, single);
        matrix.CopyRows(0, Cols, many);

        Assert.Equal(dequantized, many);
        Assert.Equal(many.AsSpan(5 * Inner, Inner).ToArray(), single);
    }

    [Fact]
    public void ARotatedWeightIsUsedWithItsActivationTransformed()
    {
        // Folding the weight and leaving the activation alone would be a silently different model,
        // so this pins that Gemm applies the rotation the matrix carries. It compares against the
        // dequantized weights against the explicitly rotated activation, not against the
        // unquantized product: ternary is lossy enough on random weights that the latter would
        // pass with the rotation missing.
        const int Cols = 8;
        const int Inner = 128;

        float[] weights = Activations(Cols * Inner, seed: 23);
        float[] activations = Activations(Inner, seed: 29);

        var rotation = new HadamardRotation(Inner);
        float[] folded = (float[])weights.Clone();
        rotation.Fold(folded, Cols, Inner);

        int rowBytes = (int)GgmlType.PQ2_0.RowSize(Inner);
        byte[] packed = new byte[rowBytes * Cols];
        float[] dequantized = new float[Cols * Inner];
        for (int r = 0; r < Cols; r++)
        {
            TernaryBlocks.EncodePq20(folded.AsSpan(r * Inner, Inner), packed.AsSpan(r * rowBytes, rowBytes));
            TernaryBlocks.DecodePq20(packed.AsSpan(r * rowBytes, rowBytes), dequantized.AsSpan(r * Inner, Inner));
        }

        float[] rotated = (float[])activations.Clone();
        rotation.Apply(rotated, 1, Inner);

        float[] expected = new float[Cols];
        Gemm.Linear(
            rotated, 1, Inner, WeightMatrix.FromFloats(dequantized, Cols, Inner), default, expected, Cols);

        float[] actual = new float[Cols];
        Gemm.Linear(
            activations,
            1,
            Inner,
            WeightMatrix.CreateQuantized(packed, GgmlType.PQ2_0, Cols, Inner, rotation),
            default,
            actual,
            Cols);

        for (int i = 0; i < Cols; i++)
        {
            Assert.Equal(expected[i], actual[i], 1e-4f);
        }

        // And the rotation is not a no-op: the same weights without it give a different answer.
        float[] unrotated = new float[Cols];
        Gemm.Linear(
            activations,
            1,
            Inner,
            WeightMatrix.CreateQuantized(packed, GgmlType.PQ2_0, Cols, Inner),
            default,
            unrotated,
            Cols);

        Assert.NotEqual(expected[0], unrotated[0], 1e-2f);
    }

    [Fact]
    public void AQuantizedMatrixReportsItsOwnRowStride()
    {
        var matrix = WeightMatrix.CreateQuantized(
            new byte[28 * 4], GgmlType.PTQ1_0, rows: 4, cols: 128);

        Assert.True(matrix.IsQuantized);
        Assert.Equal(28, matrix.RowByteLength);
        Assert.False(matrix.TryGetFloats(out _));
    }

    [Fact]
    public void AnInputWidthTheGroupDoesNotDivideIsRefused() =>
        Assert.Throws<ArgumentException>(() =>
            WeightMatrix.CreateQuantized(new byte[1024], GgmlType.PTQ1_0, rows: 4, cols: 200));

    private static (byte[] Packed, float[] Dequantized) Weights(GgmlType type, int rows, int cols, int seed)
    {
        var random = new Random(seed);
        float[] source = new float[rows * cols];
        for (int i = 0; i < source.Length; i++)
        {
            source[i] = (float)((random.NextDouble() * 2) - 1) * 0.05f;
        }

        int rowBytes = (int)type.RowSize(cols);
        byte[] packed = new byte[rowBytes * rows];
        float[] dequantized = new float[source.Length];

        for (int r = 0; r < rows; r++)
        {
            TernaryBlocks.Encode(type, source.AsSpan(r * cols, cols), packed.AsSpan(r * rowBytes, rowBytes));
            TernaryBlocks.Decode(type, packed.AsSpan(r * rowBytes, rowBytes), dequantized.AsSpan(r * cols, cols));
        }

        return (packed, dequantized);
    }

    private static float[] Activations(int count, int seed)
    {
        var random = new Random(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = (float)((random.NextDouble() * 2) - 1);
        }

        return values;
    }
}
