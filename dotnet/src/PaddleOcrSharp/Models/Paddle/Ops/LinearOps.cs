using PaddleOcrSharp.Core;
using PaddleOcrSharp.Formats.Paddle;

namespace PaddleOcrSharp.Models.Paddle.Ops;

/// <summary>Matrix products, normalisations and softmax.</summary>
internal static class LinearOps
{
    /// <summary>
    /// Batched matrix product with optional transposes, matching <c>pd_op.matmul</c>.
    /// </summary>
    public static PaddleTensor MatMul(PaddleTensor x, PaddleTensor y, bool transposeX, bool transposeY)
    {
        int[] xShape = Promote(x.Shape);
        int[] yShape = Promote(y.Shape);

        int xRows = transposeX ? xShape[^1] : xShape[^2];
        int xCols = transposeX ? xShape[^2] : xShape[^1];
        int yRows = transposeY ? yShape[^1] : yShape[^2];
        int yCols = transposeY ? yShape[^2] : yShape[^1];

        if (xCols != yRows)
        {
            throw new InvalidOperationException(
                $"matmul shape mismatch: [{string.Join(",", x.Shape)}] x [{string.Join(",", y.Shape)}] " +
                $"(transposeX={transposeX}, transposeY={transposeY}).");
        }

        int[] xBatch = xShape[..^2];
        int[] yBatch = yShape[..^2];
        int[] batchShape = Broadcast.ResultShape(xBatch, yBatch);
        int batches = PaddleTensor.ElementCount(batchShape);

        int[] resultShape = [.. batchShape, xRows, yCols];
        PaddleTensor result = PaddleTensor.Float(resultShape);

        int[] xBatchStrides = Broadcast.StridesFor(xBatch, batchShape);
        int[] yBatchStrides = Broadcast.StridesFor(yBatch, batchShape);

        int xMatrix = xShape[^2] * xShape[^1];
        int yMatrix = yShape[^2] * yShape[^1];
        int outMatrix = xRows * yCols;

        ReadOnlyMemory<float> xData = x.FloatMemory;
        ReadOnlyMemory<float> yData = y.FloatMemory;
        Memory<float> output = result.FloatMemory;

        int[] counters = new int[batchShape.Length];
        for (int b = 0; b < batches; b++)
        {
            int xOffset = 0;
            int yOffset = 0;
            for (int axis = 0; axis < batchShape.Length; axis++)
            {
                xOffset += counters[axis] * xBatchStrides[axis];
                yOffset += counters[axis] * yBatchStrides[axis];
            }

            Gemm.MatMul(
                xData.Slice(xOffset * xMatrix, xMatrix),
                xRows,
                xCols,
                transposeX,
                yData.Slice(yOffset * yMatrix, yMatrix),
                yCols,
                transposeY,
                output.Slice(b * outMatrix, outMatrix));

            for (int axis = batchShape.Length - 1; axis >= 0; axis--)
            {
                if (++counters[axis] < batchShape[axis])
                {
                    break;
                }

                counters[axis] = 0;
            }
        }

        // Paddle drops the padded dimension when an input was rank-1.
        if (x.Rank == 1 && y.Rank == 1)
        {
            return result.Reshaped([]);
        }

        if (x.Rank == 1)
        {
            var shape = new List<int>(resultShape);
            shape.RemoveAt(shape.Count - 2);
            return result.Reshaped([.. shape]);
        }

        if (y.Rank == 1)
        {
            var shape = new List<int>(resultShape);
            shape.RemoveAt(shape.Count - 1);
            return result.Reshaped([.. shape]);
        }

        return result;
    }

    /// <summary>Batched matrix product without transposes, matching <c>pd_op.bmm</c>.</summary>
    public static PaddleTensor Bmm(PaddleTensor x, PaddleTensor y) => MatMul(x, y, false, false);

    /// <summary>
    /// Layer normalisation over the trailing dimensions from <paramref name="beginNormAxis"/>.
    /// </summary>
    public static PaddleTensor LayerNorm(
        PaddleTensor input,
        PaddleTensor? scale,
        PaddleTensor? bias,
        float epsilon,
        int beginNormAxis)
    {
        int width = 1;
        for (int i = beginNormAxis; i < input.Rank; i++)
        {
            width *= input.Shape[i];
        }

        PaddleTensor result = PaddleTensor.Float([.. input.Shape]);
        input.FloatSpan.CopyTo(result.FloatSpan);

        Norms.LayerNorm(
            result.FloatSpan,
            width,
            scale is not null ? scale.FloatSpan : OnesFor(width),
            bias is not null ? bias.FloatSpan : ReadOnlySpan<float>.Empty,
            epsilon);

        return result;
    }

    private static float[] OnesFor(int width)
    {
        float[] ones = new float[width];
        Array.Fill(ones, 1f);
        return ones;
    }

    /// <summary>Softmax along <paramref name="axis"/>.</summary>
    public static PaddleTensor Softmax(PaddleTensor input, int axis)
    {
        int rank = input.Rank;
        axis = axis < 0 ? axis + rank : axis;

        PaddleTensor result = PaddleTensor.Float([.. input.Shape]);
        input.FloatSpan.CopyTo(result.FloatSpan);

        int width = input.Shape[axis];
        int inner = 1;
        for (int i = axis + 1; i < rank; i++)
        {
            inner *= input.Shape[i];
        }

        int outer = width == 0 ? 0 : input.Count / (width * inner);
        Span<float> data = result.FloatSpan;

        if (inner == 1)
        {
            for (int o = 0; o < outer; o++)
            {
                Kernels.Softmax(data.Slice(o * width, width));
            }

            return result;
        }

        Span<float> scratch = new float[width];
        for (int o = 0; o < outer; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                int baseIndex = (o * width * inner) + i;
                for (int w = 0; w < width; w++)
                {
                    scratch[w] = data[baseIndex + (w * inner)];
                }

                Kernels.Softmax(scratch);

                for (int w = 0; w < width; w++)
                {
                    data[baseIndex + (w * inner)] = scratch[w];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Inference-time batch normalisation:
    /// <c>(x − mean) / sqrt(var + eps) · scale + bias</c> over the channel dimension.
    /// </summary>
    public static PaddleTensor BatchNorm(
        PaddleTensor input,
        PaddleTensor mean,
        PaddleTensor variance,
        PaddleTensor scale,
        PaddleTensor bias,
        float epsilon,
        string dataFormat)
    {
        if (dataFormat != "NCHW")
        {
            throw new NotSupportedException($"batch_norm data format '{dataFormat}' is not supported.");
        }

        int channels = input.Shape[1];
        int spatial = 1;
        for (int i = 2; i < input.Rank; i++)
        {
            spatial *= input.Shape[i];
        }

        int batch = input.Shape[0];
        PaddleTensor result = PaddleTensor.Float([.. input.Shape]);

        ReadOnlySpan<float> source = input.FloatSpan;
        Span<float> destination = result.FloatSpan;
        ReadOnlySpan<float> means = mean.FloatSpan;
        ReadOnlySpan<float> variances = variance.FloatSpan;
        ReadOnlySpan<float> scales = scale.FloatSpan;
        ReadOnlySpan<float> biases = bias.FloatSpan;

        float[] multiplier = new float[channels];
        float[] offset = new float[channels];
        for (int c = 0; c < channels; c++)
        {
            float inverse = 1f / MathF.Sqrt(variances[c] + epsilon);
            multiplier[c] = scales[c] * inverse;
            offset[c] = biases[c] - (means[c] * multiplier[c]);
        }

        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < channels; c++)
            {
                int start = ((n * channels) + c) * spatial;
                float m = multiplier[c];
                float o = offset[c];
                for (int i = 0; i < spatial; i++)
                {
                    destination[start + i] = (source[start + i] * m) + o;
                }
            }
        }

        return result;
    }

    private static int[] Promote(int[] shape) => shape.Length switch
    {
        0 => [1, 1],
        1 => [1, shape[0]],
        _ => shape,
    };
}
