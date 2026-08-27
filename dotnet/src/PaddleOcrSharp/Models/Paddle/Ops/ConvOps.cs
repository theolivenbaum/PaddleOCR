using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Models.Paddle.Ops;

/// <summary>Convolution and pooling, in NCHW layout.</summary>
public static class ConvOps
{
    /// <summary>
    /// 2-D convolution, matching <c>pd_op.conv2d</c> and <c>pd_op.depthwise_conv2d</c>.
    /// </summary>
    /// <remarks>
    /// Implemented as im2col followed by a GEMM per group, which keeps the inner loop a
    /// contiguous dot product and lets the whole thing thread over output rows. Depthwise
    /// convolutions take a direct path instead: with one channel per group, im2col would
    /// materialise a column buffer far larger than the work it saves.
    /// </remarks>
    public static PaddleTensor Conv2d(
        PaddleTensor input,
        PaddleTensor weight,
        int[] strides,
        int[] paddings,
        int[] dilations,
        int groups,
        string paddingAlgorithm,
        string dataFormat)
    {
        if (dataFormat != "NCHW")
        {
            throw new NotSupportedException($"conv2d data format '{dataFormat}' is not supported.");
        }

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outChannels = weight.Shape[0];
        int kernelHeight = weight.Shape[2];
        int kernelWidth = weight.Shape[3];

        (int padTop, int padBottom, int padLeft, int padRight) = ResolvePadding(
            paddingAlgorithm,
            paddings,
            inHeight,
            inWidth,
            kernelHeight,
            kernelWidth,
            strides,
            dilations);

        int outHeight = ((inHeight + padTop + padBottom - (dilations[0] * (kernelHeight - 1)) - 1) / strides[0]) + 1;
        int outWidth = ((inWidth + padLeft + padRight - (dilations[1] * (kernelWidth - 1)) - 1) / strides[1]) + 1;

        PaddleTensor result = PaddleTensor.Float([batch, outChannels, outHeight, outWidth]);
        if (result.Count == 0)
        {
            return result;
        }

        if (groups == inChannels && groups == outChannels)
        {
            Depthwise(
                input, weight, result,
                batch, inChannels, inHeight, inWidth,
                kernelHeight, kernelWidth, outHeight, outWidth,
                strides, dilations, padTop, padLeft);
            return result;
        }

        int inGroupChannels = inChannels / groups;
        int outGroupChannels = outChannels / groups;
        int patch = inGroupChannels * kernelHeight * kernelWidth;

        ReadOnlySpan<float> weights = weight.FloatSpan;
        float[] weightArray = weight.Floats!;

        for (int n = 0; n < batch; n++)
        {
            for (int g = 0; g < groups; g++)
            {
                int inputBase = ((n * inChannels) + (g * inGroupChannels)) * inHeight * inWidth;
                int outputBase = ((n * outChannels) + (g * outGroupChannels)) * outHeight * outWidth;
                int weightBase = g * outGroupChannels * patch;

                Parallel.For(0, outHeight, () => TensorPool.Rent(outWidth * patch), (oy, _, column) =>
                {
                    Span<float> columns = column.Span;
                    Im2ColRow(
                        input.Floats!, inputBase, inGroupChannels, inHeight, inWidth,
                        kernelHeight, kernelWidth, strides, dilations, padTop, padLeft,
                        oy, outWidth, columns);

                    // One column at a time, four filters at a time: the column stays in L1 while
                    // the filters stream past.
                    ReadOnlySpan<float> filters = weightArray.AsSpan(weightBase, outGroupChannels * patch);
                    int planeStride = outHeight * outWidth;
                    int rowOffset = outputBase + (oy * outWidth);

                    for (int ox = 0; ox < outWidth; ox++)
                    {
                        ReadOnlySpan<float> patchColumn = columns.Slice(ox * patch, patch);
                        int oc = 0;

                        for (; oc <= outGroupChannels - 4; oc += 4)
                        {
                            Gemm.Dot4(
                                patchColumn, filters, oc * patch, patch,
                                out float a0, out float a1, out float a2, out float a3);

                            result.Floats![rowOffset + (oc * planeStride) + ox] = a0;
                            result.Floats![rowOffset + ((oc + 1) * planeStride) + ox] = a1;
                            result.Floats![rowOffset + ((oc + 2) * planeStride) + ox] = a2;
                            result.Floats![rowOffset + ((oc + 3) * planeStride) + ox] = a3;
                        }

                        for (; oc < outGroupChannels; oc++)
                        {
                            result.Floats![rowOffset + (oc * planeStride) + ox] =
                                Gemm.Dot(patchColumn, filters.Slice(oc * patch, patch));
                        }
                    }

                    return column;
                },
                column => column.Dispose());
            }
        }

        _ = weights;
        return result;
    }

    private static void Im2ColRow(
        float[] input,
        int inputBase,
        int channels,
        int inHeight,
        int inWidth,
        int kernelHeight,
        int kernelWidth,
        int[] strides,
        int[] dilations,
        int padTop,
        int padLeft,
        int outY,
        int outWidth,
        Span<float> columns)
    {
        int patch = channels * kernelHeight * kernelWidth;
        columns[..(outWidth * patch)].Clear();

        for (int c = 0; c < channels; c++)
        {
            int channelBase = inputBase + (c * inHeight * inWidth);
            for (int ky = 0; ky < kernelHeight; ky++)
            {
                int iy = (outY * strides[0]) - padTop + (ky * dilations[0]);
                if ((uint)iy >= (uint)inHeight)
                {
                    continue;
                }

                int rowBase = channelBase + (iy * inWidth);
                for (int kx = 0; kx < kernelWidth; kx++)
                {
                    int patchOffset = (((c * kernelHeight) + ky) * kernelWidth) + kx;
                    int start = -padLeft + (kx * dilations[1]);

                    for (int ox = 0; ox < outWidth; ox++)
                    {
                        int ix = start + (ox * strides[1]);
                        if ((uint)ix < (uint)inWidth)
                        {
                            columns[(ox * patch) + patchOffset] = input[rowBase + ix];
                        }
                    }
                }
            }
        }
    }

    private static void Depthwise(
        PaddleTensor input,
        PaddleTensor weight,
        PaddleTensor result,
        int batch,
        int channels,
        int inHeight,
        int inWidth,
        int kernelHeight,
        int kernelWidth,
        int outHeight,
        int outWidth,
        int[] strides,
        int[] dilations,
        int padTop,
        int padLeft)
    {
        float[] source = input.Floats!;
        float[] filters = weight.Floats!;
        float[] destination = result.Floats!;
        int kernelSize = kernelHeight * kernelWidth;

        Parallel.For(0, batch * channels, index =>
        {
            int n = index / channels;
            int c = index % channels;
            int inputBase = ((n * channels) + c) * inHeight * inWidth;
            int filterBase = c * kernelSize;
            int outputBase = ((n * channels) + c) * outHeight * outWidth;

            for (int oy = 0; oy < outHeight; oy++)
            {
                for (int ox = 0; ox < outWidth; ox++)
                {
                    float sum = 0f;
                    for (int ky = 0; ky < kernelHeight; ky++)
                    {
                        int iy = (oy * strides[0]) - padTop + (ky * dilations[0]);
                        if ((uint)iy >= (uint)inHeight)
                        {
                            continue;
                        }

                        int rowBase = inputBase + (iy * inWidth);
                        int filterRow = filterBase + (ky * kernelWidth);

                        for (int kx = 0; kx < kernelWidth; kx++)
                        {
                            int ix = (ox * strides[1]) - padLeft + (kx * dilations[1]);
                            if ((uint)ix < (uint)inWidth)
                            {
                                sum += source[rowBase + ix] * filters[filterRow + kx];
                            }
                        }
                    }

                    destination[outputBase + (oy * outWidth) + ox] = sum;
                }
            }
        });
    }

    /// <summary>2-D pooling, matching <c>pd_op.pool2d</c>.</summary>
    public static PaddleTensor Pool2d(
        PaddleTensor input,
        int[] kernel,
        int[] strides,
        int[] paddings,
        bool ceilMode,
        bool exclusive,
        bool globalPooling,
        bool adaptive,
        string poolingType,
        string paddingAlgorithm,
        string dataFormat)
    {
        if (dataFormat != "NCHW")
        {
            throw new NotSupportedException($"pool2d data format '{dataFormat}' is not supported.");
        }

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        if (globalPooling)
        {
            kernel = [inHeight, inWidth];
            paddings = [0, 0];
        }

        if (adaptive)
        {
            return Adaptive(input, kernel, poolingType);
        }

        (int padTop, int padBottom, int padLeft, int padRight) = ResolvePadding(
            paddingAlgorithm, paddings, inHeight, inWidth, kernel[0], kernel[1], strides, [1, 1]);

        int outHeight = OutputExtent(inHeight, kernel[0], strides[0], padTop, padBottom, ceilMode);
        int outWidth = OutputExtent(inWidth, kernel[1], strides[1], padLeft, padRight, ceilMode);

        PaddleTensor result = PaddleTensor.Float([batch, channels, outHeight, outWidth]);
        float[] source = input.Floats!;
        float[] destination = result.Floats!;
        bool isMax = poolingType == "max";

        Parallel.For(0, batch * channels, index =>
        {
            int inputBase = index * inHeight * inWidth;
            int outputBase = index * outHeight * outWidth;

            for (int oy = 0; oy < outHeight; oy++)
            {
                int startY = (oy * strides[0]) - padTop;
                int endY = Math.Min(startY + kernel[0], inHeight);
                int clampedStartY = Math.Max(startY, 0);

                for (int ox = 0; ox < outWidth; ox++)
                {
                    int startX = (ox * strides[1]) - padLeft;
                    int endX = Math.Min(startX + kernel[1], inWidth);
                    int clampedStartX = Math.Max(startX, 0);

                    if (isMax)
                    {
                        float best = float.NegativeInfinity;
                        for (int y = clampedStartY; y < endY; y++)
                        {
                            int rowBase = inputBase + (y * inWidth);
                            for (int x = clampedStartX; x < endX; x++)
                            {
                                best = Math.Max(best, source[rowBase + x]);
                            }
                        }

                        destination[outputBase + (oy * outWidth) + ox] = best;
                    }
                    else
                    {
                        float sum = 0f;
                        for (int y = clampedStartY; y < endY; y++)
                        {
                            int rowBase = inputBase + (y * inWidth);
                            for (int x = clampedStartX; x < endX; x++)
                            {
                                sum += source[rowBase + x];
                            }
                        }

                        int area = exclusive
                            ? (endY - clampedStartY) * (endX - clampedStartX)
                            : kernel[0] * kernel[1];
                        destination[outputBase + (oy * outWidth) + ox] = area > 0 ? sum / area : 0f;
                    }
                }
            }
        });

        return result;
    }

    private static PaddleTensor Adaptive(PaddleTensor input, int[] output, string poolingType)
    {
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];
        int outHeight = output[0];
        int outWidth = output[1];

        PaddleTensor result = PaddleTensor.Float([batch, channels, outHeight, outWidth]);
        float[] source = input.Floats!;
        float[] destination = result.Floats!;
        bool isMax = poolingType == "max";

        Parallel.For(0, batch * channels, index =>
        {
            int inputBase = index * inHeight * inWidth;
            int outputBase = index * outHeight * outWidth;

            for (int oy = 0; oy < outHeight; oy++)
            {
                int startY = oy * inHeight / outHeight;
                int endY = (((oy + 1) * inHeight) + outHeight - 1) / outHeight;

                for (int ox = 0; ox < outWidth; ox++)
                {
                    int startX = ox * inWidth / outWidth;
                    int endX = (((ox + 1) * inWidth) + outWidth - 1) / outWidth;

                    float accumulator = isMax ? float.NegativeInfinity : 0f;
                    for (int y = startY; y < endY; y++)
                    {
                        int rowBase = inputBase + (y * inWidth);
                        for (int x = startX; x < endX; x++)
                        {
                            accumulator = isMax
                                ? Math.Max(accumulator, source[rowBase + x])
                                : accumulator + source[rowBase + x];
                        }
                    }

                    destination[outputBase + (oy * outWidth) + ox] = isMax
                        ? accumulator
                        : accumulator / ((endY - startY) * (endX - startX));
                }
            }
        });

        return result;
    }

    private static int OutputExtent(int size, int kernel, int stride, int padBefore, int padAfter, bool ceilMode)
    {
        int numerator = size + padBefore + padAfter - kernel;
        int extent = ceilMode
            ? ((numerator + stride - 1) / stride) + 1
            : (numerator / stride) + 1;

        // Paddle drops a trailing window that starts inside the padding.
        if (ceilMode && (extent - 1) * stride >= size + padBefore)
        {
            extent--;
        }

        return Math.Max(extent, 1);
    }

    /// <summary>
    /// Resolves the four padding amounts, handling the <c>SAME</c> and <c>VALID</c> algorithms.
    /// </summary>
    private static (int Top, int Bottom, int Left, int Right) ResolvePadding(
        string algorithm,
        int[] paddings,
        int inHeight,
        int inWidth,
        int kernelHeight,
        int kernelWidth,
        int[] strides,
        int[] dilations)
    {
        switch (algorithm)
        {
            case "VALID":
                return (0, 0, 0, 0);

            case "SAME":
            {
                int neededHeight = Math.Max(
                    0,
                    ((((inHeight + strides[0] - 1) / strides[0]) - 1) * strides[0])
                        + (dilations[0] * (kernelHeight - 1)) + 1 - inHeight);
                int neededWidth = Math.Max(
                    0,
                    ((((inWidth + strides[1] - 1) / strides[1]) - 1) * strides[1])
                        + (dilations[1] * (kernelWidth - 1)) + 1 - inWidth);

                return (neededHeight / 2, neededHeight - (neededHeight / 2),
                    neededWidth / 2, neededWidth - (neededWidth / 2));
            }

            default:
                return paddings.Length switch
                {
                    2 => (paddings[0], paddings[0], paddings[1], paddings[1]),
                    4 => (paddings[0], paddings[1], paddings[2], paddings[3]),
                    1 => (paddings[0], paddings[0], paddings[0], paddings[0]),
                    _ => throw new NotSupportedException(
                        $"Unsupported padding array of length {paddings.Length}."),
                };
        }
    }
}
