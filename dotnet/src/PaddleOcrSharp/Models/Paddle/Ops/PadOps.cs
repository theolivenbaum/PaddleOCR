namespace PaddleOcrSharp.Models.Paddle.Ops;

/// <summary>Spatial padding, as used by <c>pad3d</c>.</summary>
/// <remarks>
/// Paddle's <c>pad3d</c> takes six paddings ordered
/// <c>(left, right, top, bottom, front, back)</c> — width first, then height, then depth — and
/// applies them to the last three spatial axes of an <c>NCDHW</c> (or <c>NDHWC</c>) tensor. UVDoc
/// reaches it through the usual trick of unsqueezing an <c>NCHW</c> feature map to <c>NC1HW</c>
/// so that a reflect-padded 2-D convolution can be expressed with the 3-D operator.
/// </remarks>
internal static class PadOps
{
    /// <summary>How to fill the padded border.</summary>
    public enum Mode
    {
        /// <summary>Fill with a constant.</summary>
        Constant,

        /// <summary>Mirror across the border without repeating the edge element.</summary>
        Reflect,

        /// <summary>Repeat the edge element.</summary>
        Replicate,

        /// <summary>Wrap around to the opposite edge.</summary>
        Circular,
    }

    /// <summary>Parses Paddle's <c>mode</c> attribute.</summary>
    public static Mode ParseMode(string mode) => mode switch
    {
        "constant" => Mode.Constant,
        "reflect" => Mode.Reflect,
        "replicate" => Mode.Replicate,
        "circular" => Mode.Circular,
        _ => throw new NotSupportedException($"Paddle padding mode '{mode}' is not implemented."),
    };

    /// <summary>Applies <c>pad3d</c> to a rank-5 tensor.</summary>
    /// <param name="input">The tensor to pad.</param>
    /// <param name="paddings">Six values ordered <c>(left, right, top, bottom, front, back)</c>.</param>
    /// <param name="mode">How to fill the border.</param>
    /// <param name="value">The fill value for <see cref="Mode.Constant"/>.</param>
    /// <param name="channelsLast">Whether the layout is <c>NDHWC</c> rather than <c>NCDHW</c>.</param>
    public static PaddleTensor Pad3d(
        PaddleTensor input,
        ReadOnlySpan<int> paddings,
        Mode mode,
        float value,
        bool channelsLast)
    {
        if (input.Rank != 5)
        {
            throw new NotSupportedException($"pad3d expects a rank-5 tensor but got rank {input.Rank}.");
        }

        if (paddings.Length != 6)
        {
            throw new NotSupportedException($"pad3d expects six paddings but got {paddings.Length}.");
        }

        // (left, right) is the innermost spatial axis, (front, back) the outermost.
        int spatial = channelsLast ? 1 : 2;
        Span<int> before = stackalloc int[5];
        Span<int> after = stackalloc int[5];

        for (int axis = 0; axis < 3; axis++)
        {
            before[spatial + axis] = paddings[(2 - axis) * 2];
            after[spatial + axis] = paddings[((2 - axis) * 2) + 1];
        }

        return Pad(input, before, after, mode, value);
    }

    /// <summary>Applies per-axis padding to a tensor of any rank.</summary>
    public static PaddleTensor Pad(
        PaddleTensor input,
        ReadOnlySpan<int> before,
        ReadOnlySpan<int> after,
        Mode mode,
        float value)
    {
        int rank = input.Rank;
        int[] shape = new int[rank];
        for (int axis = 0; axis < rank; axis++)
        {
            shape[axis] = input.Shape[axis] + before[axis] + after[axis];
            if (shape[axis] <= 0)
            {
                throw new NotSupportedException($"Padding axis {axis} to {shape[axis]} elements is not valid.");
            }
        }

        PaddleTensor result = PaddleTensor.Float(shape);
        ReadOnlySpan<float> source = input.FloatSpan;
        Span<float> destination = result.FloatSpan;

        // Strides of the *input*, so a walk over the output can gather directly.
        Span<int> strides = stackalloc int[rank];
        int stride = 1;
        for (int axis = rank - 1; axis >= 0; axis--)
        {
            strides[axis] = stride;
            stride *= input.Shape[axis];
        }

        Span<int> index = stackalloc int[rank];
        int count = result.Count;

        for (int flat = 0; flat < count; flat++)
        {
            int remainder = flat;
            for (int axis = rank - 1; axis >= 0; axis--)
            {
                index[axis] = remainder % shape[axis];
                remainder /= shape[axis];
            }

            int offset = 0;
            bool outside = false;

            for (int axis = 0; axis < rank && !outside; axis++)
            {
                int coordinate = Map(index[axis] - before[axis], input.Shape[axis], mode);
                if (coordinate < 0)
                {
                    outside = true;
                    break;
                }

                offset += coordinate * strides[axis];
            }

            destination[flat] = outside ? value : source[offset];
        }

        return result;
    }

    /// <summary>Maps a padded coordinate back into the source, or returns -1 for constant fill.</summary>
    private static int Map(int coordinate, int length, Mode mode)
    {
        if (coordinate >= 0 && coordinate < length)
        {
            return coordinate;
        }

        switch (mode)
        {
            case Mode.Replicate:
                return Math.Clamp(coordinate, 0, length - 1);

            case Mode.Circular:
                return ((coordinate % length) + length) % length;

            case Mode.Reflect:
                if (length == 1)
                {
                    return 0;
                }

                // Mirror repeatedly, which is what Paddle does for paddings wider than the axis.
                while (coordinate < 0 || coordinate >= length)
                {
                    coordinate = coordinate < 0 ? -coordinate : (2 * (length - 1)) - coordinate;
                }

                return coordinate;

            default:
                return -1;
        }
    }
}
