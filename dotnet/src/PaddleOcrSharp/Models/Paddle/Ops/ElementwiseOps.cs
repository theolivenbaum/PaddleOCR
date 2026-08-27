using System.Numerics.Tensors;
using PaddleOcrSharp.Core;
using PaddleOcrSharp.Formats.Paddle;

namespace PaddleOcrSharp.Models.Paddle.Ops;

/// <summary>The element-wise operators the graph uses, with vectorised paths for float32.</summary>
/// <remarks>
/// A layout forward pass runs 192 additions, 81 ReLUs and 36 multiplies over feature maps with
/// millions of elements. Routing those through a per-element delegate costs more than the
/// convolutions do, so the shapes that actually occur — equal shapes, a scalar operand, and a
/// trailing-dimension broadcast — get dedicated kernels and everything else falls back to the
/// generic broadcast walk.
/// </remarks>
internal static class ElementwiseOps
{
    /// <summary>Which binary operation to apply.</summary>
    public enum Binary
    {
        /// <summary>Addition.</summary>
        Add,

        /// <summary>Subtraction.</summary>
        Subtract,

        /// <summary>Multiplication.</summary>
        Multiply,

        /// <summary>Division.</summary>
        Divide,
    }

    /// <summary>Which unary activation to apply.</summary>
    public enum Unary
    {
        /// <summary>Rectified linear unit.</summary>
        Relu,

        /// <summary>Logistic sigmoid.</summary>
        Sigmoid,

        /// <summary>Sigmoid-weighted linear unit.</summary>
        Silu,

        /// <summary>Exact (erf-based) GELU.</summary>
        GeluErf,

        /// <summary>Tanh-approximated GELU.</summary>
        GeluTanh,

        /// <summary>Natural logarithm.</summary>
        Log,

        /// <summary>Floor.</summary>
        Floor,

        /// <summary>Hard swish: <c>x · clamp(x + 3, 0, 6) / 6</c>.</summary>
        HardSwish,
    }

    /// <summary>Applies <paramref name="operation"/> element-wise with broadcasting.</summary>
    public static PaddleTensor Apply(PaddleTensor left, PaddleTensor right, Binary operation)
    {
        if (left.IsFloat && right.IsFloat && TryFast(left, right, operation, out PaddleTensor? fast))
        {
            return fast;
        }

        return Broadcast.Apply(left, right, operation switch
        {
            Binary.Add => static (a, b) => a + b,
            Binary.Subtract => static (a, b) => a - b,
            Binary.Multiply => static (a, b) => a * b,
            _ => static (a, b) => a / b,
        });
    }

    /// <summary>Applies <paramref name="activation"/> element-wise.</summary>
    public static PaddleTensor Apply(PaddleTensor input, Unary activation)
    {
        if (!input.IsFloat)
        {
            return SlowUnary(input, activation);
        }

        PaddleTensor result = PaddleTensor.Float([.. input.Shape]);
        ReadOnlySpan<float> source = input.FloatSpan;
        Span<float> destination = result.FloatSpan;

        switch (activation)
        {
            case Unary.Relu:
                TensorPrimitives.Max(source, 0f, destination);
                break;
            case Unary.Sigmoid:
                TensorPrimitives.Sigmoid(source, destination);
                break;
            case Unary.Silu:
                source.CopyTo(destination);
                Kernels.Silu(destination);
                break;
            case Unary.GeluTanh:
                source.CopyTo(destination);
                Kernels.GeluTanh(destination);
                break;
            case Unary.GeluErf:
                source.CopyTo(destination);
                Kernels.GeluErf(destination);
                break;
            case Unary.Log:
                TensorPrimitives.Log(source, destination);
                break;
            case Unary.HardSwish:
                for (int i = 0; i < source.Length; i++)
                {
                    destination[i] = source[i] * Math.Clamp(source[i] + 3f, 0f, 6f) * (1f / 6f);
                }

                break;
            default:
                for (int i = 0; i < source.Length; i++)
                {
                    destination[i] = MathF.Floor(source[i]);
                }

                break;
        }

        return result;
    }

    /// <summary>Computes <c>x · scale + bias</c>, or <c>(x + bias) · scale</c>.</summary>
    public static PaddleTensor Scale(PaddleTensor input, double scale, double bias, bool biasAfterScale)
    {
        if (!input.IsFloat)
        {
            PaddleTensor integers = PaddleTensor.Allocate([.. input.Shape], input.Dtype);
            ReadOnlySpan<long> source = input.IntSpan;
            Span<long> destination = integers.IntSpan;
            for (int i = 0; i < source.Length; i++)
            {
                destination[i] = (long)(biasAfterScale ? (source[i] * scale) + bias : (source[i] + bias) * scale);
            }

            return integers;
        }

        PaddleTensor result = PaddleTensor.Float([.. input.Shape]);
        ReadOnlySpan<float> values = input.FloatSpan;
        Span<float> output = result.FloatSpan;

        float factor = (float)scale;
        float offset = (float)bias;

        if (biasAfterScale)
        {
            TensorPrimitives.Multiply(values, factor, output);
            if (offset != 0f)
            {
                TensorPrimitives.Add(output, offset, output);
            }
        }
        else
        {
            TensorPrimitives.Add(values, offset, output);
            TensorPrimitives.Multiply(output, factor, output);
        }

        return result;
    }

    /// <summary>Clamps every element into <c>[low, high]</c>.</summary>
    public static PaddleTensor Clip(PaddleTensor input, double low, double high)
    {
        if (!input.IsFloat)
        {
            PaddleTensor integers = PaddleTensor.Allocate([.. input.Shape], input.Dtype);
            ReadOnlySpan<long> source = input.IntSpan;
            Span<long> destination = integers.IntSpan;
            for (int i = 0; i < source.Length; i++)
            {
                destination[i] = Math.Clamp(source[i], (long)low, (long)high);
            }

            return integers;
        }

        PaddleTensor result = PaddleTensor.Float([.. input.Shape]);
        ReadOnlySpan<float> values = input.FloatSpan;
        Span<float> output = result.FloatSpan;

        TensorPrimitives.Max(values, (float)low, output);
        TensorPrimitives.Min(output, (float)high, output);
        return result;
    }

    /// <summary>Hard sigmoid: <c>clamp(slope · x + offset, 0, 1)</c>.</summary>
    public static PaddleTensor HardSigmoid(PaddleTensor input, float slope, float offset)
    {
        PaddleTensor result = PaddleTensor.Float([.. input.Shape]);
        ReadOnlySpan<float> source = input.FloatSpan;
        Span<float> destination = result.FloatSpan;

        TensorPrimitives.Multiply(source, slope, destination);
        TensorPrimitives.Add(destination, offset, destination);
        TensorPrimitives.Max(destination, 0f, destination);
        TensorPrimitives.Min(destination, 1f, destination);
        return result;
    }

    /// <summary>Parametric ReLU: <c>max(0, x) + alpha · min(0, x)</c>.</summary>
    /// <param name="input">The activations.</param>
    /// <param name="alpha">One slope (<c>mode = "all"</c>), one per channel, or one per element.</param>
    /// <param name="mode">Paddle's <c>mode</c> attribute: <c>all</c>, <c>channel</c> or <c>element</c>.</param>
    /// <param name="channelsLast">Whether the layout is channels-last rather than <c>NCHW</c>.</param>
    public static PaddleTensor PRelu(
        PaddleTensor input,
        PaddleTensor alpha,
        string mode,
        bool channelsLast)
    {
        PaddleTensor result = PaddleTensor.Float([.. input.Shape]);
        ReadOnlySpan<float> source = input.FloatSpan;
        ReadOnlySpan<float> slopes = alpha.FloatSpan;
        Span<float> destination = result.FloatSpan;

        switch (mode)
        {
            case "all":
            {
                float slope = slopes[0];
                for (int i = 0; i < source.Length; i++)
                {
                    destination[i] = source[i] >= 0f ? source[i] : slope * source[i];
                }

                break;
            }

            case "channel":
            {
                int channelAxis = channelsLast ? input.Rank - 1 : 1;
                int inner = 1;
                for (int axis = channelAxis + 1; axis < input.Rank; axis++)
                {
                    inner *= input.Shape[axis];
                }

                int channels = input.Rank > 1 ? input.Shape[channelAxis] : 1;
                for (int i = 0; i < source.Length; i++)
                {
                    float slope = slopes[(i / inner) % channels];
                    destination[i] = source[i] >= 0f ? source[i] : slope * source[i];
                }

                break;
            }

            default:
            {
                // "element": one slope per element of a single sample, broadcast over the batch.
                int stride = slopes.Length;
                for (int i = 0; i < source.Length; i++)
                {
                    float slope = slopes[i % stride];
                    destination[i] = source[i] >= 0f ? source[i] : slope * source[i];
                }

                break;
            }
        }

        return result;
    }

    private static bool TryFast(
        PaddleTensor left,
        PaddleTensor right,
        Binary operation,
        out PaddleTensor result)
    {
        // Same shape: a straight vectorised pass.
        if (left.Count == right.Count && left.Shape.AsSpan().SequenceEqual(right.Shape))
        {
            result = PaddleTensor.Float([.. left.Shape]);
            Run(left.FloatSpan, right.FloatSpan, result.FloatSpan, operation);
            return true;
        }

        // Scalar on either side.
        if (right.Count == 1)
        {
            result = PaddleTensor.Float([.. left.Shape]);
            RunScalar(left.FloatSpan, right.FloatSpan[0], result.FloatSpan, operation, scalarOnRight: true);
            return true;
        }

        if (left.Count == 1)
        {
            result = PaddleTensor.Float([.. right.Shape]);
            RunScalar(right.FloatSpan, left.FloatSpan[0], result.FloatSpan, operation, scalarOnRight: false);
            return true;
        }

        // Trailing-dimension broadcast, e.g. [N, C, H, W] against [1, C, 1, 1] or [..., W].
        int[] shape = Broadcast.ResultShape(left.Shape, right.Shape);
        int count = PaddleTensor.ElementCount(shape);

        if (left.Count == count && right.Count > 0 && count % right.Count == 0 && IsTrailingBlock(right, shape))
        {
            result = PaddleTensor.Float(shape);
            RunTiled(left.FloatSpan, right.FloatSpan, result.FloatSpan, operation, rightIsSecond: true);
            return true;
        }

        if (right.Count == count && left.Count > 0 && count % left.Count == 0 && IsTrailingBlock(left, shape))
        {
            result = PaddleTensor.Float(shape);
            RunTiled(right.FloatSpan, left.FloatSpan, result.FloatSpan, operation, rightIsSecond: false);
            return true;
        }

        result = null!;
        return false;
    }

    /// <summary>
    /// Whether <paramref name="tensor"/> aligns with the trailing dimensions of
    /// <paramref name="shape"/>, so it can be applied as a repeating tile.
    /// </summary>
    private static bool IsTrailingBlock(PaddleTensor tensor, int[] shape)
    {
        int offset = shape.Length - tensor.Rank;
        if (offset < 0)
        {
            return false;
        }

        for (int i = 0; i < tensor.Rank; i++)
        {
            if (tensor.Shape[i] != shape[offset + i])
            {
                return false;
            }
        }

        return true;
    }

    private static void Run(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> y, Binary operation)
    {
        switch (operation)
        {
            case Binary.Add:
                TensorPrimitives.Add(a, b, y);
                break;
            case Binary.Subtract:
                TensorPrimitives.Subtract(a, b, y);
                break;
            case Binary.Multiply:
                TensorPrimitives.Multiply(a, b, y);
                break;
            default:
                TensorPrimitives.Divide(a, b, y);
                break;
        }
    }

    private static void RunScalar(
        ReadOnlySpan<float> values,
        float scalar,
        Span<float> y,
        Binary operation,
        bool scalarOnRight)
    {
        switch (operation)
        {
            case Binary.Add:
                TensorPrimitives.Add(values, scalar, y);
                break;
            case Binary.Multiply:
                TensorPrimitives.Multiply(values, scalar, y);
                break;
            case Binary.Subtract when scalarOnRight:
                TensorPrimitives.Subtract(values, scalar, y);
                break;
            case Binary.Subtract:
                TensorPrimitives.Negate(values, y);
                TensorPrimitives.Add(y, scalar, y);
                break;
            case Binary.Divide when scalarOnRight:
                TensorPrimitives.Multiply(values, 1f / scalar, y);
                break;
            default:
                for (int i = 0; i < values.Length; i++)
                {
                    y[i] = scalar / values[i];
                }

                break;
        }
    }

    private static void RunTiled(
        ReadOnlySpan<float> big,
        ReadOnlySpan<float> tile,
        Span<float> y,
        Binary operation,
        bool rightIsSecond)
    {
        int block = tile.Length;
        for (int offset = 0; offset < big.Length; offset += block)
        {
            Span<float> target = y.Slice(offset, block);
            if (rightIsSecond)
            {
                Run(big.Slice(offset, block), tile, target, operation);
            }
            else
            {
                Run(tile, big.Slice(offset, block), target, operation);
            }
        }
    }

    private static PaddleTensor SlowUnary(PaddleTensor input, Unary activation)
    {
        PaddleTensor result = PaddleTensor.Allocate([.. input.Shape], input.Dtype);
        ReadOnlySpan<long> source = input.IntSpan;
        Span<long> destination = result.IntSpan;

        for (int i = 0; i < source.Length; i++)
        {
            double value = source[i];
            destination[i] = (long)(activation switch
            {
                Unary.Relu => Math.Max(0, value),
                Unary.Sigmoid => 1.0 / (1.0 + Math.Exp(-value)),
                Unary.Silu => value / (1.0 + Math.Exp(-value)),
                Unary.Log => Math.Log(value),
                Unary.HardSwish => value * Math.Clamp(value + 3.0, 0.0, 6.0) / 6.0,
                Unary.Floor => Math.Floor(value),
                _ => value,
            });
        }

        return result;
    }
}
