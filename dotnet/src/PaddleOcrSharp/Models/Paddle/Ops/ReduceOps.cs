using PaddleOcrSharp.Formats.Paddle;

namespace PaddleOcrSharp.Models.Paddle.Ops;

/// <summary>Reductions, sorting and selection.</summary>
public static class ReduceOps
{
    /// <summary>How a reduction combines values.</summary>
    public enum Kind
    {
        /// <summary>Sum of the reduced elements.</summary>
        Sum,

        /// <summary>Largest of the reduced elements.</summary>
        Max,

        /// <summary>Smallest of the reduced elements.</summary>
        Min,

        /// <summary>Whether any reduced element is non-zero.</summary>
        Any,
    }

    /// <summary>Reduces <paramref name="input"/> along <paramref name="axes"/>.</summary>
    /// <param name="input">Tensor to reduce.</param>
    /// <param name="axes">Axes to reduce; empty means all of them.</param>
    /// <param name="keepDimension">Whether reduced axes stay as size-1 dimensions.</param>
    /// <param name="kind">Reduction to apply.</param>
    public static PaddleTensor Reduce(PaddleTensor input, long[] axes, bool keepDimension, Kind kind)
    {
        int rank = input.Rank;
        var reduced = new HashSet<int>(
            axes.Length == 0
                ? Enumerable.Range(0, rank)
                : axes.Select(axis => (int)(axis < 0 ? axis + rank : axis)));

        var shape = new List<int>();
        for (int i = 0; i < rank; i++)
        {
            if (!reduced.Contains(i))
            {
                shape.Add(input.Shape[i]);
            }
            else if (keepDimension)
            {
                shape.Add(1);
            }
        }

        PaddleDType dtype = kind == Kind.Any ? PaddleDType.Bool : input.Dtype;
        PaddleTensor result = PaddleTensor.Allocate([.. shape], dtype);

        double seed = kind switch
        {
            Kind.Sum or Kind.Any => 0,
            Kind.Max => double.NegativeInfinity,
            _ => double.PositiveInfinity,
        };

        double[] accumulator = new double[result.Count];
        Array.Fill(accumulator, seed);

        int[] strides = PaddleTensor.Strides(input.Shape);
        int[] outputStrides = new int[rank];
        int stride = 1;
        for (int i = rank - 1; i >= 0; i--)
        {
            if (reduced.Contains(i))
            {
                outputStrides[i] = 0;
            }
            else
            {
                outputStrides[i] = stride;
                stride *= input.Shape[i];
            }
        }

        int[] counters = new int[rank];
        for (int i = 0; i < input.Count; i++)
        {
            int target = 0;
            for (int axis = 0; axis < rank; axis++)
            {
                target += counters[axis] * outputStrides[axis];
            }

            double value = input.GetDouble(i);
            accumulator[target] = kind switch
            {
                Kind.Sum => accumulator[target] + value,
                Kind.Max => Math.Max(accumulator[target], value),
                Kind.Min => Math.Min(accumulator[target], value),
                _ => (accumulator[target] != 0 || value != 0) ? 1 : 0,
            };

            for (int axis = rank - 1; axis >= 0; axis--)
            {
                if (++counters[axis] < input.Shape[axis])
                {
                    break;
                }

                counters[axis] = 0;
            }
        }

        _ = strides;

        for (int i = 0; i < result.Count; i++)
        {
            if (result.IsFloat)
            {
                result.Floats![i] = (float)accumulator[i];
            }
            else
            {
                result.Ints![i] = (long)accumulator[i];
            }
        }

        return result;
    }

    /// <summary>
    /// Top-<paramref name="k"/> values and their indices along <paramref name="axis"/>.
    /// </summary>
    public static (PaddleTensor Values, PaddleTensor Indices) TopK(
        PaddleTensor input,
        int k,
        int axis,
        bool largest,
        bool sorted)
    {
        int rank = input.Rank;
        axis = axis < 0 ? axis + rank : axis;

        int[] shape = [.. input.Shape];
        shape[axis] = k;

        PaddleTensor values = PaddleTensor.Allocate(shape, input.Dtype);
        PaddleTensor indices = PaddleTensor.Int(shape);

        int width = input.Shape[axis];
        int inner = 1;
        for (int i = axis + 1; i < rank; i++)
        {
            inner *= input.Shape[i];
        }

        int outer = width == 0 ? 0 : input.Count / (width * inner);
        int[] order = new int[width];

        for (int o = 0; o < outer; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                int sourceBase = (o * width * inner) + i;
                for (int w = 0; w < width; w++)
                {
                    order[w] = w;
                }

                // Paddle's top-k is stable: ties keep their original order.
                int[] ranked = [.. order];
                Array.Sort(ranked, (left, right) =>
                {
                    double a = input.GetDouble(sourceBase + (left * inner));
                    double b = input.GetDouble(sourceBase + (right * inner));
                    int comparison = largest ? b.CompareTo(a) : a.CompareTo(b);
                    return comparison != 0 ? comparison : left.CompareTo(right);
                });

                int targetBase = (o * k * inner) + i;
                for (int j = 0; j < k; j++)
                {
                    int source = sourceBase + (ranked[j] * inner);
                    int target = targetBase + (j * inner);
                    if (values.IsFloat)
                    {
                        values.Floats![target] = input.Floats![source];
                    }
                    else
                    {
                        values.Ints![target] = input.Ints![source];
                    }

                    indices.Ints![target] = ranked[j];
                }
            }
        }

        _ = sorted;
        return (values, indices);
    }

    /// <summary>Sorted values and their indices along <paramref name="axis"/>.</summary>
    public static (PaddleTensor Values, PaddleTensor Indices) ArgSort(
        PaddleTensor input,
        int axis,
        bool descending,
        bool stable)
    {
        int rank = input.Rank;
        axis = axis < 0 ? axis + rank : axis;
        int width = input.Shape[axis];
        (PaddleTensor values, PaddleTensor indices) = TopK(input, width, axis, descending, sorted: true);
        _ = stable;
        return (values, indices);
    }

    /// <summary>Element-wise selection: <c>condition ? x : y</c>, with broadcasting.</summary>
    public static PaddleTensor Where(PaddleTensor condition, PaddleTensor x, PaddleTensor y)
    {
        int[] shape = Broadcast.ResultShape(Broadcast.ResultShape(condition.Shape, x.Shape), y.Shape);
        PaddleTensor result = PaddleTensor.Allocate(shape, x.Dtype);

        int[] conditionStrides = Broadcast.StridesFor(condition.Shape, shape);
        int[] xStrides = Broadcast.StridesFor(x.Shape, shape);
        int[] yStrides = Broadcast.StridesFor(y.Shape, shape);
        int[] counters = new int[shape.Length];

        int conditionOffset = 0;
        int xOffset = 0;
        int yOffset = 0;

        for (int i = 0; i < result.Count; i++)
        {
            bool take = condition.GetLong(conditionOffset) != 0;
            if (result.IsFloat)
            {
                result.Floats![i] = take ? x.Floats![xOffset] : y.Floats![yOffset];
            }
            else
            {
                result.Ints![i] = take ? x.GetLong(xOffset) : y.GetLong(yOffset);
            }

            for (int axis = shape.Length - 1; axis >= 0; axis--)
            {
                counters[axis]++;
                conditionOffset += conditionStrides[axis];
                xOffset += xStrides[axis];
                yOffset += yStrides[axis];

                if (counters[axis] < shape[axis])
                {
                    break;
                }

                conditionOffset -= conditionStrides[axis] * shape[axis];
                xOffset -= xStrides[axis] * shape[axis];
                yOffset -= yStrides[axis] * shape[axis];
                counters[axis] = 0;
            }
        }

        return result;
    }
}
