namespace PaddleOcrSharp.Models.Paddle.Ops;

/// <summary>Scatter-style kernels that write into a copy of their first operand.</summary>
internal static class IndexOps
{
    /// <summary>
    /// Writes <paramref name="value"/> into <paramref name="input"/> at the positions addressed by
    /// <paramref name="indices"/>, matching <c>pd_op.index_put</c>.
    /// </summary>
    /// <param name="input">Tensor to update; left untouched, a copy is returned.</param>
    /// <param name="indices">One index tensor per leading dimension, broadcast against each other.</param>
    /// <param name="value">Values to write, broadcast across the addressed positions.</param>
    /// <param name="accumulate">Whether to add to the existing values rather than replace them.</param>
    public static PaddleTensor IndexPut(
        PaddleTensor input,
        IReadOnlyList<PaddleTensor> indices,
        PaddleTensor value,
        bool accumulate)
    {
        PaddleTensor result = input.Clone();

        int[] shape = indices[0].Shape;
        foreach (PaddleTensor index in indices.Skip(1))
        {
            shape = Broadcast.ResultShape(shape, index.Shape);
        }

        int count = PaddleTensor.ElementCount(shape);
        int[][] strides = [.. indices.Select(index => Broadcast.StridesFor(index.Shape, shape))];
        int[] inputStrides = PaddleTensor.Strides(input.Shape);

        int tail = 1;
        for (int i = indices.Count; i < input.Rank; i++)
        {
            tail *= input.Shape[i];
        }

        int[] counters = new int[shape.Length];
        int[] offsets = new int[indices.Count];

        for (int i = 0; i < count; i++)
        {
            int target = 0;
            for (int d = 0; d < indices.Count; d++)
            {
                long index = indices[d].GetLong(offsets[d]);
                if (index < 0)
                {
                    index += input.Shape[d];
                }

                target += (int)index * inputStrides[d];
            }

            for (int t = 0; t < tail; t++)
            {
                int source = value.Count == 1 ? 0 : ((i * tail) + t) % value.Count;
                if (result.IsFloat)
                {
                    float incoming = (float)value.GetDouble(source);
                    result.Floats![target + t] = accumulate ? result.Floats[target + t] + incoming : incoming;
                }
                else
                {
                    long incoming = value.GetLong(source);
                    result.Ints![target + t] = accumulate ? result.Ints[target + t] + incoming : incoming;
                }
            }

            for (int axis = shape.Length - 1; axis >= 0; axis--)
            {
                counters[axis]++;
                for (int d = 0; d < indices.Count; d++)
                {
                    offsets[d] += strides[d][axis];
                }

                if (counters[axis] < shape[axis])
                {
                    break;
                }

                for (int d = 0; d < indices.Count; d++)
                {
                    offsets[d] -= strides[d][axis] * shape[axis];
                }

                counters[axis] = 0;
            }
        }

        return result;
    }

    /// <summary>
    /// Writes <paramref name="value"/> into a strided window of <paramref name="input"/>, matching
    /// <c>pd_op.set_value_with_tensor_</c>.
    /// </summary>
    /// <remarks>
    /// With no axes given the assignment covers the whole tensor, which is how the exported graph
    /// uses it. The <c>decrease_axes</c> and <c>none_axes</c> arguments only reshape the value
    /// operand and are handled by the broadcast of <paramref name="value"/>.
    /// </remarks>
    public static PaddleTensor SetValue(
        PaddleTensor input,
        PaddleTensor value,
        long[] starts,
        long[] ends,
        long[] steps,
        long[] axes,
        long[] decreaseAxes,
        long[] noneAxes)
    {
        PaddleTensor result = input.Clone();

        if (axes.Length == 0)
        {
            axes = [.. Enumerable.Range(0, Math.Min(starts.Length, input.Rank)).Select(i => (long)i)];
        }

        int rank = input.Rank;
        int[] begin = new int[rank];
        int[] extent = [.. input.Shape];
        int[] stride = new int[rank];
        Array.Fill(stride, 1);

        for (int i = 0; i < axes.Length; i++)
        {
            int axis = (int)(axes[i] < 0 ? axes[i] + rank : axes[i]);
            int size = input.Shape[axis];

            long start = i < starts.Length ? starts[i] : 0;
            long end = i < ends.Length ? ends[i] : size;
            long step = i < steps.Length ? steps[i] : 1;

            start = Math.Clamp(start < 0 ? start + size : start, 0, size);
            end = Math.Clamp(end < 0 ? end + size : end, 0, size);

            begin[axis] = (int)start;
            stride[axis] = (int)Math.Max(1, step);
            extent[axis] = (int)Math.Max(0, ((end - start) + stride[axis] - 1) / stride[axis]);
        }

        int count = PaddleTensor.ElementCount(extent);
        int[] inputStrides = PaddleTensor.Strides(input.Shape);
        int[] counters = new int[rank];

        for (int i = 0; i < count; i++)
        {
            int target = 0;
            for (int axis = 0; axis < rank; axis++)
            {
                target += (begin[axis] + (counters[axis] * stride[axis])) * inputStrides[axis];
            }

            int source = value.Count == 1 ? 0 : i % value.Count;
            if (result.IsFloat)
            {
                result.Floats![target] = (float)value.GetDouble(source);
            }
            else
            {
                result.Ints![target] = value.GetLong(source);
            }

            for (int axis = rank - 1; axis >= 0; axis--)
            {
                if (++counters[axis] < extent[axis])
                {
                    break;
                }

                counters[axis] = 0;
            }
        }

        _ = decreaseAxes;
        _ = noneAxes;
        return result;
    }
}
