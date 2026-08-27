using PaddleOcrSharp.Formats.Paddle;

namespace PaddleOcrSharp.Models.Paddle.Ops;

/// <summary>Shape-manipulating kernels: layout, indexing and assembly.</summary>
public static class ShapeOps
{
    /// <summary>Permutes the dimensions of <paramref name="input"/>.</summary>
    public static PaddleTensor Transpose(PaddleTensor input, int[] permutation)
    {
        int rank = input.Rank;
        int[] shape = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            shape[i] = input.Shape[permutation[i]];
        }

        PaddleTensor result = PaddleTensor.Allocate(shape, input.Dtype);
        if (result.Count == 0)
        {
            return result;
        }

        int[] sourceStrides = PaddleTensor.Strides(input.Shape);
        int[] mappedStrides = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            mappedStrides[i] = sourceStrides[permutation[i]];
        }

        int[] counters = new int[rank];
        int sourceOffset = 0;

        for (int i = 0; i < result.Count; i++)
        {
            if (input.IsFloat)
            {
                result.Floats![i] = input.Floats![sourceOffset];
            }
            else
            {
                result.Ints![i] = input.Ints![sourceOffset];
            }

            for (int axis = rank - 1; axis >= 0; axis--)
            {
                counters[axis]++;
                sourceOffset += mappedStrides[axis];
                if (counters[axis] < shape[axis])
                {
                    break;
                }

                sourceOffset -= mappedStrides[axis] * shape[axis];
                counters[axis] = 0;
            }
        }

        return result;
    }

    /// <summary>Merges dimensions <paramref name="start"/> through <paramref name="stop"/>.</summary>
    public static PaddleTensor Flatten(PaddleTensor input, int start, int stop)
    {
        int rank = input.Rank;
        start = start < 0 ? start + rank : start;
        stop = stop < 0 ? stop + rank : stop;

        var shape = new List<int>();
        for (int i = 0; i < start; i++)
        {
            shape.Add(input.Shape[i]);
        }

        int merged = 1;
        for (int i = start; i <= stop; i++)
        {
            merged *= input.Shape[i];
        }

        shape.Add(merged);
        for (int i = stop + 1; i < rank; i++)
        {
            shape.Add(input.Shape[i]);
        }

        return input.Reshaped([.. shape]);
    }

    /// <summary>Inserts size-1 dimensions at <paramref name="axes"/>.</summary>
    public static PaddleTensor Unsqueeze(PaddleTensor input, ReadOnlySpan<long> axes)
    {
        var shape = new List<int>(input.Shape);
        foreach (long axis in axes)
        {
            int index = (int)(axis < 0 ? axis + shape.Count + 1 : axis);
            shape.Insert(index, 1);
        }

        return input.Reshaped([.. shape]);
    }

    /// <summary>Removes size-1 dimensions at <paramref name="axes"/>, or all of them when empty.</summary>
    public static PaddleTensor Squeeze(PaddleTensor input, ReadOnlySpan<long> axes)
    {
        var shape = new List<int>(input.Shape);
        if (axes.Length == 0)
        {
            shape.RemoveAll(dimension => dimension == 1);
        }
        else
        {
            var toRemove = new SortedSet<int>();
            foreach (long axis in axes)
            {
                int index = (int)(axis < 0 ? axis + shape.Count : axis);
                if (shape[index] == 1)
                {
                    toRemove.Add(index);
                }
            }

            foreach (int index in toRemove.Reverse())
            {
                shape.RemoveAt(index);
            }
        }

        return input.Reshaped([.. shape]);
    }

    /// <summary>
    /// Extracts a strided range along <paramref name="axes"/>, matching <c>pd_op.slice</c>.
    /// </summary>
    public static PaddleTensor Slice(
        PaddleTensor input,
        long[] axes,
        long[] starts,
        long[] ends,
        long[] decreaseAxes)
    {
        int rank = input.Rank;
        int[] begin = new int[rank];
        int[] shape = [.. input.Shape];

        for (int i = 0; i < axes.Length; i++)
        {
            int axis = (int)(axes[i] < 0 ? axes[i] + rank : axes[i]);
            int size = input.Shape[axis];

            long start = starts[i] < 0 ? starts[i] + size : starts[i];
            long end = ends[i] < 0 ? ends[i] + size : ends[i];

            start = Math.Clamp(start, 0, size);
            end = Math.Clamp(end, 0, size);

            begin[axis] = (int)start;
            shape[axis] = (int)Math.Max(0, end - start);
        }

        PaddleTensor result = PaddleTensor.Allocate(shape, input.Dtype);
        if (result.Count > 0)
        {
            CopyWindow(input, begin, shape, result);
        }

        if (decreaseAxes.Length > 0)
        {
            var reduced = new List<int>(shape);
            foreach (int axis in decreaseAxes.Select(value => (int)value).OrderByDescending(value => value))
            {
                reduced.RemoveAt(axis);
            }

            return result.Reshaped([.. reduced]);
        }

        return result;
    }

    private static void CopyWindow(PaddleTensor input, int[] begin, int[] shape, PaddleTensor result)
    {
        int rank = input.Rank;
        int[] strides = PaddleTensor.Strides(input.Shape);

        int sourceOffset = 0;
        for (int i = 0; i < rank; i++)
        {
            sourceOffset += begin[i] * strides[i];
        }

        int[] counters = new int[rank];
        int offset = sourceOffset;

        for (int i = 0; i < result.Count; i++)
        {
            if (input.IsFloat)
            {
                result.Floats![i] = input.Floats![offset];
            }
            else
            {
                result.Ints![i] = input.Ints![offset];
            }

            for (int axis = rank - 1; axis >= 0; axis--)
            {
                counters[axis]++;
                offset += strides[axis];
                if (counters[axis] < shape[axis])
                {
                    break;
                }

                offset -= strides[axis] * shape[axis];
                counters[axis] = 0;
            }
        }
    }

    /// <summary>Joins tensors along <paramref name="axis"/>.</summary>
    public static PaddleTensor Concat(IReadOnlyList<PaddleTensor> inputs, int axis)
    {
        PaddleTensor first = inputs[0];
        int rank = first.Rank;
        axis = axis < 0 ? axis + rank : axis;

        int[] shape = [.. first.Shape];
        shape[axis] = inputs.Sum(tensor => tensor.Shape[axis]);

        PaddleTensor result = PaddleTensor.Allocate(shape, first.Dtype);

        int outer = 1;
        for (int i = 0; i < axis; i++)
        {
            outer *= shape[i];
        }

        int inner = 1;
        for (int i = axis + 1; i < rank; i++)
        {
            inner *= shape[i];
        }

        int destinationRow = shape[axis] * inner;
        int written = 0;

        foreach (PaddleTensor input in inputs)
        {
            int sourceRow = input.Shape[axis] * inner;
            for (int o = 0; o < outer; o++)
            {
                int source = o * sourceRow;
                int destination = (o * destinationRow) + written;
                if (input.IsFloat)
                {
                    input.Floats.AsSpan(source, sourceRow).CopyTo(result.Floats.AsSpan(destination));
                }
                else
                {
                    input.Ints.AsSpan(source, sourceRow).CopyTo(result.Ints.AsSpan(destination));
                }
            }

            written += sourceRow;
        }

        return result;
    }

    /// <summary>Stacks tensors along a new dimension at <paramref name="axis"/>.</summary>
    public static PaddleTensor Stack(IReadOnlyList<PaddleTensor> inputs, int axis)
    {
        PaddleTensor first = inputs[0];
        int rank = first.Rank + 1;
        axis = axis < 0 ? axis + rank : axis;

        var shape = new List<int>(first.Shape);
        shape.Insert(axis, inputs.Count);

        PaddleTensor result = PaddleTensor.Allocate([.. shape], first.Dtype);

        int inner = 1;
        for (int i = axis; i < first.Rank; i++)
        {
            inner *= first.Shape[i];
        }

        int outer = first.Count == 0 ? 0 : first.Count / Math.Max(1, inner);

        for (int index = 0; index < inputs.Count; index++)
        {
            PaddleTensor input = inputs[index];
            for (int o = 0; o < outer; o++)
            {
                int source = o * inner;
                int destination = ((o * inputs.Count) + index) * inner;
                if (input.IsFloat)
                {
                    input.Floats.AsSpan(source, inner).CopyTo(result.Floats.AsSpan(destination));
                }
                else
                {
                    input.Ints.AsSpan(source, inner).CopyTo(result.Ints.AsSpan(destination));
                }
            }
        }

        return result;
    }

    /// <summary>Splits <paramref name="input"/> into sections along <paramref name="axis"/>.</summary>
    public static PaddleTensor[] Split(PaddleTensor input, IReadOnlyList<int> sections, int axis)
    {
        int rank = input.Rank;
        axis = axis < 0 ? axis + rank : axis;

        var results = new PaddleTensor[sections.Count];
        long start = 0;

        for (int i = 0; i < sections.Count; i++)
        {
            results[i] = Slice(
                input,
                [axis],
                [start],
                [start + sections[i]],
                []);
            start += sections[i];
        }

        return results;
    }

    /// <summary>Repeats <paramref name="input"/> along each dimension.</summary>
    public static PaddleTensor Tile(PaddleTensor input, long[] repeats)
    {
        int rank = Math.Max(input.Rank, repeats.Length);
        int[] sourceShape = new int[rank];
        long[] counts = new long[rank];

        for (int i = 0; i < rank; i++)
        {
            int sourceIndex = i - (rank - input.Rank);
            sourceShape[i] = sourceIndex >= 0 ? input.Shape[sourceIndex] : 1;

            int repeatIndex = i - (rank - repeats.Length);
            counts[i] = repeatIndex >= 0 ? repeats[repeatIndex] : 1;
        }

        int[] shape = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            shape[i] = (int)(sourceShape[i] * counts[i]);
        }

        PaddleTensor result = PaddleTensor.Allocate(shape, input.Dtype);
        if (result.Count == 0)
        {
            return result;
        }

        int[] sourceStrides = PaddleTensor.Strides(sourceShape);
        int[] counters = new int[rank];

        for (int i = 0; i < result.Count; i++)
        {
            int offset = 0;
            for (int axis = 0; axis < rank; axis++)
            {
                offset += (counters[axis] % sourceShape[axis]) * sourceStrides[axis];
            }

            if (input.IsFloat)
            {
                result.Floats![i] = input.Floats![offset];
            }
            else
            {
                result.Ints![i] = input.Ints![offset];
            }

            for (int axis = rank - 1; axis >= 0; axis--)
            {
                if (++counters[axis] < shape[axis])
                {
                    break;
                }

                counters[axis] = 0;
            }
        }

        return result;
    }

    /// <summary>Broadcasts <paramref name="input"/> to <paramref name="shape"/>.</summary>
    public static PaddleTensor Expand(PaddleTensor input, long[] shape)
    {
        int rank = shape.Length;
        int[] target = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            int sourceIndex = i - (rank - input.Rank);
            int sourceDimension = sourceIndex >= 0 ? input.Shape[sourceIndex] : 1;
            target[i] = shape[i] == -1 ? sourceDimension : (int)shape[i];
        }

        long[] repeats = new long[rank];
        for (int i = 0; i < rank; i++)
        {
            int sourceIndex = i - (rank - input.Rank);
            int sourceDimension = sourceIndex >= 0 ? input.Shape[sourceIndex] : 1;
            repeats[i] = sourceDimension == 0 ? 0 : target[i] / sourceDimension;
        }

        return Tile(input, repeats).Reshaped(target);
    }

    /// <summary>Reverses <paramref name="input"/> along <paramref name="axes"/>.</summary>
    public static PaddleTensor Flip(PaddleTensor input, long[] axes)
    {
        PaddleTensor result = PaddleTensor.Allocate([.. input.Shape], input.Dtype);
        int rank = input.Rank;
        int[] strides = PaddleTensor.Strides(input.Shape);
        var flipped = axes.Select(axis => (int)(axis < 0 ? axis + rank : axis)).ToHashSet();
        int[] counters = new int[rank];

        for (int i = 0; i < result.Count; i++)
        {
            int offset = 0;
            for (int axis = 0; axis < rank; axis++)
            {
                int index = flipped.Contains(axis) ? input.Shape[axis] - 1 - counters[axis] : counters[axis];
                offset += index * strides[axis];
            }

            if (input.IsFloat)
            {
                result.Floats![i] = input.Floats![offset];
            }
            else
            {
                result.Ints![i] = input.Ints![offset];
            }

            for (int axis = rank - 1; axis >= 0; axis--)
            {
                if (++counters[axis] < input.Shape[axis])
                {
                    break;
                }

                counters[axis] = 0;
            }
        }

        return result;
    }

    /// <summary>
    /// Gathers slices addressed by the trailing dimension of <paramref name="indices"/>.
    /// </summary>
    public static PaddleTensor GatherNd(PaddleTensor input, PaddleTensor indices)
    {
        int indexDepth = indices.Shape[^1];
        int groups = indices.Count / Math.Max(1, indexDepth);

        int sliceSize = 1;
        for (int i = indexDepth; i < input.Rank; i++)
        {
            sliceSize *= input.Shape[i];
        }

        var shape = new List<int>();
        for (int i = 0; i < indices.Rank - 1; i++)
        {
            shape.Add(indices.Shape[i]);
        }

        for (int i = indexDepth; i < input.Rank; i++)
        {
            shape.Add(input.Shape[i]);
        }

        PaddleTensor result = PaddleTensor.Allocate([.. shape], input.Dtype);
        int[] strides = PaddleTensor.Strides(input.Shape);

        for (int g = 0; g < groups; g++)
        {
            int offset = 0;
            for (int d = 0; d < indexDepth; d++)
            {
                long index = indices.GetLong((g * indexDepth) + d);
                if (index < 0)
                {
                    index += input.Shape[d];
                }

                offset += (int)index * strides[d];
            }

            if (input.IsFloat)
            {
                input.Floats.AsSpan(offset, sliceSize).CopyTo(result.Floats.AsSpan(g * sliceSize));
            }
            else
            {
                input.Ints.AsSpan(offset, sliceSize).CopyTo(result.Ints.AsSpan(g * sliceSize));
            }
        }

        return result;
    }

    /// <summary>An identity matrix of <paramref name="rows"/> × <paramref name="columns"/>.</summary>
    public static PaddleTensor Eye(int rows, int columns, PaddleDType dtype)
    {
        PaddleTensor result = PaddleTensor.Zeros([rows, columns], dtype);
        for (int i = 0; i < Math.Min(rows, columns); i++)
        {
            int index = (i * columns) + i;
            if (result.IsFloat)
            {
                result.Floats![index] = 1f;
            }
            else
            {
                result.Ints![index] = 1;
            }
        }

        return result;
    }

    /// <summary>Cartesian index grids, as <c>paddle.meshgrid</c> produces them.</summary>
    public static PaddleTensor[] MeshGrid(IReadOnlyList<PaddleTensor> inputs)
    {
        int rank = inputs.Count;
        int[] shape = [.. inputs.Select(tensor => tensor.Count)];
        var results = new PaddleTensor[rank];

        for (int axis = 0; axis < rank; axis++)
        {
            PaddleTensor source = inputs[axis];
            PaddleTensor result = PaddleTensor.Allocate(shape, source.Dtype);
            int[] counters = new int[rank];

            for (int i = 0; i < result.Count; i++)
            {
                int index = counters[axis];
                if (source.IsFloat)
                {
                    result.Floats![i] = source.Floats![index];
                }
                else
                {
                    result.Ints![i] = source.Ints![index];
                }

                for (int a = rank - 1; a >= 0; a--)
                {
                    if (++counters[a] < shape[a])
                    {
                        break;
                    }

                    counters[a] = 0;
                }
            }

            results[axis] = result;
        }

        return results;
    }
}
