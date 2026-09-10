using PaddleOcrSharp.Core;
using PaddleOcrSharp.Formats.Paddle;

namespace PaddleOcrSharp.Models.Paddle.Ops;

/// <summary>NumPy-style broadcasting shared by the element-wise kernels.</summary>
internal static class Broadcast
{
    /// <summary>
    /// Computes the broadcast result shape of <paramref name="left"/> and <paramref name="right"/>.
    /// </summary>
    public static int[] ResultShape(ReadOnlySpan<int> left, ReadOnlySpan<int> right)
    {
        int rank = Math.Max(left.Length, right.Length);
        int[] shape = new int[rank];

        for (int i = 0; i < rank; i++)
        {
            int l = i < rank - left.Length ? 1 : left[i - (rank - left.Length)];
            int r = i < rank - right.Length ? 1 : right[i - (rank - right.Length)];

            if (l != r && l != 1 && r != 1)
            {
                throw new InvalidOperationException(
                    $"Cannot broadcast [{string.Join(",", left.ToArray())}] with " +
                    $"[{string.Join(",", right.ToArray())}].");
            }

            shape[i] = Math.Max(l, r);
        }

        return shape;
    }

    /// <summary>
    /// Strides for walking <paramref name="shape"/> while producing <paramref name="target"/>,
    /// with zero strides on broadcast dimensions.
    /// </summary>
    public static int[] StridesFor(ReadOnlySpan<int> shape, ReadOnlySpan<int> target)
    {
        int rank = target.Length;
        int[] strides = new int[rank];
        int stride = 1;

        for (int i = shape.Length - 1; i >= 0; i--)
        {
            int targetIndex = i + (rank - shape.Length);
            strides[targetIndex] = shape[i] == 1 && target[targetIndex] != 1 ? 0 : stride;
            stride *= shape[i];
        }

        return strides;
    }

    /// <summary>
    /// Seeds a walk's per-axis counters so it can start at <paramref name="index"/> rather than at
    /// zero, which is what lets these walks be split across threads.
    /// </summary>
    /// <param name="index">Linear index into the result.</param>
    /// <param name="shape">The result shape being walked.</param>
    /// <param name="counters">Receives the per-axis position of <paramref name="index"/>.</param>
    public static void Seed(int index, int[] shape, int[] counters)
    {
        for (int axis = shape.Length - 1; axis >= 0; axis--)
        {
            counters[axis] = shape[axis] == 0 ? 0 : index % shape[axis];
            index = shape[axis] == 0 ? 0 : index / shape[axis];
        }
    }

    /// <summary>An operand's offset at the position <paramref name="counters"/> names.</summary>
    /// <param name="counters">Per-axis position, from <see cref="Seed"/>.</param>
    /// <param name="strides">The operand's strides, from <see cref="StridesFor"/>.</param>
    /// <returns>The operand's linear offset.</returns>
    public static int Offset(int[] counters, int[] strides)
    {
        int offset = 0;
        for (int axis = 0; axis < counters.Length; axis++)
        {
            offset += counters[axis] * strides[axis];
        }

        return offset;
    }

    /// <summary>
    /// Applies <paramref name="operation"/> element-wise with broadcasting over two float tensors.
    /// </summary>
    public static PaddleTensor Apply(PaddleTensor left, PaddleTensor right, Func<double, double, double> operation)
    {
        int[] shape = ResultShape(left.Shape, right.Shape);
        bool floatResult = left.IsFloat || right.IsFloat;
        PaddleDType dtype = floatResult
            ? PaddleDType.Float32
            : (left.Dtype == PaddleDType.Int64 || right.Dtype == PaddleDType.Int64
                ? PaddleDType.Int64
                : PaddleDType.Int32);

        PaddleTensor result = PaddleTensor.Allocate(shape, dtype);
        Iterate(left, right, shape, (index, l, r) =>
        {
            double value = operation(l, r);
            if (floatResult)
            {
                result.Floats![index] = (float)value;
            }
            else
            {
                result.Ints![index] = (long)value;
            }
        });

        return result;
    }

    /// <summary>
    /// Applies a predicate element-wise with broadcasting, producing a boolean tensor.
    /// </summary>
    public static PaddleTensor Compare(PaddleTensor left, PaddleTensor right, Func<double, double, bool> predicate)
    {
        int[] shape = ResultShape(left.Shape, right.Shape);
        PaddleTensor result = PaddleTensor.Int(shape, PaddleDType.Bool);
        int count = result.Count;

        // Comparing a whole feature map against a threshold is the shape that actually shows up
        // (a mask logit against zero, over twelve million elements), so it skips the per-element
        // delegate and the double conversion behind it.
        if (left.IsFloat && right.IsFloat && right.Count == 1 && left.Count == count)
        {
            float[] source = left.Floats!;
            long[] destination = result.Ints!;
            float threshold = right.FloatSpan[0];

            Parallelism.Chunked(count, (start, end) =>
            {
                for (int i = start; i < end; i++)
                {
                    destination[i] = predicate(source[i], threshold) ? 1 : 0;
                }
            });

            return result;
        }

        Iterate(left, right, shape, (index, l, r) => result.Ints![index] = predicate(l, r) ? 1 : 0);
        return result;
    }

    private static void Iterate(
        PaddleTensor left,
        PaddleTensor right,
        int[] shape,
        Action<int, double, double> emit)
    {
        int count = PaddleTensor.ElementCount(shape);
        if (count == 0)
        {
            return;
        }

        // Fast path: identical shapes need no index arithmetic at all.
        if (left.Count == count && right.Count == count && shape.Length == left.Rank && shape.Length == right.Rank)
        {
            Parallelism.Chunked(count, (start, end) =>
            {
                for (int i = start; i < end; i++)
                {
                    emit(i, left.GetDouble(i), right.GetDouble(i));
                }
            });

            return;
        }

        int[] leftStrides = StridesFor(left.Shape, shape);
        int[] rightStrides = StridesFor(right.Shape, shape);

        // Split by result index; each chunk seeds its own counters, so the walk no longer has to
        // start at zero. The mask head's broadcasts are over the leading dimensions of twelve
        // million elements, which is where this operator's time is.
        Parallelism.Chunked(count, (start, end) =>
        {
            int[] counters = new int[shape.Length];
            Seed(start, shape, counters);
            int leftOffset = Offset(counters, leftStrides);
            int rightOffset = Offset(counters, rightStrides);

            for (int i = start; i < end; i++)
            {
                emit(i, left.GetDouble(leftOffset), right.GetDouble(rightOffset));

                for (int axis = shape.Length - 1; axis >= 0; axis--)
                {
                    counters[axis]++;
                    leftOffset += leftStrides[axis];
                    rightOffset += rightStrides[axis];

                    if (counters[axis] < shape[axis])
                    {
                        break;
                    }

                    leftOffset -= leftStrides[axis] * shape[axis];
                    rightOffset -= rightStrides[axis] * shape[axis];
                    counters[axis] = 0;
                }
            }
        });
    }
}
