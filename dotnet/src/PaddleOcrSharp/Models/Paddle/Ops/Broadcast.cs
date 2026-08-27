using PaddleOcrSharp.Formats.Paddle;

namespace PaddleOcrSharp.Models.Paddle.Ops;

/// <summary>NumPy-style broadcasting shared by the element-wise kernels.</summary>
public static class Broadcast
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
            for (int i = 0; i < count; i++)
            {
                emit(i, left.GetDouble(i), right.GetDouble(i));
            }

            return;
        }

        int[] leftStrides = StridesFor(left.Shape, shape);
        int[] rightStrides = StridesFor(right.Shape, shape);
        int[] counters = new int[shape.Length];

        int leftOffset = 0;
        int rightOffset = 0;

        for (int i = 0; i < count; i++)
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
    }
}
