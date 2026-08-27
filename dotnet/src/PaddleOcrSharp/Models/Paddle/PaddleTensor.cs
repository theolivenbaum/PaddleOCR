using PaddleOcrSharp.Core;
using PaddleOcrSharp.Formats.Paddle;

namespace PaddleOcrSharp.Models.Paddle;

/// <summary>
/// A dense tensor inside the Paddle graph interpreter.
/// </summary>
/// <remarks>
/// Floating-point tensors are held as <see cref="float"/> and everything integral (including
/// booleans, which Paddle stores as bytes) as <see cref="long"/>. Two storage classes cover every
/// dtype the shipped inference programs use, and keeping indices in <c>long</c> avoids a second
/// integer width in the shape-manipulation kernels.
/// </remarks>
public sealed class PaddleTensor
{
    private PaddleTensor(PaddleDType dtype, int[] shape, float[]? floats, long[]? ints)
    {
        Dtype = dtype;
        Shape = shape;
        Floats = floats;
        Ints = ints;
        Count = ElementCount(shape);
    }

    /// <summary>Element type.</summary>
    public PaddleDType Dtype { get; }

    /// <summary>Dimensions, outermost first.</summary>
    public int[] Shape { get; private set; }

    /// <summary>Float storage, or <see langword="null"/> for integral tensors.</summary>
    public float[]? Floats { get; }

    /// <summary>Integer storage, or <see langword="null"/> for float tensors.</summary>
    public long[]? Ints { get; }

    /// <summary>Number of elements.</summary>
    public int Count { get; private set; }

    /// <summary>Number of dimensions.</summary>
    public int Rank => Shape.Length;

    /// <summary>Whether the tensor holds floats.</summary>
    public bool IsFloat => Floats is not null;

    /// <summary>Float storage as a span.</summary>
    public Span<float> FloatSpan => (Floats ?? throw new InvalidOperationException(
        $"Tensor of dtype {Dtype} has no float storage.")).AsSpan(0, Count);

    /// <summary>Integer storage as a span.</summary>
    public Span<long> IntSpan => (Ints ?? throw new InvalidOperationException(
        $"Tensor of dtype {Dtype} has no integer storage.")).AsSpan(0, Count);

    /// <summary>
    /// Allocates a float tensor whose contents are undefined.
    /// </summary>
    /// <remarks>
    /// Intermediate tensors in a graph are always written in full by the operator that produces
    /// them, and zeroing them first is not free: a single <c>[1, 256, 200, 200]</c> feature map is
    /// 41 MB, and a layout forward pass produces hundreds of them. Use <see cref="Zeros(int[],
    /// PaddleDType)"/> for the few operators that accumulate into their output.
    /// </remarks>
    public static PaddleTensor Float(int[] shape, PaddleDType dtype = PaddleDType.Float32) =>
        new(dtype, shape, GC.AllocateUninitializedArray<float>(ElementCount(shape)), null);

    /// <summary>Allocates an integer tensor whose contents are undefined.</summary>
    public static PaddleTensor Int(int[] shape, PaddleDType dtype = PaddleDType.Int64) =>
        new(dtype, shape, null, GC.AllocateUninitializedArray<long>(ElementCount(shape)));

    /// <summary>Allocates a tensor of the given dtype whose contents are undefined.</summary>
    public static PaddleTensor Allocate(int[] shape, PaddleDType dtype) =>
        dtype.IsFloat() ? Float(shape, dtype) : Int(shape, dtype);

    /// <summary>Allocates a zero-filled tensor, for operators that accumulate into their output.</summary>
    public static PaddleTensor Zeros(int[] shape, PaddleDType dtype = PaddleDType.Float32)
    {
        PaddleTensor tensor = Allocate(shape, dtype);
        if (tensor.IsFloat)
        {
            tensor.FloatSpan.Clear();
        }
        else
        {
            tensor.IntSpan.Clear();
        }

        return tensor;
    }

    /// <summary>Wraps existing float storage.</summary>
    public static PaddleTensor FromFloats(float[] values, int[] shape, PaddleDType dtype = PaddleDType.Float32) =>
        new(dtype, shape, values, null);

    /// <summary>Wraps existing integer storage.</summary>
    public static PaddleTensor FromInts(long[] values, int[] shape, PaddleDType dtype = PaddleDType.Int64) =>
        new(dtype, shape, null, values);

    /// <summary>A rank-1 integer tensor holding <paramref name="values"/>.</summary>
    public static PaddleTensor Vector(params long[] values) =>
        FromInts(values, [values.Length]);

    /// <summary>A rank-0 float tensor.</summary>
    public static PaddleTensor Scalar(float value) => FromFloats([value], []);

    /// <summary>Number of elements implied by <paramref name="shape"/>.</summary>
    public static int ElementCount(ReadOnlySpan<int> shape)
    {
        int count = 1;
        foreach (int dimension in shape)
        {
            count *= dimension;
        }

        return count;
    }

    /// <summary>Row-major strides of <paramref name="shape"/>.</summary>
    public static int[] Strides(ReadOnlySpan<int> shape)
    {
        int[] strides = new int[shape.Length];
        int stride = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }

        return strides;
    }

    /// <summary>Reads element <paramref name="index"/> as a double, whatever the storage is.</summary>
    public double GetDouble(int index) => Floats is not null ? Floats[index] : Ints![index];

    /// <summary>Reads element <paramref name="index"/> as a long, whatever the storage is.</summary>
    public long GetLong(int index) => Ints is not null ? Ints[index] : (long)Floats![index];

    /// <summary>Reassigns the shape without touching the data.</summary>
    public PaddleTensor WithShape(int[] shape)
    {
        int count = ElementCount(shape);
        if (count != Count)
        {
            throw new ArgumentException(
                $"Cannot view {Count} elements as [{string.Join(",", shape)}] ({count} elements).",
                nameof(shape));
        }

        Shape = shape;
        return this;
    }

    /// <summary>A tensor sharing this one's storage but with a different declared shape.</summary>
    public PaddleTensor Reshaped(int[] shape)
    {
        int count = ElementCount(shape);
        if (count != Count)
        {
            throw new ArgumentException(
                $"Cannot reshape {Count} elements to [{string.Join(",", shape)}] ({count} elements).",
                nameof(shape));
        }

        return new PaddleTensor(Dtype, shape, Floats, Ints) { Count = count };
    }

    /// <summary>Converts to the requested dtype, copying when the storage class changes.</summary>
    public PaddleTensor Cast(PaddleDType dtype)
    {
        if (dtype == Dtype)
        {
            return this;
        }

        if (dtype.IsFloat() && IsFloat)
        {
            return new PaddleTensor(dtype, Shape, Floats, null);
        }

        if (!dtype.IsFloat() && !IsFloat)
        {
            if (dtype == PaddleDType.Bool)
            {
                long[] booleans = new long[Count];
                for (int i = 0; i < Count; i++)
                {
                    booleans[i] = Ints![i] != 0 ? 1 : 0;
                }

                return FromInts(booleans, Shape, dtype);
            }

            if (dtype == PaddleDType.Int32)
            {
                long[] narrowed = new long[Count];
                for (int i = 0; i < Count; i++)
                {
                    narrowed[i] = (int)Ints![i];
                }

                return FromInts(narrowed, Shape, dtype);
            }

            return new PaddleTensor(dtype, Shape, null, Ints);
        }

        if (dtype.IsFloat())
        {
            float[] values = new float[Count];
            for (int i = 0; i < Count; i++)
            {
                values[i] = Ints![i];
            }

            return FromFloats(values, Shape, dtype);
        }

        long[] integers = new long[Count];
        for (int i = 0; i < Count; i++)
        {
            integers[i] = dtype == PaddleDType.Bool
                ? (Floats![i] != 0f ? 1 : 0)
                : (long)Floats![i];
        }

        return FromInts(integers, Shape, dtype);
    }

    /// <summary>Deep copy.</summary>
    public PaddleTensor Clone()
    {
        PaddleTensor copy = Allocate([.. Shape], Dtype);
        if (IsFloat)
        {
            FloatSpan.CopyTo(copy.FloatSpan);
        }
        else
        {
            IntSpan.CopyTo(copy.IntSpan);
        }

        return copy;
    }

    /// <summary>Materialises a parameter from a memory-mapped weight file.</summary>
    public static PaddleTensor FromParameter(PaddleParameter parameter)
    {
        if (parameter.Dtype.IsFloat())
        {
            float[] values = new float[parameter.Count];
            FloatConversion.ToFloat(parameter.Bytes.Span, ToCoreDType(parameter.Dtype), values);
            return FromFloats(values, parameter.Shape, PaddleDType.Float32);
        }

        long[] integers = new long[parameter.Count];
        ReadOnlySpan<byte> bytes = parameter.Bytes.Span;
        switch (parameter.Dtype)
        {
            case PaddleDType.Int64:
                System.Runtime.InteropServices.MemoryMarshal.Cast<byte, long>(bytes)[..parameter.Count]
                    .CopyTo(integers);
                break;
            case PaddleDType.Int32:
            {
                ReadOnlySpan<int> source =
                    System.Runtime.InteropServices.MemoryMarshal.Cast<byte, int>(bytes);
                for (int i = 0; i < parameter.Count; i++)
                {
                    integers[i] = source[i];
                }

                break;
            }
            case PaddleDType.Bool or PaddleDType.UInt8:
            {
                for (int i = 0; i < parameter.Count; i++)
                {
                    integers[i] = bytes[i];
                }

                break;
            }
            default:
                throw new NotSupportedException($"Cannot load a {parameter.Dtype} parameter.");
        }

        return FromInts(integers, parameter.Shape, parameter.Dtype);
    }

    private static DType ToCoreDType(PaddleDType dtype) => dtype switch
    {
        PaddleDType.Float32 => DType.Float32,
        PaddleDType.Float16 => DType.Float16,
        PaddleDType.BFloat16 => DType.BFloat16,
        _ => throw new NotSupportedException($"{dtype} is not a supported float dtype."),
    };

    /// <inheritdoc />
    public override string ToString() => $"{Dtype}[{string.Join(",", Shape)}]";
}
