using PaddleOcrSharp.Formats.Paddle;
using PaddleOcrSharp.Models.Paddle.Ops;

namespace PaddleOcrSharp.Models.Paddle;

/// <summary>
/// Executes a Paddle PIR inference program.
/// </summary>
/// <remarks>
/// <para>
/// PP-DocLayoutV3, the document orientation classifier and UVDoc all ship as Paddle inference
/// graphs rather than as a documented module tree, and there is no upstream PyTorch definition to
/// port from. Interpreting the exported graph directly is therefore both less code and strictly
/// more faithful than reconstructing the architecture by hand: every operator here is our own
/// kernel, and the result is exact by construction rather than by inspection.
/// </para>
/// <para>
/// Operations arrive in SSA execution order, so a single forward pass over the list suffices.
/// Intermediate values are released as soon as their last consumer has run, which keeps a
/// 91-convolution backbone at a bounded working set.
/// </para>
/// </remarks>
public sealed class PirInterpreter : IDisposable
{
    private readonly PirProgram _program;
    private readonly PaddleParameterFile _parameters;
    private readonly Dictionary<int, PaddleTensor> _constants = [];
    private readonly int[] _lastUse;
    private readonly int _valueCount;

    private PirInterpreter(PirProgram program, PaddleParameterFile parameters)
    {
        _program = program;
        _parameters = parameters;

        int maximum = 0;
        foreach (PirOperation operation in program.Operations)
        {
            foreach (int id in operation.Outputs)
            {
                maximum = Math.Max(maximum, id);
            }

            foreach (int id in operation.Inputs)
            {
                maximum = Math.Max(maximum, id);
            }
        }

        _valueCount = maximum + 1;
        _lastUse = new int[_valueCount];
        Array.Fill(_lastUse, -1);

        for (int index = 0; index < program.Operations.Count; index++)
        {
            foreach (int id in program.Operations[index].Inputs)
            {
                if (id > 0)
                {
                    _lastUse[id] = index;
                }
            }
        }
    }

    /// <summary>Names of the program's feed inputs.</summary>
    public IReadOnlyList<string> InputNames => _program.Inputs;

    /// <summary>Loads a program and its weights from a model directory.</summary>
    /// <param name="directory">Directory holding <c>inference.json</c> and <c>inference.pdiparams</c>.</param>
    public static PirInterpreter Load(string directory)
    {
        PirProgram program = PirProgram.Load(Path.Combine(directory, "inference.json"));
        PaddleParameterFile parameters = PaddleParameterFile.Read(
            Path.Combine(directory, "inference.pdiparams"), program.Parameters);
        return new PirInterpreter(program, parameters);
    }

    /// <summary>
    /// Runs the program.
    /// </summary>
    /// <param name="inputs">Feed tensors, keyed by the names in <see cref="InputNames"/>.</param>
    /// <param name="trace">Optional recorder for intermediate values.</param>
    /// <returns>The fetched outputs, keyed by fetch name.</returns>
    public Dictionary<string, PaddleTensor> Run(
        IReadOnlyDictionary<string, PaddleTensor> inputs,
        IPirTrace? trace = null,
        PirProfile? profile = null)
    {
        object?[] values = new object?[_valueCount];
        var outputs = new Dictionary<string, PaddleTensor>(StringComparer.Ordinal);

        for (int index = 0; index < _program.Operations.Count; index++)
        {
            PirOperation operation = _program.Operations[index];
            long started = profile is null ? 0 : System.Diagnostics.Stopwatch.GetTimestamp();

            try
            {
                Execute(operation, values, inputs, outputs);
            }
            catch (Exception exception) when (exception is not NotSupportedException)
            {
                throw new InvalidOperationException(
                    $"Operation {index} ({operation}) at '{operation.StructName}' failed: {exception.Message}",
                    exception);
            }

            profile?.Add(operation.Name, System.Diagnostics.Stopwatch.GetElapsedTime(started));
            trace?.Record(index, operation, operation.Outputs.Select(id => id > 0 ? values[id] : null).ToArray());

            foreach (int id in operation.Inputs)
            {
                if (id > 0 && _lastUse[id] == index && !_constants.ContainsKey(id))
                {
                    values[id] = null;
                }
            }
        }

        return outputs;
    }

    private void Execute(
        PirOperation operation,
        object?[] values,
        IReadOnlyDictionary<string, PaddleTensor> inputs,
        Dictionary<string, PaddleTensor> outputs)
    {
        switch (operation.Name)
        {
            case "parameter":
            {
                int id = operation.Outputs[0];
                if (!_constants.TryGetValue(id, out PaddleTensor? parameter))
                {
                    parameter = PaddleTensor.FromParameter(_parameters[operation.ParameterName!]);
                    _constants[id] = parameter;
                }

                values[id] = parameter;
                return;
            }

            case "data":
            {
                string name = operation.Attribute("name")!.AsString();
                if (!inputs.TryGetValue(name, out PaddleTensor? tensor))
                {
                    throw new ArgumentException($"Program input '{name}' was not supplied.", nameof(inputs));
                }

                values[operation.Outputs[0]] = tensor;
                return;
            }

            case "fetch":
            {
                string name = operation.Attribute("name")!.AsString();
                outputs[name] = Tensor(values, operation.Inputs[0]);
                values[operation.Outputs[0]] = outputs[name];
                return;
            }

            case "builtin.combine":
            {
                values[operation.Outputs[0]] = operation.Inputs
                    .Select(id => Tensor(values, id))
                    .ToArray();
                return;
            }

            case "builtin.split":
            {
                PaddleTensor[] list = List(values, operation.Inputs[0]);
                if (list.Length != operation.Outputs.Length)
                {
                    throw new InvalidOperationException(
                        $"builtin.split expects {operation.Outputs.Length} elements but the list holds {list.Length}.");
                }

                for (int i = 0; i < operation.Outputs.Length; i++)
                {
                    values[operation.Outputs[i]] = list[i];
                }

                return;
            }
        }

        object[] results = Dispatch(operation, values);
        for (int i = 0; i < operation.Outputs.Length && i < results.Length; i++)
        {
            if (operation.Outputs[i] > 0)
            {
                values[operation.Outputs[i]] = results[i];
            }
        }
    }

    private object[] Dispatch(PirOperation operation, object?[] values)
    {
        switch (operation.Name)
        {
            case "full":
            {
                long[] shape = operation.Attribute("shape")!.AsLongArray();
                double value = operation.Attribute("value")!.AsDouble();
                PaddleDType dtype = PaddleDTypeExtensions.FromName(operation.Attribute("dtype")!.AsString());
                return [Fill([.. shape.Select(x => (int)x)], value, dtype)];
            }

            case "full_int_array":
            {
                long[] value = operation.Attribute("value")!.AsLongArray();
                return [PaddleTensor.FromInts(value, [value.Length])];
            }

            case "full_like":
            {
                PaddleTensor reference = Tensor(values, operation.Inputs[0]);
                PaddleTensor fill = Tensor(values, operation.Inputs[1]);
                PaddleDType dtype = PaddleDTypeExtensions.FromName(operation.Attribute("dtype")!.AsString());
                return [Fill([.. reference.Shape], fill.GetDouble(0), dtype)];
            }

            case "full_with_tensor":
            {
                PaddleTensor first = Tensor(values, operation.Inputs[0]);
                PaddleTensor second = Tensor(values, operation.Inputs[1]);

                // Paddle's operand order for this op has moved between releases; the shape operand
                // is the rank-1 integer vector, which identifies it unambiguously.
                (PaddleTensor shapeTensor, PaddleTensor valueTensor) =
                    !second.IsFloat && second.Rank == 1 && second.Count <= 8 && first.Count == 1
                        ? (second, first)
                        : (first, second);

                int[] shape = [.. Enumerable.Range(0, shapeTensor.Count).Select(i => (int)shapeTensor.GetLong(i))];
                PaddleDType dtype = PaddleDTypeExtensions.FromName(operation.Attribute("dtype")!.AsString());
                return [Fill(shape, valueTensor.GetDouble(0), dtype)];
            }

            case "assign_value_":
            {
                long[] shape = operation.Attribute("shape")!.AsLongArray();
                PaddleDType dtype = PaddleDTypeExtensions.FromName(operation.Attribute("dtype")!.AsString());
                double[] data = operation.Attribute("values")!.AsDoubleArray();
                int[] dimensions = [.. shape.Select(x => (int)x)];

                // The attribute may carry fewer values than the shape declares; the remainder is
                // zero, so this one cannot start from an uninitialised buffer.
                PaddleTensor result = PaddleTensor.Zeros(dimensions, dtype);
                for (int i = 0; i < result.Count && i < data.Length; i++)
                {
                    if (result.IsFloat)
                    {
                        result.Floats![i] = (float)data[i];
                    }
                    else
                    {
                        result.Ints![i] = (long)data[i];
                    }
                }

                return [result];
            }

            case "arange":
            {
                double start = Tensor(values, operation.Inputs[0]).GetDouble(0);
                double end = Tensor(values, operation.Inputs[1]).GetDouble(0);
                double step = Tensor(values, operation.Inputs[2]).GetDouble(0);
                PaddleDType dtype = PaddleDTypeExtensions.FromName(operation.Attribute("dtype")!.AsString());

                int count = step == 0 ? 0 : Math.Max(0, (int)Math.Ceiling((end - start) / step));
                PaddleTensor result = PaddleTensor.Allocate([count], dtype);
                for (int i = 0; i < count; i++)
                {
                    double value = start + (i * step);
                    if (result.IsFloat)
                    {
                        result.Floats![i] = (float)value;
                    }
                    else
                    {
                        result.Ints![i] = (long)value;
                    }
                }

                return [result];
            }

            case "eye":
            {
                int rows = (int)Tensor(values, operation.Inputs[0]).GetLong(0);
                int columns = (int)Tensor(values, operation.Inputs[1]).GetLong(0);
                PaddleDType dtype = PaddleDTypeExtensions.FromName(operation.Attribute("dtype")!.AsString());
                return [ShapeOps.Eye(rows, columns, dtype)];
            }

            case "shape64":
            {
                PaddleTensor input = Tensor(values, operation.Inputs[0]);
                return [PaddleTensor.FromInts([.. input.Shape.Select(x => (long)x)], [input.Rank])];
            }

            case "cast":
            {
                PaddleDType dtype = PaddleDTypeExtensions.FromName(operation.Attribute("dtype")!.AsString());
                return [Tensor(values, operation.Inputs[0]).Cast(dtype)];
            }

            case "share_data_":
                return [Tensor(values, operation.Inputs[0])];

            case "dropout":
            {
                // is_test is always true in an exported inference program, so no element is ever
                // dropped. "upscale_in_train" did the rescaling during training and is the
                // identity here; "downgrade_in_infer" instead shrinks the activations by 1 - p at
                // inference, and p arrives as the third input rather than as an attribute.
                PaddleTensor input = Tensor(values, operation.Inputs[0]);
                string mode = operation.Attribute("mode")?.AsString() ?? "upscale_in_train";

                if (mode != "downgrade_in_infer")
                {
                    return [input, input];
                }

                PaddleTensor? probability = Optional(values, operation.Inputs.ElementAtOrDefault(2));
                double keep = 1.0 - (probability is null ? 0.0 : probability.FloatSpan[0]);
                PaddleTensor scaled = ElementwiseOps.Scale(input, keep, 0.0, biasAfterScale: true);
                return [scaled, scaled];
            }

            case "conv2d":
            case "depthwise_conv2d":
            {
                return
                [
                    ConvOps.Conv2d(
                        Tensor(values, operation.Inputs[0]),
                        Tensor(values, operation.Inputs[1]),
                        operation.Attribute("strides")!.AsIntArray(),
                        operation.Attribute("paddings")!.AsIntArray(),
                        operation.Attribute("dilations")!.AsIntArray(),
                        operation.Attribute("groups")!.AsInt(),
                        operation.Attribute("padding_algorithm")?.AsString() ?? "EXPLICIT",
                        operation.Attribute("data_format")?.AsString() ?? "NCHW"),
                ];
            }

            case "batch_norm_":
            {
                PaddleTensor result = LinearOps.BatchNorm(
                    Tensor(values, operation.Inputs[0]),
                    Tensor(values, operation.Inputs[1]),
                    Tensor(values, operation.Inputs[2]),
                    Tensor(values, operation.Inputs[3]),
                    Tensor(values, operation.Inputs[4]),
                    (float)operation.Attribute("epsilon")!.AsDouble(),
                    operation.Attribute("data_format")?.AsString() ?? "NCHW");

                // Only the first result is consumed downstream; the running statistics and the
                // saved mean/variance are training-time artefacts.
                return [result];
            }

            case "pool2d":
            {
                PaddleTensor kernel = Tensor(values, operation.Inputs[1]);
                return
                [
                    ConvOps.Pool2d(
                        Tensor(values, operation.Inputs[0]),
                        [.. Enumerable.Range(0, kernel.Count).Select(i => (int)kernel.GetLong(i))],
                        operation.Attribute("strides")!.AsIntArray(),
                        operation.Attribute("paddings")!.AsIntArray(),
                        operation.Attribute("ceil_mode")?.AsBool() ?? false,
                        operation.Attribute("exclusive")?.AsBool() ?? true,
                        operation.Attribute("global_pooling")?.AsBool() ?? false,
                        operation.Attribute("adaptive")?.AsBool() ?? false,
                        operation.Attribute("pooling_type")!.AsString(),
                        operation.Attribute("padding_algorithm")?.AsString() ?? "EXPLICIT",
                        operation.Attribute("data_format")?.AsString() ?? "NCHW"),
                ];
            }

            case "matmul":
                return
                [
                    LinearOps.MatMul(
                        Tensor(values, operation.Inputs[0]),
                        Tensor(values, operation.Inputs[1]),
                        operation.Attribute("transpose_x")?.AsBool() ?? false,
                        operation.Attribute("transpose_y")?.AsBool() ?? false),
                ];

            case "bmm":
                return [LinearOps.Bmm(Tensor(values, operation.Inputs[0]), Tensor(values, operation.Inputs[1]))];

            case "einsum":
                return [EinsumOps.Apply(operation.Attribute("equation")!.AsString(), List(values, operation.Inputs[0]))];

            case "layer_norm":
                return
                [
                    LinearOps.LayerNorm(
                        Tensor(values, operation.Inputs[0]),
                        Optional(values, operation.Inputs.ElementAtOrDefault(1)),
                        Optional(values, operation.Inputs.ElementAtOrDefault(2)),
                        (float)operation.Attribute("epsilon")!.AsDouble(),
                        operation.Attribute("begin_norm_axis")!.AsInt()),
                ];

            case "softmax":
                return [LinearOps.Softmax(Tensor(values, operation.Inputs[0]), operation.Attribute("axis")!.AsInt())];

            case "relu":
                return [ElementwiseOps.Apply(Tensor(values, operation.Inputs[0]), ElementwiseOps.Unary.Relu)];

            case "sigmoid":
                return [ElementwiseOps.Apply(Tensor(values, operation.Inputs[0]), ElementwiseOps.Unary.Sigmoid)];

            case "silu":
                return [ElementwiseOps.Apply(Tensor(values, operation.Inputs[0]), ElementwiseOps.Unary.Silu)];

            case "gelu":
            {
                bool approximate = operation.Attribute("approximate")?.AsBool() ?? false;
                return
                [
                    ElementwiseOps.Apply(
                        Tensor(values, operation.Inputs[0]),
                        approximate ? ElementwiseOps.Unary.GeluTanh : ElementwiseOps.Unary.GeluErf),
                ];
            }

            case "hardswish":
                return [ElementwiseOps.Apply(Tensor(values, operation.Inputs[0]), ElementwiseOps.Unary.HardSwish)];

            case "hardsigmoid":
                return
                [
                    ElementwiseOps.HardSigmoid(
                        Tensor(values, operation.Inputs[0]),
                        (float)operation.Attribute("slope")!.AsDouble(),
                        (float)operation.Attribute("offset")!.AsDouble()),
                ];

            case "prelu":
                return
                [
                    ElementwiseOps.PRelu(
                        Tensor(values, operation.Inputs[0]),
                        Tensor(values, operation.Inputs[1]),
                        operation.Attribute("mode")?.AsString() ?? "all",
                        (operation.Attribute("data_format")?.AsString() ?? "NCHW") is "NHWC" or "NDHWC"),
                ];

            case "pad3d":
            {
                PaddleTensor input = Tensor(values, operation.Inputs[0]);
                PaddleTensor paddings = Tensor(values, operation.Inputs[1]);

                int[] amounts = new int[paddings.Count];
                for (int i = 0; i < amounts.Length; i++)
                {
                    amounts[i] = (int)paddings.GetLong(i);
                }

                return
                [
                    PadOps.Pad3d(
                        input,
                        amounts,
                        PadOps.ParseMode(operation.Attribute("mode")?.AsString() ?? "constant"),
                        (float)(operation.Attribute("pad_value")?.AsDouble() ?? 0.0),
                        (operation.Attribute("data_format")?.AsString() ?? "NCDHW") == "NDHWC"),
                ];
            }

            case "log":
                return [ElementwiseOps.Apply(Tensor(values, operation.Inputs[0]), ElementwiseOps.Unary.Log)];

            case "floor":
                return [ElementwiseOps.Apply(Tensor(values, operation.Inputs[0]), ElementwiseOps.Unary.Floor)];

            case "add":
                return [Binary(values, operation, ElementwiseOps.Binary.Add)];

            case "subtract":
                return [Binary(values, operation, ElementwiseOps.Binary.Subtract)];

            case "multiply":
                return [Binary(values, operation, ElementwiseOps.Binary.Multiply)];

            case "divide":
                return [Binary(values, operation, ElementwiseOps.Binary.Divide)];

            case "remainder":
                return
                [
                    Broadcast.Apply(
                        Tensor(values, operation.Inputs[0]),
                        Tensor(values, operation.Inputs[1]),
                        // Paddle follows Python: the result takes the divisor's sign.
                        static (a, b) => a - (Math.Floor(a / b) * b)),
                ];

            case "floor_divide":
                return
                [
                    Broadcast.Apply(
                        Tensor(values, operation.Inputs[0]),
                        Tensor(values, operation.Inputs[1]),
                        static (a, b) => Math.Floor(a / b)),
                ];

            case "greater_than":
                return [Broadcast.Compare(Tensor(values, operation.Inputs[0]), Tensor(values, operation.Inputs[1]), static (a, b) => a > b)];

            case "scale":
            {
                PaddleTensor input = Tensor(values, operation.Inputs[0]);
                double factor = operation.Inputs.Length > 1 && operation.Inputs[1] > 0
                    ? Tensor(values, operation.Inputs[1]).GetDouble(0)
                    : 1.0;
                double bias = operation.Attribute("bias")?.AsDouble() ?? 0.0;
                bool biasAfterScale = operation.Attribute("bias_after_scale")?.AsBool() ?? true;

                return [ElementwiseOps.Scale(input, factor, bias, biasAfterScale)];
            }

            case "clip":
            {
                PaddleTensor input = Tensor(values, operation.Inputs[0]);
                double low = Tensor(values, operation.Inputs[1]).GetDouble(0);
                double high = Tensor(values, operation.Inputs[2]).GetDouble(0);
                return [ElementwiseOps.Clip(input, low, high)];
            }

            case "where":
                return
                [
                    ReduceOps.Where(
                        Tensor(values, operation.Inputs[0]),
                        Tensor(values, operation.Inputs[1]),
                        Tensor(values, operation.Inputs[2])),
                ];

            case "reshape":
            {
                PaddleTensor input = Tensor(values, operation.Inputs[0]);
                PaddleTensor shapeTensor = Tensor(values, operation.Inputs[1]);
                int[] shape = ResolveShape(shapeTensor, input);
                return [input.Reshaped(shape), PaddleTensor.Vector([.. input.Shape.Select(x => (long)x)])];
            }

            case "flatten":
            {
                PaddleTensor input = Tensor(values, operation.Inputs[0]);
                PaddleTensor result = ShapeOps.Flatten(
                    input,
                    operation.Attribute("start_axis")!.AsInt(),
                    operation.Attribute("stop_axis")!.AsInt());
                return [result, PaddleTensor.Vector([.. input.Shape.Select(x => (long)x)])];
            }

            case "transpose":
                return [ShapeOps.Transpose(Tensor(values, operation.Inputs[0]), operation.Attribute("perm")!.AsIntArray())];

            case "unsqueeze":
            {
                PaddleTensor input = Tensor(values, operation.Inputs[0]);
                long[] axes = AxisArgument(operation, values, 1);
                return [ShapeOps.Unsqueeze(input, axes), PaddleTensor.Vector([.. input.Shape.Select(x => (long)x)])];
            }

            case "squeeze":
            {
                PaddleTensor input = Tensor(values, operation.Inputs[0]);
                long[] axes = AxisArgument(operation, values, 1);
                return [ShapeOps.Squeeze(input, axes), PaddleTensor.Vector([.. input.Shape.Select(x => (long)x)])];
            }

            case "slice":
            {
                PaddleTensor input = Tensor(values, operation.Inputs[0]);
                PaddleTensor starts = Tensor(values, operation.Inputs[1]);
                PaddleTensor ends = Tensor(values, operation.Inputs[2]);
                return
                [
                    ShapeOps.Slice(
                        input,
                        operation.Attribute("axes")!.AsLongArray(),
                        [.. Enumerable.Range(0, starts.Count).Select(starts.GetLong)],
                        [.. Enumerable.Range(0, ends.Count).Select(ends.GetLong)],
                        operation.Attribute("decrease_axis")?.AsLongArray() ?? []),
                ];
            }

            case "concat":
            {
                PaddleTensor[] list = List(values, operation.Inputs[0]);
                int axis = (int)Tensor(values, operation.Inputs[1]).GetLong(0);
                return [ShapeOps.Concat(list, axis)];
            }

            case "stack":
            {
                PaddleTensor[] list = List(values, operation.Inputs[0]);
                return [ShapeOps.Stack(list, operation.Attribute("axis")?.AsInt() ?? 0)];
            }

            case "split":
            {
                PaddleTensor input = Tensor(values, operation.Inputs[0]);
                PaddleTensor sections = Tensor(values, operation.Inputs[1]);
                int axis = (int)Tensor(values, operation.Inputs[2]).GetLong(0);
                int[] parts = [.. Enumerable.Range(0, sections.Count).Select(i => (int)sections.GetLong(i))];
                return [ShapeOps.Split(input, parts, axis)];
            }

            case "split_with_num":
            {
                PaddleTensor input = Tensor(values, operation.Inputs[0]);
                int axis = (int)Tensor(values, operation.Inputs[1]).GetLong(0);
                int number = operation.Attribute("num")!.AsInt();
                int rank = input.Rank;
                int resolved = axis < 0 ? axis + rank : axis;
                int size = input.Shape[resolved] / number;
                return [ShapeOps.Split(input, [.. Enumerable.Repeat(size, number)], resolved)];
            }

            case "tile":
                return [ShapeOps.Tile(Tensor(values, operation.Inputs[0]), ToLongArray(Tensor(values, operation.Inputs[1])))];

            case "expand":
                return [ShapeOps.Expand(Tensor(values, operation.Inputs[0]), ToLongArray(Tensor(values, operation.Inputs[1])))];

            case "flip":
                return [ShapeOps.Flip(Tensor(values, operation.Inputs[0]), operation.Attribute("axis")!.AsLongArray())];

            case "gather_nd":
                return [ShapeOps.GatherNd(Tensor(values, operation.Inputs[0]), Tensor(values, operation.Inputs[1]))];

            case "meshgrid":
                return [ShapeOps.MeshGrid(List(values, operation.Inputs[0]))];

            case "sum":
                return
                [
                    ReduceOps.Reduce(
                        Tensor(values, operation.Inputs[0]),
                        ToLongArray(Tensor(values, operation.Inputs[1])),
                        operation.Attribute("keepdim")?.AsBool() ?? false,
                        ReduceOps.Kind.Sum),
                ];

            case "max":
                return
                [
                    ReduceOps.Reduce(
                        Tensor(values, operation.Inputs[0]),
                        ToLongArray(Tensor(values, operation.Inputs[1])),
                        operation.Attribute("keepdim")?.AsBool() ?? false,
                        ReduceOps.Kind.Max),
                ];

            case "min":
                return
                [
                    ReduceOps.Reduce(
                        Tensor(values, operation.Inputs[0]),
                        ToLongArray(Tensor(values, operation.Inputs[1])),
                        operation.Attribute("keepdim")?.AsBool() ?? false,
                        ReduceOps.Kind.Min),
                ];

            case "any":
                return
                [
                    ReduceOps.Reduce(
                        Tensor(values, operation.Inputs[0]),
                        operation.Attribute("axis")?.AsLongArray() ?? [],
                        operation.Attribute("keepdim")?.AsBool() ?? false,
                        ReduceOps.Kind.Any),
                ];

            case "topk":
            {
                PaddleTensor input = Tensor(values, operation.Inputs[0]);
                int k = (int)Tensor(values, operation.Inputs[1]).GetLong(0);
                (PaddleTensor topValues, PaddleTensor indices) = ReduceOps.TopK(
                    input,
                    k,
                    operation.Attribute("axis")?.AsInt() ?? -1,
                    operation.Attribute("largest")?.AsBool() ?? true,
                    operation.Attribute("sorted")?.AsBool() ?? true);
                return [topValues, indices];
            }

            case "argsort":
            {
                (PaddleTensor sorted, PaddleTensor indices) = ReduceOps.ArgSort(
                    Tensor(values, operation.Inputs[0]),
                    operation.Attribute("axis")?.AsInt() ?? -1,
                    operation.Attribute("descending")?.AsBool() ?? false,
                    operation.Attribute("stable")?.AsBool() ?? false);
                return [sorted, indices];
            }

            case "grid_sample":
                return
                [
                    SamplingOps.GridSample(
                        Tensor(values, operation.Inputs[0]),
                        Tensor(values, operation.Inputs[1]),
                        operation.Attribute("mode")?.AsString() ?? "bilinear",
                        operation.Attribute("padding_mode")?.AsString() ?? "zeros",
                        operation.Attribute("align_corners")?.AsBool() ?? true),
                ];

            case "bilinear_interp":
            case "nearest_interp":
            {
                PaddleTensor input = Tensor(values, operation.Inputs[0]);
                (int height, int width) = ResolveInterpolationSize(operation, values, input);
                bool alignCorners = operation.Attribute("align_corners")?.AsBool() ?? false;
                string format = operation.Attribute("data_format")?.AsString() ?? "NCHW";

                return operation.Name == "bilinear_interp"
                    ? [SamplingOps.BilinearInterp(
                        input, height, width, alignCorners, operation.Attribute("align_mode")?.AsInt() ?? 0, format)]
                    : [SamplingOps.NearestInterp(input, height, width, alignCorners, format)];
            }

            case "index_put":
                return
                [
                    IndexOps.IndexPut(
                        Tensor(values, operation.Inputs[0]),
                        List(values, operation.Inputs[1]),
                        Tensor(values, operation.Inputs[2]),
                        operation.Attribute("accumulate")?.AsBool() ?? false),
                ];

            case "set_value_with_tensor_":
                return
                [
                    IndexOps.SetValue(
                        Tensor(values, operation.Inputs[0]),
                        Tensor(values, operation.Inputs[1]),
                        ToLongArray(Tensor(values, operation.Inputs[2])),
                        ToLongArray(Tensor(values, operation.Inputs[3])),
                        ToLongArray(Tensor(values, operation.Inputs[4])),
                        operation.Attribute("axes")?.AsLongArray() ?? [],
                        operation.Attribute("decrease_axes")?.AsLongArray() ?? [],
                        operation.Attribute("none_axes")?.AsLongArray() ?? []),
                ];

            default:
                throw new NotSupportedException(
                    $"Paddle operator '{operation.Name}' is not implemented ({operation.StructName}).");
        }
    }

    private static PaddleTensor Binary(object?[] values, PirOperation operation, ElementwiseOps.Binary kind) =>
        ElementwiseOps.Apply(
            Tensor(values, operation.Inputs[0]), Tensor(values, operation.Inputs[1]), kind);

    private static (int Height, int Width) ResolveInterpolationSize(
        PirOperation operation,
        object?[] values,
        PaddleTensor input)
    {
        int height = operation.Attribute("out_h")?.AsInt() ?? -1;
        int width = operation.Attribute("out_w")?.AsInt() ?? -1;

        // Paddle offers the target three ways, in falling priority: an "OutSize" tensor holding
        // [h, w], a "SizeTensor" list of rank-0 tensors (one per spatial axis, which is how a
        // size computed at run time from another tensor's shape arrives), and a scale.
        if (operation.Inputs.Length > 1 && operation.Inputs[1] > 0)
        {
            PaddleTensor size = Tensor(values, operation.Inputs[1]);
            height = (int)size.GetLong(0);
            width = (int)size.GetLong(1);
        }
        else if (operation.Inputs.Length > 2 && operation.Inputs[2] > 0)
        {
            PaddleTensor[] parts = List(values, operation.Inputs[2]);
            if (parts.Length >= 2)
            {
                height = (int)parts[0].GetLong(0);
                width = (int)parts[1].GetLong(0);
            }
        }

        if (height > 0 && width > 0)
        {
            return (height, width);
        }

        double[] scale = operation.Attribute("scale")?.AsDoubleArray() ?? [];
        if (scale.Length == 0 && operation.Inputs.Length > 3 && operation.Inputs[3] > 0)
        {
            PaddleTensor factors = Tensor(values, operation.Inputs[3]);
            scale = new double[factors.Count];
            for (int i = 0; i < scale.Length; i++)
            {
                scale[i] = factors.IsFloat ? factors.FloatSpan[i] : factors.GetLong(i);
            }
        }

        double scaleY = scale.Length > 0 ? scale[0] : 1.0;
        double scaleX = scale.Length > 1 ? scale[1] : scaleY;

        return ((int)(input.Shape[2] * scaleY), (int)(input.Shape[3] * scaleX));
    }

    private static long[] AxisArgument(PirOperation operation, object?[] values, int inputIndex)
    {
        if (operation.Inputs.Length > inputIndex && operation.Inputs[inputIndex] > 0)
        {
            return ToLongArray(Tensor(values, operation.Inputs[inputIndex]));
        }

        return operation.Attribute("axis")?.AsLongArray() ?? operation.Attribute("axes")?.AsLongArray() ?? [];
    }

    /// <summary>
    /// Resolves a reshape target, applying Paddle's two placeholders: <c>0</c> copies the input's
    /// dimension at the same position and <c>-1</c> is inferred from the element count.
    /// </summary>
    private static int[] ResolveShape(PaddleTensor shapeTensor, PaddleTensor input)
    {
        int[] shape = new int[shapeTensor.Count];
        int inferred = -1;
        int known = 1;

        for (int i = 0; i < shape.Length; i++)
        {
            int value = (int)shapeTensor.GetLong(i);
            if (value == 0)
            {
                if (i >= input.Rank)
                {
                    throw new InvalidOperationException(
                        $"Reshape uses the copy placeholder at axis {i} but the input has rank {input.Rank}.");
                }

                value = input.Shape[i];
            }

            shape[i] = value;
            if (value < 0)
            {
                inferred = i;
            }
            else
            {
                known *= value;
            }
        }

        if (inferred >= 0)
        {
            shape[inferred] = known == 0 ? 0 : input.Count / known;
        }

        return shape;
    }

    private static long[] ToLongArray(PaddleTensor tensor) =>
        [.. Enumerable.Range(0, tensor.Count).Select(tensor.GetLong)];

    private static PaddleTensor Fill(int[] shape, double value, PaddleDType dtype)
    {
        PaddleTensor result = PaddleTensor.Allocate(shape, dtype);
        if (result.IsFloat)
        {
            result.FloatSpan.Fill((float)value);
        }
        else
        {
            result.IntSpan.Fill((long)value);
        }

        return result;
    }

    private static PaddleTensor Map(PaddleTensor input, Func<double, double> operation)
    {
        PaddleTensor result = PaddleTensor.Allocate([.. input.Shape], input.Dtype);
        if (input.IsFloat)
        {
            ReadOnlySpan<float> source = input.FloatSpan;
            Span<float> destination = result.FloatSpan;
            for (int i = 0; i < source.Length; i++)
            {
                destination[i] = (float)operation(source[i]);
            }
        }
        else
        {
            ReadOnlySpan<long> source = input.IntSpan;
            Span<long> destination = result.IntSpan;
            for (int i = 0; i < source.Length; i++)
            {
                destination[i] = (long)operation(source[i]);
            }
        }

        return result;
    }

    private static PaddleTensor Tensor(object?[] values, int id) => values[id] switch
    {
        PaddleTensor tensor => tensor,
        PaddleTensor[] list when list.Length == 1 => list[0],
        null => throw new InvalidOperationException($"Value %{id} has not been produced."),
        _ => throw new InvalidOperationException($"Value %{id} is a tensor list, not a tensor."),
    };

    private static PaddleTensor? Optional(object?[] values, int id) =>
        id > 0 ? values[id] as PaddleTensor : null;

    private static PaddleTensor[] List(object?[] values, int id) => values[id] switch
    {
        PaddleTensor[] list => list,
        PaddleTensor tensor => [tensor],
        _ => throw new InvalidOperationException($"Value %{id} is not a tensor list."),
    };

    /// <inheritdoc />
    public void Dispose()
    {
        _constants.Clear();
        _parameters.Dispose();
    }
}

/// <summary>
/// Accumulates per-operator wall-clock time, for finding the expensive part of a graph.
/// </summary>
public sealed class PirProfile
{
    private readonly Dictionary<string, (TimeSpan Elapsed, int Count)> _entries = new(StringComparer.Ordinal);

    /// <summary>Records one operator's execution.</summary>
    public void Add(string name, TimeSpan elapsed)
    {
        _entries.TryGetValue(name, out (TimeSpan Elapsed, int Count) entry);
        _entries[name] = (entry.Elapsed + elapsed, entry.Count + 1);
    }

    /// <summary>Per-operator totals, slowest first.</summary>
    public IEnumerable<(string Name, TimeSpan Elapsed, int Count)> ByCost() => _entries
        .Select(entry => (entry.Key, entry.Value.Elapsed, entry.Value.Count))
        .OrderByDescending(entry => entry.Elapsed);

    /// <summary>Total time across every operator.</summary>
    public TimeSpan Total => _entries.Values.Aggregate(TimeSpan.Zero, (sum, entry) => sum + entry.Elapsed);

    /// <summary>A human-readable table of the costliest operators.</summary>
    public string Report(int top = 15)
    {
        var builder = new System.Text.StringBuilder();
        TimeSpan total = Total;
        builder.AppendLine($"{"operator",-24}{"total",10}{"calls",8}{"share",8}");
        foreach ((string name, TimeSpan elapsed, int count) in ByCost().Take(top))
        {
            double share = total > TimeSpan.Zero ? elapsed / total : 0;
            builder.AppendLine($"{name,-24}{elapsed.TotalMilliseconds,9:F0}ms{count,8}{share,7:P1}");
        }

        builder.AppendLine($"{"total",-24}{total.TotalMilliseconds,9:F0}ms");
        return builder.ToString();
    }
}

/// <summary>Receives intermediate values while a program runs, for parity debugging.</summary>
public interface IPirTrace
{
    /// <summary>Records the results of one operation.</summary>
    /// <param name="index">Position of the operation in the program.</param>
    /// <param name="operation">The operation that just ran.</param>
    /// <param name="results">Its results; entries may be tensors, tensor lists or null.</param>
    void Record(int index, PirOperation operation, object?[] results);
}
