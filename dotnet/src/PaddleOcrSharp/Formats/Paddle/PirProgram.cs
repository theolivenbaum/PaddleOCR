using System.Text.Json;

namespace PaddleOcrSharp.Formats.Paddle;

/// <summary>An attribute value attached to a PIR operation.</summary>
public sealed class PirAttribute
{
    /// <summary>Attribute name, e.g. <c>strides</c>.</summary>
    public required string Name { get; init; }

    /// <summary>PIR type tag, e.g. <c>0.a_i32</c>.</summary>
    public required string Kind { get; init; }

    /// <summary>Raw value: a boxed scalar, a string, or a <see cref="PirAttribute"/> array.</summary>
    public object? Value { get; init; }

    /// <summary>Reads the attribute as a boolean.</summary>
    public bool AsBool() => Value is bool value ? value : Convert.ToBoolean(Value);

    /// <summary>Reads the attribute as a 32-bit integer.</summary>
    public int AsInt() => Convert.ToInt32(Value);

    /// <summary>Reads the attribute as a 64-bit integer.</summary>
    public long AsLong() => Convert.ToInt64(Value);

    /// <summary>Reads the attribute as a double.</summary>
    public double AsDouble() => Convert.ToDouble(Value);

    /// <summary>Reads the attribute as a string.</summary>
    public string AsString() => Value as string ?? Value?.ToString() ?? string.Empty;

    /// <summary>Reads the attribute as an integer array, accepting both arrays and int-arrays.</summary>
    public long[] AsLongArray() => Value switch
    {
        long[] values => values,
        PirAttribute[] items => [.. items.Select(item => item.AsLong())],
        null => [],
        _ => [AsLong()],
    };

    /// <summary>Reads the attribute as a 32-bit integer array.</summary>
    public int[] AsIntArray() => [.. AsLongArray().Select(value => (int)value)];

    /// <summary>Reads the attribute as a double array.</summary>
    public double[] AsDoubleArray() => Value switch
    {
        long[] values => [.. values.Select(value => (double)value)],
        PirAttribute[] items => [.. items.Select(item => item.AsDouble())],
        null => [],
        _ => [AsDouble()],
    };
}

/// <summary>One operation in a PIR block.</summary>
public sealed class PirOperation
{
    /// <summary>Operation name without the dialect version, e.g. <c>conv2d</c>.</summary>
    public required string Name { get; init; }

    /// <summary>Ids of the values consumed.</summary>
    public required int[] Inputs { get; init; }

    /// <summary>Ids of the values produced.</summary>
    public required int[] Outputs { get; init; }

    /// <summary>Declared type of each output.</summary>
    public required PirValueType[] OutputTypes { get; init; }

    /// <summary>Attributes, keyed by name.</summary>
    public required IReadOnlyDictionary<string, PirAttribute> Attributes { get; init; }

    /// <summary>Parameter name for <c>builtin.parameter</c> operations.</summary>
    public string? ParameterName { get; init; }

    /// <summary>Gets an attribute, or <see langword="null"/> when absent.</summary>
    public PirAttribute? Attribute(string name) =>
        Attributes.TryGetValue(name, out PirAttribute? attribute) ? attribute : null;

    /// <summary>The module path Paddle recorded, useful when tracing a mismatch.</summary>
    public string StructName => Attribute("struct_name")?.AsString() ?? string.Empty;

    /// <inheritdoc />
    public override string ToString() =>
        $"{Name}({string.Join(", ", Inputs.Select(id => $"%{id}"))}) -> {string.Join(", ", Outputs.Select(id => $"%{id}"))}";
}

/// <summary>Declared type of a PIR value.</summary>
/// <param name="IsVector">Whether the value is a list of tensors rather than a tensor.</param>
/// <param name="Dtype">Element type of the tensor (or of the first element of the list).</param>
/// <param name="Shape">Declared shape; <c>-1</c> marks a dynamic dimension.</param>
/// <param name="Elements">Element types when <paramref name="IsVector"/> is set.</param>
public readonly record struct PirValueType(
    bool IsVector,
    PaddleDType Dtype,
    int[] Shape,
    PirValueType[]? Elements);

/// <summary>
/// A parsed Paddle PIR inference program (<c>inference.json</c>).
/// </summary>
/// <remarks>
/// <para>
/// The exported JSON is a compact SSA listing: each operation names its dialect-qualified op
/// (<c>1.conv2d</c>), its attributes under <c>A</c>, its input value ids under <c>I</c> and its
/// results — with declared types — under <c>O</c>. Parameter declarations use the short form
/// <c>{"#": "p", "A": [.., .., .., name], "O": {...}}</c>.
/// </para>
/// <para>
/// Operations appear in a valid execution order, so an interpreter can run them in sequence.
/// </para>
/// </remarks>
public sealed class PirProgram
{
    private PirProgram(IReadOnlyList<PirOperation> operations)
    {
        Operations = operations;
        Parameters = [.. operations
            .Where(op => op.ParameterName is not null)
            .Select(op => op.ParameterName!)];
        Inputs = [.. operations
            .Where(op => op.Name == "data")
            .Select(op => op.Attribute("name")?.AsString() ?? string.Empty)];
    }

    /// <summary>Operations in execution order.</summary>
    public IReadOnlyList<PirOperation> Operations { get; }

    /// <summary>Names of every declared parameter, in program order.</summary>
    public IReadOnlyList<string> Parameters { get; }

    /// <summary>Names of the program's feed inputs.</summary>
    public IReadOnlyList<string> Inputs { get; }

    /// <summary>Parses <c>inference.json</c> from <paramref name="path"/>.</summary>
    public static PirProgram Load(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Parse(stream);
    }

    /// <summary>Parses a PIR program from a stream.</summary>
    public static PirProgram Parse(Stream stream)
    {
        using JsonDocument document = JsonDocument.Parse(
            stream,
            new JsonDocumentOptions { MaxDepth = 512 });

        JsonElement root = document.RootElement;
        JsonElement block = root
            .GetProperty("program")
            .GetProperty("regions")[0]
            .GetProperty("blocks")[0];

        var operations = new List<PirOperation>();
        foreach (JsonElement element in block.GetProperty("ops").EnumerateArray())
        {
            operations.Add(ParseOperation(element));
        }

        return new PirProgram(operations);
    }

    private static PirOperation ParseOperation(JsonElement element)
    {
        string tag = element.GetProperty("#").GetString()!;

        if (tag == "p")
        {
            // {"#": "p", "A": [scope, persistable, trainable, name], "O": {"%": id, "TT": ...}}
            JsonElement attributes = element.GetProperty("A");
            string name = attributes[attributes.GetArrayLength() - 1].GetString()!;
            JsonElement output = element.GetProperty("O");

            return new PirOperation
            {
                Name = "parameter",
                Inputs = [],
                Outputs = [output.GetProperty("%").GetInt32()],
                OutputTypes = [ParseType(output.GetProperty("TT"))],
                Attributes = new Dictionary<string, PirAttribute>(StringComparer.Ordinal),
                ParameterName = name,
            };
        }

        // Tags are "<dialect-version>.<op>". Version 0 is the builtin dialect and version 1 the
        // pd_op dialect, and both define a `split`, so the dialect has to survive into the name.
        int dot = tag.IndexOf('.');
        string opName = dot < 0
            ? tag
            : tag[..dot] == "0" ? "builtin." + tag[(dot + 1)..] : tag[(dot + 1)..];

        var inputs = new List<int>();
        if (element.TryGetProperty("I", out JsonElement inputElement)
            && inputElement.ValueKind is JsonValueKind.Array)
        {
            foreach (JsonElement input in inputElement.EnumerateArray())
            {
                inputs.Add(input.GetProperty("%").GetInt32());
            }
        }

        var outputs = new List<int>();
        var outputTypes = new List<PirValueType>();
        if (element.TryGetProperty("O", out JsonElement outputElement)
            && outputElement.ValueKind is JsonValueKind.Array)
        {
            foreach (JsonElement output in outputElement.EnumerateArray())
            {
                outputs.Add(output.TryGetProperty("%", out JsonElement id) ? id.GetInt32() : -1);
                outputTypes.Add(output.TryGetProperty("TT", out JsonElement type)
                    ? ParseType(type)
                    : new PirValueType(false, PaddleDType.Float32, [], null));
            }
        }

        var attributesByName = new Dictionary<string, PirAttribute>(StringComparer.Ordinal);
        if (element.TryGetProperty("A", out JsonElement attributeElement)
            && attributeElement.ValueKind is JsonValueKind.Array)
        {
            foreach (JsonElement entry in attributeElement.EnumerateArray())
            {
                if (entry.ValueKind is not JsonValueKind.Object || !entry.TryGetProperty("N", out JsonElement name))
                {
                    continue;
                }

                PirAttribute attribute = ParseAttribute(name.GetString()!, entry.GetProperty("AT"));
                attributesByName[attribute.Name] = attribute;
            }
        }

        return new PirOperation
        {
            Name = opName,
            Inputs = [.. inputs],
            Outputs = [.. outputs],
            OutputTypes = [.. outputTypes],
            Attributes = attributesByName,
        };
    }

    private static PirAttribute ParseAttribute(string name, JsonElement element)
    {
        string kind = element.GetProperty("#").GetString()!;
        JsonElement data = element.GetProperty("D");

        object? value = kind switch
        {
            "0.a_bool" => data.GetBoolean(),
            "0.a_i32" => data.GetInt32(),
            "0.a_i64" => data.GetInt64(),
            "0.a_f32" => data.GetSingle(),
            "0.a_f64" => data.GetDouble(),
            "0.a_str" => data.GetString(),
            "1.a_dtype" => data.GetString(),
            "1.a_intarray" => (object)data.EnumerateArray().Select(item => item.GetInt64()).ToArray(),
            "1.a_place" => data.ToString(),
            "0.a_array" => (object)data
                .EnumerateArray()
                .Select(item => ParseAttribute(name, item))
                .ToArray(),
            _ => data.ToString(),
        };

        return new PirAttribute { Name = name, Kind = kind, Value = value };
    }

    private static PirValueType ParseType(JsonElement element)
    {
        string kind = element.GetProperty("#").GetString()!;
        JsonElement data = element.GetProperty("D");

        if (kind == "0.t_vec")
        {
            PirValueType[] elements = [.. data.EnumerateArray().Select(ParseType)];
            return new PirValueType(
                true,
                elements.Length > 0 ? elements[0].Dtype : PaddleDType.Float32,
                [],
                elements);
        }

        if (kind == "0.t_dtensor")
        {
            PaddleDType dtype = PaddleDTypeExtensions.FromPirTag(data[0].GetProperty("#").GetString()!);
            int[] shape = [.. data[1].EnumerateArray().Select(item => item.GetInt32())];
            return new PirValueType(false, dtype, shape, null);
        }

        return new PirValueType(false, PaddleDType.Float32, [], null);
    }
}
