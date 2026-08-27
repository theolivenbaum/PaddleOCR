namespace PaddleOcrSharp.Models.Paddle.Ops;

/// <summary>
/// A small Einstein-summation evaluator, enough for the contractions the exported graphs use.
/// </summary>
/// <remarks>
/// Explicit output notation (<c>lhs-&gt;rhs</c>) and repeated labels are supported; ellipses are
/// not, because no shipped program uses them.
/// </remarks>
public static class EinsumOps
{
    /// <summary>Evaluates <paramref name="equation"/> over <paramref name="operands"/>.</summary>
    public static PaddleTensor Apply(string equation, IReadOnlyList<PaddleTensor> operands)
    {
        if (equation.Contains("...", StringComparison.Ordinal))
        {
            throw new NotSupportedException("Ellipsis notation is not supported in einsum equations.");
        }

        string[] sides = equation.Split("->", StringSplitOptions.TrimEntries);
        string[] inputs = sides[0].Split(',', StringSplitOptions.TrimEntries);

        if (inputs.Length != operands.Count)
        {
            throw new ArgumentException(
                $"Equation '{equation}' names {inputs.Length} operands but {operands.Count} were supplied.",
                nameof(operands));
        }

        var sizes = new Dictionary<char, int>();
        for (int i = 0; i < inputs.Length; i++)
        {
            string labels = inputs[i];
            for (int axis = 0; axis < labels.Length; axis++)
            {
                int size = operands[i].Shape[axis];
                if (sizes.TryGetValue(labels[axis], out int existing) && existing != size && existing != 1 && size != 1)
                {
                    throw new InvalidOperationException(
                        $"Label '{labels[axis]}' has conflicting extents {existing} and {size}.");
                }

                sizes[labels[axis]] = Math.Max(existing, size);
            }
        }

        string output = sides.Length > 1
            ? sides[1]
            : new string([.. sizes.Keys
                .Where(label => inputs.Sum(labels => labels.Count(c => c == label)) == 1)
                .OrderBy(label => label)]);

        char[] summed = [.. sizes.Keys.Where(label => !output.Contains(label))];
        char[] all = [.. output, .. summed];

        int[] outputShape = [.. output.Select(label => sizes[label])];
        PaddleTensor result = PaddleTensor.Zeros(outputShape);

        int[][] operandStrides = new int[operands.Count][];
        for (int i = 0; i < operands.Count; i++)
        {
            int[] strides = PaddleTensor.Strides(operands[i].Shape);
            int[] mapped = new int[all.Length];
            for (int axis = 0; axis < inputs[i].Length; axis++)
            {
                int position = Array.IndexOf(all, inputs[i][axis]);
                mapped[position] += operands[i].Shape[axis] == 1 ? 0 : strides[axis];
            }

            operandStrides[i] = mapped;
        }

        int[] extents = [.. all.Select(label => sizes[label])];
        int total = 1;
        foreach (int extent in extents)
        {
            total *= extent;
        }

        int outputStride = 1;
        int[] outputStrides = new int[all.Length];
        for (int axis = output.Length - 1; axis >= 0; axis--)
        {
            outputStrides[axis] = outputStride;
            outputStride *= extents[axis];
        }

        int[] counters = new int[all.Length];
        for (int i = 0; i < total; i++)
        {
            double product = 1;
            for (int operand = 0; operand < operands.Count; operand++)
            {
                int offset = 0;
                for (int axis = 0; axis < all.Length; axis++)
                {
                    offset += counters[axis] * operandStrides[operand][axis];
                }

                product *= operands[operand].GetDouble(offset);
            }

            int target = 0;
            for (int axis = 0; axis < output.Length; axis++)
            {
                target += counters[axis] * outputStrides[axis];
            }

            result.Floats![target] += (float)product;

            for (int axis = all.Length - 1; axis >= 0; axis--)
            {
                if (++counters[axis] < extents[axis])
                {
                    break;
                }

                counters[axis] = 0;
            }
        }

        return result;
    }
}
