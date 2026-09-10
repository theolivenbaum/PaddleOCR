using PaddleOcrSharp.Core;
namespace PaddleOcrSharp.Models.Paddle.Ops;

/// <summary>
/// A small Einstein-summation evaluator, enough for the contractions the exported graphs use.
/// </summary>
/// <remarks>
/// Explicit output notation (<c>lhs-&gt;rhs</c>) and repeated labels are supported; ellipses are
/// not, because no shipped program uses them.
/// </remarks>
internal static class EinsumOps
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

        // The summed axes are the tail of `all`, so each output element owns one contiguous run
        // of the walk — which is what lets the walk be split, and lets each operand's offset be
        // carried forward instead of recomputed from the counters for every term. The graph's one
        // call is a 300x300 product over 256, 23 million terms, and recomputing two operands'
        // offsets from three axes apiece for each of them cost 88 ms.
        int outputCount = output.Length;
        int summedTotal = 1;
        for (int axis = outputCount; axis < all.Length; axis++)
        {
            summedTotal *= extents[axis];
        }

        int outputTotal = summedTotal == 0 ? 0 : total / Math.Max(1, summedTotal);
        int operandCount = operands.Count;

        Parallelism.Chunked(outputTotal, (from, to) =>
        {
            int[] counters = new int[all.Length];
            int[] offsets = new int[operandCount];

            int remainder = from;
            for (int axis = outputCount - 1; axis >= 0; axis--)
            {
                counters[axis] = extents[axis] == 0 ? 0 : remainder % extents[axis];
                remainder = extents[axis] == 0 ? 0 : remainder / extents[axis];
            }

            for (int o = from; o < to; o++)
            {
                for (int operand = 0; operand < operandCount; operand++)
                {
                    int offset = 0;
                    for (int axis = 0; axis < outputCount; axis++)
                    {
                        offset += counters[axis] * operandStrides[operand][axis];
                    }

                    offsets[operand] = offset;
                }

                for (int axis = outputCount; axis < all.Length; axis++)
                {
                    counters[axis] = 0;
                }

                // float, term by term, in this order: the original accumulated into a zeroed
                // float tensor, and a double accumulator here would not give the same answer.
                float sum = 0f;
                for (int s = 0; s < summedTotal; s++)
                {
                    double product = 1;
                    for (int operand = 0; operand < operandCount; operand++)
                    {
                        product *= operands[operand].GetDouble(offsets[operand]);
                    }

                    sum += (float)product;

                    for (int axis = all.Length - 1; axis >= outputCount; axis--)
                    {
                        counters[axis]++;
                        for (int operand = 0; operand < operandCount; operand++)
                        {
                            offsets[operand] += operandStrides[operand][axis];
                        }

                        if (counters[axis] < extents[axis])
                        {
                            break;
                        }

                        for (int operand = 0; operand < operandCount; operand++)
                        {
                            offsets[operand] -= operandStrides[operand][axis] * extents[axis];
                        }

                        counters[axis] = 0;
                    }
                }

                result.Floats![o] = sum;

                for (int axis = outputCount - 1; axis >= 0; axis--)
                {
                    if (++counters[axis] < extents[axis])
                    {
                        break;
                    }

                    counters[axis] = 0;
                }
            }
        });

        return result;
    }
}
