namespace PaddleOcrSharp.Quantization;

/// <summary>
/// Accumulates <c>H = Σ xᵀx</c> per weight while the unquantized model runs, which is the input
/// GPTQ needs and the only thing about a checkpoint that a converter cannot get from the file.
/// </summary>
/// <remarks>
/// <para>
/// A recorder is installed for a scope with <see cref="Use"/>; while one is active
/// <see cref="Core.Gemm.Linear"/> hands it every activation that reaches a named weight. Outside a
/// scope the cost is one null check per matrix product, which is nothing beside the product.
/// </para>
/// <para>
/// It is an <see cref="AsyncLocal{T}"/> for the same reason <c>Parallelism</c> is: a block's tower
/// and decoder run on pool threads when <c>BlockConcurrency</c> is above one, where a
/// thread-static set on the calling thread would not be seen. Accumulation is therefore locked per
/// tensor — calibration is an offline pass over a handful of pages, so the contention is
/// irrelevant and losing an update would not be.
/// </para>
/// <para>
/// A Hessian is <c>cols²</c> doubles: 8.5 MB at the decoder's 1024, 148 MB at the projector's
/// 4608. <see cref="Use"/> therefore takes the set of weights to record, so a calibration pass
/// holds only what the policy is actually going to quantize with feedback.
/// </para>
/// </remarks>
public sealed class ActivationRecorder
{
    private static readonly AsyncLocal<ActivationRecorder?> Scope = new();

    private readonly Dictionary<string, Accumulator> _accumulators = new(StringComparer.Ordinal);

    private ActivationRecorder(IEnumerable<string> names)
    {
        foreach (string name in names)
        {
            _accumulators[name] = new Accumulator();
        }
    }

    /// <summary>The recorder installed for the current scope, if any.</summary>
    public static ActivationRecorder? Current => Scope.Value;

    /// <summary>The weights this recorder is collecting for.</summary>
    public IReadOnlyCollection<string> Names => _accumulators.Keys;

    /// <summary>Installs a recorder until the returned scope is disposed.</summary>
    /// <param name="names">The weight names to accumulate Hessians for.</param>
    /// <returns>A scope that removes the recorder, and the recorder itself.</returns>
    public static (ActivationRecorder Recorder, IDisposable Scope) Use(IEnumerable<string> names)
    {
        var recorder = new ActivationRecorder(names);
        ActivationRecorder? previous = Scope.Value;
        Scope.Value = recorder;
        return (recorder, new Restore(previous));
    }

    /// <summary>
    /// Accumulates <c>xᵀx</c> for <paramref name="name"/>.
    /// </summary>
    /// <param name="name">Weight name; ignored when this recorder does not track it.</param>
    /// <param name="activations"><c>[rows, cols]</c>, row-major.</param>
    /// <param name="rows">Number of activation rows.</param>
    /// <param name="cols">Row length, which is the weight's input width.</param>
    public void Record(string name, ReadOnlySpan<float> activations, int rows, int cols)
    {
        if (!_accumulators.TryGetValue(name, out Accumulator? accumulator))
        {
            return;
        }

        accumulator.Add(activations, rows, cols);
    }

    /// <summary>
    /// The Hessian for <paramref name="name"/> as a dense <c>[cols, cols]</c> array, normalised by
    /// the number of rows seen, or <see langword="null"/> when nothing reached it.
    /// </summary>
    /// <param name="name">Weight name.</param>
    public double[]? Hessian(string name) =>
        _accumulators.TryGetValue(name, out Accumulator? accumulator) ? accumulator.Result() : null;

    /// <summary>How many activation rows were seen for <paramref name="name"/>.</summary>
    /// <param name="name">Weight name.</param>
    public long Samples(string name) =>
        _accumulators.TryGetValue(name, out Accumulator? accumulator) ? accumulator.Rows : 0;

    private sealed class Accumulator
    {
        private readonly Lock _gate = new();
        private double[]? _sum;

        public long Rows { get; private set; }

        public void Add(ReadOnlySpan<float> activations, int rows, int cols)
        {
            lock (_gate)
            {
                _sum ??= new double[(long)cols * cols];
                if (_sum.Length != (long)cols * cols)
                {
                    throw new InvalidOperationException(
                        "The same weight was seen with two different input widths during calibration.");
                }

                for (int r = 0; r < rows; r++)
                {
                    ReadOnlySpan<float> row = activations.Slice(r * cols, cols);
                    for (int i = 0; i < cols; i++)
                    {
                        double value = row[i];
                        if (value == 0)
                        {
                            continue;
                        }

                        int offset = i * cols;
                        for (int j = i; j < cols; j++)
                        {
                            _sum[offset + j] += value * row[j];
                        }
                    }
                }

                Rows += rows;
            }
        }

        public double[]? Result()
        {
            lock (_gate)
            {
                if (_sum is null || Rows == 0)
                {
                    return null;
                }

                int cols = (int)Math.Round(Math.Sqrt(_sum.Length));
                double[] result = new double[_sum.Length];
                double inverse = 1.0 / Rows;

                for (int i = 0; i < cols; i++)
                {
                    for (int j = i; j < cols; j++)
                    {
                        double value = _sum[(i * cols) + j] * inverse;
                        result[(i * cols) + j] = value;
                        result[(j * cols) + i] = value;
                    }
                }

                return result;
            }
        }
    }

    private sealed class Restore(ActivationRecorder? previous) : IDisposable
    {
        public void Dispose() => Scope.Value = previous;
    }
}
