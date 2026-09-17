using PaddleOcrSharp.Core;
using PaddleOcrSharp.Formats.Gguf;

namespace PaddleOcrSharp.Quantization;

/// <summary>Settings for one tensor's conversion.</summary>
/// <param name="Scheme">The block layout to pack into.</param>
/// <param name="ScaleSearchSteps">
/// How many candidate scales to try per group. <c>0</c> takes <c>amax</c>, which is exactly what
/// the fork's reference encoder does and is right for weights that are already ternary; anything
/// above that searches, which is what real-valued weights need.
/// </param>
/// <param name="ScaleSearchFloor">The smallest candidate, as a fraction of <c>amax</c>.</param>
/// <param name="Damping">
/// GPTQ ridge, as a fraction of the mean diagonal of the Hessian. Ignored without one.
/// </param>
/// <param name="BlockColumns">How many columns GPTQ carries error across before catching up.</param>
/// <remarks>
/// A record class rather than a record struct, deliberately. A struct's parameterless constructor
/// bypasses the primary one and zeroes every field, so <c>new QuantizationOptions()</c> would
/// silently mean <see cref="GgmlType.F32"/> — a default that is not a ternary layout at all, and
/// one that only shows up several calls later as "not a ternary block type".
/// </remarks>
public sealed record QuantizationOptions(
    GgmlType Scheme = GgmlType.PTQ1_0,
    int ScaleSearchSteps = 24,
    float ScaleSearchFloor = 0.35f,
    double Damping = 0.01,
    int BlockColumns = 128);

/// <summary>What one tensor's conversion cost, in error and in bytes.</summary>
/// <param name="Name">Tensor name.</param>
/// <param name="Scheme">The layout it was packed into.</param>
/// <param name="Rows">Output features.</param>
/// <param name="Cols">Input features.</param>
/// <param name="Rotated">Whether it was folded into the rotated basis.</param>
/// <param name="RelativeError">
/// <c>‖W − Ŵ‖_F / ‖W‖_F</c> — how much of the tensor's energy the quantizer lost.
/// </param>
/// <param name="WorstRowCosine">
/// The <i>smallest</i> cosine similarity over the output rows. The mean hides the failure that
/// matters: one destroyed row is one destroyed attention head or one destroyed logit, and a
/// thousand healthy rows do not make up for it.
/// </param>
/// <param name="ZeroFraction">Share of weights that quantized to the zero trit.</param>
/// <param name="Bytes">Bytes the packed tensor occupies.</param>
public readonly record struct TensorQuantizationReport(
    string Name,
    GgmlType Scheme,
    int Rows,
    int Cols,
    bool Rotated,
    double RelativeError,
    double WorstRowCosine,
    double ZeroFraction,
    long Bytes)
{
    /// <summary>Bits per weight the packed tensor actually costs.</summary>
    public double BitsPerWeight => Bytes * 8.0 / ((long)Rows * Cols);
}

/// <summary>
/// Turns a float weight matrix into ternary codes and packs it.
/// </summary>
/// <remarks>
/// <para>
/// This is the part of Bonsai's pipeline that does <b>not</b> transfer. Their ternary values come
/// out of quantization-aware training; <c>quantize_row_ptq1_0_ref</c> in the fork is a container
/// writer that assumes the weights arriving are already ternary at group 128, which is why it takes
/// <c>d = amax</c> and discards an imatrix argument with the comment that "ternary codes come from
/// the weights themselves". Ours will never arrive that way, so the choice of trits is a real
/// optimisation problem and this is where it is solved.
/// </para>
/// <para>
/// Three levers, in the order they matter:
/// </para>
/// <list type="number">
/// <item>
/// <description>
/// <b>The rotation</b> (<see cref="HadamardRotation"/>), applied before anything here sees the
/// weights. Sub-2-bit error is set by the ratio of a group's largest magnitude to its typical one,
/// and spreading each coordinate over a 1024-wide block is the cheapest way to shrink it.
/// </description>
/// </item>
/// <item>
/// <description>
/// <b>The scale.</b> <c>amax</c> is optimal only for an already-ternary group: for a real-valued
/// one it spends the whole range on the single largest weight and rounds most of the rest to zero.
/// The scale that minimises squared error is smaller, and it is found by trying a grid — there is
/// no closed form, because the assignment changes as the scale moves.
/// </description>
/// </item>
/// <item>
/// <description>
/// <b>Error feedback.</b> With a Hessian <c>H = Σ xᵀx</c> from real activations, GPTQ quantizes
/// column by column and pushes each column's error into the columns not yet decided, weighted by
/// <c>H⁻¹</c>. It does not reduce the error in the weights; it moves the error into directions the
/// activations do not visit.
/// </description>
/// </item>
/// </list>
/// </remarks>
public static class TernaryQuantizer
{
    /// <summary>
    /// Quantizes and packs <paramref name="weights"/>.
    /// </summary>
    /// <param name="name">Tensor name, for the report.</param>
    /// <param name="weights">
    /// <c>[rows, cols]</c>, row-major. Overwritten: folded in place when
    /// <paramref name="rotation"/> is given, and used as scratch by the GPTQ path.
    /// </param>
    /// <param name="rows">Output features.</param>
    /// <param name="cols">Input features; must be a multiple of the layout's group size.</param>
    /// <param name="options">Conversion settings.</param>
    /// <param name="rotation">The basis to fold into, or <see langword="null"/>.</param>
    /// <param name="hessian">
    /// <c>[cols, cols]</c> row-major <c>Σ xᵀx</c> over a calibration set, already in the rotated
    /// basis when <paramref name="rotation"/> is given, or <see langword="null"/> for plain
    /// round-to-nearest.
    /// </param>
    /// <param name="packed">Receives the packed blocks.</param>
    public static TensorQuantizationReport Quantize(
        string name,
        Span<float> weights,
        int rows,
        int cols,
        QuantizationOptions options,
        HadamardRotation? rotation,
        double[]? hessian,
        Span<byte> packed)
    {
        int group = options.Scheme.BlockSize();
        if (cols % group != 0)
        {
            throw new ArgumentException(
                $"'{name}' has {cols} input features, which is not a multiple of the {group}-weight group.",
                nameof(cols));
        }

        if (rotation is not null)
        {
            rotation.Fold(weights, rows, cols);
        }

        using PooledBuffer original = TensorPool.Rent(rows * cols);
        weights[..(rows * cols)].CopyTo(original.Span);

        if (hessian is not null)
        {
            QuantizeWithFeedback(weights, rows, cols, options, hessian);
        }
        else
        {
            QuantizeRowwise(weights, rows, cols, options);
        }

        // `weights` now holds the dequantized values; packing them is exact, because every group
        // is already ternary at this group size and `amax` recovers the scale it was built from.
        int rowBytes = checked((int)options.Scheme.RowSize(cols));
        for (int r = 0; r < rows; r++)
        {
            TernaryBlocks.Encode(
                options.Scheme,
                weights.Slice(r * cols, cols),
                packed.Slice(r * rowBytes, rowBytes));
        }

        return Measure(name, options.Scheme, original.Span, weights, rows, cols, rotation is not null, (long)rows * rowBytes);
    }

    /// <summary>
    /// Replaces every group with its best ternary approximation, row by row.
    /// </summary>
    private static void QuantizeRowwise(Span<float> weights, int rows, int cols, QuantizationOptions options)
    {
        int group = options.Scheme.BlockSize();
        for (int r = 0; r < rows; r++)
        {
            Span<float> row = weights.Slice(r * cols, cols);
            for (int offset = 0; offset < cols; offset += group)
            {
                QuantizeGroup(row.Slice(offset, group), options);
            }
        }
    }

    /// <summary>
    /// Replaces one group with <c>t · s</c>, choosing <c>s</c> to minimise the squared error.
    /// </summary>
    /// <remarks>
    /// At a candidate scale the assignment is fixed — <c>t = clamp(round(w/s), −1, 1)</c> — so the
    /// error is a piecewise-quadratic function of <c>s</c> with breakpoints wherever a weight
    /// changes trit. Rather than enumerate those, the grid walks down from <c>amax</c>, which is
    /// what every published low-bit quantizer does and is enough: the optimum sits well inside the
    /// interval and the function is flat around it.
    /// </remarks>
    private static void QuantizeGroup(Span<float> group, QuantizationOptions options)
    {
        float amax = 0f;
        for (int i = 0; i < group.Length; i++)
        {
            amax = MathF.Max(amax, MathF.Abs(group[i]));
        }

        if (amax == 0f)
        {
            return;
        }

        float bestScale = amax;
        double bestError = GroupError(group, amax);

        for (int step = 1; step <= options.ScaleSearchSteps; step++)
        {
            float fraction = 1f - ((1f - options.ScaleSearchFloor) * step / options.ScaleSearchSteps);
            float candidate = amax * fraction;
            double error = GroupError(group, candidate);
            if (error < bestError)
            {
                bestError = error;
                bestScale = candidate;
            }
        }

        // The scale is stored as fp16, so quantize against the value the decoder will actually see.
        float scale = (float)(Half)bestScale;
        if (scale <= 0f)
        {
            scale = amax;
        }

        for (int i = 0; i < group.Length; i++)
        {
            group[i] = Trit(group[i], scale) * scale;
        }
    }

    private static double GroupError(ReadOnlySpan<float> group, float scale)
    {
        if (scale <= 0f)
        {
            return double.PositiveInfinity;
        }

        double error = 0;
        for (int i = 0; i < group.Length; i++)
        {
            double difference = group[i] - (Trit(group[i], scale) * scale);
            error += difference * difference;
        }

        return error;
    }

    private static float Trit(float value, float scale) =>
        Math.Clamp(MathF.Round(value / scale, MidpointRounding.AwayFromZero), -1f, 1f);

    /// <summary>
    /// GPTQ: quantize the columns in order, pushing each column's error into the ones still to be
    /// decided.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The objective is <c>‖(W − Ŵ) X‖²</c> rather than <c>‖W − Ŵ‖²</c> — what matters is the
    /// error the activations actually see. With <c>H = X Xᵀ</c> and the columns fixed in order,
    /// the optimal update after deciding column <c>i</c> is to subtract
    /// <c>(wᵢ − ŵᵢ) / [H⁻¹]ᵢᵢ</c> times row <c>i</c> of the Cholesky factor of <c>H⁻¹</c> from the
    /// remaining columns.
    /// </para>
    /// <para>
    /// Group scales are chosen at the group boundary from the weights <i>as they then stand</i>,
    /// which is the whole reason the error feedback can be absorbed: a later column's scale already
    /// accounts for the correction it received.
    /// </para>
    /// </remarks>
    private static void QuantizeWithFeedback(
        Span<float> weights,
        int rows,
        int cols,
        QuantizationOptions options,
        double[] hessian)
    {
        int group = options.Scheme.BlockSize();
        double[] inverse = CholeskyOfInverse(hessian, cols, options.Damping);

        using PooledBuffer column = TensorPool.Rent(rows);
        using PooledBuffer error = TensorPool.Rent(rows * options.BlockColumns);

        for (int start = 0; start < cols; start += options.BlockColumns)
        {
            int width = Math.Min(options.BlockColumns, cols - start);
            Span<float> errors = error.Span[..(rows * width)];
            errors.Clear();

            for (int local = 0; local < width; local++)
            {
                int i = start + local;

                if (i % group == 0)
                {
                    // A group's scale is decided once, from the current state of its columns, and
                    // then used for every column in it — which is what the block layout stores.
                    DecideGroupScales(weights, rows, cols, i, group, options, column.Span);
                }

                double diagonal = inverse[(i * cols) + i];
                for (int r = 0; r < rows; r++)
                {
                    int index = (r * cols) + i;
                    float original = weights[index];
                    float scale = column.Span[r];
                    float quantized = scale > 0f ? Trit(original, scale) * scale : 0f;
                    weights[index] = quantized;
                    errors[(r * width) + local] = (float)((original - quantized) / diagonal);
                }

                // Carry this column's error into the rest of the block.
                for (int next = local + 1; next < width; next++)
                {
                    float coefficient = (float)inverse[(i * cols) + start + next];
                    if (coefficient == 0f)
                    {
                        continue;
                    }

                    for (int r = 0; r < rows; r++)
                    {
                        weights[(r * cols) + start + next] -= errors[(r * width) + local] * coefficient;
                    }
                }
            }

            // Then into every column after the block, one pass instead of one per column.
            for (int local = 0; local < width; local++)
            {
                int i = start + local;
                for (int next = start + width; next < cols; next++)
                {
                    float coefficient = (float)inverse[(i * cols) + next];
                    if (coefficient == 0f)
                    {
                        continue;
                    }

                    for (int r = 0; r < rows; r++)
                    {
                        weights[(r * cols) + next] -= errors[(r * width) + local] * coefficient;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Picks each row's scale for the group starting at <paramref name="start"/>.
    /// </summary>
    private static void DecideGroupScales(
        ReadOnlySpan<float> weights,
        int rows,
        int cols,
        int start,
        int group,
        QuantizationOptions options,
        Span<float> scales)
    {
        Span<float> scratch = stackalloc float[Math.Min(group, 256)];
        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> slice = weights.Slice((r * cols) + start, group);
            slice.CopyTo(scratch[..group]);
            QuantizeGroup(scratch[..group], options);

            // QuantizeGroup wrote t·s; the scale is the largest magnitude it produced.
            float scale = 0f;
            for (int i = 0; i < group; i++)
            {
                scale = MathF.Max(scale, MathF.Abs(scratch[i]));
            }

            scales[r] = scale;
        }
    }

    /// <summary>
    /// Returns the upper Cholesky factor of <c>H⁻¹</c>, damped, as a dense <c>[n, n]</c> array.
    /// </summary>
    /// <remarks>
    /// Columns the calibration never excited have a zero diagonal and no defined inverse. Zeroing
    /// the weight there — as the reference GPTQ does — would be a decision the data does not
    /// support, so the diagonal is set to one instead, which makes the column quantize on its own
    /// with no feedback either way.
    /// </remarks>
    private static double[] CholeskyOfInverse(double[] hessian, int n, double damping)
    {
        double[] matrix = new double[(long)n * n];
        Array.Copy(hessian, matrix, matrix.Length);

        double mean = 0;
        for (int i = 0; i < n; i++)
        {
            mean += matrix[(i * n) + i];
        }

        mean /= n;
        double ridge = Math.Max(mean * damping, 1e-8);

        for (int i = 0; i < n; i++)
        {
            if (matrix[(i * n) + i] <= 0)
            {
                for (int j = 0; j < n; j++)
                {
                    matrix[(i * n) + j] = 0;
                    matrix[(j * n) + i] = 0;
                }

                matrix[(i * n) + i] = 1;
                continue;
            }

            matrix[(i * n) + i] += ridge;
        }

        Cholesky(matrix, n);
        InvertFromCholesky(matrix, n);
        Cholesky(matrix, n);

        // Re-expose the factor as upper-triangular rows, which is the orientation the sweep reads.
        double[] upper = new double[(long)n * n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                upper[(i * n) + j] = matrix[(j * n) + i];
            }
        }

        return upper;
    }

    /// <summary>In-place lower Cholesky factorisation; the upper triangle is zeroed.</summary>
    private static void Cholesky(double[] matrix, int n)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = matrix[(i * n) + j];
                for (int k = 0; k < j; k++)
                {
                    sum -= matrix[(i * n) + k] * matrix[(j * n) + k];
                }

                if (i == j)
                {
                    matrix[(i * n) + i] = Math.Sqrt(Math.Max(sum, 1e-12));
                }
                else
                {
                    matrix[(i * n) + j] = sum / matrix[(j * n) + j];
                }
            }

            for (int j = i + 1; j < n; j++)
            {
                matrix[(i * n) + j] = 0;
            }
        }
    }

    /// <summary>Replaces a lower Cholesky factor with the inverse of the matrix it came from.</summary>
    private static void InvertFromCholesky(double[] lower, int n)
    {
        // Invert the factor in place: L becomes L⁻¹, still lower triangular.
        for (int i = 0; i < n; i++)
        {
            lower[(i * n) + i] = 1.0 / lower[(i * n) + i];
            for (int j = 0; j < i; j++)
            {
                double sum = 0;
                for (int k = j; k < i; k++)
                {
                    sum -= lower[(i * n) + k] * lower[(k * n) + j];
                }

                lower[(i * n) + j] = sum * lower[(i * n) + i];
            }
        }

        // Then H⁻¹ = L⁻ᵀ L⁻¹, which is symmetric, so only the lower triangle is computed.
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = 0;
                for (int k = i; k < n; k++)
                {
                    sum += lower[(k * n) + i] * lower[(k * n) + j];
                }

                lower[(i * n) + j] = sum;
            }
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                lower[(i * n) + j] = lower[(j * n) + i];
            }
        }
    }

    private static TensorQuantizationReport Measure(
        string name,
        GgmlType scheme,
        ReadOnlySpan<float> original,
        ReadOnlySpan<float> quantized,
        int rows,
        int cols,
        bool rotated,
        long bytes)
    {
        double total = 0;
        double energy = 0;
        double worst = 1.0;
        long zeros = 0;

        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> a = original.Slice(r * cols, cols);
            ReadOnlySpan<float> b = quantized.Slice(r * cols, cols);

            double dot = 0;
            double normA = 0;
            double normB = 0;

            for (int i = 0; i < cols; i++)
            {
                double difference = a[i] - b[i];
                total += difference * difference;
                energy += (double)a[i] * a[i];
                dot += (double)a[i] * b[i];
                normA += (double)a[i] * a[i];
                normB += (double)b[i] * b[i];
                if (b[i] == 0f)
                {
                    zeros++;
                }
            }

            if (normA > 0 && normB > 0)
            {
                worst = Math.Min(worst, dot / Math.Sqrt(normA * normB));
            }
        }

        return new TensorQuantizationReport(
            name,
            scheme,
            rows,
            cols,
            rotated,
            energy > 0 ? Math.Sqrt(total / energy) : 0,
            worst,
            (double)zeros / ((long)rows * cols),
            bytes);
    }
}
