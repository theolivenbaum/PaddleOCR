using System.Runtime.CompilerServices;
using System.Numerics;
using System.Runtime.Intrinsics;
using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Quantization;

/// <summary>
/// The fixed blockwise rotation a ternary checkpoint's weights are stored in, and the matching
/// activation-side transform.
/// </summary>
/// <remarks>
/// <para>
/// A weight matrix is folded as <c>W ← W Rᵀ</c> with <c>R = (1/√n) Hₙ S</c>, so that the product
/// the model wants, <c>W x</c>, is recovered as <c>(W Rᵀ)(R x)</c>. <c>Hₙ</c> is the
/// Sylvester–Walsh–Hadamard matrix — <c>H[r][c] = −1</c> exactly when <c>popcount(r &amp; c)</c>
/// is odd — and <c>S</c> is a fixed diagonal of ±1. <c>R</c> is orthogonal, so the fold is exact
/// in real arithmetic and the model function is unchanged.
/// </para>
/// <para>
/// The point is what it does to the <i>distribution</i>. Quantization error at a given bit width is
/// set by the ratio between a group's largest magnitude and its typical one, and a rotation spreads
/// each coordinate over its whole block, which is what makes sub-2-bit post-training quantization
/// work at all. It is the one part of Bonsai's approach that transfers without retraining.
/// </para>
/// <para>
/// This mirrors <c>PrismML-Eng/llama.cpp</c>: the transform is <b>sign flip first, then the
/// normalized rotation</b> (<c>llm_graph_context::build_lora_mm</c> and the comment on
/// <c>llama_hadamard_transform</c>), and the rotation matrix there is built from the same parity
/// rule (<c>llama-model.cpp</c>). The fork materialises an <c>n × n</c> matrix and multiplies; this
/// does the same arithmetic as an in-place fast Walsh–Hadamard transform, which is
/// <c>O(n log n)</c> instead of <c>O(n²)</c> and produces the same values because both are sums
/// and differences of the same terms in the same order.
/// </para>
/// </remarks>
public sealed class HadamardRotation
{
    private readonly Dictionary<int, float[]> _signs;

    /// <summary>Creates a rotation.</summary>
    /// <param name="blockSize">Rotation block, a power of two; Bonsai 2 uses 1024.</param>
    /// <param name="signs">
    /// The ±1 diagonals keyed by the input width they apply to, or <see langword="null"/> for the
    /// identity sign mode. Each vector's length must be a multiple of <paramref name="blockSize"/>.
    /// </param>
    public HadamardRotation(int blockSize, IReadOnlyDictionary<int, float[]>? signs = null)
    {
        if (blockSize <= 0 || (blockSize & (blockSize - 1)) != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(blockSize), blockSize, "Block size must be a power of two.");
        }

        BlockSize = blockSize;
        _signs = [];

        if (signs is null)
        {
            return;
        }

        foreach ((int width, float[] vector) in signs)
        {
            if (vector.Length != width || width % blockSize != 0)
            {
                throw new ArgumentException(
                    $"Sign vector for width {width} must have {width} entries and a width that is a multiple of {blockSize}.",
                    nameof(signs));
            }

            _signs[width] = vector;
        }
    }

    /// <summary>The rotation block size.</summary>
    public int BlockSize { get; }

    /// <summary>Whether the rotation carries explicit sign vectors.</summary>
    public bool HasSigns => _signs.Count > 0;

    /// <summary>The input widths an explicit sign vector is stored for.</summary>
    public IReadOnlyCollection<int> SignWidths => _signs.Keys;

    /// <summary>The sign vector for <paramref name="width"/>, or an empty span when unsigned.</summary>
    public ReadOnlySpan<float> Signs(int width) =>
        _signs.TryGetValue(width, out float[]? vector) ? vector : [];

    /// <summary>
    /// Builds the deterministic sign vector this port uses when a policy asks for explicit signs.
    /// </summary>
    /// <param name="width">Vector length; must be a multiple of the rotation block.</param>
    /// <param name="seed">Stream seed, so a conversion is reproducible from the policy alone.</param>
    /// <remarks>
    /// Any fixed ±1 diagonal serves the purpose — the rotation is orthogonal whichever signs it
    /// carries — so what matters is that the converter and the runner agree, which they do because
    /// the vector is written into the file rather than regenerated from the seed at load.
    /// <see cref="Random"/> with an explicit seed is stable across .NET versions for the
    /// <see cref="Random(int)"/> constructor's legacy algorithm, and the file makes it moot.
    /// </remarks>
    public static float[] BuildSigns(int width, int seed)
    {
        var random = new Random(seed);
        float[] signs = new float[width];
        for (int i = 0; i < width; i++)
        {
            signs[i] = random.Next(2) == 0 ? -1f : 1f;
        }

        return signs;
    }

    /// <summary>
    /// Applies <c>R</c> to each row of <paramref name="activations"/>, in place.
    /// </summary>
    /// <param name="activations">
    /// <c>[rows, width]</c>, row-major. <paramref name="width"/> must be a multiple of
    /// <see cref="BlockSize"/>.
    /// </param>
    /// <param name="rows">Number of rows.</param>
    /// <param name="width">Row length.</param>
    public void Apply(Span<float> activations, int rows, int width)
    {
        if (width % BlockSize != 0)
        {
            throw new ArgumentException(
                $"A width of {width} is not a multiple of the {BlockSize}-wide rotation block.", nameof(width));
        }

        ReadOnlySpan<float> signs = Signs(width);
        if (!signs.IsEmpty)
        {
            for (int r = 0; r < rows; r++)
            {
                Kernels.MultiplyInPlace(activations.Slice(r * width, width), signs);
            }
        }
        else if (_signs.Count > 0)
        {
            throw new InvalidOperationException(
                $"The rotation has explicit signs but none for width {width}; the file is incomplete.");
        }

        float scale = 1f / MathF.Sqrt(BlockSize);
        for (int r = 0; r < rows; r++)
        {
            Span<float> row = activations.Slice(r * width, width);
            for (int offset = 0; offset < width; offset += BlockSize)
            {
                Transform(row.Slice(offset, BlockSize));
            }

            Kernels.Scale(row, scale);
        }
    }

    /// <summary>
    /// Folds a weight matrix into the rotated basis: <c>W ← W Rᵀ</c>, one row at a time.
    /// </summary>
    /// <param name="weights"><c>[rows, width]</c>, row-major; overwritten.</param>
    /// <param name="rows">Number of output rows.</param>
    /// <param name="width">Input width.</param>
    /// <remarks>
    /// <para>
    /// Row <c>i</c> of <c>W Rᵀ</c> is <c>(wᵢᵀ Rᵀ)ᵀ = R wᵢ</c>, so folding a weight is the same
    /// operation as transforming an activation. That is not a coincidence worth hiding: it is why
    /// <see cref="Fold"/> and <see cref="Apply"/> are one implementation, and so why a change to
    /// one cannot silently disagree with the other.
    /// </para>
    /// <para>
    /// <c>R</c> is orthogonal — <c>RᵀR = Sᵀ Hᵀ H S / n = S² = I</c> — so
    /// <c>(W Rᵀ)(R x) = W x</c> exactly, in real arithmetic. What the fold changes is the
    /// distribution the quantizer then sees, not the function.
    /// </para>
    /// </remarks>
    public void Fold(Span<float> weights, int rows, int width) => Apply(weights, rows, width);

    /// <summary>
    /// In-place unnormalized fast Walsh–Hadamard transform of one block.
    /// </summary>
    private static void Transform(Span<float> block)
    {
        int length = block.Length;
        int vectorWidth = Vector<float>.Count;

        for (int half = 1; half < length; half <<= 1)
        {
            if (half >= vectorWidth && Vector.IsHardwareAccelerated)
            {
                TransformStageVector(block, half, vectorWidth);
            }
            else
            {
                TransformStageScalar(block, half);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void TransformStageScalar(Span<float> block, int half)
    {
        for (int start = 0; start < block.Length; start += half << 1)
        {
            for (int i = start; i < start + half; i++)
            {
                float a = block[i];
                float b = block[i + half];
                block[i] = a + b;
                block[i + half] = a - b;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void TransformStageVector(Span<float> block, int half, int vectorWidth)
    {
        for (int start = 0; start < block.Length; start += half << 1)
        {
            for (int i = start; i < start + half; i += vectorWidth)
            {
                var a = new Vector<float>(block.Slice(i, vectorWidth));
                var b = new Vector<float>(block.Slice(i + half, vectorWidth));
                (a + b).CopyTo(block.Slice(i, vectorWidth));
                (a - b).CopyTo(block.Slice(i + half, vectorWidth));
            }
        }
    }
}
