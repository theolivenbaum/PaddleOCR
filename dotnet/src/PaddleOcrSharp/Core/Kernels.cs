using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace PaddleOcrSharp.Core;

/// <summary>
/// Vectorised element-wise kernels shared by the vision tower, the language model and the
/// layout detector.
/// </summary>
public static class Kernels
{
    private const float SqrtTwoOverPi = 0.7978845608028654f;
    private const float GeluTanhCoefficient = 0.044715f;

    /// <summary><c>destination += source</c>.</summary>
    public static void AddInPlace(Span<float> destination, ReadOnlySpan<float> source) =>
        TensorPrimitives.Add(destination, source[..destination.Length], destination);

    /// <summary><c>destination += source · scale</c>.</summary>
    /// <summary>
    /// Adds <paramref name="source"/>, scaled four different ways, into four destinations.
    /// </summary>
    /// <remarks>
    /// The point is the single load of <paramref name="source"/>: a GEMM inner loop that updates
    /// four output rows against one row of the right-hand matrix gets four multiply-adds per
    /// loaded element instead of one, which is what turns it from load-bound into compute-bound.
    /// </remarks>
    public static void AddScaled4(
        Span<float> d0,
        Span<float> d1,
        Span<float> d2,
        Span<float> d3,
        ReadOnlySpan<float> source,
        float s0,
        float s1,
        float s2,
        float s3)
    {
        int length = source.Length;
        int i = 0;

        if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            Vector256<float> v0 = Vector256.Create(s0);
            Vector256<float> v1 = Vector256.Create(s1);
            Vector256<float> v2 = Vector256.Create(s2);
            Vector256<float> v3 = Vector256.Create(s3);

            for (; i <= length - Vector256<float>.Count; i += Vector256<float>.Count)
            {
                Vector256<float> x = Vector256.LoadUnsafe(in source[i]);
                Vector256.FusedMultiplyAdd(x, v0, Vector256.LoadUnsafe(in d0[i])).StoreUnsafe(ref d0[i]);
                Vector256.FusedMultiplyAdd(x, v1, Vector256.LoadUnsafe(in d1[i])).StoreUnsafe(ref d1[i]);
                Vector256.FusedMultiplyAdd(x, v2, Vector256.LoadUnsafe(in d2[i])).StoreUnsafe(ref d2[i]);
                Vector256.FusedMultiplyAdd(x, v3, Vector256.LoadUnsafe(in d3[i])).StoreUnsafe(ref d3[i]);
            }
        }

        for (; i < length; i++)
        {
            float x = source[i];
            d0[i] += x * s0;
            d1[i] += x * s1;
            d2[i] += x * s2;
            d3[i] += x * s3;
        }
    }

    public static void AddScaled(Span<float> destination, ReadOnlySpan<float> source, float scale)
    {
        int length = destination.Length;
        int i = 0;

        if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            Vector256<float> scaleVector = Vector256.Create(scale);
            for (; i <= length - Vector256<float>.Count; i += Vector256<float>.Count)
            {
                Vector256<float> acc = Vector256.FusedMultiplyAdd(
                    Vector256.LoadUnsafe(in source[i]), scaleVector, Vector256.LoadUnsafe(in destination[i]));
                acc.StoreUnsafe(ref destination[i]);
            }
        }

        for (; i < length; i++)
        {
            destination[i] += source[i] * scale;
        }
    }

    /// <summary><c>destination *= source</c>.</summary>
    public static void MultiplyInPlace(Span<float> destination, ReadOnlySpan<float> source) =>
        TensorPrimitives.Multiply(destination, source[..destination.Length], destination);

    /// <summary><c>destination *= scale</c>.</summary>
    public static void Scale(Span<float> destination, float scale) =>
        TensorPrimitives.Multiply(destination, scale, destination);

    /// <summary>
    /// SiLU (a.k.a. swish): <c>x · sigmoid(x)</c> — the activation of the ERNIE MLP.
    /// </summary>
    public static void Silu(Span<float> values)
    {
        int length = values.Length;
        int i = 0;

        if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            Vector256<float> one = Vector256.Create(1f);
            for (; i <= length - Vector256<float>.Count; i += Vector256<float>.Count)
            {
                Vector256<float> x = Vector256.LoadUnsafe(in values[i]);
                Vector256<float> sigmoid = one / (one + Exp(-x));
                (x * sigmoid).StoreUnsafe(ref values[i]);
            }
        }

        for (; i < length; i++)
        {
            float x = values[i];
            values[i] = x / (1f + MathF.Exp(-x));
        }
    }

    /// <summary>
    /// Exact GELU: <c>0.5 · x · (1 + erf(x / √2))</c>. This is what
    /// <c>transformers.activations.GELUActivation</c> uses for the projector.
    /// </summary>
    public static void GeluErf(Span<float> values)
    {
        for (int i = 0; i < values.Length; i++)
        {
            float x = values[i];
            values[i] = 0.5f * x * (1f + Erf(x * 0.70710678118654752f));
        }
    }

    /// <summary>
    /// Tanh-approximated GELU (<c>gelu_pytorch_tanh</c>), the activation of the vision MLP.
    /// </summary>
    public static void GeluTanh(Span<float> values)
    {
        int length = values.Length;
        int i = 0;

        if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            Vector256<float> half = Vector256.Create(0.5f);
            Vector256<float> one = Vector256.Create(1f);
            Vector256<float> two = Vector256.Create(2f);
            Vector256<float> alpha = Vector256.Create(SqrtTwoOverPi);
            Vector256<float> beta = Vector256.Create(GeluTanhCoefficient);

            for (; i <= length - Vector256<float>.Count; i += Vector256<float>.Count)
            {
                Vector256<float> x = Vector256.LoadUnsafe(in values[i]);
                Vector256<float> inner = alpha * (x + (beta * x * x * x));

                // tanh(z) = 2·sigmoid(2z) − 1, which reuses the vectorised exp below.
                Vector256<float> tanh = (two / (one + Exp(-two * inner))) - one;
                (half * x * (one + tanh)).StoreUnsafe(ref values[i]);
            }
        }

        for (; i < length; i++)
        {
            float x = values[i];
            float inner = SqrtTwoOverPi * (x + (GeluTanhCoefficient * x * x * x));
            values[i] = 0.5f * x * (1f + MathF.Tanh(inner));
        }
    }

    /// <summary>
    /// In-place softmax over <paramref name="values"/>, computed in float32 exactly as
    /// <c>nn.functional.softmax(..., dtype=torch.float32)</c> does upstream.
    /// </summary>
    public static void Softmax(Span<float> values)
    {
        if (values.IsEmpty)
        {
            return;
        }

        float max = TensorPrimitives.Max(values);
        if (float.IsNegativeInfinity(max))
        {
            // Every position is masked out; upstream would produce NaN here, but a fully masked
            // row never reaches a real model output, so emit a uniform row instead of NaNs.
            values.Fill(0f);
            return;
        }

        float sum = 0f;
        int index = 0;

        if (Vector256.IsHardwareAccelerated && values.Length >= Vector256<float>.Count)
        {
            Vector256<float> shift = Vector256.Create(max);
            Vector256<float> total = Vector256<float>.Zero;

            for (; index <= values.Length - Vector256<float>.Count; index += Vector256<float>.Count)
            {
                Vector256<float> e = Exp(Vector256.LoadUnsafe(in values[index]) - shift);
                e.StoreUnsafe(ref values[index]);
                total += e;
            }

            sum = Vector256.Sum(total);
        }

        for (; index < values.Length; index++)
        {
            float e = MathF.Exp(values[index] - max);
            values[index] = e;
            sum += e;
        }

        if (sum > 0f)
        {
            TensorPrimitives.Multiply(values, 1f / sum, values);
        }
    }

    /// <summary>Numerically stable error function, matching <c>std::erf</c> to ~1e-7.</summary>
    public static float Erf(float x)
    {
        // Abramowitz &amp; Stegun 7.1.26 is only good to 1e-7, which is below float32 resolution
        // for the |x| range GELU cares about.
        float sign = x < 0f ? -1f : 1f;
        float ax = MathF.Abs(x);

        const float A1 = 0.254829592f;
        const float A2 = -0.284496736f;
        const float A3 = 1.421413741f;
        const float A4 = -1.453152027f;
        const float A5 = 1.061405429f;
        const float P = 0.3275911f;

        float t = 1f / (1f + (P * ax));
        float y = 1f - ((((((((A5 * t) + A4) * t) + A3) * t) + A2) * t) + A1) * t * MathF.Exp(-ax * ax);
        return sign * y;
    }

    /// <summary>Vectorised <c>exp</c> with float32 accuracy, used by the activation kernels.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Exp(Vector256<float> x)
    {
        // Range reduction: e^x = 2^k · e^r with k = round(x / ln2) and r = x − k·ln2.
        Vector256<float> log2E = Vector256.Create(1.44269504088896341f);
        Vector256<float> ln2Hi = Vector256.Create(0.693359375f);
        Vector256<float> ln2Lo = Vector256.Create(-2.12194440e-4f);

        Vector256<float> clamped = Vector256.Min(Vector256.Max(x, Vector256.Create(-88f)), Vector256.Create(88f));
        Vector256<float> k = Vector256.Round(clamped * log2E);
        Vector256<float> r = Vector256.FusedMultiplyAdd(k, -ln2Hi, clamped);
        r = Vector256.FusedMultiplyAdd(k, -ln2Lo, r);

        // Degree-6 minimax polynomial for e^r on [−ln2/2, ln2/2].
        Vector256<float> p = Vector256.Create(1.9875691500e-4f);
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(1.3981999507e-3f));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(8.3334519073e-3f));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(4.1665795894e-2f));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(1.6666665459e-1f));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(5.0000001201e-1f));
        p = Vector256.FusedMultiplyAdd(p, r * r, r);
        p += Vector256.Create(1f);

        // Scale by 2^k by injecting the exponent directly.
        Vector256<int> exponent = (Vector256.ConvertToInt32(k) + Vector256.Create(127)) << 23;
        return p * exponent.AsSingle();
    }
}
