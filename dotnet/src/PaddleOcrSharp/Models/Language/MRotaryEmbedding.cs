namespace PaddleOcrSharp.Models.Language;

/// <summary>
/// The 3-D multimodal rotary embedding of the ERNIE decoder.
/// </summary>
/// <remarks>
/// <para>
/// Port of <c>RotaryEmbedding</c> + <c>apply_multimodal_rotary_pos_emb</c>. Upstream builds
/// <c>cos</c>/<c>sin</c> for all three axes and then interleaves channel ranges from them
/// according to <c>mrope_section</c> repeated twice:
/// </para>
/// <code>
/// sections = [16, 24, 24, 16, 24, 24]
/// cos = cat([chunk_i from axis (i % 3) for i, chunk_i in enumerate(cos.split(sections))])
/// </code>
/// <para>
/// Because <c>emb = cat(freqs, freqs)</c>, channel <c>j</c> and channel <c>j + headDim/2</c>
/// always share a frequency, and the second three sections select the same axes as the first
/// three. That reduces the whole construction to one axis assignment per frequency index, which
/// is what <see cref="_axisOfFrequency"/> stores.
/// </para>
/// </remarks>
public sealed class MRotaryEmbedding
{
    private readonly double[] _inverseFrequencies;
    private readonly byte[] _axisOfFrequency;

    /// <summary>Builds the frequency table and the per-channel axis assignment.</summary>
    public MRotaryEmbedding(LanguageConfig config)
    {
        HeadDim = config.HeadDim;
        HalfHeadDim = config.HeadDim / 2;

        _inverseFrequencies = new double[HalfHeadDim];
        for (int i = 0; i < HalfHeadDim; i++)
        {
            _inverseFrequencies[i] = 1.0 / Math.Pow(config.RopeTheta, (2.0 * i) / config.HeadDim);
        }

        int[] sections = config.MRopeSection;
        if (sections.Sum() * 2 != config.HeadDim)
        {
            throw new ArgumentException(
                $"mrope_section {string.Join(",", sections)} must sum to headDim/2 ({config.HeadDim / 2}).",
                nameof(config));
        }

        _axisOfFrequency = new byte[HalfHeadDim];
        int cursor = 0;
        for (int section = 0; section < sections.Length; section++)
        {
            for (int i = 0; i < sections[section]; i++)
            {
                _axisOfFrequency[cursor++] = (byte)(section % 3);
            }
        }
    }

    /// <summary>Width of one head.</summary>
    public int HeadDim { get; }

    /// <summary>Half the head width; the split point of <c>rotate_half</c>.</summary>
    public int HalfHeadDim { get; }

    /// <summary>
    /// Fills one token's rotation factors from its <c>(temporal, height, width)</c> position.
    /// </summary>
    /// <param name="temporal">Temporal position id.</param>
    /// <param name="height">Height position id.</param>
    /// <param name="width">Width position id.</param>
    /// <param name="cos">Receives <see cref="HalfHeadDim"/> cosines.</param>
    /// <param name="sin">Receives <see cref="HalfHeadDim"/> sines.</param>
    public void Fill(int temporal, int height, int width, Span<float> cos, Span<float> sin)
    {
        for (int i = 0; i < HalfHeadDim; i++)
        {
            int position = _axisOfFrequency[i] switch
            {
                0 => temporal,
                1 => height,
                _ => width,
            };

            double angle = position * _inverseFrequencies[i];
            cos[i] = (float)Math.Cos(angle);
            sin[i] = (float)Math.Sin(angle);
        }
    }

    /// <summary>
    /// Applies <c>x · cos + rotate_half(x) · sin</c> in place to one head's vector.
    /// </summary>
    /// <param name="head">The head vector, <see cref="HeadDim"/> elements.</param>
    /// <param name="cos">Cosines from <see cref="Fill"/>.</param>
    /// <param name="sin">Sines from <see cref="Fill"/>.</param>
    public void Apply(Span<float> head, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin)
    {
        int half = HalfHeadDim;
        for (int i = 0; i < half; i++)
        {
            float low = head[i];
            float high = head[half + i];
            head[i] = (low * cos[i]) - (high * sin[i]);
            head[half + i] = (high * cos[i]) + (low * sin[i]);
        }
    }
}
