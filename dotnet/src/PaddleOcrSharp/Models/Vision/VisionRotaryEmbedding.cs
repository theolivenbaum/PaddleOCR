namespace PaddleOcrSharp.Models.Vision;

/// <summary>
/// The 2-D rotary embedding applied inside the vision tower's attention.
/// </summary>
/// <remarks>
/// <para>
/// Port of <c>SigLIPRotaryEmbedding</c> plus the assembly in <c>PaddleOCREncoder.forward</c>:
/// frequencies are built for <c>dim = headDim / 2</c>, looked up separately for the patch's row
/// and column, concatenated, then duplicated so the tensor covers the full head width:
/// </para>
/// <code>
/// rope = freqs[[row, col]].flatten()      // headDim / 2 values
/// rope = rope.repeat(2)                   // headDim values
/// cos, sin = rope.cos(), rope.sin()
/// </code>
/// <para>
/// The rotation itself is <c>rotate_half</c> style (split at the midpoint), not interleaved.
/// </para>
/// </remarks>
public sealed class VisionRotaryEmbedding
{
    private readonly float[] _inverseFrequencies;

    /// <summary>Builds the frequency table for a head of <paramref name="headDim"/> elements.</summary>
    public VisionRotaryEmbedding(int headDim, float theta)
    {
        // Upstream constructs SigLIPRotaryEmbedding(head_dim // 2) and that module halves again.
        int dim = headDim / 2;
        int count = dim / 2;
        _inverseFrequencies = new float[count];
        for (int i = 0; i < count; i++)
        {
            _inverseFrequencies[i] = (float)(1.0 / Math.Pow(theta, (2.0 * i) / dim));
        }

        HeadDim = headDim;
        HalfHeadDim = headDim / 2;
    }

    /// <summary>Width of one attention head.</summary>
    public int HeadDim { get; }

    /// <summary>Half the head width, the split point of <c>rotate_half</c>.</summary>
    public int HalfHeadDim { get; }

    /// <summary>
    /// Fills <paramref name="cos"/> and <paramref name="sin"/> with per-token rotation factors.
    /// </summary>
    /// <param name="gridHeight">Patch rows of the image.</param>
    /// <param name="gridWidth">Patch columns of the image.</param>
    /// <param name="cos">Receives <c>[gridHeight · gridWidth, headDim]</c> cosines.</param>
    /// <param name="sin">Receives <c>[gridHeight · gridWidth, headDim]</c> sines.</param>
    public void Fill(int gridHeight, int gridWidth, Span<float> cos, Span<float> sin)
    {
        int tokens = gridHeight * gridWidth;
        int count = _inverseFrequencies.Length;

        if (cos.Length < tokens * HeadDim || sin.Length < tokens * HeadDim)
        {
            throw new ArgumentException($"Rotary buffers must hold {tokens * HeadDim} elements.");
        }

        for (int token = 0; token < tokens; token++)
        {
            int row = token / gridWidth;
            int column = token % gridWidth;

            Span<float> cosRow = cos.Slice(token * HeadDim, HeadDim);
            Span<float> sinRow = sin.Slice(token * HeadDim, HeadDim);

            for (int i = 0; i < count; i++)
            {
                float rowAngle = row * _inverseFrequencies[i];
                float columnAngle = column * _inverseFrequencies[i];

                float cosRowValue = MathF.Cos(rowAngle);
                float sinRowValue = MathF.Sin(rowAngle);
                float cosColumnValue = MathF.Cos(columnAngle);
                float sinColumnValue = MathF.Sin(columnAngle);

                // Layout is [row freqs | column freqs] duplicated across the two halves.
                cosRow[i] = cosRowValue;
                cosRow[count + i] = cosColumnValue;
                cosRow[HalfHeadDim + i] = cosRowValue;
                cosRow[HalfHeadDim + count + i] = cosColumnValue;

                sinRow[i] = sinRowValue;
                sinRow[count + i] = sinColumnValue;
                sinRow[HalfHeadDim + i] = sinRowValue;
                sinRow[HalfHeadDim + count + i] = sinColumnValue;
            }
        }
    }

    /// <summary>
    /// Applies the rotation in place to one head's vector:
    /// <c>x · cos + rotate_half(x) · sin</c>.
    /// </summary>
    public void Apply(Span<float> head, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin)
    {
        int half = HalfHeadDim;
        for (int i = 0; i < half; i++)
        {
            float low = head[i];
            float high = head[half + i];
            head[i] = (low * cos[i]) - (high * sin[i]);
            head[half + i] = (high * cos[half + i]) + (low * sin[half + i]);
        }
    }
}
