namespace PaddleOcrSharp.Models.Language;

/// <summary>
/// The three position-id streams (temporal, height, width) the 3-D rotary embedding consumes.
/// </summary>
public sealed class PositionIds
{
    /// <summary>Creates position ids for <paramref name="length"/> tokens.</summary>
    public PositionIds(int length)
    {
        Temporal = new int[length];
        Height = new int[length];
        Width = new int[length];
    }

    /// <summary>Temporal ids.</summary>
    public int[] Temporal { get; }

    /// <summary>Height ids.</summary>
    public int[] Height { get; }

    /// <summary>Width ids.</summary>
    public int[] Width { get; }

    /// <summary>Number of positions.</summary>
    public int Length => Temporal.Length;

    /// <summary>Highest id across all three streams, or −1 for an empty sequence.</summary>
    public int Max
    {
        get
        {
            int max = -1;
            for (int i = 0; i < Temporal.Length; i++)
            {
                max = Math.Max(max, Math.Max(Temporal[i], Math.Max(Height[i], Width[i])));
            }

            return max;
        }
    }

    /// <summary>Plain 1-D positions <c>0..length-1</c> on all three axes, as a pure-text prompt gets.</summary>
    public static PositionIds Sequential(int length, int offset = 0)
    {
        var ids = new PositionIds(length);
        for (int i = 0; i < length; i++)
        {
            ids.Temporal[i] = offset + i;
            ids.Height[i] = offset + i;
            ids.Width[i] = offset + i;
        }

        return ids;
    }

    /// <summary>Sets all three streams of position <paramref name="index"/>.</summary>
    public void Set(int index, int temporal, int height, int width)
    {
        Temporal[index] = temporal;
        Height[index] = height;
        Width[index] = width;
    }
}

/// <summary>
/// Builds the multimodal position ids for a prompt containing image placeholders.
/// </summary>
/// <remarks>
/// Port of <c>PaddleOCRVLForConditionalGeneration.get_rope_index</c>. Text runs advance all three
/// axes together; each image block instead lays a <c>(t, h/merge, w/merge)</c> grid over the axes,
/// and the next text run resumes from <c>max(previous ids) + 1</c>.
/// </remarks>
public static class RopeIndex
{
    /// <summary>
    /// Computes position ids for <paramref name="tokens"/>.
    /// </summary>
    /// <param name="tokens">Prompt token ids.</param>
    /// <param name="imageGrids">
    /// One <c>(t, h, w)</c> patch grid per image, in the order the placeholders appear.
    /// </param>
    /// <param name="config">Decoder configuration, for the image token id.</param>
    /// <param name="spatialMergeSize">Patches merged per side by the projector (2).</param>
    /// <returns>
    /// The position ids and the <c>rope delta</c> upstream returns, i.e.
    /// <c>max(id) + 1 − tokenCount</c>, which offsets positions during incremental decoding.
    /// </returns>
    public static (PositionIds Ids, int Delta) Compute(
        ReadOnlySpan<int> tokens,
        IReadOnlyList<(int Temporal, int Height, int Width)> imageGrids,
        LanguageConfig config,
        int spatialMergeSize)
    {
        var ids = new PositionIds(tokens.Length);

        int cursor = 0;
        int nextStart = 0;
        int imageIndex = 0;
        int written = 0;

        while (imageIndex < imageGrids.Count)
        {
            int placeholder = IndexOf(tokens, config.ImageTokenId, cursor);
            if (placeholder < 0)
            {
                break;
            }

            (int t, int h, int w) = imageGrids[imageIndex];
            int gridHeight = h / spatialMergeSize;
            int gridWidth = w / spatialMergeSize;

            int textLength = placeholder - cursor;
            for (int i = 0; i < textLength; i++)
            {
                int value = nextStart + i;
                ids.Set(written++, value, value, value);
            }

            int gridBase = nextStart + textLength;
            for (int ti = 0; ti < t; ti++)
            {
                for (int hi = 0; hi < gridHeight; hi++)
                {
                    for (int wi = 0; wi < gridWidth; wi++)
                    {
                        // Upstream derives the temporal ids from
                        // `expanded_range * second_per_grid_t * tokens_per_second`, and images pass
                        // `second_per_grid_t = 0`, so every image patch gets temporal id 0.
                        ids.Set(written++, gridBase, gridBase + hi, gridBase + wi);
                    }
                }
            }

            // The next run resumes at max(ids in this block) + 1. With the temporal ids pinned to
            // the block base, that maximum comes from the spatial axes alone.
            nextStart = gridBase + Math.Max(gridHeight, gridWidth);
            cursor = placeholder + (t * gridHeight * gridWidth);
            imageIndex++;
        }

        for (int i = cursor; i < tokens.Length; i++)
        {
            int value = nextStart + (i - cursor);
            ids.Set(written++, value, value, value);
        }

        if (written != tokens.Length)
        {
            throw new InvalidOperationException(
                $"Position ids cover {written} tokens but the prompt has {tokens.Length}; " +
                "the image placeholders and the supplied grids disagree.");
        }

        return (ids, ids.Max + 1 - tokens.Length);
    }

    private static int IndexOf(ReadOnlySpan<int> tokens, int value, int start)
    {
        for (int i = start; i < tokens.Length; i++)
        {
            if (tokens[i] == value)
            {
                return i;
            }
        }

        return -1;
    }
}
