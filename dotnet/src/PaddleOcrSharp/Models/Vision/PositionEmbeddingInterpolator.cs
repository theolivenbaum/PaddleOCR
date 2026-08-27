using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Models.Vision;

/// <summary>
/// Resamples the pretrained 27×27 absolute position grid onto an arbitrary patch grid.
/// </summary>
/// <remarks>
/// <para>
/// Port of <c>PaddleOCRVisionEmbeddings.interpolate_pos_encoding</c>, which reshapes the
/// <c>[729, 1152]</c> embedding table to <c>[1, 1152, 27, 27]</c> and calls
/// <c>F.interpolate(mode="bilinear", align_corners=False)</c>.
/// </para>
/// <para>
/// Upstream keeps an LFU cache of up to 20 grids because a page yields many crops that share a
/// shape; this type does the same with a plain most-used-wins eviction.
/// </para>
/// </remarks>
public sealed class PositionEmbeddingInterpolator
{
    private readonly float[] _table;
    private readonly int _gridSize;
    private readonly int _width;
    private readonly int _maxCacheEntries;
    private readonly Dictionary<(int Height, int Width), float[]> _cache = [];
    private readonly Dictionary<(int Height, int Width), long> _hits = [];
    private readonly Lock _gate = new();

    /// <summary>Creates an interpolator over a pretrained embedding table.</summary>
    /// <param name="table">The <c>[gridSize², width]</c> embedding table, row-major.</param>
    /// <param name="gridSize">Side length of the pretrained grid (27).</param>
    /// <param name="width">Embedding width (1152).</param>
    /// <param name="maxCacheEntries">Maximum distinct grids to retain; upstream uses 20.</param>
    public PositionEmbeddingInterpolator(float[] table, int gridSize, int width, int maxCacheEntries = 20)
    {
        if (table.Length < gridSize * gridSize * width)
        {
            throw new ArgumentException("Embedding table is smaller than the declared grid.", nameof(table));
        }

        _table = table;
        _gridSize = gridSize;
        _width = width;
        _maxCacheEntries = maxCacheEntries;
    }

    /// <summary>
    /// Returns the position embedding for a <paramref name="height"/> × <paramref name="width"/>
    /// patch grid, as <c>[height · width, embeddingWidth]</c>.
    /// </summary>
    public ReadOnlySpan<float> Get(int height, int width)
    {
        lock (_gate)
        {
            if (_cache.TryGetValue((height, width), out float[]? cached))
            {
                _hits[(height, width)]++;
                return cached;
            }

            if (_cache.Count >= _maxCacheEntries)
            {
                (int Height, int Width) coldest = _hits.MinBy(entry => entry.Value).Key;
                _cache.Remove(coldest);
                _hits.Remove(coldest);
            }

            float[] resampled = Interpolate(height, width);
            _cache[(height, width)] = resampled;
            _hits[(height, width)] = 1;
            return resampled;
        }
    }

    /// <summary>
    /// Bilinear resample of the grid, matching <c>F.interpolate(align_corners=False)</c>.
    /// </summary>
    /// <remarks>
    /// PyTorch maps output index <c>i</c> to <c>(i + 0.5) · in/out − 0.5</c>, clamps the result at
    /// zero, truncates for the low index and clamps the high index to <c>in − 1</c>. Reproducing
    /// the clamp-at-zero (rather than clamping the weight) is what keeps the first row and column
    /// identical to upstream.
    /// </remarks>
    private float[] Interpolate(int outHeight, int outWidth)
    {
        float[] result = new float[outHeight * outWidth * _width];

        float scaleY = (float)_gridSize / outHeight;
        float scaleX = (float)_gridSize / outWidth;

        for (int oy = 0; oy < outHeight; oy++)
        {
            float sourceY = ((oy + 0.5f) * scaleY) - 0.5f;
            if (sourceY < 0f)
            {
                sourceY = 0f;
            }

            int y0 = (int)sourceY;
            int y1 = y0 < _gridSize - 1 ? y0 + 1 : y0;
            float wy1 = sourceY - y0;
            float wy0 = 1f - wy1;

            for (int ox = 0; ox < outWidth; ox++)
            {
                float sourceX = ((ox + 0.5f) * scaleX) - 0.5f;
                if (sourceX < 0f)
                {
                    sourceX = 0f;
                }

                int x0 = (int)sourceX;
                int x1 = x0 < _gridSize - 1 ? x0 + 1 : x0;
                float wx1 = sourceX - x0;
                float wx0 = 1f - wx1;

                Span<float> target = result.AsSpan((((oy * outWidth) + ox) * _width), _width);
                ReadOnlySpan<float> p00 = _table.AsSpan((((y0 * _gridSize) + x0) * _width), _width);
                ReadOnlySpan<float> p01 = _table.AsSpan((((y0 * _gridSize) + x1) * _width), _width);
                ReadOnlySpan<float> p10 = _table.AsSpan((((y1 * _gridSize) + x0) * _width), _width);
                ReadOnlySpan<float> p11 = _table.AsSpan((((y1 * _gridSize) + x1) * _width), _width);

                float w00 = wy0 * wx0;
                float w01 = wy0 * wx1;
                float w10 = wy1 * wx0;
                float w11 = wy1 * wx1;

                for (int c = 0; c < _width; c++)
                {
                    target[c] = (w00 * p00[c]) + (w01 * p01[c]) + (w10 * p10[c]) + (w11 * p11[c]);
                }
            }
        }

        return result;
    }
}
