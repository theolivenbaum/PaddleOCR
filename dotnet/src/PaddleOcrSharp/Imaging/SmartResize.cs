namespace PaddleOcrSharp.Imaging;

/// <summary>
/// Native-resolution sizing rule shared by the Qwen2-VL family and PaddleOCR-VL.
/// </summary>
/// <remarks>
/// Direct port of <c>smart_resize</c> in <c>image_processing_paddleocr_vl.py</c>. Python's
/// <c>round</c> is round-half-to-even, which is also .NET's default for
/// <see cref="Math.Round(double)"/> — the two agree without extra care.
/// </remarks>
public static class SmartResize
{
    /// <summary>Largest permitted ratio between the long and short side.</summary>
    public const int MaxAspectRatio = 200;

    /// <summary>
    /// Computes the resize target that keeps both sides a multiple of <paramref name="factor"/>
    /// and the pixel count inside <c>[minPixels, maxPixels]</c>, preserving aspect ratio as
    /// closely as possible.
    /// </summary>
    /// <param name="height">Source height.</param>
    /// <param name="width">Source width.</param>
    /// <param name="factor">Alignment, normally <c>patchSize × mergeSize</c> (28).</param>
    /// <param name="minPixels">Lower bound on <c>height × width</c> of the result.</param>
    /// <param name="maxPixels">Upper bound on <c>height × width</c> of the result.</param>
    public static (int Height, int Width) Compute(
        int height,
        int width,
        int factor,
        int minPixels,
        int maxPixels)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(factor);

        double h = height;
        double w = width;

        if (h < factor)
        {
            w = Math.Round(w * factor / h, MidpointRounding.ToEven);
            h = factor;
        }

        if (w < factor)
        {
            h = Math.Round(h * factor / w, MidpointRounding.ToEven);
            w = factor;
        }

        double ratio = Math.Max(h, w) / Math.Min(h, w);
        if (ratio > MaxAspectRatio)
        {
            throw new ArgumentException(
                $"Absolute aspect ratio must be smaller than {MaxAspectRatio}, got {ratio}.");
        }

        int hBar = (int)Math.Round(h / factor, MidpointRounding.ToEven) * factor;
        int wBar = (int)Math.Round(w / factor, MidpointRounding.ToEven) * factor;

        if ((long)hBar * wBar > maxPixels)
        {
            double beta = Math.Sqrt(h * w / maxPixels);
            hBar = (int)Math.Floor(h / beta / factor) * factor;
            wBar = (int)Math.Floor(w / beta / factor) * factor;
        }
        else if ((long)hBar * wBar < minPixels)
        {
            double beta = Math.Sqrt(minPixels / (h * w));
            hBar = (int)Math.Ceiling(h * beta / factor) * factor;
            wBar = (int)Math.Ceiling(w * beta / factor) * factor;
        }

        return (hBar, wBar);
    }
}
