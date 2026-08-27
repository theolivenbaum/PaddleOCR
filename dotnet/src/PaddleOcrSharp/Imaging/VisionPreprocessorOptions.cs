namespace PaddleOcrSharp.Imaging;

/// <summary>
/// Settings of <c>PaddleOCRVLImageProcessor</c>. Defaults come from the checkpoint's
/// <c>preprocessor_config.json</c>.
/// </summary>
public sealed record VisionPreprocessorOptions
{
    /// <summary>Defaults as shipped with PaddleOCR-VL-1.6.</summary>
    public static VisionPreprocessorOptions Default { get; } = new();

    /// <summary>Spatial patch size of the vision encoder.</summary>
    public int PatchSize { get; init; } = 14;

    /// <summary>Number of patches merged per side by the projector.</summary>
    public int MergeSize { get; init; } = 2;

    /// <summary>Temporal patch size; images use 1.</summary>
    public int TemporalPatchSize { get; init; } = 1;

    /// <summary>Lower bound on resized pixel count.</summary>
    public int MinPixels { get; init; } = 112_896;

    /// <summary>Upper bound on resized pixel count.</summary>
    public int MaxPixels { get; init; } = 1_003_520;

    /// <summary>Scale applied to raw byte values before normalisation.</summary>
    public double RescaleFactor { get; init; } = 1.0 / 255.0;

    /// <summary>Per-channel mean.</summary>
    public float[] ImageMean { get; init; } = [0.5f, 0.5f, 0.5f];

    /// <summary>Per-channel standard deviation.</summary>
    public float[] ImageStd { get; init; } = [0.5f, 0.5f, 0.5f];

    /// <summary>Alignment enforced by <see cref="SmartResize"/>.</summary>
    public int Factor => PatchSize * MergeSize;

    /// <summary>Returns a copy with a different pixel budget, as the pipeline does per block label.</summary>
    public VisionPreprocessorOptions WithPixelBudget(int minPixels, int maxPixels) =>
        this with { MinPixels = minPixels, MaxPixels = maxPixels };
}
