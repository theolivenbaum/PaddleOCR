namespace PaddleOcrSharp.Models.Vision;

/// <summary>
/// Vision-tower hyper-parameters, mirroring <c>PaddleOCRVisionConfig</c>.
/// </summary>
/// <remarks>Defaults are the values in PaddleOCR-VL-1.6's <c>config.json</c>.</remarks>
public sealed record VisionConfig
{
    /// <summary>Configuration shipped with PaddleOCR-VL-1.6.</summary>
    public static VisionConfig Default { get; } = new();

    /// <summary>Embedding width of the tower.</summary>
    public int HiddenSize { get; init; } = 1152;

    /// <summary>Width of the MLP hidden layer.</summary>
    public int IntermediateSize { get; init; } = 4304;

    /// <summary>Number of encoder layers.</summary>
    public int NumHiddenLayers { get; init; } = 27;

    /// <summary>Number of attention heads.</summary>
    public int NumAttentionHeads { get; init; } = 16;

    /// <summary>Input channels.</summary>
    public int NumChannels { get; init; } = 3;

    /// <summary>Side length the position-embedding grid was pretrained at.</summary>
    public int ImageSize { get; init; } = 384;

    /// <summary>Spatial patch size.</summary>
    public int PatchSize { get; init; } = 14;

    /// <summary>Layer-norm epsilon inside the tower.</summary>
    public float LayerNormEps { get; init; } = 1e-6f;

    /// <summary>Patches merged per side by the projector.</summary>
    public int SpatialMergeSize { get; init; } = 2;

    /// <summary>Base period of the 2-D rotary embedding.</summary>
    public float RopeTheta { get; init; } = 10000f;

    /// <summary>Width of one attention head.</summary>
    public int HeadDim => HiddenSize / NumAttentionHeads;

    /// <summary>
    /// Side length of the pretrained position grid: <c>floor(imageSize / patchSize)</c>, i.e. 27.
    /// </summary>
    public int PositionGridSize => ImageSize / PatchSize;

    /// <summary>Number of pretrained position embeddings.</summary>
    public int NumPositions => PositionGridSize * PositionGridSize;

    /// <summary>Elements in one flattened patch (<c>C · P · P</c>).</summary>
    public int PatchVectorLength => NumChannels * PatchSize * PatchSize;
}
