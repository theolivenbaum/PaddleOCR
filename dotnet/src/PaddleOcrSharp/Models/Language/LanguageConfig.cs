namespace PaddleOcrSharp.Models.Language;

/// <summary>
/// Hyper-parameters of the ERNIE-4.5 decoder, mirroring <c>PaddleOCRVLConfig</c>.
/// </summary>
/// <remarks>Defaults are PaddleOCR-VL-1.6's <c>config.json</c>.</remarks>
public sealed record LanguageConfig
{
    /// <summary>Configuration shipped with PaddleOCR-VL-1.6.</summary>
    public static LanguageConfig Default { get; } = new();

    /// <summary>Vocabulary size.</summary>
    public int VocabSize { get; init; } = 103_424;

    /// <summary>Embedding width.</summary>
    public int HiddenSize { get; init; } = 1024;

    /// <summary>MLP hidden width.</summary>
    public int IntermediateSize { get; init; } = 3072;

    /// <summary>Number of decoder layers.</summary>
    public int NumHiddenLayers { get; init; } = 18;

    /// <summary>Number of query heads.</summary>
    public int NumAttentionHeads { get; init; } = 16;

    /// <summary>Number of key/value heads (grouped-query attention).</summary>
    public int NumKeyValueHeads { get; init; } = 2;

    /// <summary>Width of one head.</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>RMS-norm epsilon.</summary>
    public float RmsNormEps { get; init; } = 1e-5f;

    /// <summary>Base period of the rotary embedding.</summary>
    public double RopeTheta { get; init; } = 500_000.0;

    /// <summary>Channel split of the 3-D rotary embedding across (temporal, height, width).</summary>
    public int[] MRopeSection { get; init; } = [16, 24, 24];

    /// <summary>Maximum position the rotary embedding was trained for.</summary>
    public int MaxPositionEmbeddings { get; init; } = 131_072;

    /// <summary>Token id that image features are scattered into.</summary>
    public int ImageTokenId { get; init; } = 100_295;

    /// <summary>Token id that video features are scattered into.</summary>
    public int VideoTokenId { get; init; } = 101_307;

    /// <summary>Token id that precedes a vision block.</summary>
    public int VisionStartTokenId { get; init; } = 101_305;

    /// <summary>Token id that closes a vision block.</summary>
    public int VisionEndTokenId { get; init; } = 101_306;

    /// <summary>End-of-sequence token id.</summary>
    public int EosTokenId { get; init; } = 2;

    /// <summary>Padding token id.</summary>
    public int PadTokenId { get; init; }

    /// <summary>Query heads sharing one key/value head.</summary>
    public int NumKeyValueGroups => NumAttentionHeads / NumKeyValueHeads;

    /// <summary>Total width of the query projection.</summary>
    public int QueryWidth => NumAttentionHeads * HeadDim;

    /// <summary>Total width of the key or value projection.</summary>
    public int KeyValueWidth => NumKeyValueHeads * HeadDim;

    /// <summary>Softmax scale of the attention.</summary>
    public float AttentionScale => 1f / MathF.Sqrt(HeadDim);
}
