namespace PaddleOcrSharp.Models.Layout;

/// <summary>How overlapping boxes of related classes are resolved.</summary>
public enum LayoutMergeMode
{
    /// <summary>Keep both boxes.</summary>
    Union,

    /// <summary>Drop the box contained by another.</summary>
    Large,

    /// <summary>Drop the box that contains another.</summary>
    Small,
}

/// <summary>
/// Post-processing settings for the layout detector, mirroring the shipped pipeline
/// configuration for PaddleOCR-VL-1.6.
/// </summary>
public sealed record LayoutOptions
{
    /// <summary>
    /// The per-class merge modes the shipped pipeline configuration sets.
    /// </summary>
    /// <remarks>
    /// Declared before <see cref="Default"/>: static initialisers run in declaration order, so a
    /// later declaration would leave <see cref="MergeModes"/> null on the default instance.
    /// </remarks>
    public static IReadOnlyDictionary<int, LayoutMergeMode> DefaultMergeModes { get; } =
        new Dictionary<int, LayoutMergeMode>
        {
            [3] = LayoutMergeMode.Large,   // chart
            [5] = LayoutMergeMode.Large,   // display_formula
            [6] = LayoutMergeMode.Large,   // doc_title
            [15] = LayoutMergeMode.Large,  // inline_formula
            [17] = LayoutMergeMode.Large,  // paragraph_title
        };

    /// <summary>Defaults from <c>deploy/paddleocr_vl_docker/pipeline_config_vllm.yaml</c>.</summary>
    public static LayoutOptions Default { get; } = new();

    /// <summary>Score threshold applied to every class.</summary>
    public float Threshold { get; init; } = 0.3f;

    /// <summary>Whether layout-aware non-maximum suppression runs.</summary>
    public bool Nms { get; init; } = true;

    /// <summary>IoU above which two boxes of the same class suppress each other.</summary>
    public float NmsIouSameClass { get; init; } = 0.6f;

    /// <summary>IoU above which two boxes of different classes suppress each other.</summary>
    public float NmsIouDifferentClass { get; init; } = 0.98f;

    /// <summary>Horizontal and vertical expansion applied to every surviving box.</summary>
    public (float Horizontal, float Vertical) UnclipRatio { get; init; } = (1.0f, 1.0f);

    /// <summary>Containment resolution per class id; classes not listed default to union.</summary>
    public IReadOnlyDictionary<int, LayoutMergeMode> MergeModes { get; init; } = DefaultMergeModes;

    /// <summary>Whether page-sized detections are dropped, as upstream's large-image filter does.</summary>
    public bool FilterPageSizedBoxes { get; init; } = true;
}
