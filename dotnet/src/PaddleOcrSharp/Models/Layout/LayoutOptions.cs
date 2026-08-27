namespace PaddleOcrSharp.Models.Layout;

/// <summary>
/// Which shape a detected region is reduced to.
/// </summary>
/// <remarks>
/// The detector emits a mask per region as well as a box. <c>layout_shape_mode</c> upstream;
/// the default is <see cref="Auto"/>.
/// </remarks>
public enum LayoutShapeMode
{
    /// <summary>The region's polygon, whatever shape the mask gives it.</summary>
    Poly,

    /// <summary>The minimum-area rectangle enclosing the polygon, which may be rotated.</summary>
    Quad,

    /// <summary>The axis-aligned bounding box; the masks are not used at all.</summary>
    Rect,

    /// <summary>Rectangle, quad or polygon, whichever fits the region without over-claiming.</summary>
    Auto,
}

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

    /// <summary>Score threshold applied to every class without one of its own.</summary>
    public float Threshold { get; init; } = 0.3f;

    /// <summary>
    /// Per-class score thresholds, keyed by class id.
    /// </summary>
    /// <remarks>
    /// A class listed here uses its own threshold; upstream defaults an unlisted class to
    /// <c>0.5</c> when any per-class threshold is given at all, rather than to the shared one.
    /// </remarks>
    public IReadOnlyDictionary<int, float>? ClassThresholds { get; init; }

    /// <summary>
    /// Which shape each region is reduced to. Anything but <see cref="LayoutShapeMode.Rect"/>
    /// runs the detector's mask head through <see cref="LayoutPolygons"/>.
    /// </summary>
    public LayoutShapeMode ShapeMode { get; init; } = LayoutShapeMode.Auto;

    /// <summary>Whether layout-aware non-maximum suppression runs.</summary>
    public bool Nms { get; init; } = true;

    /// <summary>IoU above which two boxes of the same class suppress each other.</summary>
    public float NmsIouSameClass { get; init; } = 0.6f;

    /// <summary>IoU above which two boxes of different classes suppress each other.</summary>
    public float NmsIouDifferentClass { get; init; } = 0.98f;

    /// <summary>Horizontal and vertical expansion applied to every surviving box.</summary>
    public (float Horizontal, float Vertical) UnclipRatio { get; init; } = (1.0f, 1.0f);

    /// <summary>
    /// Per-class box expansion, keyed by class id; a class not listed is left alone.
    /// </summary>
    public IReadOnlyDictionary<int, (float Horizontal, float Vertical)>? ClassUnclipRatios { get; init; }

    /// <summary>Containment resolution per class id; classes not listed default to union.</summary>
    public IReadOnlyDictionary<int, LayoutMergeMode> MergeModes { get; init; } = DefaultMergeModes;

    /// <summary>Whether page-sized detections are dropped, as upstream's large-image filter does.</summary>
    public bool FilterPageSizedBoxes { get; init; } = true;
}
