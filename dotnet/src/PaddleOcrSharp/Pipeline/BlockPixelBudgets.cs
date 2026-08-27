namespace PaddleOcrSharp.Pipeline;

/// <summary>
/// Per-label pixel budgets for the VL model's <c>smart_resize</c>.
/// </summary>
/// <remarks>
/// <para>
/// Port of the <c>*_min_pixels</c> / <c>*_max_pixels</c> knobs the pipeline reads out of
/// <c>vlm_kwargs</c>. Each label's budget defaults to the pipeline's own default pair, so leaving
/// this alone reproduces upstream exactly; raising a ceiling gives that block type more
/// resolution at proportionally more time in the vision tower.
/// </para>
/// <para>
/// Spotting is deliberately absent: upstream fixes it at
/// <c>112896 … 1605632</c> rather than reading it from the configuration.
/// </para>
/// </remarks>
public sealed record BlockPixelBudgets
{
    /// <summary>The pipeline's defaults, identical for every label.</summary>
    public static BlockPixelBudgets Default { get; } = new();

    /// <summary>Lower bound applied to any label without one of its own.</summary>
    public int MinPixels { get; init; } = BlockPrompt.DefaultMinPixels;

    /// <summary>Upper bound applied to any label without one of its own.</summary>
    public int MaxPixels { get; init; } = BlockPrompt.DefaultMaxPixels;

    /// <summary>Lower bound for plain text blocks.</summary>
    public int? OcrMinPixels { get; init; }

    /// <summary>Upper bound for plain text blocks.</summary>
    public int? OcrMaxPixels { get; init; }

    /// <summary>Lower bound for table blocks.</summary>
    public int? TableMinPixels { get; init; }

    /// <summary>Upper bound for table blocks.</summary>
    public int? TableMaxPixels { get; init; }

    /// <summary>Lower bound for chart blocks.</summary>
    public int? ChartMinPixels { get; init; }

    /// <summary>Upper bound for chart blocks.</summary>
    public int? ChartMaxPixels { get; init; }

    /// <summary>Lower bound for formula blocks.</summary>
    public int? FormulaMinPixels { get; init; }

    /// <summary>Upper bound for formula blocks.</summary>
    public int? FormulaMaxPixels { get; init; }

    /// <summary>Lower bound for seal blocks.</summary>
    public int? SealMinPixels { get; init; }

    /// <summary>Upper bound for seal blocks.</summary>
    public int? SealMaxPixels { get; init; }

    /// <summary>The budget <paramref name="label"/> should be recognised with.</summary>
    /// <param name="label">Layout label.</param>
    /// <returns>The lower and upper pixel bounds.</returns>
    public (int MinPixels, int MaxPixels) For(string label)
    {
        if (label == "spotting")
        {
            return (BlockPrompt.DefaultMinPixels, BlockPrompt.SpottingMaxPixels);
        }

        // The dispatch mirrors BlockPrompt.For: the instruction and the budget are chosen by the
        // same branch upstream, so they must not be able to disagree here.
        return label switch
        {
            "table" => (TableMinPixels ?? MinPixels, TableMaxPixels ?? MaxPixels),
            "chart" => (ChartMinPixels ?? MinPixels, ChartMaxPixels ?? MaxPixels),
            "seal" => (SealMinPixels ?? MinPixels, SealMaxPixels ?? MaxPixels),
            _ when label.Contains("formula", StringComparison.Ordinal) && label != "formula_number" =>
                (FormulaMinPixels ?? MinPixels, FormulaMaxPixels ?? MaxPixels),
            _ => (OcrMinPixels ?? MinPixels, OcrMaxPixels ?? MaxPixels),
        };
    }
}
