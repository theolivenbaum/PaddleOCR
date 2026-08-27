using PaddleOcrSharp.Imaging;

namespace PaddleOcrSharp.Pipeline;

/// <summary>
/// Maps a layout label onto the instruction and pixel budget the VL model should receive.
/// </summary>
/// <remarks>
/// Port of the dispatch in <c>_paddleocr_vl_collect_page_vlm_entries_core</c>. Every budget
/// defaults to the pipeline's <c>112896 … 1003520</c>; only spotting raises the ceiling. The
/// budgets themselves live in <see cref="BlockPixelBudgets"/>, which is configurable the way
/// upstream's <c>vlm_kwargs</c> are.
/// </remarks>
public static class BlockPrompt
{
    /// <summary>Instruction for a plain text block.</summary>
    public const string Ocr = "OCR:";

    /// <summary>Instruction for a table block.</summary>
    public const string Table = "Table Recognition:";

    /// <summary>Instruction for a formula block.</summary>
    public const string Formula = "Formula Recognition:";

    /// <summary>Instruction for a chart block.</summary>
    public const string Chart = "Chart Recognition:";

    /// <summary>Instruction for a seal block.</summary>
    public const string Seal = "Seal Recognition:";

    /// <summary>Instruction for whole-page spotting.</summary>
    public const string Spotting = "Spotting:";

    /// <summary>Default lower pixel bound for every block type.</summary>
    public const int DefaultMinPixels = 112_896;

    /// <summary>Default upper pixel bound for every block type.</summary>
    public const int DefaultMaxPixels = 1_003_520;

    /// <summary>Upper pixel bound used for spotting, which needs the extra resolution.</summary>
    public const int SpottingMaxPixels = 1_605_632;

    /// <summary>
    /// Chooses the instruction for <paramref name="label"/>.
    /// </summary>
    /// <param name="label">Layout label.</param>
    /// <param name="useChartRecognition">Whether charts are recognised rather than kept as images.</param>
    /// <param name="useSealRecognition">Whether seals are recognised rather than kept as images.</param>
    public static string For(string label, bool useChartRecognition = false, bool useSealRecognition = false) =>
        label switch
        {
            "table" => Table,
            "chart" when useChartRecognition => Chart,
            "seal" when useSealRecognition => Seal,
            "spotting" => Spotting,
            _ when label.Contains("formula", StringComparison.Ordinal) && label != "formula_number" => Formula,
            _ => Ocr,
        };

    /// <summary>Preprocessing options for <paramref name="label"/>.</summary>
    /// <param name="label">Layout label.</param>
    /// <param name="budgets">Per-label pixel budgets; the pipeline's defaults when omitted.</param>
    /// <param name="baseline">Options to start from, for callers that changed something else.</param>
    public static VisionPreprocessorOptions Options(
        string label,
        BlockPixelBudgets? budgets = null,
        VisionPreprocessorOptions? baseline = null)
    {
        VisionPreprocessorOptions options = baseline ?? VisionPreprocessorOptions.Default;
        (int minPixels, int maxPixels) = (budgets ?? BlockPixelBudgets.Default).For(label);
        return options.WithPixelBudget(minPixels, maxPixels);
    }
}
