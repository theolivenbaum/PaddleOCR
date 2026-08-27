using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Pipeline;

namespace PaddleOcrSharp.Tests.Pipeline;

/// <summary>
/// The per-label pixel budgets, against the dispatch in
/// <c>_paddleocr_vl_collect_page_vlm_entries_core</c>.
/// </summary>
public class BlockPixelBudgetsTests
{
    [Theory]
    [InlineData("text")]
    [InlineData("table")]
    [InlineData("chart")]
    [InlineData("seal")]
    [InlineData("formula")]
    [InlineData("inline_formula")]
    [InlineData("formula_number")]
    [InlineData("paragraph_title")]
    public void DefaultsAreThePipelineDefaultsForEveryLabel(string label)
    {
        Assert.Equal(
            (BlockPrompt.DefaultMinPixels, BlockPrompt.DefaultMaxPixels),
            BlockPixelBudgets.Default.For(label));
    }

    [Fact]
    public void SpottingRaisesTheCeiling()
    {
        // Upstream hard-codes spotting rather than reading it from the configuration, so an
        // override must not reach it.
        var budgets = new BlockPixelBudgets { MaxPixels = 200_000 };

        Assert.Equal(
            (BlockPrompt.DefaultMinPixels, BlockPrompt.SpottingMaxPixels),
            budgets.For("spotting"));
    }

    [Fact]
    public void APerLabelCeilingOverridesTheSharedOne()
    {
        var budgets = new BlockPixelBudgets { MaxPixels = 500_000, TableMaxPixels = 1_400_000 };

        Assert.Equal(1_400_000, budgets.For("table").MaxPixels);
        Assert.Equal(500_000, budgets.For("text").MaxPixels);
        Assert.Equal(500_000, budgets.For("chart").MaxPixels);
    }

    [Fact]
    public void FormulaNumberIsTextNotFormula()
    {
        // The label contains "formula" but upstream excludes it explicitly, so it takes the OCR
        // instruction and the OCR budget.
        var budgets = new BlockPixelBudgets { OcrMaxPixels = 111_111, FormulaMaxPixels = 222_222 };

        Assert.Equal(111_111, budgets.For("formula_number").MaxPixels);
        Assert.Equal(222_222, budgets.For("inline_formula").MaxPixels);
        Assert.Equal(BlockPrompt.Ocr, BlockPrompt.For("formula_number"));
    }

    [Fact]
    public void OptionsCarryTheBudgetIntoPreprocessing()
    {
        var budgets = new BlockPixelBudgets { TableMinPixels = 50_000, TableMaxPixels = 600_000 };

        VisionPreprocessorOptions options = BlockPrompt.Options("table", budgets);

        Assert.Equal(50_000, options.MinPixels);
        Assert.Equal(600_000, options.MaxPixels);

        // Everything else about the preprocessing is untouched.
        Assert.Equal(VisionPreprocessorOptions.Default.Factor, options.Factor);
    }
}
