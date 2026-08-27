using PaddleOcrSharp.Models.Layout;
using PaddleOcrSharp.Models.Paddle;

namespace PaddleOcrSharp.Tests.Models;

/// <summary>
/// The detector's post-processing chain, driven with a hand-built detection tensor so the
/// individual rules are visible without running the graph.
/// </summary>
public class LayoutPostProcessTests
{
    /// <summary>Builds an <c>[N, 7]</c> detection tensor: class, score, box, reading order.</summary>
    private static PaddleTensor Detections(params (int Class, float Score, float[] Box, int Order)[] rows)
    {
        float[] data = new float[rows.Length * 7];

        for (int i = 0; i < rows.Length; i++)
        {
            data[i * 7] = rows[i].Class;
            data[(i * 7) + 1] = rows[i].Score;
            rows[i].Box.CopyTo(data, (i * 7) + 2);
            data[(i * 7) + 6] = rows[i].Order;
        }

        return PaddleTensor.FromFloats(data, [rows.Length, 7]);
    }

    private static LayoutDetector Detector() =>
        LayoutDetector.ForTesting(PaddleOcrSharp.Pipeline.BlockLabels.All);

    [Fact]
    public void BoxesAreRoundedToWholePixels()
    {
        // Half to even, as `np.round` does: 10.5 down to 10, 11.5 up to 12.
        PaddleTensor detections = Detections(
            (22, 0.9f, [10.5f, 20.4f, 11.5f, 30.6f], 0));

        IReadOnlyList<LayoutBox> boxes = Detector().PostProcess(
            detections, null, 800, 1000, LayoutOptions.Default with { Nms = false });

        LayoutBox box = Assert.Single(boxes);
        Assert.Equal(10f, box.Left);
        Assert.Equal(20f, box.Top);
        Assert.Equal(12f, box.Right);
        Assert.Equal(31f, box.Bottom);
    }

    [Fact]
    public void APerClassThresholdOptsEveryOtherClassIntoTheDefault()
    {
        // Naming one class sends the rest to 0.5 rather than to the shared threshold, so the
        // 0.4-scoring block of an unnamed class goes even though the shared threshold is 0.3.
        PaddleTensor detections = Detections(
            (22, 0.35f, [10, 10, 100, 40], 0),
            (17, 0.40f, [10, 50, 100, 80], 1));

        IReadOnlyList<LayoutBox> boxes = Detector().PostProcess(
            detections,
            null,
            800,
            1000,
            LayoutOptions.Default with
            {
                Nms = false,
                ClassThresholds = new Dictionary<int, float> { [22] = 0.3f },
            });

        Assert.Equal(["text"], boxes.Select(box => box.Label));
    }

    [Fact]
    public void OnlyTheNamedClassesAreExpanded()
    {
        PaddleTensor detections = Detections(
            (22, 0.9f, [100, 100, 200, 200], 0),
            (17, 0.9f, [100, 300, 200, 400], 1));

        IReadOnlyList<LayoutBox> boxes = Detector().PostProcess(
            detections,
            null,
            800,
            1000,
            LayoutOptions.Default with
            {
                Nms = false,
                ClassUnclipRatios = new Dictionary<int, (float, float)> { [22] = (2f, 1f) },
            });

        LayoutBox expanded = boxes.Single(box => box.Label == "text");
        LayoutBox untouched = boxes.Single(box => box.Label == "paragraph_title");

        Assert.Equal(50f, expanded.Left);
        Assert.Equal(250f, expanded.Right);
        Assert.Equal(100f, expanded.Top);

        Assert.Equal(100f, untouched.Left);
        Assert.Equal(200f, untouched.Right);
    }
}
