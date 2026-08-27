using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models.Paddle;
using PaddleOcrSharp.Pipeline;

namespace PaddleOcrSharp.Models.Layout;

/// <summary>
/// PP-DocLayoutV3, the RT-DETR document layout detector the PaddleOCR-VL-1.6 pipeline uses.
/// </summary>
/// <remarks>
/// <para>
/// The model runs through <see cref="PirInterpreter"/>. Pre-processing follows the shipped
/// <c>inference.yml</c> — bicubic resize to 800×800 with OpenCV semantics, a 1/255 rescale, no
/// mean/std normalisation, NCHW — and post-processing follows PaddleX's <c>DetPostProcess</c>:
/// threshold, layout-aware NMS, the page-sized-image filter, per-class containment resolution,
/// reading-order sort, then unclipping.
/// </para>
/// <para>
/// The detector's own post-process recovers the page size as <c>im_shape / scale_factor</c>, so
/// feeding the true page size with a unit scale factor puts the fetched boxes straight into
/// original-image coordinates.
/// </para>
/// </remarks>
public sealed class LayoutDetector : IDisposable
{
    private const int InputSize = 800;

    private readonly PirInterpreter _interpreter;
    private readonly string[] _labels;

    private LayoutDetector(PirInterpreter interpreter, string[] labels)
    {
        _interpreter = interpreter;
        _labels = labels;
    }

    /// <summary>Class names, indexed by class id.</summary>
    public IReadOnlyList<string> Labels => _labels;

    /// <summary>Loads the detector from a model directory.</summary>
    /// <param name="directory">
    /// Directory holding <c>inference.json</c>, <c>inference.pdiparams</c> and, optionally,
    /// <c>inference.yml</c> with the label list.
    /// </param>
    public static LayoutDetector Load(string directory)
    {
        PirInterpreter interpreter = PirInterpreter.Load(directory);
        string[] labels = ReadLabels(Path.Combine(directory, "inference.yml")) ?? BlockLabels.All;
        return new LayoutDetector(interpreter, labels);
    }

    /// <summary>Detects layout regions on <paramref name="page"/>.</summary>
    /// <param name="page">The page image.</param>
    /// <param name="options">Post-processing settings.</param>
    public IReadOnlyList<LayoutBox> Detect(RgbImage page, LayoutOptions? options = null) =>
        Detect(page, options, profile: null);

    /// <summary>Detects layout regions, optionally recording per-operator timings.</summary>
    /// <param name="page">The page image.</param>
    /// <param name="options">Post-processing settings.</param>
    /// <param name="profile">Receives per-operator wall-clock time when supplied.</param>
    public IReadOnlyList<LayoutBox> Detect(RgbImage page, LayoutOptions? options, PirProfile? profile)
    {
        LayoutOptions settings = options ?? LayoutOptions.Default;

        using RgbImage resized = OpenCvResize.ResizeBicubic(page, InputSize, InputSize);

        PaddleTensor image = PaddleTensor.Float([1, 3, InputSize, InputSize]);
        Span<float> pixels = image.FloatSpan;
        int plane = InputSize * InputSize;

        for (int y = 0; y < InputSize; y++)
        {
            ReadOnlySpan<byte> row = resized.Row(y);
            int rowBase = y * InputSize;
            for (int x = 0; x < InputSize; x++)
            {
                int offset = x * 3;
                pixels[rowBase + x] = row[offset] * (1f / 255f);
                pixels[plane + rowBase + x] = row[offset + 1] * (1f / 255f);
                pixels[(2 * plane) + rowBase + x] = row[offset + 2] * (1f / 255f);
            }
        }

        var inputs = new Dictionary<string, PaddleTensor>(StringComparer.Ordinal)
        {
            ["image"] = image,
            ["im_shape"] = PaddleTensor.FromFloats([page.Height, page.Width], [1, 2]),
            ["scale_factor"] = PaddleTensor.FromFloats([1f, 1f], [1, 2]),
        };

        Dictionary<string, PaddleTensor> outputs = _interpreter.Run(inputs, trace: null, profile);
        PaddleTensor detections = outputs["fetch_name_0"];

        return PostProcess(detections, page.Width, page.Height, settings);
    }

    /// <summary>
    /// Applies the pipeline's post-processing chain to the raw <c>[N, 7]</c> detection tensor.
    /// </summary>
    internal IReadOnlyList<LayoutBox> PostProcess(
        PaddleTensor detections,
        int pageWidth,
        int pageHeight,
        LayoutOptions options)
    {
        int rows = detections.Shape[0];
        int columns = detections.Shape[1];
        ReadOnlySpan<float> data = detections.FloatSpan;

        var boxes = new List<LayoutBox>();
        for (int row = 0; row < rows; row++)
        {
            int classId = (int)data[row * columns];
            float score = data[(row * columns) + 1];
            if (classId < 0 || score <= options.Threshold)
            {
                continue;
            }

            boxes.Add(new LayoutBox(
                classId,
                classId < _labels.Length ? _labels[classId] : "unknown",
                score,
                data[(row * columns) + 2],
                data[(row * columns) + 3],
                data[(row * columns) + 4],
                data[(row * columns) + 5],
                columns > 6 ? (int)data[(row * columns) + 6] : row));
        }

        if (options.Nms)
        {
            boxes = ApplyNms(boxes, options.NmsIouSameClass, options.NmsIouDifferentClass);
        }

        if (options.FilterPageSizedBoxes && boxes.Count > 1)
        {
            boxes = FilterPageSizedImages(boxes, pageWidth, pageHeight);
        }

        boxes = ResolveContainment(boxes, options.MergeModes);
        boxes.Sort(static (left, right) => left.ReadingOrder.CompareTo(right.ReadingOrder));

        if (options.UnclipRatio != (1f, 1f))
        {
            for (int i = 0; i < boxes.Count; i++)
            {
                boxes[i] = Unclip(boxes[i], options.UnclipRatio);
            }
        }

        return boxes;
    }

    /// <summary>
    /// Greedy NMS with a looser threshold across classes than within one, matching PaddleX's
    /// <c>nms(boxes, iou_same, iou_diff)</c>.
    /// </summary>
    private static List<LayoutBox> ApplyNms(List<LayoutBox> boxes, float iouSame, float iouDifferent)
    {
        var order = Enumerable.Range(0, boxes.Count).ToList();
        order.Sort((left, right) => boxes[right].Score.CompareTo(boxes[left].Score));

        var kept = new List<LayoutBox>();
        var remaining = new List<int>(order);

        while (remaining.Count > 0)
        {
            int current = remaining[0];
            remaining.RemoveAt(0);
            kept.Add(boxes[current]);

            var survivors = new List<int>(remaining.Count);
            foreach (int candidate in remaining)
            {
                float threshold = boxes[current].ClassId == boxes[candidate].ClassId ? iouSame : iouDifferent;
                if (InflatedIou(boxes[current], boxes[candidate]) < threshold)
                {
                    survivors.Add(candidate);
                }
            }

            remaining = survivors;
        }

        return kept;
    }

    /// <summary>
    /// IoU with the inclusive pixel convention (<c>+1</c> on each extent) that PaddleX's
    /// <c>iou</c> helper uses.
    /// </summary>
    private static float InflatedIou(LayoutBox a, LayoutBox b)
    {
        float width = Math.Max(0f, Math.Min(a.Right, b.Right) - Math.Max(a.Left, b.Left) + 1f);
        float height = Math.Max(0f, Math.Min(a.Bottom, b.Bottom) - Math.Max(a.Top, b.Top) + 1f);
        float intersection = width * height;

        float areaA = (a.Right - a.Left + 1f) * (a.Bottom - a.Top + 1f);
        float areaB = (b.Right - b.Left + 1f) * (b.Bottom - b.Top + 1f);
        float union = areaA + areaB - intersection;

        return union <= 0f ? 0f : intersection / union;
    }

    /// <summary>
    /// Drops <c>image</c> detections that cover almost the whole page, which are usually the
    /// scan background rather than a figure.
    /// </summary>
    private List<LayoutBox> FilterPageSizedImages(List<LayoutBox> boxes, int pageWidth, int pageHeight)
    {
        int imageClass = Array.IndexOf(_labels, "image");
        if (imageClass < 0)
        {
            return boxes;
        }

        float areaThreshold = pageWidth > pageHeight ? 0.82f : 0.93f;
        float pageArea = (float)pageWidth * pageHeight;

        var filtered = new List<LayoutBox>(boxes.Count);
        foreach (LayoutBox box in boxes)
        {
            if (box.ClassId != imageClass)
            {
                filtered.Add(box);
                continue;
            }

            LayoutBox clamped = box.ClampTo(pageWidth, pageHeight);
            if (clamped.Area <= areaThreshold * pageArea)
            {
                filtered.Add(box);
            }
        }

        return filtered.Count == 0 ? boxes : filtered;
    }

    /// <summary>
    /// Resolves boxes that sit inside other boxes, per class.
    /// </summary>
    /// <remarks>
    /// Containment uses upstream's asymmetric test: box A counts as inside box B when 90% of A's
    /// area is covered by B.
    /// </remarks>
    private static List<LayoutBox> ResolveContainment(
        List<LayoutBox> boxes,
        IReadOnlyDictionary<int, LayoutMergeMode> modes)
    {
        if (boxes.Count < 2 || modes.Count == 0)
        {
            return boxes;
        }

        bool[] keep = new bool[boxes.Count];
        Array.Fill(keep, true);

        foreach ((int classId, LayoutMergeMode mode) in modes)
        {
            if (mode == LayoutMergeMode.Union)
            {
                continue;
            }

            bool[] containsOther = new bool[boxes.Count];
            bool[] containedByOther = new bool[boxes.Count];

            for (int i = 0; i < boxes.Count; i++)
            {
                for (int j = 0; j < boxes.Count; j++)
                {
                    if (i == j)
                    {
                        continue;
                    }

                    bool relevant = mode == LayoutMergeMode.Large
                        ? boxes[j].ClassId == classId
                        : boxes[i].ClassId == classId;

                    if (relevant && IsContained(boxes[i], boxes[j]))
                    {
                        containedByOther[i] = true;
                        containsOther[j] = true;
                    }
                }
            }

            for (int i = 0; i < boxes.Count; i++)
            {
                keep[i] &= mode == LayoutMergeMode.Large
                    ? !containedByOther[i]
                    : !containsOther[i] || containedByOther[i];
            }
        }

        var result = new List<LayoutBox>(boxes.Count);
        for (int i = 0; i < boxes.Count; i++)
        {
            if (keep[i])
            {
                result.Add(boxes[i]);
            }
        }

        return result;
    }

    private static bool IsContained(LayoutBox inner, LayoutBox outer) =>
        inner.Area > 0 && inner.IntersectionWith(outer) / inner.Area >= 0.9f;

    private static LayoutBox Unclip(LayoutBox box, (float Horizontal, float Vertical) ratio)
    {
        float centerX = box.Left + (box.Width / 2f);
        float centerY = box.Top + (box.Height / 2f);
        float width = box.Width * ratio.Horizontal;
        float height = box.Height * ratio.Vertical;

        return box with
        {
            Left = centerX - (width / 2f),
            Top = centerY - (height / 2f),
            Right = centerX + (width / 2f),
            Bottom = centerY + (height / 2f),
        };
    }

    /// <summary>Reads <c>label_list</c> out of the model's <c>inference.yml</c>.</summary>
    private static string[]? ReadLabels(string path)
    {
        if (!File.Exists(path))
        {
            return null;
        }

        var labels = new List<string>();
        bool inList = false;

        foreach (string line in File.ReadLines(path))
        {
            if (line.StartsWith("label_list:", StringComparison.Ordinal))
            {
                inList = true;
                continue;
            }

            if (!inList)
            {
                continue;
            }

            if (line.StartsWith("- ", StringComparison.Ordinal))
            {
                labels.Add(line[2..].Trim());
            }
            else if (line.Length > 0 && !char.IsWhiteSpace(line[0]))
            {
                break;
            }
        }

        return labels.Count > 0 ? [.. labels] : null;
    }

    /// <inheritdoc />
    public void Dispose() => _interpreter.Dispose();
}
