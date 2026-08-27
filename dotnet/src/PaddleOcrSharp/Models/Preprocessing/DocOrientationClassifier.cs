using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models.Paddle;

namespace PaddleOcrSharp.Models.Preprocessing;

/// <summary>
/// PP-LCNet_x1_0_doc_ori: predicts whether a page is rotated by 0°, 90°, 180° or 270°.
/// </summary>
/// <remarks>
/// Pre-processing follows the model's <c>inference.yml</c>: resize the short side to 256, centre
/// crop to 224, scale by 1/255 and normalise with the ImageNet statistics, then NCHW.
/// </remarks>
public sealed class DocOrientationClassifier : IDisposable
{
    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] StandardDeviation = [0.229f, 0.224f, 0.225f];
    private static readonly int[] Angles = [0, 90, 180, 270];

    private const int ShortSide = 256;
    private const int CropSize = 224;

    private readonly PirInterpreter _interpreter;

    private DocOrientationClassifier(PirInterpreter interpreter) => _interpreter = interpreter;

    /// <summary>Loads the classifier from a model directory.</summary>
    public static DocOrientationClassifier Load(string directory) => new(PirInterpreter.Load(directory));

    /// <summary>Predicts the page's rotation.</summary>
    /// <param name="page">The page image.</param>
    /// <returns>The predicted angle in degrees and its score.</returns>
    public (int Angle, float Score) Predict(RgbImage page)
    {
        using RgbImage prepared = PrepareInput(page);
        PaddleTensor input = PaddleTensor.Float([1, 3, CropSize, CropSize]);

        Span<float> pixels = input.FloatSpan;
        int plane = CropSize * CropSize;

        for (int y = 0; y < CropSize; y++)
        {
            ReadOnlySpan<byte> row = prepared.Row(y);
            for (int x = 0; x < CropSize; x++)
            {
                int offset = x * 3;
                int index = (y * CropSize) + x;
                for (int c = 0; c < 3; c++)
                {
                    pixels[(c * plane) + index] =
                        ((row[offset + c] * (1f / 255f)) - Mean[c]) / StandardDeviation[c];
                }
            }
        }

        Dictionary<string, PaddleTensor> outputs = _interpreter.Run(
            new Dictionary<string, PaddleTensor>(StringComparer.Ordinal) { ["x"] = input });

        PaddleTensor logits = outputs.Values.First();
        Span<float> scores = logits.FloatSpan;

        int best = 0;
        for (int i = 1; i < scores.Length && i < Angles.Length; i++)
        {
            if (scores[i] > scores[best])
            {
                best = i;
            }
        }

        return (Angles[best], scores[best]);
    }

    /// <summary>Rotates <paramref name="page"/> so that its text is upright.</summary>
    public RgbImage Correct(RgbImage page)
    {
        (int angle, _) = Predict(page);
        return angle == 0 ? page.Clone() : Rotate(page, angle);
    }

    /// <summary>
    /// Rotates an image counter-clockwise by 90, 180 or 270 degrees.
    /// </summary>
    /// <remarks>
    /// Counter-clockwise because that is the direction <c>rotate_image</c> turns — it goes through
    /// <c>cv2.getRotationMatrix2D</c>, where a positive angle is counter-clockwise — and the angle
    /// it is given is the classifier's own prediction. Turning the other way would take a page the
    /// model called 90 degrees and leave it at 270.
    /// </remarks>
    public static RgbImage Rotate(RgbImage image, int degrees)
    {
        int normalised = ((degrees % 360) + 360) % 360;
        if (normalised == 0)
        {
            return image.Clone();
        }

        bool swapAxes = normalised is 90 or 270;
        RgbImage result = RgbImage.Rent(
            swapAxes ? image.Height : image.Width,
            swapAxes ? image.Width : image.Height);

        for (int y = 0; y < image.Height; y++)
        {
            ReadOnlySpan<byte> row = image.Row(y);
            for (int x = 0; x < image.Width; x++)
            {
                (int tx, int ty) = normalised switch
                {
                    90 => (y, image.Width - 1 - x),
                    180 => (image.Width - 1 - x, image.Height - 1 - y),
                    _ => (image.Height - 1 - y, x),
                };

                Span<byte> target = result.Row(ty);
                target[tx * 3] = row[x * 3];
                target[(tx * 3) + 1] = row[(x * 3) + 1];
                target[(tx * 3) + 2] = row[(x * 3) + 2];
            }
        }

        return result;
    }

    /// <summary>Resizes the short side to 256 and centre-crops to 224.</summary>
    private static RgbImage PrepareInput(RgbImage page)
    {
        float scale = (float)ShortSide / Math.Min(page.Width, page.Height);
        int width = Math.Max(1, (int)MathF.Round(page.Width * scale));
        int height = Math.Max(1, (int)MathF.Round(page.Height * scale));

        using RgbImage resized = OpenCvResize.ResizeBicubic(page, width, height);

        int left = (width - CropSize) / 2;
        int top = (height - CropSize) / 2;
        return resized.Crop(left, top, left + CropSize, top + CropSize);
    }

    /// <inheritdoc />
    public void Dispose() => _interpreter.Dispose();
}
