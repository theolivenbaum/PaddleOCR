using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models.Paddle;

namespace PaddleOcrSharp.Models.Preprocessing;

/// <summary>
/// UVDoc: flattens a photographed or scanned page that is curled or creased.
/// </summary>
/// <remarks>
/// The model takes the image scaled to <c>[0, 1]</c> in NCHW with no resize and no mean/std
/// normalisation, and returns a warped image in the same layout, which
/// <c>DocTrPostProcess</c> scales back by 255.
/// </remarks>
public sealed class DocumentUnwarper : IDisposable
{
    private readonly PirInterpreter _interpreter;

    private DocumentUnwarper(PirInterpreter interpreter) => _interpreter = interpreter;

    /// <summary>Loads the model from a model directory.</summary>
    public static DocumentUnwarper Load(string directory) => new(PirInterpreter.Load(directory));

    /// <summary>Returns a flattened copy of <paramref name="page"/>.</summary>
    public RgbImage Unwarp(RgbImage page)
    {
        PaddleTensor input = PaddleTensor.Float([1, 3, page.Height, page.Width]);
        Span<float> pixels = input.FloatSpan;
        int plane = page.Width * page.Height;

        for (int y = 0; y < page.Height; y++)
        {
            ReadOnlySpan<byte> row = page.Row(y);
            int rowBase = y * page.Width;
            for (int x = 0; x < page.Width; x++)
            {
                int offset = x * 3;
                pixels[rowBase + x] = row[offset] * (1f / 255f);
                pixels[plane + rowBase + x] = row[offset + 1] * (1f / 255f);
                pixels[(2 * plane) + rowBase + x] = row[offset + 2] * (1f / 255f);
            }
        }

        Dictionary<string, PaddleTensor> outputs = _interpreter.Run(
            new Dictionary<string, PaddleTensor>(StringComparer.Ordinal) { ["image"] = input });

        PaddleTensor warped = outputs.Values.First();
        int height = warped.Shape[2];
        int width = warped.Shape[3];
        int outputPlane = width * height;

        RgbImage result = RgbImage.Rent(width, height);
        ReadOnlySpan<float> values = warped.FloatSpan;

        for (int y = 0; y < height; y++)
        {
            Span<byte> row = result.Row(y);
            int rowBase = y * width;
            for (int x = 0; x < width; x++)
            {
                int offset = x * 3;
                row[offset] = Saturate(values[rowBase + x]);
                row[offset + 1] = Saturate(values[outputPlane + rowBase + x]);
                row[offset + 2] = Saturate(values[(2 * outputPlane) + rowBase + x]);
            }
        }

        return result;
    }

    private static byte Saturate(float value)
    {
        int scaled = (int)MathF.Round(value * 255f);
        return scaled switch
        {
            < 0 => 0,
            > 255 => 255,
            _ => (byte)scaled,
        };
    }

    /// <inheritdoc />
    public void Dispose() => _interpreter.Dispose();
}
