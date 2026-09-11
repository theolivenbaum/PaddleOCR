using PaddleOcrSharp.Imaging;
using PDFtoImage;
using SkiaSharp;

namespace PaddleOcrSharp.Pdf;

/// <summary>
/// Renders PDF pages to images so the document pipeline can parse them.
/// </summary>
/// <remarks>
/// This lives in its own assembly because it is the one part of the port that needs a native
/// component: PDFium does the rasterisation. Nothing in the model path depends on it.
/// </remarks>
public static class PdfRasterizer
{
    /// <summary>Resolution used when none is given, matching upstream's.</summary>
    /// <remarks>
    /// PaddleX renders a page with <c>zoom=2.0</c> over the PDF's natural 72 dpi
    /// (<c>inference/utils/io/readers.py</c>), so 144 is what the pipeline this port reproduces
    /// actually sees. The port defaulted to 200 for long enough to distort a comparison: at 200
    /// a page carries 1.93x the pixels, which is 1.93x the patches for the vision tower to
    /// encode, so the port was doing appreciably more work than upstream on the same file and
    /// still being timed against it. Raise it with <c>--dpi</c> where a scan of small type needs
    /// it; the model's own pixel budget caps a block either way.
    /// </remarks>
    public const int DefaultDpi = 144;

    /// <summary>Number of pages in the document.</summary>
    public static int GetPageCount(string path, string? password = null) =>
        Conversion.GetPageCount(File.ReadAllBytes(path), password);

    /// <summary>Renders every page of <paramref name="path"/>.</summary>
    /// <param name="path">Path to the PDF.</param>
    /// <param name="dpi">Rendering resolution.</param>
    /// <param name="password">Password for an encrypted document.</param>
    /// <param name="maxPages">Optional cap on the number of pages rendered.</param>
    public static IEnumerable<RgbImage> Render(
        string path,
        int dpi = DefaultDpi,
        string? password = null,
        int maxPages = 0)
    {
        byte[] bytes = File.ReadAllBytes(path);
        return Render(bytes, dpi, password, maxPages);
    }

    /// <summary>Renders every page of an in-memory PDF.</summary>
    public static IEnumerable<RgbImage> Render(
        byte[] pdf,
        int dpi = DefaultDpi,
        string? password = null,
        int maxPages = 0)
    {
        int pages = Conversion.GetPageCount(pdf, password);
        if (maxPages > 0)
        {
            pages = Math.Min(pages, maxPages);
        }

        for (int index = 0; index < pages; index++)
        {
            using SKBitmap bitmap = Conversion.ToImage(
                pdf,
                index,
                password,
                new RenderOptions(Dpi: dpi, WithAnnotations: true, WithFormFill: true));

            yield return FromBitmap(bitmap);
        }
    }

    private static RgbImage FromBitmap(SKBitmap bitmap)
    {
        using SKBitmap rgba = bitmap.ColorType == SKColorType.Rgba8888
            ? bitmap.Copy()
            : bitmap.Copy(SKColorType.Rgba8888);

        RgbImage image = RgbImage.Rent(rgba.Width, rgba.Height);
        ReadOnlySpan<byte> source = rgba.GetPixelSpan();

        for (int y = 0; y < rgba.Height; y++)
        {
            ReadOnlySpan<byte> sourceRow = source.Slice(y * rgba.RowBytes, rgba.Width * 4);
            Span<byte> targetRow = image.Row(y);

            for (int x = 0; x < rgba.Width; x++)
            {
                byte r = sourceRow[(x * 4) + 0];
                byte g = sourceRow[(x * 4) + 1];
                byte b = sourceRow[(x * 4) + 2];
                byte a = sourceRow[(x * 4) + 3];

                if (a != 255)
                {
                    // PDFium leaves unpainted areas transparent; a document page is white there.
                    int inverse = 255 - a;
                    r = (byte)(((r * a) + (255 * inverse)) / 255);
                    g = (byte)(((g * a) + (255 * inverse)) / 255);
                    b = (byte)(((b * a) + (255 * inverse)) / 255);
                }

                targetRow[(x * 3) + 0] = r;
                targetRow[(x * 3) + 1] = g;
                targetRow[(x * 3) + 2] = b;
            }
        }

        return image;
    }
}
