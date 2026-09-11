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
    /// <param name="capToSource">
    /// Whether to render a scanned page at its own resolution when that is lower than
    /// <paramref name="dpi"/>. See <see cref="GetSourceDpi(byte[], int, string?)"/>.
    /// </param>
    public static IEnumerable<RgbImage> Render(
        string path,
        int dpi = DefaultDpi,
        string? password = null,
        int maxPages = 0,
        bool capToSource = true)
    {
        byte[] bytes = File.ReadAllBytes(path);
        return Render(bytes, dpi, password, maxPages, capToSource);
    }

    /// <summary>Renders every page of an in-memory PDF.</summary>
    /// <param name="pdf">The document bytes.</param>
    /// <param name="dpi">Rendering resolution.</param>
    /// <param name="password">Password for an encrypted document.</param>
    /// <param name="maxPages">Optional cap on the number of pages rendered.</param>
    /// <param name="capToSource">
    /// Whether to render a scanned page at its own resolution when that is lower than
    /// <paramref name="dpi"/>. See <see cref="GetSourceDpi(byte[], int, string?)"/>.
    /// </param>
    public static IEnumerable<RgbImage> Render(
        byte[] pdf,
        int dpi = DefaultDpi,
        string? password = null,
        int maxPages = 0,
        bool capToSource = true)
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
                new RenderOptions(
                    Dpi: capToSource ? EffectiveDpi(pdf, index, password, dpi) : dpi,
                    WithAnnotations: true,
                    WithFormFill: true));

            yield return FromBitmap(bitmap);
        }
    }

    /// <summary>
    /// The resolution a scanned page's own content can supply, or <see langword="null"/> when the
    /// page is not a scan and so has no such limit.
    /// </summary>
    /// <remarks>
    /// A page counts as a scan when it puts no visible text or vector art on the page and its
    /// image objects cover essentially all of it — an invisible OCR text layer over the image does
    /// not disqualify it. The figure is the highest resolution among those images, as PDFium
    /// reports it, which accounts for how each image was scaled into its box.
    /// </remarks>
    /// <param name="pdf">The document bytes.</param>
    /// <param name="page">Zero-based page index.</param>
    /// <param name="password">Password for an encrypted document.</param>
    /// <returns>The page's own resolution in dpi, or <see langword="null"/> when it is not a scan.</returns>
    public static double? GetSourceDpi(byte[] pdf, int page, string? password = null) =>
        PdfSource.GetSourceDpi(pdf, page, password);

    /// <summary>
    /// The resolution a page is worth rendering at: what was asked for, or what the page can
    /// supply, whichever is lower.
    /// </summary>
    /// <remarks>
    /// Only ever lowers the figure. Rendering a 96 dpi scan at 144 interpolates it to 1.5x the
    /// pixels and so 2.25x the patches for the vision tower to encode, and the model reads the
    /// same characters from them — over this repository's scanned corpus, 96 to 200 dpi cost 94%
    /// more wall time for 0.02 points of character accuracy. Rendering *below* the request is
    /// never done for a page that could supply more, because that would lose real detail.
    /// </remarks>
    private static int EffectiveDpi(byte[] pdf, int page, string? password, int requested)
    {
        double? source = PdfSource.GetSourceDpi(pdf, page, password);
        if (source is not > 0)
        {
            return requested;
        }

        // Round up: a page whose images land at 95.6 dpi still wants all 96 of them.
        int native = (int)Math.Ceiling(source.Value - 0.001);
        return Math.Clamp(native, 1, requested);
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
