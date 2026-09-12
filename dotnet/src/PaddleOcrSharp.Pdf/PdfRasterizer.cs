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

    /// <summary>Largest page raster produced by default, in pixels.</summary>
    /// <remarks>
    /// <para>
    /// A dpi is a resolution per inch of page, and a PDF's declared page size is not always the
    /// size of the document someone scanned. <c>pdf_scanned/nougat_010_scanned.pdf</c> is one
    /// 6623x4678 scan of an A-series landscape page — its aspect ratio is √2 to three decimals —
    /// placed on a MediaBox of 4967x3508 points, which claims the page is 69 inches by 49. At 144
    /// dpi that renders to 9934x7017: seventy megapixels, half of them interpolated, because the
    /// scan itself carries only thirty-one. The pipeline then encoded ten megapixels of block
    /// crops out of it and spent eight minutes on the page.
    /// </para>
    /// <para>
    /// The budget is set to leave every ordinary page alone. A mis-declared MediaBox is common and
    /// usually mild: most of <c>test_documents/pdf_scanned</c> declares 1275x1650 points, which is
    /// a letter page written in 150-dpi pixels, and renders to 8.4 megapixels here. Twelve clears
    /// all of those — so those files come out byte-identical, and so does anything up to ISO A2 —
    /// while still catching the two that are not mild: the file above at seventy megapixels, and
    /// <c>nougat_009_scanned.pdf</c>, which claims to be 87 inches tall, at twenty-seven.
    /// </para>
    /// <para>
    /// A page over the budget is rendered at the resolution that fits, which for the file above is
    /// 350 dpi of its real A4-landscape page rather than 144 dpi of a page size it does not have.
    /// Pass <c>--max-page-pixels 0</c> to render at the requested dpi whatever the page claims to
    /// be, and a smaller value to hold large scans down harder.
    /// </para>
    /// </remarks>
    public const int DefaultMaxPagePixels = 12_000_000;

    /// <summary>Number of pages in the document.</summary>
    public static int GetPageCount(string path, string? password = null) =>
        Conversion.GetPageCount(File.ReadAllBytes(path), password);

    /// <summary>Renders every page of <paramref name="path"/>.</summary>
    /// <param name="path">Path to the PDF.</param>
    /// <param name="dpi">Rendering resolution.</param>
    /// <param name="password">Password for an encrypted document.</param>
    /// <param name="maxPages">Optional cap on the number of pages rendered.</param>
    /// <param name="maxPagePixels">
    /// Largest raster to produce for one page; a page that would exceed it is rendered at the
    /// resolution that fits. Zero renders at <paramref name="dpi"/> whatever the page claims to be.
    /// </param>
    public static IEnumerable<RgbImage> Render(
        string path,
        int dpi = DefaultDpi,
        string? password = null,
        int maxPages = 0,
        int maxPagePixels = DefaultMaxPagePixels)
    {
        byte[] bytes = File.ReadAllBytes(path);
        return Render(bytes, dpi, password, maxPages, maxPagePixels);
    }

    /// <summary>Renders every page of an in-memory PDF.</summary>
    /// <param name="pdf">The document.</param>
    /// <param name="dpi">Rendering resolution.</param>
    /// <param name="password">Password for an encrypted document.</param>
    /// <param name="maxPages">Optional cap on the number of pages rendered.</param>
    /// <param name="maxPagePixels">
    /// Largest raster to produce for one page; a page that would exceed it is rendered at the
    /// resolution that fits. Zero renders at <paramref name="dpi"/> whatever the page claims to be.
    /// </param>
    public static IEnumerable<RgbImage> Render(
        byte[] pdf,
        int dpi = DefaultDpi,
        string? password = null,
        int maxPages = 0,
        int maxPagePixels = DefaultMaxPagePixels)
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
                    Dpi: ResolutionFor(pdf, index, password, dpi, maxPagePixels),
                    WithAnnotations: true,
                    WithFormFill: true));

            yield return FromBitmap(bitmap);
        }
    }

    /// <summary>
    /// The dpi to render a page of the given size at: the requested one, or the largest that keeps
    /// the raster inside <paramref name="maxPagePixels"/>.
    /// </summary>
    /// <param name="widthPoints">Page width in PostScript points.</param>
    /// <param name="heightPoints">Page height in points.</param>
    /// <param name="dpi">Requested resolution.</param>
    /// <param name="maxPagePixels">Pixel budget, or zero for none.</param>
    /// <returns>The resolution to render at, never below 1.</returns>
    public static int ResolutionFor(double widthPoints, double heightPoints, int dpi, int maxPagePixels)
    {
        if (maxPagePixels <= 0 || widthPoints <= 0 || heightPoints <= 0)
        {
            return dpi;
        }

        double pixels = widthPoints / 72 * dpi * (heightPoints / 72 * dpi);
        return pixels <= maxPagePixels
            ? dpi
            : Math.Max(1, (int)(dpi * Math.Sqrt(maxPagePixels / pixels)));
    }

    private static int ResolutionFor(byte[] pdf, int index, string? password, int dpi, int maxPagePixels)
    {
        if (maxPagePixels <= 0)
        {
            return dpi;
        }

        System.Drawing.SizeF points = Conversion.GetPageSize(pdf, index, password);
        return ResolutionFor(points.Width, points.Height, dpi, maxPagePixels);
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
