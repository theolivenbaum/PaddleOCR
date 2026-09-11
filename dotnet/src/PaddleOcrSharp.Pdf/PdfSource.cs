using System.Runtime.InteropServices;
using PDFtoImage;

namespace PaddleOcrSharp.Pdf;

/// <summary>
/// What resolution a PDF page's own content can supply, read from the page's objects.
/// </summary>
/// <remarks>
/// <para>
/// A scanned PDF is a photograph of a page wrapped in a page box, and the photograph has a fixed
/// resolution. Rendering such a page above that resolution interpolates rather than reveals: the
/// renderer produces more pixels, the vision tower encodes proportionally more patches, and the
/// model reads exactly the same text. Measured over this repository's scanned corpus, whose pages
/// are all 96 dpi, going from 96 to 200 dpi cost 94% more wall time and moved character accuracy
/// by 0.02 points — within the run-to-run noise of the metric.
/// </para>
/// <para>
/// PDFium already knows the number: <c>FPDFImageObj_GetImageMetadata</c> reports an image
/// object's pixel dimensions and the dpi those pixels land on the page at, which accounts for how
/// the image was scaled into its box. What this type adds is deciding when that number describes
/// the whole page rather than one illustration on it.
/// </para>
/// <para>
/// The library is initialised by <c>PDFtoImage</c>, which this assembly already uses to render;
/// every entry point here forces that initialisation first and treats any failure as "unknown",
/// so a page that cannot be inspected is simply rendered at the resolution asked for. PDFium is
/// not thread-safe, and neither is this.
/// </para>
/// </remarks>
internal static partial class PdfSource
{
    private const string Library = "pdfium";

    /// <summary>Page-object types, from <c>fpdf_edit.h</c>.</summary>
    private const int Text = 1;
    private const int Image = 3;

    /// <summary>Text drawn in render mode 3 puts no ink on the page — an OCR layer over a scan.</summary>
    private const int Invisible = 3;

    /// <summary>
    /// Fraction of the page that image objects must cover before the page counts as a scan.
    /// </summary>
    /// <remarks>
    /// A scan covers all of it; the illustrations in a typeset document cover a fraction. The
    /// margin below 1 is for scans whose image is inset by a hairline, and for pages whose box
    /// was cropped after scanning.
    /// </remarks>
    private const double ScanCoverage = 0.9;

    /// <summary>
    /// The resolution the page's own content can supply, or <see langword="null"/> when the page
    /// is not a scan and so has no such limit.
    /// </summary>
    /// <param name="pdf">The document bytes.</param>
    /// <param name="page">Zero-based page index.</param>
    /// <param name="password">Password for an encrypted document.</param>
    public static double? GetSourceDpi(byte[] pdf, int page, string? password)
    {
        // Forces PDFtoImage to initialise PDFium, which the calls below need and this assembly
        // does not own. Cheap, and it is what Render does first anyway.
        try
        {
            if (page < 0 || page >= Conversion.GetPageCount(pdf, password))
            {
                return null;
            }
        }
        catch (Exception)
        {
            return null;
        }

        GCHandle pinned = GCHandle.Alloc(pdf, GCHandleType.Pinned);
        nint document = 0;
        nint handle = 0;

        try
        {
            document = FPDF_LoadMemDocument(pinned.AddrOfPinnedObject(), pdf.Length, password);
            if (document == 0)
            {
                return null;
            }

            handle = FPDF_LoadPage(document, page);
            return handle == 0 ? null : Inspect(handle);
        }
        catch (DllNotFoundException)
        {
            return null;
        }
        catch (EntryPointNotFoundException)
        {
            return null;
        }
        finally
        {
            if (handle != 0)
            {
                FPDF_ClosePage(handle);
            }

            if (document != 0)
            {
                FPDF_CloseDocument(document);
            }

            pinned.Free();
        }
    }

    /// <summary>
    /// The page's source resolution, or <see langword="null"/> when its content is not one scan.
    /// </summary>
    /// <remarks>
    /// Two conditions, both conservative, because the cost of wrongly deciding a page is a scan is
    /// a blurred render and the cost of missing one is only the render we would have done anyway.
    /// The page must carry no visible text — an invisible OCR layer over the scan does not count —
    /// and its image objects must cover <see cref="ScanCoverage"/> of it. Where several images
    /// tile or overlap the page, the highest of their resolutions wins, so a page banded into
    /// strips of different quality is rendered for its best strip rather than its worst.
    /// </remarks>
    private static double? Inspect(nint page)
    {
        float width = FPDF_GetPageWidthF(page);
        float height = FPDF_GetPageHeightF(page);
        if (width <= 0 || height <= 0)
        {
            return null;
        }

        int count = FPDFPage_CountObjects(page);
        double covered = 0;
        double dpi = 0;

        for (int index = 0; index < count; index++)
        {
            nint obj = FPDFPage_GetObject(page, index);
            if (obj == 0)
            {
                continue;
            }

            int type = FPDFPageObj_GetType(obj);

            if (type == Text)
            {
                if (FPDFTextObj_GetTextRenderMode(obj) != Invisible)
                {
                    return null;
                }

                continue;
            }

            if (type != Image)
            {
                // A path or a shading is drawn at the device resolution, so the page has detail
                // above whatever its images carry.
                return null;
            }

            if (FPDFImageObj_GetImageMetadata(obj, page, out ImageMetadata metadata) != 0)
            {
                dpi = Math.Max(dpi, Math.Max(metadata.HorizontalDpi, metadata.VerticalDpi));
            }

            if (FPDFPageObj_GetBounds(obj, out float left, out float bottom, out float right, out float top) != 0)
            {
                covered += Math.Max(0f, right - left) * Math.Max(0f, top - bottom);
            }
        }

        if (dpi <= 0 || covered / (width * (double)height) < ScanCoverage)
        {
            return null;
        }

        return dpi;
    }

    /// <summary>Mirrors <c>FPDF_IMAGEOBJ_METADATA</c> in <c>fpdf_edit.h</c>.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct ImageMetadata
    {
        public uint Width;
        public uint Height;
        public float HorizontalDpi;
        public float VerticalDpi;
        public uint BitsPerPixel;
        public int Colorspace;
        public int MarkedContentId;
    }

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    private static partial nint FPDF_LoadMemDocument(nint data, int size, string? password);

    [LibraryImport(Library)]
    private static partial void FPDF_CloseDocument(nint document);

    [LibraryImport(Library)]
    private static partial nint FPDF_LoadPage(nint document, int index);

    [LibraryImport(Library)]
    private static partial void FPDF_ClosePage(nint page);

    [LibraryImport(Library)]
    private static partial float FPDF_GetPageWidthF(nint page);

    [LibraryImport(Library)]
    private static partial float FPDF_GetPageHeightF(nint page);

    [LibraryImport(Library)]
    private static partial int FPDFPage_CountObjects(nint page);

    [LibraryImport(Library)]
    private static partial nint FPDFPage_GetObject(nint page, int index);

    [LibraryImport(Library)]
    private static partial int FPDFPageObj_GetType(nint obj);

    [LibraryImport(Library)]
    private static partial int FPDFTextObj_GetTextRenderMode(nint obj);

    [LibraryImport(Library)]
    private static partial int FPDFImageObj_GetImageMetadata(nint obj, nint page, out ImageMetadata metadata);

    [LibraryImport(Library)]
    private static partial int FPDFPageObj_GetBounds(
        nint obj, out float left, out float bottom, out float right, out float top);
}
