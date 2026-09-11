using System.IO.Compression;
using System.Text;
using PaddleOcrSharp.Pdf;

namespace PaddleOcrSharp.Tests.Pdf;

/// <summary>
/// Pins when a page is treated as a scan whose resolution caps the render.
/// </summary>
/// <remarks>
/// The documents are built here rather than loaded, so the test says exactly which property of a
/// page it is asserting on and needs no corpus. Each is one inch square, so an <c>n × n</c> image
/// filling it lands at <c>n</c> dpi.
/// </remarks>
public class PdfSourceDpiTests
{
    [Fact]
    public void APageThatIsNothingButAnImageReportsThatImagesResolution()
    {
        byte[] pdf = OnePage(pixels: 200, coverage: 1.0, withVisibleText: false);

        Assert.Equal(200, PdfRasterizer.GetSourceDpi(pdf, 0) ?? 0, 0.5);
    }

    [Fact]
    public void VisibleTextOnThePageMeansItIsNotAScan()
    {
        byte[] pdf = OnePage(pixels: 200, coverage: 1.0, withVisibleText: true);

        Assert.Null(PdfRasterizer.GetSourceDpi(pdf, 0));
    }

    [Fact]
    public void AnIllustrationThatDoesNotCoverThePageIsNotAScan()
    {
        // The same image, drawn small: a figure in a typeset page, not a photograph of one.
        byte[] pdf = OnePage(pixels: 200, coverage: 0.3, withVisibleText: false);

        Assert.Null(PdfRasterizer.GetSourceDpi(pdf, 0));
    }

    [Fact]
    public void APageOutsideTheDocumentHasNoResolution()
    {
        byte[] pdf = OnePage(pixels: 200, coverage: 1.0, withVisibleText: false);

        Assert.Null(PdfRasterizer.GetSourceDpi(pdf, 7));
    }

    [Fact]
    public void SomethingThatIsNotAPdfIsUnknownRatherThanAnError()
    {
        Assert.Null(PdfRasterizer.GetSourceDpi([1, 2, 3, 4], 0));
    }

    /// <summary>
    /// A one-page, one-inch-square PDF holding one image, optionally with a line of real text.
    /// </summary>
    /// <param name="pixels">Width and height of the image in pixels, so also its dpi at full coverage.</param>
    /// <param name="coverage">Fraction of the page side the image is drawn across.</param>
    /// <param name="withVisibleText">Whether to draw a line of text over it.</param>
    private static byte[] OnePage(int pixels, double coverage, bool withVisibleText)
    {
        byte[] image = Deflate(Enumerable.Repeat((byte)255, pixels * pixels * 3).ToArray());

        double side = 72 * coverage;
        var content = new StringBuilder($"q {side} 0 0 {side} 0 0 cm /Im0 Do Q");
        if (withVisibleText)
        {
            content.Append("\nBT /F1 12 Tf 5 5 Td (hi) Tj ET");
        }

        byte[] stream = Encoding.ASCII.GetBytes(content.ToString());

        List<byte[]> objects =
        [
            Ascii("<< /Type /Catalog /Pages 2 0 R >>"),
            Ascii("<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
            Ascii(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 72 72] /Resources << "
                + "/XObject << /Im0 4 0 R >> /Font << /F1 6 0 R >> >> /Contents 5 0 R >>"),
            Concat(
                Ascii(
                    $"<< /Type /XObject /Subtype /Image /Width {pixels} /Height {pixels} "
                    + $"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
                    + $"/Length {image.Length} >>\nstream\n"),
                image,
                Ascii("\nendstream")),
            Concat(Ascii($"<< /Length {stream.Length} >>\nstream\n"), stream, Ascii("\nendstream")),
            Ascii("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
        ];

        var document = new MemoryStream();
        document.Write(Ascii("%PDF-1.7\n"));

        var offsets = new List<long>();
        for (int index = 0; index < objects.Count; index++)
        {
            offsets.Add(document.Length);
            document.Write(Ascii($"{index + 1} 0 obj\n"));
            document.Write(objects[index]);
            document.Write(Ascii("\nendobj\n"));
        }

        long start = document.Length;
        document.Write(Ascii($"xref\n0 {objects.Count + 1}\n0000000000 65535 f \n"));
        foreach (long offset in offsets)
        {
            document.Write(Ascii($"{offset:D10} 00000 n \n"));
        }

        document.Write(Ascii(
            $"trailer\n<< /Size {objects.Count + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n"));
        return document.ToArray();
    }

    private static byte[] Ascii(string text) => Encoding.ASCII.GetBytes(text);

    private static byte[] Concat(params byte[][] parts) => [.. parts.SelectMany(part => part)];

    private static byte[] Deflate(byte[] data)
    {
        var compressed = new MemoryStream();
        using (var deflate = new ZLibStream(compressed, CompressionLevel.Fastest, leaveOpen: true))
        {
            deflate.Write(data);
        }

        return compressed.ToArray();
    }
}
