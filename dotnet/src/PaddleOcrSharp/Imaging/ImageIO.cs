using SkiaSharp;

namespace PaddleOcrSharp.Imaging;

/// <summary>
/// Image decoding and encoding, backed by SkiaSharp.
/// </summary>
/// <remarks>
/// Skia is used only for container handling (PNG/JPEG/WEBP/BMP decode, EXIF orientation) — never
/// for resampling. Resizing goes through <see cref="PilResize"/> so the pixels the model sees are
/// bit-identical to the Python pipeline's.
/// </remarks>
public static class ImageIO
{
    /// <summary>Decodes an image file into an interleaved RGB buffer.</summary>
    public static RgbImage Load(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Load(stream);
    }

    /// <summary>Decodes an image from a stream into an interleaved RGB buffer.</summary>
    public static RgbImage Load(Stream stream)
    {
        using SKCodec? codec = SKCodec.Create(new SKManagedStream(stream, disposeManagedStream: false));
        if (codec is null)
        {
            throw new InvalidDataException("Stream does not contain a decodable image.");
        }

        var info = new SKImageInfo(
            codec.Info.Width, codec.Info.Height, SKColorType.Rgba8888, SKAlphaType.Unpremul);

        using var bitmap = new SKBitmap(info);
        SKCodecResult result = codec.GetPixels(info, bitmap.GetPixels());
        if (result is not (SKCodecResult.Success or SKCodecResult.IncompleteInput))
        {
            throw new InvalidDataException($"Image could not be decoded ({result}).");
        }

        using SKBitmap oriented = ApplyOrientation(bitmap, codec.EncodedOrigin);
        return FromBitmap(oriented);
    }

    /// <summary>Decodes an image already held in memory.</summary>
    public static RgbImage Load(ReadOnlySpan<byte> bytes)
    {
        using var stream = new MemoryStream(bytes.ToArray(), writable: false);
        return Load(stream);
    }

    /// <summary>Writes an image as PNG.</summary>
    public static void SavePng(RgbImage image, string path)
    {
        using SKBitmap bitmap = ToBitmap(image);
        using SKData data = bitmap.Encode(SKEncodedImageFormat.Png, 100);
        using FileStream output = File.Create(path);
        data.SaveTo(output);
    }

    /// <summary>Encodes an image as PNG bytes, used when embedding figures in markdown output.</summary>
    public static byte[] EncodePng(RgbImage image)
    {
        using SKBitmap bitmap = ToBitmap(image);
        using SKData data = bitmap.Encode(SKEncodedImageFormat.Png, 100);
        return data.ToArray();
    }

    private static RgbImage FromBitmap(SKBitmap bitmap)
    {
        RgbImage image = RgbImage.Rent(bitmap.Width, bitmap.Height);
        ReadOnlySpan<byte> source = bitmap.GetPixelSpan();
        int sourceStride = bitmap.RowBytes;
        bool hasAlpha = bitmap.AlphaType != SKAlphaType.Opaque;

        for (int y = 0; y < bitmap.Height; y++)
        {
            ReadOnlySpan<byte> sourceRow = source.Slice(y * sourceStride, bitmap.Width * 4);
            Span<byte> targetRow = image.Row(y);

            for (int x = 0; x < bitmap.Width; x++)
            {
                byte r = sourceRow[(x * 4) + 0];
                byte g = sourceRow[(x * 4) + 1];
                byte b = sourceRow[(x * 4) + 2];
                byte a = sourceRow[(x * 4) + 3];

                if (hasAlpha && a != 255)
                {
                    // Upstream loads with PIL and calls `convert("RGB")`, which composites onto
                    // black; matching that keeps transparent PNGs identical.
                    r = (byte)((r * a) / 255);
                    g = (byte)((g * a) / 255);
                    b = (byte)((b * a) / 255);
                }

                targetRow[(x * 3) + 0] = r;
                targetRow[(x * 3) + 1] = g;
                targetRow[(x * 3) + 2] = b;
            }
        }

        return image;
    }

    private static SKBitmap ToBitmap(RgbImage image)
    {
        var bitmap = new SKBitmap(new SKImageInfo(image.Width, image.Height, SKColorType.Rgba8888, SKAlphaType.Opaque));
        Span<byte> target = GetWritablePixels(bitmap);

        for (int y = 0; y < image.Height; y++)
        {
            ReadOnlySpan<byte> sourceRow = image.Row(y);
            Span<byte> targetRow = target.Slice(y * bitmap.RowBytes, image.Width * 4);
            for (int x = 0; x < image.Width; x++)
            {
                targetRow[(x * 4) + 0] = sourceRow[(x * 3) + 0];
                targetRow[(x * 4) + 1] = sourceRow[(x * 3) + 1];
                targetRow[(x * 4) + 2] = sourceRow[(x * 3) + 2];
                targetRow[(x * 4) + 3] = 255;
            }
        }

        return bitmap;
    }

    private static unsafe Span<byte> GetWritablePixels(SKBitmap bitmap) =>
        new((void*)bitmap.GetPixels(), bitmap.RowBytes * bitmap.Height);

    private static SKBitmap ApplyOrientation(SKBitmap bitmap, SKEncodedOrigin origin)
    {
        if (origin is SKEncodedOrigin.Default or SKEncodedOrigin.TopLeft)
        {
            return bitmap.Copy();
        }

        bool swapAxes = origin is SKEncodedOrigin.LeftTop or SKEncodedOrigin.RightTop
            or SKEncodedOrigin.RightBottom or SKEncodedOrigin.LeftBottom;

        int width = swapAxes ? bitmap.Height : bitmap.Width;
        int height = swapAxes ? bitmap.Width : bitmap.Height;

        var rotated = new SKBitmap(new SKImageInfo(width, height, bitmap.ColorType, bitmap.AlphaType));
        using (var canvas = new SKCanvas(rotated))
        {
            switch (origin)
            {
                case SKEncodedOrigin.TopRight:
                    canvas.Scale(-1, 1, width / 2f, 0);
                    break;
                case SKEncodedOrigin.BottomRight:
                    canvas.RotateDegrees(180, width / 2f, height / 2f);
                    break;
                case SKEncodedOrigin.BottomLeft:
                    canvas.Scale(1, -1, 0, height / 2f);
                    break;
                case SKEncodedOrigin.LeftTop:
                    canvas.Translate(width, 0);
                    canvas.RotateDegrees(90);
                    canvas.Scale(1, -1, 0, bitmap.Height / 2f);
                    break;
                case SKEncodedOrigin.RightTop:
                    canvas.Translate(width, 0);
                    canvas.RotateDegrees(90);
                    break;
                case SKEncodedOrigin.RightBottom:
                    canvas.Translate(0, height);
                    canvas.RotateDegrees(270);
                    canvas.Scale(1, -1, 0, bitmap.Height / 2f);
                    break;
                case SKEncodedOrigin.LeftBottom:
                    canvas.Translate(0, height);
                    canvas.RotateDegrees(270);
                    break;
            }

            canvas.DrawBitmap(bitmap, 0, 0);
        }

        return rotated;
    }
}
