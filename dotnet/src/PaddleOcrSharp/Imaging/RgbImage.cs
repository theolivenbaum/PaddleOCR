using System.Buffers;

namespace PaddleOcrSharp.Imaging;

/// <summary>
/// An 8-bit RGB image stored interleaved (height × width × 3), the layout every stage of the
/// pipeline exchanges before the final conversion to float tensors.
/// </summary>
public sealed class RgbImage : IDisposable
{
    private byte[]? _pixels;
    private readonly bool _pooled;

    private RgbImage(byte[] pixels, int width, int height, bool pooled)
    {
        _pixels = pixels;
        _pooled = pooled;
        Width = width;
        Height = height;
    }

    /// <summary>Width in pixels.</summary>
    public int Width { get; }

    /// <summary>Height in pixels.</summary>
    public int Height { get; }

    /// <summary>Bytes per row.</summary>
    public int Stride => Width * 3;

    /// <summary>Total number of bytes.</summary>
    public int Length => Width * Height * 3;

    /// <summary>The interleaved RGB bytes.</summary>
    public Span<byte> Pixels => (_pixels ?? throw new ObjectDisposedException(nameof(RgbImage)))
        .AsSpan(0, Length);

    /// <summary>Row <paramref name="y"/> of the image.</summary>
    public Span<byte> Row(int y) => Pixels.Slice(y * Stride, Stride);

    /// <summary>Allocates an uninitialised image backed by a pooled buffer.</summary>
    public static RgbImage Rent(int width, int height)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        return new RgbImage(ArrayPool<byte>.Shared.Rent(width * height * 3), width, height, pooled: true);
    }

    /// <summary>Wraps an existing interleaved RGB buffer without copying.</summary>
    public static RgbImage Wrap(byte[] pixels, int width, int height)
    {
        if (pixels.Length < width * height * 3)
        {
            throw new ArgumentException(
                $"Buffer of {pixels.Length} bytes is too small for a {width}×{height} RGB image.",
                nameof(pixels));
        }

        return new RgbImage(pixels, width, height, pooled: false);
    }

    /// <summary>Copies <paramref name="pixels"/> into a new image.</summary>
    public static RgbImage From(ReadOnlySpan<byte> pixels, int width, int height)
    {
        byte[] copy = new byte[width * height * 3];
        pixels[..copy.Length].CopyTo(copy);
        return Wrap(copy, width, height);
    }

    /// <summary>
    /// Extracts the axis-aligned rectangle <c>[x0, x1) × [y0, y1)</c>, clamped to the image.
    /// </summary>
    public RgbImage Crop(int x0, int y0, int x1, int y1)
    {
        x0 = Math.Clamp(x0, 0, Width);
        x1 = Math.Clamp(x1, 0, Width);
        y0 = Math.Clamp(y0, 0, Height);
        y1 = Math.Clamp(y1, 0, Height);

        int width = Math.Max(1, x1 - x0);
        int height = Math.Max(1, y1 - y0);

        RgbImage result = Rent(width, height);
        for (int y = 0; y < height; y++)
        {
            Pixels.Slice(((y0 + y) * Stride) + (x0 * 3), width * 3).CopyTo(result.Row(y));
        }

        return result;
    }

    /// <summary>Deep copy into a non-pooled image.</summary>
    public RgbImage Clone() => From(Pixels, Width, Height);

    /// <inheritdoc />
    public void Dispose()
    {
        if (_pooled && _pixels is not null)
        {
            ArrayPool<byte>.Shared.Return(_pixels);
        }

        _pixels = null;
    }
}
