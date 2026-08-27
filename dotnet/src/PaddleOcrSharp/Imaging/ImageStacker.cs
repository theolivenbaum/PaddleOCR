namespace PaddleOcrSharp.Imaging;

/// <summary>How a narrower image is placed inside a wider canvas when stacking.</summary>
public enum StackAlignment
{
    /// <summary>Flush left.</summary>
    Left,

    /// <summary>Horizontally centred.</summary>
    Center,

    /// <summary>Flush right.</summary>
    Right,
}

/// <summary>
/// Stacks crops vertically onto one canvas, which is how the pipeline turns a paragraph split
/// across several detections into a single recognition.
/// </summary>
/// <remarks>
/// Port of <c>merge_images</c>. The alignment list has one entry per join, and each join is
/// applied to everything placed so far: aligning right, then left, shifts the earlier images
/// again. Reproducing that iterative shift is what keeps the offsets identical to upstream's.
/// </remarks>
public static class ImageStacker
{
    /// <summary>Width and height a vertical stack of <paramref name="images"/> would need.</summary>
    public static (int Width, int Height) MeasureStack(IReadOnlyList<RgbImage> images) =>
        (images.Max(image => image.Width), images.Sum(image => image.Height));

    /// <summary>
    /// Stacks <paramref name="images"/> vertically onto a white canvas.
    /// </summary>
    /// <param name="images">Images to stack, top first.</param>
    /// <param name="alignments">One alignment per join; must be <c>images.Count - 1</c> long.</param>
    public static RgbImage Stack(IReadOnlyList<RgbImage> images, IReadOnlyList<StackAlignment> alignments)
    {
        ArgumentOutOfRangeException.ThrowIfZero(images.Count);

        if (images.Count == 1)
        {
            return images[0].Clone();
        }

        if (alignments.Count != images.Count - 1)
        {
            throw new ArgumentException(
                $"Need {images.Count - 1} alignments for {images.Count} images, got {alignments.Count}.",
                nameof(alignments));
        }

        int[] offsets = new int[images.Count];
        int mergedWidth = images[0].Width;

        for (int i = 1; i < images.Count; i++)
        {
            int stepWidth = Math.Max(mergedWidth, images[i].Width);
            (int shift, int placement) = alignments[i - 1] switch
            {
                StackAlignment.Center => ((stepWidth - mergedWidth) / 2, (stepWidth - images[i].Width) / 2),
                StackAlignment.Right => (stepWidth - mergedWidth, stepWidth - images[i].Width),
                _ => (0, 0),
            };

            for (int k = 0; k < i; k++)
            {
                offsets[k] += shift;
            }

            offsets[i] = placement;
            mergedWidth = stepWidth;
        }

        int totalHeight = images.Sum(image => image.Height);
        RgbImage canvas = RgbImage.Rent(mergedWidth, totalHeight);
        canvas.Pixels.Fill(255);

        int y = 0;
        for (int i = 0; i < images.Count; i++)
        {
            RgbImage image = images[i];
            for (int row = 0; row < image.Height; row++)
            {
                image.Row(row).CopyTo(canvas.Row(y + row)[(offsets[i] * 3)..]);
            }

            y += image.Height;
        }

        return canvas;
    }
}
