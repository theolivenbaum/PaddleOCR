using System.Text.RegularExpressions;
using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models.Layout;
using SkiaSharp;

namespace PaddleOcrSharp.Pipeline;

/// <summary>A picture the page contains, as <c>gather_imgs</c> collects them.</summary>
/// <param name="Path">File name the figure is written to, which also identifies it.</param>
/// <param name="Box">The figure's region, in page coordinates.</param>
public readonly record struct DocumentFigure(string Path, LayoutBox Box);

/// <summary>A figure that sits inside a table and was replaced by a placeholder token.</summary>
/// <param name="Token">The placeholder written into the table image, e.g. <c>[F23]</c>.</param>
/// <param name="Path">File name of the figure the token stands for.</param>
public readonly record struct TokenizedFigure(string Token, string Path);

/// <summary>
/// Replaces figures inside a table with short placeholder tokens before recognition, then puts
/// image references back into the recognised HTML.
/// </summary>
/// <remarks>
/// <para>
/// Port of <c>tokenize_figure_of_table</c> / <c>untokenize_figure_of_table</c>. A figure fully
/// contained in a table's box is painted over with a white rectangle carrying a token such as
/// <c>[F23]</c>, so the model emits that token in the cell instead of trying to read the picture.
/// Figures shorter than 25 px on a side are painted over without a token — too small to be read
/// back reliably — and are still removed from the page's block list.
/// </para>
/// <para>
/// The token digits avoid <c>0</c>, <c>1</c> and <c>9</c>, which the model confuses with letters.
/// Upstream shuffles the numbering with a seeded RNG; the mapping only has to be a stable
/// bijection, so this assigns them in order rather than reproducing Python's Mersenne Twister.
/// </para>
/// <para>
/// The token is drawn with SkiaSharp rather than OpenCV's Hershey font. What matters is that the
/// model can read it, not which glyphs draw it.
/// </para>
/// </remarks>
public static partial class TableFigureTokenizer
{
    /// <summary>Figures with a side shorter than this are blanked but not tokenised.</summary>
    public const int MinimumTokenizableSide = 25;

    [GeneratedRegex(@"\[F(\d+)\]")]
    private static partial Regex TokenPattern();

    /// <summary>
    /// Paints placeholder tokens over the figures contained in <paramref name="table"/>.
    /// </summary>
    /// <param name="tableCrop">The table's crop; a modified copy is returned.</param>
    /// <param name="table">The table's region, in page coordinates.</param>
    /// <param name="figures">Every picture on the page.</param>
    /// <returns>
    /// The painted crop, the figures that were given a token, and the paths of every figure the
    /// table swallowed — including the ones too small to tokenise, which are covered over all the
    /// same and so must not also appear as blocks of their own.
    /// </returns>
    public static (RgbImage Image, IReadOnlyList<TokenizedFigure> Tokens, IReadOnlyList<string> Absorbed)
        Tokenize(RgbImage tableCrop, LayoutBox table, IReadOnlyList<DocumentFigure> figures)
    {
        var contained = new List<int>();
        for (int i = 0; i < figures.Count; i++)
        {
            LayoutBox figure = figures[i].Box;
            if (figure.Left >= table.Left
                && figure.Top >= table.Top
                && figure.Right <= table.Right
                && figure.Bottom <= table.Bottom)
            {
                contained.Add(i);
            }
        }

        if (contained.Count == 0)
        {
            return (tableCrop.Clone(), [], []);
        }

        RgbImage painted = tableCrop.Clone();
        var tokens = new List<TokenizedFigure>(contained.Count);
        var absorbed = new List<string>(contained.Count);

        // Numbered over the whole figure list, so a figure's token does not depend on which
        // table happens to contain it.
        int[] numbers = TokenNumbers(figures.Count);

        foreach (int index in contained)
        {
            DocumentFigure entry = figures[index];
            LayoutBox figure = entry.Box;
            absorbed.Add(entry.Path);

            int left = (int)(figure.Left - table.Left);
            int top = (int)(figure.Top - table.Top);
            int right = (int)(figure.Right - table.Left);
            int bottom = (int)(figure.Bottom - table.Top);

            if (Math.Min(figure.Width, figure.Height) < MinimumTokenizableSide)
            {
                // Still covered so the model does not try to read it, but too small to carry a
                // legible token.
                Blank(painted, left, top, right, bottom);
                continue;
            }

            string token = $"[F{numbers[index]}]";
            Paint(painted, left, top, right, bottom, token);
            tokens.Add(new TokenizedFigure(token, entry.Path));
        }

        return (painted, tokens, absorbed);
    }

    /// <summary>
    /// Replaces the placeholder tokens in <paramref name="html"/> with image references.
    /// </summary>
    /// <param name="html">Recognised table HTML.</param>
    /// <param name="figures">Figures produced by <see cref="Tokenize"/>.</param>
    /// <param name="imageDirectory">Directory prefix written into the <c>src</c> attribute.</param>
    /// <param name="textOf">
    /// The recognised text of the figure at a path, if any, which follows its image. A figure the
    /// page no longer has a block for returns null, and its token is left as it was.
    /// </param>
    public static string Untokenize(
        string html,
        IReadOnlyList<TokenizedFigure> figures,
        string imageDirectory,
        Func<string, string?>? textOf = null)
    {
        if (figures.Count == 0)
        {
            return html;
        }

        Dictionary<string, TokenizedFigure> byToken = figures.ToDictionary(
            figure => figure.Token, figure => figure, StringComparer.Ordinal);

        return TokenPattern().Replace(html, match =>
        {
            if (!byToken.TryGetValue(match.Value, out TokenizedFigure figure))
            {
                return match.Value;
            }

            string? text = textOf?.Invoke(figure.Path);
            if (textOf is not null && text is null)
            {
                return match.Value;
            }

            string path = $"{imageDirectory}/{figure.Path}"
                .Replace("-\n", string.Empty, StringComparison.Ordinal)
                .Replace("\n", " ", StringComparison.Ordinal);

            // The doubled quote after `Image` is upstream's, and it is what ends up in the HTML
            // a consumer reads, so it is reproduced rather than tidied.
            string tag = $"<img src=\"{path}\" alt=\"Image\"\" />";

            return text is { Length: > 0 } ? $"{tag}\n\n{text}\n\n" : tag;
        });
    }

    /// <summary>
    /// The first <paramref name="count"/> non-negative integers whose digits avoid 0, 1 and 9.
    /// </summary>
    private static int[] TokenNumbers(int count)
    {
        var numbers = new List<int>(count);
        for (int candidate = 0; numbers.Count < count; candidate++)
        {
            if (!candidate.ToString().Any(digit => digit is '0' or '1' or '9'))
            {
                numbers.Add(candidate);
            }
        }

        return [.. numbers];
    }

    private static void Blank(RgbImage image, int left, int top, int right, int bottom)
    {
        left = Math.Clamp(left, 0, image.Width);
        right = Math.Clamp(right, 0, image.Width);
        top = Math.Clamp(top, 0, image.Height);
        bottom = Math.Clamp(bottom, 0, image.Height);

        for (int y = top; y < bottom; y++)
        {
            image.Row(y).Slice(left * 3, Math.Max(0, right - left) * 3).Fill(255);
        }
    }

    private static void Paint(RgbImage image, int left, int top, int right, int bottom, string token)
    {
        Blank(image, left, top, right, bottom);

        int width = Math.Max(1, right - left);
        int height = Math.Max(1, bottom - top);

        using var typeface = SKTypeface.FromFamilyName(
            "sans-serif", SKFontStyleWeight.Bold, SKFontStyleWidth.Normal, SKFontStyleSlant.Upright);

        // Grow the text until it fills 90% of the shorter side, as upstream's binary search does.
        float size = 8f;
        var font = new SKFont(typeface, size);
        try
        {
            float limit = Math.Min(width, height) * 0.9f;
            while (size < 400f)
            {
                font.Size = size + 1f;
                float measured = font.MeasureText(token);
                SKFontMetrics metrics = font.Metrics;
                if (measured > width * 0.9f || metrics.Descent - metrics.Ascent > limit)
                {
                    break;
                }

                size += 1f;
            }

            font.Size = size;

            using var surface = new SKBitmap(new SKImageInfo(width, height, SKColorType.Rgba8888, SKAlphaType.Opaque));
            using (var canvas = new SKCanvas(surface))
            {
                canvas.Clear(SKColors.White);
                using var paint = new SKPaint { Color = SKColors.Black, IsAntialias = true };

                float textWidth = font.MeasureText(token);
                SKFontMetrics metrics = font.Metrics;
                float x = (width - textWidth) / 2f;
                float y = ((height - (metrics.Descent - metrics.Ascent)) / 2f) - metrics.Ascent;
                canvas.DrawText(token, x, y, font, paint);
            }

            CopyInto(surface, image, left, top);
        }
        finally
        {
            font.Dispose();
        }
    }

    private static void CopyInto(SKBitmap source, RgbImage target, int left, int top)
    {
        ReadOnlySpan<byte> pixels = source.GetPixelSpan();

        for (int y = 0; y < source.Height; y++)
        {
            int targetY = top + y;
            if ((uint)targetY >= (uint)target.Height)
            {
                continue;
            }

            ReadOnlySpan<byte> sourceRow = pixels.Slice(y * source.RowBytes, source.Width * 4);
            Span<byte> targetRow = target.Row(targetY);

            for (int x = 0; x < source.Width; x++)
            {
                int targetX = left + x;
                if ((uint)targetX >= (uint)target.Width)
                {
                    continue;
                }

                targetRow[(targetX * 3) + 0] = sourceRow[(x * 4) + 0];
                targetRow[(targetX * 3) + 1] = sourceRow[(x * 4) + 1];
                targetRow[(targetX * 3) + 2] = sourceRow[(x * 4) + 2];
            }
        }
    }
}
