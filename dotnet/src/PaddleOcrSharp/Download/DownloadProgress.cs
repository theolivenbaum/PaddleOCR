namespace PaddleOcrSharp.Download;

/// <summary>Progress of one file in a model download.</summary>
/// <param name="Model">Model being fetched.</param>
/// <param name="File">Repository-relative file path.</param>
/// <param name="BytesReceived">Bytes written so far.</param>
/// <param name="TotalBytes">Total size, or <see langword="null"/> when the server did not report one.</param>
/// <param name="Cached">Whether the file was already present and no transfer happened.</param>
public readonly record struct DownloadProgress(
    string Model,
    string File,
    long BytesReceived,
    long? TotalBytes,
    bool Cached)
{
    /// <summary>Fraction complete in <c>[0, 1]</c>, or <see langword="null"/> when the size is unknown.</summary>
    public double? Fraction => TotalBytes is > 0 ? (double)BytesReceived / TotalBytes.Value : null;
}
