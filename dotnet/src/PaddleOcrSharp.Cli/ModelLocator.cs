using PaddleOcrSharp.Download;

namespace PaddleOcrSharp.Cli;

/// <summary>Resolves a checkpoint directory from the command line, environment or cache.</summary>
public static class ModelLocator
{
    /// <summary>
    /// Finds the layout detector, downloading it when <paramref name="allowDownload"/> is set.
    /// </summary>
    public static async Task<string> ResolveLayoutAsync(
        CommandLine command,
        bool allowDownload,
        CancellationToken cancellationToken = default)
    {
        string? explicitPath = command.Get("layout-dir")
            ?? Environment.GetEnvironmentVariable("PP_DOCLAYOUT_V3_DIR");

        if (!string.IsNullOrEmpty(explicitPath))
        {
            if (!File.Exists(Path.Combine(explicitPath, "inference.json")))
            {
                throw new DirectoryNotFoundException($"'{explicitPath}' does not contain inference.json.");
            }

            return explicitPath;
        }

        using var downloader = new ModelDownloader(command.Get("cache"), command.Get("endpoint"));
        ModelDescriptor model = ModelCatalog.PpDocLayoutV3;

        if (downloader.IsComplete(model))
        {
            return downloader.DirectoryFor(model);
        }

        if (!allowDownload)
        {
            throw new FileNotFoundException(
                $"Model '{model.Name}' is not in the cache at {downloader.DirectoryFor(model)}. " +
                "Run `paddleocr-sharp download` first, or pass --layout-dir.");
        }

        Console.Error.WriteLine($"Fetching {model.Name} into {downloader.CacheRoot} …");
        return await downloader.EnsureAsync(model, progress: null, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Finds the VL checkpoint, downloading it when <paramref name="allowDownload"/> is set.
    /// </summary>
    public static async Task<string> ResolveVLAsync(
        CommandLine command,
        bool allowDownload,
        CancellationToken cancellationToken = default)
    {
        string? explicitPath = command.Get("model-dir")
            ?? Environment.GetEnvironmentVariable("PADDLEOCR_VL_DIR");

        if (!string.IsNullOrEmpty(explicitPath))
        {
            if (!File.Exists(Path.Combine(explicitPath, "model.safetensors")))
            {
                throw new DirectoryNotFoundException(
                    $"'{explicitPath}' does not contain model.safetensors.");
            }

            return explicitPath;
        }

        using var downloader = new ModelDownloader(command.Get("cache"), command.Get("endpoint"));
        ModelDescriptor model = ModelCatalog.PaddleOcrVL16;

        if (downloader.IsComplete(model))
        {
            return downloader.DirectoryFor(model);
        }

        if (!allowDownload)
        {
            throw new FileNotFoundException(
                $"Model '{model.Name}' is not in the cache at {downloader.DirectoryFor(model)}. " +
                "Run `paddleocr-sharp download` first, or pass --model-dir.");
        }

        Console.Error.WriteLine($"Fetching {model.Name} into {downloader.CacheRoot} …");
        return await downloader.EnsureAsync(model, progress: null, cancellationToken).ConfigureAwait(false);
    }
}
