using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;

namespace PaddleOcrSharp.Download;

/// <summary>
/// Fetches model files from a Hugging Face-compatible endpoint into a local cache.
/// </summary>
/// <remarks>
/// <para>
/// Transfers resume from a <c>.part</c> file, are verified against the size and ETag the server
/// reports, and are published by an atomic rename so a killed process never leaves a truncated
/// file that later looks complete. A cross-process lock file makes concurrent CLI invocations
/// safe.
/// </para>
/// <para>
/// The endpoint can be redirected with <c>HF_ENDPOINT</c> for mirrors, and the cache root with
/// <c>PADDLEOCR_SHARP_CACHE</c>.
/// </para>
/// </remarks>
public sealed class ModelDownloader : IDisposable
{
    private const string DefaultEndpoint = "https://huggingface.co";

    private readonly HttpClient _client;
    private readonly bool _ownsClient;

    /// <summary>Creates a downloader with its own <see cref="HttpClient"/>.</summary>
    /// <param name="cacheRoot">Cache directory; defaults to the user cache.</param>
    /// <param name="endpoint">Base URL; defaults to <c>HF_ENDPOINT</c> or huggingface.co.</param>
    /// <param name="token">Optional bearer token for gated repositories.</param>
    public ModelDownloader(string? cacheRoot = null, string? endpoint = null, string? token = null)
        : this(CreateClient(token), ownsClient: true, cacheRoot, endpoint)
    {
    }

    /// <summary>Creates a downloader over a caller-owned <see cref="HttpClient"/>.</summary>
    public ModelDownloader(HttpClient client, bool ownsClient, string? cacheRoot = null, string? endpoint = null)
    {
        _client = client;
        _ownsClient = ownsClient;
        CacheRoot = cacheRoot ?? DefaultCacheRoot();
        Endpoint = (endpoint
            ?? Environment.GetEnvironmentVariable("HF_ENDPOINT")
            ?? DefaultEndpoint).TrimEnd('/');
    }

    /// <summary>Root of the local model cache.</summary>
    public string CacheRoot { get; }

    /// <summary>Base URL files are fetched from.</summary>
    public string Endpoint { get; }

    /// <summary>Directory a model resolves to, whether or not it has been downloaded.</summary>
    public string DirectoryFor(ModelDescriptor model) =>
        Path.Combine(CacheRoot, model.Name, Sanitise(model.Revision));

    /// <summary>Whether every required file of <paramref name="model"/> is already present.</summary>
    public bool IsComplete(ModelDescriptor model)
    {
        string directory = DirectoryFor(model);
        return model.Files
            .Where(file => file.Required)
            .All(file => File.Exists(Path.Combine(directory, file.Path)));
    }

    /// <summary>
    /// Ensures every file of <paramref name="model"/> is present and returns its directory.
    /// </summary>
    public async Task<string> EnsureAsync(
        ModelDescriptor model,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        string directory = DirectoryFor(model);
        Directory.CreateDirectory(directory);

        using FileStream guard = await AcquireLockAsync(directory, cancellationToken).ConfigureAwait(false);

        foreach (ModelFile file in model.Files)
        {
            try
            {
                await FetchAsync(model, file, directory, progress, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception exception) when (!file.Required && exception is not OperationCanceledException)
            {
                // Optional files (chat templates, generation configs) are absent from some
                // revisions; the model still loads without them.
            }
        }

        return directory;
    }

    private async Task FetchAsync(
        ModelDescriptor model,
        ModelFile file,
        string directory,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        string target = Path.Combine(directory, file.Path);
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);

        string url = $"{Endpoint}/{model.Repository}/resolve/{model.Revision}/{file.Path}";
        RemoteFileInfo? remote = await HeadAsync(url, cancellationToken).ConfigureAwait(false);

        if (File.Exists(target))
        {
            var existing = new FileInfo(target);
            bool sizeMatches = remote?.Length is null || remote.Value.Length == existing.Length;
            if (sizeMatches)
            {
                progress?.Report(new DownloadProgress(
                    model.Name, file.Path, existing.Length, existing.Length, Cached: true));
                return;
            }

            File.Delete(target);
        }

        if (remote is null)
        {
            throw new HttpRequestException($"'{url}' is not reachable and no cached copy exists.");
        }

        string partial = target + ".part";
        long offset = File.Exists(partial) ? new FileInfo(partial).Length : 0;
        if (offset > 0 && remote.Value.Length is { } total && (offset > total || !remote.Value.SupportsRange))
        {
            File.Delete(partial);
            offset = 0;
        }

        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        if (offset > 0)
        {
            request.Headers.Range = new RangeHeaderValue(offset, null);
        }

        using HttpResponseMessage response = await _client
            .SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
            .ConfigureAwait(false);

        if (offset > 0 && response.StatusCode != HttpStatusCode.PartialContent)
        {
            // The server ignored the range; start over rather than concatenating a full body
            // onto the partial file.
            offset = 0;
        }

        response.EnsureSuccessStatusCode();

        await using (FileStream output = new(
            partial,
            offset > 0 ? FileMode.Append : FileMode.Create,
            FileAccess.Write,
            FileShare.None,
            bufferSize: 1 << 20,
            useAsync: true))
        {
            await using Stream input = await response.Content
                .ReadAsStreamAsync(cancellationToken)
                .ConfigureAwait(false);

            byte[] buffer = new byte[1 << 20];
            long received = offset;
            int read;
            while ((read = await input.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
            {
                await output.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                received += read;
                progress?.Report(new DownloadProgress(
                    model.Name, file.Path, received, remote.Value.Length, Cached: false));
            }
        }

        if (remote.Value.Length is { } expected && new FileInfo(partial).Length != expected)
        {
            long actual = new FileInfo(partial).Length;
            File.Delete(partial);
            throw new IOException($"'{file.Path}' downloaded {actual} bytes but {expected} were expected.");
        }

        if (remote.Value.Sha256 is { } digest && !await MatchesAsync(partial, digest, cancellationToken).ConfigureAwait(false))
        {
            File.Delete(partial);
            throw new IOException($"'{file.Path}' failed its SHA-256 check.");
        }

        File.Move(partial, target, overwrite: true);
    }

    private async Task<RemoteFileInfo?> HeadAsync(string url, CancellationToken cancellationToken)
    {
        try
        {
            using var request = new HttpRequestMessage(HttpMethod.Head, url);
            using HttpResponseMessage response = await _client
                .SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
                .ConfigureAwait(false);

            if (!response.IsSuccessStatusCode)
            {
                return null;
            }

            // Hugging Face returns the LFS object's SHA-256 in X-Linked-Etag for large files and
            // a git blob hash in ETag for small ones; only the former is a content digest.
            string? sha256 = null;
            if (response.Headers.TryGetValues("X-Linked-Etag", out IEnumerable<string>? linked))
            {
                string value = linked.First().Trim('"');
                if (value.Length == 64 && value.All(Uri.IsHexDigit))
                {
                    sha256 = value;
                }
            }

            return new RemoteFileInfo(
                response.Content.Headers.ContentLength,
                sha256,
                response.Headers.AcceptRanges.Contains("bytes"));
        }
        catch (HttpRequestException)
        {
            return null;
        }
        catch (TaskCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            return null;
        }
    }

    private static async Task<bool> MatchesAsync(string path, string expected, CancellationToken cancellationToken)
    {
        await using FileStream stream = File.OpenRead(path);
        byte[] digest = await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false);
        return Convert.ToHexStringLower(digest) == expected;
    }

    private static async Task<FileStream> AcquireLockAsync(string directory, CancellationToken cancellationToken)
    {
        string path = Path.Combine(directory, ".download.lock");
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                return new FileStream(
                    path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, 1, FileOptions.DeleteOnClose);
            }
            catch (IOException)
            {
                await Task.Delay(200, cancellationToken).ConfigureAwait(false);
            }
        }
    }

    private static HttpClient CreateClient(string? token)
    {
        var handler = new SocketsHttpHandler
        {
            AutomaticDecompression = DecompressionMethods.All,
            PooledConnectionLifetime = TimeSpan.FromMinutes(5),
        };

        var client = new HttpClient(handler)
        {
            // Large weight files stream for minutes; the per-read cancellation token is what
            // actually bounds a stalled transfer.
            Timeout = Timeout.InfiniteTimeSpan,
        };

        client.DefaultRequestHeaders.UserAgent.ParseAdd("PaddleOcrSharp/0.1");

        string? bearer = token ?? Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrEmpty(bearer))
        {
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", bearer);
        }

        return client;
    }

    private static string DefaultCacheRoot()
    {
        string? configured = Environment.GetEnvironmentVariable("PADDLEOCR_SHARP_CACHE");
        if (!string.IsNullOrEmpty(configured))
        {
            return configured;
        }

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string cache = Environment.GetEnvironmentVariable("XDG_CACHE_HOME")
            ?? Path.Combine(home, ".cache");
        return Path.Combine(cache, "paddleocr-sharp");
    }

    private static string Sanitise(string value) =>
        string.Concat(value.Select(c => Path.GetInvalidFileNameChars().Contains(c) ? '_' : c));

    /// <inheritdoc />
    public void Dispose()
    {
        if (_ownsClient)
        {
            _client.Dispose();
        }
    }

    private readonly record struct RemoteFileInfo(long? Length, string? Sha256, bool SupportsRange);
}
