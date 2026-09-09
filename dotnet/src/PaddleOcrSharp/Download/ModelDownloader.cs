using System.Net;
using System.Net.Http.Headers;

namespace PaddleOcrSharp.Download;

/// <summary>
/// Fetches model files from a static HTTP mirror into a local cache.
/// </summary>
/// <remarks>
/// <para>
/// A file's URL is <c>{endpoint}/{remote path}/{file path}</c> — the mirror is a plain directory
/// tree, so <c>https://models.curiosity.ai/paddleocr-vl/config.json</c> is the whole address of
/// that file. Nothing here is specific to a model host's API.
/// </para>
/// <para>
/// Transfers resume from a <c>.part</c> file, are verified against the length the server reports,
/// and are published by an atomic rename so a killed process never leaves a truncated file that
/// later looks complete. A cross-process lock file makes concurrent CLI invocations safe.
/// </para>
/// <para>
/// The endpoint can be redirected with <c>PADDLEOCR_SHARP_MODELS_URL</c> for a private or local
/// mirror, and the cache root with <c>PADDLEOCR_SHARP_CACHE</c>.
/// </para>
/// </remarks>
public sealed class ModelDownloader : IDisposable
{
    private const string DefaultEndpoint = "https://models.curiosity.ai";
    private const string EndpointVariable = "PADDLEOCR_SHARP_MODELS_URL";
    private const string TokenVariable = "PADDLEOCR_SHARP_MODELS_TOKEN";

    private readonly HttpClient _client;
    private readonly bool _ownsClient;

    // Set once a mirror answers 405 to a HEAD; it will answer 405 to every other one too.
    private volatile bool _headUnsupported;

    /// <summary>Creates a downloader with its own <see cref="HttpClient"/>.</summary>
    /// <param name="cacheRoot">Cache directory; defaults to the user cache.</param>
    /// <param name="endpoint">
    /// Base URL; defaults to <c>PADDLEOCR_SHARP_MODELS_URL</c> or models.curiosity.ai.
    /// </param>
    /// <param name="token">
    /// Optional bearer token, for a mirror that is not public. The default endpoint needs none.
    /// </param>
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
            ?? Environment.GetEnvironmentVariable(EndpointVariable)
            ?? DefaultEndpoint).TrimEnd('/');
    }

    /// <summary>Root of the local model cache.</summary>
    public string CacheRoot { get; }

    /// <summary>Base URL files are fetched from.</summary>
    public string Endpoint { get; }

    /// <summary>Directory a model resolves to, whether or not it has been downloaded.</summary>
    public string DirectoryFor(ModelDescriptor model) => Path.Combine(CacheRoot, model.Name);

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

        string url = $"{Endpoint}/{model.RemotePath}/{file.Path}";
        RemoteFileInfo? remote = await ProbeAsync(url, cancellationToken).ConfigureAwait(false);

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
        string validatorFile = partial + ".etag";
        string? validator = remote.Value.Validator?.Tag;

        long offset = File.Exists(partial) ? new FileInfo(partial).Length : 0;
        if (offset > 0 && remote.Value.Length is { } total && (offset > total || !remote.Value.SupportsRange))
        {
            offset = 0;
        }

        if (offset > 0 && !ResumeIsSafe(validatorFile, validator))
        {
            offset = 0;
        }

        if (offset == 0)
        {
            File.Delete(partial);
            File.Delete(validatorFile);
        }

        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        if (offset > 0)
        {
            request.Headers.Range = new RangeHeaderValue(offset, null);

            // This validator is the one just probed, so a server that honours the header will
            // almost always find it current and answer 206. What it buys is the narrow race
            // where the file changes between the probe and this request; a partial file left by
            // an earlier run is the recorded validator's job, not this header's.
            if (remote.Value.Validator is { } tag)
            {
                request.Headers.IfRange = new RangeConditionHeaderValue(tag);
            }
        }

        if (validator is not null)
        {
            // Written before the transfer, so an interrupted one leaves a partial file that can
            // still be matched against the mirror on the next run.
            await File.WriteAllTextAsync(validatorFile, validator, cancellationToken).ConfigureAwait(false);
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

        File.Move(partial, target, overwrite: true);
        File.Delete(validatorFile);
    }

    /// <summary>
    /// Whether a <c>.part</c> file may be appended to rather than started over.
    /// </summary>
    /// <remarks>
    /// Appending to a partial copy of a file that has since changed splices two files into one of
    /// exactly the length the server reports, so neither the length check nor a resumed transfer
    /// can detect it. <c>If-Range</c> does not help: the validator it would carry is the one just
    /// probed, which is current by construction, and models.curiosity.ai ignores the header
    /// anyway. So the validator in force when the partial file was started is recorded beside it
    /// and compared here. A mirror that publishes no usable validator cannot be checked at all,
    /// and its partial file is discarded rather than trusted.
    /// </remarks>
    private static bool ResumeIsSafe(string validatorFile, string? validator)
    {
        if (validator is null || !File.Exists(validatorFile))
        {
            return false;
        }

        return File.ReadAllText(validatorFile).Trim() == validator;
    }

    /// <summary>
    /// Reads a file's length, validator and range support without transferring it.
    /// </summary>
    /// <remarks>
    /// HEAD is the cheap way to ask, but a mirror is entitled to refuse it — the R2 bucket behind
    /// models.curiosity.ai answers <c>405</c> with <c>Allow: PUT, GET, DELETE</c>. A one-byte
    /// ranged GET is the fallback: its <c>Content-Range</c> carries the full length, and a
    /// <c>206</c> proves range support outright rather than promising it in a header.
    /// </remarks>
    private async Task<RemoteFileInfo?> ProbeAsync(string url, CancellationToken cancellationToken)
    {
        if (!_headUnsupported)
        {
            MetadataProbe head = await RequestMetadataAsync(url, HttpMethod.Head, cancellationToken)
                .ConfigureAwait(false);
            if (head.Info is { } info)
            {
                return info;
            }

            _headUnsupported = head.MethodRejected;
        }

        MetadataProbe ranged = await RequestMetadataAsync(url, HttpMethod.Get, cancellationToken)
            .ConfigureAwait(false);
        return ranged.Info;
    }

    private async Task<MetadataProbe> RequestMetadataAsync(
        string url,
        HttpMethod method,
        CancellationToken cancellationToken)
    {
        try
        {
            using var request = new HttpRequestMessage(method, url);
            if (method == HttpMethod.Get)
            {
                request.Headers.Range = new RangeHeaderValue(0, 0);
            }

            using HttpResponseMessage response = await _client
                .SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
                .ConfigureAwait(false);

            if (!response.IsSuccessStatusCode)
            {
                return new MetadataProbe(
                    null,
                    MethodRejected: response.StatusCode == HttpStatusCode.MethodNotAllowed);
            }

            bool partial = response.StatusCode == HttpStatusCode.PartialContent;

            // A static mirror publishes no content digest: an S3-compatible ETag is an MD5 only
            // for a single-part upload, and a CDN in front of it may rewrite the header. So the
            // ETag is kept as what it reliably is — an opaque validator — and length is what the
            // finished file is checked against. A weak tag is no use as a validator and is dropped.
            EntityTagHeaderValue? etag = response.Headers.ETag is { IsWeak: false } strong ? strong : null;

            return new MetadataProbe(
                new RemoteFileInfo(
                    partial ? response.Content.Headers.ContentRange?.Length : response.Content.Headers.ContentLength,
                    etag,
                    partial || response.Headers.AcceptRanges.Contains("bytes")),
                MethodRejected: false);
        }
        catch (HttpRequestException)
        {
            return default;
        }
        catch (TaskCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            return default;
        }
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

        string? bearer = token ?? Environment.GetEnvironmentVariable(TokenVariable);
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

    /// <inheritdoc />
    public void Dispose()
    {
        if (_ownsClient)
        {
            _client.Dispose();
        }
    }

    private readonly record struct RemoteFileInfo(
        long? Length,
        EntityTagHeaderValue? Validator,
        bool SupportsRange);

    private readonly record struct MetadataProbe(RemoteFileInfo? Info, bool MethodRejected);
}
