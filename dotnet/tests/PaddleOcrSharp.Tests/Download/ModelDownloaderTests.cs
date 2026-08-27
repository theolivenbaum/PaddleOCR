using System.Net;
using PaddleOcrSharp.Download;

namespace PaddleOcrSharp.Tests.Download;

/// <summary>
/// Exercises the downloader against a stub transport: cache hits, resume, verification and the
/// optional-file policy, without touching the network.
/// </summary>
public class ModelDownloaderTests : IDisposable
{
    private readonly string _cache = Path.Combine(
        Path.GetTempPath(), "paddleocr-sharp-tests", Guid.NewGuid().ToString("N"));

    [Fact]
    public async Task DownloadsEveryFileAndReportsProgress()
    {
        var handler = new StubHandler
        {
            Files =
            {
                ["a.json"] = "{\"a\":1}"u8.ToArray(),
                ["weights.bin"] = Enumerable.Range(0, 4096).Select(i => (byte)i).ToArray(),
            },
        };

        using var downloader = new ModelDownloader(new HttpClient(handler), ownsClient: true, _cache, "https://stub");
        var model = new ModelDescriptor("stub", "org/stub", "main", [new ModelFile("a.json"), new ModelFile("weights.bin")]);

        var reports = new Collector();
        string directory = await downloader.EnsureAsync(model, reports, TestContext.Current.CancellationToken);

        Assert.True(File.Exists(Path.Combine(directory, "a.json")));
        Assert.Equal(4096, new FileInfo(Path.Combine(directory, "weights.bin")).Length);
        Assert.True(downloader.IsComplete(model));
    }

    [Fact]
    public async Task SecondRunUsesTheCache()
    {
        var handler = new StubHandler { Files = { ["a.json"] = "{}"u8.ToArray() } };
        using var downloader = new ModelDownloader(new HttpClient(handler), ownsClient: true, _cache, "https://stub");
        var model = new ModelDescriptor("stub", "org/stub", "main", [new ModelFile("a.json")]);

        await downloader.EnsureAsync(model, null, TestContext.Current.CancellationToken);
        int firstBodyCount = handler.BodyRequests;

        var reports = new Collector();
        await downloader.EnsureAsync(model, reports, TestContext.Current.CancellationToken);

        Assert.Equal(firstBodyCount, handler.BodyRequests);
        Assert.Contains(reports.Reports, report => report.Cached);
    }

    [Fact]
    public async Task PartialFileIsResumed()
    {
        byte[] payload = Enumerable.Range(0, 1000).Select(i => (byte)(i % 251)).ToArray();
        var handler = new StubHandler { Files = { ["weights.bin"] = payload } };

        var model = new ModelDescriptor("stub", "org/stub", "main", [new ModelFile("weights.bin")]);
        using var downloader = new ModelDownloader(new HttpClient(handler), ownsClient: true, _cache, "https://stub");

        string directory = downloader.DirectoryFor(model);
        Directory.CreateDirectory(directory);
        await File.WriteAllBytesAsync(
            Path.Combine(directory, "weights.bin.part"), payload[..400], TestContext.Current.CancellationToken);

        await downloader.EnsureAsync(model, null, TestContext.Current.CancellationToken);

        Assert.Equal(
            payload,
            await File.ReadAllBytesAsync(
                Path.Combine(directory, "weights.bin"), TestContext.Current.CancellationToken));
        Assert.True(handler.SawRangeRequest);
    }

    [Fact]
    public async Task TruncatedResponseIsRejected()
    {
        var handler = new StubHandler
        {
            Files = { ["weights.bin"] = new byte[500] },
            TruncateBodyTo = 100,
        };

        using var downloader = new ModelDownloader(new HttpClient(handler), ownsClient: true, _cache, "https://stub");
        var model = new ModelDescriptor("stub", "org/stub", "main", [new ModelFile("weights.bin")]);

        await Assert.ThrowsAsync<IOException>(() =>
            downloader.EnsureAsync(model, null, TestContext.Current.CancellationToken));
        Assert.False(File.Exists(Path.Combine(downloader.DirectoryFor(model), "weights.bin")));
    }

    [Fact]
    public async Task MissingOptionalFileIsTolerated()
    {
        var handler = new StubHandler { Files = { ["a.json"] = "{}"u8.ToArray() } };
        using var downloader = new ModelDownloader(new HttpClient(handler), ownsClient: true, _cache, "https://stub");

        var model = new ModelDescriptor(
            "stub", "org/stub", "main", [new ModelFile("a.json"), new ModelFile("optional.json", Required: false)]);

        string directory = await downloader.EnsureAsync(model, null, TestContext.Current.CancellationToken);

        Assert.True(File.Exists(Path.Combine(directory, "a.json")));
        Assert.False(File.Exists(Path.Combine(directory, "optional.json")));
    }

    [Fact]
    public async Task MissingRequiredFileFails()
    {
        var handler = new StubHandler();
        using var downloader = new ModelDownloader(new HttpClient(handler), ownsClient: true, _cache, "https://stub");
        var model = new ModelDescriptor("stub", "org/stub", "main", [new ModelFile("missing.json")]);

        await Assert.ThrowsAnyAsync<Exception>(() =>
            downloader.EnsureAsync(model, null, TestContext.Current.CancellationToken));
    }

    [Fact]
    public void CatalogueResolvesKnownModels()
    {
        Assert.NotNull(ModelCatalog.Find("PaddleOCR-VL-1.6"));
        Assert.NotNull(ModelCatalog.Find("pp-doclayoutv3"));
        Assert.Null(ModelCatalog.Find("nonexistent"));
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Directory.Exists(_cache))
        {
            Directory.Delete(_cache, recursive: true);
        }
    }

    /// <summary>
    /// A synchronous progress sink. <see cref="Progress{T}"/> posts to the thread pool, so a
    /// report can still be in flight when the test asserts on it.
    /// </summary>
    private sealed class Collector : IProgress<DownloadProgress>
    {
        private readonly List<DownloadProgress> _reports = [];
        private readonly Lock _gate = new();

        public IReadOnlyList<DownloadProgress> Reports
        {
            get
            {
                lock (_gate)
                {
                    return [.. _reports];
                }
            }
        }

        public void Report(DownloadProgress value)
        {
            lock (_gate)
            {
                _reports.Add(value);
            }
        }
    }

    /// <summary>An in-memory stand-in for the Hugging Face file endpoint.</summary>
    private sealed class StubHandler : HttpMessageHandler
    {
        public Dictionary<string, byte[]> Files { get; } = new(StringComparer.Ordinal);

        public int BodyRequests { get; private set; }

        public bool SawRangeRequest { get; private set; }

        public int? TruncateBodyTo { get; init; }

        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request,
            CancellationToken cancellationToken)
        {
            string name = request.RequestUri!.Segments[^1];

            if (!Files.TryGetValue(name, out byte[]? payload))
            {
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
            }

            if (request.Method == HttpMethod.Head)
            {
                var head = new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new ByteArrayContent([]),
                };
                head.Content.Headers.ContentLength = payload.Length;
                head.Headers.AcceptRanges.Add("bytes");
                return Task.FromResult(head);
            }

            BodyRequests++;

            int offset = 0;
            HttpStatusCode status = HttpStatusCode.OK;
            if (request.Headers.Range?.Ranges.FirstOrDefault()?.From is { } from)
            {
                SawRangeRequest = true;
                offset = (int)from;
                status = HttpStatusCode.PartialContent;
            }

            byte[] body = payload[offset..];
            if (TruncateBodyTo is { } limit && body.Length > limit)
            {
                body = body[..limit];
            }

            return Task.FromResult(new HttpResponseMessage(status) { Content = new ByteArrayContent(body) });
        }
    }
}
