using System.Net;
using System.Net.Http.Headers;
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
        var model = new ModelDescriptor("stub", "stub", [new ModelFile("a.json"), new ModelFile("weights.bin")]);

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
        var model = new ModelDescriptor("stub", "stub", [new ModelFile("a.json")]);

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
        var handler = new StubHandler { Files = { ["weights.bin"] = payload }, ETag = "\"v1\"" };

        var model = new ModelDescriptor("stub", "stub", [new ModelFile("weights.bin")]);
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
        Assert.True(handler.SawIfRange);
    }

    /// <summary>
    /// A <c>.part</c> left over from an earlier copy of the file must not be appended to: the
    /// result would be a splice of two files at exactly the length the server reports, which the
    /// size check cannot detect. If-Range is what makes the server send the whole body instead.
    /// </summary>
    [Fact]
    public async Task StalePartialFileIsDiscardedRatherThanSpliced()
    {
        byte[] payload = Enumerable.Range(0, 1000).Select(i => (byte)(i % 251)).ToArray();
        var handler = new StubHandler
        {
            Files = { ["weights.bin"] = payload },
            ETag = "\"v2\"",
            ContentChanged = true,
        };

        var model = new ModelDescriptor("stub", "stub", [new ModelFile("weights.bin")]);
        using var downloader = new ModelDownloader(new HttpClient(handler), ownsClient: true, _cache, "https://stub");

        string directory = downloader.DirectoryFor(model);
        Directory.CreateDirectory(directory);
        await File.WriteAllBytesAsync(
            Path.Combine(directory, "weights.bin.part"),
            Enumerable.Repeat((byte)0xEE, 400).ToArray(),
            TestContext.Current.CancellationToken);

        await downloader.EnsureAsync(model, null, TestContext.Current.CancellationToken);

        Assert.Equal(
            payload,
            await File.ReadAllBytesAsync(
                Path.Combine(directory, "weights.bin"), TestContext.Current.CancellationToken));
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
        var model = new ModelDescriptor("stub", "stub", [new ModelFile("weights.bin")]);

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
            "stub", "stub", [new ModelFile("a.json"), new ModelFile("optional.json", Required: false)]);

        string directory = await downloader.EnsureAsync(model, null, TestContext.Current.CancellationToken);

        Assert.True(File.Exists(Path.Combine(directory, "a.json")));
        Assert.False(File.Exists(Path.Combine(directory, "optional.json")));
    }

    [Fact]
    public async Task MissingRequiredFileFails()
    {
        var handler = new StubHandler();
        using var downloader = new ModelDownloader(new HttpClient(handler), ownsClient: true, _cache, "https://stub");
        var model = new ModelDescriptor("stub", "stub", [new ModelFile("missing.json")]);

        await Assert.ThrowsAnyAsync<Exception>(() =>
            downloader.EnsureAsync(model, null, TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task RequestsTheMirrorsFlatLayout()
    {
        var handler = new StubHandler { Files = { ["config.json"] = "{}"u8.ToArray() } };
        using var downloader = new ModelDownloader(
            new HttpClient(handler), ownsClient: true, _cache, "https://models.example/");
        var model = new ModelDescriptor("stub", "paddleocr-vl", [new ModelFile("config.json")]);

        await downloader.EnsureAsync(model, null, TestContext.Current.CancellationToken);

        Assert.NotEmpty(handler.Requests);
        Assert.All(
            handler.Requests,
            uri => Assert.Equal("https://models.example/paddleocr-vl/config.json", uri));
    }

    [Fact]
    public void ModelDirectoryIsJustTheModelName()
    {
        using var downloader = new ModelDownloader(
            new HttpClient(new StubHandler()), ownsClient: true, _cache, "https://stub");

        Assert.Equal(
            Path.Combine(_cache, "PaddleOCR-VL-1.6"),
            downloader.DirectoryFor(ModelCatalog.PaddleOcrVL16));
    }

    /// <summary>
    /// The remote paths are the mirror's directory names; a typo here is a 404 at first run, so
    /// they are pinned rather than derived from the model name.
    /// </summary>
    [Fact]
    public void CatalogueMirrorsTheBucketsDirectoryNames()
    {
        Assert.Equal("paddleocr-vl", ModelCatalog.PaddleOcrVL16.RemotePath);
        Assert.Equal("pp-doclayoutv3", ModelCatalog.PpDocLayoutV3.RemotePath);
        Assert.Equal("pp-lcnet-x1-0-doc-ori", ModelCatalog.DocOrientationClassifier.RemotePath);
        Assert.Equal("uvdoc", ModelCatalog.DocUnwarping.RemotePath);
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

    /// <summary>An in-memory stand-in for the model mirror.</summary>
    private sealed class StubHandler : HttpMessageHandler
    {
        public Dictionary<string, byte[]> Files { get; } = new(StringComparer.Ordinal);

        public List<string> Requests { get; } = [];

        public int BodyRequests { get; private set; }

        public bool SawRangeRequest { get; private set; }

        public bool SawIfRange { get; private set; }

        public int? TruncateBodyTo { get; init; }

        /// <summary>ETag to advertise, quoted as the header syntax requires.</summary>
        public string? ETag { get; init; }

        /// <summary>Answers a conditional range request as though the file had changed.</summary>
        public bool ContentChanged { get; init; }

        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request,
            CancellationToken cancellationToken)
        {
            string name = request.RequestUri!.Segments[^1];
            Requests.Add(request.RequestUri.ToString());

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
                if (ETag is { } tag)
                {
                    head.Headers.ETag = new EntityTagHeaderValue(tag);
                }

                return Task.FromResult(head);
            }

            BodyRequests++;

            int offset = 0;
            HttpStatusCode status = HttpStatusCode.OK;
            if (request.Headers.Range?.Ranges.FirstOrDefault()?.From is { } from)
            {
                SawRangeRequest = true;
                SawIfRange |= request.Headers.IfRange is not null;

                if (request.Headers.IfRange is not null && ContentChanged)
                {
                    // RFC 9110: a failed If-Range is answered with the whole representation.
                    return Task.FromResult(
                        new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(payload) });
                }

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
