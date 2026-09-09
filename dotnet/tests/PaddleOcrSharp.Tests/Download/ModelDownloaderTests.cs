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
        await File.WriteAllTextAsync(
            Path.Combine(directory, "weights.bin.part.etag"), "\"v1\"", TestContext.Current.CancellationToken);

        await downloader.EnsureAsync(model, null, TestContext.Current.CancellationToken);

        Assert.Equal(
            payload,
            await File.ReadAllBytesAsync(
                Path.Combine(directory, "weights.bin"), TestContext.Current.CancellationToken));
        Assert.True(handler.SawRangeRequest);
        Assert.True(handler.SawIfRange);
    }

    /// <summary>
    /// A server may answer a ranged request with the whole representation — because it ignores
    /// the header, or because <c>If-Range</c> failed. The partial file must then be replaced
    /// rather than appended to.
    /// </summary>
    [Fact]
    public async Task ServerThatIgnoresTheRangeRestartsTheTransfer()
    {
        byte[] payload = Enumerable.Range(0, 1000).Select(i => (byte)(i % 251)).ToArray();
        var handler = new StubHandler
        {
            Files = { ["weights.bin"] = payload },
            ETag = "\"v1\"",
            IgnoresRangeRequests = true,
        };

        var model = new ModelDescriptor("stub", "stub", [new ModelFile("weights.bin")]);
        using var downloader = new ModelDownloader(new HttpClient(handler), ownsClient: true, _cache, "https://stub");

        string directory = downloader.DirectoryFor(model);
        Directory.CreateDirectory(directory);
        await File.WriteAllBytesAsync(
            Path.Combine(directory, "weights.bin.part"),
            payload[..400],
            TestContext.Current.CancellationToken);
        await File.WriteAllTextAsync(
            Path.Combine(directory, "weights.bin.part.etag"), "\"v1\"", TestContext.Current.CancellationToken);

        await downloader.EnsureAsync(model, null, TestContext.Current.CancellationToken);

        Assert.Equal(
            payload,
            await File.ReadAllBytesAsync(
                Path.Combine(directory, "weights.bin"), TestContext.Current.CancellationToken));
    }

    /// <summary>
    /// A <c>.part</c> left over from an earlier copy of the file must not be appended to: the
    /// result would splice two files into one of exactly the length the server reports, which the
    /// size check cannot see. <c>If-Range</c> cannot catch this — the validator it carries is the
    /// one just probed — so the validator recorded when the partial file was started is the only
    /// thing standing between an interrupted transfer and a corrupt model.
    /// </summary>
    [Fact]
    public async Task StalePartialFileIsDiscardedRatherThanSpliced()
    {
        byte[] payload = Enumerable.Range(0, 1000).Select(i => (byte)(i % 251)).ToArray();
        var handler = new StubHandler { Files = { ["weights.bin"] = payload }, ETag = "\"v2\"" };

        var model = new ModelDescriptor("stub", "stub", [new ModelFile("weights.bin")]);
        using var downloader = new ModelDownloader(new HttpClient(handler), ownsClient: true, _cache, "https://stub");

        string directory = downloader.DirectoryFor(model);
        Directory.CreateDirectory(directory);
        await File.WriteAllBytesAsync(
            Path.Combine(directory, "weights.bin.part"),
            Enumerable.Repeat((byte)0xEE, 400).ToArray(),
            TestContext.Current.CancellationToken);
        await File.WriteAllTextAsync(
            Path.Combine(directory, "weights.bin.part.etag"), "\"v1\"", TestContext.Current.CancellationToken);

        await downloader.EnsureAsync(model, null, TestContext.Current.CancellationToken);

        Assert.Equal(
            payload,
            await File.ReadAllBytesAsync(
                Path.Combine(directory, "weights.bin"), TestContext.Current.CancellationToken));
        Assert.False(
            File.Exists(Path.Combine(directory, "weights.bin.part.etag")),
            "the recorded validator should be cleaned up with the finished file");
    }

    /// <summary>
    /// The R2 bucket behind models.curiosity.ai answers HEAD with <c>405 Allow: PUT, GET,
    /// DELETE</c>, so the length has to come from a one-byte ranged GET instead — and the refusal
    /// is remembered, since a mirror that refuses HEAD for one file refuses it for all of them.
    /// </summary>
    [Fact]
    public async Task LengthComesFromARangedGetWhenHeadIsRejected()
    {
        byte[] weights = Enumerable.Range(0, 4096).Select(i => (byte)i).ToArray();
        var handler = new StubHandler
        {
            Files = { ["a.json"] = "{\"a\":1}"u8.ToArray(), ["weights.bin"] = weights },
            RejectHead = true,
            ETag = "\"v1\"",
        };

        using var downloader = new ModelDownloader(new HttpClient(handler), ownsClient: true, _cache, "https://stub");
        var model = new ModelDescriptor("stub", "stub", [new ModelFile("a.json"), new ModelFile("weights.bin")]);

        string directory = await downloader.EnsureAsync(model, null, TestContext.Current.CancellationToken);

        Assert.Equal(
            weights,
            await File.ReadAllBytesAsync(
                Path.Combine(directory, "weights.bin"), TestContext.Current.CancellationToken));
        Assert.Equal(2, handler.ProbeRequests);
        Assert.Equal(1, handler.HeadRequests);
        Assert.True(downloader.IsComplete(model));
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

    /// <summary>
    /// An in-memory stand-in for the model mirror, close enough to the real one to be worth
    /// asserting against: it answers ranged requests with a <c>Content-Range</c>, can refuse HEAD
    /// the way an R2 bucket does, and ignores <c>If-Range</c> unless told to honour it.
    /// </summary>
    private sealed class StubHandler : HttpMessageHandler
    {
        public Dictionary<string, byte[]> Files { get; } = new(StringComparer.Ordinal);

        public List<string> Requests { get; } = [];

        /// <summary>Requests that transferred the file, excluding one-byte metadata probes.</summary>
        public int BodyRequests { get; private set; }

        public int HeadRequests { get; private set; }

        public int ProbeRequests { get; private set; }

        public bool SawRangeRequest { get; private set; }

        public bool SawIfRange { get; private set; }

        public int? TruncateBodyTo { get; init; }

        /// <summary>ETag to advertise, quoted as the header syntax requires.</summary>
        public string? ETag { get; init; }

        /// <summary>Answers HEAD with 405, as the bucket behind models.curiosity.ai does.</summary>
        public bool RejectHead { get; init; }

        /// <summary>Answers a ranged request with the whole body instead of a 206.</summary>
        public bool IgnoresRangeRequests { get; init; }

        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request,
            CancellationToken cancellationToken)
        {
            string name = request.RequestUri!.Segments[^1];
            Requests.Add(request.RequestUri.ToString());

            if (request.Method == HttpMethod.Head)
            {
                HeadRequests++;
                if (RejectHead)
                {
                    var rejected = new HttpResponseMessage(HttpStatusCode.MethodNotAllowed);
                    rejected.Content.Headers.Allow.Add("GET");
                    return Task.FromResult(rejected);
                }
            }

            if (!Files.TryGetValue(name, out byte[]? payload))
            {
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
            }

            if (request.Method == HttpMethod.Head)
            {
                var head = new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent([]) };
                head.Content.Headers.ContentLength = payload.Length;
                head.Headers.AcceptRanges.Add("bytes");
                Tag(head);
                return Task.FromResult(head);
            }

            RangeItemHeaderValue? range = request.Headers.Range?.Ranges.FirstOrDefault();
            bool probe = range is { From: 0, To: 0 };

            if (probe)
            {
                ProbeRequests++;
            }
            else
            {
                BodyRequests++;
            }

            if (range?.From is { } from)
            {
                if (!probe)
                {
                    SawRangeRequest = true;
                    SawIfRange |= request.Headers.IfRange is not null;

                    if (IgnoresRangeRequests)
                    {
                        return Task.FromResult(Whole(payload));
                    }
                }

                int start = (int)from;
                int last = range.To is { } to
                    ? Math.Min((int)to, payload.Length - 1)
                    : payload.Length - 1;

                byte[] slice = payload[start..(last + 1)];
                if (TruncateBodyTo is { } cap && slice.Length > cap)
                {
                    slice = slice[..cap];
                }

                var partial = new HttpResponseMessage(HttpStatusCode.PartialContent)
                {
                    Content = new ByteArrayContent(slice),
                };
                partial.Content.Headers.ContentRange = new ContentRangeHeaderValue(start, last, payload.Length);
                Tag(partial);
                return Task.FromResult(partial);
            }

            return Task.FromResult(Whole(payload));
        }

        private HttpResponseMessage Whole(byte[] payload)
        {
            byte[] body = TruncateBodyTo is { } cap && payload.Length > cap ? payload[..cap] : payload;
            var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(body) };
            response.Content.Headers.ContentLength = payload.Length;
            response.Headers.AcceptRanges.Add("bytes");
            Tag(response);
            return response;
        }

        private void Tag(HttpResponseMessage response)
        {
            if (ETag is { } tag)
            {
                response.Headers.ETag = new EntityTagHeaderValue(tag);
            }
        }
    }
}
