using PaddleOcrSharp.Models;

namespace PaddleOcrSharp.Tests.Fixtures;

/// <summary>
/// Opens the PaddleOCR-VL checkpoint once for every test that needs real weights.
/// </summary>
/// <remarks>
/// The checkpoint is 1.8 GB and is not committed. Set <c>PADDLEOCR_VL_DIR</c> to point at a
/// download; tests that need it skip when it is absent.
/// </remarks>
public sealed class CheckpointFixture : IDisposable
{
    private readonly Lazy<WeightStore?> _weights;

    /// <summary>Opens the checkpoint lazily.</summary>
    public CheckpointFixture() => _weights = new Lazy<WeightStore?>(Load);

    /// <summary>Directory holding <c>model.safetensors</c>, or <see langword="null"/>.</summary>
    public static string? Directory
    {
        get
        {
            string? configured = Environment.GetEnvironmentVariable("PADDLEOCR_VL_DIR");
            if (!string.IsNullOrEmpty(configured) && File.Exists(Path.Combine(configured, "model.safetensors")))
            {
                return configured;
            }

            string fallback = Path.Combine("/home/user/ref/vl16");
            return File.Exists(Path.Combine(fallback, "model.safetensors")) ? fallback : null;
        }
    }

    /// <summary>Whether the checkpoint is available.</summary>
    public static bool Available => Directory is not null;

    /// <summary>The opened checkpoint.</summary>
    public WeightStore Weights => _weights.Value
        ?? throw new InvalidOperationException("Checkpoint is unavailable; call RequireOrSkip first.");

    /// <summary>Skips the calling test when <c>tokenizer.json</c> is not available.</summary>
    public static void RequireTokenizerOrSkip()
    {
        if (Directory is null || !File.Exists(Path.Combine(Directory, "tokenizer.json")))
        {
            Assert.Skip("tokenizer.json not found; set PADDLEOCR_VL_DIR to a download.");
        }
    }

    /// <summary>Skips the calling test when the checkpoint is not present.</summary>
    public void RequireOrSkip()
    {
        if (!Available)
        {
            Assert.Skip("PaddleOCR-VL checkpoint not found; set PADDLEOCR_VL_DIR to a download.");
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_weights.IsValueCreated)
        {
            _weights.Value?.Dispose();
        }
    }

    private static WeightStore? Load()
    {
        string? directory = Directory;
        return directory is null ? null : WeightStore.Open(Path.Combine(directory, "model.safetensors"));
    }
}

/// <summary>Groups tests that share the opened checkpoint.</summary>
[CollectionDefinition(Name)]
public sealed class CheckpointCollection : ICollectionFixture<CheckpointFixture>
{
    /// <summary>Collection name.</summary>
    public const string Name = "checkpoint";
}
