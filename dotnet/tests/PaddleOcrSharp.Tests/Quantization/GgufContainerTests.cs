using PaddleOcrSharp.Formats.Gguf;

namespace PaddleOcrSharp.Tests.Quantization;

/// <summary>The GGUF reader and writer, against each other and against the format's own rules.</summary>
public class GgufContainerTests : IDisposable
{
    private readonly string _directory = Directory.CreateTempSubdirectory("gguf-tests").FullName;

    [Fact]
    public void AWrittenFileReadsBackWithItsMetadataAndTensors()
    {
        string path = Path.Combine(_directory, "round-trip.gguf");
        float[] values = [1f, 2f, 3f, 4f, 5f, 6f];
        byte[] bytes = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);

        using (var writer = new GgufWriter(path))
        {
            writer.Add("general.architecture", "paddleocr-vl");
            writer.Add("paddleocr.quantization.group_size", 128u);
            writer.Add("prism.hadamard.weight_names", GgufValue.From(["a.weight", "b.weight"]));
            writer.Add("prism.hadamard.sign_values", GgufValue.From([1, -1, 1, -1]));
            writer.Declare("a.weight", GgmlType.F32, [3, 2]);
            writer.BeginData();
            writer.Write("a.weight", bytes);
            writer.Complete();
        }

        using GgufFile file = GgufFile.Open(path);

        Assert.Equal("paddleocr-vl", file.Meta("general.architecture")!.AsString());
        Assert.Equal(128u, file.Meta("paddleocr.quantization.group_size")!.AsUInt32());
        Assert.Equal(["a.weight", "b.weight"], file.Meta("prism.hadamard.weight_names")!.AsStringArray());
        Assert.Equal([1, -1, 1, -1], file.Meta("prism.hadamard.sign_values")!.AsInt32Array());

        GgufTensor tensor = file["a.weight"];
        Assert.Equal(GgmlType.F32, tensor.Type);
        Assert.Equal([3L, 2L], tensor.Ne);

        // GGUF lists the fastest-varying dimension first; the port's shapes are outermost first.
        Assert.Equal([2, 3], tensor.Shape);
        Assert.Equal(bytes, tensor.Bytes.ToArray());
    }

    [Fact]
    public void TensorDataStartsAtTheDeclaredAlignment()
    {
        string path = Path.Combine(_directory, "aligned.gguf");

        using (var writer = new GgufWriter(path))
        {
            writer.Add("general.architecture", "paddleocr-vl");
            writer.Declare("first", GgmlType.F32, [3]);
            writer.Declare("second", GgmlType.F32, [3]);
            writer.BeginData();
            writer.Write("first", new byte[12]);
            writer.Write("second", new byte[12]);
            writer.Complete();
        }

        using GgufFile file = GgufFile.Open(path);

        // Twelve bytes of data followed by twenty of padding: the second tensor has to begin on a
        // multiple of the alignment, which is what lets a reader map it directly.
        Assert.Equal(32, file.Alignment);
        Assert.Equal(2, file.Count);
    }

    [Fact]
    public void AQuantizedTensorTakesItsBlockLayoutsSize()
    {
        string path = Path.Combine(_directory, "quantized.gguf");

        using (var writer = new GgufWriter(path))
        {
            writer.Add("general.architecture", "paddleocr-vl");
            long length = writer.Declare("q.weight", GgmlType.PTQ1_0, [256, 4]);

            // Four rows of two 28-byte blocks.
            Assert.Equal(4 * 2 * 28, length);

            writer.BeginData();
            writer.Write("q.weight", new byte[length]);
            writer.Complete();
        }

        using GgufFile file = GgufFile.Open(path);
        Assert.Equal(1024, file["q.weight"].ElementCount);
        Assert.Equal(224, file["q.weight"].Bytes.Length);
    }

    [Fact]
    public void ARowThatDoesNotDivideTheGroupIsRefused()
    {
        string path = Path.Combine(_directory, "misaligned.gguf");
        using var writer = new GgufWriter(path);

        // 200 is not a multiple of 128, so there is no way to lay the blocks out — which is the
        // constraint that keeps the vision MLP's 4304-wide projection out of the ternary path.
        Assert.Throws<ArgumentException>(() => writer.Declare("bad.weight", GgmlType.PTQ1_0, [200, 4]));
    }

    [Fact]
    public void TensorsMustBeWrittenInDeclarationOrder()
    {
        string path = Path.Combine(_directory, "order.gguf");
        using var writer = new GgufWriter(path);

        writer.Declare("first", GgmlType.F32, [4]);
        writer.Declare("second", GgmlType.F32, [4]);
        writer.BeginData();

        Assert.Throws<InvalidOperationException>(() => writer.Write("second", new byte[16]));
    }

    [Fact]
    public void AnUnfinishedFileIsReportedRatherThanLeftShort()
    {
        string path = Path.Combine(_directory, "incomplete.gguf");
        using var writer = new GgufWriter(path);

        writer.Declare("only", GgmlType.F32, [4]);
        writer.BeginData();

        Assert.Throws<InvalidOperationException>(writer.Complete);
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
        try
        {
            Directory.Delete(_directory, recursive: true);
        }
        catch (IOException)
        {
            // A leftover temp directory is not worth failing a test run over.
        }
    }
}
