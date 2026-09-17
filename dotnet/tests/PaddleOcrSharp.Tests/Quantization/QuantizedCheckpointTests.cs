using PaddleOcrSharp.Core;
using PaddleOcrSharp.Formats.Gguf;
using PaddleOcrSharp.Models;
using PaddleOcrSharp.Quantization;

namespace PaddleOcrSharp.Tests.Quantization;

/// <summary>
/// The policy, the quantization header and the store that reads them — everything between the
/// codec and the model.
/// </summary>
public class QuantizedCheckpointTests : IDisposable
{
    private readonly string _directory = Directory.CreateTempSubdirectory("quantized-tests").FullName;

    [Theory]
    [InlineData("*norm*", "model.layers.3.input_layernorm.weight", true)]
    [InlineData("*.bias", "vision_model.encoder.layers.0.mlp.fc1.bias", true)]
    [InlineData("*.bias", "vision_model.encoder.layers.0.mlp.fc1.weight", false)]
    [InlineData("*embed_tokens*", "model.embed_tokens.weight", true)]
    [InlineData("model.layers.?.mlp*", "model.layers.7.mlp.up_proj.weight", true)]
    [InlineData("model.layers.?.mlp*", "model.layers.17.mlp.up_proj.weight", false)]
    public void GlobsMatchTheWayAPolicyExpects(string pattern, string name, bool expected) =>
        Assert.Equal(expected, QuantizationPolicy.Matches(pattern, name));

    [Fact]
    public void TheStartingPolicyQuantizesTheHeadAndSparesTheNorms()
    {
        QuantizationPolicy policy = QuantizationPolicy.Recommended;

        // lm_head is 106 M parameters re-read on every generated token, so it is the one tensor
        // whose quantization the whole decode win rests on.
        Assert.Equal(GgmlType.PTQ1_0, policy.SchemeFor("lm_head.weight"));
        Assert.Equal(GgmlType.PTQ1_0, policy.SchemeFor("model.layers.4.self_attn.q_proj.weight"));
        Assert.Equal(GgmlType.F32, policy.SchemeFor("model.layers.4.input_layernorm.weight"));
        Assert.Equal(GgmlType.F32, policy.SchemeFor("vision_model.encoder.layers.0.mlp.fc1.bias"));
        Assert.Equal(GgmlType.BF16, policy.SchemeFor("model.embed_tokens.weight"));
    }

    [Fact]
    public void APolicySurvivesBeingWrittenAndReadBack()
    {
        string path = Path.Combine(_directory, "policy.json");
        var policy = new QuantizationPolicy
        {
            DefaultScheme = "pq2_0",
            Rules = [new QuantizationRule("*norm*", "f32"), new QuantizationRule("*head*", "ptq1_0", Rotate: false)],
            Rotation = new RotationPolicy(1024, "explicit", 7),
        };

        policy.Save(path);
        QuantizationPolicy loaded = QuantizationPolicy.Load(path);

        Assert.Equal(GgmlType.PQ2_0, loaded.SchemeFor("model.layers.0.mlp.up_proj.weight"));
        Assert.Equal(GgmlType.F32, loaded.SchemeFor("model.norm.weight"));
        Assert.Equal(GgmlType.PTQ1_0, loaded.SchemeFor("lm_head.weight"));
        Assert.True(loaded.RotatesFor("model.layers.0.mlp.up_proj.weight"));
        Assert.False(loaded.RotatesFor("lm_head.weight"));
        Assert.Equal(1024, loaded.Rotation.BlockSize);
    }

    [Fact]
    public void TheRotationHeaderSurvivesTheFile()
    {
        string path = Path.Combine(_directory, "rotated.gguf");
        var rotation = new HadamardRotation(
            128, new Dictionary<int, float[]> { [256] = HadamardRotation.BuildSigns(256, 3) });

        var metadata = new QuantizationMetadata
        {
            Rotation = rotation,
            RotatedWeights = new HashSet<string>(["a.weight"], StringComparer.Ordinal),
            Method = "gptq+hadamard",
            Calibration = "four pages",
            Source = "PaddleOCR-VL-1.6",
        };

        using (var writer = new GgufWriter(path))
        {
            writer.Add("general.architecture", "paddleocr-vl");
            metadata.Write(writer);
            writer.Declare("a.weight", GgmlType.F32, [4]);
            writer.BeginData();
            writer.Write("a.weight", new byte[16]);
            writer.Complete();
        }

        using GgufFile file = GgufFile.Open(path);
        QuantizationMetadata read = QuantizationMetadata.Read(file);

        Assert.Equal("gptq+hadamard", read.Method);
        Assert.Equal("four pages", read.Calibration);
        Assert.NotNull(read.Rotation);
        Assert.Equal(128, read.Rotation!.BlockSize);
        Assert.Equal(rotation.Signs(256).ToArray(), read.Rotation.Signs(256).ToArray());
        Assert.NotNull(read.For("a.weight"));
        Assert.Null(read.For("b.weight"));
    }

    [Fact]
    public void AFileThatDeclaresARotationWithoutNamingItsWeightsIsRefused()
    {
        string path = Path.Combine(_directory, "nameless.gguf");

        using (var writer = new GgufWriter(path))
        {
            writer.Add("prism.hadamard.version", 1u);
            writer.Add("prism.hadamard.block_size", 128u);
            writer.Add("prism.hadamard.transform", QuantizationMetadata.TransformName);
            writer.Add("prism.hadamard.axis", QuantizationMetadata.AxisName);
            writer.Add("prism.hadamard.sign_mode", "identity");
            writer.Add("prism.hadamard.weight_names", GgufValue.From(Array.Empty<string>()));
            writer.Declare("a.weight", GgmlType.F32, [4]);
            writer.BeginData();
            writer.Write("a.weight", new byte[16]);
            writer.Complete();
        }

        using GgufFile file = GgufFile.Open(path);

        // Half-rotated maths is not a worse model, it is a different one, and it fails silently.
        Assert.Throws<InvalidDataException>(() => QuantizationMetadata.Read(file));
    }

    [Fact]
    public void AStoreServesAQuantizedTensorAsAWeightMatrix()
    {
        const int Rows = 4;
        const int Cols = 256;
        string path = Path.Combine(_directory, "store.gguf");

        var random = new Random(53);
        float[] weights = new float[Rows * Cols];
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = (random.Next(3) - 1) * 0.0625f;
        }

        int rowBytes = (int)GgmlType.PTQ1_0.RowSize(Cols);
        byte[] packed = new byte[rowBytes * Rows];
        for (int r = 0; r < Rows; r++)
        {
            TernaryBlocks.EncodePtq10(weights.AsSpan(r * Cols, Cols), packed.AsSpan(r * rowBytes, rowBytes));
        }

        using (var writer = new GgufWriter(path))
        {
            writer.Add("general.architecture", "paddleocr-vl");
            writer.Add("paddleocr.quantization.method", "rtn");
            writer.Declare("model.layers.0.mlp.up_proj.weight", GgmlType.PTQ1_0, [Cols, Rows]);
            writer.BeginData();
            writer.Write("model.layers.0.mlp.up_proj.weight", packed);
            writer.Complete();
        }

        using WeightStore store = WeightStore.Open(path);

        Assert.True(store.HasQuantizedWeights);
        Assert.Equal("rtn", store.Quantization!.Method);

        WeightMatrix matrix = store.Matrix("model.layers.0.mlp.up_proj.weight");
        Assert.True(matrix.IsQuantized);
        Assert.Equal(Rows, matrix.Rows);
        Assert.Equal(Cols, matrix.Cols);
        Assert.Equal("model.layers.0.mlp.up_proj.weight", matrix.Name);

        float[] decoded = new float[Rows * Cols];
        matrix.CopyTo(decoded);
        for (int i = 0; i < weights.Length; i++)
        {
            Assert.Equal(weights[i], decoded[i], 1e-4f);
        }
    }

    [Fact]
    public void ARecorderCollectsTheHessianAProductSees()
    {
        const int Cols = 128;
        const int Rows = 2;

        float[] weights = new float[Rows * Cols];
        var matrix = WeightMatrix.Create(
            System.Runtime.InteropServices.MemoryMarshal.AsBytes<float>(weights).ToArray(),
            DType.Float32,
            Rows,
            Cols,
            "probe.weight");

        float[] activations = new float[Cols];
        for (int i = 0; i < Cols; i++)
        {
            activations[i] = i % 3;
        }

        float[] output = new float[Rows];
        (ActivationRecorder recorder, IDisposable scope) = ActivationRecorder.Use(["probe.weight"]);
        using (scope)
        {
            Gemm.Linear(activations, 1, Cols, matrix, default, output, Rows);
        }

        double[] hessian = recorder.Hessian("probe.weight")!;
        Assert.Equal(1, recorder.Samples("probe.weight"));

        for (int i = 0; i < Cols; i++)
        {
            for (int j = 0; j < Cols; j++)
            {
                Assert.Equal(activations[i] * activations[j], hessian[(i * Cols) + j], 1e-6);
            }
        }
    }

    [Fact]
    public void NoRecorderMeansNoCollection()
    {
        Assert.Null(ActivationRecorder.Current);
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
