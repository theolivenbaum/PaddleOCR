using System.Text.Json;
using PaddleOcrSharp.Models.Language;
using PaddleOcrSharp.Models.Vision;

namespace PaddleOcrSharp.Models;

/// <summary>
/// The pair of configurations read from a checkpoint's <c>config.json</c>.
/// </summary>
/// <param name="Language">Decoder hyper-parameters.</param>
/// <param name="Vision">Vision-tower hyper-parameters.</param>
public sealed record ModelConfiguration(LanguageConfig Language, VisionConfig Vision)
{
    /// <summary>The configuration of PaddleOCR-VL-1.6, used when no file is available.</summary>
    public static ModelConfiguration Default { get; } = new(LanguageConfig.Default, VisionConfig.Default);

    /// <summary>Reads <c>config.json</c> from <paramref name="directory"/>.</summary>
    public static ModelConfiguration Load(string directory)
    {
        string path = Path.Combine(directory, "config.json");
        if (!File.Exists(path))
        {
            return Default;
        }

        using FileStream stream = File.OpenRead(path);
        using JsonDocument document = JsonDocument.Parse(stream);
        JsonElement root = document.RootElement;

        LanguageConfig language = LanguageConfig.Default with
        {
            VocabSize = Int(root, "vocab_size", LanguageConfig.Default.VocabSize),
            HiddenSize = Int(root, "hidden_size", LanguageConfig.Default.HiddenSize),
            IntermediateSize = Int(root, "intermediate_size", LanguageConfig.Default.IntermediateSize),
            NumHiddenLayers = Int(root, "num_hidden_layers", LanguageConfig.Default.NumHiddenLayers),
            NumAttentionHeads = Int(root, "num_attention_heads", LanguageConfig.Default.NumAttentionHeads),
            NumKeyValueHeads = Int(root, "num_key_value_heads", LanguageConfig.Default.NumKeyValueHeads),
            HeadDim = Int(root, "head_dim", LanguageConfig.Default.HeadDim),
            RmsNormEps = (float)Double(root, "rms_norm_eps", LanguageConfig.Default.RmsNormEps),
            RopeTheta = Double(root, "rope_theta", LanguageConfig.Default.RopeTheta),
            MaxPositionEmbeddings = Int(
                root, "max_position_embeddings", LanguageConfig.Default.MaxPositionEmbeddings),
            ImageTokenId = Int(root, "image_token_id", LanguageConfig.Default.ImageTokenId),
            VideoTokenId = Int(root, "video_token_id", LanguageConfig.Default.VideoTokenId),
            VisionStartTokenId = Int(root, "vision_start_token_id", LanguageConfig.Default.VisionStartTokenId),
            VisionEndTokenId = Int(root, "vision_end_token_id", LanguageConfig.Default.VisionEndTokenId),
            PadTokenId = Int(root, "pad_token_id", LanguageConfig.Default.PadTokenId),
            MRopeSection = ReadMRopeSection(root) ?? LanguageConfig.Default.MRopeSection,
        };

        VisionConfig vision = VisionConfig.Default;
        if (root.TryGetProperty("vision_config", out JsonElement visionElement))
        {
            vision = vision with
            {
                HiddenSize = Int(visionElement, "hidden_size", vision.HiddenSize),
                IntermediateSize = Int(visionElement, "intermediate_size", vision.IntermediateSize),
                NumHiddenLayers = Int(visionElement, "num_hidden_layers", vision.NumHiddenLayers),
                NumAttentionHeads = Int(visionElement, "num_attention_heads", vision.NumAttentionHeads),
                NumChannels = Int(visionElement, "num_channels", vision.NumChannels),
                ImageSize = Int(visionElement, "image_size", vision.ImageSize),
                PatchSize = Int(visionElement, "patch_size", vision.PatchSize),
                LayerNormEps = (float)Double(visionElement, "layer_norm_eps", vision.LayerNormEps),
                SpatialMergeSize = Int(visionElement, "spatial_merge_size", vision.SpatialMergeSize),
            };
        }

        return new ModelConfiguration(language, vision);
    }

    private static int[]? ReadMRopeSection(JsonElement root)
    {
        if (!root.TryGetProperty("rope_scaling", out JsonElement scaling)
            || scaling.ValueKind is not JsonValueKind.Object
            || !scaling.TryGetProperty("mrope_section", out JsonElement section)
            || section.ValueKind is not JsonValueKind.Array)
        {
            return null;
        }

        return [.. section.EnumerateArray().Select(value => value.GetInt32())];
    }

    private static int Int(JsonElement element, string name, int fallback) =>
        element.TryGetProperty(name, out JsonElement value) && value.ValueKind is JsonValueKind.Number
            ? value.GetInt32()
            : fallback;

    private static double Double(JsonElement element, string name, double fallback) =>
        element.TryGetProperty(name, out JsonElement value) && value.ValueKind is JsonValueKind.Number
            ? value.GetDouble()
            : fallback;
}
