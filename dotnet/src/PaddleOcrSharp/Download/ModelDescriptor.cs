namespace PaddleOcrSharp.Download;

/// <summary>One file inside a model repository.</summary>
/// <param name="Path">Path relative to the repository root.</param>
/// <param name="Required">Whether the model cannot be used without it.</param>
public readonly record struct ModelFile(string Path, bool Required = true);

/// <summary>A downloadable model.</summary>
/// <param name="Name">Short local name, also the cache directory.</param>
/// <param name="Repository">Hugging Face repository, e.g. <c>PaddlePaddle/PaddleOCR-VL-1.6</c>.</param>
/// <param name="Revision">Git revision or branch.</param>
/// <param name="Files">Files to fetch.</param>
public sealed record ModelDescriptor(
    string Name,
    string Repository,
    string Revision,
    IReadOnlyList<ModelFile> Files);

/// <summary>The models the pipeline can fetch.</summary>
public static class ModelCatalog
{
    /// <summary>The 0.9B vision-language model that does the actual recognition.</summary>
    public static ModelDescriptor PaddleOcrVL16 { get; } = new(
        "PaddleOCR-VL-1.6",
        "PaddlePaddle/PaddleOCR-VL-1.6",
        "main",
        [
            new ModelFile("config.json"),
            new ModelFile("generation_config.json", Required: false),
            new ModelFile("model.safetensors"),
            new ModelFile("preprocessor_config.json"),
            new ModelFile("processor_config.json", Required: false),
            new ModelFile("tokenizer.json"),
            new ModelFile("tokenizer_config.json", Required: false),
            new ModelFile("added_tokens.json", Required: false),
            new ModelFile("special_tokens_map.json", Required: false),
            new ModelFile("chat_template.jinja", Required: false),
        ]);

    /// <summary>The RT-DETR layout detector used by the 1.6 pipeline.</summary>
    public static ModelDescriptor PpDocLayoutV3 { get; } = new(
        "PP-DocLayoutV3",
        "PaddlePaddle/PP-DocLayoutV3",
        "main",
        [
            new ModelFile("inference.json"),
            new ModelFile("inference.pdiparams"),
            new ModelFile("inference.yml"),
        ]);

    /// <summary>Document orientation classifier, used when doc preprocessing is enabled.</summary>
    public static ModelDescriptor DocOrientationClassifier { get; } = new(
        "PP-LCNet_x1_0_doc_ori",
        "PaddlePaddle/PP-LCNet_x1_0_doc_ori",
        "main",
        [
            new ModelFile("inference.json"),
            new ModelFile("inference.pdiparams"),
            new ModelFile("inference.yml"),
            new ModelFile("config.json", Required: false),
        ]);

    /// <summary>Document unwarping model, used when doc preprocessing is enabled.</summary>
    public static ModelDescriptor DocUnwarping { get; } = new(
        "UVDoc",
        "PaddlePaddle/UVDoc",
        "main",
        [
            new ModelFile("inference.json"),
            new ModelFile("inference.pdiparams"),
            new ModelFile("inference.yml"),
            new ModelFile("config.json", Required: false),
        ]);

    /// <summary>Every model the catalogue knows about.</summary>
    public static IReadOnlyList<ModelDescriptor> All { get; } =
        [PaddleOcrVL16, PpDocLayoutV3, DocOrientationClassifier, DocUnwarping];

    /// <summary>Finds a model by <see cref="ModelDescriptor.Name"/>, case-insensitively.</summary>
    public static ModelDescriptor? Find(string name) =>
        All.FirstOrDefault(model => string.Equals(model.Name, name, StringComparison.OrdinalIgnoreCase));
}
