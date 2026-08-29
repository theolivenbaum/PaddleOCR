using System.Diagnostics;
using PaddleOcrSharp.Core;
using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models.Language;
using PaddleOcrSharp.Models.Vision;
using PaddleOcrSharp.Text;

namespace PaddleOcrSharp.Models;

/// <summary>
/// The PaddleOCR-VL vision-language model: a NaViT SigLIP tower, a 2×2 merging projector and an
/// ERNIE-4.5 decoder, wired together the way
/// <c>PaddleOCRVLForConditionalGeneration.forward</c> wires them.
/// </summary>
public sealed class PaddleOcrVLModel : IDisposable
{
    private readonly WeightStore _weights;
    private readonly bool _ownsWeights;

    private PaddleOcrVLModel(
        WeightStore weights,
        bool ownsWeights,
        ModelConfiguration configuration,
        BpeTokenizer tokenizer)
    {
        _weights = weights;
        _ownsWeights = ownsWeights;
        Configuration = configuration;
        Tokenizer = tokenizer;
        Vision = new VisionTower(weights, configuration.Vision, configuration.Language.HiddenSize);
        Decoder = new ErnieDecoder(weights, configuration.Language);
    }

    /// <summary>Model hyper-parameters.</summary>
    public ModelConfiguration Configuration { get; }

    /// <summary>The checkpoint's tokenizer.</summary>
    public BpeTokenizer Tokenizer { get; }

    /// <summary>The vision tower and projector.</summary>
    public VisionTower Vision { get; }

    /// <summary>The ERNIE decoder.</summary>
    public ErnieDecoder Decoder { get; }

    /// <summary>
    /// Loads a model from a directory holding <c>model.safetensors</c>, <c>config.json</c> and
    /// <c>tokenizer.json</c>.
    /// </summary>
    public static PaddleOcrVLModel Load(string directory)
    {
        ModelConfiguration configuration = ModelConfiguration.Load(directory);
        var tokenizer = BpeTokenizer.FromFile(Path.Combine(directory, "tokenizer.json"));
        WeightStore weights = WeightStore.Open(Path.Combine(directory, "model.safetensors"));
        return new PaddleOcrVLModel(weights, ownsWeights: true, configuration, tokenizer);
    }

    /// <summary>Builds a model over an already-open checkpoint, for tests and tooling.</summary>
    public static PaddleOcrVLModel FromWeights(
        WeightStore weights,
        ModelConfiguration configuration,
        BpeTokenizer tokenizer) => new(weights, ownsWeights: false, configuration, tokenizer);

    /// <summary>
    /// Builds the prompt token ids for one image and one instruction.
    /// </summary>
    /// <remarks>
    /// Mirrors the checkpoint's chat template followed by
    /// <c>PaddleOCRVLProcessor.__call__</c>'s placeholder expansion:
    /// <c>&lt;|begin_of_sentence|&gt;User: &lt;|IMAGE_START|&gt;</c>, then one
    /// <c>&lt;|IMAGE_PLACEHOLDER|&gt;</c> per merged patch, then
    /// <c>&lt;|IMAGE_END|&gt;{instruction}\nAssistant:\n</c>.
    /// </remarks>
    public int[] BuildPrompt(ImageGrid grid, string instruction)
    {
        int imageTokens = grid.TokenCount(Configuration.Vision.SpatialMergeSize);
        var ids = new List<int>(imageTokens + 32);

        Tokenizer.EncodeInto("<|begin_of_sentence|>User: <|IMAGE_START|>", ids);
        for (int i = 0; i < imageTokens; i++)
        {
            ids.Add(Configuration.Language.ImageTokenId);
        }

        Tokenizer.EncodeInto($"<|IMAGE_END|>{instruction}\nAssistant:\n", ids);
        return [.. ids];
    }

    /// <summary>
    /// Runs the model on one image and returns the decoded text.
    /// </summary>
    /// <param name="image">Image to recognise; already cropped to the block of interest.</param>
    /// <param name="instruction">Task prompt, e.g. <c>"OCR:"</c>.</param>
    /// <param name="preprocessing">Pixel-budget settings for this block.</param>
    /// <param name="generation">Decoding settings.</param>
    /// <param name="profile">Optional collector for what this call cost.</param>
    /// <param name="label">Name this call is reported under when <paramref name="profile"/> is given.</param>
    /// <param name="cancellationToken">Cancels generation between tokens.</param>
    public string Recognize(
        RgbImage image,
        string instruction,
        VisionPreprocessorOptions? preprocessing = null,
        GenerationOptions? generation = null,
        RecognitionProfile? profile = null,
        string? label = null,
        CancellationToken cancellationToken = default)
    {
        using PreprocessedImage preprocessed = VisionPreprocessor.Preprocess(
            image, preprocessing ?? VisionPreprocessorOptions.Default);

        return Recognize(preprocessed, instruction, generation, profile, label, cancellationToken);
    }

    /// <summary>Runs the model on an already-preprocessed image.</summary>
    /// <param name="image">Preprocessed block image.</param>
    /// <param name="instruction">Task prompt, e.g. <c>"OCR:"</c>.</param>
    /// <param name="generation">Decoding settings.</param>
    /// <param name="profile">Optional collector for what this call cost.</param>
    /// <param name="label">Name this call is reported under when <paramref name="profile"/> is given.</param>
    /// <param name="cancellationToken">Cancels generation between tokens.</param>
    public string Recognize(
        PreprocessedImage image,
        string instruction,
        GenerationOptions? generation = null,
        RecognitionProfile? profile = null,
        string? label = null,
        CancellationToken cancellationToken = default)
    {
        GenerationOptions options = generation ?? GenerationOptions.Default;
        int[] prompt = BuildPrompt(image.Grid, instruction);

        if (profile is null)
        {
            using Tensor plain = Vision.Encode(image);
            return Tokenizer.Decode(
                Generate(prompt, plain, image.Grid, options, out _, cancellationToken),
                options.SkipSpecialTokens);
        }

        long threadStart = GC.GetAllocatedBytesForCurrentThread();
        long totalStart = GC.GetTotalAllocatedBytes(precise: false);
        long visionStart = Stopwatch.GetTimestamp();

        using Tensor imageEmbeddings = Vision.Encode(image);
        TimeSpan vision = Stopwatch.GetElapsedTime(visionStart);

        List<int> generated = Generate(
            prompt, imageEmbeddings, image.Grid, options, out GenerationStats stats, cancellationToken);

        profile.Add(new RecognitionRecord(
            label ?? string.Empty,
            image.Grid.Height * image.Grid.Width * image.Grid.Temporal,
            stats.PromptTokens,
            stats.GeneratedTokens,
            stats.HitTokenBudget,
            stats.StoppedEarly,
            vision,
            stats.Prefill,
            stats.Decode,
            stats.DecodeLogits,
            GC.GetAllocatedBytesForCurrentThread() - threadStart,
            Math.Max(0, GC.GetTotalAllocatedBytes(precise: false) - totalStart)));

        return Tokenizer.Decode(generated, options.SkipSpecialTokens);
    }

    /// <summary>
    /// Prefills the prompt and decodes until the end-of-sequence token or the token budget.
    /// </summary>
    /// <param name="prompt">Prompt token ids, with image placeholders already expanded.</param>
    /// <param name="imageEmbeddings">Projected image features, one row per placeholder.</param>
    /// <param name="grid">Patch grid of the image, for the 3-D rope index.</param>
    /// <param name="options">Decoding settings.</param>
    /// <param name="cancellationToken">Cancels generation between tokens.</param>
    public List<int> Generate(
        int[] prompt,
        Tensor imageEmbeddings,
        ImageGrid grid,
        GenerationOptions options,
        CancellationToken cancellationToken = default) =>
        Generate(prompt, imageEmbeddings, grid, options, out _, cancellationToken);

    /// <summary>
    /// Prefills the prompt and decodes until the stop token, the token budget, or - when
    /// <see cref="GenerationOptions.StopOnRepetition"/> is set - the point at which the output has
    /// provably fallen into a loop.
    /// </summary>
    /// <param name="prompt">Prompt token ids, with image placeholders already expanded.</param>
    /// <param name="imageEmbeddings">Projected image features, one row per placeholder.</param>
    /// <param name="grid">Patch grid of the image, for the 3-D rope index.</param>
    /// <param name="options">Decoding settings.</param>
    /// <param name="stats">What the call cost.</param>
    /// <param name="cancellationToken">Cancels generation between tokens.</param>
    public List<int> Generate(
        int[] prompt,
        Tensor imageEmbeddings,
        ImageGrid grid,
        GenerationOptions options,
        out GenerationStats stats,
        CancellationToken cancellationToken = default)
    {
        LanguageConfig config = Configuration.Language;
        int width = config.HiddenSize;

        (PositionIds positions, int delta) = RopeIndex.Compute(
            prompt,
            [(grid.Temporal, grid.Height, grid.Width)],
            config,
            Configuration.Vision.SpatialMergeSize);

        using Tensor hidden = Tensor.Rent(prompt.Length, width);
        Decoder.Embed(prompt, hidden.Span);
        ScatterImageEmbeddings(prompt, imageEmbeddings, hidden.Span, config.ImageTokenId, width);

        long prefillStart = Stopwatch.GetTimestamp();

        using KvCache cache = Decoder.CreateCache(prompt.Length + Math.Min(options.MaxNewTokens, 1024));
        Decoder.Forward(hidden.Memory, prompt.Length, positions, cache);

        using PooledBuffer logits = TensorPool.Rent(config.VocabSize);
        Decoder.Logits(hidden.Memory.Slice((prompt.Length - 1) * width, width), logits.Memory);

        TimeSpan prefill = Stopwatch.GetElapsedTime(prefillStart);
        long decodeStart = Stopwatch.GetTimestamp();

        var sampler = new Sampler(options);
        var generated = new List<int>(Math.Min(options.MaxNewTokens, 512));

        using Tensor step = Tensor.Rent(1, width);
        int nextPosition = prompt.Length + delta;

        var loop = new LoopDetector(options);
        bool hitBudget = false;
        bool stoppedEarly = false;
        long logitsTicks = 0;

        // One reusable slot: `Embed([token], ...)` would allocate a fresh array per token.
        Span<int> one = stackalloc int[1];

        for (int i = 0; i < options.MaxNewTokens; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            int token = sampler.Select(logits.Span, generated);
            if (token == config.EosTokenId)
            {
                break;
            }

            generated.Add(token);

            // A block whose decoder has fallen into a cycle will not leave it: every later token is
            // produced from a state the loop already reproduced. Nothing after this point survives
            // `RepetitionTruncator`, so continuing to the budget only buys a longer string to throw
            // away - and, because attention re-reads the whole cache, an increasingly expensive one.
            if (loop.IsLooping(generated))
            {
                stoppedEarly = true;
                break;
            }

            if (i == options.MaxNewTokens - 1)
            {
                hitBudget = true;
                break;
            }

            one[0] = token;
            Decoder.Embed(one, step.Span);

            // Generated tokens are pure text, so all three rope axes advance together from the
            // prompt's final position plus the delta `get_rope_index` returned.
            PositionIds stepPosition = PositionIds.Sequential(1, nextPosition + i);
            Decoder.Forward(step.Memory, 1, stepPosition, cache);

            long logitsStart = Stopwatch.GetTimestamp();
            Decoder.Logits(step.Memory, logits.Memory);
            logitsTicks += Stopwatch.GetTimestamp() - logitsStart;
        }

        stats = new GenerationStats(
            prompt.Length,
            generated.Count,
            hitBudget,
            stoppedEarly,
            prefill,
            Stopwatch.GetElapsedTime(decodeStart),
            Stopwatch.GetElapsedTime(0, logitsTicks));

        return generated;
    }

    private static void ScatterImageEmbeddings(
        ReadOnlySpan<int> prompt,
        Tensor imageEmbeddings,
        Span<float> hidden,
        int imageTokenId,
        int width)
    {
        int rows = imageEmbeddings.Length / width;
        int cursor = 0;

        for (int i = 0; i < prompt.Length; i++)
        {
            if (prompt[i] != imageTokenId)
            {
                continue;
            }

            if (cursor >= rows)
            {
                throw new InvalidOperationException(
                    $"Prompt has more image placeholders than the {rows} features produced by the vision tower.");
            }

            imageEmbeddings.Span.Slice(cursor * width, width).CopyTo(hidden.Slice(i * width, width));
            cursor++;
        }

        if (cursor != rows)
        {
            throw new InvalidOperationException(
                $"Vision tower produced {rows} features but the prompt has {cursor} image placeholders.");
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_ownsWeights)
        {
            _weights.Dispose();
        }
    }
}
