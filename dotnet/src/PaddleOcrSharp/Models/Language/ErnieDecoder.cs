using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Models.Language;

/// <summary>
/// The ERNIE-4.5 causal decoder of PaddleOCR-VL, with a key/value cache and grouped-query
/// attention.
/// </summary>
/// <remarks>
/// Port of <c>Ernie4_5Model</c> / <c>Ernie4_5DecoderLayer</c> / <c>Ernie4_5Attention</c> from
/// <c>modeling_paddleocr_vl.py</c>. All projections are bias-free (<c>use_bias = false</c>) and
/// the norms are RMS norms in float32.
/// </remarks>
public sealed class ErnieDecoder
{
    private readonly LanguageConfig _config;
    private readonly MRotaryEmbedding _rope;
    private readonly LayerWeights[] _layers;
    private readonly float[] _finalNorm;
    private readonly WeightMatrix _embeddings;
    private readonly WeightMatrix _lmHead;

    /// <summary>Loads the decoder from a checkpoint.</summary>
    /// <param name="weights">Open checkpoint.</param>
    /// <param name="config">Decoder hyper-parameters.</param>
    /// <param name="prefix">Parameter-name prefix; the shipped checkpoint uses <c>model.</c>.</param>
    /// <param name="lmHeadName">Name of the output projection.</param>
    public ErnieDecoder(
        WeightStore weights,
        LanguageConfig config,
        string prefix = "model.",
        string lmHeadName = "lm_head.weight")
    {
        _config = config;
        _rope = new MRotaryEmbedding(config);
        _embeddings = weights.Matrix(prefix + "embed_tokens.weight");
        _lmHead = weights.Matrix(lmHeadName);
        _finalNorm = weights.Vector(prefix + "norm.weight");

        _layers = new LayerWeights[config.NumHiddenLayers];
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            string layer = $"{prefix}layers.{i}.";
            _layers[i] = new LayerWeights
            {
                InputNorm = weights.Vector(layer + "input_layernorm.weight"),
                PostAttentionNorm = weights.Vector(layer + "post_attention_layernorm.weight"),
                Query = weights.Matrix(layer + "self_attn.q_proj.weight"),
                Key = weights.Matrix(layer + "self_attn.k_proj.weight"),
                Value = weights.Matrix(layer + "self_attn.v_proj.weight"),
                Output = weights.Matrix(layer + "self_attn.o_proj.weight"),
                Gate = weights.Matrix(layer + "mlp.gate_proj.weight"),
                Up = weights.Matrix(layer + "mlp.up_proj.weight"),
                Down = weights.Matrix(layer + "mlp.down_proj.weight"),
            };
        }
    }

    /// <summary>Decoder configuration.</summary>
    public LanguageConfig Config => _config;

    /// <summary>Creates a cache sized for this decoder.</summary>
    public KvCache CreateCache(int capacity = 1024) =>
        new(_config.NumHiddenLayers, _config.KeyValueWidth, capacity);

    /// <summary>
    /// Looks up token embeddings into <paramref name="destination"/>, shaped
    /// <c>[tokens.Length, hiddenSize]</c>.
    /// </summary>
    public void Embed(ReadOnlySpan<int> tokens, Span<float> destination)
    {
        int width = _config.HiddenSize;
        for (int i = 0; i < tokens.Length; i++)
        {
            _embeddings.CopyRow(tokens[i], destination.Slice(i * width, width));
        }
    }

    /// <summary>
    /// Runs the decoder over <paramref name="hidden"/> and appends the new keys and values to
    /// <paramref name="cache"/>.
    /// </summary>
    /// <param name="hidden">Input embeddings, <c>[tokens, hiddenSize]</c>. Overwritten in place.</param>
    /// <param name="tokenCount">Number of tokens in <paramref name="hidden"/>.</param>
    /// <param name="positions">Position ids for those tokens.</param>
    /// <param name="cache">Key/value cache; its current length is the number of past tokens.</param>
    /// <param name="trace">Optional activation recorder.</param>
    public void Forward(
        Memory<float> hidden,
        int tokenCount,
        PositionIds positions,
        KvCache cache,
        Vision.VisionTrace? trace = null)
    {
        if (positions.Length < tokenCount)
        {
            throw new ArgumentException("Fewer position ids than tokens.", nameof(positions));
        }

        int width = _config.HiddenSize;
        int headDim = _config.HeadDim;
        int queryHeads = _config.NumAttentionHeads;
        int keyValueHeads = _config.NumKeyValueHeads;
        int groups = _config.NumKeyValueGroups;
        int queryWidth = _config.QueryWidth;
        int keyValueWidth = _config.KeyValueWidth;

        int pastLength = cache.Length;
        cache.Reserve(tokenCount);

        using PooledBuffer normed = TensorPool.Rent(tokenCount * width);
        using PooledBuffer queries = TensorPool.Rent(tokenCount * queryWidth);
        using PooledBuffer keys = TensorPool.Rent(tokenCount * keyValueWidth);
        using PooledBuffer values = TensorPool.Rent(tokenCount * keyValueWidth);
        using PooledBuffer attention = TensorPool.Rent(tokenCount * queryWidth);
        using PooledBuffer gate = TensorPool.Rent(tokenCount * _config.IntermediateSize);
        using PooledBuffer up = TensorPool.Rent(tokenCount * _config.IntermediateSize);
        using PooledBuffer cos = TensorPool.Rent(tokenCount * _rope.HalfHeadDim);
        using PooledBuffer sin = TensorPool.Rent(tokenCount * _rope.HalfHeadDim);

        for (int i = 0; i < tokenCount; i++)
        {
            _rope.Fill(
                positions.Temporal[i],
                positions.Height[i],
                positions.Width[i],
                cos.Span.Slice(i * _rope.HalfHeadDim, _rope.HalfHeadDim),
                sin.Span.Slice(i * _rope.HalfHeadDim, _rope.HalfHeadDim));
        }

        for (int layerIndex = 0; layerIndex < _layers.Length; layerIndex++)
        {
            LayerWeights layer = _layers[layerIndex];

            hidden.Span[..(tokenCount * width)].CopyTo(normed.Span);
            Norms.RmsNormParallel(normed.Memory, width, layer.InputNorm, _config.RmsNormEps);

            Gemm.Linear(normed.Memory, tokenCount, width, layer.Query, default, queries.Memory, queryWidth);
            Gemm.Linear(normed.Memory, tokenCount, width, layer.Key, default, keys.Memory, keyValueWidth);
            Gemm.Linear(normed.Memory, tokenCount, width, layer.Value, default, values.Memory, keyValueWidth);

            ApplyRope(queries.Span, tokenCount, queryHeads, headDim, cos.Span, sin.Span);
            ApplyRope(keys.Span, tokenCount, keyValueHeads, headDim, cos.Span, sin.Span);

            keys.Span[..(tokenCount * keyValueWidth)].CopyTo(cache.KeySlot(layerIndex, pastLength, tokenCount));
            values.Span[..(tokenCount * keyValueWidth)].CopyTo(cache.ValueSlot(layerIndex, pastLength, tokenCount));

            CausalAttention(
                queries.Memory,
                cache,
                layerIndex,
                pastLength,
                tokenCount,
                queryHeads,
                keyValueHeads,
                groups,
                headDim,
                attention.Memory);

            Gemm.Linear(attention.Memory, tokenCount, queryWidth, layer.Output, default, normed.Memory, width);
            Kernels.AddInPlace(hidden.Span[..(tokenCount * width)], normed.Span);

            hidden.Span[..(tokenCount * width)].CopyTo(normed.Span);
            Norms.RmsNormParallel(normed.Memory, width, layer.PostAttentionNorm, _config.RmsNormEps);

            Gemm.Linear(
                normed.Memory, tokenCount, width, layer.Gate, default, gate.Memory, _config.IntermediateSize);
            Gemm.Linear(
                normed.Memory, tokenCount, width, layer.Up, default, up.Memory, _config.IntermediateSize);
            Kernels.Silu(gate.Span);
            Kernels.MultiplyInPlace(gate.Span, up.Span);
            Gemm.Linear(
                gate.Memory, tokenCount, _config.IntermediateSize, layer.Down, default, normed.Memory, width);
            Kernels.AddInPlace(hidden.Span[..(tokenCount * width)], normed.Span);

            trace?.Record($"lm_layer{layerIndex}", hidden.Span, tokenCount, width);
        }

        cache.Advance(tokenCount);

        Norms.RmsNormParallel(hidden[..(tokenCount * width)], width, _finalNorm, _config.RmsNormEps);
        trace?.Record("lm_norm", hidden.Span, tokenCount, width);
    }

    /// <summary>
    /// Projects one hidden state onto the vocabulary.
    /// </summary>
    /// <remarks>
    /// Only the row that will be sampled is projected: the head is a 103 424 × 1024 matrix, so
    /// running it over a whole prefill would dominate the forward pass for no benefit.
    /// </remarks>
    public void Logits(ReadOnlyMemory<float> hiddenRow, Memory<float> logits) =>
        Gemm.Linear(hiddenRow, 1, _config.HiddenSize, _lmHead, default, logits, _config.VocabSize);

    private void ApplyRope(
        Span<float> projection,
        int tokenCount,
        int heads,
        int headDim,
        ReadOnlySpan<float> cos,
        ReadOnlySpan<float> sin)
    {
        int width = heads * headDim;
        int half = _rope.HalfHeadDim;

        for (int token = 0; token < tokenCount; token++)
        {
            ReadOnlySpan<float> tokenCos = cos.Slice(token * half, half);
            ReadOnlySpan<float> tokenSin = sin.Slice(token * half, half);
            Span<float> row = projection.Slice(token * width, width);

            for (int head = 0; head < heads; head++)
            {
                _rope.Apply(row.Slice(head * headDim, headDim), tokenCos, tokenSin);
            }
        }
    }

    /// <summary>
    /// Grouped-query causal attention against the cache.
    /// </summary>
    /// <remarks>
    /// Query <c>i</c> of this batch sits at absolute position <c>pastLength + i</c> and may attend
    /// to keys <c>0 .. pastLength + i</c>. Work is parallelised over (head, query) pairs; each
    /// worker keeps one score buffer for the whole visible history.
    /// </remarks>
    private void CausalAttention(
        ReadOnlyMemory<float> queries,
        KvCache cache,
        int layer,
        int pastLength,
        int tokenCount,
        int queryHeads,
        int keyValueHeads,
        int groups,
        int headDim,
        Memory<float> output)
    {
        int queryWidth = queryHeads * headDim;
        int keyValueWidth = keyValueHeads * headDim;
        int visible = pastLength + tokenCount;
        float scale = _config.AttentionScale;

        // `Advance` runs once after every layer has been processed, so the cache's published
        // length still excludes this batch; read the window that covers it explicitly.
        ReadOnlyMemory<float> keyMemory = cache.KeyWindow(layer, visible);
        ReadOnlyMemory<float> valueMemory = cache.ValueWindow(layer, visible);

        Parallel.For(
            0,
            queryHeads * tokenCount,
            () => TensorPool.Rent(visible),
            (index, _, scores) =>
            {
                int head = index / tokenCount;
                int token = index % tokenCount;
                int keyValueHead = head / groups;
                int limit = pastLength + token + 1;

                ReadOnlySpan<float> query = queries.Span.Slice((token * queryWidth) + (head * headDim), headDim);
                ReadOnlySpan<float> keySpan = keyMemory.Span;
                ReadOnlySpan<float> valueSpan = valueMemory.Span;
                Span<float> scoreSpan = scores.Span[..limit];

                int keyOffset = keyValueHead * headDim;
                for (int j = 0; j < limit; j++)
                {
                    scoreSpan[j] = Gemm.Dot(query, keySpan.Slice((j * keyValueWidth) + keyOffset, headDim)) * scale;
                }

                Kernels.Softmax(scoreSpan);

                Span<float> target = output.Span.Slice((token * queryWidth) + (head * headDim), headDim);
                target.Clear();
                for (int j = 0; j < limit; j++)
                {
                    float weight = scoreSpan[j];
                    if (weight != 0f)
                    {
                        Kernels.AddScaled(target, valueSpan.Slice((j * keyValueWidth) + keyOffset, headDim), weight);
                    }
                }

                return scores;
            },
            scores => scores.Dispose());
    }

    private sealed class LayerWeights
    {
        public required float[] InputNorm { get; init; }

        public required float[] PostAttentionNorm { get; init; }

        public required WeightMatrix Query { get; init; }

        public required WeightMatrix Key { get; init; }

        public required WeightMatrix Value { get; init; }

        public required WeightMatrix Output { get; init; }

        public required WeightMatrix Gate { get; init; }

        public required WeightMatrix Up { get; init; }

        public required WeightMatrix Down { get; init; }
    }
}
