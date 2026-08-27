using System.Text;
using System.Text.Json;
using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Text;

/// <summary>
/// The SentencePiece-style byte-level-fallback BPE tokenizer shipped with PaddleOCR-VL, loaded
/// from a Hugging Face <c>tokenizer.json</c>.
/// </summary>
/// <remarks>
/// <para>
/// The checkpoint's configuration is narrow and this implementation targets it exactly rather
/// than the whole <c>tokenizers</c> feature surface:
/// </para>
/// <list type="bullet">
///   <item>normalizer: a single <c>Replace(" " → "▁")</c>;</item>
///   <item>pre-tokenizer: none, so a segment is BPE-merged as one word;</item>
///   <item>model: BPE with <c>byte_fallback</c> and <c>fuse_unk</c>;</item>
///   <item>decoder: <c>Replace("▁" → " ")</c>, then <c>ByteFallback</c>, then <c>Fuse</c>.</item>
/// </list>
/// <para>
/// Anything else in the file is rejected at load time rather than silently ignored, so a future
/// checkpoint with a different pipeline fails loudly instead of producing wrong tokens.
/// </para>
/// </remarks>
public sealed class BpeTokenizer
{
    private const char SentencePieceSpace = '▁';

    private readonly Dictionary<string, int> _vocab;
    private readonly string[] _idToToken;
    private readonly Dictionary<(int Left, int Right), (int Rank, int Id)> _merges;
    private readonly AddedToken[] _addedTokens;
    private readonly Dictionary<string, AddedToken> _addedByContent;
    private readonly int _unknownId;
    private readonly int[] _byteFallbackIds;
    private readonly bool _fuseUnknown;

    private BpeTokenizer(
        Dictionary<string, int> vocab,
        string[] idToToken,
        Dictionary<(int, int), (int, int)> merges,
        AddedToken[] addedTokens,
        int unknownId,
        int[] byteFallbackIds,
        bool fuseUnknown)
    {
        _vocab = vocab;
        _idToToken = idToToken;
        _merges = merges;
        _addedTokens = addedTokens;
        _unknownId = unknownId;
        _byteFallbackIds = byteFallbackIds;
        _fuseUnknown = fuseUnknown;
        _addedByContent = addedTokens.ToDictionary(token => token.Content, StringComparer.Ordinal);
    }

    /// <summary>Number of entries in the vocabulary, including added tokens.</summary>
    public int VocabSize => _idToToken.Length;

    /// <summary>The tokens that bypass BPE.</summary>
    public IReadOnlyList<AddedToken> AddedTokens => _addedTokens;

    /// <summary>Loads a tokenizer from a <c>tokenizer.json</c> file.</summary>
    public static BpeTokenizer FromFile(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return FromJson(stream);
    }

    /// <summary>Loads a tokenizer from a <c>tokenizer.json</c> stream.</summary>
    public static BpeTokenizer FromJson(Stream stream)
    {
        using JsonDocument document = JsonDocument.Parse(stream);
        JsonElement root = document.RootElement;

        ValidateNormalizer(root);
        ValidatePreTokenizer(root);
        ValidateDecoder(root);

        JsonElement model = root.GetProperty("model");
        string type = model.GetProperty("type").GetString() ?? string.Empty;
        if (type != "BPE")
        {
            throw new NotSupportedException($"Only BPE tokenizers are supported, found '{type}'.");
        }

        if (model.TryGetProperty("dropout", out JsonElement dropout) && dropout.ValueKind is not JsonValueKind.Null)
        {
            throw new NotSupportedException("BPE dropout is not supported.");
        }

        if (model.TryGetProperty("continuing_subword_prefix", out JsonElement prefix)
            && prefix.ValueKind is not JsonValueKind.Null
            && prefix.GetString() is { Length: > 0 })
        {
            throw new NotSupportedException("continuing_subword_prefix is not supported.");
        }

        if (model.TryGetProperty("end_of_word_suffix", out JsonElement suffix)
            && suffix.ValueKind is not JsonValueKind.Null
            && suffix.GetString() is { Length: > 0 })
        {
            throw new NotSupportedException("end_of_word_suffix is not supported.");
        }

        bool byteFallback = model.TryGetProperty("byte_fallback", out JsonElement fallback)
            && fallback.ValueKind is JsonValueKind.True;
        bool fuseUnknown = model.TryGetProperty("fuse_unk", out JsonElement fuse)
            && fuse.ValueKind is JsonValueKind.True;

        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (JsonProperty entry in model.GetProperty("vocab").EnumerateObject())
        {
            vocab[entry.Name] = entry.Value.GetInt32();
        }

        var added = new List<AddedToken>();
        if (root.TryGetProperty("added_tokens", out JsonElement addedElement))
        {
            foreach (JsonElement entry in addedElement.EnumerateArray())
            {
                var token = new AddedToken(
                    entry.GetProperty("id").GetInt32(),
                    entry.GetProperty("content").GetString()!,
                    entry.TryGetProperty("special", out JsonElement special) && special.ValueKind is JsonValueKind.True,
                    entry.TryGetProperty("lstrip", out JsonElement lstrip) && lstrip.ValueKind is JsonValueKind.True,
                    entry.TryGetProperty("rstrip", out JsonElement rstrip) && rstrip.ValueKind is JsonValueKind.True,
                    entry.TryGetProperty("normalized", out JsonElement normalized)
                        && normalized.ValueKind is JsonValueKind.True);

                added.Add(token);
                vocab.TryAdd(token.Content, token.Id);
            }
        }

        int maxId = vocab.Count == 0 ? -1 : vocab.Values.Max();
        string[] idToToken = new string[maxId + 1];
        foreach ((string token, int id) in vocab)
        {
            idToToken[id] = token;
        }

        var merges = new Dictionary<(int, int), (int, int)>();
        int rank = 0;
        foreach (JsonElement entry in model.GetProperty("merges").EnumerateArray())
        {
            string left;
            string right;
            if (entry.ValueKind is JsonValueKind.Array)
            {
                left = entry[0].GetString()!;
                right = entry[1].GetString()!;
            }
            else
            {
                string text = entry.GetString()!;
                int space = text.IndexOf(' ');
                left = text[..space];
                right = text[(space + 1)..];
            }

            if (vocab.TryGetValue(left, out int leftId)
                && vocab.TryGetValue(right, out int rightId)
                && vocab.TryGetValue(left + right, out int mergedId))
            {
                merges.TryAdd((leftId, rightId), (rank, mergedId));
            }

            rank++;
        }

        string unknown = model.TryGetProperty("unk_token", out JsonElement unk) && unk.ValueKind is JsonValueKind.String
            ? unk.GetString()!
            : "<unk>";
        int unknownId = vocab.TryGetValue(unknown, out int id2) ? id2 : 0;

        int[] byteIds = new int[256];
        if (byteFallback)
        {
            for (int b = 0; b < 256; b++)
            {
                byteIds[b] = vocab.TryGetValue($"<0x{b:X2}>", out int byteId) ? byteId : -1;
            }
        }
        else
        {
            Array.Fill(byteIds, -1);
        }

        return new BpeTokenizer(vocab, idToToken, merges, [.. added], unknownId, byteIds, fuseUnknown);
    }

    /// <summary>Encodes <paramref name="text"/> into token ids.</summary>
    public List<int> Encode(string text)
    {
        var ids = new List<int>(text.Length / 2 + 8);
        EncodeInto(text, ids);
        return ids;
    }

    /// <summary>Appends the encoding of <paramref name="text"/> to <paramref name="ids"/>.</summary>
    public void EncodeInto(string text, List<int> ids)
    {
        int cursor = 0;
        while (cursor < text.Length)
        {
            (int start, AddedToken token) = FindNextAddedToken(text, cursor);
            if (start < 0)
            {
                EncodeSegment(text.AsSpan(cursor), ids);
                return;
            }

            if (start > cursor)
            {
                EncodeSegment(text.AsSpan(cursor, start - cursor), ids);
            }

            ids.Add(token.Id);
            cursor = start + token.Content.Length;
        }
    }

    /// <summary>Looks up the id of a token string, or <c>-1</c> when it is absent.</summary>
    public int TokenToId(string token) => _vocab.TryGetValue(token, out int id) ? id : -1;

    /// <summary>Looks up the literal text of a token id.</summary>
    public string IdToToken(int id) =>
        (uint)id < (uint)_idToToken.Length ? _idToToken[id] ?? string.Empty : string.Empty;

    /// <summary>Whether <paramref name="id"/> is one of the added tokens.</summary>
    public bool IsAddedToken(int id) =>
        (uint)id < (uint)_idToToken.Length
        && _idToToken[id] is { } token
        && _addedByContent.ContainsKey(token);

    /// <summary>
    /// Decodes token ids back to text, running the checkpoint's decoder chain:
    /// <c>Replace("▁" → " ")</c>, then <c>ByteFallback</c>, then <c>Fuse</c>.
    /// </summary>
    /// <param name="ids">Token ids.</param>
    /// <param name="skipSpecialTokens">Whether tokens marked special are dropped.</param>
    public string Decode(IEnumerable<int> ids, bool skipSpecialTokens = true)
    {
        var output = new StringBuilder();
        var pendingBytes = new List<byte>();

        void FlushBytes()
        {
            if (pendingBytes.Count > 0)
            {
                output.Append(Encoding.UTF8.GetString(pendingBytes.ToArray()));
                pendingBytes.Clear();
            }
        }

        foreach (int id in ids)
        {
            string token = IdToToken(id);
            if (token.Length == 0)
            {
                continue;
            }

            if (skipSpecialTokens && _addedByContent.TryGetValue(token, out AddedToken added) && added.Special)
            {
                continue;
            }

            if (TryParseByteToken(token, out byte value))
            {
                pendingBytes.Add(value);
                continue;
            }

            FlushBytes();
            output.Append(token.Replace(SentencePieceSpace, ' '));
        }

        FlushBytes();
        return output.ToString();
    }

    private (int Start, AddedToken Token) FindNextAddedToken(string text, int from)
    {
        int bestStart = -1;
        AddedToken best = default;

        // Added-token contents all start with '<' in this checkpoint, so anchoring the scan on
        // that character keeps the search linear instead of quadratic in the vocabulary size.
        for (int i = from; i < text.Length; i++)
        {
            if (text[i] != '<')
            {
                continue;
            }

            int close = text.IndexOf('>', i + 1);
            while (close >= 0)
            {
                string candidate = text[i..(close + 1)];
                if (_addedByContent.TryGetValue(candidate, out AddedToken token))
                {
                    if (bestStart < 0 || i < bestStart || candidate.Length > best.Content.Length)
                    {
                        bestStart = i;
                        best = token;
                    }

                    break;
                }

                close = text.IndexOf('>', close + 1);
            }

            if (bestStart >= 0)
            {
                return (bestStart, best);
            }
        }

        return (-1, default);
    }

    private void EncodeSegment(ReadOnlySpan<char> segment, List<int> ids)
    {
        if (segment.IsEmpty)
        {
            return;
        }

        // Normalizer: Replace(" " -> "▁").
        Span<char> normalized = segment.Length <= 256 ? stackalloc char[segment.Length] : new char[segment.Length];
        for (int i = 0; i < segment.Length; i++)
        {
            normalized[i] = segment[i] == ' ' ? SentencePieceSpace : segment[i];
        }

        var symbols = new List<int>(normalized.Length + 4);
        BuildInitialSymbols(normalized, symbols);
        ApplyMerges(symbols);

        if (_fuseUnknown)
        {
            int previous = -1;
            foreach (int id in symbols)
            {
                if (id == _unknownId && previous == _unknownId)
                {
                    continue;
                }

                ids.Add(id);
                previous = id;
            }
        }
        else
        {
            ids.AddRange(symbols);
        }
    }

    private void BuildInitialSymbols(ReadOnlySpan<char> normalized, List<int> symbols)
    {
        Span<byte> bytes = stackalloc byte[4];
        int i = 0;

        while (i < normalized.Length)
        {
            int length = char.IsHighSurrogate(normalized[i]) && i + 1 < normalized.Length ? 2 : 1;
            string character = new(normalized.Slice(i, length));
            i += length;

            if (_vocab.TryGetValue(character, out int id))
            {
                symbols.Add(id);
                continue;
            }

            // byte_fallback: emit one <0xXX> token per UTF-8 byte of the unknown character.
            int written = Encoding.UTF8.GetBytes(character, bytes);
            bool complete = written > 0;
            for (int b = 0; b < written && complete; b++)
            {
                complete = _byteFallbackIds[bytes[b]] >= 0;
            }

            if (complete)
            {
                for (int b = 0; b < written; b++)
                {
                    symbols.Add(_byteFallbackIds[bytes[b]]);
                }
            }
            else
            {
                symbols.Add(_unknownId);
            }
        }
    }

    /// <summary>
    /// Applies merges lowest-rank first.
    /// </summary>
    /// <remarks>
    /// A doubly-linked list plus a priority queue reproduces the <c>tokenizers</c> crate's
    /// algorithm in <c>O(n log n)</c>. Queue entries are lazily invalidated: a popped candidate is
    /// skipped when either endpoint has since been merged away, which is cheaper than removing
    /// stale entries eagerly.
    /// </remarks>
    private void ApplyMerges(List<int> symbols)
    {
        int count = symbols.Count;
        if (count < 2)
        {
            return;
        }

        int[] ids = TensorPool.RentInts(count);
        int[] previous = TensorPool.RentInts(count);
        int[] next = TensorPool.RentInts(count);
        bool[] removed = new bool[count];

        try
        {
            for (int i = 0; i < count; i++)
            {
                ids[i] = symbols[i];
                previous[i] = i - 1;
                next[i] = i + 1 < count ? i + 1 : -1;
            }

            // Priority packs (rank, position) so equal-rank candidates fire left to right, which
            // is how the `tokenizers` crate breaks ties. Without the position tie-break a run of
            // identical symbols merges in an arbitrary order and produces different tokens.
            var queue = new PriorityQueue<(int Left, int Merged), long>();
            for (int i = 0; i + 1 < count; i++)
            {
                if (_merges.TryGetValue((ids[i], ids[i + 1]), out (int Rank, int Id) merge))
                {
                    queue.Enqueue((i, merge.Id), Priority(merge.Rank, i));
                }
            }

            while (queue.TryDequeue(out (int Left, int Merged) candidate, out _))
            {
                (int left, int merged) = candidate;
                if (removed[left])
                {
                    continue;
                }

                int right = next[left];
                if (right < 0)
                {
                    continue;
                }

                // The queue holds stale entries: an earlier merge may have changed what sits at
                // `left` or `right` since this candidate was pushed. Re-resolving the pair and
                // checking it still produces the same token is how the `tokenizers` crate
                // invalidates them, and skipping it merges the wrong symbols together.
                if (!_merges.TryGetValue((ids[left], ids[right]), out (int Rank, int Id) current)
                    || current.Id != merged)
                {
                    continue;
                }

                ids[left] = merged;
                removed[right] = true;

                int after = next[right];
                next[left] = after;
                if (after >= 0)
                {
                    previous[after] = left;
                }

                int before = previous[left];
                if (before >= 0 && _merges.TryGetValue((ids[before], merged), out (int Rank, int Id) leftMerge))
                {
                    queue.Enqueue((before, leftMerge.Id), Priority(leftMerge.Rank, before));
                }

                if (after >= 0 && _merges.TryGetValue((merged, ids[after]), out (int Rank, int Id) rightMerge))
                {
                    queue.Enqueue((left, rightMerge.Id), Priority(rightMerge.Rank, left));
                }
            }

            symbols.Clear();
            for (int i = 0; i >= 0; i = next[i])
            {
                symbols.Add(ids[i]);
            }
        }
        finally
        {
            TensorPool.ReturnInts(ids);
            TensorPool.ReturnInts(previous);
            TensorPool.ReturnInts(next);
        }
    }

    /// <summary>Packs a merge rank and a position into one ascending priority.</summary>
    private static long Priority(int rank, int position) => ((long)rank << 32) | (uint)position;

    private static bool TryParseByteToken(string token, out byte value)
    {
        value = 0;
        if (token.Length != 6 || token[0] != '<' || token[1] != '0' || token[2] != 'x' || token[5] != '>')
        {
            return false;
        }

        return byte.TryParse(token.AsSpan(3, 2), System.Globalization.NumberStyles.HexNumber, null, out value);
    }

    private static void ValidateNormalizer(JsonElement root)
    {
        if (!root.TryGetProperty("normalizer", out JsonElement normalizer)
            || normalizer.ValueKind is JsonValueKind.Null)
        {
            return;
        }

        foreach (JsonElement step in Flatten(normalizer, "normalizers"))
        {
            string type = step.GetProperty("type").GetString() ?? string.Empty;
            bool isSpaceReplace = type == "Replace"
                && step.GetProperty("pattern").TryGetProperty("String", out JsonElement pattern)
                && pattern.GetString() == " "
                && step.GetProperty("content").GetString() == "▁";

            if (!isSpaceReplace)
            {
                throw new NotSupportedException($"Unsupported normalizer step '{type}'.");
            }
        }
    }

    private static void ValidatePreTokenizer(JsonElement root)
    {
        if (root.TryGetProperty("pre_tokenizer", out JsonElement preTokenizer)
            && preTokenizer.ValueKind is not JsonValueKind.Null)
        {
            throw new NotSupportedException("Pre-tokenizers are not supported by this implementation.");
        }
    }

    private static void ValidateDecoder(JsonElement root)
    {
        if (!root.TryGetProperty("decoder", out JsonElement decoder) || decoder.ValueKind is JsonValueKind.Null)
        {
            return;
        }

        foreach (JsonElement step in Flatten(decoder, "decoders"))
        {
            string type = step.GetProperty("type").GetString() ?? string.Empty;
            switch (type)
            {
                case "ByteFallback":
                case "Fuse":
                    break;
                case "Replace":
                    if (!step.GetProperty("pattern").TryGetProperty("String", out JsonElement pattern)
                        || pattern.GetString() != "▁"
                        || step.GetProperty("content").GetString() != " ")
                    {
                        throw new NotSupportedException("Unsupported Replace decoder step.");
                    }

                    break;
                default:
                    throw new NotSupportedException($"Unsupported decoder step '{type}'.");
            }
        }
    }

    private static IEnumerable<JsonElement> Flatten(JsonElement element, string sequenceProperty)
    {
        if (element.GetProperty("type").GetString() == "Sequence"
            && element.TryGetProperty(sequenceProperty, out JsonElement steps))
        {
            foreach (JsonElement step in steps.EnumerateArray())
            {
                yield return step;
            }
        }
        else
        {
            yield return element;
        }
    }
}
