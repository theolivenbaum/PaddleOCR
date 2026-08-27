using System.Text.Json;
using PaddleOcrSharp.Text;
using PaddleOcrSharp.Tests.Fixtures;

namespace PaddleOcrSharp.Tests.Text;

/// <summary>
/// Compares <see cref="BpeTokenizer"/> against the Hugging Face tokenizer on a mixed-script
/// corpus. Fixtures come from <c>dotnet/tools/reference/dump_tokenizer.py</c>.
/// </summary>
public class TokenizerParityTests
{
    private const string FixtureName = "tokenizer.json";

    private sealed record Case(string Text, int[] Ids, string Decoded);

    [Fact]
    public void EncodingMatchesHuggingFace()
    {
        (BpeTokenizer tokenizer, Case[] cases) = Load();

        var failures = new List<string>();
        foreach (Case testCase in cases)
        {
            int[] actual = [.. tokenizer.Encode(testCase.Text)];
            if (!actual.SequenceEqual(testCase.Ids))
            {
                failures.Add(
                    $"'{Summarise(testCase.Text)}'\n  expected: [{string.Join(", ", testCase.Ids)}]\n" +
                    $"  actual:   [{string.Join(", ", actual)}]");
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public void DecodingRoundTripsEveryCase()
    {
        (BpeTokenizer tokenizer, Case[] cases) = Load();

        var failures = new List<string>();
        foreach (Case testCase in cases)
        {
            string decoded = tokenizer.Decode(testCase.Ids, skipSpecialTokens: false);
            if (decoded != testCase.Decoded)
            {
                failures.Add($"'{Summarise(testCase.Text)}'\n  expected: '{testCase.Decoded}'\n  actual:   '{decoded}'");
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public void SpecialTokenIdsMatchTheCheckpoint()
    {
        (BpeTokenizer tokenizer, _) = Load();

        Assert.Equal(100_295, tokenizer.TokenToId("<|IMAGE_PLACEHOLDER|>"));
        Assert.Equal(101_305, tokenizer.TokenToId("<|IMAGE_START|>"));
        Assert.Equal(101_306, tokenizer.TokenToId("<|IMAGE_END|>"));
        Assert.Equal(2, tokenizer.TokenToId("</s>"));
        Assert.Equal(0, tokenizer.TokenToId("<unk>"));
    }

    private static (BpeTokenizer Tokenizer, Case[] Cases) Load()
    {
        Fixture.RequireOrSkip(FixtureName);
        CheckpointFixture.RequireTokenizerOrSkip();

        var tokenizer = BpeTokenizer.FromFile(
            Path.Combine(CheckpointFixture.Directory!, "tokenizer.json"));

        string json = File.ReadAllText(Path.Combine(Fixture.Root!, FixtureName));
        using JsonDocument document = JsonDocument.Parse(json);

        var cases = new List<Case>();
        foreach (JsonElement entry in document.RootElement.EnumerateArray())
        {
            int[] ids = [.. entry.GetProperty("ids").EnumerateArray().Select(value => value.GetInt32())];
            cases.Add(new Case(
                entry.GetProperty("text").GetString()!,
                ids,
                entry.GetProperty("decoded").GetString()!));
        }

        return (tokenizer, [.. cases]);
    }

    private static string Summarise(string text) =>
        text.Length <= 60 ? text.Replace("\n", "\\n") : text[..60].Replace("\n", "\\n") + "…";
}
