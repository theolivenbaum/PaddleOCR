namespace PaddleOcrSharp.Text;

/// <summary>A token that bypasses BPE and is matched literally in the input.</summary>
/// <param name="Id">Vocabulary id.</param>
/// <param name="Content">Literal text.</param>
/// <param name="Special">Whether the token is marked special (skipped by <c>skip_special_tokens</c>).</param>
/// <param name="LeftStrip">Whether whitespace to the left is absorbed.</param>
/// <param name="RightStrip">Whether whitespace to the right is absorbed.</param>
/// <param name="Normalized">Whether the surrounding normalizer is applied to the token's content.</param>
public readonly record struct AddedToken(
    int Id,
    string Content,
    bool Special,
    bool LeftStrip,
    bool RightStrip,
    bool Normalized);
