namespace PaddleOcrSharp.Pipeline;

/// <summary>
/// Numbers a page's blocks by their place in its reading flow.
/// </summary>
/// <remarks>
/// Port of <c>update_order_index</c>. The labels that sit out are the ones that are not prose —
/// running heads, captions, figures, tables, marginalia — together with whatever the markdown
/// settings already exclude. They keep their position in the block list and simply have no
/// number, which is how a consumer can tell the page's text flow from everything around it.
/// </remarks>
public static class BlockOrder
{
    /// <summary>
    /// Returns <paramref name="blocks"/> with <see cref="ParsedBlock.Order"/> assigned.
    /// </summary>
    /// <param name="blocks">The page's blocks, already in reading order.</param>
    /// <param name="alsoSkipped">
    /// Labels to exclude on top of <see cref="BlockLabels.SkipOrder"/>; normally the markdown
    /// settings' ignored labels, which is what upstream unions in.
    /// </param>
    public static IReadOnlyList<ParsedBlock> Assign(
        IReadOnlyList<ParsedBlock> blocks,
        IEnumerable<string>? alsoSkipped = null)
    {
        var skipped = new HashSet<string>(BlockLabels.SkipOrder, StringComparer.Ordinal);
        if (alsoSkipped is not null)
        {
            skipped.UnionWith(alsoSkipped);
        }

        var numbered = new ParsedBlock[blocks.Count];
        int order = 1;

        for (int i = 0; i < blocks.Count; i++)
        {
            numbered[i] = skipped.Contains(blocks[i].Label)
                ? blocks[i] with { Order = null }
                : blocks[i] with { Order = order++ };
        }

        return numbered;
    }
}
