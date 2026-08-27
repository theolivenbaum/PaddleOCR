namespace PaddleOcrSharp.Tests.Fixtures;

/// <summary>Locates the two document pre-processing inference programs.</summary>
public static class PreprocessingModelFixture
{
    /// <summary>Directory holding PP-LCNet_x1_0_doc_ori, or <see langword="null"/>.</summary>
    public static string? OrientationDirectory => Resolve("PP_LCNET_DOC_ORI_DIR", "/home/user/ref/docori");

    /// <summary>Directory holding UVDoc, or <see langword="null"/>.</summary>
    public static string? UnwarpDirectory => Resolve("UVDOC_DIR", "/home/user/ref/uvdoc");

    /// <summary>Skips the calling test when the orientation classifier is not present.</summary>
    public static void RequireOrientationOrSkip()
    {
        if (OrientationDirectory is null)
        {
            Assert.Skip("PP-LCNet_x1_0_doc_ori not found; set PP_LCNET_DOC_ORI_DIR to a download.");
        }
    }

    /// <summary>Skips the calling test when UVDoc is not present.</summary>
    public static void RequireUnwarpOrSkip()
    {
        if (UnwarpDirectory is null)
        {
            Assert.Skip("UVDoc not found; set UVDOC_DIR to a download.");
        }
    }

    private static string? Resolve(string variable, string fallback)
    {
        string? configured = Environment.GetEnvironmentVariable(variable);
        if (!string.IsNullOrEmpty(configured) && File.Exists(Path.Combine(configured, "inference.json")))
        {
            return configured;
        }

        return File.Exists(Path.Combine(fallback, "inference.json")) ? fallback : null;
    }
}
