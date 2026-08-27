namespace PaddleOcrSharp.Tests.Fixtures;

/// <summary>Locates the PP-DocLayoutV3 inference program.</summary>
public static class LayoutModelFixture
{
    /// <summary>Directory holding <c>inference.json</c>, or <see langword="null"/>.</summary>
    public static string? Directory
    {
        get
        {
            string? configured = Environment.GetEnvironmentVariable("PP_DOCLAYOUT_V3_DIR");
            if (!string.IsNullOrEmpty(configured) && File.Exists(Path.Combine(configured, "inference.json")))
            {
                return configured;
            }

            const string Fallback = "/home/user/ref/layout";
            return File.Exists(Path.Combine(Fallback, "inference.json")) ? Fallback : null;
        }
    }

    /// <summary>Skips the calling test when the model is not present.</summary>
    public static void RequireOrSkip()
    {
        if (Directory is null)
        {
            Assert.Skip("PP-DocLayoutV3 model not found; set PP_DOCLAYOUT_V3_DIR to a download.");
        }
    }
}
