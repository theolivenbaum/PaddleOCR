using PaddleOcrSharp.Formats;

namespace PaddleOcrSharp.Tests.Fixtures;

/// <summary>
/// Locates the reference tensors produced by <c>dotnet/tools/reference/dump_*.py</c>.
/// </summary>
/// <remarks>
/// Fixtures are large and generated on demand, so they are not committed. Tests that need one
/// are skipped rather than failed when it is missing, which keeps <c>dotnet test</c> useful on a
/// clean clone.
/// </remarks>
public static class Fixture
{
    private static readonly Lazy<string?> RootPath = new(FindRoot);

    /// <summary>Directory holding the <c>.npz</c> fixtures, or <see langword="null"/> if absent.</summary>
    public static string? Root => RootPath.Value;

    /// <summary>Whether <paramref name="name"/> is present.</summary>
    public static bool Exists(string name) => Root is not null && File.Exists(Path.Combine(Root, name));

    /// <summary>Loads a <c>.npz</c> fixture by file name.</summary>
    public static Dictionary<string, NpyArray> Load(string name)
    {
        RequireOrSkip(name);
        return NpyFile.ReadArchive(Path.Combine(Root!, name));
    }

    /// <summary>Skips the calling test when <paramref name="name"/> has not been generated.</summary>
    public static void RequireOrSkip(string name)
    {
        if (!Exists(name))
        {
            Assert.Skip($"Fixture '{name}' not generated; run dotnet/tools/reference/dump_*.py.");
        }
    }

    private static string? FindRoot()
    {
        string? environment = Environment.GetEnvironmentVariable("PADDLEOCR_SHARP_FIXTURES");
        if (!string.IsNullOrEmpty(environment) && Directory.Exists(environment))
        {
            return environment;
        }

        DirectoryInfo? directory = new(AppContext.BaseDirectory);
        while (directory is not null)
        {
            string candidate = Path.Combine(directory.FullName, "artifacts", "fixtures");
            if (Directory.Exists(candidate))
            {
                return candidate;
            }

            directory = directory.Parent;
        }

        return null;
    }
}
