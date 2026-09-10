namespace PaddleOcrSharp.Cli;

/// <summary>
/// A minimal option parser: the CLI has a handful of verbs and no need for a dependency.
/// </summary>
public sealed class CommandLine
{
    private readonly Dictionary<string, string> _options = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<string> _positional = [];

    private CommandLine()
    {
    }

    /// <summary>The verb, or an empty string when none was given.</summary>
    public string Verb { get; private set; } = string.Empty;

    /// <summary>Arguments that are not options.</summary>
    public IReadOnlyList<string> Positional => _positional;

    /// <summary>
    /// Parses <paramref name="args"/>. The first non-option argument becomes <see cref="Verb"/>;
    /// <c>--name value</c> and <c>--flag</c> both become options.
    /// </summary>
    public static CommandLine Parse(string[] args)
    {
        var result = new CommandLine();

        for (int i = 0; i < args.Length; i++)
        {
            string argument = args[i];
            if (argument.StartsWith("--", StringComparison.Ordinal))
            {
                string name = argument[2..];
                int equals = name.IndexOf('=');
                if (equals >= 0)
                {
                    result._options[name[..equals]] = name[(equals + 1)..];
                    continue;
                }

                // A switch takes the following token only when that token is a boolean literal:
                // `--stop-on-repetition false` and `--calibrate false` have to keep working, but
                // `--profile page.pdf` must not eat the path — which is what a greedy rule did,
                // leaving the run to fail with "parse needs at least one image path".
                bool hasValue = i + 1 < args.Length
                    && !args[i + 1].StartsWith("--", StringComparison.Ordinal)
                    && !(IsSwitch(name) && !IsBooleanLiteral(args[i + 1]));

                result._options[name] = hasValue ? args[++i] : "true";
                continue;
            }

            if (result.Verb.Length == 0)
            {
                result.Verb = argument;
            }
            else
            {
                result._positional.Add(argument);
            }
        }

        return result;
    }

    /// <summary>
    /// Options that mean something on their own. Everything else needs a value, so the token
    /// after it is taken whatever it looks like.
    /// </summary>
    private static readonly HashSet<string> Switches = new(StringComparer.OrdinalIgnoreCase)
    {
        "profile",
        "layout-profile",
        "no-layout",
        "no-vl",
        "chart",
        "seal",
        "ocr-images",
        "doc-orientation",
        "doc-unwarping",
        "format-block-content",
        "merge-tables",
        "title-levels",
        "stop-on-repetition",
        "calibrate",
        "gemm",
        "help",
    };

    private static bool IsSwitch(string name) =>
        Switches.Contains(name)
        || (name.StartsWith("no-", StringComparison.OrdinalIgnoreCase) && Switches.Contains(name[3..]));

    private static bool IsBooleanLiteral(string value) =>
        value.Equals("true", StringComparison.OrdinalIgnoreCase)
        || value.Equals("false", StringComparison.OrdinalIgnoreCase);

    /// <summary>Whether <paramref name="name"/> was given.</summary>
    public bool Has(string name) => _options.ContainsKey(name);

    /// <summary>Option value, or <paramref name="fallback"/>.</summary>
    public string? Get(string name, string? fallback = null) =>
        _options.TryGetValue(name, out string? value) ? value : fallback;

    /// <summary>Option value parsed as an integer.</summary>
    public int GetInt(string name, int fallback) =>
        _options.TryGetValue(name, out string? value) && int.TryParse(value, out int parsed) ? parsed : fallback;

    /// <summary>Option value parsed as a float.</summary>
    public float GetFloat(string name, float fallback) =>
        _options.TryGetValue(name, out string? value) && float.TryParse(value, out float parsed) ? parsed : fallback;

    /// <summary>
    /// Option value parsed as a boolean; a bare flag counts as <see langword="true"/>, and
    /// <c>--no-name</c> turns a default-on switch off.
    /// </summary>
    public bool GetBool(string name, bool fallback)
    {
        if (_options.TryGetValue(name, out string? value))
        {
            return IsTrue(value);
        }

        return _options.TryGetValue("no-" + name, out string? negated) ? !IsTrue(negated) : fallback;
    }

    private static bool IsTrue(string value) => value is "true" or "1" or "yes" or "on";
}
