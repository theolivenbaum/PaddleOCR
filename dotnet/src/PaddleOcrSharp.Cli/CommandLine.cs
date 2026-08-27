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

                bool hasValue = i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal);
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

    /// <summary>Option value parsed as a boolean; a bare flag counts as <see langword="true"/>.</summary>
    public bool GetBool(string name, bool fallback) =>
        _options.TryGetValue(name, out string? value)
            ? value is "true" or "1" or "yes"
            : fallback;
}
