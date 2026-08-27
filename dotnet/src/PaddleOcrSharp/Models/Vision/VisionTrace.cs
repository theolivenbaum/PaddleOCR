namespace PaddleOcrSharp.Models.Vision;

/// <summary>
/// Collects intermediate activations so a divergence from the Python reference can be traced to
/// the layer that introduced it.
/// </summary>
/// <remarks>
/// Tracing is opt-in and off in production paths: the recorded copies are full-size activations.
/// </remarks>
public sealed class VisionTrace
{
    private readonly Dictionary<string, (float[] Values, int Rows, int Cols)> _stages = [];

    /// <summary>Stage names, in the order they were recorded.</summary>
    public IReadOnlyCollection<string> Stages => _stages.Keys;

    /// <summary>Whether <paramref name="stage"/> was recorded.</summary>
    public bool Contains(string stage) => _stages.ContainsKey(stage);

    /// <summary>Records a copy of <paramref name="values"/> under <paramref name="stage"/>.</summary>
    public void Record(string stage, ReadOnlySpan<float> values, int rows, int cols) =>
        _stages[stage] = (values[..(rows * cols)].ToArray(), rows, cols);

    /// <summary>Reads back a recorded stage.</summary>
    public (float[] Values, int Rows, int Cols) Get(string stage) => _stages[stage];
}
