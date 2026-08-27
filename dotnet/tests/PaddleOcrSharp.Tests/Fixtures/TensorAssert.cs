namespace PaddleOcrSharp.Tests.Fixtures;

/// <summary>Numeric comparison helpers with diagnostics that identify the worst element.</summary>
public static class TensorAssert
{
    /// <summary>
    /// Asserts every element of <paramref name="actual"/> is within
    /// <paramref name="absoluteTolerance"/> + <paramref name="relativeTolerance"/> · |expected|
    /// of <paramref name="expected"/>.
    /// </summary>
    public static void Close(
        ReadOnlySpan<float> expected,
        ReadOnlySpan<float> actual,
        double absoluteTolerance = 1e-5,
        double relativeTolerance = 0.0,
        string? because = null)
    {
        Assert.Equal(expected.Length, actual.Length);

        double worstError = 0;
        int worstIndex = -1;

        for (int i = 0; i < expected.Length; i++)
        {
            double e = expected[i];
            double a = actual[i];

            if (double.IsNaN(e) != double.IsNaN(a))
            {
                Assert.Fail($"NaN mismatch at [{i}]: expected {e}, actual {a}. {because}");
            }

            double allowed = absoluteTolerance + (relativeTolerance * Math.Abs(e));
            double error = Math.Abs(e - a) - allowed;
            if (error > worstError)
            {
                worstError = error;
                worstIndex = i;
            }
        }

        if (worstIndex >= 0)
        {
            Assert.Fail(
                $"Values differ at [{worstIndex}]: expected {expected[worstIndex]}, " +
                $"actual {actual[worstIndex]}, excess over tolerance {worstError:G6}. {because}");
        }
    }

    /// <summary>Mean absolute difference, for reporting rather than asserting.</summary>
    public static double MeanAbsoluteDifference(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        double sum = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            sum += Math.Abs((double)expected[i] - actual[i]);
        }

        return expected.Length == 0 ? 0 : sum / expected.Length;
    }

    /// <summary>Asserts two integer sequences are identical.</summary>
    public static void Equal(ReadOnlySpan<long> expected, ReadOnlySpan<long> actual, string? because = null)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] != actual[i])
            {
                Assert.Fail($"Values differ at [{i}]: expected {expected[i]}, actual {actual[i]}. {because}");
            }
        }
    }
}
