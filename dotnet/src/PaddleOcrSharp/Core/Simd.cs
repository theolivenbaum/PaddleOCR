using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace PaddleOcrSharp.Core;

/// <summary>
/// Which vector width the kernels use.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="Vector512.IsHardwareAccelerated"/> does not report whether AVX-512 exists. It
/// reports the runtime's <c>PreferredVectorBitWidth</c> policy, which defaults to 256 bits even
/// on hardware with full AVX-512, because 512-bit code drops the core's clock on several Intel
/// generations. The ISA is still reachable: an explicit <see cref="Vector512"/> call compiles to
/// real <c>zmm</c> instructions whenever <see cref="Avx512F"/> reports support, and the JIT
/// enregisters the locals either way. So a kernel is free to ignore the policy.
/// </para>
/// <para>
/// Ignoring it turns out to be the wrong call on the machine this port is measured on, which is
/// worth recording because the microbenchmark says the opposite. A dependency-free FMA loop runs
/// at 101 GFLOP/s through the 256-bit path and 163 through the 512-bit one. The GEMM does not:
/// every 512-bit variant tried came out behind the 256-bit kernel — the same four-by-four tile at
/// 202 GFLOP/s against 218, and a narrower four-by-two tile, chosen to fit the register file,
/// worse again at 135. The microbenchmark has no memory traffic and one thread; the GEMM has both,
/// and whatever the wider vectors gain per instruction they give back through the clock the
/// package runs at with four threads issuing them.
/// </para>
/// <para>
/// So the runtime's conservative default is followed rather than overridden, and the override is
/// left to whoever measures their own machine: <c>PADDLEOCR_SHARP_VECTOR_BITS=512</c> forces the
/// wide kernels on, <c>=256</c> forces them off. <c>paddleocr-sharp bench --gemm true</c> prints
/// both the machine's FMA ceiling and what the kernels achieve against it, which is what the
/// decision should rest on.
/// </para>
/// </remarks>
internal static class Simd
{
    /// <summary>Whether the 512-bit kernels should run.</summary>
    internal static readonly bool Use512 = Resolve512();

    /// <summary>Whether the 256-bit kernels should run.</summary>
    internal static readonly bool Use256 = Resolve256();

    private static int? RequestedBits()
    {
        string? setting = Environment.GetEnvironmentVariable("PADDLEOCR_SHARP_VECTOR_BITS");
        return int.TryParse(setting, out int bits) ? bits : null;
    }

    private static bool Resolve512() =>
        RequestedBits() is { } bits
            ? bits >= 512 && Avx512F.IsSupported && Avx512BW.IsSupported
            : Vector512.IsHardwareAccelerated;

    private static bool Resolve256() =>
        RequestedBits() is { } bits
            ? bits >= 256 && (Vector256.IsHardwareAccelerated || Avx2.IsSupported)
            : Vector256.IsHardwareAccelerated;
}
