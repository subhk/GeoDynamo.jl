using Test
using GeoDynamo

# Process-grid balance detection: the spectral pencil splits l (over r_ranks) and m
# (over θ_ranks) into CONTIGUOUS slot blocks, but triangular SH truncation makes the
# mode COUNT per block uneven — and a square grid can leave whole ranks with zero
# modes (the high-m / low-l corner has no m<=l pairs). spectral_mode_counts reports
# the per-rank mode count so setup can warn about idle ranks / imbalance.

@testset "spectral_mode_counts: idle ranks + imbalance detection" begin
    lmax = mmax = 85
    nlm = (lmax + 1) * (lmax + 2) ÷ 2   # = 3741 for mmax == lmax

    # Serial: one rank owns every mode.
    @test GeoDynamo.spectral_mode_counts(1, 1, lmax, mmax) == [nlm]

    # 2x2: the high-m / low-l corner rank owns ZERO modes.
    c22 = GeoDynamo.spectral_mode_counts(2, 2, lmax, mmax)
    @test length(c22) == 4
    @test sum(c22) == nlm
    @test minimum(c22) == 0

    # 4x1: no idle rank, but a substantial (>1.5x) mode-load imbalance.
    c41 = GeoDynamo.spectral_mode_counts(4, 1, lmax, mmax)
    @test length(c41) == 4
    @test sum(c41) == nlm
    @test minimum(c41) > 0
    @test maximum(c41) / (sum(c41) / 4) > 1.5

    # 8x1: still no idle rank, larger imbalance.
    c81 = GeoDynamo.spectral_mode_counts(8, 1, lmax, mmax)
    @test sum(c81) == nlm
    @test minimum(c81) > 0
end
