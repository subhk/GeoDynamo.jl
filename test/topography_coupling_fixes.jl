using Test
using GeoDynamo
using MPI

MPI.Initialized() || MPI.Init()

const TopoFix = GeoDynamo.bcs.topography

# Fixes for two latent topography-coupling correctness bugs found in the audit:
#  (1) lm_to_index assumed a FULL (l,m) triangle (idx = l(l+1)/2 + m + 1), which
#      disagrees with the SEQUENTIAL storage packing (m in 0:min(l,mmax)) whenever
#      mmax < lmax → wrong coefficient read, or out-of-bounds for high-l modes.
#  (2) rms_amplitude was the raw coefficient sum-of-squares, omitting the 1/4π
#      sphere-average factor and the m>0 doubling (±m share one stored slot).

@testset "topography lm_to_index matches sequential packing (truncated mmax)" begin
    lmax = 4
    mmax = 1

    # Sequential storage order: for l, for m in 0:min(l,mmax), incrementing idx.
    seq = Dict{Tuple{Int, Int}, Int}()
    idx = 0
    for l in 0:lmax, m in 0:min(l, mmax)
        idx += 1
        seq[(l, m)] = idx
    end

    for ((l, m), e) in seq
        @test TopoFix.lm_to_index(l, m, lmax, mmax) == e
    end

    # Full-triangle default (mmax = lmax) must stay the classic formula.
    for l in 0:lmax, m in 0:l
        @test TopoFix.lm_to_index(l, m, lmax, lmax) == l * (l + 1) ÷ 2 + m + 1
    end
end

@testset "topography get_coefficient round-trips for truncated mmax (no OOB)" begin
    lmax = 4
    mmax = 1
    fld = TopoFix.TopographyField(lmax, mmax, 0.35, TopoFix.INNER_BOUNDARY)

    coeffs = ComplexF64[complex(Float64(i), 0.0) for i in 1:fld.nlm]
    TopoFix.set_topography_coefficients!(fld, coeffs)

    # Every (l,m) must read back the value stored in its SEQUENTIAL slot.
    idx = 0
    for l in 0:lmax, m in 0:min(l, mmax)
        idx += 1
        @test TopoFix.get_coefficient(fld, l, m) ≈ coeffs[idx]
    end

    # (4,0) is a valid mode under mmax=1 and must not index out of bounds.
    @test isfinite(abs(TopoFix.get_coefficient(fld, 4, 0)))
end

@testset "topography rms_amplitude: sphere-average + m>0 doubling" begin
    lmax = 1
    mmax = 1
    A = 0.4

    f0 = TopoFix.TopographyField(lmax, mmax, 0.35, TopoFix.INNER_BOUNDARY)
    c0 = zeros(ComplexF64, f0.nlm)
    c0[TopoFix.lm_to_index(0, 0, lmax, mmax)] = A
    TopoFix.set_topography_coefficients!(f0, c0)
    TopoFix.update_topography_statistics!(f0)
    @test f0.rms_amplitude ≈ A / sqrt(4pi)

    f1 = TopoFix.TopographyField(lmax, mmax, 0.35, TopoFix.INNER_BOUNDARY)
    c1 = zeros(ComplexF64, f1.nlm)
    c1[TopoFix.lm_to_index(1, 1, lmax, mmax)] = A
    TopoFix.set_topography_coefficients!(f1, c1)
    TopoFix.update_topography_statistics!(f1)
    @test f1.rms_amplitude ≈ A * sqrt(2 / (4pi))   # m>0 ⇒ counted twice
end

@testset "Stefan target loop iterates m ≥ 0 only (no ±m double-count)" begin
    # The OUTER target-mode loop (lowercase l,m) must iterate m ≥ 0, matching the
    # thermal/velocity/magnetic coupling siblings. The bug `for m in -l:l` wrote
    # both +m and -m into the SAME abs(m) storage slot ⇒ doubled every m≠0
    # correction. (The inner SOURCE loop `for M in -L:L` legitimately spans ±M.)
    src = read(joinpath(@__DIR__, "..", "src", "bcs", "topography", "stefan_condition.jl"), String)
    @test !occursin("for m in -l:l", src)
    @test occursin("for m in 0:min(l, mmax)", src)
end
