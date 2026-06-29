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

@testset "get_spectral_radial_derivative gathers m>0 modes (not just l-slot axis)" begin
    # The gather was gated by `idx in field.pencil.axes_local[1]`, i.e. the l-slot
    # axis (1:lmax+1), used as if it were a mode-index range. Any canonical mode
    # index > lmax+1 (every m>0 mode) was skipped ⇒ profile stayed zero ⇒ the
    # derivative came back exactly 0. An m=0 mode (idx ≤ lmax+1) was unaffected, so
    # an m>0 mode and an m=0 mode of the SAME degree l, holding the SAME radial
    # profile, must yield the SAME (nonzero) radial derivative.
    lmax = 2; mmax = 2; nr = 8
    cfg = GeoDynamo.create_shtnskit_config(lmax = lmax, mmax = mmax,
        nlat = 8, nlon = 12, nr = nr)
    dom = GeoDynamo.create_radial_domain(nr)
    field = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    ∂r = GeoDynamo.create_derivative_matrix(Float64, 1, dom)

    # Same non-constant radial profile in the (2,0) [m=0, idx≤lmax+1] and
    # (2,2) [m>0, idx>lmax+1] storage slots.
    real3 = parent(field.data_real)
    for (l, m) in ((2, 0), (2, 2))
        idx = TopoFix.lm_to_spectral_index(l, m, cfg)
        slot = GeoDynamo.local_spectral_storage_slot(cfg, idx)
        for k in 1:nr
            real3[slot[1], slot[2], k] = Float64(k)
        end
    end

    d_m0 = TopoFix.get_spectral_radial_derivative(field, 2, 0, 0.0,
        TopoFix.OUTER_BOUNDARY; ∂r = ∂r, domain = dom)
    d_m2 = TopoFix.get_spectral_radial_derivative(field, 2, 2, 0.0,
        TopoFix.OUTER_BOUNDARY; ∂r = ∂r, domain = dom)

    @test abs(d_m0) > 0
    @test d_m2 ≈ d_m0
end
