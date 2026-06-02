using Test, GeoDynamo, SHTnsKit, MPI, PencilArrays

const FINALIZE_MPI_P3_TRANSPOSE = !MPI.Initialized()
FINALIZE_MPI_P3_TRANSPOSE && MPI.Init()

@testset "Alm <-> spec_solve identity (dealiased)" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=24, nr=8)  # dealiased nlon=24
    plan = GeoDynamo.get_disttranspose_plan(cfg)
    Alm = SHTnsKit.allocate_spectral(plan)
    p = parent(Alm)
    for i in eachindex(p)
        p[i] = ComplexF64(i + 100 * (MPI.Comm_rank(MPI.COMM_WORLD) + 1))
    end
    a0 = copy(p)
    solve = GeoDynamo.to_spec_solve(cfg, Alm, plan)
    GeoDynamo.from_spec_solve!(cfg, Alm, solve, plan)
    @test parent(Alm) == a0
end

# ===========================================================================
# Scalar transform via DistTransposePlan (Phase 3 Task 2)
# ===========================================================================
#
# Gate 1 (roundtrip): spec → phys → spec must preserve the (l,m,r) coefficients
#   to < 1e-10 (gathered over COMM_WORLD), at np=1 and every r×θ grid.
# Gate 2 (np-consistency / partition-neutrality): the global temperature
#   spectral state after one roundtrip must be IDENTICAL regardless of MPI grid
#   (it equals the deterministic seed exactly — the roundtrip is the identity).
#
# Seed only physically valid modes: l ≥ m, m ≤ mmax, and m=0 ⇒ imag = 0.
# Values depend only on the GLOBAL (l,m,r) indices so they are grid-independent.

# Deterministic, grid-independent seed value for global indices (l,m,r).
_p3_scalar_seed_real(l, m, r) = sinpi(0.1 * (l + 1) + 0.07 * m - 0.013 * r)
_p3_scalar_seed_imag(l, m, r) = m == 0 ? 0.0 : cospi(0.05 * l - 0.09 * (m + 1) + 0.017 * r)

# Seed the field's spectral storage from the deterministic global seed.
function _p3_seed_scalar!(spec, cfg)
    sr = parent(spec.data_real)
    si = parent(spec.data_imag)
    fill!(sr, 0.0)
    fill!(si, 0.0)
    r_range = GeoDynamo.range_local(cfg.pencils.spec, 3)
    for lm_idx in GeoDynamo.local_spectral_mode_indices(cfg)
        lm_idx <= cfg.nlm || continue
        l = cfg.l_values[lm_idx]
        m = cfg.m_values[lm_idx]
        (0 <= m <= cfg.mmax && m <= l <= cfg.lmax) || continue
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_idx)
        slot === nothing && continue
        for r_idx in r_range
            lr = r_idx - first(r_range) + 1
            lr > size(sr, 3) && continue
            GeoDynamo.set_local_spectral_value!(sr, slot, lr, _p3_scalar_seed_real(l, m, r_idx))
            GeoDynamo.set_local_spectral_value!(si, slot, lr, _p3_scalar_seed_imag(l, m, r_idx))
        end
    end
    return spec
end

# Gather global (nlm, nr) real/imag spectral arrays over COMM_WORLD.
function _p3_gather_scalar(spec, cfg, nr, comm)
    nlm = cfg.nlm
    gr = zeros(Float64, nlm, nr)
    gi = zeros(Float64, nlm, nr)
    sr = parent(spec.data_real)
    si = parent(spec.data_imag)
    r_range = GeoDynamo.range_local(cfg.pencils.spec, 3)
    for lm_idx in GeoDynamo.local_spectral_mode_indices(cfg)
        lm_idx <= nlm || continue
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_idx)
        slot === nothing && continue
        for r_idx in r_range
            lr = r_idx - first(r_range) + 1
            lr > size(sr, 3) && continue
            gr[lm_idx, r_idx] = GeoDynamo.local_spectral_value(sr, slot, lr)
            gi[lm_idx, r_idx] = GeoDynamo.local_spectral_value(si, slot, lr)
        end
    end
    MPI.Allreduce!(gr, +, comm)
    MPI.Allreduce!(gi, +, comm)
    return gr, gi
end

@testset "scalar spec→phys→spec roundtrip (DistTransposePlan, dealiased)" begin
    comm   = MPI.COMM_WORLD
    rank   = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    lmax, mmax, nlat, nlon, nr = 8, 8, 12, 24, 8   # dealiased nlon=24 > 2*mmax+1
    cfg = GeoDynamo.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon, nr=nr)
    dom = GeoDynamo.create_radial_domain(nr)

    # Spectral fields on the (redefined) spec pencil; physical on the r pencil
    # (θ-dist / φ-local / r-dist) — matching the real temperature field layout.
    spec_in  = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    spec_out = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    phys     = GeoDynamo.create_shtns_physical_field(Float64, cfg, dom, cfg.pencils.r)

    _p3_seed_scalar!(spec_in, cfg)

    # The scalar transforms must use the DistTransposePlan path (Phase 3 T2),
    # not the Phase-2 gather path. Track invocations via a module counter.
    c0 = GeoDynamo._SCALAR_DISTTRANSPOSE_COUNT[]
    GeoDynamo.scalar_spectral_to_physical!(spec_in, phys)
    GeoDynamo.scalar_physical_to_spectral!(phys, spec_out)
    @test GeoDynamo._SCALAR_DISTTRANSPOSE_COUNT[] == c0 + 2  # synthesis + analysis

    # Gate 1: roundtrip identity (global, gathered).
    gin_r,  gin_i  = _p3_gather_scalar(spec_in,  cfg, nr, comm)
    gout_r, gout_i = _p3_gather_scalar(spec_out, cfg, nr, comm)
    maxdiff = max(maximum(abs, gout_r .- gin_r), maximum(abs, gout_i .- gin_i))
    if rank == 0
        println("[p3-scalar grid=$(get(ENV,"GEODYNAMO_PROC_GRID","$(nprocs)x1"))] roundtrip maxdiff=$maxdiff")
    end
    @test maxdiff < 1e-10

    # Gate 2: partition-neutrality. The gathered global seed must equal the
    # deterministic reference recomputed from global (l,m,r) — identical on EVERY
    # grid, proving the redefined spec pencil + transform are physics-neutral.
    ref_r = zeros(Float64, cfg.nlm, nr)
    ref_i = zeros(Float64, cfg.nlm, nr)
    for lm_idx in 1:cfg.nlm
        l = cfg.l_values[lm_idx]; m = cfg.m_values[lm_idx]
        (0 <= m <= cfg.mmax && m <= l <= cfg.lmax) || continue
        for r_idx in 1:nr
            ref_r[lm_idx, r_idx] = _p3_scalar_seed_real(l, m, r_idx)
            ref_i[lm_idx, r_idx] = _p3_scalar_seed_imag(l, m, r_idx)
        end
    end
    @test maximum(abs, gout_r .- ref_r) < 1e-10
    @test maximum(abs, gout_i .- ref_i) < 1e-10
end

FINALIZE_MPI_P3_TRANSPOSE && MPI.Finalize()
