using Test, GeoDynamo, MPI, PencilArrays
# Owns MPI lifecycle only if it had to initialize (so later appended @testsets and
# the suite runner don't double-init / prematurely finalize). Mirrors the
# FINALIZE_* guard pattern in test/mpi_parallel_invariants.jl.
const FINALIZE_MPI_THETA_DIST = !MPI.Initialized()
FINALIZE_MPI_THETA_DIST && MPI.Init()

function _theta_assert_only_valid_modes_seeded(spec, cfg)
    sr = parent(spec.data_real)
    si = parent(spec.data_imag)
    lm_map = GeoDynamo.local_spectral_lm_map(cfg)
    @test size(lm_map) == size(sr)[1:2]
    for slot in CartesianIndices(lm_map)
        lm_map[slot] != 0 && continue
        @test all(iszero, @view sr[slot[1], slot[2], :])
        @test all(iszero, @view si[slot[1], slot[2], :])
    end
    return nothing
end

function _theta_seed_scalar!(spec, cfg)
    sr = parent(spec.data_real)
    si = parent(spec.data_imag)
    fill!(sr, 0.0)
    fill!(si, 0.0)
    r_range = GeoDynamo.range_local(cfg.pencils.spec, 3)
    for lm_idx in GeoDynamo.local_spectral_mode_indices(cfg)
        l = cfg.l_values[lm_idx]
        m = cfg.m_values[lm_idx]
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_idx)
        slot === nothing && continue
        for r_idx in r_range
            local_r = r_idx - first(r_range) + 1
            GeoDynamo.set_local_spectral_value!(
                sr, slot, local_r, 0.7 / (l + m + 1))
            GeoDynamo.set_local_spectral_value!(
                si, slot, local_r, m == 0 ? 0.0 : -0.15 / (l + m + 1))
        end
    end
    return spec
end

function _theta_seed_vector!(toroidal, poloidal, cfg)
    tr = parent(toroidal.data_real)
    ti = parent(toroidal.data_imag)
    pr = parent(poloidal.data_real)
    pi_ = parent(poloidal.data_imag)
    fill!(tr, 0.0)
    fill!(ti, 0.0)
    fill!(pr, 0.0)
    fill!(pi_, 0.0)
    r_range = GeoDynamo.range_local(cfg.pencils.spec, 3)
    for lm_idx in GeoDynamo.local_spectral_mode_indices(cfg)
        l = cfg.l_values[lm_idx]
        m = cfg.m_values[lm_idx]
        l == 0 && continue
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_idx)
        slot === nothing && continue
        for r_idx in r_range
            local_r = r_idx - first(r_range) + 1
            radial_factor = 1 + 0.2 * r_idx
            GeoDynamo.set_local_spectral_value!(
                tr, slot, local_r, 0.5 / (l + m + 1))
            GeoDynamo.set_local_spectral_value!(
                ti, slot, local_r, m == 0 ? 0.0 : -0.08 / (l + m + 1))
            GeoDynamo.set_local_spectral_value!(
                pr, slot, local_r, 0.3 * radial_factor / (l + m + 1))
            GeoDynamo.set_local_spectral_value!(
                pi_, slot, local_r,
                m == 0 ? 0.0 : -0.1 * radial_factor / (l + m + 1))
        end
    end
    return toroidal, poloidal
end

@testset "1D-theta layout + prototype pencil" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=20, nr=4)
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    rloc = PencilArrays.range_local(cfg.pencils.r)
    @test length(rloc[2]) == 20          # φ FULLY LOCAL (1:nlon)
    @test length(rloc[3]) == 4           # r LOCAL (1:nr)
    # θ is distributed: under np>1 the local θ-slab is a strict subset of 1:nlat.
    # (On np=1 it is the full 1:12, so the split-match below is trivially true —
    #  the non-trivial multi-rank check runs under `mpiexec -n 2+` in Task 6.)
    @test nprocs == 1 || length(rloc[1]) < 12
    p2 = cfg.pencils.theta_phys
    p2loc = PencilArrays.range_local(p2)
    @test length(p2loc[2]) == 20         # φ full on the 2D prototype
    @test p2loc[1] == rloc[1]            # θ-split MATCHES pencils.r (critical)
end

# ============================================================================
# Task 3: scalar dist-transform roundtrip (spec → phys → spec)
# ============================================================================

@testset "scalar dist transform roundtrip" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=20, nr=4)
    dom = GeoDynamo.create_radial_domain(4)
    tf  = GeoDynamo.create_shtns_temperature_field(Float64, cfg, dom)
    # Seed every valid mode owned by this rank through the global-mode mapping
    # so the roundtrip has non-trivial, decomposition-independent data.
    _theta_seed_scalar!(tf.spectral, cfg)
    sr = parent(tf.spectral.data_real); si = parent(tf.spectral.data_imag)
    _theta_assert_only_valid_modes_seeded(tf.spectral, cfg)
    sr0 = copy(sr); si0 = copy(si)
    GeoDynamo.scalar_spectral_to_physical!(tf.spectral, tf.temperature)
    GeoDynamo.scalar_physical_to_spectral!(tf.temperature, tf.spectral)
    @test maximum(abs.(parent(tf.spectral.data_real) .- sr0)) < 1e-10
    @test maximum(abs.(parent(tf.spectral.data_imag) .- si0)) < 1e-10
end

# ============================================================================
# Task 5: vector dist-transform roundtrip (tor/pol spec → phys → tor/pol spec)
# ============================================================================

@testset "vector dist transform roundtrip" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=20, nr=4)
    dom = GeoDynamo.create_radial_domain(4)
    vf  = GeoDynamo.create_shtns_velocity_fields(Float64, cfg, dom)
    _theta_seed_vector!(vf.toroidal, vf.poloidal, cfg)
    tr = parent(vf.toroidal.data_real); ti = parent(vf.toroidal.data_imag)
    pr = parent(vf.poloidal.data_real); pi_ = parent(vf.poloidal.data_imag)
    _theta_assert_only_valid_modes_seeded(vf.toroidal, cfg)
    _theta_assert_only_valid_modes_seeded(vf.poloidal, cfg)
    tr0=copy(tr); ti0=copy(ti); pr0=copy(pr); pi0=copy(pi_)
    GeoDynamo.shtnskit_vector_synthesis!(vf.toroidal, vf.poloidal, vf.velocity; domain=dom)
    # Assert synthesized physical field is finite and non-zero (defence-in-depth:
    # catches a θ-split bug where the prototype gives the wrong rank's rows)
    vt_data = parent(vf.velocity.θ_component.data)
    @test any(x -> abs(x) > 1e-10, vt_data)
    @test all(isfinite, vt_data)
    GeoDynamo.shtnskit_vector_analysis!(vf.velocity, vf.toroidal, vf.poloidal; domain=dom)
    @test maximum(abs.(parent(vf.toroidal.data_real) .- tr0)) < 1e-8
    @test maximum(abs.(parent(vf.toroidal.data_imag) .- ti0)) < 1e-8
    @test maximum(abs.(parent(vf.poloidal.data_real) .- pr0)) < 1e-8
    @test maximum(abs.(parent(vf.poloidal.data_imag) .- pi0)) < 1e-8
end

# ============================================================================
# Phase-1b: solver vector dist transform roundtrip
# Tests vector_spectral_to_physical! / vector_physical_to_spectral! in
# src/solver/numerics.jl after migration to dist_*_sphtor.
# ============================================================================

@testset "solver vector dist transform roundtrip" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=20, nr=4)
    dom = GeoDynamo.create_radial_domain(4)
    vf  = GeoDynamo.create_shtns_velocity_fields(Float64, cfg, dom)
    tr  = parent(vf.toroidal.data_real); ti  = parent(vf.toroidal.data_imag)
    pr  = parent(vf.poloidal.data_real); pii = parent(vf.poloidal.data_imag)
    _theta_seed_vector!(vf.toroidal, vf.poloidal, cfg)
    _theta_assert_only_valid_modes_seeded(vf.toroidal, cfg)
    _theta_assert_only_valid_modes_seeded(vf.poloidal, cfg)
    tr0 = copy(tr); ti0 = copy(ti); pr0 = copy(pr); pi0 = copy(pii)

    # Call the LIVE SOLVER vector transforms (numerics.jl, in GeoDynamo module scope).
    # domain=dom enables v_r (poloidal radial component) synthesis.
    GeoDynamo.vector_spectral_to_physical!(vf.toroidal, vf.poloidal, vf.velocity; domain=dom)

    # Assert synthesized physical field is finite and non-zero.
    vt_data = parent(vf.velocity.θ_component.data)
    @test any(x -> abs(x) > 1e-10, vt_data)
    @test all(isfinite, vt_data)

    GeoDynamo.vector_physical_to_spectral!(vf.velocity, vf.toroidal, vf.poloidal; domain=dom)

    @test maximum(abs.(parent(vf.toroidal.data_real) .- tr0)) < 1e-8
    @test maximum(abs.(parent(vf.toroidal.data_imag) .- ti0)) < 1e-8
    @test maximum(abs.(parent(vf.poloidal.data_real) .- pr0)) < 1e-8
    @test maximum(abs.(parent(vf.poloidal.data_imag) .- pi0)) < 1e-8
end

# ============================================================================
# Tasks 5 (vector transform roundtrips): APPEND new @testset blocks
# ABOVE this line — the MPI.Finalize() below must remain the last statement.
# ============================================================================

FINALIZE_MPI_THETA_DIST && MPI.Finalize()
