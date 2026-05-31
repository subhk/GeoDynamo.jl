using Test, GeoDynamo, MPI, PencilArrays
# Owns MPI lifecycle only if it had to initialize (so later appended @testsets and
# the suite runner don't double-init / prematurely finalize). Mirrors the
# FINALIZE_* guard pattern in test/mpi_parallel_invariants.jl.
const FINALIZE_MPI_THETA_DIST = !MPI.Initialized()
FINALIZE_MPI_THETA_DIST && MPI.Init()

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
    # Seed a few owned spectral modes directly so the roundtrip has something
    # non-trivial to preserve.
    sr = parent(tf.spectral.data_real); si = parent(tf.spectral.data_imag)
    sr .= 0; si .= 0
    for k in 1:size(sr, 3)
        sr[min(2, size(sr, 1)), 1, k] = 0.7
        if size(sr, 2) >= 2
            sr[min(3, size(sr, 1)), 2, k] = 0.3
            si[min(3, size(sr, 1)), 2, k] = -0.15
        end
    end
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
    tr = parent(vf.toroidal.data_real); ti = parent(vf.toroidal.data_imag)
    pr = parent(vf.poloidal.data_real); pi_ = parent(vf.poloidal.data_imag)
    tr .= 0; ti .= 0; pr .= 0; pi_ .= 0
    for k in 1:size(tr,3)
        tr[min(2,size(tr,1)),1,k] = 0.5
        if size(tr,2) >= 2; pr[min(3,size(pr,1)),2,k] = 0.3; pi_[min(3,size(pr,1)),2,k] = -0.1; end
    end
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
    tr .= 0; ti .= 0; pr .= 0; pii .= 0
    for k in 1:size(tr, 3)
        tr[min(2, size(tr, 1)), 1, k] = 0.5
        if size(tr, 2) >= 2
            pr[min(3, size(pr, 1)), 2, k]  =  0.3
            pii[min(3, size(pii, 1)), 2, k] = -0.1
        end
    end
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
