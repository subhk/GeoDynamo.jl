using Test
using MPI
using Random

const Ball = GeoDynamo.GeoDynamoBall
const FINALIZE_MPI = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

@testset "Ball finiteness at r=0 for derived quantities" begin
    if MPI.Finalized()
        @warn "MPI already finalized before ball finiteness tests; skipping"
        return
    end

    if !MPI.Initialized()
        MPI.Init()
    end

    comm = GeoDynamo.get_comm()

    # Small config
    lmax = 6;
    mmax = 6
    nlat = max(lmax + 2, 12)
    nlon = max(2lmax + 1, 24)
    nr = 6

    cfg = GeoDynamo.create_shtnskit_config(
        lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr)
    dom = Ball.create_ball_radial_domain(nr)

    # Create velocity workspace for all potentially active threads
    # Use Threads.maxthreads() if available, otherwise use a safe buffer
    nthreads_workspace = if isdefined(Threads, :maxthreads)
        Threads.maxthreads()
    else
        max(Threads.nthreads(), 4)
    end
    ws = GeoDynamo.create_velocity_workspace(Float64, nr, nthreads_workspace)
    GeoDynamo.set_velocity_workspace!(ws)

    # Velocity vorticity finiteness at r=0
    vfields = GeoDynamo.create_shtns_velocity_fields(Float64, cfg, dom;
        params = GeoDynamo.SolverParameters(
            geometry = :ball,
            radius_ratio = 0.0,
            nr_inner = 0,
            nr = nr,
            lmax = lmax,
            mmax = mmax,
            nlat = nlat,
            nlon = nlon
        ))
    # Fill toroidal/poloidal spectra with random values (no regularity enforced here)
    randn!(parent(vfields.𝒯.data_real))
    randn!(parent(vfields.𝒯.data_imag))
    randn!(parent(vfields.𝒫.data_real))
    randn!(parent(vfields.𝒫.data_imag))

    GeoDynamo.compute_vorticity_spectral_full!(vfields, dom)

    ω_tor_r = parent(vfields.ζᵀ.data_real)
    ω_tor_i = parent(vfields.ζᵀ.data_imag)
    ω_pol_r = parent(vfields.ζᴾ.data_real)
    ω_pol_i = parent(vfields.ζᴾ.data_imag)

    r_range = GeoDynamo.range_local(cfg.pencils.spec, 3)
    lm_range = GeoDynamo.local_spectral_mode_indices(cfg)
    if 1 in r_range && !isempty(lm_range)
        local_r = 1 - first(r_range) + 1
        # Expect inner plane to be finite and effectively zero by guard
        for lm_idx in lm_range
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_idx)
            slot === nothing && continue
            @test isfinite(ω_tor_r[slot[1], slot[2], local_r]) &&
                  isfinite(ω_tor_i[slot[1], slot[2], local_r])
            @test isfinite(ω_pol_r[slot[1], slot[2], local_r]) &&
                  isfinite(ω_pol_i[slot[1], slot[2], local_r])
            @test ω_tor_r[slot[1], slot[2], local_r] ≈ 0.0 atol=1e-12
            @test ω_tor_i[slot[1], slot[2], local_r] ≈ 0.0 atol=1e-12
            @test ω_pol_r[slot[1], slot[2], local_r] ≈ 0.0 atol=1e-12
            @test ω_pol_i[slot[1], slot[2], local_r] ≈ 0.0 atol=1e-12
        end
    end

    # Magnetic current density finiteness at r=0
    mfields = GeoDynamo.create_shtns_magnetic_fields(Float64, cfg, dom, dom)
    randn!(parent(mfields.𝒯.data_real))
    randn!(parent(mfields.𝒯.data_imag))
    randn!(parent(mfields.𝒫.data_real))
    randn!(parent(mfields.𝒫.data_imag))

    GeoDynamo.compute_current_density_spectral!(mfields, dom)

    j_tor_r = parent(mfields.work_tor.data_real)
    j_tor_i = parent(mfields.work_tor.data_imag)
    j_pol_r = parent(mfields.work_pol.data_real)
    j_pol_i = parent(mfields.work_pol.data_imag)

    r_range = GeoDynamo.range_local(cfg.pencils.spec, 3)
    lm_range = GeoDynamo.local_spectral_mode_indices(cfg)
    if 1 in r_range && !isempty(lm_range)
        local_r = 1 - first(r_range) + 1
        for lm_idx in lm_range
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_idx)
            slot === nothing && continue
            @test isfinite(j_tor_r[slot[1], slot[2], local_r]) &&
                  isfinite(j_tor_i[slot[1], slot[2], local_r])
            @test isfinite(j_pol_r[slot[1], slot[2], local_r]) &&
                  isfinite(j_pol_i[slot[1], slot[2], local_r])
            @test j_tor_r[slot[1], slot[2], local_r] ≈ 0.0 atol=1e-12
            @test j_tor_i[slot[1], slot[2], local_r] ≈ 0.0 atol=1e-12
            @test j_pol_r[slot[1], slot[2], local_r] ≈ 0.0 atol=1e-12
            @test j_pol_i[slot[1], slot[2], local_r] ≈ 0.0 atol=1e-12
        end
    end

    if MPI.Initialized()
        MPI.Barrier(comm)
        if FINALIZE_MPI && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
