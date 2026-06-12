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

    # The velocity field now owns its scratch workspace and builds it lazily
    # (sized for the active threads) on the first vorticity call — no global
    # workspace registration needed.

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
    randn!(parent(vfields.toroidal.data_real))
    randn!(parent(vfields.toroidal.data_imag))
    randn!(parent(vfields.poloidal.data_real))
    randn!(parent(vfields.poloidal.data_imag))

    GeoDynamo.compute_vorticity_spectral_full!(vfields, dom)

    ω_tor_r = parent(vfields.ζᵀ.data_real)
    ω_tor_i = parent(vfields.ζᵀ.data_imag)
    ω_pol_r = parent(vfields.ζᴾ.data_real)
    ω_pol_i = parent(vfields.ζᴾ.data_imag)

    r_range = GeoDynamo.range_local(cfg.pencils.spec, 3)
    lm_range = GeoDynamo.local_spectral_mode_indices(cfg)
    if 1 in r_range && !isempty(lm_range)
        local_r = 1 - first(r_range) + 1
        # Off-center grid: inner node is at r₁ > 0, so 1/r operators are finite
        # and the vorticity values at the inner node are generally non-zero.
        # We only verify finiteness here; exact-zero assertions were artifacts of
        # the old r=0 hack (zeroed 1/r columns) and no longer apply.
        for lm_idx in lm_range
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_idx)
            slot === nothing && continue
            @test isfinite(ω_tor_r[slot[1], slot[2], local_r]) &&
                  isfinite(ω_tor_i[slot[1], slot[2], local_r])
            @test isfinite(ω_pol_r[slot[1], slot[2], local_r]) &&
                  isfinite(ω_pol_i[slot[1], slot[2], local_r])
        end
    end

    # Magnetic current density finiteness at r=0
    mfields = GeoDynamo.create_shtns_magnetic_fields(Float64, cfg, dom, dom)
    randn!(parent(mfields.toroidal.data_real))
    randn!(parent(mfields.toroidal.data_imag))
    randn!(parent(mfields.poloidal.data_real))
    randn!(parent(mfields.poloidal.data_imag))

    GeoDynamo.compute_current_density_spectral!(mfields, dom)

    j_tor_r = parent(mfields.work_tor.data_real)
    j_tor_i = parent(mfields.work_tor.data_imag)
    j_pol_r = parent(mfields.work_pol.data_real)
    j_pol_i = parent(mfields.work_pol.data_imag)

    r_range = GeoDynamo.range_local(cfg.pencils.spec, 3)
    lm_range = GeoDynamo.local_spectral_mode_indices(cfg)
    if 1 in r_range && !isempty(lm_range)
        local_r = 1 - first(r_range) + 1
        # Off-center grid: inner node is at r₁ > 0, so current density values
        # are generally non-zero there. Verify finiteness only; exact-zero
        # assertions were artifacts of the old r=0 hack and no longer apply.
        for lm_idx in lm_range
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_idx)
            slot === nothing && continue
            @test isfinite(j_tor_r[slot[1], slot[2], local_r]) &&
                  isfinite(j_tor_i[slot[1], slot[2], local_r])
            @test isfinite(j_pol_r[slot[1], slot[2], local_r]) &&
                  isfinite(j_pol_i[slot[1], slot[2], local_r])
        end
    end

    if MPI.Initialized()
        MPI.Barrier(comm)
        if FINALIZE_MPI && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
