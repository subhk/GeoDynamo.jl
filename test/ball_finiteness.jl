using Test
using Geodynamo
using MPI

const Ball = Geodynamo.GeodynamoBall
const FINALIZE_MPI = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

@testset "Ball finiteness at r=0 for derived quantities" begin
    if MPI.Finalized()
        @warn "MPI already finalized before ball finiteness tests; skipping"
        return
    end

    if !MPI.Initialized()
        MPI.Init()
    end

    comm = Geodynamo.get_comm()

    # Small config
    lmax = 6; mmax = 6
    nlat = max(lmax + 2, 12)
    nlon = max(2lmax + 1, 24)
    nr   = 6

    cfg = Geodynamo.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon, nr=nr)
    dom = Ball.create_ball_radial_domain(nr)

    # Velocity vorticity finiteness at r=0
    vfields = Geodynamo.create_shtns_velocity_fields(Float64, cfg, dom, cfg.pencils, cfg.pencils.spec)
    # Fill toroidal/poloidal spectra with random values (no regularity enforced here)
    parent(vfields.toroidal.data_real) .= randn.(Float64)
    parent(vfields.toroidal.data_imag) .= randn.(Float64)
    parent(vfields.poloidal.data_real) .= randn.(Float64)
    parent(vfields.poloidal.data_imag) .= randn.(Float64)

    Geodynamo.compute_vorticity_spectral_full!(vfields, dom)

    ω_tor_r = parent(vfields.vort_toroidal.data_real)
    ω_tor_i = parent(vfields.vort_toroidal.data_imag)
    ω_pol_r = parent(vfields.vort_poloidal.data_real)
    ω_pol_i = parent(vfields.vort_poloidal.data_imag)

    r_range = Geodynamo.range_local(cfg.pencils.spec, 3)
    lm_range = Geodynamo.range_local(cfg.pencils.spec, 1)
    if 1 in r_range && !isempty(lm_range)
        local_r = 1 - first(r_range) + 1
        # Expect inner plane to be finite and effectively zero by guard
        for k in 1:length(lm_range)
            @test isfinite(ω_tor_r[k, 1, local_r]) && isfinite(ω_tor_i[k, 1, local_r])
            @test isfinite(ω_pol_r[k, 1, local_r]) && isfinite(ω_pol_i[k, 1, local_r])
            @test ω_tor_r[k, 1, local_r] ≈ 0.0 atol=1e-12
            @test ω_tor_i[k, 1, local_r] ≈ 0.0 atol=1e-12
            @test ω_pol_r[k, 1, local_r] ≈ 0.0 atol=1e-12
            @test ω_pol_i[k, 1, local_r] ≈ 0.0 atol=1e-12
        end
    end

    # Magnetic current density finiteness at r=0
    mfields = Geodynamo.create_shtns_magnetic_fields(Float64, cfg, dom, dom)
    parent(mfields.toroidal.data_real) .= randn.(Float64)
    parent(mfields.toroidal.data_imag) .= randn.(Float64)
    parent(mfields.poloidal.data_real) .= randn.(Float64)
    parent(mfields.poloidal.data_imag) .= randn.(Float64)

    Geodynamo.compute_current_density_spectral!(mfields, dom)

    j_tor_r = parent(mfields.work_tor.data_real)
    j_tor_i = parent(mfields.work_tor.data_imag)
    j_pol_r = parent(mfields.work_pol.data_real)
    j_pol_i = parent(mfields.work_pol.data_imag)

    r_range = Geodynamo.range_local(cfg.pencils.spec, 3)
    lm_range = Geodynamo.range_local(cfg.pencils.spec, 1)
    if 1 in r_range && !isempty(lm_range)
        local_r = 1 - first(r_range) + 1
        for k in 1:length(lm_range)
            @test isfinite(j_tor_r[k, 1, local_r]) && isfinite(j_tor_i[k, 1, local_r])
            @test isfinite(j_pol_r[k, 1, local_r]) && isfinite(j_pol_i[k, 1, local_r])
            @test j_tor_r[k, 1, local_r] ≈ 0.0 atol=1e-12
            @test j_tor_i[k, 1, local_r] ≈ 0.0 atol=1e-12
            @test j_pol_r[k, 1, local_r] ≈ 0.0 atol=1e-12
            @test j_pol_i[k, 1, local_r] ≈ 0.0 atol=1e-12
        end
    end

    if MPI.Initialized()
        MPI.Barrier(comm)
        if FINALIZE_MPI && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
