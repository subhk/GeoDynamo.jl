using Test
using MPI

const Ball = GeoDynamo.GeoDynamoBall
const FINALIZE_MPI_INITIAL = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

function assert_ball_scalar_regularity(spec, cfg; atol=1e-12)
    lm_range = GeoDynamo.local_spectral_mode_indices(cfg)
    r_range = GeoDynamo.range_local(cfg.pencils.spec, 3)
    if !(1 in r_range)
        return
    end
    r_local = 1 - first(r_range) + 1
    real = parent(spec.data_real)
    imag = parent(spec.data_imag)
    for lm_idx in lm_range
        if cfg.l_values[lm_idx] > 0
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_idx)
            slot === nothing && continue
            @test real[slot[1], slot[2], r_local] ≈ 0.0 atol=atol
            @test imag[slot[1], slot[2], r_local] ≈ 0.0 atol=atol
        end
    end
end

function assert_ball_vector_regularity(tor_spec, pol_spec, cfg; atol=1e-12)
    lm_range = GeoDynamo.local_spectral_mode_indices(cfg)
    r_range = GeoDynamo.range_local(cfg.pencils.spec, 3)
    if !(1 in r_range)
        return
    end
    r_local = 1 - first(r_range) + 1
    for spec in (tor_spec, pol_spec)
        real = parent(spec.data_real)
        imag = parent(spec.data_imag)
        for lm_idx in lm_range
            if cfg.l_values[lm_idx] >= 1
                slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_idx)
                slot === nothing && continue
                @test real[slot[1], slot[2], r_local] ≈ 0.0 atol=atol
                @test imag[slot[1], slot[2], r_local] ≈ 0.0 atol=atol
            end
        end
    end
end

@testset "Initial Conditions" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping initial condition tests"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    lmax = 4
    mmax = 4
    nlat = max(lmax + 2, 10)
    nlon = max(2lmax + 1, 16)
    nr = 6

    cfg = GeoDynamo.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon, nr=nr)
    shell = GeoDynamo.create_radial_domain(nr)
    ball = Ball.create_ball_radial_domain(nr)

    @testset "Random generators are reproducible with a seed" begin
        temp_a = GeoDynamo.create_shtns_temperature_field(Float64, cfg, shell)
        temp_b = GeoDynamo.create_shtns_temperature_field(Float64, cfg, shell)

        GeoDynamo.generate_random_initial_conditions!(temp_a, :temperature, amplitude=0.2, modes_range=1:3, seed=7)
        GeoDynamo.generate_random_initial_conditions!(temp_b, :temperature, amplitude=0.2, modes_range=1:3, seed=7)

        @test parent(temp_a.spectral.data_real) == parent(temp_b.spectral.data_real)
        @test parent(temp_a.spectral.data_imag) == parent(temp_b.spectral.data_imag)
    end

    @testset "Placeholder load/save behavior stays explicit" begin
        temp = GeoDynamo.create_shtns_temperature_field(Float64, cfg, shell)

        missing_path = joinpath(mktempdir(), "missing_initial_conditions.nc")
        @test_throws ArgumentError GeoDynamo.load_initial_conditions!(temp, :temperature, missing_path)

        existing_path, io = mktemp()
        close(io)

        loaded = @test_logs (:warn, r"NetCDF loading not implemented") GeoDynamo.load_initial_conditions!(temp, :temperature, existing_path)
        @test loaded === temp

        save_path = joinpath(mktempdir(), "saved_initial_conditions.nc")
        saved = @test_logs (:warn, r"NetCDF saving not implemented") GeoDynamo.save_initial_conditions(temp, :temperature, save_path)
        @test saved == save_path
    end

    @testset "Ball analytical presets preserve regularity at r=0" begin
        temp = GeoDynamo.create_shtns_temperature_field(Float64, cfg, ball)
        GeoDynamo.set_analytical_initial_conditions!(temp, :temperature, :hot_blob, amplitude=1.0, r_center=0.0, blob_width=0.3)
        assert_ball_scalar_regularity(temp.spectral, cfg)

        comp = GeoDynamo.create_shtns_composition_field(Float64, cfg, ball)
        GeoDynamo.set_analytical_initial_conditions!(comp, :composition, :blob, amplitude=1.0, r_center=0.0, blob_width=0.3, blob_composition=0.8)
        assert_ball_scalar_regularity(comp.spectral, cfg)

        magnetic = GeoDynamo.create_shtns_magnetic_fields(Float64, cfg, ball, ball)
        GeoDynamo.set_analytical_initial_conditions!(magnetic, :magnetic, :uniform_field, amplitude=1.0, direction=:x)
        assert_ball_vector_regularity(magnetic.toroidal, magnetic.poloidal, cfg)
    end

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_INITIAL && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
