using Test
using MPI
using NCDatasets
using LinearAlgebra
using Statistics

const FINALIZE_MPI_IO_SUB = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

# Coverage for the previously-untested I/O helpers:
#   * io/diagnostics.jl — compute_diagnostics + compute_spectral_energy_diagnostics!
#   * io/history.jl     — resolved_state_radial_grid + write_fields! (Dict path)
#   * io/utilities.jl   — validate_output, get_file_info, get_time_series,
#                         verify_all_ranks_wrote, cleanup_old_files,
#                         create_shtns_aware_output_config,
#                         validate_output_compatibility
@testset "I/O subsystem" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping I/O subsystem tests"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    lmax = 4
    mmax = 4
    nlat = max(lmax + 2, 10)
    nlon = max(2lmax + 1, 16)
    nr = 8

    cfg = GeoDynamo.create_shtnskit_config(
        lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr)
    dom = GeoDynamo.create_radial_domain(nr)
    radial_nodes = Float64.(dom.r[1:nr, 4])
    nlm = cfg.nlm

    @testset "compute_diagnostics: physical + spectral summaries" begin
        Tphys = fill(2.0, nlat, nlon, nr)
        Cphys = fill(-1.0, nlat, nlon, nr)
        mr = randn(nlm, nr)
        mi = randn(nlm, nr)
        fields = Dict{String, Any}(
            "temperature" => Tphys,
            "composition" => Cphys,
            "magnetic_toroidal" => Dict("real" => mr, "imag" => mi)
        )
        fi = GeoDynamo.extract_field_info(fields, cfg, nothing;
            radius_ratio = 0.35, radial_grid = radial_nodes)
        d = GeoDynamo.compute_diagnostics(fields, fi)

        # physical-space stats are exact for constant fields
        @test d["temp_mean"] ≈ 2.0
        @test d["temp_std"] ≈ 0.0 atol = 1e-12
        @test d["temp_min"] == 2.0
        @test d["temp_max"] == 2.0
        @test d["comp_mean"] ≈ -1.0
        @test d["comp_min"] == -1.0

        # spectral energy/rms/max match the analytic definitions
        @test d["magnetic_toroidal_energy"] ≈ 0.5 * (sum(mr .^ 2) + sum(mi .^ 2))
        @test d["magnetic_toroidal_max"] ≈ maximum(sqrt.(mr .^ 2 .+ mi .^ 2))

        # has_config -> degree-wise spectral diagnostics are added
        @test fi.has_config
        @test haskey(d, "magnetic_toroidal_peak_l")
        @test haskey(d, "magnetic_toroidal_spectral_centroid")
        @test 0.0 <= d["magnetic_toroidal_low_mode_fraction"] <= 1.0
    end

    @testset "resolved_state_radial_grid prefers explicit, then state domain" begin
        # explicit grid wins
        @test GeoDynamo.resolved_state_radial_grid(nothing, radial_nodes) === radial_nodes
        # no runtime, no explicit grid -> nothing
        @test GeoDynamo.resolved_state_radial_grid(nothing, nothing) === nothing
        # pulls nodes from state.runtime.outer_core_domain
        mock_state = (runtime = (outer_core_domain = dom,),)
        @test GeoDynamo.resolved_state_radial_grid(mock_state, nothing) ≈
              Float64.(dom.r[1:dom.N, 4])
    end

    @testset "create_shtns_aware_output_config + validate_output_compatibility" begin
        ocfg = GeoDynamo.create_shtns_aware_output_config(cfg, cfg.pencils)
        @test ocfg.spectral_lmax_output == cfg.lmax

        fields = Dict{String, Any}("temperature" => fill(0.0, nlat, nlon, nr))
        fi = GeoDynamo.extract_field_info(fields, cfg, nothing;
            radius_ratio = 0.35, radial_grid = radial_nodes)
        @test GeoDynamo.validate_output_compatibility(fi, cfg) == true

        # field info built against a different config -> incompatible (warns + false)
        cfg2 = GeoDynamo.create_shtnskit_config(
            lmax = lmax + 2, mmax = mmax + 2, nlat = nlat + 2, nlon = nlon + 4, nr = nr)
        fi2 = GeoDynamo.extract_field_info(fields, cfg2, nothing;
            radius_ratio = 0.35, radial_grid = radial_nodes)
        @test (@test_logs (:warn,) GeoDynamo.validate_output_compatibility(fi2, cfg)) == false
    end

    comm = GeoDynamo.output_comm()
    parallel_ok = try
        probe = tempname() * ".nc"
        ds0 = NCDataset(comm, probe, "c"; info = MPI.Info())
        close(ds0)
        isfile(probe) && rm(probe; force = true)
        true
    catch err
        @warn "Parallel NetCDF unavailable; skipping write_fields!/utilities file tests" error = err
        false
    end

    if parallel_ok
        tmpdir = mktempdir()
        base = GeoDynamo.default_config(Float64)
        config = GeoDynamo.OutputConfig(
            base.output_space, tmpdir, base.filename_prefix,
            base.include_metadata, base.include_grid, base.include_diagnostics,
            Float64, base.spectral_lmax_output, true,
            base.output_interval, base.restart_interval, base.max_output_time,
            base.time_tolerance)

        T_in = randn(nlat, nlon, nr)
        fields = Dict{String, Any}(
            "temperature" => copy(T_in),
            "magnetic_toroidal" =>
                Dict("real" => randn(nlm, nr), "imag" => randn(nlm, nr))
        )

        @testset "write_fields! (Dict path) writes a history file" begin
            tracker = GeoDynamo.create_time_tracker(config, 0.0)  # triggers output + restart at t=0
            metadata = Dict{String, Any}("current_time" => 0.0, "current_step" => 7)

            wrote = GeoDynamo.write_fields!(fields, tracker, metadata, config, cfg, nothing;
                geometry = :shell, radius_ratio = 0.35, radial_grid = radial_nodes)
            @test wrote == true
            @test tracker.output_count == 1
            @test tracker.restart_count == 1
            @test tracker.grid_file_written == true

            restartfile = GeoDynamo.generate_filename(
                config, 0.0, 7, "restart", 1; geometry = :shell)
            resumed = GeoDynamo.create_time_tracker(config, 0.0)
            GeoDynamo._load_restart_file(restartfile, resumed, config;
                shtns_config = cfg)
            # The checkpoint represents the state after both files emitted by
            # this write_fields! call, not the counters from before the call.
            @test resumed.output_count == 1
            @test resumed.restart_count == 1

            # the no-write path returns false without producing a file
            quiet = GeoDynamo.create_time_tracker(config, 0.0)
            quiet.last_output_time = 1.0e6
            quiet.last_restart_time = 1.0e6
            @test GeoDynamo.write_fields!(fields, quiet,
                Dict{String, Any}("current_time" => 0.0, "current_step" => 1),
                config, cfg, nothing; radial_grid = radial_nodes) == false
        end

        histfile = GeoDynamo.generate_filename(config, 0.0, 7, "output", 1; geometry = :shell)

        @testset "utilities read back the written history file" begin
            @test isfile(histfile)
            @test GeoDynamo.validate_output(histfile) == true
            @test GeoDynamo.validate_output(joinpath(tmpdir, "missing.nc")) == false

            info = GeoDynamo.get_file_info(histfile)
            @test info["step"] == 7
            @test haskey(info, "variables")
            @test haskey(info, "dimensions")

            # get_time_series picks up the hist file (restart excluded), sorted
            @test GeoDynamo.get_time_series(tmpdir) == [0.0]

            # verify_all_ranks_wrote: present, absent, dim-match, dim-mismatch
            ok, msgs, _ = GeoDynamo.verify_all_ranks_wrote(tmpdir, 1; geometry = "shell")
            @test ok == true
            @test isempty(msgs)

            missing_ok, _, _ = GeoDynamo.verify_all_ranks_wrote(tmpdir, 999; geometry = "shell")
            @test missing_ok == false

            # read a real dimension + size the same way verify_all_ranks_wrote does
            dname, dsize = NCDataset(histfile, "r") do ds
                ("r", ds.dim["r"])
            end
            match_ok, _, match_info = GeoDynamo.verify_all_ranks_wrote(
                tmpdir, 1; geometry = "shell", expected_dims = Dict(dname => dsize))
            @test match_ok == true
            @test haskey(match_info, "dimensions")

            mis_ok, _, _ = GeoDynamo.verify_all_ranks_wrote(
                tmpdir, 1; geometry = "shell", expected_dims = Dict(dname => dsize + 1))
            @test mis_ok == false
        end

        @testset "cleanup_old_files keeps the newest N by history number" begin
            cdir = mktempdir()
            for n in 1:13
                write(joinpath(cdir, "geodynamo_shell_hist_$(n).nc"), "x")
            end
            hist_pat = r"_hist_(\d+)\.nc$"
            GeoDynamo.cleanup_old_files(cdir, 10)
            remaining = filter(f -> occursin(hist_pat, f), readdir(cdir))
            @test length(remaining) == 10
            nums = sort([parse(Int, match(hist_pat, f).captures[1]) for f in remaining])
            @test nums == collect(4:13)

            # already at/under the keep count -> no-op
            GeoDynamo.cleanup_old_files(cdir, 100)
            @test count(f -> occursin(hist_pat, f), readdir(cdir)) == 10

            # empty directory -> no error
            @test GeoDynamo.cleanup_old_files(mktempdir(), 10) === nothing
        end
    end

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_IO_SUB && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
