using Test
using MPI
using NCDatasets
using LinearAlgebra

const FINALIZE_MPI_RESTART = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

# Restart I/O contract.
#
# Exercises the previously-untested restart layer (src/io/restart.jl):
#   * find_restart_files  — directory scan + newest-first ordering (pure)
#   * write_restart!       — production parallel-write path for a restart file
#   * read_restart!        — locate-and-read, restoring tracker + field data
#   * _load_restart_file   — read a specific restart file path
@testset "Restart I/O" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping restart I/O tests"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    @testset "find_restart_files filters and orders newest-first" begin
        empty_dir = mktempdir()
        @test GeoDynamo.find_restart_files(empty_dir, 0.0) == String[]

        # Directory with no restart files at all -> empty result.
        only_other = mktempdir()
        write(joinpath(only_other, "geodynamo_shell_output_1.nc"), "x")
        write(joinpath(only_other, "notes.txt"), "x")
        @test GeoDynamo.find_restart_files(only_other, 0.0) == String[]

        # Mixed directory: only the restart .nc files are returned, newest-first.
        dir = mktempdir()
        write(joinpath(dir, "geodynamo_shell_output_3.nc"), "x")   # ignored (not restart)
        write(joinpath(dir, "restart_notes.txt"), "x")             # ignored (not .nc)
        f1 = joinpath(dir, "geodynamo_shell_restart_1.nc")
        f2 = joinpath(dir, "geodynamo_shell_restart_2.nc")
        write(f1, "x")
        sleep(0.05)                                                # ensure f2 mtime > f1
        write(f2, "x")

        found = GeoDynamo.find_restart_files(dir, 0.0)
        @test length(found) == 2
        @test all(p -> endswith(p, ".nc") && occursin("restart", p), found)
        @test all(isabspath, found)
        @test Set(found) == Set([f1, f2])
        @test found[1] == f2          # newest first
        @test found[2] == f1
    end

    comm = GeoDynamo.output_comm()

    # Same parallel-NetCDF capability probe used by the write round-trip test;
    # restart write/read both go through the parallel MPI-IO path.
    parallel_ok = let probe_err = GeoDynamo.parallel_netcdf_probe(comm)
        probe_err === nothing ||
            @warn "Parallel NetCDF unavailable; skipping restart write/read round-trip" error = probe_err
        probe_err === nothing
    end

    if parallel_ok
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
        T_in = randn(cfg.nlat, cfg.nlon, nr)
        btor_real = randn(nlm, nr)
        btor_imag = randn(nlm, nr)

        fields = Dict{String, Any}(
            "temperature" => copy(T_in),
            "magnetic_toroidal" =>
                Dict("real" => copy(btor_real), "imag" => copy(btor_imag))
        )

        tmpdir = mktempdir()
        base = GeoDynamo.default_config(Float64)
        config = GeoDynamo.OutputConfig(
            base.output_space,
            tmpdir,                       # output_dir
            base.filename_prefix,
            base.include_metadata,
            base.include_grid,
            base.include_diagnostics,
            Float64,                      # output_precision
            base.spectral_lmax_output,
            true,                         # overwrite_files
            base.output_interval,
            base.restart_interval,
            base.max_output_time,
            base.time_tolerance
        )

        # Tracker state that must survive the round-trip.
        tracker = GeoDynamo.create_time_tracker(config, 0.0)
        tracker.last_output_time = 1.5
        tracker.output_count = 3
        tracker.restart_count = 0
        tracker.grid_file_written = true

        metadata = Dict{String, Any}(
            "current_time" => 2.5,
            "current_step" => 42
        )

        GeoDynamo.write_restart!(fields, tracker, metadata, config, nothing;
            shtns_config = cfg, geometry = :shell,
            radius_ratio = 0.35, radial_grid = radial_nodes)

        found = GeoDynamo.find_restart_files(tmpdir, 0.0)
        @test length(found) == 1
        @test isfile(found[1])

        @testset "read_restart! restores tracker state and field data" begin
            reader = GeoDynamo.create_time_tracker(config, 0.0)
            restart_data, md = GeoDynamo.read_restart!(
                reader, tmpdir, 2.5, config, nothing; shtns_config = cfg)

            @test md["current_time"] ≈ 2.5
            @test md["current_step"] == 42

            # Tracker scalars round-trip from the file.
            @test reader.last_output_time ≈ 1.5
            @test reader.output_count == 3
            # The checkpoint is restart #1, so a resumed writer must continue
            # at #2 instead of reusing and overwriting the file just loaded.
            @test reader.restart_count == 1
            @test reader.grid_file_written == true
            @test reader.last_restart_time ≈ 2.5

            # Field data round-trips exactly (serial / pencils=nothing path).
            @test haskey(restart_data, "temperature")
            @test size(restart_data["temperature"]) == size(T_in)
            @test maximum(abs.(restart_data["temperature"] .- T_in)) == 0.0

            @test haskey(restart_data, "magnetic_toroidal")
            @test maximum(abs.(restart_data["magnetic_toroidal"]["real"] .- btor_real)) == 0.0
            @test maximum(abs.(restart_data["magnetic_toroidal"]["imag"] .- btor_imag)) == 0.0
        end

        @testset "_load_restart_file reads an explicit restart path" begin
            reader = GeoDynamo.create_time_tracker(config, 0.0)
            restart_data, md = GeoDynamo._load_restart_file(
                found[1], reader, config; shtns_config = cfg)
            @test md["current_step"] == 42
            @test reader.output_count == 3
            @test reader.restart_count == 1
            @test maximum(abs.(restart_data["temperature"] .- T_in)) == 0.0

            @test_throws ErrorException GeoDynamo._load_restart_file(
                joinpath(tmpdir, "does_not_exist.nc"), reader, config)
        end

        @testset "read_restart! errors when no restart files exist" begin
            reader = GeoDynamo.create_time_tracker(config, 0.0)
            @test_throws ErrorException GeoDynamo.read_restart!(
                reader, mktempdir(), 0.0, config, nothing)
        end
    end

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_RESTART && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
