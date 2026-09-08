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

    @testset "solver restart snapshot preserves timestep state" begin
        params = GeoDynamo.SolverParameters(
            geometry = :shell,
            lmax = 3,
            mmax = 3,
            nlat = 10,
            nlon = 16,
            nr = 8,
            nr_inner = 4,
            radial_bandwidth = 3,
            radius_ratio = 0.35,
            Ek = 1e-3,
            Ra = 1e3,
            Pm = 1.0,
            Pr = 1.0,
            Sc = 1.0,
            timestep = 1e-4,
            timestepper = GeoDynamo.CNAB2(),
            include_magnetic = true,
            include_composition = true,
            magnetic_inner_bc = :conducting_inner_core,
            internal_heating = 2.5,
            compositional_source = 1.25,
        )
        source = GeoDynamo.initialize_solver_state(Float64; params)
        GeoDynamo.initialize_solver_fields!(source)

        function fill_spectral_pair!(field, value)
            fill!(parent(field.data_real), value)
            fill!(parent(field.data_imag), -value)
            return field
        end

        history_fields = (
            ("temperature_prev_nonlinear", source.fields.temperature.prev_nonlinear, 11.0),
            ("velocity_prev_nl_toroidal", source.fields.velocity.prev_nl_toroidal, 12.0),
            ("velocity_prev_nl_poloidal", source.fields.velocity.prev_nl_poloidal, 13.0),
            ("magnetic_prev_nl_toroidal", source.fields.magnetic.prev_nl_toroidal, 14.0),
            ("magnetic_prev_nl_poloidal", source.fields.magnetic.prev_nl_poloidal, 15.0),
            ("composition_prev_nonlinear", source.fields.composition.prev_nonlinear, 16.0),
        )
        for (_, field, value) in history_fields
            fill_spectral_pair!(field, value)
        end
        fill_spectral_pair!(source.fields.magnetic.toroidal_ic, 21.0)
        fill_spectral_pair!(source.fields.magnetic.poloidal_ic, 22.0)
        source.fields.temperature.internal_sources .= collect(31.0:38.0)
        source.fields.composition.internal_sources .= collect(41.0:48.0)
        source.runtime.timestep_state.needs_ab2_bootstrap = false

        snapshot = GeoDynamo.extract_all_fields(source)
        for (name, _, _) in history_fields
            @test haskey(snapshot, name)
        end
        @test haskey(snapshot, "magnetic_toroidal_ic")
        @test haskey(snapshot, "magnetic_poloidal_ic")
        @test haskey(snapshot, "temperature_internal_sources")
        @test haskey(snapshot, "composition_internal_sources")
        @test snapshot["needs_ab2_bootstrap"] === false

        restored = GeoDynamo.initialize_solver_state(Float64; params)
        @test all(iszero, restored.fields.temperature.internal_sources)
        @test all(iszero, restored.fields.composition.internal_sources)
        GeoDynamo.restore_fields_from_restart!(restored, snapshot)

        for (_, source_field, _) in history_fields
            restored_field = source_field === source.fields.temperature.prev_nonlinear ?
                             restored.fields.temperature.prev_nonlinear :
                             source_field === source.fields.velocity.prev_nl_toroidal ?
                             restored.fields.velocity.prev_nl_toroidal :
                             source_field === source.fields.velocity.prev_nl_poloidal ?
                             restored.fields.velocity.prev_nl_poloidal :
                             source_field === source.fields.magnetic.prev_nl_toroidal ?
                             restored.fields.magnetic.prev_nl_toroidal :
                             source_field === source.fields.magnetic.prev_nl_poloidal ?
                             restored.fields.magnetic.prev_nl_poloidal :
                             restored.fields.composition.prev_nonlinear
            @test parent(restored_field.data_real) == parent(source_field.data_real)
            @test parent(restored_field.data_imag) == parent(source_field.data_imag)
        end
        @test parent(restored.fields.magnetic.toroidal_ic.data_real) ==
              parent(source.fields.magnetic.toroidal_ic.data_real)
        @test parent(restored.fields.magnetic.toroidal_ic.data_imag) ==
              parent(source.fields.magnetic.toroidal_ic.data_imag)
        @test parent(restored.fields.magnetic.poloidal_ic.data_real) ==
              parent(source.fields.magnetic.poloidal_ic.data_real)
        @test parent(restored.fields.magnetic.poloidal_ic.data_imag) ==
              parent(source.fields.magnetic.poloidal_ic.data_imag)
        @test restored.runtime.timestep_state.needs_ab2_bootstrap === false
        @test restored.fields.temperature.internal_sources ==
              source.fields.temperature.internal_sources
        @test restored.fields.composition.internal_sources ==
              source.fields.composition.internal_sources
        @test restored.is_initialized

        # A restored CNAB2 state must take the same *next* step as the source.
        # This catches a checkpoint that copies the visible solution but drops
        # lagged nonlinear terms, source profiles, or conducting-inner-core state.
        GeoDynamo.solver_step!(source)
        GeoDynamo.solver_step!(restored)
        source_after_step = GeoDynamo.extract_all_fields(source)
        restored_after_step = GeoDynamo.extract_all_fields(restored)
        @test keys(restored_after_step) == keys(source_after_step)
        for (name, expected) in source_after_step
            actual = restored_after_step[name]
            if expected isa AbstractDict
                @test actual["real"] ≈ expected["real"] rtol = 1e-12 atol = 1e-12
                @test actual["imag"] ≈ expected["imag"] rtol = 1e-12 atol = 1e-12
            elseif expected isa AbstractArray
                @test actual ≈ expected rtol = 1e-12 atol = 1e-12
            else
                @test actual == expected
            end
        end
        @test restored.step == source.step
        @test restored.time == source.time
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
        nr_inner = 4
        @test nr_inner != nr
        restart_spectral = Dict(
            name => Dict("real" => randn(nlm, nr), "imag" => randn(nlm, nr))
            for name in (
                "temperature_prev_nonlinear",
                "velocity_prev_nl_toroidal",
                "velocity_prev_nl_poloidal",
                "magnetic_prev_nl_toroidal",
                "magnetic_prev_nl_poloidal",
                "composition_prev_nonlinear",
            )
        )
        restart_spectral["magnetic_toroidal_ic"] =
            Dict("real" => randn(nlm, nr_inner), "imag" => randn(nlm, nr_inner))
        restart_spectral["magnetic_poloidal_ic"] =
            Dict("real" => randn(nlm, nr_inner), "imag" => randn(nlm, nr_inner))

        fields = Dict{String, Any}(
            "temperature" => copy(T_in),
            "magnetic_toroidal" =>
                Dict("real" => copy(btor_real), "imag" => copy(btor_imag)),
            "temperature_internal_sources" => collect(51.0:58.0),
            "composition_internal_sources" => collect(61.0:68.0),
            "needs_ab2_bootstrap" => false,
            "previous_dt" => 3e-5,
        )
        merge!(fields, restart_spectral)

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

        @testset "restart files ignore lossy history layout selections" begin
            # A restart must contain both representations needed by
            # restore_fields_from_restart!, even when the caller reuses a
            # PHYSICAL_ONLY or SPECTRAL_ONLY history configuration.
            for layout in (GeoDynamo.PHYSICAL_ONLY, GeoDynamo.SPECTRAL_ONLY)
                layout_dir = mktempdir()
                layout_config = GeoDynamo.OutputConfig(
                    layout,
                    layout_dir,
                    config.filename_prefix,
                    config.include_metadata,
                    config.include_grid,
                    config.include_diagnostics,
                    config.output_precision,
                    config.spectral_lmax_output,
                    config.overwrite_files,
                    config.output_interval,
                    config.restart_interval,
                    config.max_output_time,
                    config.time_tolerance,
                )
                layout_tracker = GeoDynamo.create_time_tracker(layout_config, 0.0)
                GeoDynamo.write_restart!(fields, layout_tracker, metadata,
                    layout_config, nothing;
                    shtns_config = cfg, geometry = :shell,
                    radius_ratio = 0.35, radial_grid = radial_nodes)
                layout_path = only(GeoDynamo.find_restart_files(layout_dir, 0.0))
                NCDataset(layout_path, "r") do ds
                    @test haskey(ds, "temperature")
                    @test haskey(ds, "magnetic_toroidal_real")
                    @test haskey(ds, "temperature_prev_nonlinear_real")
                end
            end
        end

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
            for (name, expected) in restart_spectral
                @test haskey(restart_data, name)
                @test restart_data[name]["real"] == expected["real"]
                @test restart_data[name]["imag"] == expected["imag"]
            end
            @test restart_data["needs_ab2_bootstrap"] === false
            @test restart_data["previous_dt"] == 3e-5
            @test restart_data["temperature_internal_sources"] == collect(51.0:58.0)
            @test restart_data["composition_internal_sources"] == collect(61.0:68.0)
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


        @testset "AB2 bootstrap true survives a NetCDF round-trip" begin
            fields["needs_ab2_bootstrap"] = true
            tracker.restart_count = 1
            true_metadata = Dict{String, Any}(
                "current_time" => 3.0,
                "current_step" => 43,
            )
            GeoDynamo.write_restart!(fields, tracker, true_metadata, config, nothing;
                shtns_config = cfg, geometry = :shell,
                radius_ratio = 0.35, radial_grid = radial_nodes)
            true_path = only(filter(path -> endswith(path, "restart_2.nc"),
                GeoDynamo.find_restart_files(tmpdir, 0.0)))
            reader = GeoDynamo.create_time_tracker(config, 0.0)
            restart_data, _ = GeoDynamo._load_restart_file(
                true_path, reader, config; shtns_config = cfg)
            @test restart_data["needs_ab2_bootstrap"] === true
        end

        @testset "read_restart! errors when no restart files exist" begin
            reader = GeoDynamo.create_time_tracker(config, 0.0)
            @test_throws ErrorException GeoDynamo.read_restart!(
                reader, mktempdir(), 0.0, config, nothing)
        end

        @testset "NetCDF checkpoint reproduces the next CNAB2 step" begin
            trajectory_params = GeoDynamo.SolverParameters(
                geometry = :shell,
                lmax = 3,
                mmax = 3,
                nlat = 10,
                nlon = 16,
                nr = 8,
                nr_inner = 4,
                radial_bandwidth = 3,
                radius_ratio = 0.35,
                Ek = 1e-3,
                Ra = 1e3,
                Pm = 1.0,
                Pr = 1.0,
                Sc = 1.0,
                timestep = 1e-4,
                timestepper = GeoDynamo.CNAB2(),
                include_magnetic = true,
                include_composition = true,
                magnetic_inner_bc = :conducting_inner_core,
            )
            continuing = GeoDynamo.initialize_solver_state(
                Float64; params = trajectory_params)
            GeoDynamo.initialize_solver_fields!(continuing)
            continuing.fields.temperature.internal_sources .= collect(71.0:78.0)
            continuing.fields.composition.internal_sources .= collect(81.0:88.0)
            GeoDynamo.solver_step!(continuing)
            @test continuing.runtime.timestep_state.needs_ab2_bootstrap === false

            trajectory_fields = GeoDynamo.extract_all_fields(continuing)
            trajectory_dir = mktempdir()
            trajectory_config = GeoDynamo.OutputConfig(
                config.output_space,
                trajectory_dir,
                config.filename_prefix,
                config.include_metadata,
                config.include_grid,
                config.include_diagnostics,
                config.output_precision,
                config.spectral_lmax_output,
                config.overwrite_files,
                config.output_interval,
                config.restart_interval,
                config.max_output_time,
                config.time_tolerance,
            )
            trajectory_tracker = GeoDynamo.create_time_tracker(
                trajectory_config, continuing.time)
            trajectory_metadata = Dict{String, Any}(
                "current_time" => continuing.time,
                "current_step" => continuing.step,
            )
            trajectory_cfg = continuing.runtime.shtns_config
            trajectory_r = Float64.(continuing.runtime.outer_core_domain.r[
                1:continuing.runtime.outer_core_domain.N, 4])
            GeoDynamo.write_restart!(trajectory_fields, trajectory_tracker,
                trajectory_metadata, trajectory_config, trajectory_cfg.pencils;
                shtns_config = trajectory_cfg,
                geometry = :shell,
                radius_ratio = trajectory_params.radius_ratio,
                radial_grid = trajectory_r)

            resumed = GeoDynamo.initialize_solver_state(
                Float64; params = trajectory_params)
            resumed_cfg = resumed.runtime.shtns_config
            reader = GeoDynamo.create_time_tracker(trajectory_config, 0.0)
            loaded, loaded_metadata = GeoDynamo.read_restart!(reader,
                trajectory_dir, continuing.time, trajectory_config,
                resumed_cfg.pencils; shtns_config = resumed_cfg)
            GeoDynamo.restore_fields_from_restart!(resumed, loaded)
            GeoDynamo.reset_solver_clock!(resumed;
                time = loaded_metadata["current_time"],
                step = loaded_metadata["current_step"])

            @test size(parent(resumed.fields.magnetic.toroidal_ic.data_real), 3) == 4
            @test size(parent(resumed.fields.magnetic.toroidal.data_real), 3) == 8
            GeoDynamo.solver_step!(continuing)
            GeoDynamo.solver_step!(resumed)
            expected_after_restart = GeoDynamo.extract_all_fields(continuing)
            actual_after_restart = GeoDynamo.extract_all_fields(resumed)
            @test keys(actual_after_restart) == keys(expected_after_restart)
            for (name, expected) in expected_after_restart
                actual = actual_after_restart[name]
                if expected isa AbstractDict
                    @test actual["real"] ≈ expected["real"] rtol = 1e-12 atol = 1e-12
                    @test actual["imag"] ≈ expected["imag"] rtol = 1e-12 atol = 1e-12
                elseif expected isa AbstractArray
                    @test actual ≈ expected rtol = 1e-12 atol = 1e-12
                else
                    @test actual == expected
                end
            end
            @test resumed.step == continuing.step
            @test resumed.time == continuing.time
        end
    end

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_RESTART && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
