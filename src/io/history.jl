# ================================================================================
# Main Output Function
# ================================================================================

"""
    write_fields!(state, tracker, metadata, config, shtns_config, pencils)
    write_fields!(fields, tracker, metadata, config, shtns_config, pencils)

All-ranks entry point: synchronizes the output decision via MPI.Bcast, then
extracts fields only if output is actually needed.

The `state` overload accepts any runtime object for which `extract_all_fields`
is defined. The `fields` overload accepts an already extracted field dictionary.
Both overloads must be called by every rank on every timestep to avoid MPI
deadlocks in collective output decisions.
"""
function write_fields!(state, tracker::TimeTracker,
        metadata::Dict{String, Any}, config::OutputConfig = output_config_from_parameters(),
        shtns_config::Union{SHTnsKitConfig, Nothing} = nothing,
        pencils::Union{NamedTuple, Nothing} = nothing;
        geometry::Symbol = :shell,
        radius_ratio::Float64 = 0.35,
        radial_grid::Union{AbstractVector{<:Real}, Nothing} = nothing)
    comm = output_comm()
    current_time = metadata["current_time"]

    # Synchronize output decision across all ranks (collective)
    local_output = should_output_now(tracker, current_time, config) ? 1 : 0
    local_restart = should_restart_now(tracker, current_time, config) ? 1 : 0
    so = MPI.Bcast(Ref(local_output), 0, comm)[] != 0
    sr = MPI.Bcast(Ref(local_restart), 0, comm)[] != 0

    if !so && !sr
        return false
    end

    # Only now extract the (expensive) field copies
    fields = extract_all_fields(state)
    resolved_shtns_config = shtns_config === nothing && hasproperty(state, :runtime) ?
                            getproperty(state.runtime, :shtns_config) : shtns_config
    resolved_pencils = pencils === nothing && resolved_shtns_config !== nothing ?
                       resolved_shtns_config.pencils : pencils
    resolved_radial_grid = resolved_state_radial_grid(state, radial_grid)
    return write_fields!(
        fields, tracker, metadata, config, resolved_shtns_config, resolved_pencils;
        geometry = geometry, radius_ratio = radius_ratio, radial_grid = resolved_radial_grid)
end

"""
    resolved_state_radial_grid(state, radial_grid)

Return the true radial collocation nodes for the writer: an explicit
`radial_grid` wins, otherwise the outer-core `RadialDomain` carried on
`state.runtime.outer_core_domain` is used when present. Falls back to `nothing` so the
`fields`-only path keeps working without a runtime.
"""
function resolved_state_radial_grid(state, radial_grid)
    radial_grid !== nothing && return radial_grid
    hasproperty(state, :runtime) || return nothing
    rt = state.runtime
    hasproperty(rt, :outer_core_domain) || return nothing
    dom = getproperty(rt, :outer_core_domain)
    return Float64.(dom.r[1:dom.N, 4])
end

function write_fields!(fields::Dict{String, Any}, tracker::TimeTracker,
        metadata::Dict{String, Any}, config::OutputConfig = output_config_from_parameters(),
        shtns_config::Union{SHTnsKitConfig, Nothing} = nothing,
        pencils::Union{NamedTuple, Nothing} = nothing;
        geometry::Symbol = :shell,
        radius_ratio::Float64 = 0.35,
        radial_grid::Union{AbstractVector{<:Real}, Nothing} = nothing)
    comm = output_comm()
    rank = MPI.Comm_rank(comm)
    current_time = metadata["current_time"]
    current_step = metadata["current_step"]

    # When called from a state object, the decision is already made and this path
    # always writes. When called directly with a Dict, re-check.
    local_output = should_output_now(tracker, current_time, config) ? 1 : 0
    local_restart = should_restart_now(tracker, current_time, config) ? 1 : 0
    should_output = MPI.Bcast(Ref(local_output), 0, comm)[] != 0
    should_restart = MPI.Bcast(Ref(local_restart), 0, comm)[] != 0

    if !should_output && !should_restart
        return false
    end

    # Rank 0 ensures output directory exists
    if rank == 0 && !isdir(config.output_dir)
        try
            mkpath(config.output_dir)
            println("Rank 0: Created output directory: $(config.output_dir)")
        catch e
            error("Failed to create output directory '$(config.output_dir)': $e")
        end
        if !isdir(config.output_dir)
            error("Output directory '$(config.output_dir)' does not exist after mkpath()")
        end
    end
    MPI.Barrier(comm)

    # Extract field information
    field_info = extract_field_info(fields, shtns_config, pencils;
        radius_ratio = radius_ratio, radial_grid = radial_grid)

    # Write grid file once (rank 0 only)
    if !tracker.grid_file_written && config.include_grid
        write_grid_file!(config, field_info, shtns_config, metadata; geometry = geometry)
        tracker.grid_file_written = true
        MPI.Barrier(comm)
    end

    # Regular output - all ranks write collectively to single file
    if should_output
        output_number = tracker.output_count + 1
        filename = generate_filename(config, current_time, current_step, "output",
            output_number; geometry = geometry)

        if rank == 0
            println("Time $(round(current_time, digits=6)): Writing parallel output #$(output_number)")
        end

        write_start_time = time()

        # Compute diagnostics (local to each rank)
        diagnostics = compute_diagnostics(fields, field_info)
        diagnostics["output_time"] = current_time
        diagnostics["output_number"] = Float64(output_number)

        # Open file collectively
        ds = create_parallel_netcdf(
            filename, config, field_info, metadata, comm; geometry = geometry)

        try
            # Define dimensions and variables (collective)
            setup_dimensions!(ds, field_info, config)
            available_fields = collect(keys(fields))
            setup_variables!(ds, field_info, config, available_fields)
            setup_diagnostic_variables!(ds, diagnostics, config)

            # Write data (each rank writes its portion)
            write_coordinate_data!(ds, field_info, config)
            write_field_data!(ds, fields, config, field_info)
            write_time_data!(ds, current_time, current_step, config)
            write_diagnostics!(ds, diagnostics, config)

            # Flush all pending writes before close (critical for parallel MPI-IO)
            NCDatasets.sync(ds)
        finally
            close(ds)
        end

        write_duration = time() - write_start_time
        if rank == 0 && isfile(filename)
            file_size = filesize(filename) / (1024*1024)
            println("Parallel write complete: $(basename(filename)) " *
                    "($(round(file_size, digits=2)) MB in $(round(write_duration, digits=2))s)")
        end
    end

    # Restart file
    if should_restart
        write_restart!(fields, tracker, metadata, config, pencils;
            shtns_config = shtns_config,
            geometry = geometry, radius_ratio = radius_ratio)
    end

    # Update tracker
    update_tracker!(tracker, current_time, config, should_output, should_restart)

    if rank == 0 && should_output
        println("Next output: $(round(tracker.next_output_time, digits=6))")
    end

    return should_output || should_restart
end
