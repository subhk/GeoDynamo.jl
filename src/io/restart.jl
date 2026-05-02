# ================================================================================
# Restart Functions (Parallel I/O)
# ================================================================================

"""
    write_restart!(fields, tracker, metadata, config[, pencils]; shtns_config=nothing, geometry=:shell, radius_ratio=0.35)

Write a restart NetCDF file using the same parallel field layout as history
output.

The restart file also stores enough `TimeTracker` state for a resumed run to
continue output and restart numbering without clobbering earlier files.
"""
function write_restart!(fields::Dict{String,Any}, tracker::TimeTracker,
                        metadata::Dict{String,Any}, config::OutputConfig,
                        pencils::Union{NamedTuple,Nothing}=nothing;
                        shtns_config::Union{SHTnsKitConfig,Nothing}=nothing,
                        geometry::Symbol = :shell,
                        radius_ratio::Float64 = 0.35)
    comm = output_comm()
    rank = MPI.Comm_rank(comm)
    current_time = metadata["current_time"]
    current_step = metadata["current_step"]

    restart_number = tracker.restart_count + 1
    filename = generate_filename(config, current_time, current_step, "restart", restart_number; geometry=geometry)

    if rank == 0
        println("Writing parallel restart #$(restart_number): $(basename(filename))")
    end

    field_info = extract_field_info(fields, shtns_config, pencils; radius_ratio=radius_ratio)

    restart_metadata = copy(metadata)
    restart_metadata["restart_time"] = current_time
    restart_metadata["last_output_time"] = tracker.last_output_time
    restart_metadata["output_count"] = tracker.output_count
    restart_metadata["restart_count"] = tracker.restart_count

    ds = create_parallel_netcdf(filename, config, field_info, restart_metadata, comm; geometry=geometry)

    try
        setup_dimensions!(ds, field_info, config)
        available_fields = collect(keys(fields))
        setup_variables!(ds, field_info, config, available_fields)

        # Additional restart variables (scalar, written by rank 0)
        defDim(ds, "scalar", 1)
        defVar(ds, "last_output_time", config.output_precision, ("scalar",))
        defVar(ds, "output_count", Int32, ("scalar",))
        defVar(ds, "restart_count", Int32, ("scalar",))
        defVar(ds, "grid_file_written", Int32, ("scalar",))

        # Write data
        write_coordinate_data!(ds, field_info, config)
        write_field_data!(ds, fields, config, field_info)
        write_time_data!(ds, current_time, current_step, config)

        # Restart-specific data (rank 0 only)
        if rank == 0
            ds["last_output_time"][1] = config.output_precision(tracker.last_output_time)
            ds["output_count"][1] = Int32(tracker.output_count)
            ds["restart_count"][1] = Int32(tracker.restart_count)
            ds["grid_file_written"][1] = Int32(tracker.grid_file_written ? 1 : 0)
        end

        # Flush all pending writes before close (critical for restart integrity)
        NCDatasets.sync(ds)
    finally
        close(ds)
    end
end

"""
    read_restart!(tracker, restart_dir, restart_time, config[, pencils]; shtns_config=nothing)

Find and read a restart file near `restart_time`.

All ranks collectively open the selected file. With pencils, each rank reads
only its local slices; without pencils, each rank reads full field arrays. The
passed tracker is updated from the restart metadata.
"""
function read_restart!(tracker::TimeTracker, restart_dir::String,
                        restart_time::Float64, config::OutputConfig,
                        pencils::Union{NamedTuple,Nothing}=nothing;
                        shtns_config::Union{SHTnsKitConfig,Nothing}=nothing)
    comm = output_comm()
    rank = MPI.Comm_rank(comm)

    restart_files = find_restart_files(restart_dir, restart_time)

    if isempty(restart_files)
        error("No restart files found near time $restart_time in $restart_dir")
    end

    filename = restart_files[1]
    if rank == 0
        println("Reading parallel restart: $(basename(filename))")
    end

    restart_data = Dict{String, Any}()
    metadata = Dict{String, Any}()

    # All ranks open the file collectively for reading
    ds = NCDataset(filename, "r"; comm=comm, info=MPI.Info())

    try
        # Read metadata (all ranks read scalars)
        if haskey(ds, "time")
            metadata["current_time"] = Float64(ds["time"][1])
        end
        if haskey(ds, "step")
            metadata["current_step"] = Int(ds["step"][1])
        end

        # Read tracker state
        if haskey(ds, "last_output_time")
            tracker.last_output_time = Float64(ds["last_output_time"][1])
        end
        if haskey(ds, "output_count")
            tracker.output_count = Int(ds["output_count"][1])
        end
        if haskey(ds, "restart_count")
            tracker.restart_count = Int(ds["restart_count"][1])
        end
        if haskey(ds, "grid_file_written")
            tracker.grid_file_written = ds["grid_file_written"][1] != 0
        end
        if haskey(metadata, "current_time")
            tracker.last_restart_time = metadata["current_time"]
        end
        tracker.next_output_time = tracker.last_output_time + config.output_interval
        tracker.next_restart_time = tracker.last_restart_time + config.restart_interval

        # Read fields - each rank reads its local slice
        if haskey(ds, "temperature") && pencils !== nothing
            θ_range = range_local(pencils.r, 1)
            φ_range = range_local(pencils.r, 2)
            restart_data["temperature"] = Array(ds["temperature"][θ_range, φ_range, :])
        elseif haskey(ds, "temperature")
            restart_data["temperature"] = Array(ds["temperature"][:, :, :])
        end

        if haskey(ds, "composition") && pencils !== nothing
            θ_range = range_local(pencils.r, 1)
            φ_range = range_local(pencils.r, 2)
            restart_data["composition"] = Array(ds["composition"][θ_range, φ_range, :])
        elseif haskey(ds, "composition")
            restart_data["composition"] = Array(ds["composition"][:, :, :])
        end

        for component in ["velocity_toroidal", "velocity_poloidal",
                          "magnetic_toroidal", "magnetic_poloidal",
                          "temperature_spectral", "composition_spectral"]
            real_name = "$(component)_real"
            imag_name = "$(component)_imag"

            if haskey(ds, real_name) && haskey(ds, imag_name)
                if pencils !== nothing
                    lm_range = range_local(pencils.spec, 1)
                    r_range = range_local(pencils.spec, 3)
                    real_slice = Array(ds[real_name][lm_range, r_range])
                    imag_slice = Array(ds[imag_name][lm_range, r_range])
                    if shtns_config !== nothing
                        real_data, imag_data = unpack_local_spectral_coefficients(real_slice, imag_slice, shtns_config)
                        restart_data[component] = Dict(
                            "real" => real_data,
                            "imag" => imag_data,
                        )
                    else
                        restart_data[component] = Dict(
                            "real" => real_slice,
                            "imag" => imag_slice,
                        )
                    end
                else
                    restart_data[component] = Dict(
                        "real" => Array(ds[real_name][:, :]),
                        "imag" => Array(ds[imag_name][:, :])
                    )
                end
            end
        end
    finally
        close(ds)
    end

    if rank == 0
        println("Restart completed from time $(get(metadata, "current_time", "unknown"))")
    end

    return restart_data, metadata
end

"""
    _load_restart_file(filepath, tracker, config; pencils=nothing)

Load simulation state from a specific restart NetCDF file path using parallel I/O.
All ranks open the file collectively and read their local slices.
"""
function _load_restart_file(filepath::String, tracker::TimeTracker, config::OutputConfig;
                           pencils::Union{NamedTuple,Nothing}=nothing,
                           shtns_config::Union{SHTnsKitConfig,Nothing}=nothing)
    comm = output_comm()
    rank = MPI.Comm_rank(comm)

    if !isfile(filepath)
        error("Rank $rank: Restart file not found: $filepath")
    end

    if rank == 0
        println("Loading parallel restart from $(basename(filepath))")
    end

    restart_data = Dict{String, Any}()
    metadata = Dict{String, Any}()

    ds = NCDataset(filepath, "r"; comm=comm, info=MPI.Info())

    try
        # Read time and step metadata
        if haskey(ds, "time")
            metadata["current_time"] = Float64(ds["time"][1])
        end
        if haskey(ds, "step")
            metadata["current_step"] = Int(ds["step"][1])
        end

        # Read tracker state
        if haskey(ds, "last_output_time")
            tracker.last_output_time = Float64(ds["last_output_time"][1])
        end
        if haskey(ds, "output_count")
            tracker.output_count = Int(ds["output_count"][1])
        end
        if haskey(ds, "restart_count")
            tracker.restart_count = Int(ds["restart_count"][1])
        end
        if haskey(ds, "grid_file_written")
            tracker.grid_file_written = ds["grid_file_written"][1] != 0
        end
        if haskey(metadata, "current_time")
            tracker.last_restart_time = metadata["current_time"]
        end
        tracker.next_output_time = tracker.last_output_time + config.output_interval
        tracker.next_restart_time = tracker.last_restart_time + config.restart_interval

        # Read fields with parallel slicing
        if haskey(ds, "temperature")
            if pencils !== nothing
                θ_range = range_local(pencils.r, 1)
                φ_range = range_local(pencils.r, 2)
                restart_data["temperature"] = Array(ds["temperature"][θ_range, φ_range, :])
            else
                restart_data["temperature"] = Array(ds["temperature"][:, :, :])
            end
        end

        if haskey(ds, "composition")
            if pencils !== nothing
                θ_range = range_local(pencils.r, 1)
                φ_range = range_local(pencils.r, 2)
                restart_data["composition"] = Array(ds["composition"][θ_range, φ_range, :])
            else
                restart_data["composition"] = Array(ds["composition"][:, :, :])
            end
        end

        for component in ["velocity_toroidal", "velocity_poloidal",
                          "magnetic_toroidal", "magnetic_poloidal",
                          "temperature_spectral", "composition_spectral"]
            real_name = "$(component)_real"
            imag_name = "$(component)_imag"

            if haskey(ds, real_name) && haskey(ds, imag_name)
                if pencils !== nothing
                    lm_range = range_local(pencils.spec, 1)
                    r_range = range_local(pencils.spec, 3)
                    real_slice = Array(ds[real_name][lm_range, r_range])
                    imag_slice = Array(ds[imag_name][lm_range, r_range])
                    if shtns_config !== nothing
                        real_data, imag_data = unpack_local_spectral_coefficients(real_slice, imag_slice, shtns_config)
                        restart_data[component] = Dict(
                            "real" => real_data,
                            "imag" => imag_data,
                        )
                    else
                        restart_data[component] = Dict(
                            "real" => real_slice,
                            "imag" => imag_slice,
                        )
                    end
                else
                    restart_data[component] = Dict(
                        "real" => Array(ds[real_name][:, :]),
                        "imag" => Array(ds[imag_name][:, :])
                    )
                end
            end
        end
    finally
        close(ds)
    end

    if rank == 0
        println("Restart loaded: time=$(get(metadata, "current_time", "unknown")), " *
                "step=$(get(metadata, "current_step", "unknown"))")
    end

    return restart_data, metadata
end

# ================================================================================
# Utility Functions
# ================================================================================

"""
    find_restart_files(restart_dir, target_time)

Return restart NetCDF files in newest-first order.

`target_time` is accepted for API compatibility; current restart filenames are
counter-based, so modification time is used as the selection proxy.
"""
function find_restart_files(restart_dir::String, target_time::Float64)
    files = readdir(restart_dir)
    # Match single-file restart pattern: geodynamo_shell_restart_N.nc
    restart_files = filter(f -> endswith(f, ".nc") && contains(f, "restart"), files)

    if isempty(restart_files)
        return String[]
    end

    # Sort by modification time (most recent first) as a proxy for closest time
    full_paths = [joinpath(restart_dir, f) for f in restart_files]
    sort!(full_paths, by=mtime, rev=true)
    return full_paths
end
