# ================================================================================
# Restart Functions (Parallel I/O)
# ================================================================================

"""
    _scan_output_count(dir, prefix, geometry, kind) -> Int

Highest numbered `<prefix>_<geometry>_<kind>_N.nc` file present in `dir`, or 0
when `dir` does not exist. Directory access failures are reported to the caller;
they must not be mistaken for an empty directory because doing so can reuse an
existing output number and overwrite data.

Purely local and MPI-free on purpose: each caller decides how to make the answer
unanimous (rank 0 scans, then broadcasts), and keeping the scan itself collective-free
is what lets it be tested serially.
"""
function _scan_output_count(dir::String, prefix::String, geometry::Symbol, kind::String)
    isdir(dir) || return 0
    # `prefix` is the free-form `OutputConfig.filename_prefix`, so it has to be escaped
    # before it goes into a pattern: "run(1)" would add a second capture group and make
    # `only(matched.captures)` throw, and "run[" would fail in the Regex constructor —
    # both from inside a checkpoint write, aborting the run.
    quoted = replace(prefix, r"([\\^$.|?*+()\[\]{}])" => s"\\\1")
    pattern = Regex("^$(quoted)_$(geometry)_$(kind)_(\\d+)\\.nc\\z")
    count = 0
    for filename in readdir(dir)
        matched = match(pattern, filename)
        matched === nothing && continue
        parsed = tryparse(Int, only(matched.captures))
        parsed === nothing || (count = max(count, parsed))
    end
    return count
end

"""
    _collective_scan_output_count(dir, prefix, geometry, kind, comm) -> Int

Scan on rank 0 and broadcast either the result or the scan failure. Every rank
therefore leaves this collective along the same path, and an unreadable output
directory cannot silently masquerade as an empty one.
"""
function _collective_scan_output_count(dir::String, prefix::String,
        geometry::Symbol, kind::String, comm)
    return root_value(comm, "Could not scan output directory '$dir' for numbered $kind files") do
        _scan_output_count(dir, prefix, geometry, kind)
    end
end

"""
    _persisted_output_count(tracker, did_output, config, geometry;
                            history_dirs=()) -> Int

The history-file count to record in a restart file: the larger of what `tracker`
believes and what the run's history directories actually contain.

`tracker` alone is not enough because it is not always the tracker that produced the
history files. `CheckpointWriter` writes its checkpoints through its OWN private
tracker, built from a config whose `output_interval` is `Inf`, so that tracker sits at
`output_count == 0` for the life of the run no matter how many history files the run
emitted. A resume that trusted it started numbering at 1 again and — with
`overwrite_files = true` — deleted the existing `hist_1`. The directory is the one
witness that is right on both paths.

Which directory, though, is not `config.output_dir` on the checkpoint path:
`_run_output_writer!(::CheckpointWriter, ...)` (api/output_writers.jl) builds that
config with `output_dir = ow.path`, the CHECKPOINT directory, which holds no
`<prefix>_<geometry>_hist_N.nc` at all. `history_dirs` carries the run's real history
directories (`_history_output_dirs`); empty means "scan `config.output_dir`", which is
right for the legacy `write_fields!` path where the two are the same directory.

Rank 0 scans and broadcasts, so every rank writes the same value into the file — one
collective per directory, hence the requirement that `history_dirs` be identical on
every rank (`_freeze_output_writers!` enforces it at construction).
"""
function _persisted_output_count(tracker::TimeTracker, did_output::Bool,
        config::OutputConfig, geometry::Symbol;
        history_dirs = ())
    from_tracker = tracker.output_count + (did_output ? 1 : 0)
    comm = MPI.Initialized() ? output_comm() : nothing
    dirs = isempty(history_dirs) ? (config.output_dir,) : history_dirs
    on_disk = 0
    for dir in dirs
        on_disk = max(on_disk, _collective_scan_output_count(
            dir, config.filename_prefix, geometry, "hist", comm))
    end
    return max(from_tracker, on_disk)
end

"""
    _require_restart_file_everywhere(filepath, comm)

Verify that EVERY rank can see `filepath`, and raise on every rank if any cannot.

Checking on rank 0 and broadcasting catches "missing everywhere", but not the case the
path selection is guarded against in the first place: a checkpoint sitting on node-local
scratch, visible to rank 0 and to nothing else. Those ranks would fail alone inside the
collective `NCDataset` open while the ranks that can see the file block inside it — the
hang, arriving as an opaque NetCDF error. Reducing the per-rank `isfile` turns it into a
unanimous, legible abort.

Collective; every rank must call it together.
"""
function _require_restart_file_everywhere(filepath::String, comm)
    missing_here = !isfile(filepath)
    missing_anywhere = any_rank(missing_here, comm)
    missing_anywhere || return nothing
    rank = (comm === nothing || !MPI.Initialized()) ? 0 : MPI.Comm_rank(comm)
    detail = missing_here ? "not visible on rank $rank" :
             "visible on rank $rank but not on every rank (node-local scratch?)"
    error("Restart file unusable: $detail: $filepath")
end

"""
    _restart_path_for_all_ranks(restart_dir, restart_time) -> String

The restart file every rank must open, chosen once by rank 0 and broadcast.

`find_restart_files` is a rank-LOCAL `readdir` plus an mtime/stored-time sort, and the
path it returns is handed straight to the COLLECTIVE `NCDataset(comm, ...)` open in
`_load_restart_file`. On node-local scratch, or an NFS mount with stale attribute
caching, two ranks can list the directory differently: they then collectively open
DIFFERENT files — hanging in MPI-IO, or silently mixing two checkpoints — and a rank
whose listing came back empty raises alone while the others block in the open. Choosing
once on rank 0 removes both, and stops every rank from re-opening each candidate to read
its stored time.

The not-found `error` is raised on every rank, so an absent checkpoint aborts the run
instead of deadlocking it. That requires rank 0's scan to be TOTAL: `find_restart_files`
opens with a bare `readdir`, which throws `SystemError` on a missing, unmounted or
unreadable directory — and a throw on rank 0 alone leaves the others blocked in the
broadcast below forever. Every such failure is therefore folded into "no candidates",
which the broadcast turns into a unanimous error.
"""
function _restart_path_for_all_ranks(restart_dir::String, restart_time::Float64)
    comm = MPI.Initialized() ? output_comm() : nothing
    selected = root_value(comm, "Selecting restart file") do
        candidates = try
            isdir(restart_dir) ? find_restart_files(restart_dir, restart_time) : String[]
        catch err
            @warn "Restart directory could not be scanned" restart_dir exception = err
            String[]
        end
        isempty(candidates) ? "" : candidates[1]
    end
    isempty(selected) && error(
        "No readable restart files found near time $restart_time in $restart_dir")
    return selected
end

function _write_restart_tracker_data_collectively!(ds,
        persisted_last_output_time::Real, persisted_output_count::Integer,
        restart_number::Integer, grid_file_written::Bool,
        config::OutputConfig, comm)
    run_on_root!(comm, "Writing NetCDF restart tracker data") do
        ds["last_output_time"][1] = config.output_precision(persisted_last_output_time)
        ds["output_count"][1] = Int32(persisted_output_count)
        ds["restart_count"][1] = Int32(restart_number)
        ds["grid_file_written"][1] = Int32(grid_file_written ? 1 : 0)
    end
    return nothing
end

"""
    _lossless_restart_config(config) -> OutputConfig

Return a restart-specific copy of `config` that always writes the mixed
physical/spectral layout. History output may intentionally select only one
representation, but a checkpoint cannot inherit that lossy choice because
`restore_fields_from_restart!` needs both parts of the solver state.
"""
function _lossless_restart_config(config::OutputConfig)
    config.output_space == MIXED_FIELDS && return config
    return OutputConfig(
        MIXED_FIELDS,
        config.output_dir,
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
end

"""
    write_restart!(fields, tracker, metadata, config[, pencils];
                   shtns_config=nothing, geometry=:shell, radius_ratio=0.35,
                   did_output=false)

Write a restart NetCDF file through the parallel field-I/O path. Restart files
always use the lossless mixed physical/spectral layout, even when `config`
selects a single representation for ordinary history output.

The restart file also stores enough `TimeTracker` state for a resumed run to
continue output and restart numbering without clobbering earlier files.
Set `did_output=true` when a history file was successfully emitted immediately
before this checkpoint so the persisted tracker includes that file as well.

The persisted history count is `max` of what `tracker` knows and what the run's
history directories actually hold — see `_persisted_output_count` for why the tracker
alone is not enough. Pass `history_dirs` when `config.output_dir` is NOT where the
history files live, as it is not for a `CheckpointWriter`.
"""
function write_restart!(fields::Dict{String, Any}, tracker::TimeTracker,
        metadata::Dict{String, Any}, config::OutputConfig,
        pencils::Union{NamedTuple, Nothing} = nothing;
        shtns_config::Union{SHTnsKitConfig, Nothing} = nothing,
        geometry::Symbol = :shell,
        radius_ratio::Float64 = 0.35,
        radial_grid::Union{AbstractVector{<:Real}, Nothing} = nothing,
        did_output::Bool = false,
        history_dirs = ())
    # Checkpoints are solver state, not presentation output. Do not let a
    # history-only layout silently omit fields that the restart loader requires.
    config = _lossless_restart_config(config)
    comm = output_comm()
    rank = MPI.Comm_rank(comm)
    current_time = metadata["current_time"]
    current_step = metadata["current_step"]

    restart_number = tracker.restart_count + 1
    persisted_output_count = _persisted_output_count(tracker, did_output, config, geometry;
        history_dirs = history_dirs)
    persisted_last_output_time = did_output ? current_time : tracker.last_output_time
    filename = generate_filename(
        config, current_time, current_step, "restart", restart_number; geometry = geometry)

    if rank == 0
        println("Writing parallel restart #$(restart_number): $(basename(filename))")
    end

    field_info = extract_field_info(fields, shtns_config, pencils;
        radius_ratio = radius_ratio, radial_grid = radial_grid)

    restart_metadata = copy(metadata)
    restart_metadata["restart_time"] = current_time
    restart_metadata["last_output_time"] = persisted_last_output_time
    restart_metadata["output_count"] = persisted_output_count
    restart_metadata["restart_count"] = restart_number

    ds = create_parallel_netcdf(
        filename, config, field_info, restart_metadata, comm; geometry = geometry)

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
        setup_restart_variables!(ds, fields, field_info, config)

        # Leaving define mode is collective for parallel NetCDF. Do it on
        # every rank before any root-only payload write can fail.
        NCDatasets.sync(ds)

        # Write data
        write_coordinate_data!(ds, field_info, config)
        write_field_data!(ds, fields, config, field_info)
        write_restart_field_data!(ds, fields, config, field_info)
        write_time_data!(ds, current_time, current_step, config)

        # Restart-specific data are root-written but failure is collective, so
        # every rank either reaches the sync below or aborts before it.
        _write_restart_tracker_data_collectively!(ds,
            persisted_last_output_time, persisted_output_count,
            restart_number, tracker.grid_file_written, config, comm)

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
        pencils::Union{NamedTuple, Nothing} = nothing;
        shtns_config::Union{SHTnsKitConfig, Nothing} = nothing)
    comm = output_comm()
    rank = MPI.Comm_rank(comm)

    filename = _restart_path_for_all_ranks(restart_dir, restart_time)

    # Locate-then-delegate. This function used to carry its own ~90-line copy of the
    # read — same metadata reads, same tracker restoration, same per-field slicing,
    # differing only in `&&`-chained vs nested-`if` form — so a field added to one
    # reader, or a slicing fix applied to one, silently skipped the other depending on
    # which entry point the caller used. Locating the file is the only thing that is
    # actually specific to this entry point.
    return _load_restart_file(filename, tracker, config;
        pencils = pencils, shtns_config = shtns_config)
end

"""
    _load_restart_file(filepath, tracker, config; pencils=nothing)

Load simulation state from a specific restart NetCDF file path using parallel I/O.
All ranks open the file collectively and read their local slices.
"""
function _load_restart_file(filepath::String, tracker::TimeTracker, config::OutputConfig;
        pencils::Union{NamedTuple, Nothing} = nothing,
        shtns_config::Union{SHTnsKitConfig, Nothing} = nothing)
    comm = output_comm()
    rank = MPI.Comm_rank(comm)

    # Unanimous for the same reason the path is: this check sits directly in front of the
    # collective `NCDataset` open below, so a rank-local verdict lets one rank abort while
    # the others block in the open.
    _require_restart_file_everywhere(filepath, comm)

    if rank == 0
        println("Loading parallel restart from $(basename(filepath))")
    end

    restart_data = Dict{String, Any}()
    metadata = Dict{String, Any}()

    ds = NCDataset(comm, filepath, "r"; info = MPI.Info())

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
                r_range = range_local(pencils.r, 3)  # r is distributed under r×θ (Phase 2)
                restart_data["temperature"] = Array(ds["temperature"][θ_range, φ_range, r_range])
            else
                restart_data["temperature"] = Array(ds["temperature"][:, :, :])
            end
        end

        if haskey(ds, "composition")
            if pencils !== nothing
                θ_range = range_local(pencils.r, 1)
                φ_range = range_local(pencils.r, 2)
                r_range = range_local(pencils.r, 3)  # r is distributed under r×θ (Phase 2)
                restart_data["composition"] = Array(ds["composition"][θ_range, φ_range, r_range])
            else
                restart_data["composition"] = Array(ds["composition"][:, :, :])
            end
        end

        for component in ("velocity_toroidal", "velocity_poloidal",
            "magnetic_toroidal", "magnetic_poloidal",
            "temperature_spectral", "composition_spectral",
            RESTART_HISTORY_SPECTRAL_COMPONENTS...)
            real_name = "$(component)_real"
            imag_name = "$(component)_imag"

            if haskey(ds, real_name) && haskey(ds, imag_name)
                if pencils !== nothing
                    if shtns_config !== nothing
                        mode_indices = local_spectral_mode_indices(shtns_config)
                        r_range = range_local(shtns_config.pencils.spec, 3)
                        real_slice = read_local_spectral_coefficients(ds[real_name], mode_indices, r_range)
                        imag_slice = read_local_spectral_coefficients(ds[imag_name], mode_indices, r_range)
                        real_data,
                        imag_data = unpack_local_spectral_coefficients(real_slice, imag_slice, shtns_config)
                        restart_data[component] = Dict(
                            "real" => real_data,
                            "imag" => imag_data
                        )
                    else
                        lm_range, r_range = _legacy_linear_spectral_io_ranges(pencils)
                        real_slice = Array(ds[real_name][lm_range, r_range])
                        imag_slice = Array(ds[imag_name][lm_range, r_range])
                        restart_data[component] = Dict(
                            "real" => real_slice,
                            "imag" => imag_slice
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

        for component in RESTART_INNER_CORE_SPECTRAL_COMPONENTS
            real_name = "$(component)_real"
            imag_name = "$(component)_imag"
            if haskey(ds, real_name) && haskey(ds, imag_name)
                if pencils !== nothing
                    nr_inner = size(ds[real_name], 2)
                    r_inner_range = 1:nr_inner
                    if shtns_config !== nothing
                        mode_indices = local_spectral_mode_indices(shtns_config)
                        real_slice = read_local_spectral_coefficients(
                            ds[real_name], mode_indices, r_inner_range)
                        imag_slice = read_local_spectral_coefficients(
                            ds[imag_name], mode_indices, r_inner_range)
                        real_data, imag_data =
                            unpack_local_inner_core_spectral_coefficients(
                                real_slice, imag_slice, shtns_config)
                        restart_data[component] = Dict(
                            "real" => real_data,
                            "imag" => imag_data,
                        )
                    else
                        lm_range, _ = _legacy_linear_spectral_io_ranges(pencils)
                        restart_data[component] = Dict(
                            "real" => Array(ds[real_name][lm_range, r_inner_range]),
                            "imag" => Array(ds[imag_name][lm_range, r_inner_range]),
                        )
                    end
                else
                    restart_data[component] = Dict(
                        "real" => Array(ds[real_name][:, :]),
                        "imag" => Array(ds[imag_name][:, :]),
                    )
                end
            end
        end

        for profile in RESTART_SOURCE_PROFILES
            if haskey(ds, profile)
                restart_data[profile] = Array(ds[profile][:])
            end
        end
        if haskey(ds, "needs_ab2_bootstrap")
            restart_data["needs_ab2_bootstrap"] =
                ds["needs_ab2_bootstrap"][1] != 0
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

Return restart NetCDF files ordered best-match first.

`target_time <= 0` is the "restart from the latest checkpoint" sentinel and
orders files newest-first by modification time (cheap; no files opened). A
positive `target_time` is honored: each checkpoint's stored simulation `time`
is read and files are ordered by closeness to `target_time`, so a requested
restart time actually selects the nearest snapshot rather than the newest file.
Files whose stored time cannot be read fall back to the end (newest-first).
"""
function find_restart_files(restart_dir::String, target_time::Float64)
    files = readdir(restart_dir)
    # Match single-file restart pattern: geodynamo_shell_restart_N.nc
    restart_files = filter(f -> endswith(f, ".nc") && contains(f, "restart"), files)

    if isempty(restart_files)
        return String[]
    end

    full_paths = [joinpath(restart_dir, f) for f in restart_files]

    # Sentinel: caller wants the most recent checkpoint, no need to open files.
    if target_time <= 0
        sort!(full_paths, by = mtime, rev = true)
        return full_paths
    end

    # Honor a meaningful target_time by reading each checkpoint's stored time.
    stored_time = function (path)
        try
            return NCDataset(path, "r") do ds
                if haskey(ds, "time")
                    return Float64(ds["time"][1])
                elseif haskey(ds.attrib, "current_time")
                    return Float64(ds.attrib["current_time"])
                end
                return nothing
            end
        catch
            return nothing
        end
    end

    times = Dict(p => stored_time(p) for p in full_paths)
    known = [p for p in full_paths if times[p] !== nothing]
    unknown = [p for p in full_paths if times[p] === nothing]
    sort!(known, by = p -> abs(times[p] - target_time))
    sort!(unknown, by = mtime, rev = true)
    return vcat(known, unknown)
end
