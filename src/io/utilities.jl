"""
    validate_output(filename)

Return `true` when `filename` can be opened as NetCDF and has the minimum
history-file variables required by the output reader helpers.
"""
function validate_output(filename::String)
    try
        NCDataset(filename, "r") do ds
            for var in ["time", "step", "r"]
                if !haskey(ds, var)
                    return false
                end
            end
        end
        return true
    catch
        return false
    end
end

"""
    verify_all_ranks_wrote(output_dir, hist_number; geometry="shell", expected_dims=nothing)

Verify that the single parallel output file exists for a given history number
and has correct global dimensions.
"""
function verify_all_ranks_wrote(output_dir::String, hist_number::Int;
        geometry::String = "shell",
        expected_dims::Union{Dict{String, Int}, Nothing} = nothing)
    # Look for the single shared file
    pattern = Regex("$(geometry)_hist_$(hist_number)\\.nc")
    files = readdir(output_dir)
    matching = filter(f -> occursin(pattern, f), files)

    if isempty(matching)
        return (false,
            ["File not found"],
            Dict{String, Any}(
                "hist_number" => hist_number, "error" => "No matching file found"))
    end

    filepath = joinpath(output_dir, matching[1])
    info = Dict{String, Any}(
        "hist_number" => hist_number,
        "file" => filepath,
        "file_size_MB" => filesize(filepath) / (1024*1024)
    )

    # Verify dimensions if expected values provided
    if expected_dims !== nothing
        # A bare `return` from inside the `do` closure only exits the closure, so
        # the mismatch verdict is captured here and propagated after the block.
        mismatch_msg = nothing
        try
            NCDataset(filepath, "r") do ds
                for (dim_name, expected_size) in expected_dims
                    if haskey(ds.dim, dim_name)
                        actual_size = ds.dim[dim_name]
                        if actual_size != expected_size
                            info["error"] = "Dimension $dim_name: expected $expected_size, got $actual_size"
                            mismatch_msg = "Dimension mismatch"
                            return
                        end
                    end
                end
                info["dimensions"] = Dict(String(k) => v for (k, v) in ds.dim)
                info["variables"] = collect(keys(ds))
            end
        catch e
            info["error"] = string(e)
            return (false, ["Read error"], info)
        end
        if mismatch_msg !== nothing
            return (false, [mismatch_msg], info)
        end
    end

    return (true, String[], info)
end

"""
    cleanup_old_files(output_dir, keep_last_n=10)

Remove older history files from `output_dir`, keeping the most recent
`keep_last_n` files by history number.
"""
function cleanup_old_files(output_dir::String, keep_last_n::Int = 10)
    files = readdir(output_dir)
    # Match hist_N.nc pattern (no rank in name)
    hist_pattern = r"_hist_(\d+)\.nc$"
    output_files = filter(f -> occursin(hist_pattern, f), files)

    if isempty(output_files)
        return
    end

    # Extract history numbers
    file_numbers = Tuple{String, Int}[]
    for file in output_files
        m = match(hist_pattern, file)
        if m !== nothing
            push!(file_numbers, (file, parse(Int, m.captures[1])))
        end
    end

    sort!(file_numbers, by = x -> x[2], rev = true)

    if length(file_numbers) <= keep_last_n
        return
    end

    for (file, _) in file_numbers[(keep_last_n + 1):end]
        try
            rm(joinpath(output_dir, file))
            println("Removed old file: $file")
        catch e
            @warn "Failed to remove file: $file"
        end
    end

    println("Cleanup completed. Kept last $keep_last_n output files.")
end

"""
    get_time_series(output_dir)

Read and return sorted simulation times from history NetCDF files in
`output_dir`.

Unreadable files are skipped so a partially written or unrelated file does not
break post-processing.
"""
function get_time_series(output_dir::String)
    files = readdir(output_dir)
    hist_pattern = r"_hist_(\d+)\.nc$"
    hist_files = filter(f -> occursin(hist_pattern, f) && !contains(f, "restart"), files)

    times = Float64[]
    for file in hist_files
        filepath = joinpath(output_dir, file)
        try
            NCDataset(filepath, "r") do ds
                if haskey(ds, "time")
                    push!(times, Float64(ds["time"][1]))
                end
            end
        catch
            continue
        end
    end

    return sort(times)
end

"""
    get_file_info(filename)

Return basic metadata for one NetCDF output file: time, step, dimensions, and
variable names.
"""
function get_file_info(filename::String)
    info = Dict{String, Any}()
    NCDataset(filename, "r") do ds
        if haskey(ds, "time")
            info["time"] = Float64(ds["time"][1])
        end
        if haskey(ds, "step")
            info["step"] = Int(ds["step"][1])
        end
        info["dimensions"] = Dict(String(k) => v for (k, v) in ds.dim)
        info["variables"] = collect(keys(ds))
    end
    return info
end

# ================================================================================
# Enhanced Configuration and Integration Functions
# ================================================================================

"""
    create_shtns_aware_output_config(shtns_config, pencils; base_config=default_config())

Return an `OutputConfig` whose spectral output limit follows the supplied
SHTns configuration.

The `pencils` argument is currently accepted to mirror setup call sites that
construct output configuration together with decomposition metadata.
"""
function create_shtns_aware_output_config(
        shtns_config::SHTnsKitConfig, pencils::NamedTuple;
        base_config::OutputConfig = default_config())
    return OutputConfig(
        base_config.output_space,
        base_config.output_dir,
        base_config.filename_prefix,
        base_config.include_metadata,
        base_config.include_grid,
        base_config.include_diagnostics,
        base_config.output_precision,
        shtns_config.lmax,
        base_config.overwrite_files,
        base_config.output_interval,
        base_config.restart_interval,
        base_config.max_output_time,
        base_config.time_tolerance
    )
end

"""
    validate_output_compatibility(field_info, shtns_config)

Check that field metadata and SHTns metadata describe the same grid and
spectral layout.

Returns `false` and emits a warning listing all mismatches; returns `true` when
the layouts are compatible.
"""
function validate_output_compatibility(field_info::FieldInfo, shtns_config::SHTnsKitConfig)
    errors = String[]

    if field_info.nlm != shtns_config.nlm
        push!(errors, "Field nlm ($(field_info.nlm)) != SHTns nlm ($(shtns_config.nlm))")
    end
    if field_info.nlat != shtns_config.nlat
        push!(errors, "Field nlat ($(field_info.nlat)) != SHTns nlat ($(shtns_config.nlat))")
    end
    if field_info.nlon != shtns_config.nlon
        push!(errors, "Field nlon ($(field_info.nlon)) != SHTns nlon ($(shtns_config.nlon))")
    end
    if !isempty(field_info.l_values) && !isempty(shtns_config.l_values)
        if length(field_info.l_values) != length(shtns_config.l_values)
            push!(errors, "l_values length mismatch")
        elseif field_info.l_values != shtns_config.l_values
            push!(errors, "l_values content mismatch")
        end
    end

    if !isempty(errors)
        @warn "Output compatibility validation failed:\n" * join(errors, "\n")
        return false
    end

    return true
end
