# ================================================================================
# Parallel NetCDF Support Check
# ================================================================================

"""
    check_parallel_netcdf_support(comm)

Verify that parallel NetCDF (MPI-IO via HDF5) is available at runtime.
Called once at initialization. Errors immediately if not supported.
"""
function check_parallel_netcdf_support(comm)
    # Generate filename on rank 0 and broadcast — all ranks must use the same
    # path for the collective NCDataset open, otherwise it will deadlock.
    rank = MPI.Comm_rank(comm)
    tmpfile = rank == 0 ? tempname() * ".nc" : ""
    tmpfile = MPI.bcast(tmpfile, comm; root=0)
    try
        ds = NCDataset(tmpfile, "c"; comm=comm, info=MPI.Info())
        close(ds)
    catch e
        error("Parallel NetCDF (MPI-IO) is required but not available. " *
              "Please install HDF5 with parallel support: " *
              "set ENV[\"JULIA_HDF5_PATH\"] to a parallel-enabled HDF5 installation " *
              "and rebuild HDF5_jll. Error: $e")
    finally
        # Only rank 0 cleans up to avoid filesystem races
        if rank == 0 && isfile(tmpfile)
            rm(tmpfile; force=true)
        end
        MPI.Barrier(comm)
    end
end

# ================================================================================
# Filename Generation
# ================================================================================

"""
    generate_filename(config, time, step, file_type="output", output_number=1; geometry=:shell)

Build the on-disk NetCDF path for history, restart, or auxiliary output files.

Files are numbered by writer counters rather than by `time` or `step`, which
keeps names stable across floating-point time representations.
"""
function generate_filename(config::OutputConfig, time::Float64, step::Int,
                            file_type::String = "output", output_number::Int = 1;
                            geometry::Symbol = :shell)
    geom = string(geometry)

    filename = if file_type == "output"
        "$(config.filename_prefix)_$(geom)_hist_$(output_number).nc"
    elseif file_type == "restart"
        "$(config.filename_prefix)_$(geom)_restart_$(output_number).nc"
    else
        "$(config.filename_prefix)_$(geom)_$(file_type)_$(output_number).nc"
    end

    return joinpath(config.output_dir, filename)
end

# ================================================================================
# Parallel NetCDF File Creation
# ================================================================================

"""
    create_parallel_netcdf(filename, config, field_info, metadata, comm)

Open a new NetCDF file in parallel mode. All ranks call this collectively.
Defines global dimensions and variables based on field_info.
"""
function create_parallel_netcdf(filename::String, config::OutputConfig,
                                field_info::FieldInfo, metadata::Dict{String,Any}, comm;
                                geometry::Symbol = :shell)
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if config.overwrite_files && isfile(filename) && rank == 0
        rm(filename)
    end
    MPI.Barrier(comm)

    ds = NCDataset(filename, "c"; comm=comm, info=MPI.Info())

    # Global attributes
    ds.attrib["title"] = "GeoDynamo Simulation Output"
    ds.attrib["source"] = "GeoDynamo simulation code"
    ds.attrib["history"] = "Created on $(now())"
    ds.attrib["Conventions"] = "CF-1.8"
    ds.attrib["mpi_total_ranks"] = nprocs

    # Add simulation metadata
    if config.include_metadata
        if !haskey(metadata, "geometry")
            metadata["geometry"] = string(geometry)
        end
        for (key, value) in metadata
            try
                ds.attrib[key] = value
            catch
                # Skip problematic attributes
            end
        end
    end

    return ds
end

# ================================================================================
# Dimension and Variable Setup
# ================================================================================

"""
    setup_dimensions!(ds, field_info, config)

Define the global NetCDF dimensions shared by coordinates and field variables.

This is part of the collective parallel NetCDF setup path; every rank that
opened `ds` must reach this call in the same order.
"""
function setup_dimensions!(ds, field_info::FieldInfo, config::OutputConfig)
    # Time dimension
    defDim(ds, "time", 1)

    # Radial (shared by all fields)
    if field_info.nr > 0
        defDim(ds, "r", field_info.nr)
    end

    # Physical coordinates (for temperature/composition)
    if config.output_space == MIXED_FIELDS || config.output_space == PHYSICAL_ONLY
        if field_info.nlat > 0
            defDim(ds, "theta", field_info.nlat)
        end
        if field_info.nlon > 0
            defDim(ds, "phi", field_info.nlon)
        end
    end

    # Spectral dimension
    if config.output_space == MIXED_FIELDS || config.output_space == SPECTRAL_ONLY
        if field_info.nlm > 0
            defDim(ds, "spectral_mode", field_info.nlm)
        end
    end
end

"""
    setup_variables!(ds, field_info, config, available_fields)

Define NetCDF variables for time, coordinates, and available simulation fields.

The writer currently supports the mixed GeoDynamo layout: physical
temperature/composition and spectral toroidal/poloidal components.
"""
function setup_variables!(ds, field_info::FieldInfo, config::OutputConfig,
                          available_fields::Vector{String})
    T = config.output_precision

    # Time and step variables
    defVar(ds, "time", T, ("time",); attrib=Dict(
        "long_name" => "simulation_time", "units" => "dimensionless"))
    defVar(ds, "step", Int32, ("time",); attrib=Dict(
        "long_name" => "simulation_step"))

    # Coordinate variables
    if field_info.nr > 0
        defVar(ds, "r", T, ("r",); attrib=Dict(
            "long_name" => "radial_coordinate", "units" => "dimensionless"))
    end

    if config.output_space == MIXED_FIELDS || config.output_space == PHYSICAL_ONLY
        if field_info.nlat > 0
            defVar(ds, "theta", T, ("theta",); attrib=Dict(
                "long_name" => "colatitude", "units" => "radians"))
        end
        if field_info.nlon > 0
            defVar(ds, "phi", T, ("phi",); attrib=Dict(
                "long_name" => "azimuthal_angle", "units" => "radians"))
        end
    end

    if config.output_space == MIXED_FIELDS || config.output_space == SPECTRAL_ONLY
        if field_info.nlm > 0
            defVar(ds, "l_values", Int32, ("spectral_mode",); attrib=Dict(
                "long_name" => "spherical_harmonic_degree"))
            defVar(ds, "m_values", Int32, ("spectral_mode",); attrib=Dict(
                "long_name" => "spherical_harmonic_order"))
        end
    end

    # Field variables
    if config.output_space == MIXED_FIELDS
        # Temperature: Physical space (theta, phi, r)
        if "temperature" in available_fields && field_info.nlat > 0 && field_info.nlon > 0
            defVar(ds, "temperature", T, ("theta", "phi", "r"); attrib=Dict(
                "long_name" => "temperature", "units" => "dimensionless",
                "representation" => "physical_space"))
        end

        # Composition: Physical space (theta, phi, r)
        if "composition" in available_fields && field_info.nlat > 0 && field_info.nlon > 0
            defVar(ds, "composition", T, ("theta", "phi", "r"); attrib=Dict(
                "long_name" => "composition", "units" => "dimensionless",
                "representation" => "physical_space"))
        end

        # Spectral fields (spectral_mode, r)
        if field_info.nlm > 0
            for component in ["velocity_toroidal", "velocity_poloidal",
                              "magnetic_toroidal", "magnetic_poloidal",
                              "temperature_spectral", "composition_spectral"]
                if component in available_fields
                    defVar(ds, "$(component)_real", T, ("spectral_mode", "r"); attrib=Dict(
                        "long_name" => "$(component)_real_coefficients",
                        "representation" => "spectral_space"))
                    defVar(ds, "$(component)_imag", T, ("spectral_mode", "r"); attrib=Dict(
                        "long_name" => "$(component)_imaginary_coefficients",
                        "representation" => "spectral_space"))
                end
            end
        end
    end
end

"""
    setup_diagnostic_variables!(ds, diagnostics, config)

Create scalar diagnostic variables for the diagnostics dictionary.

No variables are created when diagnostics are disabled or the dictionary is
empty.
"""
function setup_diagnostic_variables!(ds, diagnostics::Dict{String,Float64}, config::OutputConfig)
    if !config.include_diagnostics || isempty(diagnostics)
        return
    end

    if !haskey(ds.dim, "scalar")
        defDim(ds, "scalar", 1)
    end

    for (name, _) in diagnostics
        defVar(ds, "diag_$(name)", config.output_precision, ("scalar",); attrib=Dict(
            "long_name" => replace(name, "_" => " ")))
    end
end

# ================================================================================
# Data Writing (Parallel Offset Writes)
# ================================================================================

"""
    write_coordinate_data!(ds, field_info, config)

Write coordinate arrays. Only rank 0 writes coordinates (they are global/shared).
"""
function write_coordinate_data!(ds, field_info::FieldInfo, config::OutputConfig)
    rank = MPI.Comm_rank(output_comm())

    # Coordinates are the same on all ranks; only rank 0 writes them
    if rank == 0
        T = config.output_precision
        if !isempty(field_info.theta) && haskey(ds, "theta")
            ds["theta"][:] = T.(field_info.theta)
        end
        if !isempty(field_info.phi) && haskey(ds, "phi")
            ds["phi"][:] = T.(field_info.phi)
        end
        if !isempty(field_info.r) && haskey(ds, "r")
            ds["r"][:] = T.(field_info.r)
        end
        if !isempty(field_info.l_values) && haskey(ds, "l_values")
            ds["l_values"][:] = Int32.(field_info.l_values)
            ds["m_values"][:] = Int32.(field_info.m_values)
        end
    end
end

@inline function local_spectral_io_ranges(field_info::FieldInfo)
    if field_info.has_config
        spec_pencil = (field_info.config::SHTnsKitConfig).pencils.spec
        return range_local(spec_pencil, 1), range_local(spec_pencil, 3)
    elseif field_info.has_pencils
        return range_local(field_info.pencils.spec, 1), range_local(field_info.pencils.spec, 3)
    else
        return 1:field_info.nlm, 1:field_info.nr
    end
end

"""
    pack_local_spectral_coefficients(real_data, imag_data, field_info)

Convert local spectral storage into the `(spectral_mode, r)` slab written to
NetCDF.

Mapped local storage keeps the future 2D spectral-pencil migration localized to
this packing step instead of leaking `[local_lm, 1, local_r]` assumptions into
I/O code.
"""
function pack_local_spectral_coefficients(real_data::AbstractArray,
                                          imag_data::AbstractArray,
                                          field_info::FieldInfo)
    if ndims(real_data) == 2 && ndims(imag_data) == 2
        return real_data, imag_data
    elseif ndims(real_data) == 3 && ndims(imag_data) == 3
        field_info.has_config || throw(ArgumentError(
            "Packing 3D spectral storage for NetCDF output requires SHTns configuration metadata.",
        ))

        config = field_info.config::SHTnsKitConfig
        lm_map = local_spectral_lm_map(config)
        lm_range, r_range = local_spectral_io_ranges(field_info)
        packed_real = zeros(eltype(real_data), length(lm_range), length(r_range))
        packed_imag = zeros(eltype(imag_data), length(lm_range), length(r_range))
        (isempty(lm_range) || isempty(r_range)) && return packed_real, packed_imag
        lm_first = first(lm_range)
        r_first = first(r_range)

        for slot in CartesianIndices(lm_map)
            global_lm = lm_map[slot]
            global_lm == 0 && continue
            row = global_lm - lm_first + 1
            (1 <= row <= size(packed_real, 1)) || continue
            for global_r in r_range
                col = global_r - r_first + 1
                local_r = col
                if local_r <= size(real_data, 3) && local_r <= size(imag_data, 3)
                    packed_real[row, col] = local_spectral_value(real_data, slot, local_r)
                    packed_imag[row, col] = local_spectral_value(imag_data, slot, local_r)
                end
            end
        end

        return packed_real, packed_imag
    end

    throw(ArgumentError(
        "Expected spectral coefficient arrays with 2 or 3 dimensions, got $(ndims(real_data)) and $(ndims(imag_data)).",
    ))
end

"""
    unpack_local_spectral_coefficients(real_data, imag_data, config)

Rebuild local spectral storage from a NetCDF `(spectral_mode, r)` slab using
the configured local spectral-slot mapping.
"""
function unpack_local_spectral_coefficients(real_data::AbstractMatrix,
                                            imag_data::AbstractMatrix,
                                            config::SHTnsKitConfig)
    spec_pencil = config.pencils.spec
    local_shape = size_local(spec_pencil)
    unpacked_real = zeros(eltype(real_data), local_shape[1], local_shape[2], local_shape[3])
    unpacked_imag = zeros(eltype(imag_data), local_shape[1], local_shape[2], local_shape[3])
    lm_map = local_spectral_lm_map(config)
    lm_range = range_local(spec_pencil, 1)
    isempty(lm_range) && return unpacked_real, unpacked_imag
    lm_first = first(lm_range)

    for slot in CartesianIndices(lm_map)
        global_lm = lm_map[slot]
        global_lm == 0 && continue
        row = global_lm - lm_first + 1
        (1 <= row <= size(real_data, 1)) || continue
        for local_r in axes(unpacked_real, 3)
            if local_r <= size(real_data, 2) && local_r <= size(imag_data, 2)
                set_local_spectral_value!(unpacked_real, slot, local_r, real_data[row, local_r])
                set_local_spectral_value!(unpacked_imag, slot, local_r, imag_data[row, local_r])
            end
        end
    end

    return unpacked_real, unpacked_imag
end

"""
    write_field_data!(ds, fields, config, field_info)

Write field data using parallel offset writes. Each rank writes its local pencil
slice at the correct global position.
"""
function write_field_data!(ds, fields::Dict{String,Any}, config::OutputConfig,
                          field_info::FieldInfo)
    T = config.output_precision
    pencils = field_info.has_pencils ? field_info.pencils : nothing

    # Temperature: physical space, r-pencil (theta,phi distributed, r local)
    if haskey(fields, "temperature") && haskey(ds, "temperature")
        T_data = T.(fields["temperature"])
        if pencils !== nothing
            θ_range = range_local(pencils.r, 1)
            φ_range = range_local(pencils.r, 2)
            ds["temperature"][θ_range, φ_range, :] = T_data
        else
            ds["temperature"][:, :, :] = T_data
        end
    end

    # Composition: physical space, r-pencil (theta,phi distributed, r local)
    if haskey(fields, "composition") && haskey(ds, "composition")
        C_data = T.(fields["composition"])
        if pencils !== nothing
            θ_range = range_local(pencils.r, 1)
            φ_range = range_local(pencils.r, 2)
            ds["composition"][θ_range, φ_range, :] = C_data
        else
            ds["composition"][:, :, :] = C_data
        end
    end

    # Spectral fields: spec-pencil (lm and r distributed)
    for component in ["velocity_toroidal", "velocity_poloidal",
                      "magnetic_toroidal", "magnetic_poloidal",
                      "temperature_spectral", "composition_spectral"]
        if haskey(fields, component)
            field_data = fields[component]
            if haskey(field_data, "real") && haskey(field_data, "imag")
                real_name = "$(component)_real"
                imag_name = "$(component)_imag"

                if haskey(ds, real_name) && haskey(ds, imag_name)
                    real_data = field_data["real"]
                    imag_data = field_data["imag"]
                    real_data, imag_data = pack_local_spectral_coefficients(real_data, imag_data, field_info)

                    real_out = T.(real_data)
                    imag_out = T.(imag_data)

                    if pencils !== nothing
                        lm_range, r_range = local_spectral_io_ranges(field_info)
                        ds[real_name][lm_range, r_range] = real_out
                        ds[imag_name][lm_range, r_range] = imag_out
                    else
                        ds[real_name][:, :] = real_out
                        ds[imag_name][:, :] = imag_out
                    end
                end
            end
        end
    end
end

"""
    write_time_data!(ds, time, step, config)

Write scalar simulation time and step values.

Only rank 0 writes these shared scalar variables; field arrays are handled by
the distributed write path.
"""
function write_time_data!(ds, time::Float64, step::Int, config::OutputConfig)
    rank = MPI.Comm_rank(output_comm())
    # Only rank 0 writes scalar time/step data
    if rank == 0
        ds["time"][1] = config.output_precision(time)
        ds["step"][1] = Int32(step)
    end
end

"""
    write_diagnostics!(ds, diagnostics, config)

Write scalar diagnostics into `diag_*` variables when diagnostics are enabled.

The values are expected to be already globally reduced by `compute_diagnostics`.
"""
function write_diagnostics!(ds, diagnostics::Dict{String,Float64}, config::OutputConfig)
    if !config.include_diagnostics
        return
    end
    rank = MPI.Comm_rank(output_comm())
    if rank == 0
        for (name, value) in diagnostics
            var_name = "diag_$(name)"
            if haskey(ds, var_name)
                ds[var_name][1] = config.output_precision(value)
            end
        end
    end
end

# ================================================================================
# Grid File Writing (One-time, Rank 0 only)
# ================================================================================

"""
    write_grid_file!(config, field_info, shtns_config, metadata)

Write a separate grid file containing coordinate and grid information.
Written only once by rank 0 at the start of the simulation.
"""
function write_grid_file!(config::OutputConfig, field_info::FieldInfo,
                         shtns_config::Union{SHTnsKitConfig,Nothing},
                         metadata::Dict{String,Any};
                         geometry::Symbol = :shell)
    rank = MPI.Comm_rank(output_comm())

    if rank != 0
        return
    end

    geom = string(geometry)
    grid_filename = joinpath(config.output_dir,
                            "$(config.filename_prefix)_$(geom)_grid.nc")

    if config.overwrite_files && isfile(grid_filename)
        rm(grid_filename)
    end

    T = config.output_precision

    NCDataset(grid_filename, "c") do ds
        ds.attrib["title"] = "GeoDynamo Simulation Grid Information"
        ds.attrib["description"] = "Grid coordinates and geometry information"
        ds.attrib["source"] = "GeoDynamo simulation code"
        ds.attrib["created"] = string(now())
        ds.attrib["Conventions"] = "CF-1.8"

        if config.include_metadata
            if !haskey(metadata, "geometry")
                metadata["geometry"] = string(geometry)
            end
            for (key, value) in metadata
                try
                    ds.attrib[key] = value
                catch
                end
            end
        end

        # Define dimensions and coordinate variables
        if field_info.nr > 0
            defDim(ds, "r", field_info.nr)
            defVar(ds, "r", T, ("r",); attrib=Dict(
                "long_name" => "radial_coordinate", "units" => "dimensionless",
                "description" => "Normalized radial coordinate (inner=0.35, outer=1.0)"))
        end

        if field_info.nlat > 0
            defDim(ds, "theta", field_info.nlat)
            defVar(ds, "theta", T, ("theta",); attrib=Dict(
                "long_name" => "colatitude", "units" => "radians",
                "description" => "Colatitude from north pole (0 to pi)"))
        end

        if field_info.nlon > 0
            defDim(ds, "phi", field_info.nlon)
            defVar(ds, "phi", T, ("phi",); attrib=Dict(
                "long_name" => "azimuthal_angle", "units" => "radians",
                "description" => "Longitude angle (0 to 2pi)"))
        end

        if field_info.nlm > 0
            defDim(ds, "spectral_mode", field_info.nlm)
            defVar(ds, "l_values", Int32, ("spectral_mode",); attrib=Dict(
                "long_name" => "spherical_harmonic_degree"))
            defVar(ds, "m_values", Int32, ("spectral_mode",); attrib=Dict(
                "long_name" => "spherical_harmonic_order"))
        end

        # Add SHTns-specific info
        if shtns_config !== nothing
            ds.attrib["shtns_lmax"] = shtns_config.lmax
            ds.attrib["shtns_mmax"] = shtns_config.mmax
            ds.attrib["shtns_nlm"] = shtns_config.nlm
            ds.attrib["shtns_nlat"] = shtns_config.nlat
            ds.attrib["shtns_nlon"] = shtns_config.nlon
            ds.attrib["grid_type_theta"] = "gaussian"
            ds.attrib["grid_type_phi"] = "equispaced"

            if !isempty(shtns_config.gauss_weights) && field_info.nlat > 0
                defVar(ds, "gauss_weights", T, ("theta",); attrib=Dict(
                    "long_name" => "gaussian_quadrature_weights"))
            end
        end

        # Write coordinate data
        if !isempty(field_info.r)
            ds["r"][:] = T.(field_info.r)
        end
        if !isempty(field_info.theta)
            ds["theta"][:] = T.(field_info.theta)
        end
        if !isempty(field_info.phi)
            ds["phi"][:] = T.(field_info.phi)
        end
        if !isempty(field_info.l_values)
            ds["l_values"][:] = Int32.(field_info.l_values)
            ds["m_values"][:] = Int32.(field_info.m_values)
        end
        if shtns_config !== nothing && !isempty(shtns_config.gauss_weights) && haskey(ds, "gauss_weights")
            ds["gauss_weights"][:] = T.(shtns_config.gauss_weights)
        end
    end

    println("Rank 0: Successfully wrote grid file: $grid_filename")
end
