# ================================================================================
# NetCDF Output Writer - Parallel I/O (MPI-IO) for GeoDynamo Simulation
# All ranks write concurrently to a single shared NetCDF file per timestep.
# Uses NCDatasets.jl with parallel HDF5 backend.
# ================================================================================

using MPI
using NCDatasets
using LinearAlgebra
using Statistics
using Dates
using Printf

# Use get_comm() from pencil_decomps.jl for lazy MPI initialization
const _output_comm = Ref{MPI.Comm}()
const _output_comm_initialized = Ref(false)

function output_comm()
    if !_output_comm_initialized[]
        _output_comm[] = get_comm()
        _output_comm_initialized[] = true
    end
    return _output_comm[]
end

# ================================================================================
# Configuration
# ================================================================================

@enum OutputSpace begin
    MIXED_FIELDS        # Spectral for velocity/magnetic, physical for temperature/composition
    PHYSICAL_ONLY       # Output only physical space data
    SPECTRAL_ONLY       # Output only spectral coefficients
end

struct OutputConfig
    output_space::OutputSpace
    output_dir::String
    filename_prefix::String
    include_metadata::Bool
    include_grid::Bool
    include_diagnostics::Bool
    output_precision::DataType
    spectral_lmax_output::Int
    overwrite_files::Bool

    # Time-based control
    output_interval::Float64
    restart_interval::Float64
    max_output_time::Float64
    time_tolerance::Float64
end

function default_config(; precision::Type{<:AbstractFloat}=Float64)
    return OutputConfig(
        MIXED_FIELDS,       # spectral for velocity/magnetic, physical for temperature
        "./output",         # output directory
        "geodynamo",        # filename prefix
        true,               # include metadata
        true,               # include grid
        true,               # include diagnostics
        precision,          # output precision
        -1,                 # spectral lmax (-1 = all)
        true,               # overwrite files
        0.1,                # output every 0.1 time units
        1.0,                # restart every 1.0 time units
        Inf,                # max output time
        1e-10               # time tolerance
    )
end

default_config(T::Type{<:AbstractFloat}) = default_config(; precision=T)

@inline function resolve_output_precision(sym::Symbol)
    sym === :float32 && return Float32
    sym === :float64 && return Float64
    @warn "Unknown output precision symbol $(sym); defaulting to Float64"
    return Float64
end

function output_config_from_parameters(; base_config::OutputConfig=default_config())
    params = get_parameters()
    precision = resolve_output_precision(params.output_precision)

    return OutputConfig(
        base_config.output_space,
        base_config.output_dir,
        base_config.filename_prefix,
        base_config.include_metadata,
        base_config.include_grid,
        base_config.include_diagnostics,
        precision,
        base_config.spectral_lmax_output,
        base_config.overwrite_files,
        base_config.output_interval,
        base_config.restart_interval,
        base_config.max_output_time,
        base_config.time_tolerance
    )
end

function with_output_precision(config::OutputConfig, ::Type{T}) where {T<:AbstractFloat}
    return OutputConfig(
        config.output_space,
        config.output_dir,
        config.filename_prefix,
        config.include_metadata,
        config.include_grid,
        config.include_diagnostics,
        T,
        config.spectral_lmax_output,
        config.overwrite_files,
        config.output_interval,
        config.restart_interval,
        config.max_output_time,
        config.time_tolerance
    )
end

# ================================================================================
# Time Tracking
# ================================================================================

mutable struct TimeTracker
    last_output_time::Float64
    last_restart_time::Float64
    output_count::Int
    restart_count::Int
    next_output_time::Float64
    next_restart_time::Float64
    grid_file_written::Bool
end

function create_time_tracker(config::OutputConfig, start_time::Float64 = 0.0)
    return TimeTracker(
        start_time - config.output_interval,
        start_time - config.restart_interval,
        0, 0,
        start_time, start_time,
        false
    )
end

function should_output_now(tracker::TimeTracker, current_time::Float64, config::OutputConfig)
    time_since_output = current_time - tracker.last_output_time
    return (time_since_output >= config.output_interval - config.time_tolerance &&
            current_time <= config.max_output_time)
end

function should_restart_now(tracker::TimeTracker, current_time::Float64, config::OutputConfig)
    time_since_restart = current_time - tracker.last_restart_time
    return time_since_restart >= config.restart_interval - config.time_tolerance
end

function update_tracker!(tracker::TimeTracker, current_time::Float64,
                        config::OutputConfig, did_output::Bool, did_restart::Bool)
    if did_output
        tracker.last_output_time = current_time
        tracker.output_count += 1
        tracker.next_output_time = current_time + config.output_interval
    end
    if did_restart
        tracker.last_restart_time = current_time
        tracker.restart_count += 1
        tracker.next_restart_time = current_time + config.restart_interval
    end
end

function time_to_next_output(tracker::TimeTracker, current_time::Float64, config::OutputConfig)
    time_to_output = tracker.next_output_time - current_time
    time_to_restart = tracker.next_restart_time - current_time
    return min(time_to_output, time_to_restart, config.output_interval)
end

# ================================================================================
# Field Information
# ================================================================================

struct FieldInfo
    # Physical dimensions (for temperature/composition)
    nlat::Int
    nlon::Int
    nr::Int

    # Spectral dimensions (for velocity/magnetic)
    nlm::Int

    # Coordinate arrays
    theta::Vector{Float64}
    phi::Vector{Float64}
    r::Vector{Float64}
    l_values::Vector{Int}
    m_values::Vector{Int}

    # Pencil decomposition information
    has_pencils::Bool
    pencils::NamedTuple
    has_config::Bool
    config::Union{SHTnsKitConfig,Nothing}

    # Local range information
    local_ranges::Dict{Symbol, UnitRange{Int}}
end

# Default constructor for FieldInfo without pencils/config
function FieldInfo()
    return FieldInfo(0, 0, 0, 0, Float64[], Float64[], Float64[],
                     Int[], Int[], false, NamedTuple(), false, nothing, Dict{Symbol, UnitRange{Int}}())
end

# Constructor with pencils and config
function FieldInfo(nlat::Int, nlon::Int, nr::Int, nlm::Int,
                   theta::Vector{Float64}, phi::Vector{Float64}, r::Vector{Float64},
                   l_values::Vector{Int}, m_values::Vector{Int},
                   pencils::NamedTuple, config::SHTnsKitConfig,
                   local_ranges::Dict{Symbol, UnitRange{Int}})
    return FieldInfo(nlat, nlon, nr, nlm, theta, phi, r, l_values, m_values,
                     true, pencils, true, config, local_ranges)
end

function extract_field_info(fields::Dict{String,Any}, config::Union{SHTnsKitConfig,Nothing}=nothing,
                           pencils::Union{NamedTuple,Nothing}=nothing)
    nlat = 0
    nlon = 0
    nr = 0
    nlm = 0

    # Get physical dimensions from temperature
    if haskey(fields, "temperature")
        temp_dims = size(fields["temperature"])
        nlat, nlon, nr = temp_dims[1], temp_dims[2], temp_dims[3]
    end

    # Get physical dimensions from composition if temperature not available
    if nlat == 0 && haskey(fields, "composition")
        comp_dims = size(fields["composition"])
        nlat, nlon, nr = comp_dims[1], comp_dims[2], comp_dims[3]
    end

    # Get spectral dimensions
    if haskey(fields, "velocity_toroidal") && haskey(fields["velocity_toroidal"], "real")
        spec_dims = size(fields["velocity_toroidal"]["real"])
        nlm = spec_dims[1]
        if nr == 0
            nr = spec_dims[end]
        end
    elseif haskey(fields, "magnetic_toroidal") && haskey(fields["magnetic_toroidal"], "real")
        spec_dims = size(fields["magnetic_toroidal"]["real"])
        nlm = spec_dims[1]
        if nr == 0
            nr = spec_dims[end]
        end
    elseif haskey(fields, "temperature_spectral") && haskey(fields["temperature_spectral"], "real")
        spec_dims = size(fields["temperature_spectral"]["real"])
        nlm = spec_dims[1]
        if nr == 0
            nr = spec_dims[end]
        end
    elseif haskey(fields, "composition_spectral") && haskey(fields["composition_spectral"], "real")
        spec_dims = size(fields["composition_spectral"]["real"])
        nlm = spec_dims[1]
        if nr == 0
            nr = spec_dims[end]
        end
    end

    # Create coordinate arrays
    theta = nlat > 0 ? collect(range(0, π, length=nlat)) : Float64[]
    phi = nlon > 0 ? collect(range(0, 2π, length=nlon)) : Float64[]
    rratio = try get_parameters().d_rratio catch; 0.35 end
    r = nr > 0 ? collect(range(rratio, 1.0, length=nr)) : Float64[]

    # Create l,m values for spectral modes
    l_values = Int[]
    m_values = Int[]
    if nlm > 0
        # Invert nlm = (lmax+1)(lmax+2)/2 via quadratic formula
        lmax = Int(floor((-3 + sqrt(1 + 8*nlm)) / 2))
        for l in 0:lmax
            for m in 0:l
                if length(l_values) < nlm
                    push!(l_values, l)
                    push!(m_values, m)
                end
            end
        end
    end

    # Extract local range information if pencils are provided
    local_ranges = Dict{Symbol, UnitRange{Int}}()
    if pencils !== nothing
        try
            local_ranges[:spec] = range_local(pencils.spec, 1)
            local_ranges[:r] = range_local(pencils.r, 3)
            local_ranges[:θ] = range_local(pencils.θ, 1)
            local_ranges[:φ] = range_local(pencils.φ, 2)
        catch
            # Fallback if pencil ranges not available
        end
    end

    # Use config information if available
    if config !== nothing
        nlat = config.nlat
        nlon = config.nlon
        nlm = config.nlm
        l_values = config.l_values[1:min(length(config.l_values), nlm)]
        m_values = config.m_values[1:min(length(config.m_values), nlm)]
        theta = config.theta_grid
        phi = config.phi_grid
    end

    # Create FieldInfo with type-stable constructor
    if pencils !== nothing && config !== nothing
        return FieldInfo(nlat, nlon, nr, nlm, theta, phi, r, l_values, m_values,
                         pencils, config, local_ranges)
    else
        dummy_pencils = NamedTuple()

        return FieldInfo(nlat, nlon, nr, nlm, theta, phi, r, l_values, m_values,
                         pencils !== nothing, pencils !== nothing ? pencils : dummy_pencils,
                         config !== nothing, config,
                         local_ranges)
    end
end

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

function generate_filename(config::OutputConfig, time::Float64, step::Int,
                            file_type::String = "output", output_number::Int = 1)
    geom = try
        string(get_parameters().geometry)
    catch
        "shell"
    end

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
                                field_info::FieldInfo, metadata::Dict{String,Any}, comm)
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
            try
                metadata["geometry"] = string(get_parameters().geometry)
            catch
                metadata["geometry"] = "shell"
            end
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

                    # Handle 3D arrays (nlm, 1, nr) -> squeeze dim 2
                    if ndims(real_data) == 3
                        real_data = real_data[:, 1, :]
                        imag_data = imag_data[:, 1, :]
                    end

                    real_out = T.(real_data)
                    imag_out = T.(imag_data)

                    if pencils !== nothing
                        lm_range = range_local(pencils.spec, 1)
                        r_range = range_local(pencils.spec, 3)
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

function write_time_data!(ds, time::Float64, step::Int, config::OutputConfig)
    rank = MPI.Comm_rank(output_comm())
    # Only rank 0 writes scalar time/step data
    if rank == 0
        ds["time"][1] = config.output_precision(time)
        ds["step"][1] = Int32(step)
    end
end

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
                         metadata::Dict{String,Any})
    rank = MPI.Comm_rank(output_comm())

    if rank != 0
        return
    end

    geom = try
        string(get_parameters().geometry)
    catch
        "shell"
    end
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
                try
                    metadata["geometry"] = string(get_parameters().geometry)
                catch
                    metadata["geometry"] = "shell"
                end
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

# ================================================================================
# Diagnostics Computation
# ================================================================================

function compute_diagnostics(fields::Dict{String,Any}, field_info::FieldInfo)
    diagnostics = Dict{String, Float64}()
    comm = get_comm()
    nprocs = (comm !== nothing && MPI.Comm_size(comm) > 1) ? MPI.Comm_size(comm) : 1
    use_mpi = nprocs > 1

    # Helper: reduce a scalar across all ranks
    _global_sum(x)  = use_mpi ? MPI.Allreduce(x, MPI.SUM, comm) : x
    _global_max(x)  = use_mpi ? MPI.Allreduce(x, MPI.MAX, comm) : x
    _global_min(x)  = use_mpi ? MPI.Allreduce(x, MPI.MIN, comm) : x

    # Physical-space fields: need global reduction for correct mean/min/max/std
    for (key, prefix) in [("temperature", "temp"), ("composition", "comp")]
        if haskey(fields, key)
            F = fields[key]
            local_n   = Float64(length(F))
            local_sum = Float64(sum(F))
            local_min = Float64(minimum(F))
            local_max = Float64(maximum(F))

            global_n   = _global_sum(local_n)
            global_sum = _global_sum(local_sum)
            global_min = _global_min(local_min)
            global_max = _global_max(local_max)
            global_mean = global_sum / global_n

            # Two-pass variance via parallel algorithm: sum of (x - global_mean)^2
            local_sq_sum = sum(x -> (Float64(x) - global_mean)^2, F)
            global_sq_sum = _global_sum(local_sq_sum)

            diagnostics["$(prefix)_mean"] = global_mean
            diagnostics["$(prefix)_std"]  = sqrt(global_sq_sum / global_n)
            diagnostics["$(prefix)_min"]  = global_min
            diagnostics["$(prefix)_max"]  = global_max
        end
    end

    # Spectral fields: reduce energy, rms, max across ranks
    for component in ["velocity_toroidal", "velocity_poloidal",
                      "magnetic_toroidal", "magnetic_poloidal",
                      "temperature_spectral", "composition_spectral"]
        if haskey(fields, component)
            field_data = fields[component]
            if haskey(field_data, "real") && haskey(field_data, "imag")
                real_part = field_data["real"]
                imag_part = field_data["imag"]

                local_energy = zero(Float64)
                local_max_mag = zero(Float64)
                local_count = Float64(length(real_part))
                for i in eachindex(real_part, imag_part)
                    magnitude_sq = Float64(real_part[i])^2 + Float64(imag_part[i])^2
                    local_energy += magnitude_sq
                    local_max_mag = max(local_max_mag, sqrt(magnitude_sq))
                end

                global_energy = _global_sum(local_energy)
                global_max_mag = _global_max(local_max_mag)
                global_count = _global_sum(local_count)

                diagnostics["$(component)_energy"] = 0.5 * global_energy
                diagnostics["$(component)_rms"] = sqrt(global_energy / global_count)
                diagnostics["$(component)_max"] = global_max_mag

                if field_info.has_config && !isempty(field_info.l_values)
                    compute_spectral_energy_diagnostics!(diagnostics, component,
                                                        real_part, imag_part, field_info)
                end
            end
        end
    end

    return diagnostics
end

function compute_spectral_energy_diagnostics!(diagnostics::Dict{String,Float64},
                                            component::String,
                                            real_part::AbstractArray,
                                            imag_part::AbstractArray,
                                            field_info::FieldInfo)
    if !field_info.has_config
        return
    end

    config = field_info.config
    l_values = config.l_values

    l_max = maximum(l_values)
    l_energies = zeros(Float64, l_max + 1)

    for (idx, l) in enumerate(l_values)
        if idx <= size(real_part, 1)
            l_energy = zero(eltype(real_part))
            for j in axes(real_part, 2), k in axes(real_part, 3)
                l_energy += real_part[idx, j, k]^2 + imag_part[idx, j, k]^2
            end
            l_energies[l + 1] += l_energy
        end
    end

    total_energy = sum(l_energies)
    if total_energy > 0
        peak_l = argmax(l_energies) - 1
        diagnostics["$(component)_peak_l"] = Float64(peak_l)

        spectral_centroid = sum((0:l_max) .* l_energies) / total_energy
        diagnostics["$(component)_spectral_centroid"] = spectral_centroid

        low_mode_energy = sum(l_energies[1:min(11, length(l_energies))])
        diagnostics["$(component)_low_mode_fraction"] = low_mode_energy / total_energy
    end
end

# ================================================================================
# Main Output Function
# ================================================================================

function write_fields!(fields::Dict{String,Any}, tracker::TimeTracker,
                        metadata::Dict{String,Any}, config::OutputConfig = output_config_from_parameters(),
                        shtns_config::Union{SHTnsKitConfig,Nothing} = nothing,
                        pencils::Union{NamedTuple,Nothing} = nothing)
    comm = output_comm()
    rank = MPI.Comm_rank(comm)
    current_time = metadata["current_time"]
    current_step = metadata["current_step"]

    should_output = should_output_now(tracker, current_time, config)
    should_restart = should_restart_now(tracker, current_time, config)

    if !should_output && !should_restart
        return false
    end

    # Rank 0 ensures output directory exists
    if rank == 0 && !isdir(config.output_dir)
        mkpath(config.output_dir)
        println("Rank 0: Created output directory: $(config.output_dir)")
    end
    MPI.Barrier(comm)

    # Extract field information
    field_info = extract_field_info(fields, shtns_config, pencils)

    # Write grid file once (rank 0 only)
    if !tracker.grid_file_written && config.include_grid
        write_grid_file!(config, field_info, shtns_config, metadata)
        tracker.grid_file_written = true
        MPI.Barrier(comm)
    end

    # Regular output - all ranks write collectively to single file
    if should_output
        output_number = tracker.output_count + 1
        filename = generate_filename(config, current_time, current_step, "output", output_number)

        if rank == 0
            println("Time $(round(current_time, digits=6)): Writing parallel output #$(output_number)")
        end

        write_start_time = time()

        # Compute diagnostics (local to each rank)
        diagnostics = compute_diagnostics(fields, field_info)
        diagnostics["output_time"] = current_time
        diagnostics["output_number"] = Float64(output_number)

        # Open file collectively
        ds = create_parallel_netcdf(filename, config, field_info, metadata, comm)

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
        write_restart!(fields, tracker, metadata, config, pencils)
    end

    # Update tracker
    update_tracker!(tracker, current_time, config, should_output, should_restart)

    if rank == 0 && should_output
        println("Next output: $(round(tracker.next_output_time, digits=6))")
    end

    return should_output || should_restart
end

# ================================================================================
# Restart Functions (Parallel I/O)
# ================================================================================

function write_restart!(fields::Dict{String,Any}, tracker::TimeTracker,
                        metadata::Dict{String,Any}, config::OutputConfig,
                        pencils::Union{NamedTuple,Nothing}=nothing)
    comm = output_comm()
    rank = MPI.Comm_rank(comm)
    current_time = metadata["current_time"]
    current_step = metadata["current_step"]

    restart_number = tracker.restart_count + 1
    filename = generate_filename(config, current_time, current_step, "restart", restart_number)

    if rank == 0
        println("Writing parallel restart #$(restart_number): $(basename(filename))")
    end

    field_info = extract_field_info(fields, nothing, pencils)

    restart_metadata = copy(metadata)
    restart_metadata["restart_time"] = current_time
    restart_metadata["last_output_time"] = tracker.last_output_time
    restart_metadata["output_count"] = tracker.output_count
    restart_metadata["restart_count"] = tracker.restart_count

    ds = create_parallel_netcdf(filename, config, field_info, restart_metadata, comm)

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
    finally
        close(ds)
    end
end

function read_restart!(tracker::TimeTracker, restart_dir::String,
                        restart_time::Float64, config::OutputConfig,
                        pencils::Union{NamedTuple,Nothing}=nothing)
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
                    restart_data[component] = Dict(
                        "real" => Array(ds[real_name][lm_range, r_range]),
                        "imag" => Array(ds[imag_name][lm_range, r_range])
                    )
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
                           pencils::Union{NamedTuple,Nothing}=nothing)
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
                    restart_data[component] = Dict(
                        "real" => Array(ds[real_name][lm_range, r_range]),
                        "imag" => Array(ds[imag_name][lm_range, r_range])
                    )
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
                               geometry::String="shell",
                               expected_dims::Union{Dict{String,Int},Nothing}=nothing)
    # Look for the single shared file
    pattern = Regex("$(geometry)_hist_$(hist_number)\\.nc")
    files = readdir(output_dir)
    matching = filter(f -> occursin(pattern, f), files)

    if isempty(matching)
        return (false, ["File not found"], Dict{String,Any}(
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
        try
            NCDataset(filepath, "r") do ds
                for (dim_name, expected_size) in expected_dims
                    if haskey(ds.dim, dim_name)
                        actual_size = ds.dim[dim_name]
                        if actual_size != expected_size
                            info["error"] = "Dimension $dim_name: expected $expected_size, got $actual_size"
                            return (false, ["Dimension mismatch"], info)
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
    end

    return (true, String[], info)
end

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

    sort!(file_numbers, by = x -> x[2], rev=true)

    if length(file_numbers) <= keep_last_n
        return
    end

    for (file, _) in file_numbers[(keep_last_n+1):end]
        try
            rm(joinpath(output_dir, file))
            println("Removed old file: $file")
        catch e
            @warn "Failed to remove file: $file"
        end
    end

    println("Cleanup completed. Kept last $keep_last_n output files.")
end

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

function create_shtns_aware_output_config(shtns_config::SHTnsKitConfig, pencils::NamedTuple;
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
