# ================================================================================
# NetCDF Output Writer - Complete Module for Geodynamo Simulation
# Spectral for velocity/magnetic, physical for temperature/composition
# ================================================================================

using MPI
using NetCDF  
using LinearAlgebra
using Statistics
using Dates
using Printf

const comm = MPI.COMM_WORLD

# ================================================================================
# Configuration
# ================================================================================

@enum OutputSpace begin
    MIXED_FIELDS        # Spectral for velocity/magnetic, physical for temperature/composition
    PHYSICAL_ONLY       # Output only physical space data
    SPECTRAL_ONLY       # Output only spectral coefficients
end

@enum FileNaming begin
    RANK_TIME          # filename_rank_XXXX_time_Y.nc
    TIME_RANK          # filename_time_Y_rank_XXXX.nc
end

struct OutputConfig
    output_space::OutputSpace
    naming_scheme::FileNaming
    output_dir::String
    filename_prefix::String
    compression_level::Int
    include_metadata::Bool
    include_grid::Bool
    include_diagnostics::Bool
    output_precision::DataType
    spectral_lmax_output::Int
    overwrite_files::Bool
    independent_writes::Bool

    # Time-based control
    output_interval::Float64
    restart_interval::Float64
    max_output_time::Float64
    time_tolerance::Float64
end

function default_config(; precision::Type{<:AbstractFloat}=Float64,
                        independent_writes::Bool=true)
    return OutputConfig(
        MIXED_FIELDS,       # spectral for velocity/magnetic, physical for temperature
        RANK_TIME,          # naming scheme
        "./output",         # output directory
        "geodynamo",        # filename prefix
        6,                  # compression level
        true,               # include metadata
        true,               # include grid
        true,               # include diagnostics
        precision,          # output precision
        -1,                 # spectral lmax (-1 = all)
        true,               # overwrite files
        independent_writes, # allow ranks to write without synchronization
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
    independent = params.independent_output_files

    return OutputConfig(
        base_config.output_space,
        base_config.naming_scheme,
        base_config.output_dir,
        base_config.filename_prefix,
        base_config.compression_level,
        base_config.include_metadata,
        base_config.include_grid,
        base_config.include_diagnostics,
        precision,
        base_config.spectral_lmax_output,
        base_config.overwrite_files,
        independent,
        base_config.output_interval,
        base_config.restart_interval,
        base_config.max_output_time,
        base_config.time_tolerance
    )
end

function with_output_precision(config::OutputConfig, ::Type{T}) where {T<:AbstractFloat}
    return OutputConfig(
        config.output_space,
        config.naming_scheme,
        config.output_dir,
        config.filename_prefix,
        config.compression_level,
        config.include_metadata,
        config.include_grid,
        config.include_diagnostics,
        T,
        config.spectral_lmax_output,
        config.overwrite_files,
        config.independent_writes,
        config.output_interval,
        config.restart_interval,
        config.max_output_time,
        config.time_tolerance
    )
end

function with_independent_writes(config::OutputConfig, flag::Bool)
    return OutputConfig(
        config.output_space,
        config.naming_scheme,
        config.output_dir,
        config.filename_prefix,
        config.compression_level,
        config.include_metadata,
        config.include_grid,
        config.include_diagnostics,
        config.output_precision,
        config.spectral_lmax_output,
        config.overwrite_files,
        flag,
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
    pencils::Union{NamedTuple, Nothing}
    config::Union{SHTnsKitConfig, Nothing}
    
    # Local range information
    local_ranges::Dict{Symbol, UnitRange{Int}}
end

# Default constructor for FieldInfo
function FieldInfo()
    return FieldInfo(0, 0, 0, 0, Float64[], Float64[], Float64[], 
                     Int[], Int[], nothing, nothing, Dict{Symbol, UnitRange{Int}}())
end

function extract_field_info(fields::Dict{String,Any}, config::Union{SHTnsKitConfig,Nothing}=nothing, 
                           pencils::Union{NamedTuple,Nothing}=nothing)
    # Extract dimensions from available fields with enhanced config integration
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
    
    # Get spectral dimensions from velocity, magnetic, temperature, or composition
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
    r = nr > 0 ? collect(range(0.35, 1.0, length=nr)) : Float64[]
    
    # Create l,m values for spectral modes
    l_values = Int[]
    m_values = Int[]
    if nlm > 0
        # Generate (l,m) pairs - simplified version
        lmax = Int(sqrt(nlm)) + 1
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
    
    return FieldInfo(nlat, nlon, nr, nlm, theta, phi, r, l_values, m_values, 
                     pencils, config, local_ranges)
end

# ================================================================================
# Filename Generation
# ================================================================================

function generate_filename(config::OutputConfig, time::Float64, step::Int, 
                            rank::Int, file_type::String = "output")
    time_str = @sprintf("%.6f", time)
    time_str = replace(time_str, "." => "p")
    # Add geometry tag from global parameters for clarity
    geom = try
        string(get_parameters().geometry)
    catch
        "shell"
    end
    geom_tag = "geom_" * geom
    
    filename = if config.naming_scheme == RANK_TIME
        "$(config.filename_prefix)_$(file_type)_$(geom_tag)_rank_$(lpad(rank, 4, '0'))_time_$(time_str).nc"
    else  # TIME_RANK
        "$(config.filename_prefix)_$(file_type)_$(geom_tag)_time_$(time_str)_rank_$(lpad(rank, 4, '0')).nc"
    end
    
    return joinpath(config.output_dir, filename)
end

# ================================================================================
# NetCDF File Operations
# ================================================================================

function create_netcdf_file(filename::String, config::OutputConfig, 
                            field_info::FieldInfo, metadata::Dict{String,Any})
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    if config.overwrite_files && isfile(filename)
        rm(filename)
    end
    
    nc_file = NetCDF.create(filename, NetCDF.NcVar[]; mode=NetCDF.NC_NETCDF4)
    
    # Global attributes
    NetCDF.putatt(nc_file, "title", "Geodynamo Simulation Output")
    NetCDF.putatt(nc_file, "source", "Geodynamo simulation code")
    NetCDF.putatt(nc_file, "history", "Created on $(now()) by rank $rank")
    NetCDF.putatt(nc_file, "Conventions", "CF-1.8")
    NetCDF.putatt(nc_file, "mpi_rank", rank)
    NetCDF.putatt(nc_file, "mpi_total_ranks", nprocs)
    
    # Add simulation metadata
    if config.include_metadata
        # Ensure geometry attribute exists
        if !haskey(metadata, "geometry")
            try
                metadata["geometry"] = string(get_parameters().geometry)
            catch
                metadata["geometry"] = "shell"
            end
        end
        for (key, value) in metadata
            try
                NetCDF.putatt(nc_file, key, value)
            catch
                # Skip problematic attributes
            end
        end
    end

    # Define metadata record variables (as dataset fields)
    # Create a dedicated scalar dimension for metadata
    meta_dim = NetCDF.defDim(nc_file, "meta", 1)
    # Geometry recorded as a scalar string variable
    NetCDF.defVar(nc_file, "geometry", String, (meta_dim,))
    
    return nc_file
end

function setup_coordinates!(nc_file, field_info::FieldInfo, config::OutputConfig)
    if !config.include_grid
        return
    end
    
    # Time (always present)
    time_dim = NetCDF.defDim(nc_file, "time", 1)
    time_var = NetCDF.defVar(nc_file, "time", config.output_precision, (time_dim,))
    NetCDF.putatt(nc_file, time_var, "long_name", "simulation_time")
    NetCDF.putatt(nc_file, time_var, "units", "dimensionless")
    
    step_var = NetCDF.defVar(nc_file, "step", Int32, (time_dim,))
    NetCDF.putatt(nc_file, step_var, "long_name", "simulation_step")
    
    # Radial (shared by all fields)
    if field_info.nr > 0
        r_dim = NetCDF.defDim(nc_file, "r", field_info.nr)
        r_var = NetCDF.defVar(nc_file, "r", config.output_precision, (r_dim,))
        NetCDF.putatt(nc_file, r_var, "long_name", "radial_coordinate")
        NetCDF.putatt(nc_file, r_var, "units", "dimensionless")
        NetCDF.putatt(nc_file, r_var, "valid_range", [0.35, 1.0])
    end
    
    # Physical coordinates (for temperature/composition)
    if config.output_space == MIXED_FIELDS || config.output_space == PHYSICAL_ONLY
        if field_info.nlat > 0
            theta_dim = NetCDF.defDim(nc_file, "theta", field_info.nlat)
            theta_var = NetCDF.defVar(nc_file, "theta", config.output_precision, (theta_dim,))
            NetCDF.putatt(nc_file, theta_var, "long_name", "colatitude")
            NetCDF.putatt(nc_file, theta_var, "units", "radians")
            NetCDF.putatt(nc_file, theta_var, "valid_range", [0.0, π])
        end
        
        if field_info.nlon > 0
            phi_dim = NetCDF.defDim(nc_file, "phi", field_info.nlon)
            phi_var = NetCDF.defVar(nc_file, "phi", config.output_precision, (phi_dim,))
            NetCDF.putatt(nc_file, phi_var, "long_name", "azimuthal_angle")
            NetCDF.putatt(nc_file, phi_var, "units", "radians")
            NetCDF.putatt(nc_file, phi_var, "valid_range", [0.0, 2π])
        end
    end
    
    # Spectral coordinates (for velocity/magnetic)
    if config.output_space == MIXED_FIELDS || config.output_space == SPECTRAL_ONLY
        if field_info.nlm > 0
            lm_dim = NetCDF.defDim(nc_file, "spectral_mode", field_info.nlm)
            
            l_var = NetCDF.defVar(nc_file, "l_values", Int32, (lm_dim,))
            NetCDF.putatt(nc_file, l_var, "long_name", "spherical_harmonic_degree")
            
            m_var = NetCDF.defVar(nc_file, "m_values", Int32, (lm_dim,))
            NetCDF.putatt(nc_file, m_var, "long_name", "spherical_harmonic_order")
        end
    end
end

function setup_field_variables!(nc_file, field_info::FieldInfo, config::OutputConfig, 
                                available_fields::Vector{String})
    if config.output_space == MIXED_FIELDS
        # Temperature: Physical space
        if "temperature" in available_fields && field_info.nlat > 0 && field_info.nlon > 0
            dims = ["theta", "phi", "r"]
            var_dims = tuple([NetCDF.dimid(nc_file, d) for d in dims]...)
            
            temp_var = NetCDF.defVar(nc_file, "temperature", config.output_precision, var_dims)
            NetCDF.putatt(nc_file, temp_var, "long_name", "temperature")
            NetCDF.putatt(nc_file, temp_var, "units", "dimensionless")
            NetCDF.putatt(nc_file, temp_var, "representation", "physical_space")
            
            if config.compression_level > 0
                NetCDF.defVarDeflate(nc_file, temp_var, true, true, config.compression_level)
            end
        end
        
        # Composition: Physical space
        if "composition" in available_fields && field_info.nlat > 0 && field_info.nlon > 0
            dims = ["theta", "phi", "r"]
            var_dims = tuple([NetCDF.dimid(nc_file, d) for d in dims]...)
            
            comp_var = NetCDF.defVar(nc_file, "composition", config.output_precision, var_dims)
            NetCDF.putatt(nc_file, comp_var, "long_name", "composition")
            NetCDF.putatt(nc_file, comp_var, "units", "dimensionless")
            NetCDF.putatt(nc_file, comp_var, "representation", "physical_space")
            
            if config.compression_level > 0
                NetCDF.defVarDeflate(nc_file, comp_var, true, true, config.compression_level)
            end
        end
        
        # Velocity, Magnetic, Temperature, Composition: Spectral space
        if field_info.nlm > 0
            spec_dims = tuple([NetCDF.dimid(nc_file, "spectral_mode"), NetCDF.dimid(nc_file, "r")]...)
            
            # All spectral components (velocity, magnetic, temperature, composition)
            for component in ["velocity_toroidal", "velocity_poloidal", "magnetic_toroidal", "magnetic_poloidal", "temperature_spectral", "composition_spectral"]
                if component in available_fields
                    real_var = NetCDF.defVar(nc_file, "$(component)_real", config.output_precision, spec_dims)
                    imag_var = NetCDF.defVar(nc_file, "$(component)_imag", config.output_precision, spec_dims)
                    
                    NetCDF.putatt(nc_file, real_var, "long_name", "$(component)_real_coefficients")
                    NetCDF.putatt(nc_file, real_var, "representation", "spectral_space")
                    NetCDF.putatt(nc_file, imag_var, "long_name", "$(component)_imaginary_coefficients")
                    NetCDF.putatt(nc_file, imag_var, "representation", "spectral_space")
                    
                    if config.compression_level > 0
                        NetCDF.defVarDeflate(nc_file, real_var, true, true, config.compression_level)
                        NetCDF.defVarDeflate(nc_file, imag_var, true, true, config.compression_level)
                    end
                end
            end
        end
    end
end

function setup_diagnostics!(nc_file, diagnostics::Dict{String,Float64}, config::OutputConfig)
    if !config.include_diagnostics || isempty(diagnostics)
        return
    end
    
    scalar_dim = NetCDF.defDim(nc_file, "scalar", 1)
    
    for (name, value) in diagnostics
        var = NetCDF.defVar(nc_file, "diag_$(name)", config.output_precision, (scalar_dim,))
        NetCDF.putatt(nc_file, var, "long_name", replace(name, "_" => " "))
    end
end

# ================================================================================
# Data Writing
# ================================================================================

function write_coordinates!(nc_file, field_info::FieldInfo, config::OutputConfig)
    # Write coordinate data
    if !isempty(field_info.theta)
        NetCDF.putvar(nc_file, "theta", config.output_precision.(field_info.theta))
    end
    if !isempty(field_info.phi)
        NetCDF.putvar(nc_file, "phi", config.output_precision.(field_info.phi))
    end
    if !isempty(field_info.r)
        NetCDF.putvar(nc_file, "r", config.output_precision.(field_info.r))
    end
    if !isempty(field_info.l_values)
        NetCDF.putvar(nc_file, "l_values", field_info.l_values)
        NetCDF.putvar(nc_file, "m_values", field_info.m_values)
    end
end

function write_field_data!(nc_file, fields::Dict{String,Any}, config::OutputConfig, 
                          field_info::FieldInfo=FieldInfo())
    # Write temperature (physical space) with enhanced memory access
    if haskey(fields, "temperature") && NetCDF.varid(nc_file, "temperature") != -1
        T_data = fields["temperature"]
        
        # Use enhanced data copying for large arrays
        if length(T_data) > 10000
            data_out = similar(T_data, config.output_precision)
            copyto!(data_out, T_data)
        else
            data_out = config.output_precision.(T_data)
        end
        
        NetCDF.putvar(nc_file, "temperature", data_out)
    end
    
    # Write composition (physical space) with enhanced memory access
    if haskey(fields, "composition") && NetCDF.varid(nc_file, "composition") != -1
        C_data = fields["composition"]
        
        # Use enhanced data copying for large arrays
        if length(C_data) > 10000
            data_out = similar(C_data, config.output_precision)
            copyto!(data_out, C_data)
        else
            data_out = config.output_precision.(C_data)
        end
        
        NetCDF.putvar(nc_file, "composition", data_out)
    end
    
    # Write all spectral fields (velocity, magnetic, temperature, composition) with memory optimization
    for component in ["velocity_toroidal", "velocity_poloidal", "magnetic_toroidal", "magnetic_poloidal", "temperature_spectral", "composition_spectral"]
        if haskey(fields, component)
            field_data = fields[component]
            if haskey(field_data, "real") && haskey(field_data, "imag")
                real_name = "$(component)_real"
                imag_name = "$(component)_imag"
                
                if NetCDF.varid(nc_file, real_name) != -1 && NetCDF.varid(nc_file, imag_name) != -1
                    # Handle 3D arrays (nlm, 1, nr) -> (nlm, nr) with enhanced processing
                    real_data = field_data["real"]
                    imag_data = field_data["imag"]
                    
                    if ndims(real_data) == 3
                        real_data = real_data[:, 1, :]
                        imag_data = imag_data[:, 1, :]
                    end
                    
                    # Use enhanced conversion for large spectral arrays
                    if length(real_data) > 5000
                        real_out = similar(real_data, config.output_precision)
                        imag_out = similar(imag_data, config.output_precision)
                        copyto!(real_out, real_data)
                        copyto!(imag_out, imag_data)
                    else
                        real_out = config.output_precision.(real_data)
                        imag_out = config.output_precision.(imag_data)
                    end
                    
                    # Use parallel I/O if available and data is large enough
                    if length(real_out) > 100000 && field_info.pencils !== nothing
                        write_spectral_data_parallel!(nc_file, real_name, imag_name, 
                                                     real_out, imag_out, field_info)
                    else
                        NetCDF.putvar(nc_file, real_name, real_out)
                        NetCDF.putvar(nc_file, imag_name, imag_out)
                    end
                end
            end
        end
    end
end

function write_time_data!(nc_file, time::Float64, step::Int, output_num::Int, config::OutputConfig)
    NetCDF.putvar(nc_file, "time", [config.output_precision(time)])
    NetCDF.putvar(nc_file, "step", [Int32(step)])
    # Also write geometry metadata field if present
    try
        NetCDF.putvar(nc_file, "geometry", [string(get_parameters().geometry)])
    catch
        # Ignore if variable not defined or NetCDF backend does not support String variables
    end
end

function write_diagnostics!(nc_file, diagnostics::Dict{String,Float64}, config::OutputConfig)
    if !config.include_diagnostics
        return
    end
    
    for (name, value) in diagnostics
        var_name = "diag_$(name)"
        if NetCDF.varid(nc_file, var_name) != -1
            NetCDF.putvar(nc_file, var_name, [config.output_precision(value)])
        end
    end
end


# ================================================================================
# Enhanced Parallel I/O Functions  
# ================================================================================

"""
    write_spectral_data_parallel!(nc_file, real_name, imag_name, real_data, imag_data, field_info)
    
Write spectral data using parallel I/O strategies based on pencil decomposition
"""
function write_spectral_data_parallel!(nc_file, real_name::String, imag_name::String,
                                      real_data::AbstractArray, imag_data::AbstractArray,
                                      field_info::FieldInfo)
    # For now, fall back to regular write - parallel NetCDF would require additional setup
    # In a full implementation, this would use collective I/O operations
    NetCDF.putvar(nc_file, real_name, real_data)
    NetCDF.putvar(nc_file, imag_name, imag_data)
end


"""
    create_memory_efficient_output_buffer(data_size::Int, precision::DataType)
    
Create appropriately sized buffer for output operations
"""
function create_memory_efficient_output_buffer(data_size::Int, precision::DataType)
    # Use memory mapping for very large arrays
    if data_size > 1000000  # 1M elements
        return zeros(precision, data_size)
    else
        return Vector{precision}(undef, data_size)
    end
end


"""
    optimize_field_data_layout!(field_data::Dict, field_info::FieldInfo)
    
Optimize data layout for output based on pencil decomposition
"""
function optimize_field_data_layout!(field_data::Dict, field_info::FieldInfo)
    if field_info.pencils === nothing
        return field_data
    end
    
    # This would implement data reorganization for optimal I/O
    # For now, return as-is since full implementation requires transpose operations
    return field_data
end


"""
    batch_write_spectral_fields!(nc_file, fields::Dict, config::OutputConfig, field_info::FieldInfo)
    
Write multiple spectral fields in batched operations for better I/O performance
"""
function batch_write_spectral_fields!(nc_file, fields::Dict{String,Any}, 
                                     config::OutputConfig, field_info::FieldInfo)
    # Collect all spectral components for batch processing
    spectral_components = String[]
    for component in ["velocity_toroidal", "velocity_poloidal", "magnetic_toroidal", "magnetic_poloidal", "temperature_spectral", "composition_spectral"]
        if haskey(fields, component)
            push!(spectral_components, component)
        end
    end
    
    if isempty(spectral_components)
        return
    end
    
    # Process in batches to reduce memory pressure
    batch_size = min(2, length(spectral_components))  # Process 2 components at a time
    
    for i in 1:batch_size:length(spectral_components)
        batch_end = min(i + batch_size - 1, length(spectral_components))
        batch_components = spectral_components[i:batch_end]
        
        # Pre-allocate buffers for the batch
        batch_data = Dict{String, Any}()
        
        for component in batch_components
            field_data = fields[component]
            if haskey(field_data, "real") && haskey(field_data, "imag")
                # Process data layout optimization
                enhanced_data = optimize_field_data_layout!(field_data, field_info)
                batch_data[component] = enhanced_data
            end
        end
        
        # Write the batch
        for component in batch_components
            if haskey(batch_data, component)
                write_single_spectral_component!(nc_file, component, batch_data[component], config)
            end
        end
        
        # Clear batch data to free memory
        empty!(batch_data)
    end
end


"""
    write_single_spectral_component!(nc_file, component, field_data, config)
    
Write a single spectral component with enhanced memory handling
"""
function write_single_spectral_component!(nc_file, component::String, field_data::Dict, 
                                         config::OutputConfig)
    real_name = "$(component)_real"
    imag_name = "$(component)_imag"
    
    if NetCDF.varid(nc_file, real_name) == -1 || NetCDF.varid(nc_file, imag_name) == -1
        return
    end
    
    real_data = field_data["real"]
    imag_data = field_data["imag"]
    
    # Handle 3D arrays (nlm, 1, nr) -> (nlm, nr)
    if ndims(real_data) == 3
        real_data = real_data[:, 1, :]
        imag_data = imag_data[:, 1, :]
    end
    
    # Memory-efficient conversion
    data_size = length(real_data)
    if data_size > 10000
        real_buffer = create_memory_efficient_output_buffer(data_size, config.output_precision)
        imag_buffer = create_memory_efficient_output_buffer(data_size, config.output_precision)
        
        copyto!(real_buffer, real_data)
        copyto!(imag_buffer, imag_data)
        
        NetCDF.putvar(nc_file, real_name, reshape(real_buffer, size(real_data)))
        NetCDF.putvar(nc_file, imag_name, reshape(imag_buffer, size(imag_data)))
    else
        NetCDF.putvar(nc_file, real_name, config.output_precision.(real_data))
        NetCDF.putvar(nc_file, imag_name, config.output_precision.(imag_data))
    end
end

# ================================================================================
# Grid File Writing (One-time)
# ================================================================================

"""
    write_grid_file!(config::OutputConfig, field_info::FieldInfo,
                    shtns_config::Union{SHTnsKitConfig,Nothing},
                    metadata::Dict{String,Any})

Write a separate grid file containing all coordinate and grid information.
This is written only once by rank 0 at the start of the simulation.
"""
function write_grid_file!(config::OutputConfig, field_info::FieldInfo,
                         shtns_config::Union{SHTnsKitConfig,Nothing},
                         metadata::Dict{String,Any})
    rank = MPI.Comm_rank(comm)

    # Only rank 0 writes the grid file
    if rank != 0
        return
    end

    # Generate grid filename
    geom = try
        string(get_parameters().geometry)
    catch
        "shell"
    end
    geom_tag = "geom_" * geom
    grid_filename = joinpath(config.output_dir,
                            "$(config.filename_prefix)_grid_$(geom_tag).nc")

    # Remove existing grid file if overwrite is enabled
    if config.overwrite_files && isfile(grid_filename)
        rm(grid_filename)
    end

    # Create grid file
    nc_file = NetCDF.create(grid_filename, NetCDF.NcVar[]; mode=NetCDF.NC_NETCDF4)

    try
        # Global attributes
        NetCDF.putatt(nc_file, "title", "Geodynamo Simulation Grid Information")
        NetCDF.putatt(nc_file, "description", "Grid coordinates and geometry information")
        NetCDF.putatt(nc_file, "source", "Geodynamo simulation code")
        NetCDF.putatt(nc_file, "created", string(now()))
        NetCDF.putatt(nc_file, "Conventions", "CF-1.8")

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
                    NetCDF.putatt(nc_file, key, value)
                catch
                    # Skip problematic attributes
                end
            end
        end

        # Define dimensions
        if field_info.nr > 0
            r_dim = NetCDF.defDim(nc_file, "r", field_info.nr)
        end

        if field_info.nlat > 0
            theta_dim = NetCDF.defDim(nc_file, "theta", field_info.nlat)
        end

        if field_info.nlon > 0
            phi_dim = NetCDF.defDim(nc_file, "phi", field_info.nlon)
        end

        if field_info.nlm > 0
            lm_dim = NetCDF.defDim(nc_file, "spectral_mode", field_info.nlm)
        end

        # Define coordinate variables
        if field_info.nr > 0
            r_var = NetCDF.defVar(nc_file, "r", config.output_precision, (r_dim,))
            NetCDF.putatt(nc_file, r_var, "long_name", "radial_coordinate")
            NetCDF.putatt(nc_file, r_var, "units", "dimensionless")
            NetCDF.putatt(nc_file, r_var, "valid_range", [0.35, 1.0])
            NetCDF.putatt(nc_file, r_var, "description", "Normalized radial coordinate (inner=0.35, outer=1.0)")
        end

        if field_info.nlat > 0
            theta_var = NetCDF.defVar(nc_file, "theta", config.output_precision, (theta_dim,))
            NetCDF.putatt(nc_file, theta_var, "long_name", "colatitude")
            NetCDF.putatt(nc_file, theta_var, "units", "radians")
            NetCDF.putatt(nc_file, theta_var, "valid_range", [0.0, π])
            NetCDF.putatt(nc_file, theta_var, "description", "Colatitude from north pole (0 to π)")
        end

        if field_info.nlon > 0
            phi_var = NetCDF.defVar(nc_file, "phi", config.output_precision, (phi_dim,))
            NetCDF.putatt(nc_file, phi_var, "long_name", "azimuthal_angle")
            NetCDF.putatt(nc_file, phi_var, "units", "radians")
            NetCDF.putatt(nc_file, phi_var, "valid_range", [0.0, 2π])
            NetCDF.putatt(nc_file, phi_var, "description", "Longitude angle (0 to 2π)")
        end

        if field_info.nlm > 0
            l_var = NetCDF.defVar(nc_file, "l_values", Int32, (lm_dim,))
            NetCDF.putatt(nc_file, l_var, "long_name", "spherical_harmonic_degree")
            NetCDF.putatt(nc_file, l_var, "description", "Spherical harmonic degree l")

            m_var = NetCDF.defVar(nc_file, "m_values", Int32, (lm_dim,))
            NetCDF.putatt(nc_file, m_var, "long_name", "spherical_harmonic_order")
            NetCDF.putatt(nc_file, m_var, "description", "Spherical harmonic order m")
        end

        # Add SHTns-specific information if available
        if shtns_config !== nothing
            NetCDF.putatt(nc_file, "shtns_lmax", shtns_config.lmax)
            NetCDF.putatt(nc_file, "shtns_mmax", shtns_config.mmax)
            NetCDF.putatt(nc_file, "shtns_nlm", shtns_config.nlm)
            NetCDF.putatt(nc_file, "shtns_nlat", shtns_config.nlat)
            NetCDF.putatt(nc_file, "shtns_nlon", shtns_config.nlon)
            NetCDF.putatt(nc_file, "grid_type_theta", "gaussian")
            NetCDF.putatt(nc_file, "grid_type_phi", "equispaced")

            # Define and write Gaussian quadrature weights if available
            if !isempty(shtns_config.gauss_weights)
                gauss_dim = NetCDF.dimid(nc_file, "theta")
                weights_var = NetCDF.defVar(nc_file, "gauss_weights", config.output_precision, (gauss_dim,))
                NetCDF.putatt(nc_file, weights_var, "long_name", "gaussian_quadrature_weights")
                NetCDF.putatt(nc_file, weights_var, "description", "Quadrature weights for Gaussian grid integration")
            end
        end

        # End definition mode
        NetCDF.endDef(nc_file)

        # Write coordinate data
        if !isempty(field_info.r)
            NetCDF.putvar(nc_file, "r", config.output_precision.(field_info.r))
        end

        if !isempty(field_info.theta)
            NetCDF.putvar(nc_file, "theta", config.output_precision.(field_info.theta))
        end

        if !isempty(field_info.phi)
            NetCDF.putvar(nc_file, "phi", config.output_precision.(field_info.phi))
        end

        if !isempty(field_info.l_values)
            NetCDF.putvar(nc_file, "l_values", field_info.l_values)
            NetCDF.putvar(nc_file, "m_values", field_info.m_values)
        end

        # Write Gaussian weights if available
        if shtns_config !== nothing && !isempty(shtns_config.gauss_weights)
            try
                NetCDF.putvar(nc_file, "gauss_weights", config.output_precision.(shtns_config.gauss_weights))
            catch
                # Skip if weights cannot be written
            end
        end

        println("Rank 0: Successfully wrote grid file: $grid_filename")

    finally
        NetCDF.close(nc_file)
    end
end

# ================================================================================
# Diagnostics Computation
# ================================================================================

function compute_diagnostics(fields::Dict{String,Any}, field_info::FieldInfo)
    diagnostics = Dict{String, Float64}()
    
    # Temperature statistics
    if haskey(fields, "temperature")
        T = fields["temperature"]
        diagnostics["temp_mean"] = mean(T)
        diagnostics["temp_std"] = std(T)
        diagnostics["temp_min"] = minimum(T)
        diagnostics["temp_max"] = maximum(T)
        
        # Add radial profile statistics if field_info has ranges
        if haskey(field_info.local_ranges, :r) && !isempty(field_info.local_ranges[:r])
            r_range = field_info.local_ranges[:r]
            if length(r_range) > 1
                radial_mean = mean(T, dims=(1,2))[:, :, :]
                diagnostics["temp_radial_variation"] = std(radial_mean)
            end
        end
    end
    
    # Composition statistics
    if haskey(fields, "composition")
        C = fields["composition"]
        diagnostics["comp_mean"] = mean(C)
        diagnostics["comp_std"] = std(C)
        diagnostics["comp_min"] = minimum(C)
        diagnostics["comp_max"] = maximum(C)
        
        # Add radial profile statistics if field_info has ranges
        if haskey(field_info.local_ranges, :r) && !isempty(field_info.local_ranges[:r])
            r_range = field_info.local_ranges[:r]
            if length(r_range) > 1
                radial_mean = mean(C, dims=(1,2))[:, :, :]
                diagnostics["comp_radial_variation"] = std(radial_mean)
            end
        end
    end
    
    # Enhanced spectral field statistics with config-aware processing
    for component in ["velocity_toroidal", "velocity_poloidal", "magnetic_toroidal", "magnetic_poloidal", "temperature_spectral", "composition_spectral"]
        if haskey(fields, component)
            field_data = fields[component]
            if haskey(field_data, "real") && haskey(field_data, "imag")
                real_part = field_data["real"]
                imag_part = field_data["imag"]
                
                # In-place computation to avoid temporary arrays
                energy = zero(eltype(real_part))
                for i in eachindex(real_part, imag_part)
                    magnitude_sq = real_part[i]^2 + imag_part[i]^2
                    energy += magnitude_sq
                end
                diagnostics["$(component)_energy"] = 0.5 * energy
                
                # Compute RMS and max without temporary arrays
                sum_magnitude_sq = zero(eltype(real_part))
                max_magnitude = zero(eltype(real_part))
                for i in eachindex(real_part, imag_part)
                    magnitude_sq = real_part[i]^2 + imag_part[i]^2
                    magnitude = sqrt(magnitude_sq)
                    sum_magnitude_sq += magnitude_sq
                    max_magnitude = max(max_magnitude, magnitude)
                end
                
                diagnostics["$(component)_rms"] = sqrt(sum_magnitude_sq / length(real_part))
                diagnostics["$(component)_max"] = max_magnitude
                
                # Add spectral energy distribution if l_values are available
                if field_info.config !== nothing && !isempty(field_info.l_values)
                    compute_spectral_energy_diagnostics!(diagnostics, component, 
                                                        real_part, imag_part, field_info)
                end
            end
        end
    end
    
    return diagnostics
end


"""
    compute_spectral_energy_diagnostics!(diagnostics, component, real_part, imag_part, field_info)
    
Compute spectral energy distribution diagnostics using SHTns configuration
"""
function compute_spectral_energy_diagnostics!(diagnostics::Dict{String,Float64}, 
                                            component::String,
                                            real_part::AbstractArray, 
                                            imag_part::AbstractArray,
                                            field_info::FieldInfo)
    if field_info.config === nothing
        return
    end
    
    config = field_info.config
    l_values = config.l_values
    
    # Compute energy per l mode
    l_max = maximum(l_values)
    l_energies = zeros(Float64, l_max + 1)
    
    for (idx, l) in enumerate(l_values)
        if idx <= size(real_part, 1)
            # In-place computation for l-mode energy
            l_energy = zero(eltype(real_part))
            for j in axes(real_part, 2), k in axes(real_part, 3)
                l_energy += real_part[idx, j, k]^2 + imag_part[idx, j, k]^2
            end
            l_energies[l + 1] += l_energy
        end
    end
    
    # Store key spectral diagnostics
    total_energy = sum(l_energies)
    if total_energy > 0
        # Peak spectral degree
        peak_l = argmax(l_energies) - 1
        diagnostics["$(component)_peak_l"] = Float64(peak_l)
        
        # Spectral centroid (weighted average l)
        spectral_centroid = sum((0:l_max) .* l_energies) / total_energy
        diagnostics["$(component)_spectral_centroid"] = spectral_centroid
        
        # Energy in low modes (l <= 10)
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
    rank = MPI.Comm_rank(comm)
    current_time = metadata["current_time"]
    current_step = metadata["current_step"]
    
    # Check if output is needed
    should_output = should_output_now(tracker, current_time, config)
    should_restart = should_restart_now(tracker, current_time, config)
    
    if !should_output && !should_restart
        return false
    end
    
    # Ensure output directory exists for each rank
    if config.independent_writes
        dir_existed = isdir(config.output_dir)
        mkpath(config.output_dir)
        if rank == 0 && !dir_existed
            println("Created output directory: $(config.output_dir)")
        end
    else
        dir_existed = isdir(config.output_dir)
        if rank == 0 && !dir_existed
            mkpath(config.output_dir)
            println("Created output directory: $(config.output_dir)")
        end
        MPI.Barrier(comm)
    end
    
    # Extract field information (needed for both grid file and regular output)
    field_info = extract_field_info(fields, shtns_config, pencils)

    # Write grid file once at the first output (only rank 0)
    if !tracker.grid_file_written && config.include_grid
        write_grid_file!(config, field_info, shtns_config, metadata)
        tracker.grid_file_written = true
        # Ensure all ranks know the grid file has been written
        if !config.independent_writes
            MPI.Barrier(comm)
        end
    end

    # Regular output
    if should_output
        if rank == 0 && !config.independent_writes
            println("Time $(current_time): Writing mixed field output")
        end

        # Generate filename
        filename = generate_filename(config, current_time, current_step, rank, "output")
        println("Rank $rank: Writing $filename")
        
        # Compute diagnostics
        diagnostics = compute_diagnostics(fields, field_info)
        diagnostics["output_time"] = current_time
        diagnostics["output_number"] = tracker.output_count + 1
        
        # Create NetCDF file
        nc_file = create_netcdf_file(filename, config, field_info, metadata)
        
        try
            # Setup file structure
            setup_coordinates!(nc_file, field_info, config)
            
            available_fields = collect(keys(fields))
            setup_field_variables!(nc_file, field_info, config, available_fields)
            setup_diagnostics!(nc_file, diagnostics, config)
            
            # End definition mode
            NetCDF.endDef(nc_file)
            
            # Write all data
            write_coordinates!(nc_file, field_info, config)
            write_field_data!(nc_file, fields, config)
            write_time_data!(nc_file, current_time, current_step, tracker.output_count + 1, config)
            write_diagnostics!(nc_file, diagnostics, config)
            
            println("Rank $rank: Successfully wrote output at time $current_time")
            
        finally
            NetCDF.close(nc_file)
        end
    end
    
    # Restart file
    if should_restart
        write_restart!(fields, tracker, metadata, config)
    end
    
    # Update tracker
    update_tracker!(tracker, current_time, config, should_output, should_restart)
    
    if !config.independent_writes
        MPI.Barrier(comm)
        if rank == 0 && should_output
            println("All ranks completed output at time $current_time")
            println("Next output: $(tracker.next_output_time)")
        end
    end

    return should_output || should_restart
end

# ================================================================================
# Restart Functions
# ================================================================================

function write_restart!(fields::Dict{String,Any}, tracker::TimeTracker, 
                        metadata::Dict{String,Any}, config::OutputConfig)
    rank = MPI.Comm_rank(comm)
    current_time = metadata["current_time"]
    current_step = metadata["current_step"]
    
    # Generate restart filename
    filename = generate_filename(config, current_time, current_step, rank, "restart")
    println("Rank $rank: Writing restart $filename")
    
    # Extract field information
    field_info = extract_field_info(fields)
    
    # Enhanced metadata for restart
    restart_metadata = copy(metadata)
    restart_metadata["restart_time"] = current_time
    restart_metadata["last_output_time"] = tracker.last_output_time
    restart_metadata["output_count"] = tracker.output_count
    restart_metadata["restart_count"] = tracker.restart_count
    
    nc_file = create_netcdf_file(filename, config, field_info, restart_metadata)
    
    try
        # Setup restart file (use same structure as regular output)
        setup_coordinates!(nc_file, field_info, config)
        available_fields = collect(keys(fields))
        setup_field_variables!(nc_file, field_info, config, available_fields)
        
        # Additional restart variables
        scalar_dim = NetCDF.defDim(nc_file, "scalar", 1)
        NetCDF.defVar(nc_file, "last_output_time", config.output_precision, (scalar_dim,))
        NetCDF.defVar(nc_file, "output_count", Int32, (scalar_dim,))
        NetCDF.defVar(nc_file, "restart_count", Int32, (scalar_dim,))
        NetCDF.defVar(nc_file, "grid_file_written", Int32, (scalar_dim,))
        
        # End definition mode
        NetCDF.endDef(nc_file)
        
        # Write all data
        write_coordinates!(nc_file, field_info, config)
        write_field_data!(nc_file, fields, config)
        write_time_data!(nc_file, current_time, current_step, tracker.restart_count + 1, config)
        
        # Write restart-specific data
        NetCDF.putvar(nc_file, "last_output_time", [config.output_precision(tracker.last_output_time)])
        NetCDF.putvar(nc_file, "output_count", [Int32(tracker.output_count)])
        NetCDF.putvar(nc_file, "restart_count", [Int32(tracker.restart_count)])
        NetCDF.putvar(nc_file, "grid_file_written", [Int32(tracker.grid_file_written ? 1 : 0)])
        
    finally
        NetCDF.close(nc_file)
    end
end

function read_restart!(tracker::TimeTracker, restart_dir::String, 
                        restart_time::Float64, config::OutputConfig)
    rank = MPI.Comm_rank(comm)
    
    # Find restart file for this rank near the target time
    restart_files = find_restart_files(restart_dir, restart_time, rank)
    
    if isempty(restart_files)
        error("Rank $rank: No restart files found near time $restart_time")
    end
    
    filename = restart_files[1]
    println("Rank $rank: Reading restart $filename")
    
    restart_data = Dict{String, Any}()
    metadata = Dict{String, Any}()
    
    nc_file = NetCDF.open(filename, NC_NOWRITE)

    try
        # Read field data for this rank
        var_names = collect(keys(NetCDF.varnames(nc_file)))

        # Read metadata (broadcast from rank 0)
        if rank == 0
            metadata["current_time"] = NetCDF.readvar(nc_file, "time")[1]
            metadata["current_step"] = NetCDF.readvar(nc_file, "step")[1]

            # Read tracker state
            tracker.last_output_time = NetCDF.readvar(nc_file, "last_output_time")[1]
            tracker.output_count = NetCDF.readvar(nc_file, "output_count")[1]
            tracker.restart_count = NetCDF.readvar(nc_file, "restart_count")[1]

            # Read grid file written flag if available
            if "grid_file_written" in var_names
                tracker.grid_file_written = NetCDF.readvar(nc_file, "grid_file_written")[1] != 0
            else
                # For backwards compatibility with old restart files
                tracker.grid_file_written = true
            end

            # Update next times
            tracker.next_output_time = tracker.last_output_time + config.output_interval
            tracker.next_restart_time = tracker.last_restart_time + config.restart_interval
        end
        
        # Temperature (physical)
        if "temperature" in var_names
            restart_data["temperature"] = NetCDF.readvar(nc_file, "temperature")
        end
        
        # Composition (physical)
        if "composition" in var_names
            restart_data["composition"] = NetCDF.readvar(nc_file, "composition")
        end
        
        # Spectral fields
        for component in ["velocity_toroidal", "velocity_poloidal", "magnetic_toroidal", "magnetic_poloidal", "temperature_spectral", "composition_spectral"]
            real_name = "$(component)_real"
            imag_name = "$(component)_imag"
            
            if real_name in var_names && imag_name in var_names
                restart_data[component] = Dict(
                    "real" => NetCDF.readvar(nc_file, real_name),
                    "imag" => NetCDF.readvar(nc_file, imag_name)
                )
            end
        end
        
    finally
        NetCDF.close(nc_file)
    end
    
    # Broadcast metadata and tracker state
    for key in ["current_time", "current_step"]
        if haskey(metadata, key)
            metadata[key] = MPI.bcast(metadata[key], 0, comm)
        end
    end
    
    tracker.last_output_time = MPI.bcast(tracker.last_output_time, 0, comm)
    tracker.output_count = MPI.bcast(tracker.output_count, 0, comm)
    tracker.restart_count = MPI.bcast(tracker.restart_count, 0, comm)
    tracker.next_output_time = MPI.bcast(tracker.next_output_time, 0, comm)
    tracker.next_restart_time = MPI.bcast(tracker.next_restart_time, 0, comm)
    tracker.grid_file_written = MPI.bcast(tracker.grid_file_written, 0, comm)
    
    if rank == 0
        println("Restart completed from time $(metadata["current_time"])")
    end
    
    return restart_data, metadata
end

# ================================================================================
# Utility Functions
# ================================================================================

function find_restart_files(restart_dir::String, target_time::Float64, rank::Int)
    files = readdir(restart_dir)
    restart_files = filter(f -> endswith(f, ".nc") && contains(f, "restart") && 
                            contains(f, "rank_$(lpad(rank, 4, '0'))"), files)
    
    if isempty(restart_files)
        return String[]
    end
    
    # Extract times and find closest
    time_pattern = r"time_(\d+p\d+)"
    file_times = Tuple{String, Float64}[]
    
    for file in restart_files
        m = match(time_pattern, file)
        if m !== nothing
            time_str = replace(m.captures[1], "p" => ".")
            try
                file_time = parse(Float64, time_str)
                push!(file_times, (joinpath(restart_dir, file), file_time))
            catch
                continue
            end
        end
    end
    
    # Sort by time difference
    sort!(file_times, by = x -> abs(x[2] - target_time))
    return [ft[1] for ft in file_times]
end

function validate_output(filename::String)
    try
        nc_file = NetCDF.open(filename, NC_NOWRITE)
        
        required_vars = ["time", "step", "r"]
        for var in required_vars
            if NetCDF.varid(nc_file, var) == -1
                NetCDF.close(nc_file)
                return false
            end
        end
        
        NetCDF.close(nc_file)
        return true
    catch
        return false
    end
end

function cleanup_old_files(output_dir::String, keep_last_n::Int = 10)
    files = readdir(output_dir)
    output_files = filter(f -> endswith(f, ".nc") && contains(f, "time_") && 
                            !contains(f, "restart"), files)
    
    if isempty(output_files)
        return
    end
    
    # Extract unique times
    time_pattern = r"time_(\d+p\d+)"
    times = Set{Float64}()
    
    for file in output_files
        m = match(time_pattern, file)
        if m !== nothing
            time_str = replace(m.captures[1], "p" => ".")
            try
                push!(times, parse(Float64, time_str))
            catch
                continue
            end
        end
    end
    
    times_sorted = sort(collect(times), rev=true)
    
    if length(times_sorted) <= keep_last_n
        return
    end
    
    # Remove old files
    times_to_remove = times_sorted[(keep_last_n+1):end]
    
    for time_val in times_to_remove
        time_str = @sprintf("%.6f", time_val)
        time_str = replace(time_str, "." => "p")
        
        time_files = filter(f -> contains(f, "time_$(time_str)"), output_files)
        
        for file in time_files
            try
                rm(joinpath(output_dir, file))
                println("Removed old file: $file")
            catch e
                @warn "Failed to remove file: $file"
            end
        end
    end
    
    println("Cleanup completed. Kept last $keep_last_n time outputs.")
end

function get_time_series(output_dir::String, rank::Int = 0)
    files = readdir(output_dir)
    rank_files = filter(f -> endswith(f, ".nc") && contains(f, "rank_$(lpad(rank, 4, '0'))") && 
                        !contains(f, "restart"), files)
    
    time_pattern = r"time_(\d+p\d+)"
    times = Float64[]
    
    for file in rank_files
        m = match(time_pattern, file)
        if m !== nothing
            time_str = replace(m.captures[1], "p" => ".")
            try
                push!(times, parse(Float64, time_str))
            catch
                continue
            end
        end
    end
    
    return sort(times)
end

function find_files_in_time_range(output_dir::String, start_time::Float64, end_time::Float64, 
                                    rank::Union{Int, Nothing} = nothing)
    files = readdir(output_dir)
    output_files = filter(f -> endswith(f, ".nc") && contains(f, "time_") && 
                            !contains(f, "restart"), files)
    
    if rank !== nothing
        output_files = filter(f -> contains(f, "rank_$(lpad(rank, 4, '0'))"), output_files)
    end
    
    time_pattern = r"time_(\d+p\d+)"
    files_in_range = String[]
    
    for file in output_files
        m = match(time_pattern, file)
        if m !== nothing
            time_str = replace(m.captures[1], "p" => ".")
            try
                file_time = parse(Float64, time_str)
                if start_time <= file_time <= end_time
                    push!(files_in_range, joinpath(output_dir, file))
                end
            catch
                continue
            end
        end
    end
    
    return files_in_range
end

function get_file_info(filename::String)
    nc_file = NetCDF.open(filename, NC_NOWRITE)
    
    try
        info = Dict{String, Any}()
        info["rank"] = NetCDF.getatt(nc_file, NC_GLOBAL, "mpi_rank")
        info["time"] = NetCDF.readvar(nc_file, "time")[1]
        info["step"] = NetCDF.readvar(nc_file, "step")[1]
        
        # Get dimensions
        info["dimensions"] = Dict{String, Int}()
        for (name, dim_id) in NetCDF.dimnames(nc_file)
            info["dimensions"][name] = NetCDF.dimlen(nc_file, dim_id)
        end
        
        # Get variables
        info["variables"] = collect(keys(NetCDF.varnames(nc_file)))
        
        return info
    finally
        NetCDF.close(nc_file)
    end
end


# ================================================================================
# Enhanced Configuration and Integration Functions
# ================================================================================

"""
    create_shtns_aware_output_config(shtns_config::SHTnsKitConfig, pencils::NamedTuple; 
                                    base_config::OutputConfig = default_config())
    
Create output configuration that integrates with SHTns configuration and pencil decomposition
"""
function create_shtns_aware_output_config(shtns_config::SHTnsKitConfig, pencils::NamedTuple; 
                                         base_config::OutputConfig = default_config())
    # Create enhanced config that uses SHTns-specific optimizations
    enhanced_config = OutputConfig(
        base_config.output_space,
        base_config.naming_scheme,
        base_config.output_dir,
        base_config.filename_prefix,
        base_config.compression_level,
        base_config.include_metadata,
        base_config.include_grid,
        base_config.include_diagnostics,
        base_config.output_precision,
        shtns_config.lmax,  # Use actual lmax from config
        base_config.overwrite_files,
        base_config.independent_writes,
        base_config.output_interval,
        base_config.restart_interval,
        base_config.max_output_time,
        base_config.time_tolerance
    )
    
    return enhanced_config
end


"""
    validate_output_compatibility(field_info::FieldInfo, shtns_config::SHTnsKitConfig)
    
Validate that output field information is compatible with SHTns configuration
"""
function validate_output_compatibility(field_info::FieldInfo, shtns_config::SHTnsKitConfig)
    errors = String[]
    
    # Check spectral dimensions match
    if field_info.nlm != shtns_config.nlm
        push!(errors, "Field nlm ($(field_info.nlm)) != SHTns nlm ($(shtns_config.nlm))")
    end
    
    # Check grid dimensions
    if field_info.nlat != shtns_config.nlat
        push!(errors, "Field nlat ($(field_info.nlat)) != SHTns nlat ($(shtns_config.nlat))")
    end
    
    if field_info.nlon != shtns_config.nlon
        push!(errors, "Field nlon ($(field_info.nlon)) != SHTns nlon ($(shtns_config.nlon))")
    end
    
    # Check l,m values consistency
    if !isempty(field_info.l_values) && !isempty(shtns_config.l_values)
        if length(field_info.l_values) != length(shtns_config.l_values)
            push!(errors, "l_values length mismatch")
        elseif field_info.l_values != shtns_config.l_values
            push!(errors, "l_values content mismatch")
        end
    end
    
    if !isempty(errors)
        @warn "Output compatibility validation failed:\\n" * join(errors, "\\n")
        return false
    end
    
    return true
end


"""
    setup_shtns_metadata!(nc_file, shtns_config::SHTnsKitConfig, pencils::NamedTuple)
    
Add SHTns-specific metadata to NetCDF file
"""
function setup_shtns_metadata!(nc_file, shtns_config::SHTnsKitConfig, pencils::NamedTuple)
    # Add SHTns configuration metadata
    NetCDF.putatt(nc_file, "shtns_lmax", shtns_config.lmax)
    NetCDF.putatt(nc_file, "shtns_mmax", shtns_config.mmax)
    NetCDF.putatt(nc_file, "shtns_nlm", shtns_config.nlm)
    NetCDF.putatt(nc_file, "shtns_nlat", shtns_config.nlat)
    NetCDF.putatt(nc_file, "shtns_nlon", shtns_config.nlon)
    
    # Add pencil decomposition info
    comm = get_comm()
    rank = get_rank()
    nprocs = get_nprocs()
    
    NetCDF.putatt(nc_file, "pencil_decomposition", "true")
    NetCDF.putatt(nc_file, "pencil_rank", rank)
    NetCDF.putatt(nc_file, "pencil_nprocs", nprocs)
    
    # Add local range information
    try
        spec_range = range_local(pencils.spec, 1)
        r_range = range_local(pencils.r, 3)
        
        NetCDF.putatt(nc_file, "local_spec_range_start", first(spec_range))
        NetCDF.putatt(nc_file, "local_spec_range_end", last(spec_range))
        NetCDF.putatt(nc_file, "local_r_range_start", first(r_range))
        NetCDF.putatt(nc_file, "local_r_range_end", last(r_range))
    catch
        # Skip range info if pencils not properly initialized
    end
end


"""
    create_enhanced_field_variables!(nc_file, field_info::FieldInfo, config::OutputConfig, 
                                    available_fields::Vector{String})
                                     
Enhanced field variable setup that leverages SHTns configuration
"""
function create_enhanced_field_variables!(nc_file, field_info::FieldInfo, config::OutputConfig, 
                                        available_fields::Vector{String})
    # Call original function first
    setup_field_variables!(nc_file, field_info, config, available_fields)
    
    # Add enhanced attributes if SHTns config is available
    if field_info.config !== nothing
        config_ref = field_info.config
        
        # Add spectral field attributes with mode information
        for component in ["velocity_toroidal", "velocity_poloidal", "magnetic_toroidal", "magnetic_poloidal"]
            if component in available_fields
                real_name = "$(component)_real"
                imag_name = "$(component)_imag"
                
                if NetCDF.varid(nc_file, real_name) != -1
                    NetCDF.putatt(nc_file, real_name, "lmax", config_ref.lmax)
                    NetCDF.putatt(nc_file, real_name, "mmax", config_ref.mmax)
                    NetCDF.putatt(nc_file, real_name, "spectral_truncation", "triangular")
                    NetCDF.putatt(nc_file, real_name, "normalization", "schmidt_semi_normalized")
                end
                
                if NetCDF.varid(nc_file, imag_name) != -1
                    NetCDF.putatt(nc_file, imag_name, "lmax", config_ref.lmax)
                    NetCDF.putatt(nc_file, imag_name, "mmax", config_ref.mmax)
                    NetCDF.putatt(nc_file, imag_name, "spectral_truncation", "triangular")
                    NetCDF.putatt(nc_file, imag_name, "normalization", "schmidt_semi_normalized")
                end
            end
        end
        
        # Add grid attributes
        if NetCDF.varid(nc_file, "theta") != -1
            NetCDF.putatt(nc_file, "theta", "grid_type", "gaussian")
            NetCDF.putatt(nc_file, "theta", "quadrature_weights_available", "yes")
        end
        
        if NetCDF.varid(nc_file, "phi") != -1
            NetCDF.putatt(nc_file, "phi", "grid_type", "equispaced")
        end
    end
end


"""
    write_enhanced_coordinates!(nc_file, field_info::FieldInfo, config::OutputConfig)
    
Write coordinates with enhanced precision and metadata from SHTns config
"""
function write_enhanced_coordinates!(nc_file, field_info::FieldInfo, config::OutputConfig)
    # Write basic coordinates
    write_coordinates!(nc_file, field_info, config)
    
    # Add enhanced coordinate data if config is available
    if field_info.config !== nothing
        shtns_cfg = field_info.config

        # Write Gaussian quadrature weights if available
        # Note: This function is called after endDef, so we can only write data if the variable exists
        if NetCDF.varid(nc_file, "gauss_weights") != -1 && !isempty(shtns_cfg.gauss_weights)
            try
                NetCDF.putvar(nc_file, "gauss_weights", shtns_cfg.gauss_weights)
            catch
                # Weights variable write failed - skip
            end
        end
    end
end

# Exports are handled by main module

# ================================================================================
# Complete Usage Example (commented out due to parsing issues)
# ================================================================================

#=
"""
Complete usage example for mixed field NetCDF output:

\```julia
using MPI
using .NetCDFOutputWriter

# Initialize MPI
MPI.Init()

# Create configuration for mixed fields
config = OutputConfig(
    MIXED_FIELDS,         # Spectral for velocity/magnetic, physical for temperature
    RANK_TIME,            # File naming
    "./mixed_output",     # Output directory
    "geodynamo_mixed",    # Filename prefix
    6,                    # Compression
    true, true, true,     # Include metadata, grid, diagnostics
    Float32,              # Single precision
    -1,                   # All spectral modes
    true,                 # Overwrite files
    true,                 # Independent rank writes
    0.05,                 # Output every 0.05 time units
    0.5,                  # Restart every 0.5 time units
    10.0,                 # Max output time
    1e-12                 # Time tolerance
)

# Alternatively, derive settings from GeodynamoParameters:
# config = output_config_from_parameters()
# config = with_output_precision(config, Float64)
# config = with_independent_writes(config, true)

# Initialize time tracker
tracker = create_time_tracker(config, 0.0)

# Simulation loop
simulation_time = 0.0
dt = 0.001
step = 0

while simulation_time < 2.0
    simulation_time += dt
    step += 1
    
    # Prepare mixed field data
    fields = Dict(
        # Option 1: Temperature/Composition in Physical space
        "temperature" => rand(Float64, 32, 64, 20),
        "composition" => rand(Float64, 32, 64, 20),
        
        # Option 2: Temperature/Composition in Spectral space (more efficient!)
        # "temperature_spectral" => Dict(
        #     "real" => rand(Float64, 100, 1, 20),
        #     "imag" => rand(Float64, 100, 1, 20)
        # ),
        # "composition_spectral" => Dict(
        #     "real" => rand(Float64, 100, 1, 20),
        #     "imag" => rand(Float64, 100, 1, 20)
        # ),
        
        # Velocity: Spectral space (toroidal/poloidal)
        "velocity_toroidal" => Dict(
            "real" => rand(Float64, 100, 1, 20),
            "imag" => rand(Float64, 100, 1, 20)
        ),
        "velocity_poloidal" => Dict(
            "real" => rand(Float64, 100, 1, 20),
            "imag" => rand(Float64, 100, 1, 20)
        ),
        
        # Magnetic: Spectral space (toroidal/poloidal)
        "magnetic_toroidal" => Dict(
            "real" => rand(Float64, 100, 1, 20),
            "imag" => rand(Float64, 100, 1, 20)
        ),
        "magnetic_poloidal" => Dict(
            "real" => rand(Float64, 100, 1, 20),
            "imag" => rand(Float64, 100, 1, 20)
        )
    )
    
    # Metadata
    metadata = Dict{String, Any}(
        "current_time" => simulation_time,
        "current_step" => step,
        "current_dt" => dt,
        "Rayleigh_number" => 1e6,
        "Ekman_number" => 1e-4
    )
    
    # Time-based output
    did_output = write_fields!(fields, tracker, metadata, config)
    
    if did_output
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        if rank == 0
            println("Mixed output at time: $simulation_time")
            println("  Temperature: Physical (32×64×20)")
            println("  Composition: Physical (32×64×20)")
            println("  Velocity: Spectral toroidal/poloidal (100×20)")
            println("  Magnetic: Spectral toroidal/poloidal (100×20)")
        end
    end
    
    # Adjust timestep for exact output timing
    time_to_next = time_to_next_output(tracker, simulation_time, config)
    if time_to_next > 0 && time_to_next < dt
        dt = time_to_next
    end
end

# Analysis tools
times = get_time_series("./mixed_output", 0)
println("Available output times: ", times)

files = find_files_in_time_range("./mixed_output", 1.0, 1.5, 0)
println("Files in range 1.0-1.5: ", length(files))

# Cleanup
cleanup_old_files("./mixed_output", 5)

MPI.Finalize()
```

This provides a complete, efficient NetCDF output system with:
- Temperature in physical space for easy visualization
- Velocity and magnetic fields in compact spectral representation
- Time-based output control
- Complete restart capability
- Robust file management utilities

File structure example:
```
geodynamo_mixed_output_rank_0000_time_1p500000.nc
├── Dimensions: theta(32), phi(64), r(20), spectral_mode(100), time(1)
├── Coordinates: theta, phi, r, l_values, m_values, time, step
├── Temperature: temperature[theta,phi,r] (physical) OR
│               temperature_spectral_real/imag[spectral_mode,r] (spectral)
├── Composition: composition[theta,phi,r] (physical) OR  
│               composition_spectral_real/imag[spectral_mode,r] (spectral)
├── Velocity: velocity_toroidal_real/imag[spectral_mode,r] (spectral)
│             velocity_poloidal_real/imag[spectral_mode,r] (spectral)
└── Magnetic: magnetic_toroidal_real/imag[spectral_mode,r] (spectral)
              magnetic_poloidal_real/imag[spectral_mode,r] (spectral)
\```
"""
=#


# # Get all files for a specific time
# time_val = 1.5
# nprocs = 4
# files = []
# for rank in 0:(nprocs-1)
#     filename = "geodynamo_output_rank_$(lpad(rank, 4, '0'))_time_1p500000.nc"
#     push!(files, filename)
# end

# # Read and combine data from all processors
# global_temperature = combine_temperature_data(files)
# global_velocity_spectral = combine_spectral_data(files, "velocity_toroidal")
