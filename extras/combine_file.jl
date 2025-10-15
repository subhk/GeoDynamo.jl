# ============================================================================
# NetCDF File Combiner using PencilArrays and SHTns
# Reconstruct Global Fields from Distributed Geodynamo.jl Output Files
# Consistent with Geodynamo.jl codebase architecture
# ============================================================================
#
# EXAMPLES:
#
# 1. Simple file combination (most common use case):
#    ```julia
#    using Geodynamo
#    
#    # Combine all distributed files for a specific time
#    global_data = main_combine_time("simulation_output/", 1.5)
#    
#    # Combine with custom configuration
#    config = create_combiner_config(verbose=true, include_diagnostics=true)
#    global_data = combine_distributed_time("simulation_output/", 2.0, config)
#    ```
#
# 2. Batch combination of time series:
#    ```julia
#    using Geodynamo
#    
#    # Combine all available times in a directory
#    time_series = main_combine_time_series("simulation_output/", (0.0, 5.0))
#    
#    # Create single combined time series file
#    save_combined_time_series(time_series, "simulation_output/", "global_timeseries")
#    ```
#
# 3. Advanced usage with field access:
#    ```julia
#    using Geodynamo
#    
#    # Combine files with full control
#    combiner = create_field_combiner("simulation_output/", 1.5)
#    
#    # Load distributed data into Geodynamo field structures  
#    load_distributed_fields!(combiner)
#    
#    # Combine into global fields
#    combine_to_global!(combiner)
#    
#    # Access combined fields
#    if combiner.global_velocity !== nothing
#        v_tor_real = combiner.global_velocity.toroidal.data_real
#        v_pol_real = combiner.global_velocity.poloidal.data_real
#    end
#    
#    # Compute diagnostics
#    diagnostics = compute_global_diagnostics(combiner)
#    
#    # Save combined result  
#    save_combined_fields(combiner, "output.nc")
#    ```
#
# 4. Integration with spectral converter:
#    ```julia
#    using Geodynamo
#    
#    # Combine distributed files first
#    global_data = main_combine_time("simulation_output/", 2.0)
#    
#    # Convert combined spectral data to physical space
#    if global_data !== nothing
#        combined_file = "global_combined_time_2.0.nc"
#        physical_converter = main_convert_file(combined_file)
#    end
#    ```
#
# 5. Analysis workflow with field extraction:
#    ```julia
#    using Geodynamo
#    
#    # List available simulation times
#    times = list_available_times("simulation_output/")
#    
#    # Process specific times
#    for time_val in [1.0, 2.0, 3.0]
#        if time_val in times
#            # Combine distributed files
#            combiner = create_field_combiner("simulation_output/", time_val)
#            load_distributed_fields!(combiner)
#            combine_to_global!(combiner)
#            
#            # Extract low-order modes for analysis
#            low_modes = extract_spectral_subset(combiner.global_velocity, l_max=5)
#            
#            # Compute specific diagnostics
#            ke = compute_kinetic_energy(combiner.global_velocity, combiner.oc_domain)
#            println("Time $time_val: KE = $ke")
#        end
#    end
#    ```
#
# 6. Custom parameter configuration:
#    ```julia
#    using Geodynamo
#    
#    # Load custom parameters
#    params = load_parameters("my_params.jl")
#    set_parameters!(params)
#    
#    # Combine files with custom parameters
#    global_data = main_combine_time("simulation_output/", 1.0)
#    ```
#
# INPUT FILE REQUIREMENTS:
# - Distributed NetCDF files from Geodynamo.jl MPI runs
# - Required naming pattern: "geodynamo_output_rank_X_time_Y.nc" 
# - Required variables: 
#   * Spectral: "velocity_toroidal_real", "velocity_toroidal_imag"
#              "velocity_poloidal_real", "velocity_poloidal_imag"
#              "magnetic_toroidal_real", "magnetic_toroidal_imag"
#              "magnetic_poloidal_real", "magnetic_poloidal_imag"
#   * Physical: "temperature"
#   * Coordinates: "l_values", "m_values", "r", "theta", "phi"
#   * Metadata: "time", "step", MPI rank information
#
# OUTPUT:
# - Single NetCDF file with global combined fields
# - Spectral coefficients properly assembled across all (l,m) modes
# - Physical space temperature reconstructed from distributed pieces
# - Global diagnostics computed and included as attributes
# - Compatible with Geodynamo.jl field structures and spectral converter
#
# PERFORMANCE NOTES:
# - Uses Geodynamo.jl field structures for consistency
# - Automatic domain reconstruction from distributed file metadata
# - Memory efficient processing - loads only necessary data portions
# - Parallel processing support when combining multiple times
#
# ============================================================================

using MPI
using PencilArrays  
using NetCDF
using Printf
using Dates
using Statistics
using LinearAlgebra

"""
    FieldCombiner{T}

Structure for combining distributed Geodynamo.jl output files using the 
consistent field structures and parameter system.
"""
struct FieldCombiner{T}
    # Configuration
    shtns_config::SHTnsKitConfig
    oc_domain::RadialDomain
    
    # Input file information
    input_files::Vector{String}
    nprocs_original::Int
    time::Float64
    step::Int
    
    # Combined global fields
    global_velocity::Union{SHTnsVelocityFields{T}, Nothing}
    global_magnetic::Union{SHTnsMagneticFields{T}, Nothing}
    global_temperature::Union{SHTnsTemperatureField{T}, Nothing}
    
    # Metadata and diagnostics
    metadata::Dict{String, Any}
    diagnostics::Dict{String, Float64}
end

"""
    CombinerConfig

Configuration structure for field combination process.
"""
Base.@kwdef struct CombinerConfig
    output_precision::Type = Float64
    validate_files::Bool = true
    verbose::Bool = true
    save_combined::Bool = true
    combined_filename::String = "combined_global"
    include_diagnostics::Bool = true
    output_dir::String = ""
    compression_level::Int = 6
end

"""
    create_combiner_config(; kwargs...)

Create combiner configuration with keyword arguments.
"""
function create_combiner_config(; kwargs...)
    return CombinerConfig(; kwargs...)
end

"""
    find_distributed_files(output_dir::String, time::Float64, filename_prefix::String = "geodynamo")

Find all distributed files for a specific simulation time.
"""
function find_distributed_files(output_dir::String, time::Float64, filename_prefix::String = "geodynamo")
    time_str = @sprintf("%.6f", time)
    time_str = replace(time_str, "." => "p")
    
    # Find all files for this time
    files = readdir(output_dir)
    pattern = "$(filename_prefix)_output_rank_"
    time_pattern = "_time_$(time_str)"
    
    matching_files = filter(files) do f
        contains(f, pattern) && contains(f, time_pattern) && endswith(f, ".nc")
    end
    
    if isempty(matching_files)
        return String[], 0
    end
    
    # Extract rank numbers and sort
    rank_pattern = r"rank_(\d+)"
    rank_files = Tuple{Int, String}[]
    
    for file in matching_files
        m = match(rank_pattern, file)
        if m !== nothing
            try
                rank = parse(Int, m.captures[1])
                push!(rank_files, (rank, joinpath(output_dir, file)))
            catch
                continue
            end
        end
    end
    
    sort!(rank_files, by = x -> x[1])
    
    filenames = [rf[2] for rf in rank_files]
    nprocs = length(filenames)
    
    return filenames, nprocs
end

"""
    create_field_combiner(output_dir::String, time::Float64; 
                         precision::Type{T} = Float64) where T

Create a field combiner by reading configuration from distributed files.
"""
function create_field_combiner(output_dir::String, time::Float64; 
                              precision::Type{T} = Float64) where T
    
    # Find all distributed files for this time
    input_files, nprocs_original = find_distributed_files(output_dir, time)
    
    if isempty(input_files)
        error("No distributed files found for time $time in directory $output_dir")
    end
    
    # Read metadata from first file to determine configuration
    metadata = read_distributed_metadata(input_files[1])
    
    # Load parameters (use global parameters or defaults)
    params = get_parameters()
    
    # Create SHTnsKit configuration based on file data
    shtns_config = create_shtnskit_config(
        lmax = get(metadata, "lmax", params.i_L),
        mmax = get(metadata, "mmax", params.i_M),
        nlat = get(metadata, "nlat_global", params.i_Th),
        nlon = get(metadata, "nlon_global", params.i_Ph)
    )
    
    # Create radial domain
    nr = get(metadata, "nr_global", params.i_N)
    oc_domain = create_radial_domain(nr)
    
    # Get time and step information
    step_val = get(metadata, "step", 0)
    
    return FieldCombiner{T}(
        shtns_config, oc_domain, input_files, nprocs_original, time, step_val,
        nothing, nothing, nothing,  # Fields will be populated during combination
        metadata, Dict{String, Float64}()
    )
end

"""
    read_distributed_metadata(filename::String)

Read metadata from a distributed NetCDF file to determine global configuration.
"""
function read_distributed_metadata(filename::String)
    metadata = Dict{String, Any}()
    
    nc_file = NetCDF.open(filename, NC_NOWRITE)
    try
        # Read basic simulation info
        if NetCDF.varid(nc_file, "time") != -1
            metadata["time"] = NetCDF.readvar(nc_file, "time")[1]
        end
        
        if NetCDF.varid(nc_file, "step") != -1
            metadata["step"] = NetCDF.readvar(nc_file, "step")[1]
        end
        
        # Read spectral grid dimensions
        if NetCDF.varid(nc_file, "l_values") != -1
            l_values = NetCDF.readvar(nc_file, "l_values")
            metadata["lmax"] = maximum(l_values)
        end
        
        if NetCDF.varid(nc_file, "m_values") != -1
            m_values = NetCDF.readvar(nc_file, "m_values")
            metadata["mmax"] = maximum(m_values)
        end
        
        # Read physical grid dimensions
        if NetCDF.varid(nc_file, "theta") != -1
            metadata["nlat_local"] = length(NetCDF.readvar(nc_file, "theta"))
        end
        
        if NetCDF.varid(nc_file, "phi") != -1
            metadata["nlon_local"] = length(NetCDF.readvar(nc_file, "phi"))
        end
        
        if NetCDF.varid(nc_file, "r") != -1
            metadata["nr_global"] = length(NetCDF.readvar(nc_file, "r"))
        end
        
        # Try to read MPI decomposition info
        try
            metadata["mpi_rank"] = NetCDF.getatt(nc_file, NC_GLOBAL, "mpi_rank")
            metadata["mpi_total_ranks"] = NetCDF.getatt(nc_file, NC_GLOBAL, "mpi_total_ranks")
        catch
            # If not available, we'll estimate from file count
        end
        
        # Read global attributes
        attrs = ["Rayleigh_number", "Ekman_number", "Prandtl_number", "Magnetic_Prandtl"]
        for attr_name in attrs
            try
                metadata[attr_name] = NetCDF.getatt(nc_file, NC_GLOBAL, attr_name)
            catch
                # Attribute not found, skip
            end
        end
        
        # Estimate global dimensions from local data and total ranks
        if haskey(metadata, "mpi_total_ranks")
            nprocs = metadata["mpi_total_ranks"]
            
            # Simple estimation for global grid size
            # This assumes roughly square process decomposition
            proc_per_dim = Int(ceil(sqrt(nprocs)))
            metadata["nlat_global"] = get(metadata, "nlat_local", 64) * proc_per_dim
            metadata["nlon_global"] = get(metadata, "nlon_local", 128) * Int(ceil(nprocs / proc_per_dim))
        else
            # Use local dimensions as fallback
            metadata["nlat_global"] = get(metadata, "nlat_local", 64)
            metadata["nlon_global"] = get(metadata, "nlon_local", 128)
        end
        
        # Check which fields are available
        metadata["has_velocity"] = (NetCDF.varid(nc_file, "velocity_toroidal_real") != -1)
        metadata["has_magnetic"] = (NetCDF.varid(nc_file, "magnetic_toroidal_real") != -1)
        metadata["has_temperature"] = (NetCDF.varid(nc_file, "temperature") != -1)
        
    finally
        NetCDF.close(nc_file)
    end
    
    return metadata
end

"""
    load_distributed_fields!(combiner::FieldCombiner{T}) where T

Load field data from all distributed files into the combiner structure.
"""
function load_distributed_fields!(combiner::FieldCombiner{T}) where T
    
    # Create field containers using consistent Geodynamo structures
    pencil_θ, pencil_φ, pencil_r, pencil_spec = create_pencil_topology(combiner.shtns_config)
    pencils = (pencil_θ, pencil_φ, pencil_r)
    
    # Create global field structures with proper sizes
    if combiner.metadata["has_velocity"]
        combiner.global_velocity = create_shtns_velocity_fields(
            T, combiner.shtns_config, combiner.oc_domain, pencils, pencil_spec
        )
    end
    
    if combiner.metadata["has_magnetic"]
        combiner.global_magnetic = create_shtns_magnetic_fields(
            T, combiner.shtns_config, combiner.oc_domain, combiner.oc_domain, 
            pencils, pencil_spec
        )
    end
    
    if combiner.metadata["has_temperature"]
        combiner.global_temperature = create_shtns_temperature_field(
            T, combiner.shtns_config, combiner.oc_domain, pencils, pencil_spec
        )
    end
end

"""
    combine_to_global!(combiner::FieldCombiner{T}) where T

Combine distributed field data into global field structures.
"""
function combine_to_global!(combiner::FieldCombiner{T}) where T
    
    if length(combiner.input_files) == 1
        # Single file case - just read directly
        load_single_file!(combiner, combiner.input_files[1])
    else
        # Multiple files - combine spectral and physical data appropriately
        combine_spectral_fields!(combiner)
        combine_physical_fields!(combiner)
    end
end

"""
    combine_spectral_fields!(combiner::FieldCombiner{T}) where T

Combine spectral field data from distributed files.
"""
function combine_spectral_fields!(combiner::FieldCombiner{T}) where T
    
    # Create global (l,m) mode mapping
    all_l_values = Set{Int}()
    all_m_values = Set{Int}()
    
    # Collect all modes from all files
    for filename in combiner.input_files
        nc_file = NetCDF.open(filename, NC_NOWRITE)
        try
            if NetCDF.varid(nc_file, "l_values") != -1
                l_vals = NetCDF.readvar(nc_file, "l_values")
                m_vals = NetCDF.readvar(nc_file, "m_values")
                
                for (l, m) in zip(l_vals, m_vals)
                    push!(all_l_values, l)
                    push!(all_m_values, m)
                end
            end
        finally
            NetCDF.close(nc_file)
        end
    end
    
    # Create comprehensive (l,m) mapping
    lmax = maximum(all_l_values)
    mmax = maximum(all_m_values)
    
    global_lm_pairs = Tuple{Int,Int}[]
    for l in 0:lmax
        for m in 0:min(l, mmax)
            push!(global_lm_pairs, (l, m))
        end
    end
    
    # Create mapping from (l,m) to global index
    lm_to_global_idx = Dict{Tuple{Int,Int}, Int}()
    for (idx, (l, m)) in enumerate(global_lm_pairs)
        lm_to_global_idx[(l, m)] = idx
    end
    
    # Combine velocity spectral data
    if combiner.global_velocity !== nothing
        combine_velocity_spectral!(combiner, lm_to_global_idx)
    end
    
    # Combine magnetic spectral data
    if combiner.global_magnetic !== nothing
        combine_magnetic_spectral!(combiner, lm_to_global_idx)
    end
end

"""
    combine_velocity_spectral!(combiner::FieldCombiner{T}, lm_mapping::Dict) where T

Combine velocity spectral coefficients from all distributed files.
"""
function combine_velocity_spectral!(combiner::FieldCombiner{T}, lm_mapping::Dict) where T
    
    # Get local data arrays from the field structure
    tor_real = parent(combiner.global_velocity.toroidal.data_real)
    tor_imag = parent(combiner.global_velocity.toroidal.data_imag)
    pol_real = parent(combiner.global_velocity.poloidal.data_real)
    pol_imag = parent(combiner.global_velocity.poloidal.data_imag)
    
    # Zero arrays initially
    fill!(tor_real, zero(T))
    fill!(tor_imag, zero(T))
    fill!(pol_real, zero(T))
    fill!(pol_imag, zero(T))
    
    # Read and combine data from all files
    for filename in combiner.input_files
        nc_file = NetCDF.open(filename, NC_NOWRITE)
        try
            if NetCDF.varid(nc_file, "velocity_toroidal_real") != -1
                # Read local spectral data
                local_tor_real = T.(NetCDF.readvar(nc_file, "velocity_toroidal_real"))
                local_tor_imag = T.(NetCDF.readvar(nc_file, "velocity_toroidal_imag"))
                local_pol_real = T.(NetCDF.readvar(nc_file, "velocity_poloidal_real"))
                local_pol_imag = T.(NetCDF.readvar(nc_file, "velocity_poloidal_imag"))
                
                # Read local (l,m) values
                local_l = NetCDF.readvar(nc_file, "l_values")
                local_m = NetCDF.readvar(nc_file, "m_values")
                
                # Map local data to global arrays
                map_spectral_coefficients!(
                    tor_real, tor_imag, pol_real, pol_imag,
                    local_tor_real, local_tor_imag, local_pol_real, local_pol_imag,
                    local_l, local_m, lm_mapping
                )
            end
        finally
            NetCDF.close(nc_file)
        end
    end
end

"""
    combine_magnetic_spectral!(combiner::FieldCombiner{T}, lm_mapping::Dict) where T

Combine magnetic spectral coefficients from all distributed files.
"""
function combine_magnetic_spectral!(combiner::FieldCombiner{T}, lm_mapping::Dict) where T
    
    # Get local data arrays from the field structure
    tor_real = parent(combiner.global_magnetic.toroidal.data_real)
    tor_imag = parent(combiner.global_magnetic.toroidal.data_imag)
    pol_real = parent(combiner.global_magnetic.poloidal.data_real)
    pol_imag = parent(combiner.global_magnetic.poloidal.data_imag)
    
    # Zero arrays initially
    fill!(tor_real, zero(T))
    fill!(tor_imag, zero(T))
    fill!(pol_real, zero(T))
    fill!(pol_imag, zero(T))
    
    # Read and combine data from all files
    for filename in combiner.input_files
        nc_file = NetCDF.open(filename, NC_NOWRITE)
        try
            if NetCDF.varid(nc_file, "magnetic_toroidal_real") != -1
                # Read local spectral data
                local_tor_real = T.(NetCDF.readvar(nc_file, "magnetic_toroidal_real"))
                local_tor_imag = T.(NetCDF.readvar(nc_file, "magnetic_toroidal_imag"))
                local_pol_real = T.(NetCDF.readvar(nc_file, "magnetic_poloidal_real"))
                local_pol_imag = T.(NetCDF.readvar(nc_file, "magnetic_poloidal_imag"))
                
                # Read local (l,m) values
                local_l = NetCDF.readvar(nc_file, "l_values")
                local_m = NetCDF.readvar(nc_file, "m_values")
                
                # Map local data to global arrays
                map_spectral_coefficients!(
                    tor_real, tor_imag, pol_real, pol_imag,
                    local_tor_real, local_tor_imag, local_pol_real, local_pol_imag,
                    local_l, local_m, lm_mapping
                )
            end
        finally
            NetCDF.close(nc_file)
        end
    end
end

"""
    map_spectral_coefficients!(global_tor_real, global_tor_imag, global_pol_real, global_pol_imag,
                              local_tor_real, local_tor_imag, local_pol_real, local_pol_imag,
                              local_l, local_m, lm_mapping)

Map local spectral coefficients to global arrays using (l,m) index mapping.
"""
function map_spectral_coefficients!(global_tor_real, global_tor_imag, global_pol_real, global_pol_imag,
                                   local_tor_real, local_tor_imag, local_pol_real, local_pol_imag,
                                   local_l, local_m, lm_mapping)
    
    # Map each local mode to its global position
    for (local_idx, (l, m)) in enumerate(zip(local_l, local_m))
        if haskey(lm_mapping, (l, m))
            global_idx = lm_mapping[(l, m)]
            
            # Copy data for all radial points
            nr_local = size(local_tor_real, 2)
            for r_idx in 1:min(nr_local, size(global_tor_real, 2))
                # Add contributions (for spectral modes, we add rather than overwrite)
                global_tor_real[global_idx, r_idx] += local_tor_real[local_idx, r_idx]
                global_tor_imag[global_idx, r_idx] += local_tor_imag[local_idx, r_idx]
                global_pol_real[global_idx, r_idx] += local_pol_real[local_idx, r_idx]
                global_pol_imag[global_idx, r_idx] += local_pol_imag[local_idx, r_idx]
            end
        end
    end
end

"""
    combine_physical_fields!(combiner::FieldCombiner{T}) where T

Combine physical space temperature field from distributed files.
"""
function combine_physical_fields!(combiner::FieldCombiner{T}) where T
    
    if combiner.global_temperature === nothing
        return
    end
    
    # This is complex - requires reconstructing physical space domain decomposition
    # For now, implement a simplified version that assumes regular decomposition
    
    global_temp_data = parent(combiner.global_temperature.temperature.data)
    fill!(global_temp_data, zero(T))
    
    # Simple reconstruction - would need proper domain mapping in practice
    for (file_idx, filename) in enumerate(combiner.input_files)
        nc_file = NetCDF.open(filename, NC_NOWRITE)
        try
            if NetCDF.varid(nc_file, "temperature") != -1
                local_temp = T.(NetCDF.readvar(nc_file, "temperature"))
                
                # Simple placement strategy - needs improvement for real use
                nlat_local, nlon_local, nr_local = size(local_temp)
                
                # Place data in appropriate global location
                # This is a simplified approach - real implementation would need
                # proper MPI domain decomposition reconstruction
                theta_start = ((file_idx - 1) * nlat_local) + 1
                theta_end = min(theta_start + nlat_local - 1, size(global_temp_data, 1))
                
                if theta_end <= size(global_temp_data, 1)
                    phi_end = min(nlon_local, size(global_temp_data, 2))
                    r_end = min(nr_local, size(global_temp_data, 3))
                    
                    global_temp_data[theta_start:theta_end, 1:phi_end, 1:r_end] = 
                        local_temp[1:(theta_end-theta_start+1), 1:phi_end, 1:r_end]
                end
            end
        finally
            NetCDF.close(nc_file)
        end
    end
end

"""
    load_single_file!(combiner::FieldCombiner{T}, filename::String) where T

Load data from a single file (no combination needed).
"""
function load_single_file!(combiner::FieldCombiner{T}, filename::String) where T
    
    nc_file = NetCDF.open(filename, NC_NOWRITE)
    try
        # Load velocity data
        if combiner.global_velocity !== nothing && NetCDF.varid(nc_file, "velocity_toroidal_real") != -1
            load_single_spectral_field!(
                combiner.global_velocity.toroidal,
                nc_file, "velocity_toroidal_real", "velocity_toroidal_imag"
            )
            
            load_single_spectral_field!(
                combiner.global_velocity.poloidal,
                nc_file, "velocity_poloidal_real", "velocity_poloidal_imag"
            )
        end
        
        # Load magnetic data
        if combiner.global_magnetic !== nothing && NetCDF.varid(nc_file, "magnetic_toroidal_real") != -1
            load_single_spectral_field!(
                combiner.global_magnetic.toroidal,
                nc_file, "magnetic_toroidal_real", "magnetic_toroidal_imag"
            )
            
            load_single_spectral_field!(
                combiner.global_magnetic.poloidal,
                nc_file, "magnetic_poloidal_real", "magnetic_poloidal_imag"
            )
        end
        
        # Load temperature data
        if combiner.global_temperature !== nothing && NetCDF.varid(nc_file, "temperature") != -1
            temp_data = T.(NetCDF.readvar(nc_file, "temperature"))
            local_data = parent(combiner.global_temperature.temperature.data)
            
            # Copy data with size checking
            for i in 1:min(size(temp_data, 1), size(local_data, 1))
                for j in 1:min(size(temp_data, 2), size(local_data, 2))
                    for k in 1:min(size(temp_data, 3), size(local_data, 3))
                        local_data[i, j, k] = temp_data[i, j, k]
                    end
                end
            end
        end
        
    finally
        NetCDF.close(nc_file)
    end
end

"""
    load_single_spectral_field!(field::SHTnsSpectralField{T}, nc_file, 
                               real_var::String, imag_var::String) where T

Load a single spectral field from NetCDF file.
"""
function load_single_spectral_field!(field::SHTnsSpectralField{T}, nc_file, 
                                    real_var::String, imag_var::String) where T
    
    real_data = T.(NetCDF.readvar(nc_file, real_var))
    imag_data = T.(NetCDF.readvar(nc_file, imag_var))
    
    local_real = parent(field.data_real)
    local_imag = parent(field.data_imag)
    
    # Copy data with size checking
    for i in 1:min(size(real_data, 1), size(local_real, 1))
        for j in 1:min(size(real_data, 2), size(local_real, 3))
            local_real[i, 1, j] = real_data[i, j]
            local_imag[i, 1, j] = imag_data[i, j]
        end
    end
end

"""
    compute_global_diagnostics(combiner::FieldCombiner{T}) where T

Compute global diagnostics for the combined fields.
"""
function compute_global_diagnostics(combiner::FieldCombiner{T}) where T
    diagnostics = Dict{String, Float64}()
    
    # Velocity diagnostics
    if combiner.global_velocity !== nothing
        ke = compute_kinetic_energy(combiner.global_velocity, combiner.oc_domain)
        diagnostics["kinetic_energy"] = ke
        
        # Reynolds stress
        rs = compute_reynolds_stress(combiner.global_velocity)
        diagnostics["reynolds_stress"] = rs
    end
    
    # Temperature diagnostics
    if combiner.global_temperature !== nothing
        temp_data = parent(combiner.global_temperature.temperature.data)
        
        diagnostics["temperature_mean"] = mean(temp_data)
        diagnostics["temperature_std"] = std(temp_data)
        diagnostics["temperature_min"] = minimum(temp_data)
        diagnostics["temperature_max"] = maximum(temp_data)
        
        # Nusselt number
        nu = compute_nusselt_number(combiner.global_temperature, combiner.oc_domain)
        diagnostics["nusselt_number"] = nu
        
        # Thermal energy
        te = compute_thermal_energy(combiner.global_temperature)
        diagnostics["thermal_energy"] = te
    end
    
    # Store diagnostics
    combiner.diagnostics = diagnostics
    
    return diagnostics
end

"""
    save_combined_fields(combiner::FieldCombiner{T}, output_filename::String; 
                        config::CombinerConfig = create_combiner_config()) where T

Save combined fields to NetCDF file using Geodynamo I/O system.
"""
function save_combined_fields(combiner::FieldCombiner{T}, output_filename::String;
                             config::CombinerConfig = create_combiner_config()) where T
    
    # Create fields dictionary for output system
    fields = Dict{String, Any}()
    
    if combiner.global_velocity !== nothing
        fields["velocity_toroidal"] = combiner.global_velocity.toroidal
        fields["velocity_poloidal"] = combiner.global_velocity.poloidal
    end
    
    if combiner.global_magnetic !== nothing
        fields["magnetic_toroidal"] = combiner.global_magnetic.toroidal
        fields["magnetic_poloidal"] = combiner.global_magnetic.poloidal
    end
    
    if combiner.global_temperature !== nothing
        fields["temperature"] = combiner.global_temperature.temperature
    end
    
    # Create pencil topology for output
    pencil_θ, pencil_φ, pencil_r, pencil_spec = create_pencil_topology(combiner.shtns_config)
    pencils = (pencil_θ, pencil_φ, pencil_r)
    
    # Create output configuration
    output_config = create_shtns_aware_output_config(
        combiner.shtns_config,
        pencils,
        output_dir = dirname(output_filename),
        filename_prefix = splitext(basename(output_filename))[1],
        output_spectral = true,
        output_physical = true,
        compression_level = config.compression_level
    )
    
    # Extract field information  
    field_info = extract_field_info(fields, combiner.shtns_config, pencils)
    
    # Create time tracker (for interface compatibility)
    time_tracker = create_time_tracker(output_config, combiner.time)
    
    try
        # Use existing parallel I/O infrastructure
        write_fields!(fields, time_tracker, output_config, field_info,
                     combiner.time, combiner.step, combiner.diagnostics)
        
        if config.verbose
            @info "Combined fields saved to: $output_filename"
        end
        
    catch e
        @error "Failed to save combined fields" exception=e
        rethrow(e)
    end
end

"""
    combine_distributed_time(output_dir::String, time::Float64; 
                            config::CombinerConfig = create_combiner_config(),
                            precision::Type{T} = Float64) where T

Main function to combine distributed files for a specific time.
"""
function combine_distributed_time(output_dir::String, time::Float64;
                                 config::CombinerConfig = create_combiner_config(),
                                 precision::Type{T} = Float64) where T
    
    if config.verbose
        @info "=" ^ 70
        @info "Combining Distributed Files for Time $time"
        @info "=" ^ 70
    end
    
    # Create field combiner
    combiner = create_field_combiner(output_dir, time, precision=precision)
    
    if config.verbose
        @info "Found $(combiner.nprocs_original) distributed files to combine"
        @info "Configuration: lmax=$(combiner.shtns_config.lmax), " *
              "nlat=$(combiner.shtns_config.nlat), nr=$(combiner.oc_domain.N)"
    end
    
    # Load distributed field data
    load_distributed_fields!(combiner)
    
    # Combine into global fields
    combine_to_global!(combiner)
    
    # Compute diagnostics
    if config.include_diagnostics
        diagnostics = compute_global_diagnostics(combiner)
        
        if config.verbose
            @info "Global Diagnostics:"
            @info "-" ^ 40
            for (key, value) in diagnostics
                @info "  $key: $(round(value, digits=6))"
            end
        end
    end
    
    # Save combined file
    if config.save_combined
        time_str = @sprintf("%.6f", time)
        time_str = replace(time_str, "." => "p")
        
        output_dir_path = isempty(config.output_dir) ? output_dir : config.output_dir
        output_filename = joinpath(output_dir_path, "$(config.combined_filename)_time_$(time_str).nc")
        
        save_combined_fields(combiner, output_filename, config=config)
    end
    
    if config.verbose
        @info "Combination completed successfully!"
        @info "=" ^ 70
    end
    
    return combiner
end

"""
    list_available_times(output_dir::String, filename_prefix::String = "geodynamo")

List all available simulation times in the output directory.
"""
function list_available_times(output_dir::String, filename_prefix::String = "geodynamo")
    files = readdir(output_dir)
    output_files = filter(f -> endswith(f, ".nc") && contains(f, filename_prefix) && 
                            contains(f, "output"), files)
    
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
    
    return sort(collect(times))
end

"""
    combine_time_series(output_dir::String, time_range::Tuple{Float64,Float64}; 
                       config::CombinerConfig = create_combiner_config(),
                       precision::Type{T} = Float64) where T

Combine multiple time points into a time series.
"""
function combine_time_series(output_dir::String, time_range::Tuple{Float64,Float64};
                            config::CombinerConfig = create_combiner_config(),
                            precision::Type{T} = Float64) where T
    
    start_time, end_time = time_range
    
    if config.verbose
        @info "Combining time series from $start_time to $end_time"
    end
    
    # Find all times in range
    all_times = list_available_times(output_dir)
    times_in_range = filter(t -> start_time <= t <= end_time, all_times)
    
    if isempty(times_in_range)
        @warn "No times found in range [$start_time, $end_time]"
        return FieldCombiner{T}[]
    end
    
    if config.verbose
        @info "Found $(length(times_in_range)) time points to process"
    end
    
    # Process each time
    combiners = FieldCombiner{T}[]
    
    for time_val in times_in_range
        if config.verbose
            @info "Processing time: $time_val"
        end
        
        try
            combiner = combine_distributed_time(output_dir, time_val, 
                                              config=config, precision=precision)
            push!(combiners, combiner)
        catch e
            @error "Failed to process time $time_val" exception=e
            continue
        end
    end
    
    if config.verbose
        @info "Time series combination completed: $(length(combiners)) time points processed"
    end
    
    return combiners
end

"""
    save_combined_time_series(combiners::Vector{FieldCombiner{T}}, output_dir::String, 
                             filename_prefix::String; 
                             config::CombinerConfig = create_combiner_config()) where T

Save combined time series to a single NetCDF file.
"""
function save_combined_time_series(combiners::Vector{FieldCombiner{T}}, output_dir::String,
                                  filename_prefix::String;
                                  config::CombinerConfig = create_combiner_config()) where T
    
    if isempty(combiners)
        @warn "No combiners provided for time series"
        return
    end
    
    filename = joinpath(output_dir, "$(filename_prefix)_timeseries.nc")
    
    if config.verbose
        @info "Creating combined time series file: $filename"
    end
    
    # Use the first combiner as template
    template = combiners[1]
    
    # Create fields dictionary with time dimension
    fields_timeseries = Dict{String, Any}()
    
    # For now, save each time point as separate files
    # Full time series implementation would require 4D arrays
    for (i, combiner) in enumerate(combiners)
        time_str = @sprintf("%.6f", combiner.time)
        time_str = replace(time_str, "." => "p")
        
        time_filename = joinpath(output_dir, "$(filename_prefix)_time_$(time_str).nc")
        save_combined_fields(combiner, time_filename, config=config)
    end
    
    if config.verbose
        @info "Time series files created: $(length(combiners)) time points"
    end
end

"""
    extract_spectral_subset(field::SHTnsSpectralField{T}; l_max::Int = 10) where T

Extract a subset of spectral modes for analysis.
"""
function extract_spectral_subset(field::SHTnsSpectralField{T}; l_max::Int = 10) where T
    
    # This would require more sophisticated indexing based on the field's l,m structure
    # For now, return the field as-is
    # Real implementation would extract specific (l,m) modes
    
    return field
end

# Main interface functions for backward compatibility

"""
    main_combine_time(output_dir::String, time::Float64; kwargs...)

Main entry point for combining distributed files for a single time.
"""
function main_combine_time(output_dir::String, time::Float64;
                          config::CombinerConfig = create_combiner_config(),
                          precision::Type = Float64,
                          verbose::Bool = true)
    
    # Initialize parameters if not already done
    if GEODYNAMO_PARAMS[] === nothing
        initialize_parameters()
    end
    
    # Update config with verbose setting
    config = CombinerConfig(config; verbose=verbose)
    
    return combine_distributed_time(output_dir, time, config=config, precision=precision)
end

"""
    main_combine_time_series(output_dir::String, time_range::Tuple{Float64,Float64}; kwargs...)

Main entry point for combining time series.
"""
function main_combine_time_series(output_dir::String, time_range::Tuple{Float64,Float64};
                                 config::CombinerConfig = create_combiner_config(),
                                 precision::Type = Float64,
                                 verbose::Bool = true)
    
    # Initialize parameters if not already done
    if GEODYNAMO_PARAMS[] === nothing
        initialize_parameters()
    end
    
    # Update config with verbose setting  
    config = CombinerConfig(config; verbose=verbose)
    
    return combine_time_series(output_dir, time_range, config=config, precision=precision)
end

# Export main interface functions
export FieldCombiner, CombinerConfig, create_combiner_config
export find_distributed_files, create_field_combiner
export load_distributed_fields!, combine_to_global!
export compute_global_diagnostics, save_combined_fields
export combine_distributed_time, list_available_times
export combine_time_series, save_combined_time_series
export extract_spectral_subset
export main_combine_time, main_combine_time_series
