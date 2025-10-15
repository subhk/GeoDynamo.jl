# ============================================================================
# Spectral to Physical Field Converter using PencilArrays and SHTns
# Consistent with Geodynamo.jl codebase architecture
# ============================================================================
#
# EXAMPLES:
#
# 1. Simple file conversion (most common use case):
#    ```julia
#    using Geodynamo
#    using MPI
#    
#    MPI.Init()
#    
#    # Convert a single spectral data file to physical space
#    converter = main_convert_file("simulation_step_001000.nc")
#    
#    # Convert with custom output filename
#    converter = main_convert_file("input.nc", output_filename="my_physical_output.nc")
#    
#    MPI.Finalize()
#    ```
#
# 2. Batch conversion of multiple files:
#    ```julia
#    using Geodynamo
#    using MPI
#    
#    MPI.Init()
#    
#    # Convert all files matching pattern in directory
#    main_batch_convert("simulation_output/",
#                       output_dir="physical_fields/",
#                       pattern="combined_global_time_")
#    
#    MPI.Finalize()
#    ```
#
# 3. Advanced usage with full control:
#    ```julia
#    using Geodynamo
#    using MPI
#    
#    MPI.Init()
#    
#    # Create converter from NetCDF file metadata
#    converter = create_spectral_converter("data.nc", precision=Float64)
#    
#    # Load spectral data into Geodynamo field structures
#    load_spectral_data!(converter, "data.nc")
#    
#    # Convert toroidal/poloidal spectral data to (r,θ,φ) physical components
#    convert_to_physical!(converter)
#    
#    # Compute global diagnostics (kinetic energy, RMS values, etc.)
#    diagnostics = compute_global_diagnostics(converter)
#    
#    # Access converted physical fields
#    if converter.velocity_fields !== nothing
#        v_r = converter.velocity_fields.velocity.r_component
#        v_theta = converter.velocity_fields.velocity.θ_component
#        v_phi = converter.velocity_fields.velocity.φ_component
#    end
#    
#    # Save to NetCDF with parallel I/O
#    save_physical_fields(converter, "output_physical.nc")
#    
#    MPI.Finalize()
#    ```
#
# 4. MPI parallel execution example:
#    ```bash
#    # Run with 4 MPI processes
#    mpirun -n 4 julia --project=. -e '
#        using Geodynamo
#        using MPI
#        MPI.Init()
#        main_batch_convert("data/", output_dir="physical/")
#        MPI.Finalize()
#    '
#    ```
#
# 5. Custom parameter configuration:
#    ```julia
#    using Geodynamo
#    
#    # Load custom parameters before conversion
#    params = load_parameters("my_simulation_params.jl")
#    set_parameters!(params)
#    
#    # Now conversions will use your custom parameters
#    converter = main_convert_file("data.nc")
#    ```
#
# INPUT FILE REQUIREMENTS:
# - NetCDF file with spectral coefficients in toroidal/poloidal form
# - Required variables: "velocity_toroidal_real_global", "velocity_toroidal_imag_global"
#                      "velocity_poloidal_real_global", "velocity_poloidal_imag_global"
#                      "magnetic_toroidal_real_global", "magnetic_toroidal_imag_global"
#                      "magnetic_poloidal_real_global", "magnetic_poloidal_imag_global"
#                      "temperature_global" (physical space)
# - Coordinate arrays: "l_values_global", "m_values_global", "r_spectral"
#                      "theta", "phi", "r_physical"
# - Time information: "time", "step"
#
# OUTPUT:
# - NetCDF file with physical space vector components (r, θ, φ)
# - Parallel I/O optimized for MPI
# - Global diagnostics included as attributes
# - Compatible with Geodynamo.jl field structures
#
# PERFORMANCE NOTES:
# - Uses optimized PencilArrays for domain decomposition
# - SHTns transforms are parallelized automatically
# - Memory usage scales with local domain size
# - I/O uses parallel NetCDF when available
#
# ============================================================================

using Geodynamo
using MPI
using PencilArrays
using NetCDF
using Printf
using Dates
using Statistics
using LinearAlgebra

"""
    SpectralToPhysicalConverter{T}

Structure for converting spectral field data to physical space using the
Geodynamo.jl infrastructure (PencilArrays + SHTns).
"""
struct SpectralToPhysicalConverter{T}
    # Configuration
    shtns_config::SHTnsKitConfig
    oc_domain::RadialDomain
    
    # Pencil decomposition  
    pencils::Tuple{Pencil{3}, Pencil{3}, Pencil{3}}  # θ, φ, r pencils
    pencil_spec::Pencil{3}                            # spectral pencil
    
    # Field containers
    velocity_fields::Union{SHTnsVelocityFields{T}, Nothing}
    magnetic_fields::Union{SHTnsMagneticFields{T}, Nothing}
    temperature_field::Union{SHTnsTemperatureField{T}, Nothing}
    
    # Metadata
    time::Float64
    step::Int
    metadata::Dict{String, Any}
end

"""
    create_spectral_converter(filename::String; precision::Type{T} = Float64) where T

Create a spectral to physical converter by reading configuration from a NetCDF file.
"""
function create_spectral_converter(filename::String; precision::Type{T} = Float64) where T
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    
    if rank == 0
        @info "Creating spectral converter from file: $filename"
    end
    
    # Read file metadata to determine configuration
    metadata = read_file_metadata(filename)
    
    # Initialize MPI if not already done
    if !MPI.Initialized()
        MPI.Init()
    end
    
    # Load parameters (use global parameters or defaults)
    params = get_parameters()
    
    # Create SHTnsKit configuration based on file data
    shtns_config = create_shtnskit_config(
        lmax = get(metadata, "lmax", params.i_L),
        mmax = get(metadata, "mmax", params.i_M),
        nlat = get(metadata, "nlat_global", params.i_Th),
        nlon = get(metadata, "nlon_global", params.i_Ph)
    )
    
    # Create pencil decomposition optimized for the grid
    pencils_nt = create_pencil_topology(shtns_config)
    pencil_θ = pencils_nt.θ
    pencil_φ = pencils_nt.φ
    pencil_r = pencils_nt.r
    pencil_spec = pencils_nt.spec
    pencils = (pencil_θ, pencil_φ, pencil_r)
    
    # Create radial domain
    nr = get(metadata, "nr_global", params.i_N)
    oc_domain = create_radial_domain(nr)
    
    # Initialize field containers (will be populated during conversion)
    velocity_fields = nothing
    magnetic_fields = nothing  
    temperature_field = nothing
    
    time = get(metadata, "time", 0.0)
    step = get(metadata, "step", 0)
    
    if rank == 0
        @info "Converter configuration:" *
              " lmax=$(shtns_config.lmax), mmax=$(shtns_config.mmax)" *
              " nlat=$(shtns_config.nlat), nlon=$(shtns_config.nlon)" *
              " nr=$nr"
    end
    
    return SpectralToPhysicalConverter{T}(
        shtns_config, oc_domain,
        pencils, pencil_spec,
        velocity_fields, magnetic_fields, temperature_field,
        time, step, metadata
    )
end

"""
    read_file_metadata(filename::String)

Read metadata from NetCDF file to determine grid configuration.
"""
function read_file_metadata(filename::String)
    metadata = Dict{String, Any}()
    
    nc_file = NetCDF.open(filename, NC_NOWRITE)
    try
        # Read time information
        if NetCDF.varid(nc_file, "time") != -1
            metadata["time"] = NetCDF.readvar(nc_file, "time")[1]
        end
        
        if NetCDF.varid(nc_file, "step") != -1
            metadata["step"] = NetCDF.readvar(nc_file, "step")[1]
        end
        
        # Read grid dimensions
        if NetCDF.varid(nc_file, "l_values_global") != -1
            l_values = NetCDF.readvar(nc_file, "l_values_global")
            metadata["lmax"] = maximum(l_values)
            metadata["nlm"] = length(l_values)
        end
        
        if NetCDF.varid(nc_file, "m_values_global") != -1
            m_values = NetCDF.readvar(nc_file, "m_values_global")
            metadata["mmax"] = maximum(m_values)
        end
        
        # Read physical grid dimensions from coordinate arrays
        if NetCDF.varid(nc_file, "theta") != -1
            metadata["nlat_global"] = length(NetCDF.readvar(nc_file, "theta"))
        end
        
        if NetCDF.varid(nc_file, "phi") != -1
            metadata["nlon_global"] = length(NetCDF.readvar(nc_file, "phi"))
        end
        
        if NetCDF.varid(nc_file, "r_physical") != -1
            metadata["nr_global"] = length(NetCDF.readvar(nc_file, "r_physical"))
        elseif NetCDF.varid(nc_file, "r_spectral") != -1
            metadata["nr_global"] = length(NetCDF.readvar(nc_file, "r_spectral"))
        end
        
        # Read global attributes
        attrs = ["Rayleigh_number", "Ekman_number", "Prandtl_number", 
                "Magnetic_Prandtl", "original_nprocs"]
        for attr_name in attrs
            try
                metadata[attr_name] = NetCDF.getatt(nc_file, NC_GLOBAL, attr_name)
            catch
                # Attribute not found, skip
            end
        end
        
        # Check which fields are available
        metadata["has_velocity"] = (NetCDF.varid(nc_file, "velocity_toroidal_real_global") != -1)
        metadata["has_magnetic"] = (NetCDF.varid(nc_file, "magnetic_toroidal_real_global") != -1)
        metadata["has_temperature"] = (NetCDF.varid(nc_file, "temperature_global") != -1)
        
    finally
        NetCDF.close(nc_file)
    end
    
    return metadata
end

"""
    load_spectral_data!(converter::SpectralToPhysicalConverter{T}, filename::String) where T

Load spectral field data from NetCDF file into Geodynamo field structures.
"""
function load_spectral_data!(converter::SpectralToPhysicalConverter{T}, filename::String) where T
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    
    if rank == 0
        @info "Loading spectral data from: $filename"
    end
    
    nc_file = NetCDF.open(filename, NC_NOWRITE)
    
    try
        # Read spectral mode information
        l_values = NetCDF.readvar(nc_file, "l_values_global")
        m_values = NetCDF.readvar(nc_file, "m_values_global")
        r_spectral = NetCDF.readvar(nc_file, "r_spectral")
        
        # Load velocity field
        if converter.metadata["has_velocity"]
            if rank == 0
                @info "Loading velocity spectral data..."
            end
            
            # Create velocity field structure
            converter.velocity_fields = create_shtns_velocity_fields(
                T, converter.shtns_config, converter.oc_domain,
                converter.pencils, converter.pencil_spec
            )
            
            # Load spectral coefficients
            load_spectral_coefficients!(
                converter.velocity_fields.toroidal,
                nc_file, "velocity_toroidal_real_global", "velocity_toroidal_imag_global",
                l_values, m_values, r_spectral
            )
            
            load_spectral_coefficients!(
                converter.velocity_fields.poloidal,
                nc_file, "velocity_poloidal_real_global", "velocity_poloidal_imag_global",
                l_values, m_values, r_spectral
            )
        end
        
        # Load magnetic field
        if converter.metadata["has_magnetic"]
            if rank == 0
                @info "Loading magnetic spectral data..."
            end
            
            # Create magnetic field structure
            converter.magnetic_fields = create_shtns_magnetic_fields(
                T, converter.shtns_config, converter.oc_domain, converter.oc_domain,
                converter.pencils, converter.pencil_spec
            )
            
            # Load spectral coefficients
            load_spectral_coefficients!(
                converter.magnetic_fields.toroidal,
                nc_file, "magnetic_toroidal_real_global", "magnetic_toroidal_imag_global",
                l_values, m_values, r_spectral
            )
            
            load_spectral_coefficients!(
                converter.magnetic_fields.poloidal,
                nc_file, "magnetic_poloidal_real_global", "magnetic_poloidal_imag_global",
                l_values, m_values, r_spectral
            )
        end
        
        # Load temperature field
        if converter.metadata["has_temperature"]
            if rank == 0
                @info "Loading temperature data..."
            end
            
            # Create temperature field structure
            converter.temperature_field = create_shtns_temperature_field(
                T, converter.shtns_config, converter.oc_domain
            )
            
            # Load physical space temperature data
            load_physical_temperature!(
                converter.temperature_field.temperature,
                nc_file, "temperature_global"
            )
        end
        
    finally
        NetCDF.close(nc_file)
    end
    
    if rank == 0
        @info "Spectral data loading completed"
    end
end

"""
    load_spectral_coefficients!(field::SHTnsSpectralField{T}, nc_file, 
                                real_var_name::String, imag_var_name::String,
                                l_values::Vector{Int}, m_values::Vector{Int}, 
                                r_spectral::Vector{Float64}) where T

Load spectral coefficients from NetCDF file into a spectral field structure.
"""
function load_spectral_coefficients!(field::SHTnsSpectralField{T}, nc_file,
                                     real_var_name::String, imag_var_name::String,
                                     l_values::Vector{Int}, m_values::Vector{Int},
                                     r_spectral::Vector{Float64}) where T
    
    # Read full spectral coefficient arrays
    real_coeffs = T.(NetCDF.readvar(nc_file, real_var_name))  # (nlm, nr)
    imag_coeffs = T.(NetCDF.readvar(nc_file, imag_var_name))  # (nlm, nr)
    
    # Get local ranges for this process
    lm_range = range_local(field.pencil, 1)  # local spectral mode range
    r_range = range_local(field.pencil, 3)   # local radial range
    
    # Get local data arrays
    local_real = parent(field.data_real)
    local_imag = parent(field.data_imag)
    
    # Copy relevant portions to local arrays
    for (local_lm, global_lm) in enumerate(lm_range)
        if global_lm <= size(real_coeffs, 1)  # Check bounds
            for (local_r, global_r) in enumerate(r_range)
                if global_r <= size(real_coeffs, 2) && local_r <= size(local_real, 3)
                    local_real[local_lm, 1, local_r] = real_coeffs[global_lm, global_r]
                    local_imag[local_lm, 1, local_r] = imag_coeffs[global_lm, global_r]
                end
            end
        end
    end
end

"""
    load_physical_temperature!(field::SHTnsPhysicalField{T}, nc_file, var_name::String) where T

Load physical space temperature data from NetCDF file.
"""
function load_physical_temperature!(field::SHTnsPhysicalField{T}, nc_file, var_name::String) where T
    
    # Read the full temperature field
    temp_data = T.(NetCDF.readvar(nc_file, var_name))  # (nlat, nlon, nr)
    
    # Get local ranges for this process
    theta_range = range_local(field.pencil, 1)  # local theta range
    phi_range = range_local(field.pencil, 2)    # local phi range  
    r_range = range_local(field.pencil, 3)      # local radial range
    
    # Get local data array
    local_data = parent(field.data)
    
    # Copy relevant portions to local array
    for (local_theta, global_theta) in enumerate(theta_range)
        if global_theta <= size(temp_data, 1)
            for (local_phi, global_phi) in enumerate(phi_range)
                if global_phi <= size(temp_data, 2)
                    for (local_r, global_r) in enumerate(r_range)
                        if global_r <= size(temp_data, 3) && 
                           local_r <= size(local_data, 3)
                            local_data[local_theta, local_phi, local_r] = 
                                temp_data[global_theta, global_phi, global_r]
                        end
                    end
                end
            end
        end
    end
end

"""
    convert_to_physical!(converter::SpectralToPhysicalConverter{T}) where T

Convert all loaded spectral data to physical space using SHTns transforms.
"""
function convert_to_physical!(converter::SpectralToPhysicalConverter{T}) where T
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    
    if rank == 0
        @info "Converting spectral data to physical space..."
    end
    
    # Convert velocity field
    if converter.velocity_fields !== nothing
        if rank == 0
            @info "Converting velocity field..."
        end
        
        # Use SHTnsKit vector synthesis to convert toroidal/poloidal to (v_r, v_θ, v_φ)
        shtnskit_vector_synthesis!(
            converter.velocity_fields.toroidal,
            converter.velocity_fields.poloidal, 
            converter.velocity_fields.velocity
        )
        
        if rank == 0
            @info "Velocity conversion completed"
        end
    end
    
    # Convert magnetic field
    if converter.magnetic_fields !== nothing
        if rank == 0
            @info "Converting magnetic field..."
        end
        
        # Use SHTnsKit vector synthesis to convert toroidal/poloidal to (B_r, B_θ, B_φ)
        shtnskit_vector_synthesis!(
            converter.magnetic_fields.toroidal,
            converter.magnetic_fields.poloidal, 
            converter.magnetic_fields.magnetic
        )
        
        if rank == 0
            @info "Magnetic conversion completed"
        end
    end
    
    # Temperature is already in physical space - ensure proper transforms if needed
    if converter.temperature_field !== nothing
        if rank == 0
            @info "Temperature data already in physical space"
        end
    end
    
    if rank == 0
        @info "Physical space conversion completed"
    end
end

"""
    compute_global_diagnostics(converter::SpectralToPhysicalConverter{T}) where T

Compute global field diagnostics using MPI reductions.
"""
function compute_global_diagnostics(converter::SpectralToPhysicalConverter{T}) where T
    diagnostics = Dict{String, Float64}()
    comm = MPI.COMM_WORLD
    
    # Velocity diagnostics
    if converter.velocity_fields !== nothing
        ke = compute_kinetic_energy(converter.velocity_fields, converter.oc_domain)
        diagnostics["kinetic_energy"] = MPI.Allreduce(ke, MPI.SUM, comm)
        
        # Get local velocity magnitude statistics
        v_r_data = parent(converter.velocity_fields.velocity.r_component.data)
        v_theta_data = parent(converter.velocity_fields.velocity.θ_component.data)
        v_phi_data = parent(converter.velocity_fields.velocity.φ_component.data)
        
        # Compute local RMS values
        local_volume = length(v_r_data)
        v_r_rms_local = sqrt(sum(v_r_data.^2) / local_volume)
        v_theta_rms_local = sqrt(sum(v_theta_data.^2) / local_volume)  
        v_phi_rms_local = sqrt(sum(v_phi_data.^2) / local_volume)
        
        # Global volume-weighted averages
        total_volume = MPI.Allreduce(Float64(local_volume), MPI.SUM, comm)
        diagnostics["velocity_r_rms"] = MPI.Allreduce(v_r_rms_local * local_volume, MPI.SUM, comm) / total_volume
        diagnostics["velocity_theta_rms"] = MPI.Allreduce(v_theta_rms_local * local_volume, MPI.SUM, comm) / total_volume
        diagnostics["velocity_phi_rms"] = MPI.Allreduce(v_phi_rms_local * local_volume, MPI.SUM, comm) / total_volume
        
        # Maximum velocity magnitude
        vel_mag_local = sqrt.(v_r_data.^2 .+ v_theta_data.^2 .+ v_phi_data.^2)
        max_vel_local = length(vel_mag_local) > 0 ? maximum(vel_mag_local) : 0.0
        diagnostics["velocity_max"] = MPI.Allreduce(max_vel_local, MPI.MAX, comm)
    end
    
    # Magnetic diagnostics  
    if converter.magnetic_fields !== nothing
        # Similar calculations for magnetic field
        B_r_data = parent(converter.magnetic_fields.magnetic.r_component.data)
        B_theta_data = parent(converter.magnetic_fields.magnetic.θ_component.data)
        B_phi_data = parent(converter.magnetic_fields.magnetic.φ_component.data)
        
        local_volume = length(B_r_data)
        magnetic_energy_local = 0.5 * sum(B_r_data.^2 .+ B_theta_data.^2 .+ B_phi_data.^2) / local_volume
        total_volume = MPI.Allreduce(Float64(local_volume), MPI.SUM, comm)
        diagnostics["magnetic_energy"] = MPI.Allreduce(magnetic_energy_local * local_volume, MPI.SUM, comm) / total_volume
        
        mag_mag_local = sqrt.(B_r_data.^2 .+ B_theta_data.^2 .+ B_phi_data.^2)
        max_mag_local = length(mag_mag_local) > 0 ? maximum(mag_mag_local) : 0.0
        diagnostics["magnetic_max"] = MPI.Allreduce(max_mag_local, MPI.MAX, comm)
    end
    
    # Temperature diagnostics
    if converter.temperature_field !== nothing
        temp_data = parent(converter.temperature_field.temperature.data)
        
        local_volume = length(temp_data)
        temp_sum_local = sum(temp_data)
        temp_min_local = length(temp_data) > 0 ? minimum(temp_data) : Inf
        temp_max_local = length(temp_data) > 0 ? maximum(temp_data) : -Inf
        
        total_volume = MPI.Allreduce(Float64(local_volume), MPI.SUM, comm)
        diagnostics["temperature_mean"] = MPI.Allreduce(temp_sum_local, MPI.SUM, comm) / total_volume
        diagnostics["temperature_min"] = MPI.Allreduce(temp_min_local, MPI.MIN, comm)
        diagnostics["temperature_max"] = MPI.Allreduce(temp_max_local, MPI.MAX, comm)
        
        # Nusselt number calculation
        nu = compute_nusselt_number(converter.temperature_field, converter.oc_domain)
        diagnostics["nusselt_number"] = MPI.Allreduce(nu, MPI.SUM, comm)
    end
    
    return diagnostics
end

"""
    save_physical_fields(converter::SpectralToPhysicalConverter{T}, output_filename::String) where T

Save converted physical space fields to NetCDF file using parallel I/O.
"""
function save_physical_fields(converter::SpectralToPhysicalConverter{T}, output_filename::String) where T
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    comm = MPI.COMM_WORLD
    
    if rank == 0
        @info "Saving physical fields to: $output_filename"
        
        # Remove existing file
        if isfile(output_filename)
            rm(output_filename)
        end
    end
    
    MPI.Barrier(comm)
    
    # Create output configuration
    fields = Dict{String, Any}()
    
    # Add velocity field data
    if converter.velocity_fields !== nothing
        fields["velocity_r"] = converter.velocity_fields.velocity.r_component
        fields["velocity_theta"] = converter.velocity_fields.velocity.θ_component  
        fields["velocity_phi"] = converter.velocity_fields.velocity.φ_component
    end
    
    # Add magnetic field data
    if converter.magnetic_fields !== nothing
        fields["magnetic_r"] = converter.magnetic_fields.magnetic.r_component
        fields["magnetic_theta"] = converter.magnetic_fields.magnetic.θ_component
        fields["magnetic_phi"] = converter.magnetic_fields.magnetic.φ_component
    end
    
    # Add temperature data
    if converter.temperature_field !== nothing
        fields["temperature"] = converter.temperature_field.temperature
    end
    
    # Create output configuration
    output_config = create_shtns_aware_output_config(
        converter.shtns_config, 
        converter.pencils,
        output_dir = dirname(output_filename),
        filename_prefix = splitext(basename(output_filename))[1],
        output_physical = true,
        output_spectral = false,
        compression_level = 6
    )
    
    # Extract field information
    field_info = extract_field_info(fields, converter.shtns_config, converter.pencils)
    
    # Compute diagnostics
    diagnostics = compute_global_diagnostics(converter)
    
    # Create time tracker (for interface compatibility)
    time_tracker = create_time_tracker(output_config, converter.time)
    
    # Create metadata dictionary with all the required information
    metadata = Dict{String,Any}(
        "current_time" => converter.time,
        "current_step" => converter.step
    )
    # Add diagnostics to metadata
    merge!(metadata, diagnostics)
    
    try
        # Use the existing parallel I/O infrastructure with correct signature
        write_fields!(fields, time_tracker, metadata, output_config)
        
        if rank == 0
            @info "Physical fields saved successfully"
        end
        
    catch e
        if rank == 0
            @error "Failed to save physical fields" exception=e
        end
        rethrow(e)
    end
end

"""
    convert_spectral_file(input_filename::String, output_filename::String = "";
                         precision::Type{T} = Float64, 
                         compute_diagnostics::Bool = true) where T

Main function to convert a single spectral data file to physical space.
"""
function convert_spectral_file(input_filename::String, output_filename::String = "";
                              precision::Type{T} = Float64,
                              compute_diagnostics::Bool = true) where T
    
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    
    if rank == 0
        @info "=" ^ 70
        @info "Converting Spectral Data to Physical Space"
        @info "Input file: $input_filename"
        @info "Using $nprocs MPI processes"
        @info "=" ^ 70
    end
    
    # Generate output filename if not provided
    if isempty(output_filename)
        base_name = splitext(basename(input_filename))[1]
        output_dir = dirname(input_filename)
        output_filename = joinpath(output_dir, "$(base_name)_physical.nc")
    end
    
    # Create converter
    converter = create_spectral_converter(input_filename, precision=precision)
    
    # Load spectral data
    load_spectral_data!(converter, input_filename)
    
    # Convert to physical space
    convert_to_physical!(converter)
    
    # Compute diagnostics
    if compute_diagnostics
        diagnostics = compute_global_diagnostics(converter)
        
        if rank == 0
            @info "Global Field Diagnostics:"
            @info "-" ^ 40
            
            if haskey(diagnostics, "kinetic_energy")
                @info "Velocity:"
                @info "  Kinetic energy: $(round(diagnostics["kinetic_energy"], digits=6))"
                @info "  RMS components: r=$(round(diagnostics["velocity_r_rms"], digits=6)), " *
                      "θ=$(round(diagnostics["velocity_theta_rms"], digits=6)), " *
                      "φ=$(round(diagnostics["velocity_phi_rms"], digits=6))"
                @info "  Max magnitude: $(round(diagnostics["velocity_max"], digits=6))"
            end
            
            if haskey(diagnostics, "magnetic_energy")
                @info "Magnetic:"
                @info "  Magnetic energy: $(round(diagnostics["magnetic_energy"], digits=6))"  
                @info "  Max magnitude: $(round(diagnostics["magnetic_max"], digits=6))"
            end
            
            if haskey(diagnostics, "temperature_mean")
                @info "Temperature:"
                @info "  Range: [$(round(diagnostics["temperature_min"], digits=6)), " *
                      "$(round(diagnostics["temperature_max"], digits=6))]"
                @info "  Mean: $(round(diagnostics["temperature_mean"], digits=6))"
                if haskey(diagnostics, "nusselt_number")
                    @info "  Nusselt number: $(round(diagnostics["nusselt_number"], digits=6))"
                end
            end
        end
    end
    
    # Save to output file
    save_physical_fields(converter, output_filename)
    
    if rank == 0
        @info "Conversion completed successfully!"
        @info "Output file: $output_filename"
        @info "=" ^ 70
    end
    
    return converter
end

"""
    batch_convert_directory(input_dir::String, output_dir::String = "";
                           pattern::String = "combined_global_time_",
                           precision::Type{T} = Float64,
                           compute_diagnostics::Bool = true) where T

Convert all spectral files matching a pattern in a directory.
"""
function batch_convert_directory(input_dir::String, output_dir::String = "";
                                pattern::String = "combined_global_time_",
                                precision::Type{T} = Float64,
                                compute_diagnostics::Bool = true) where T
    
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    
    if rank == 0
        @info "Batch converting files in directory: $input_dir"
        @info "Pattern: $pattern"
        @info "Using $nprocs MPI processes"
    end
    
    # Set default output directory
    if isempty(output_dir)
        output_dir = joinpath(input_dir, "physical_space")
    end
    
    # Create output directory
    if rank == 0 && !isdir(output_dir)
        mkpath(output_dir)
        @info "Created output directory: $output_dir"
    end
    
    MPI.Barrier(MPI.COMM_WORLD)
    
    # Find matching files (rank 0 only)
    files = String[]
    if rank == 0
        all_files = readdir(input_dir)
        files = filter(f -> contains(f, pattern) && endswith(f, ".nc"), all_files)
        sort!(files)  # Process in order
        
        if isempty(files)
            @warn "No files found matching pattern '$pattern' in $input_dir"
        else
            @info "Found $(length(files)) files to convert"
        end
    end
    
    # Broadcast file list to all processes
    num_files = MPI.bcast(length(files), 0, MPI.COMM_WORLD)
    
    if num_files == 0
        return
    end
    
    # Broadcast file names
    if rank != 0
        files = Vector{String}(undef, num_files)
    end
    
    for i in 1:num_files
        files[i] = MPI.bcast(rank == 0 ? files[i] : "", 0, MPI.COMM_WORLD)
    end
    
    # Process each file
    success_count = 0
    for (i, filename) in enumerate(files)
        input_path = joinpath(input_dir, filename)
        
        # Generate output filename
        base_name = splitext(filename)[1]
        output_filename = joinpath(output_dir, "$(base_name)_physical.nc")
        
        if rank == 0
            @info "[$i/$num_files] Processing: $filename"
        end
        
        try
            converter = convert_spectral_file(input_path, output_filename, 
                                            precision=precision,
                                            compute_diagnostics=compute_diagnostics)
            
            success_count += 1
            
        catch e
            if rank == 0
                @error "Failed to process $filename" exception=e
            end
        end
        
        MPI.Barrier(MPI.COMM_WORLD)  # Synchronize between files
    end
    
    if rank == 0
        @info "Batch conversion completed: $success_count/$num_files files processed successfully"
    end
end

# Main interface functions for backward compatibility

"""
    main_convert_file(filename::String; kwargs...)

Main entry point for converting a single file.
"""
function main_convert_file(filename::String; 
                          output_filename::String = "",
                          precision::Type = Float64,
                          compute_stats::Bool = true)
    
    # Initialize parameters if not already done
    if GEODYNAMO_PARAMS[] === nothing
        initialize_parameters()
    end
    
    return convert_spectral_file(filename, output_filename,
                               precision=precision,
                               compute_diagnostics=compute_stats)
end

"""
    main_batch_convert(input_dir::String; kwargs...)

Main entry point for batch conversion of files.
"""
function main_batch_convert(input_dir::String;
                           output_dir::String = "",
                           pattern::String = "combined_global_time_", 
                           precision::Type = Float64,
                           compute_stats::Bool = true)
    
    # Initialize parameters if not already done 
    if GEODYNAMO_PARAMS[] === nothing
        initialize_parameters()
    end
    
    return batch_convert_directory(input_dir, output_dir,
                                 pattern=pattern,
                                 precision=precision, 
                                 compute_diagnostics=compute_stats)
end

# Export main interface functions
export SpectralToPhysicalConverter
export create_spectral_converter, load_spectral_data!, convert_to_physical!
export compute_global_diagnostics, save_physical_fields
export convert_spectral_file, batch_convert_directory
export main_convert_file, main_batch_convert
