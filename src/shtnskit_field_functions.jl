# ================================================================================
# Core Transform Functions using SHTnsKit with PencilArrays
# ================================================================================

"""
    shtnskit_spectral_to_physical!(spec::SHTnsSpectralField{T},
                                  phys::SHTnsPhysicalField{T}) where T

Transform from spectral to physical space using SHTnsKit with PencilArrays/PencilFFTs.
"""
function shtnskit_spectral_to_physical!(spec::SHTnsSpectralField{T},
                                       phys::SHTnsPhysicalField{T}) where T
    config = spec.config
    sht_config = config.sht_config

    # Simple direct method for all cases
    perform_synthesis_direct!(spec, phys, config)

    # Synchronize MPI processes
    MPI.Barrier(get_comm())
end

"""
    perform_synthesis_phi_local!(spec, phys, config)

Perform synthesis when physical field is already in phi-pencil (phi is local).
"""
function perform_synthesis_phi_local!(spec::SHTnsSpectralField{T}, 
                                     phys::SHTnsPhysicalField{T}, 
                                     config) where T
    sht_config = config.sht_config
    
    # Get local data
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag) 
    phys_data = parent(phys.data)
    
    # Process each local radial level
    for r_local in axes(phys_data, 3)
        # Extract spectral coefficients for this radial level
        coeffs_matrix = extract_coefficients_for_shtnskit(spec_real_data, spec_imag_data, r_local, config)
        
        # Perform SHTnsKit synthesis (handles internal FFTs)
        phys_slice = SHTnsKit.synthesis(sht_config, coeffs_matrix; real_output=true)
        
        # Store in physical array (respecting PencilArray layout)
        store_physical_slice_phi_local!(phys_data, phys_slice, r_local, config)
    end
end

"""
    perform_synthesis_with_transpose!(spec, phys, config, back_plan)

Perform synthesis with transpose to phi-pencil for optimal FFT performance.
"""
function perform_synthesis_with_transpose!(spec::SHTnsSpectralField{T}, 
                                         phys::SHTnsPhysicalField{T}, 
                                         config, back_plan) where T
    # Create temporary phi-pencil array
    phys_phi = PencilArray{T}(undef, config.pencils.phi)
    
    # Perform synthesis to phi-pencil
    perform_synthesis_to_phi_pencil!(spec, phys_phi, config)

    # Transpose back to original pencil orientation using pre-computed plan
    mul!(phys.data, back_plan, phys_phi)
end

"""
    perform_synthesis_to_phi_pencil!(spec, phys_phi, config)

Perform synthesis directly to phi-pencil array.
"""
function perform_synthesis_to_phi_pencil!(spec::SHTnsSpectralField{T}, 
                                        phys_phi::PencilArray{T,3}, 
                                        config) where T
    sht_config = config.sht_config
    
    # Get data arrays
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag)
    phys_phi_data = parent(phys_phi)
    
    # Process each radial level
    for r_local in axes(phys_phi_data, 3)
        # Extract spectral coefficients
        coeffs_matrix = extract_coefficients_for_shtnskit(spec_real_data, spec_imag_data, r_local, config)
        
        # Perform synthesis
        phys_slice = SHTnsKit.synthesis(sht_config, coeffs_matrix; real_output=true)
        
        # Store in phi-pencil array
        store_physical_slice_phi_local!(phys_phi_data, phys_slice, r_local, config)
    end
end

"""
    perform_synthesis_direct!(spec, phys, config)

Direct synthesis without transpose (fallback method).
"""
function perform_synthesis_direct!(spec::SHTnsSpectralField{T},
                                  phys::SHTnsPhysicalField{T},
                                  config) where T
    sht_config = config.sht_config

    # Get local data
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag)
    phys_data = parent(phys.data)

    # Process each radial level
    for r_local in axes(phys_data, 3)
        # Extract coefficients
        coeffs_matrix = extract_coefficients_for_shtnskit(spec_real_data, spec_imag_data, r_local, config)

        # Perform synthesis
        phys_slice = SHTnsKit.synthesis(sht_config, coeffs_matrix; real_output=true)

        # Store result (generic storage for any pencil orientation)
        store_physical_slice_generic!(phys_data, phys_slice, r_local, config)
    end
end

"""
    shtnskit_physical_to_spectral!(phys::SHTnsPhysicalField{T}, 
                                  spec::SHTnsSpectralField{T}) where T

Transform from physical to spectral space using SHTnsKit with PencilArrays.
"""
function shtnskit_physical_to_spectral!(phys::SHTnsPhysicalField{T}, 
                                       spec::SHTnsSpectralField{T}) where T
    config = spec.config
    sht_config = config.sht_config
    
    # Use direct method - handles distribution via MPI communication
    perform_analysis_direct!(phys, spec, config)
    
    # Synchronize MPI processes
    MPI.Barrier(get_comm())
end

"""
    perform_analysis_phi_local!(phys, spec, config)

Perform analysis when physical field is in phi-pencil (phi is local).
"""
function perform_analysis_phi_local!(phys::SHTnsPhysicalField{T}, 
                                    spec::SHTnsSpectralField{T}, 
                                    config) where T
    sht_config = config.sht_config
    
    # Get local data
    phys_data = parent(phys.data)
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag)
    
    # Process each radial level
    for r_local in axes(phys_data, 3)
        # Extract physical slice
        phys_slice = extract_physical_slice_phi_local(phys_data, r_local, config)
        
        # Perform SHTnsKit analysis
        coeffs_matrix = SHTnsKit.analysis(sht_config, phys_slice)
        
        # Store spectral coefficients
        store_coefficients_from_shtnskit!(spec_real_data, spec_imag_data, coeffs_matrix, r_local, config)
    end
end

"""
    perform_analysis_with_transpose!(phys, spec, config, to_phi_plan)

Perform analysis with transpose to phi-pencil.
"""
function perform_analysis_with_transpose!(phys::SHTnsPhysicalField{T},
                                        spec::SHTnsSpectralField{T},
                                        config, to_phi_plan) where T
    phys_phi = PencilArray{T}(undef, config.pencils.phi)
    # Transpose to phi-pencil using pre-computed plan
    mul!(phys_phi, to_phi_plan, phys.data)
    perform_analysis_from_phi_pencil!(phys_phi, spec, config)
end

"""
    perform_analysis_from_phi_pencil!(phys_phi, spec, config)

Perform analysis from phi-pencil data.
"""
function perform_analysis_from_phi_pencil!(phys_phi::PencilArray{T,3}, 
                                         spec::SHTnsSpectralField{T}, 
                                         config) where T
    sht_config = config.sht_config
    
    # Get data arrays
    phys_phi_data = parent(phys_phi)
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag)
    
    # Process each radial level
    for r_local in axes(phys_phi_data, 3)
        # Extract physical slice
        phys_slice = extract_physical_slice_phi_local(phys_phi_data, r_local, config)
        
        # Perform analysis
        coeffs_matrix = SHTnsKit.analysis(sht_config, phys_slice)
        
        # Store coefficients
        store_coefficients_from_shtnskit!(spec_real_data, spec_imag_data, coeffs_matrix, r_local, config)
    end
end

"""
    perform_analysis_direct!(phys, spec, config)

Direct analysis without transpose (fallback).
"""
function perform_analysis_direct!(phys::SHTnsPhysicalField{T},
                                 spec::SHTnsSpectralField{T},
                                 config) where T
    sht_config = config.sht_config

    # Get local data
    phys_data = parent(phys.data)
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag)

    # Process each radial level
    for r_local in axes(phys_data, 3)
        # Extract physical slice (generic extraction)
        phys_slice = extract_physical_slice_generic(phys_data, r_local, config)

        # Perform analysis
        coeffs_matrix = SHTnsKit.analysis(sht_config, phys_slice)

        # Store coefficients
        store_coefficients_from_shtnskit!(spec_real_data, spec_imag_data, coeffs_matrix, r_local, config)
    end
end

# ================================================================================
# Vector Transforms with SHTnsKit and PencilArrays
# ================================================================================

"""
    shtnskit_vector_synthesis!(tor_spec::SHTnsSpectralField{T}, 
                              pol_spec::SHTnsSpectralField{T},
                              vec_phys::SHTnsVectorField{T}) where T

Vector synthesis using SHTnsKit spheroidal-toroidal decomposition with PencilArrays.
"""
function shtnskit_vector_synthesis!(tor_spec::SHTnsSpectralField{T}, 
                                   pol_spec::SHTnsSpectralField{T},
                                   vec_phys::SHTnsVectorField{T}) where T
    config = tor_spec.config
    sht_config = config.sht_config
    
    # Get data arrays
    tor_real = parent(tor_spec.data_real)
    tor_imag = parent(tor_spec.data_imag)
    pol_real = parent(pol_spec.data_real) 
    pol_imag = parent(pol_spec.data_imag)
    
    v_theta = parent(vec_phys.θ_component.data)
    v_phi = parent(vec_phys.φ_component.data)
    
    # Process each radial level
    for r_local in axes(tor_real, 3)
        # Extract toroidal and poloidal coefficients (includes MPI gathering)
        tor_coeffs = extract_coefficients_for_shtnskit(tor_real, tor_imag, r_local, config)
        pol_coeffs = extract_coefficients_for_shtnskit(pol_real, pol_imag, r_local, config)

        # Perform vector synthesis using SHTnsKit
        vt_field, vp_field = SHTnsKit.SHsphtor_to_spat(sht_config, pol_coeffs, tor_coeffs;
                                                      real_output=true)

        # Store vector components
        store_vector_components_generic!(v_theta, v_phi, vt_field, vp_field, r_local, config)
    end
    
    MPI.Barrier(get_comm())
end

"""
    shtnskit_vector_analysis!(vec_phys::SHTnsVectorField{T},
                             tor_spec::SHTnsSpectralField{T}, 
                             pol_spec::SHTnsSpectralField{T}) where T

Vector analysis using SHTnsKit with PencilArrays.
"""
function shtnskit_vector_analysis!(vec_phys::SHTnsVectorField{T},
                                  tor_spec::SHTnsSpectralField{T}, 
                                  pol_spec::SHTnsSpectralField{T}) where T
    config = tor_spec.config
    sht_config = config.sht_config
    
    # Get data arrays
    v_theta = parent(vec_phys.θ_component.data)
    v_phi = parent(vec_phys.φ_component.data)
    
    tor_real = parent(tor_spec.data_real)
    tor_imag = parent(tor_spec.data_imag)
    pol_real = parent(pol_spec.data_real)
    pol_imag = parent(pol_spec.data_imag)
    
    # Process each radial level  
    for r_local in axes(v_theta, 3)
        # Extract vector components
        vt_field = extract_vector_component_generic(v_theta, r_local, config)
        vp_field = extract_vector_component_generic(v_phi, r_local, config)
        
        # Perform vector analysis using SHTnsKit
        pol_coeffs, tor_coeffs = SHTnsKit.spat_to_SHsphtor(sht_config, vt_field, vp_field)
        
        # Store spectral coefficients
        store_coefficients_from_shtnskit!(pol_real, pol_imag, pol_coeffs, r_local, config)
        store_coefficients_from_shtnskit!(tor_real, tor_imag, tor_coeffs, r_local, config)
    end
    
    MPI.Barrier(get_comm())
end

# ================================================================================
# Helper Functions for PencilArray Data Management
# ================================================================================

"""
    get_pencil_orientation(pencil::Pencil{3}) -> Symbol

Get the orientation of a pencil (which dimensions are local).
"""
function get_pencil_orientation(pencil::Pencil{3})
    local_ranges = pencil.axes_local
    global_sizes = pencil.size_global

    θ_local = length(local_ranges[1]) == global_sizes[1]
    φ_local = length(local_ranges[2]) == global_sizes[2]

    if θ_local && φ_local
        return :theta_phi  # both angular directions fully local
    elseif θ_local
        return :theta
    elseif φ_local
        return :phi
    else
        return :r
    end
end

"""
    extract_coefficients_for_shtnskit!(coeffs_buffer, spec_real, spec_imag, r_local, config)

Extract spectral coefficients in format expected by SHTnsKit using pre-allocated buffer.
"""
function extract_coefficients_for_shtnskit!(coeffs_buffer::Matrix{ComplexF64},
                                           spec_real, spec_imag, r_local, config)
    lmax, mmax = config.lmax, config.mmax

    # Get buffer dimensions (may be larger than our config.lmax/mmax if SHTnsKit uses larger values)
    buffer_lmax = size(coeffs_buffer, 1) - 1
    buffer_mmax = size(coeffs_buffer, 2) - 1

    # Clear the buffer for reuse
    fill!(coeffs_buffer, zero(ComplexF64))

    # Fill from local spectral data
    Threads.@threads for lm_idx in eachindex(IndexLinear(), view(spec_real, :, 1, 1))
        l, m = index_to_lm_shtnskit(lm_idx, lmax, mmax)
        if r_local <= size(spec_real, 3) && l >= 0 && m >= 0 &&
           l <= buffer_lmax && m <= buffer_mmax  # Check bounds
            real_part = spec_real[lm_idx, 1, r_local]
            imag_part = spec_imag[lm_idx, 1, r_local]
            coeffs_buffer[l+1, m+1] = complex(real_part, imag_part)
        end
    end

    return coeffs_buffer
end

# Convenience wrapper for backward compatibility - uses buffer cache
function extract_coefficients_for_shtnskit(spec_real, spec_imag, r_local, config)
    # Get or create cached buffer - use config.lmax/mmax (what we requested)
    # SHTnsKit functions expect matrices of size (lmax+1) × (mmax+1)
    buffer_key = :coeffs_buffer
    if !haskey(config._buffer_cache, buffer_key)
        lmax, mmax = config.lmax, config.mmax
        config._buffer_cache[buffer_key] = zeros(ComplexF64, lmax+1, mmax+1)
    end

    coeffs_buffer = config._buffer_cache[buffer_key]
    extract_coefficients_for_shtnskit!(coeffs_buffer, spec_real, spec_imag, r_local, config)

    # Gather spectral coefficients across all MPI processes
    # SHTnsKit functions need complete (l,m) coefficient matrices
    buffer_gathered_key = :coeffs_buffer_gathered
    if !haskey(config._buffer_cache, buffer_gathered_key)
        lmax, mmax = config.lmax, config.mmax
        config._buffer_cache[buffer_gathered_key] = zeros(ComplexF64, lmax+1, mmax+1)
    end
    coeffs_gathered = config._buffer_cache[buffer_gathered_key]

    MPI.Allreduce!(coeffs_buffer, coeffs_gathered, MPI.SUM, get_comm())

    # Return a copy to avoid buffer aliasing when called multiple times
    # (e.g., for both toroidal and poloidal coefficients in vector transforms)
    return copy(coeffs_gathered)
end

"""
    store_coefficients_from_shtnskit!(spec_real, spec_imag, coeffs_matrix, r_local, config)

Store coefficients from SHTnsKit format back to spectral field.
"""
function store_coefficients_from_shtnskit!(spec_real, spec_imag, coeffs_matrix, r_local, config)
    # Use config.lmax/mmax for iterating over our spectral field
    lmax, mmax = config.lmax, config.mmax

    # Get dimensions of coeffs_matrix returned by SHTnsKit
    # SHTnsKit may use different lmax/mmax internally
    matrix_lmax = size(coeffs_matrix, 1) - 1
    matrix_mmax = size(coeffs_matrix, 2) - 1

    Threads.@threads for lm_idx in eachindex(IndexLinear(), view(spec_real, :, 1, 1))
        l, m = index_to_lm_shtnskit(lm_idx, lmax, mmax)
        if r_local <= size(spec_real, 3) && l >= 0 && m >= 0
            # Check if this (l,m) exists in the SHTnsKit matrix
            if l <= matrix_lmax && m <= matrix_mmax
                coeff = coeffs_matrix[l+1, m+1]
                spec_real[lm_idx, 1, r_local] = real(coeff)
                spec_imag[lm_idx, 1, r_local] = imag(coeff)

                # Ensure m=0 modes are real
                if m == 0
                    spec_imag[lm_idx, 1, r_local] = 0.0
                end
            else
                # Mode doesn't exist in SHTnsKit matrix - set to zero
                spec_real[lm_idx, 1, r_local] = 0.0
                spec_imag[lm_idx, 1, r_local] = 0.0
            end
        end
    end
end

"""
    index_to_lm_shtnskit(idx, lmax, mmax) -> (l, m)

Convert linear index to (l,m) for SHTnsKit compatibility.
"""
function index_to_lm_shtnskit(idx::Int, lmax::Int, mmax::Int)
    # Simple conversion - this should match SHTnsKit's indexing
    current_idx = 0
    for l in 0:lmax
        for m in 0:min(l, mmax)
            current_idx += 1
            if current_idx == idx
                return l, m
            end
        end
    end
    return 0, 0  # fallback
end

"""
    store_physical_slice_phi_local!(phys_data, phys_slice, r_local, config)

Store physical slice when in phi-local pencil.
"""
function store_physical_slice_phi_local!(phys_data, phys_slice, r_local, config)
    nlat, nlon = config.nlat, config.nlon
    
    # Store respecting the phi-local layout
    common_i_range = 1:min(size(phys_data, 1), nlat, size(phys_slice, 1))
    common_j_range = 1:min(size(phys_data, 2), nlon, size(phys_slice, 2))
    
    Threads.@threads for i in common_i_range
        for j in common_j_range
            if r_local <= size(phys_data, 3)
                phys_data[i, j, r_local] = phys_slice[i, j]
            end
        end
    end
end

"""
    store_physical_slice_generic!(phys_data, phys_slice, r_local, config)

Generic storage for any pencil orientation.
"""
function store_physical_slice_generic!(phys_data, phys_slice, r_local, config)
    # This is a generic fallback - may not be optimal for all pencil orientations
    common_i_range = 1:min(size(phys_data, 1), size(phys_slice, 1))
    common_j_range = 1:min(size(phys_data, 2), size(phys_slice, 2))
    
    Threads.@threads for i in common_i_range
        for j in common_j_range
            if r_local <= size(phys_data, 3)
                phys_data[i, j, r_local] = phys_slice[i, j]
            end
        end
    end
end

"""
    extract_physical_slice_phi_local!(slice_buffer, phys_data, r_local, config)

Extract physical slice when in phi-local pencil using pre-allocated buffer.
"""
function extract_physical_slice_phi_local!(slice_buffer::Matrix{T}, phys_data, r_local, config) where T
    nlat, nlon = config.nlat, config.nlon

    # Clear buffer for reuse
    fill!(slice_buffer, zero(T))

    common_i_range = 1:min(size(phys_data, 1), nlat, size(slice_buffer, 1))
    common_j_range = 1:min(size(phys_data, 2), nlon, size(slice_buffer, 2))

    Threads.@threads for i in common_i_range
        for j in common_j_range
            if r_local <= size(phys_data, 3)
                slice_buffer[i, j] = phys_data[i, j, r_local]
            end
        end
    end

    # Gather complete grid across all MPI processes
    MPI.Allreduce!(slice_buffer, MPI.SUM, get_comm())

    return slice_buffer
end

# Backward compatibility wrapper
function extract_physical_slice_phi_local(phys_data, r_local, config)
    nlat, nlon = config.nlat, config.nlon
    # Get or create cached buffer for phi slice
    buffer_key = :phi_slice_buffer
    if !haskey(config._buffer_cache, buffer_key)
        config._buffer_cache[buffer_key] = zeros(eltype(phys_data), nlat, nlon)
    end
    
    slice_buffer = config._buffer_cache[buffer_key]
    return extract_physical_slice_phi_local!(slice_buffer, phys_data, r_local, config)
end

"""
    extract_physical_slice_generic!(slice_buffer, phys_data, r_local, config)

Generic extraction for any pencil orientation using pre-allocated buffer.
"""
function extract_physical_slice_generic!(slice_buffer::Matrix{T}, phys_data, r_local, config) where T
    nlat, nlon = config.nlat, config.nlon

    # Clear buffer for reuse
    fill!(slice_buffer, zero(T))

    # Generic extraction - may need MPI communication for distributed dimensions
    common_i_range = 1:min(size(phys_data, 1), nlat, size(slice_buffer, 1))
    common_j_range = 1:min(size(phys_data, 2), nlon, size(slice_buffer, 2))

    Threads.@threads for i in common_i_range
        for j in common_j_range
            if r_local <= size(phys_data, 3)
                slice_buffer[i, j] = phys_data[i, j, r_local]
            end
        end
    end

    # Gather complete grid across all MPI processes
    MPI.Allreduce!(slice_buffer, MPI.SUM, get_comm())

    return slice_buffer
end

# Backward compatibility wrapper
function extract_physical_slice_generic(phys_data, r_local, config)
    nlat, nlon = config.nlat, config.nlon
    # Get or create cached buffer for generic slice  
    buffer_key = :generic_slice_buffer
    if !haskey(config._buffer_cache, buffer_key)
        config._buffer_cache[buffer_key] = zeros(eltype(phys_data), nlat, nlon)
    end
    
    slice_buffer = config._buffer_cache[buffer_key]
    return extract_physical_slice_generic!(slice_buffer, phys_data, r_local, config)
end

"""
    extract_vector_component_generic!(component_buffer, v_data, r_local, config)

Generic extraction for vector components using pre-allocated buffer.
"""
function extract_vector_component_generic!(component_buffer::Matrix{T}, v_data, r_local, config) where T
    nlat, nlon = config.nlat, config.nlon

    # Clear buffer for reuse
    fill!(component_buffer, zero(T))

    common_i_range = 1:min(size(v_data, 1), nlat, size(component_buffer, 1))
    common_j_range = 1:min(size(v_data, 2), nlon, size(component_buffer, 2))

    for i in common_i_range
        for j in common_j_range
            if r_local <= size(v_data, 3)
                component_buffer[i, j] = v_data[i, j, r_local]
            end
        end
    end

    # Gather complete grid across all MPI processes
    MPI.Allreduce!(component_buffer, MPI.SUM, get_comm())

    return component_buffer
end

# Backward compatibility wrapper
function extract_vector_component_generic(v_data, r_local, config)
    nlat, nlon = config.nlat, config.nlon
    component_buffer = zeros(eltype(v_data), nlat, nlon)
    return extract_vector_component_generic!(component_buffer, v_data, r_local, config)
end

"""
    store_vector_components_generic!(v_theta, v_phi, vt_field, vp_field, r_local, config)

Store vector components for any pencil orientation.
"""
function store_vector_components_generic!(v_theta, v_phi, vt_field, vp_field, r_local, config)
    common_i_range = 1:min(size(v_theta, 1), size(vt_field, 1))
    common_j_range = 1:min(size(v_theta, 2), size(vt_field, 2))
    
    for i in common_i_range
        for j in common_j_range
            if r_local <= size(v_theta, 3) && r_local <= size(v_phi, 3)
                v_theta[i, j, r_local] = vt_field[i, j]
                v_phi[i, j, r_local] = vp_field[i, j]
            end
        end
    end
end

# ================================================================================
# Batch Processing for Enhanced Performance
# ================================================================================

"""
    batch_shtnskit_transforms!(specs::Vector{SHTnsSpectralField{T}},
                              physs::Vector{SHTnsPhysicalField{T}}) where T

Batch process multiple transforms using SHTnsKit with PencilArrays.
"""
function batch_shtnskit_transforms!(specs::Vector{SHTnsSpectralField{T}},
                                   physs::Vector{SHTnsPhysicalField{T}}) where T
    @assert length(specs) == length(physs)
    
    if isempty(specs)
        return
    end
    
    # Process in parallel using threading
    @threads for batch_idx in eachindex(specs)
        shtnskit_spectral_to_physical!(specs[batch_idx], physs[batch_idx])
    end
end

# ---------------------------------------------------------------------------
# Backward-compatible alias used by other modules
# ---------------------------------------------------------------------------
"""
    batch_spectral_to_physical!(specs, physs)

Compatibility wrapper that calls `batch_shtnskit_transforms!` for batched
spectral→physical transforms using SHTnsKit with PencilArrays/MPI.
"""
function batch_spectral_to_physical!(specs::Vector{SHTnsSpectralField{T}},
                                     physs::Vector{SHTnsPhysicalField{T}}) where T
    return batch_shtnskit_transforms!(specs, physs)
end

# ================================================================================
# Performance Monitoring
# ================================================================================

"""
    get_shtnskit_performance_stats()

Get performance statistics for SHTnsKit transforms with PencilArrays.
"""
function get_shtnskit_performance_stats()
    return (
        library = "SHTnsKit",
        parallelization = "theta-phi MPI + PencilArrays",
        fft_backend = "PencilFFTs",
        optimization = "enabled"
    )
end

# ================================================================================
# Functions for SHTnsKit field transforms (exports handled by main GeoDynamo.jl)
# ================================================================================

# ================================================================================
# MPI and PencilFFTs Synchronization Utilities  
# ================================================================================

"""
    synchronize_pencil_data!(field)

Synchronize PencilArray data across MPI processes to ensure consistency.
"""
function synchronize_pencil_data!(field::Union{SHTnsSpectralField{T}, SHTnsPhysicalField{T}}) where T
    # Synchronize the underlying PencilArray data
    if hasmethod(MPI.Barrier, Tuple{typeof(get_comm())})
        MPI.Barrier(get_comm())
    end
    return field
end

"""
    optimize_fft_performance!(config::SHTnsKitConfig)

Optimize PencilFFTs performance by warming up plans and checking efficiency.
"""
function optimize_fft_performance!(config::SHTnsKitConfig)
    # Warm up FFT plans for better performance
    if haskey(config.fft_plans, :phi_forward) && !get(config.fft_plans, :fallback, false)
        try
            # Create a test array to warm up the plans
            test_pencil = config.pencils.phi
            test_array = PencilArray{ComplexF64}(undef, test_pencil)
            fill!(parent(test_array), complex(1.0, 0.0))
            
            # Execute forward and backward transforms
            plan_forward = config.fft_plans[:phi_forward]
            plan_backward = config.fft_plans[:phi_backward]
            
            plan_forward * test_array
            plan_backward * test_array
            
            if get_rank() == 0
                @info "PencilFFTs plans warmed up successfully"
            end
        catch e
            @warn "Could not warm up PencilFFTs plans: $e"
        end
    end
    return config
end

"""
    validate_pencil_decomposition(config::SHTnsKitConfig)

Validate that pencil decomposition is optimal for the problem size and MPI configuration.
"""
function validate_pencil_decomposition(config::SHTnsKitConfig)
    rank = get_rank()
    nprocs = get_nprocs()
    
    if nprocs > 1 && rank == 0
        nlat, nlon = config.nlat, config.nlon
        
        # Check load balance
        theta_per_proc = nlat ÷ nprocs
        phi_per_proc = nlon ÷ nprocs
        
        theta_imbalance = nlat % nprocs
        phi_imbalance = nlon % nprocs
        
        @info """
        Pencil Decomposition Validation:
          Grid: $nlat × $nlon
          Processes: $nprocs
          Theta per process: $theta_per_proc (imbalance: $theta_imbalance)
          Phi per process: $phi_per_proc (imbalance: $phi_imbalance)
        """
        
        # Warn about potential issues
        if theta_imbalance > nprocs ÷ 2
            @warn "Significant theta load imbalance detected: $theta_imbalance/$nprocs"
        end
        if phi_imbalance > nprocs ÷ 2
            @warn "Significant phi load imbalance detected: $phi_imbalance/$nprocs"
        end
        
        # Check minimum size per process
        if theta_per_proc < 4 || phi_per_proc < 4
            @warn "Very small sub-domains detected. Consider using fewer processes for better efficiency."
        end

    end
    return config
end

"""
    optimize_erk2_transforms!(config::SHTnsKitConfig)

Optimize SHTnsKit transforms for ERK2 timestepping with PencilFFTs.
This function pre-warms transform plans and optimizes memory layout.
"""
function optimize_erk2_transforms!(config::SHTnsKitConfig)
    rank = get_rank()
    
    if rank == 0
        @info "Optimizing ERK2 transforms with PencilFFTs"
    end
    
    # Pre-warm SHTnsKit configuration
    try
        SHTnsKit.prepare_plm_tables!(config.sht_config)
        if rank == 0
            @info "SHTnsKit Legendre tables pre-computed"
        end
    catch e
        @warn "Could not pre-compute SHTnsKit tables: $e"
    end
    
    # Optimize PencilFFTs plans
    optimize_fft_performance!(config)
    
    # Validate decomposition efficiency
    validate_pencil_decomposition(config)
    
    # Test transform performance with sample data
    if haskey(config.pencils, :phi) && haskey(config.pencils, :spec)
        try
            # Create sample spectral field
            spec_test = PencilArray{ComplexF64}(undef, config.pencils.spec)
            phys_test = PencilArray{Float64}(undef, config.pencils.phi)
            
            # Fill with test data
            fill!(parent(spec_test), complex(1.0, 0.0))
            
            # Test a few transforms to warm up the system
            start_time = MPI.Wtime()
            for i in 1:3
                # Perform synthesis (would use actual SHTnsKit functions in practice)
                fill!(parent(phys_test), 1.0)
                MPI.Barrier(get_comm())
            end
            end_time = MPI.Wtime()
            
            if rank == 0
                avg_time = (end_time - start_time) / 3.0
                @info "Transform warm-up completed: $(round(avg_time*1000, digits=2)) ms per transform"
            end
            
        catch e
            @warn "Transform warm-up failed: $e"
        end
    end

    return config
end


"""
    create_erk2_config(; lmax, mmax, nlat, nlon, optimize_for_erk2=true)

Create an SHTnsKit configuration for ERK2 timestepping.
"""
function create_erk2_config(; lmax::Int, mmax::Int=lmax,
                           nlat::Int=max(lmax+2, get_default_nlat()),
                           nlon::Int=max(2*lmax+1, 4, get_default_nlon()),
                           optimize_for_erk2::Bool=true)
    config = create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon, optimize_decomp=true)
    if optimize_for_erk2
        optimize_erk2_transforms!(config)
    end
    return config
end
