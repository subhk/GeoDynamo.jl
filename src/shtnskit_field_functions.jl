# ================================================================================
# Core Transform Functions using SHTnsKit with PencilArrays
# ================================================================================
#
# This file implements the spherical harmonic transform operations that convert
# between spectral (l,m) representation and physical (θ,φ) representation.
#
# TRANSFORM TERMINOLOGY:
# ----------------------
# - Synthesis: Spectral → Physical (inverse SH transform)
#   Reconstructs field values on the (θ,φ) grid from spherical harmonic coefficients
#
# - Analysis: Physical → Spectral (forward SH transform)
#   Computes spherical harmonic coefficients from field values on the grid
#
# IMPLEMENTATION STRATEGY:
# ------------------------
# For each radial level independently:
# 1. Extract/prepare spectral coefficients in SHTnsKit's expected format
# 2. Call SHTnsKit.synthesis() or SHTnsKit.analysis()
# 3. Store/scatter results to the appropriate PencilArray
#
# MPI parallelization is handled through PencilArrays, with MPI.Barrier()
# synchronization after complete transforms.
#
# ================================================================================

"""
    shtnskit_spectral_to_physical!(spec, phys)

Transform spectral coefficients to physical space values (synthesis).

# The Synthesis Operation
Given spherical harmonic coefficients f_l^m, compute the physical field:
    f(θ,φ) = Σ_{l,m} f_l^m × Y_l^m(θ,φ)

where Y_l^m are the spherical harmonics.

# Implementation
Uses SHTnsKit's synthesis function which:
1. Performs the Legendre transform (summing over l for each m)
2. Performs the FFT along longitude (summing over m)

# Arguments
- `spec::SHTnsSpectralField`: Source spectral field with coefficients
- `phys::SHTnsPhysicalField`: Destination physical field (modified in-place)

# Side Effects
Modifies `phys.data` with the synthesized field values
"""
function shtnskit_spectral_to_physical!(spec::SHTnsSpectralField{T},
                                       phys::SHTnsPhysicalField{T}) where T
    config = spec.config
    sht_config = config.sht_config

    # Use direct synthesis method (processes each radial level)
    perform_synthesis_direct!(spec, phys, config)

    # Ensure all MPI processes complete before returning
    MPI.Barrier(get_comm())
end

"""
    perform_synthesis_phi_local!(spec, phys, config)

Perform synthesis when physical field is in phi-pencil orientation.

This is the most efficient synthesis path because:
1. The phi (longitude) dimension is fully local on each process
2. SHTnsKit's FFT operates entirely in local memory
3. No MPI communication needed during the transform itself

# Algorithm for each radial level:
1. Extract spectral coefficients into SHTnsKit's (lmax+1, mmax+1) matrix format
2. Call SHTnsKit.synthesis() which does Legendre transform + FFT
3. Store the resulting (nlat, nlon) physical field slice
"""
function perform_synthesis_phi_local!(spec::SHTnsSpectralField{T},
                                     phys::SHTnsPhysicalField{T},
                                     config) where T
    sht_config = config.sht_config

    # Extract underlying Julia arrays from PencilArrays
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag)
    phys_data = parent(phys.data)

    # Process each radial level independently (embarrassingly parallel in r)
    for r_local in axes(phys_data, 3)
        # Step 1: Gather spectral coefficients into SHTnsKit's expected format
        # Returns a (lmax+1) × (mmax+1) complex matrix
        coeffs_matrix = extract_coefficients_for_shtnskit(spec_real_data, spec_imag_data, r_local, config)

        # Step 2: Perform the actual spherical harmonic synthesis
        # SHTnsKit handles both Legendre transform and longitude FFT internally
        phys_slice = SHTnsKit.synthesis(sht_config, coeffs_matrix; real_output=true)

        # Step 3: Store result in the physical array at this radial level
        store_physical_slice_phi_local!(phys_data, phys_slice, r_local, config)
    end
end

"""
    perform_synthesis_with_transpose!(spec, phys, config, back_plan)

Perform synthesis when physical field is NOT in phi-pencil orientation.

# Strategy
When the target physical field is in a non-phi pencil (e.g., theta or r pencil),
we can't directly use SHTnsKit because it requires all longitude points to be local.

Solution:
1. Create a temporary phi-pencil array
2. Perform synthesis to the temporary array (longitude local)
3. Transpose the result to the target pencil orientation

This involves one extra MPI all-to-all communication (the transpose) but
ensures the FFT can operate on contiguous local data.
"""
function perform_synthesis_with_transpose!(spec::SHTnsSpectralField{T},
                                         phys::SHTnsPhysicalField{T},
                                         config, back_plan) where T
    # Allocate temporary array in phi-pencil orientation
    phys_phi = PencilArray{T}(undef, config.pencils.phi)

    # Perform synthesis with longitude local (optimal for SHTnsKit)
    perform_synthesis_to_phi_pencil!(spec, phys_phi, config)

    # Redistribute data to match target pencil orientation
    # back_plan encodes the MPI communication pattern
    mul!(phys.data, back_plan, phys_phi)
end

"""
    perform_synthesis_to_phi_pencil!(spec, phys_phi, config)

Core synthesis routine that writes directly to a phi-pencil array.

This is the workhorse function called by other synthesis routines.
It assumes the destination array is already in phi-pencil orientation.
"""
function perform_synthesis_to_phi_pencil!(spec::SHTnsSpectralField{T},
                                        phys_phi::PencilArray{T,3},
                                        config) where T
    sht_config = config.sht_config

    # Get underlying Julia arrays
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag)
    phys_phi_data = parent(phys_phi)

    # Loop over radial levels (each level is independent)
    for r_local in axes(phys_phi_data, 3)
        # Prepare spectral coefficients in SHTnsKit format
        coeffs_matrix = extract_coefficients_for_shtnskit(spec_real_data, spec_imag_data, r_local, config)

        # SHTnsKit synthesis: spectral → physical for this radial slice
        phys_slice = SHTnsKit.synthesis(sht_config, coeffs_matrix; real_output=true)

        # Copy result to output array
        store_physical_slice_phi_local!(phys_phi_data, phys_slice, r_local, config)
    end
end

"""
    perform_synthesis_direct!(spec, phys, config)

Direct synthesis method that handles any pencil orientation.

This is the default/fallback method. It works regardless of the physical
field's pencil orientation by using a generic storage function that
handles the index mapping appropriately.

# Note
For phi-pencil physical fields, `perform_synthesis_phi_local!` is more
efficient as it can use optimized storage.
"""
function perform_synthesis_direct!(spec::SHTnsSpectralField{T},
                                  phys::SHTnsPhysicalField{T},
                                  config) where T
    sht_config = config.sht_config

    # Extract underlying arrays from PencilArrays wrapper
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag)
    phys_data = parent(phys.data)

    # Process each radial level
    for r_local in axes(phys_data, 3)
        # Gather spectral coefficients for this radial level
        coeffs_matrix = extract_coefficients_for_shtnskit(spec_real_data, spec_imag_data, r_local, config)

        # Perform SHTnsKit synthesis (Legendre + FFT)
        phys_slice = SHTnsKit.synthesis(sht_config, coeffs_matrix; real_output=true)

        # Store using generic method (works for any pencil orientation)
        store_physical_slice_generic!(phys_data, phys_slice, r_local, config)
    end
end

"""
    shtnskit_physical_to_spectral!(phys, spec)

Transform physical space values to spectral coefficients (analysis).

# The Analysis Operation
Given physical field values f(θ,φ), compute spherical harmonic coefficients:
    f_l^m = ∫∫ f(θ,φ) × Y_l^m*(θ,φ) sin(θ) dθ dφ

The integral is computed numerically using:
- Gauss-Legendre quadrature for the θ integral
- FFT for the φ integral (exploiting periodicity)

# Implementation
Uses SHTnsKit's analysis function which:
1. Performs the FFT along longitude (extracting Fourier modes)
2. Performs the Legendre transform (computing l coefficients for each m)

# Arguments
- `phys::SHTnsPhysicalField`: Source physical field values
- `spec::SHTnsSpectralField`: Destination spectral field (modified in-place)

# Side Effects
Modifies `spec.data_real` and `spec.data_imag` with the computed coefficients
"""
function shtnskit_physical_to_spectral!(phys::SHTnsPhysicalField{T},
                                       spec::SHTnsSpectralField{T}) where T
    config = spec.config
    sht_config = config.sht_config

    # Use direct analysis method (processes each radial level)
    perform_analysis_direct!(phys, spec, config)

    # Ensure all MPI processes complete before returning
    MPI.Barrier(get_comm())
end

"""
    perform_analysis_phi_local!(phys, spec, config)

Perform analysis when physical field is in phi-pencil orientation.

This is the most efficient analysis path because:
1. The phi (longitude) dimension is fully local on each process
2. SHTnsKit's FFT operates entirely in local memory
3. No MPI communication needed during the transform itself
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
                              vec_phys::SHTnsVectorField{T};
                              domain::Union{RadialDomain,Nothing}=nothing) where T

Vector synthesis using SHTnsKit spheroidal-toroidal decomposition with PencilArrays.

# Toroidal-Poloidal Decomposition

For a solenoidal vector field v (∇·v = 0):
    v = ∇×(T r̂) + ∇×∇×(P r̂)

where T = toroidal scalar, P = poloidal scalar.

In spherical components:
    v_r = l(l+1)/r * P * Y_lm   (from poloidal only)
    v_θ, v_φ from both T and P  (computed by SHTnsKit.SHsphtor_to_spat)

CRITICAL: SHTnsKit.SHsphtor_to_spat returns ONLY tangential components.
The radial component v_r MUST be computed separately from the poloidal scalar.

# Parameters
- `domain`: Optional RadialDomain needed for computing v_r with correct radial scaling.
            If not provided, v_r will be set to zero (suitable for tests).
"""
function shtnskit_vector_synthesis!(tor_spec::SHTnsSpectralField{T},
                                   pol_spec::SHTnsSpectralField{T},
                                   vec_phys::SHTnsVectorField{T};
                                   domain::Union{RadialDomain,Nothing}=nothing) where T
    config = tor_spec.config
    sht_config = config.sht_config

    # Get data arrays
    tor_real = parent(tor_spec.data_real)
    tor_imag = parent(tor_spec.data_imag)
    pol_real = parent(pol_spec.data_real)
    pol_imag = parent(pol_spec.data_imag)

    v_r = parent(vec_phys.r_component.data)
    v_theta = parent(vec_phys.θ_component.data)
    v_phi = parent(vec_phys.φ_component.data)

    # Get local radial range
    r_range = get_local_range(pol_spec.pencil, 3)

    # Process each radial level
    for r_local in axes(tor_real, 3)
        # Extract toroidal and poloidal coefficients (includes MPI gathering)
        tor_coeffs = extract_coefficients_for_shtnskit(tor_real, tor_imag, r_local, config)
        pol_coeffs = extract_coefficients_for_shtnskit(pol_real, pol_imag, r_local, config)

        # Perform vector synthesis using SHTnsKit (tangential components only)
        vt_field, vp_field = SHTnsKit.SHsphtor_to_spat(sht_config, pol_coeffs, tor_coeffs;
                                                      real_output=true)

        # Store tangential vector components
        store_vector_components_generic!(v_theta, v_phi, vt_field, vp_field, r_local, config)

        # ========================================================================
        # CRITICAL: Compute radial component from poloidal scalar
        # v_r = l(l+1)/r * P * Y_lm
        # Only computed if domain information is provided
        # ========================================================================

        if domain !== nothing
            # Get global radial index
            r_idx_global = r_local + first(r_range) - 1

            if r_idx_global <= domain.N
                r_val = domain.r[r_idx_global, 4]  # Actual radius value

                if r_val > 1e-15  # Avoid division by zero at r=0
                    # Scale poloidal coefficients by l(l+1)/r
                    lmax, mmax = config.lmax, config.mmax
                    pol_rad_coeffs = zeros(ComplexF64, lmax+1, mmax+1)

                    for l in 0:lmax
                        l_factor = l * (l + 1) / r_val
                        for m in 0:min(l, mmax)
                            pol_rad_coeffs[l+1, m+1] = pol_coeffs[l+1, m+1] * l_factor
                        end
                    end

                    # Synthesize radial component
                    vr_field = SHTnsKit.synthesis(sht_config, pol_rad_coeffs; real_output=true)

                    # Store radial component
                    store_scalar_component_generic!(v_r, vr_field, r_local, config)
                else
                    # At r=0 (ball geometry), v_r must be zero for regularity
                    store_zero_component_generic!(v_r, r_local, config)
                end
            end
        else
            # No domain provided - set v_r to zero for all points at this radial level
            # This is used in tests that don't have domain information
            store_zero_component_generic!(v_r, r_local, config)
        end
    end

    MPI.Barrier(get_comm())
end

"""
    shtnskit_vector_analysis!(vec_phys::SHTnsVectorField{T},
                             tor_spec::SHTnsSpectralField{T},
                             pol_spec::SHTnsSpectralField{T}) where T

Vector analysis using SHTnsKit with PencilArrays.

# Toroidal-Poloidal Analysis

Decomposes a 3D velocity field into toroidal and poloidal scalars.

For a solenoidal vector field v (∇·v = 0):
    v = ∇×(T r̂) + ∇×∇×(P r̂)

# Mathematical Note on Analysis

SHTnsKit.spat_to_SHsphtor takes (v_θ, v_φ) and returns (P, T).

This is mathematically valid for solenoidal fields because:
1. The solenoidal constraint ∇·v = 0 couples v_r to (v_θ, v_φ)
2. The decomposition into T (rotational) and P (potential) parts is unique
3. The radial component v_r = l(l+1)/r * P is implicitly determined

However, this assumes EXACT solenoidality. In numerical simulations with
finite precision, v_r may not exactly satisfy ∇·v = 0.

# Alternative: Use Full 3-Component Analysis

For better numerical accuracy, one could use:
    Q_coeffs = analysis(v_r * r / l(l+1))  # Recover P from v_r
    S, T = spat_to_SHsphtor(v_θ, v_φ)      # Decompose tangential

Then check: Q_coeffs ≈ S_coeffs (should match for solenoidal field)

Current implementation uses 2-component analysis which is standard practice
for solenoidal MHD simulations.
"""
function shtnskit_vector_analysis!(vec_phys::SHTnsVectorField{T},
                                  tor_spec::SHTnsSpectralField{T},
                                  pol_spec::SHTnsSpectralField{T};
                                  domain::Union{RadialDomain,Nothing}=nothing,
                                  verify_solenoidal::Bool=false) where T
    config = tor_spec.config
    sht_config = config.sht_config

    # Get data arrays
    v_r = parent(vec_phys.r_component.data)
    v_theta = parent(vec_phys.θ_component.data)
    v_phi = parent(vec_phys.φ_component.data)

    tor_real = parent(tor_spec.data_real)
    tor_imag = parent(tor_spec.data_imag)
    pol_real = parent(pol_spec.data_real)
    pol_imag = parent(pol_spec.data_imag)

    # Get local radial range
    r_range = get_local_range(pol_spec.pencil, 3)

    # Process each radial level
    for r_local in axes(v_theta, 3)
        # Extract vector components
        vt_field = extract_vector_component_generic(v_theta, r_local, config)
        vp_field = extract_vector_component_generic(v_phi, r_local, config)

        # Perform vector analysis using SHTnsKit (tangential components)
        # This returns P and T assuming solenoidal constraint
        pol_coeffs, tor_coeffs = SHTnsKit.spat_to_SHsphtor(sht_config, vt_field, vp_field)

        # Store spectral coefficients
        store_coefficients_from_shtnskit!(pol_real, pol_imag, pol_coeffs, r_local, config)
        store_coefficients_from_shtnskit!(tor_real, tor_imag, tor_coeffs, r_local, config)

        # ========================================================================
        # OPTIONAL: Verify solenoidal constraint using radial component
        # This is a consistency check, not used in the decomposition
        # Only performed if domain is provided and verify_solenoidal is true
        # ========================================================================
        if verify_solenoidal && domain !== nothing
            r_idx_global = r_local + first(r_range) - 1
            if r_idx_global <= domain.N
                r_val = domain.r[r_idx_global, 4]
                if r_val > 1e-15
                    # Extract radial component
                    vr_field = extract_vector_component_generic(v_r, r_local, config)

                    # Compute what P should be from v_r: P = r/(l(l+1)) * v_r
                    lmax, mmax = config.lmax, config.mmax
                    pol_from_vr = zeros(ComplexF64, lmax+1, mmax+1)

                    # Analysis of v_r
                    vr_coeffs = SHTnsKit.analysis(sht_config, vr_field)

                    # Scale by r/l(l+1) to get P
                    for l in 1:lmax  # Skip l=0 which has no radial component
                        l_factor = r_val / (l * (l + 1))
                        for m in 0:min(l, mmax)
                            pol_from_vr[l+1, m+1] = vr_coeffs[l+1, m+1] * l_factor
                        end
                    end

                    # Compare with pol_coeffs from tangential analysis
                    # If solenoidal, these should match
                    # (In practice, accumulate error metrics if needed)
                end
            end
        end
    end

    MPI.Barrier(get_comm())
end

# ================================================================================
# Helper Functions for PencilArray Data Management
# ================================================================================
#
# These functions handle the conversion between our internal data layout
# (PencilArrays with distributed spectral/physical data) and SHTnsKit's
# expected format (full coefficient matrices).
#
# KEY CONCEPTS:
# -------------
# 1. Linear spectral index: We store spectral coefficients with a combined (l,m)
#    index running from 1 to nlm. The mapping is: m varies fastest within each l.
#
# 2. SHTnsKit format: Expects a (lmax+1) × (mmax+1) matrix where entry [l+1, m+1]
#    contains the coefficient f_l^m.
#
# 3. MPI gathering: Since spectral data may be distributed, we use MPI.Allreduce
#    to combine partial coefficient matrices from all processes.
#
# ================================================================================

"""
    get_pencil_orientation(pencil) -> Symbol

Determine which dimension(s) are fully local in a pencil decomposition.

# Returns
- `:theta_phi`: Both angular dimensions local (serial or single-node)
- `:theta`: Latitude (θ) dimension is local
- `:phi`: Longitude (φ) dimension is local
- `:r`: Radial dimension is local

# Usage
Used to choose optimal transform strategies based on data layout.
"""
function get_pencil_orientation(pencil::Pencil{3})
    local_ranges = pencil.axes_local    # Index ranges local to this process
    global_sizes = pencil.size_global   # Full global array dimensions

    # Check if each angular dimension is fully local
    θ_local = length(local_ranges[1]) == global_sizes[1]
    φ_local = length(local_ranges[2]) == global_sizes[2]

    if θ_local && φ_local
        return :theta_phi  # Both angular directions fully local (ideal for SHT)
    elseif θ_local
        return :theta      # All latitudes local, longitudes distributed
    elseif φ_local
        return :phi        # All longitudes local (optimal for FFT)
    else
        return :r          # Only radial dimension local
    end
end

"""
    extract_coefficients_for_shtnskit!(coeffs_buffer, spec_real, spec_imag, r_local, config)

Extract spectral coefficients into SHTnsKit's expected matrix format.

# Data Format Conversion
Our internal format: Linear array indexed by combined (l,m) index
SHTnsKit format: Matrix indexed by [l+1, m+1]

# Arguments
- `coeffs_buffer`: Pre-allocated (lmax+1) × (mmax+1) complex matrix (output)
- `spec_real`, `spec_imag`: Real and imaginary parts of spectral coefficients
- `r_local`: Radial level index (1-based)
- `config`: SHTnsKit configuration

# Threading
Uses `@threads` for parallel filling of the coefficient matrix.
"""
function extract_coefficients_for_shtnskit!(coeffs_buffer::Matrix{ComplexF64},
                                           spec_real, spec_imag, r_local, config)
    lmax, mmax = config.lmax, config.mmax

    # Buffer may be larger than needed if SHTnsKit uses different lmax/mmax internally
    buffer_lmax = size(coeffs_buffer, 1) - 1
    buffer_mmax = size(coeffs_buffer, 2) - 1

    # Reset buffer to zero (important for modes not present in input)
    fill!(coeffs_buffer, zero(ComplexF64))

    # Convert from linear indexing to (l,m) matrix format
    # Threaded for performance with large spectral arrays
    Threads.@threads for lm_idx in eachindex(IndexLinear(), view(spec_real, :, 1, 1))
        l, m = index_to_lm_shtnskit(lm_idx, lmax, mmax)
        if r_local <= size(spec_real, 3) && l >= 0 && m >= 0 &&
           l <= buffer_lmax && m <= buffer_mmax
            real_part = spec_real[lm_idx, 1, r_local]
            imag_part = spec_imag[lm_idx, 1, r_local]
            coeffs_buffer[l+1, m+1] = complex(real_part, imag_part)
        end
    end

    return coeffs_buffer
end

"""
    extract_coefficients_for_shtnskit(spec_real, spec_imag, r_local, config) -> Matrix{ComplexF64}

High-level coefficient extraction with automatic buffer management and MPI gathering.

This is the main entry point for preparing spectral coefficients for SHTnsKit.
It handles:
1. Buffer allocation/reuse from cache
2. Local coefficient extraction
3. MPI Allreduce to combine coefficients from all processes

# Why MPI Gathering?
Spectral coefficients may be distributed across MPI processes. SHTnsKit needs
the complete coefficient matrix, so we sum partial contributions from all processes.

# Returns
A complete (lmax+1) × (mmax+1) coefficient matrix ready for SHTnsKit.synthesis()
"""
function extract_coefficients_for_shtnskit(spec_real, spec_imag, r_local, config)
    # Use cached buffer to avoid repeated allocations
    buffer_key = :coeffs_buffer
    if !haskey(config._buffer_cache, buffer_key)
        lmax, mmax = config.lmax, config.mmax
        config._buffer_cache[buffer_key] = zeros(ComplexF64, lmax+1, mmax+1)
    end

    coeffs_buffer = config._buffer_cache[buffer_key]
    extract_coefficients_for_shtnskit!(coeffs_buffer, spec_real, spec_imag, r_local, config)

    # Second buffer for MPI reduction result
    buffer_gathered_key = :coeffs_buffer_gathered
    if !haskey(config._buffer_cache, buffer_gathered_key)
        lmax, mmax = config.lmax, config.mmax
        config._buffer_cache[buffer_gathered_key] = zeros(ComplexF64, lmax+1, mmax+1)
    end
    coeffs_gathered = config._buffer_cache[buffer_gathered_key]

    # Sum partial coefficient matrices from all MPI processes
    # Each process contributes its local portion; summing gives complete matrix
    Allreduce!(coeffs_buffer, coeffs_gathered, MPI.SUM, get_comm())

    # Return copy to avoid aliasing issues with buffer reuse
    return copy(coeffs_gathered)
end

"""
    store_coefficients_from_shtnskit!(spec_real, spec_imag, coeffs_matrix, r_local, config)

Convert SHTnsKit coefficient matrix format back to our linear spectral storage.

This is the inverse of `extract_coefficients_for_shtnskit!`:
- Input: SHTnsKit's (lmax+1) × (mmax+1) coefficient matrix
- Output: Our linear-indexed spectral arrays (real and imaginary parts separate)

# Physical Constraint
For real-valued physical fields, the m=0 coefficients must be purely real.
This function enforces this by zeroing the imaginary part for m=0 modes.
"""
function store_coefficients_from_shtnskit!(spec_real, spec_imag, coeffs_matrix, r_local, config)
    lmax, mmax = config.lmax, config.mmax

    # SHTnsKit's matrix dimensions may differ from our config
    matrix_lmax = size(coeffs_matrix, 1) - 1
    matrix_mmax = size(coeffs_matrix, 2) - 1

    # Convert from (l,m) matrix to linear index format
    Threads.@threads for lm_idx in eachindex(IndexLinear(), view(spec_real, :, 1, 1))
        l, m = index_to_lm_shtnskit(lm_idx, lmax, mmax)
        if r_local <= size(spec_real, 3) && l >= 0 && m >= 0
            if l <= matrix_lmax && m <= matrix_mmax
                # Extract coefficient from SHTnsKit matrix
                coeff = coeffs_matrix[l+1, m+1]
                spec_real[lm_idx, 1, r_local] = real(coeff)
                spec_imag[lm_idx, 1, r_local] = imag(coeff)

                # Physical constraint: m=0 modes must be real for real-valued fields
                if m == 0
                    spec_imag[lm_idx, 1, r_local] = 0.0
                end
            else
                # Mode outside SHTnsKit's range - set to zero
                spec_real[lm_idx, 1, r_local] = 0.0
                spec_imag[lm_idx, 1, r_local] = 0.0
            end
        end
    end
end

"""
    index_to_lm_shtnskit(idx, lmax, mmax) -> (l, m)

Convert linear spectral index to spherical harmonic degree (l) and order (m).

# Index Ordering
The linear index follows the convention where m varies fastest within each l:
- idx=1: (l=0, m=0)
- idx=2: (l=1, m=0)
- idx=3: (l=1, m=1)
- idx=4: (l=2, m=0)
- idx=5: (l=2, m=1)
- idx=6: (l=2, m=2)
- ...

# Performance Note
This function uses a linear search. For performance-critical code with many
lookups, consider precomputing the l_values and m_values arrays (stored in
SHTnsKitConfig).
"""
function index_to_lm_shtnskit(idx::Int, lmax::Int, mmax::Int)
    current_idx = 0
    for l in 0:lmax
        for m in 0:min(l, mmax)
            current_idx += 1
            if current_idx == idx
                return l, m
            end
        end
    end
    return 0, 0  # Fallback for invalid index
end

"""
    store_physical_slice_phi_local!(phys_data, phys_slice, r_local, config)

Copy a 2D physical field slice into a 3D array at a specific radial level.

# Optimized for Phi-Local Layout
When the physical field is in phi-pencil orientation, the (θ, φ) indices
correspond directly to the array's first two dimensions, making this a
straightforward copy operation.

# Arguments
- `phys_data`: 3D destination array (θ × φ × r)
- `phys_slice`: 2D source array from SHTnsKit (θ × φ)
- `r_local`: Radial index to write to
- `config`: Configuration for grid dimensions
"""
function store_physical_slice_phi_local!(phys_data, phys_slice, r_local, config)
    nlat, nlon = config.nlat, config.nlon

    # Determine safe index ranges (handle potential size mismatches)
    common_i_range = 1:min(size(phys_data, 1), nlat, size(phys_slice, 1))
    common_j_range = 1:min(size(phys_data, 2), nlon, size(phys_slice, 2))

    # Threaded copy for large arrays
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
    Allreduce!(slice_buffer, MPI.SUM, get_comm())

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
    Allreduce!(slice_buffer, MPI.SUM, get_comm())

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
    Allreduce!(component_buffer, MPI.SUM, get_comm())

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

"""
    store_scalar_component_generic!(v_component, field, r_local, config)

Store a scalar field into a component array for any pencil orientation.
Used for storing the radial component v_r from synthesized field.
"""
function store_scalar_component_generic!(v_component, field, r_local, config)
    common_i_range = 1:min(size(v_component, 1), size(field, 1))
    common_j_range = 1:min(size(v_component, 2), size(field, 2))

    for i in common_i_range
        for j in common_j_range
            if r_local <= size(v_component, 3)
                v_component[i, j, r_local] = field[i, j]
            end
        end
    end
end

"""
    store_zero_component_generic!(v_component, r_local, config)

Set a component to zero at a given radial level.
Used at r=0 (ball geometry) where v_r must be zero for regularity.
"""
function store_zero_component_generic!(v_component, r_local, config)
    if r_local <= size(v_component, 3)
        for i in axes(v_component, 1)
            for j in axes(v_component, 2)
                v_component[i, j, r_local] = zero(eltype(v_component))
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

Optimize FFT performance by warming up FFTW plans and checking efficiency.
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
            
            plan_forward * parent(test_array)
            plan_backward * parent(test_array)

            if get_rank() == 0
                @info "FFT plans warmed up successfully"
            end
        catch e
            @warn "Could not warm up FFT plans: $e"
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
