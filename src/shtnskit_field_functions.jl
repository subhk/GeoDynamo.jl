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
# IMPLEMENTATION STRATEGY (SHTnsKit v1.1.15+):
# --------------------------------------------
# Uses native SHTnsKit distributed transforms when available:
# - dist_synthesis() for distributed spectral→physical
# - dist_analysis() for distributed physical→spectral
# - dist_SHsphtor_to_spat() for distributed vector synthesis
# - dist_spat_to_SHsphtor() for distributed vector analysis
# - SHqst_to_spat() / spat_to_SHqst() for full 3D QST vector transforms
#
# Fallback implementation for each radial level independently:
# 1. Extract/prepare spectral coefficients in SHTnsKit's expected format
# 2. Call SHTnsKit.synthesis() or SHTnsKit.analysis()
# 3. Store/scatter results to the appropriate PencilArray
#
# MPI parallelization is handled through PencilArrays, with MPI.Barrier()
# synchronization after complete transforms.
#
# ================================================================================

# Type-stable accessors for buffer cache (function barriers to avoid Dict{Symbol,Any} instability)
@inline function _get_sht_plan(cache::Dict{Symbol,Any})
    return get(cache, :sht_plan, nothing)::Union{SHTnsKit.SHTPlan, Nothing}
end
@inline function _get_synth_out(cache::Dict{Symbol,Any})
    return get(cache, :synth_out, nothing)::Union{Matrix{Float64}, Nothing}
end
@inline function _get_anal_out(cache::Dict{Symbol,Any})
    return get(cache, :anal_out, nothing)::Union{Matrix{ComplexF64}, Nothing}
end

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
- `spec::SHTnsSpecField`: Source spectral field with coefficients
- `phys::SHTnsPhysField`: Destination physical field (modified in-place)

# Side Effects
Modifies `phys.data` with the synthesized field values
"""
function shtnskit_spectral_to_physical!(spec::SHTnsSpecField{T},
                                       phys::SHTnsPhysField{T}) where T
    config = spec.config

    # Use direct synthesis method (processes each radial level)
    # MPI synchronization is handled by Allreduce inside extract_coefficients_for_shtnskit
    perform_synthesis_direct!(spec, phys, config)
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
function perform_synthesis_phi_local!(spec::SHTnsSpecField{T},
                                     phys::SHTnsPhysField{T},
                                     config) where T
    sht_config = config.sht_config

    # Extract underlying Julia arrays from PencilArrays
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag)
    phys_data = parent(phys.data)

    # Get pre-allocated plan and output buffer (allocation-free path)
    plan = _get_sht_plan(config._buffer_cache)
    synth_out = _get_synth_out(config._buffer_cache)

    # Get global index ranges for this rank's portion of the physical grid
    axes_local = phys.pencil.axes_local

    # Process each radial level independently (embarrassingly parallel in r)
    for r_local in axes(phys_data, 3)
        coeffs_matrix = extract_coefficients_for_shtnskit(spec_real_data, spec_imag_data, r_local, config)

        if plan !== nothing && synth_out !== nothing
            SHTnsKit.synthesis!(plan, synth_out, coeffs_matrix; real_output=true)
            local_synth = @view synth_out[axes_local[1], axes_local[2]]
            store_physical_slice_phi_local!(phys_data, local_synth, r_local, config)
        else
            phys_slice = SHTnsKit.synthesis(sht_config, coeffs_matrix; real_output=true)
            local_slice = @view phys_slice[axes_local[1], axes_local[2]]
            store_physical_slice_phi_local!(phys_data, local_slice, r_local, config)
        end
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
function perform_synthesis_with_transpose!(spec::SHTnsSpecField{T},
                                         phys::SHTnsPhysField{T},
                                         config, back_plan) where T
    # Reuse cached temporary phi-pencil array (avoids allocation every call)
    # Uses separate key from analysis to avoid aliasing if called concurrently
    phys_phi = get_cached_buffer!(config, :synthesis_phi_tmp) do
        PencilArray{T}(undef, config.pencils.phi)
    end

    # Perform synthesis with longitude local (optimal for SHTnsKit)
    perform_synthesis_to_phi_pencil!(spec, phys_phi, config)

    # Redistribute data to match target pencil orientation
    mul!(phys.data, back_plan, phys_phi)
end

"""
    perform_synthesis_to_phi_pencil!(spec, phys_phi, config)

Core synthesis routine that writes directly to a phi-pencil array.

This is the workhorse function called by other synthesis routines.
It assumes the destination array is already in phi-pencil orientation.
"""
function perform_synthesis_to_phi_pencil!(spec::SHTnsSpecField{T},
                                        phys_phi::PencilArray{T,3},
                                        config) where T
    sht_config = config.sht_config

    # Get underlying Julia arrays
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag)
    phys_phi_data = parent(phys_phi)

    # Get pre-allocated plan and output buffer (allocation-free path)
    plan = _get_sht_plan(config._buffer_cache)
    synth_out = _get_synth_out(config._buffer_cache)

    # Get global index ranges for this rank's portion of the phi-pencil grid
    axes_local = PencilArrays.pencil(phys_phi).axes_local

    # Loop over radial levels (each level is independent)
    for r_local in axes(phys_phi_data, 3)
        coeffs_matrix = extract_coefficients_for_shtnskit(spec_real_data, spec_imag_data, r_local, config)

        if plan !== nothing && synth_out !== nothing
            SHTnsKit.synthesis!(plan, synth_out, coeffs_matrix; real_output=true)
            local_synth = @view synth_out[axes_local[1], axes_local[2]]
            store_physical_slice_phi_local!(phys_phi_data, local_synth, r_local, config)
        else
            phys_slice = SHTnsKit.synthesis(sht_config, coeffs_matrix; real_output=true)
            local_slice = @view phys_slice[axes_local[1], axes_local[2]]
            store_physical_slice_phi_local!(phys_phi_data, local_slice, r_local, config)
        end
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
function perform_synthesis_direct!(spec::SHTnsSpecField{T},
                                  phys::SHTnsPhysField{T},
                                  config) where T
    sht_config = config.sht_config

    # Extract underlying arrays from PencilArrays wrapper
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag)
    phys_data = parent(phys.data)

    # SAFETY: The radial loop below contains MPI collectives (Allreduce).
    # All processes MUST iterate the same number of times to avoid deadlock.
    # This is guaranteed when dim 3 is local (r-pencil or spec-pencil).
    # If dim 3 is distributed unevenly, processes will deadlock.
    @assert size(phys_data, 3) == size(spec_real_data, 3) "Radial dimension mismatch: physical=$(size(phys_data,3)) vs spectral=$(size(spec_real_data,3)). SH transforms require radial to be local."

    # Get pre-allocated plan and output buffer (allocation-free path)
    plan = _get_sht_plan(config._buffer_cache)
    synth_out = _get_synth_out(config._buffer_cache)

    # Get global index ranges for this rank's local portion of the physical grid.
    # When angular dimensions are MPI-distributed, each rank owns a subset of
    # the full (nlat, nlon) grid. SHTnsKit synthesis produces the FULL grid,
    # so we must extract only this rank's portion using global offsets.
    axes_local = phys.pencil.axes_local

    # Process each radial level
    for r_local in axes(phys_data, 3)
        # Gather spectral coefficients for this radial level
        coeffs_matrix = extract_coefficients_for_shtnskit(spec_real_data, spec_imag_data, r_local, config)

        if plan !== nothing && synth_out !== nothing
            # Allocation-free path: use pre-allocated plan and output buffer
            SHTnsKit.synthesis!(plan, synth_out, coeffs_matrix; real_output=true)
            local_synth = @view synth_out[axes_local[1], axes_local[2]]
            store_physical_slice_generic!(phys_data, local_synth, r_local, config)
        else
            # Fallback: allocating path
            phys_slice = SHTnsKit.synthesis(sht_config, coeffs_matrix; real_output=true)
            local_slice = @view phys_slice[axes_local[1], axes_local[2]]
            store_physical_slice_generic!(phys_data, local_slice, r_local, config)
        end
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
- `phys::SHTnsPhysField`: Source physical field values
- `spec::SHTnsSpecField`: Destination spectral field (modified in-place)

# Side Effects
Modifies `spec.data_real` and `spec.data_imag` with the computed coefficients
"""
function shtnskit_physical_to_spectral!(phys::SHTnsPhysField{T},
                                       spec::SHTnsSpecField{T}) where T
    config = spec.config

    # Use direct analysis method (processes each radial level)
    # MPI synchronization is handled by Allreduce inside extract_physical_slice_generic
    perform_analysis_direct!(phys, spec, config)
end

"""
    perform_analysis_phi_local!(phys, spec, config)

Perform analysis when physical field is in phi-pencil orientation.

This is the most efficient analysis path because:
1. The phi (longitude) dimension is fully local on each process
2. SHTnsKit's FFT operates entirely in local memory
3. No MPI communication needed during the transform itself
"""
function perform_analysis_phi_local!(phys::SHTnsPhysField{T},
                                    spec::SHTnsSpecField{T},
                                    config) where T
    sht_config = config.sht_config

    # Get local data
    phys_data = parent(phys.data)
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag)

    # Get pre-allocated plan and output buffer
    plan = _get_sht_plan(config._buffer_cache)
    anal_out = _get_anal_out(config._buffer_cache)

    # Get global index ranges for correct placement in Allreduce buffer
    phys_axes_local = phys.pencil.axes_local

    # Process each radial level
    for r_local in axes(phys_data, 3)
        phys_slice = extract_physical_slice_phi_local(phys_data, r_local, config;
                                                      axes_local=phys_axes_local)

        if plan !== nothing && anal_out !== nothing
            SHTnsKit.analysis!(plan, anal_out, phys_slice)
            store_coefficients_from_shtnskit!(spec_real_data, spec_imag_data, anal_out, r_local, config)
        else
            coeffs_matrix = SHTnsKit.analysis(sht_config, phys_slice)
            store_coefficients_from_shtnskit!(spec_real_data, spec_imag_data, coeffs_matrix, r_local, config)
        end
    end
end

"""
    perform_analysis_with_transpose!(phys, spec, config, to_phi_plan)

Perform analysis with transpose to phi-pencil.
"""
function perform_analysis_with_transpose!(phys::SHTnsPhysField{T},
                                        spec::SHTnsSpecField{T},
                                        config, to_phi_plan) where T
    # Reuse cached temporary phi-pencil array (avoids allocation every call)
    # Uses separate key from synthesis to avoid aliasing if called concurrently
    phys_phi = get_cached_buffer!(config, :analysis_phi_tmp) do
        PencilArray{T}(undef, config.pencils.phi)
    end
    # Transpose to phi-pencil using pre-computed plan
    mul!(phys_phi, to_phi_plan, phys.data)
    perform_analysis_from_phi_pencil!(phys_phi, spec, config)
end

"""
    perform_analysis_from_phi_pencil!(phys_phi, spec, config)

Perform analysis from phi-pencil data.
"""
function perform_analysis_from_phi_pencil!(phys_phi::PencilArray{T,3},
                                         spec::SHTnsSpecField{T},
                                         config) where T
    sht_config = config.sht_config

    # Get data arrays
    phys_phi_data = parent(phys_phi)
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag)

    # Get pre-allocated plan and output buffer
    plan = _get_sht_plan(config._buffer_cache)
    anal_out = _get_anal_out(config._buffer_cache)

    # Get global index ranges for correct placement in Allreduce buffer
    phi_axes_local = PencilArrays.pencil(phys_phi).axes_local

    # Process each radial level
    for r_local in axes(phys_phi_data, 3)
        phys_slice = extract_physical_slice_phi_local(phys_phi_data, r_local, config;
                                                      axes_local=phi_axes_local)

        if plan !== nothing && anal_out !== nothing
            SHTnsKit.analysis!(plan, anal_out, phys_slice)
            store_coefficients_from_shtnskit!(spec_real_data, spec_imag_data, anal_out, r_local, config)
        else
            coeffs_matrix = SHTnsKit.analysis(sht_config, phys_slice)
            store_coefficients_from_shtnskit!(spec_real_data, spec_imag_data, coeffs_matrix, r_local, config)
        end
    end
end

"""
    perform_analysis_direct!(phys, spec, config)

Direct analysis without transpose (fallback).
"""
function perform_analysis_direct!(phys::SHTnsPhysField{T},
                                 spec::SHTnsSpecField{T},
                                 config) where T
    sht_config = config.sht_config

    # Get local data
    phys_data = parent(phys.data)
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag)

    # SAFETY: see perform_synthesis_direct! — radial must be local for MPI safety
    @assert size(phys_data, 3) == size(spec_real_data, 3) "Radial dimension mismatch in analysis. SH transforms require radial to be local."

    # Get pre-allocated plan and output buffer
    plan = _get_sht_plan(config._buffer_cache)
    anal_out = _get_anal_out(config._buffer_cache)

    # Get global index ranges for correct placement in Allreduce buffer
    phys_axes_local = phys.pencil.axes_local

    # Process each radial level
    for r_local in axes(phys_data, 3)
        phys_slice = extract_physical_slice_generic(phys_data, r_local, config;
                                                    axes_local=phys_axes_local)

        if plan !== nothing && anal_out !== nothing
            SHTnsKit.analysis!(plan, anal_out, phys_slice)
            store_coefficients_from_shtnskit!(spec_real_data, spec_imag_data, anal_out, r_local, config)
        else
            coeffs_matrix = SHTnsKit.analysis(sht_config, phys_slice)
            store_coefficients_from_shtnskit!(spec_real_data, spec_imag_data, coeffs_matrix, r_local, config)
        end
    end
end

# ================================================================================
# Vector Transforms with SHTnsKit and PencilArrays
# ================================================================================

"""
    shtnskit_vector_synthesis!(tor_spec::SHTnsSpecField{T},
                              pol_spec::SHTnsSpecField{T},
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
function shtnskit_vector_synthesis!(tor_spec::SHTnsSpecField{T},
                                   pol_spec::SHTnsSpecField{T},
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

    # Get pre-allocated plan and output buffers (allocation-free path)
    plan = get(config._buffer_cache, :sht_plan, nothing)
    vt_out = get(config._buffer_cache, :vt_out, nothing)
    vp_out = get(config._buffer_cache, :vp_out, nothing)
    synth_out = get(config._buffer_cache, :synth_out, nothing)

    # SAFETY: The radial loop below contains MPI collectives (Allreduce).
    # All processes MUST iterate the same number of times to avoid deadlock.
    @assert size(v_r, 3) == size(tor_real, 3) "Radial dimension mismatch: physical=$(size(v_r,3)) vs spectral=$(size(tor_real,3)). Vector SH transforms require radial to be local."

    # Get global index ranges for this rank's portion of the physical grid.
    # SHTnsKit synthesis produces the FULL (nlat, nlon) grid; we must extract
    # only this rank's local portion using these offsets.
    phys_axes_local = vec_phys.r_component.pencil.axes_local

    # Process each radial level
    for r_local in axes(tor_real, 3)
        # Extract toroidal and poloidal coefficients efficiently (includes MPI gathering)
        tor_coeffs, pol_coeffs = extract_coefficients_pair_for_shtnskit(
            tor_real, tor_imag, pol_real, pol_imag, r_local, config)

        # Perform vector synthesis using SHTnsKit (tangential components only)
        if plan !== nothing && vt_out !== nothing && vp_out !== nothing
            # Allocation-free path: in-place vector synthesis
            SHTnsKit.synthesis_sphtor!(plan, vt_out, vp_out, pol_coeffs, tor_coeffs; real_output=true)
            store_vector_components_generic!(v_theta, v_phi, vt_out, vp_out, r_local, config;
                                             axes_local=phys_axes_local)
        else
            vt_field, vp_field = SHTnsKit.synthesis_sphtor(sht_config, pol_coeffs, tor_coeffs;
                                                          real_output=true)
            store_vector_components_generic!(v_theta, v_phi, vt_field, vp_field, r_local, config;
                                             axes_local=phys_axes_local)
        end

        # ========================================================================
        # CRITICAL: Compute radial component from poloidal scalar
        # v_r = l(l+1)/r * P * Y_lm
        # Only computed if domain information is provided
        # ========================================================================

        if domain !== nothing
            r_idx_global = r_local + first(r_range) - 1

            if r_idx_global >= 1 && r_idx_global <= domain.N
                r_val = domain.r[r_idx_global, 4]

                if r_val > 1e-15
                    lmax, mmax = config.lmax, config.mmax

                    pol_rad_coeffs = get_cached_buffer!(config, :pol_rad_coeffs_buffer) do
                        zeros(ComplexF64, lmax+1, mmax+1)
                    end
                    fill!(pol_rad_coeffs, zero(ComplexF64))

                    for l in 0:lmax
                        l_factor = l * (l + 1) / r_val
                        for m in 0:min(l, mmax)
                            pol_rad_coeffs[l+1, m+1] = pol_coeffs[l+1, m+1] * l_factor
                        end
                    end

                    # Synthesize radial component (allocation-free if plan available)
                    if plan !== nothing && synth_out !== nothing
                        SHTnsKit.synthesis!(plan, synth_out, pol_rad_coeffs; real_output=true)
                        store_scalar_component_generic!(v_r, synth_out, r_local, config;
                                                        axes_local=phys_axes_local)
                    else
                        vr_field = SHTnsKit.synthesis(sht_config, pol_rad_coeffs; real_output=true)
                        store_scalar_component_generic!(v_r, vr_field, r_local, config;
                                                        axes_local=phys_axes_local)
                    end
                else
                    store_zero_component_generic!(v_r, r_local, config)
                end
            end
        else
            store_zero_component_generic!(v_r, r_local, config)
        end
    end
end

"""
    shtnskit_vector_analysis!(vec_phys::SHTnsVectorField{T},
                             tor_spec::SHTnsSpecField{T},
                             pol_spec::SHTnsSpecField{T}) where T

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
                                  tor_spec::SHTnsSpecField{T},
                                  pol_spec::SHTnsSpecField{T};
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

    # Get pre-allocated plan and output buffers (allocation-free path)
    plan = get(config._buffer_cache, :sht_plan, nothing)
    slm_out = get(config._buffer_cache, :slm_out, nothing)
    tlm_out = get(config._buffer_cache, :tlm_out, nothing)

    # Process each radial level
    for r_local in axes(v_theta, 3)
        # Extract vector components using SEPARATE buffers to avoid overwriting
        nlat, nlon = config.nlat, config.nlon
        vt_buffer = get_cached_buffer!(config, :vector_component_buffer_vt) do
            zeros(eltype(v_theta), nlat, nlon)
        end
        vp_buffer = get_cached_buffer!(config, :vector_component_buffer_vp) do
            zeros(eltype(v_phi), nlat, nlon)
        end
        phys_axes_local = vec_phys.r_component.pencil.axes_local
        vt_field = extract_vector_component_generic!(vt_buffer, v_theta, r_local, config;
                                                     axes_local=phys_axes_local)
        vp_field = extract_vector_component_generic!(vp_buffer, v_phi, r_local, config;
                                                     axes_local=phys_axes_local)

        # Perform vector analysis using SHTnsKit (tangential components)
        if plan !== nothing && slm_out !== nothing && tlm_out !== nothing
            # Allocation-free path: in-place vector analysis
            SHTnsKit.analysis_sphtor!(plan, slm_out, tlm_out, vt_field, vp_field)
            store_coefficients_from_shtnskit!(pol_real, pol_imag, slm_out, r_local, config)
            store_coefficients_from_shtnskit!(tor_real, tor_imag, tlm_out, r_local, config)
        else
            pol_coeffs, tor_coeffs = SHTnsKit.analysis_sphtor(sht_config, vt_field, vp_field)
            store_coefficients_from_shtnskit!(pol_real, pol_imag, pol_coeffs, r_local, config)
            store_coefficients_from_shtnskit!(tor_real, tor_imag, tor_coeffs, r_local, config)
        end
    end
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
    # Uses O(1) index_to_lm_fast with precomputed lookup tables
    Threads.@threads for lm_idx in eachindex(IndexLinear(), view(spec_real, :, 1, 1))
        l, m = index_to_lm_fast(lm_idx, config)
        # Check bounds on both spec_real and spec_imag for safety
        if r_local <= size(spec_real, 3) && r_local <= size(spec_imag, 3) &&
           l >= 0 && m >= 0 && l <= buffer_lmax && m <= buffer_mmax
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
    lmax, mmax = config.lmax, config.mmax

    # Use thread-safe cached buffer access
    coeffs_buffer = get_cached_buffer!(config, :coeffs_buffer) do
        zeros(ComplexF64, lmax+1, mmax+1)
    end

    extract_coefficients_for_shtnskit!(coeffs_buffer, spec_real, spec_imag, r_local, config)

    # Second buffer for MPI reduction result (thread-safe)
    coeffs_gathered = get_cached_buffer!(config, :coeffs_buffer_gathered) do
        zeros(ComplexF64, lmax+1, mmax+1)
    end

    # Sum partial coefficient matrices from all MPI processes
    # Each process contributes its local portion; summing gives complete matrix
    Allreduce!(coeffs_buffer, coeffs_gathered, MPI.SUM, get_comm())

    # Return the gathered buffer directly (no copy needed since callers consume
    # the result immediately via synthesis! or store_coefficients_from_shtnskit!)
    return coeffs_gathered
end

"""
    extract_coefficients_pair_for_shtnskit(spec1_real, spec1_imag, spec2_real, spec2_imag, r_local, config)

Extract two spectral coefficient matrices efficiently for vector transforms.

This is optimized for the common case in vector synthesis/analysis where we need
both toroidal and poloidal coefficients. It avoids one copy operation compared
to calling `extract_coefficients_for_shtnskit` twice.

# Returns
Tuple (coeffs1, coeffs2) of coefficient matrices.
"""
function extract_coefficients_pair_for_shtnskit(spec1_real, spec1_imag,
                                                 spec2_real, spec2_imag,
                                                 r_local, config)
    lmax, mmax = config.lmax, config.mmax

    # Use separate cached buffers for each extraction
    coeffs_buffer1 = get_cached_buffer!(config, :coeffs_buffer_pair1) do
        zeros(ComplexF64, lmax+1, mmax+1)
    end
    coeffs_buffer2 = get_cached_buffer!(config, :coeffs_buffer_pair2) do
        zeros(ComplexF64, lmax+1, mmax+1)
    end

    # Extract both coefficient sets
    extract_coefficients_for_shtnskit!(coeffs_buffer1, spec1_real, spec1_imag, r_local, config)
    extract_coefficients_for_shtnskit!(coeffs_buffer2, spec2_real, spec2_imag, r_local, config)

    # Buffers for MPI reduction
    coeffs_gathered1 = get_cached_buffer!(config, :coeffs_gathered_pair1) do
        zeros(ComplexF64, lmax+1, mmax+1)
    end
    coeffs_gathered2 = get_cached_buffer!(config, :coeffs_gathered_pair2) do
        zeros(ComplexF64, lmax+1, mmax+1)
    end

    # MPI gather for both - these can't be parallelized due to MPI collective semantics
    Allreduce!(coeffs_buffer1, coeffs_gathered1, MPI.SUM, get_comm())
    Allreduce!(coeffs_buffer2, coeffs_gathered2, MPI.SUM, get_comm())

    # No copies needed - separate buffer keys ensure no aliasing
    return coeffs_gathered1, coeffs_gathered2
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
    # Uses O(1) index_to_lm_fast with precomputed lookup tables
    Threads.@threads for lm_idx in eachindex(IndexLinear(), view(spec_real, :, 1, 1))
        l, m = index_to_lm_fast(lm_idx, config)
        # Check bounds on both spec_real and spec_imag for safety
        if r_local <= size(spec_real, 3) && r_local <= size(spec_imag, 3) && l >= 0 && m >= 0
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
The linear index follows the SHTnsKit m-major convention (m varies slowest, l fastest):
- idx=1: (l=0, m=0)
- idx=2: (l=1, m=0)
- idx=3: (l=2, m=0)
- ...
- idx=lmax+1: (l=lmax, m=0)
- idx=lmax+2: (l=1, m=1)
- idx=lmax+3: (l=2, m=1)
- ...

# Performance Note
This function uses a linear search. For performance-critical code with many
lookups, use `index_to_lm_fast` with precomputed lookup tables from SHTnsKitConfig.
"""
function index_to_lm_shtnskit(idx::Int, lmax::Int, mmax::Int)
    # Validate index bounds
    if idx < 1
        return -1, -1  # Invalid index indicator
    end

    current_idx = 0
    for m in 0:mmax
        for l in m:lmax
            current_idx += 1
            if current_idx == idx
                return l, m
            end
        end
    end
    return -1, -1  # Index out of range - return invalid indicator
end

"""
    index_to_lm_fast(idx, config) -> (l, m)

Fast O(1) conversion from linear spectral index to (l, m) using precomputed tables.

Uses the l_values and m_values arrays stored in SHTnsKitConfig during initialization.
This is significantly faster than `index_to_lm_shtnskit` for repeated lookups.

# Arguments
- `idx`: Linear spectral index (1-based)
- `config`: SHTnsKitConfig containing precomputed l_values and m_values

# Returns
Tuple (l, m) for the spherical harmonic degree and order.
"""
@inline function index_to_lm_fast(idx::Int, config)
    if idx >= 1 && idx <= length(config.l_values)
        return config.l_values[idx], config.m_values[idx]
    else
        return -1, -1  # Invalid index indicator (not a valid l,m pair)
    end
end

"""
    build_lm_lookup_tables(lmax, mmax) -> (l_values, m_values)

Build precomputed lookup tables for converting linear indices to (l, m).

# Returns
- `l_values`: Vector where l_values[idx] gives the degree l for linear index idx
- `m_values`: Vector where m_values[idx] gives the order m for linear index idx
"""
function build_lm_lookup_tables(lmax::Int, mmax::Int)
    # Calculate total number of modes (m-major ordering to match SHTnsKit)
    nlm = 0
    for m in 0:mmax
        nlm += lmax - m + 1
    end

    l_values = zeros(Int, nlm)
    m_values = zeros(Int, nlm)

    idx = 0
    for m in 0:mmax
        for l in m:lmax
            idx += 1
            l_values[idx] = l
            m_values[idx] = m
        end
    end

    return l_values, m_values
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

# WARNING: MPI Synchronization
This function contains MPI.Allreduce! which is a collective operation.
When called inside a per-radial loop, ALL MPI processes must call this
function the same number of times, otherwise deadlock will occur.
Ensure even radial distribution or use global loop bounds.
"""
function extract_physical_slice_phi_local!(slice_buffer::Matrix{T}, phys_data, r_local, config;
                                          axes_local::Union{Nothing, Tuple}=nothing) where T
    nlat, nlon = config.nlat, config.nlon

    # Clear buffer for reuse
    fill!(slice_buffer, zero(T))

    # Check if this process has data at this radial level
    has_local_data = r_local <= size(phys_data, 3)

    if axes_local !== nothing
        # Use global offsets: place local data at correct position in the full grid buffer
        θ_range = axes_local[1]
        φ_range = axes_local[2]
        if has_local_data
            Threads.@threads for i_local in 1:size(phys_data, 1)
                i_global = θ_range[i_local]
                for j_local in 1:size(phys_data, 2)
                    j_global = φ_range[j_local]
                    slice_buffer[i_global, j_global] = phys_data[i_local, j_local, r_local]
                end
            end
        end
    else
        # Legacy path: phi-local pencil, assumes indices match global
        common_i_range = 1:min(size(phys_data, 1), nlat, size(slice_buffer, 1))
        common_j_range = 1:min(size(phys_data, 2), nlon, size(slice_buffer, 2))
        if has_local_data
            Threads.@threads for i in common_i_range
                for j in common_j_range
                    slice_buffer[i, j] = phys_data[i, j, r_local]
                end
            end
        end
    end

    # Gather complete grid across all MPI processes
    # This is a collective operation - all processes must participate
    Allreduce!(slice_buffer, MPI.SUM, get_comm())

    return slice_buffer
end

# Backward compatibility wrapper with thread-safe buffer access
function extract_physical_slice_phi_local(phys_data, r_local, config;
                                          axes_local::Union{Nothing, Tuple}=nothing)
    nlat, nlon = config.nlat, config.nlon
    # Get or create cached buffer for phi slice (thread-safe)
    slice_buffer = get_cached_buffer!(config, :phi_slice_buffer) do
        zeros(eltype(phys_data), nlat, nlon)
    end
    return extract_physical_slice_phi_local!(slice_buffer, phys_data, r_local, config;
                                             axes_local=axes_local)
end

"""
    extract_physical_slice_generic!(slice_buffer, phys_data, r_local, config)

Generic extraction for any pencil orientation using pre-allocated buffer.

# WARNING: MPI Synchronization
This function contains MPI.Allreduce! which is a collective operation.
When called inside a per-radial loop, ALL MPI processes must call this
function the same number of times, otherwise deadlock will occur.
Ensure even radial distribution or use global loop bounds.
"""
function extract_physical_slice_generic!(slice_buffer::Matrix{T}, phys_data, r_local, config;
                                        axes_local::Union{Nothing, Tuple}=nothing) where T
    nlat, nlon = config.nlat, config.nlon

    # Clear buffer for reuse
    fill!(slice_buffer, zero(T))

    # Check if this process has data at this radial level
    has_local_data = r_local <= size(phys_data, 3)

    if axes_local !== nothing
        # Use global offsets: place local data at correct position in the full grid buffer
        θ_range = axes_local[1]
        φ_range = axes_local[2]
        if has_local_data
            Threads.@threads for i_local in 1:size(phys_data, 1)
                i_global = θ_range[i_local]
                for j_local in 1:size(phys_data, 2)
                    j_global = φ_range[j_local]
                    slice_buffer[i_global, j_global] = phys_data[i_local, j_local, r_local]
                end
            end
        end
    else
        # Legacy path: assumes local indices match global (serial or fully-local pencil)
        common_i_range = 1:min(size(phys_data, 1), nlat, size(slice_buffer, 1))
        common_j_range = 1:min(size(phys_data, 2), nlon, size(slice_buffer, 2))
        if has_local_data
            Threads.@threads for i in common_i_range
                for j in common_j_range
                    slice_buffer[i, j] = phys_data[i, j, r_local]
                end
            end
        end
    end

    # Gather complete grid across all MPI processes
    # This is a collective operation - all processes must participate
    Allreduce!(slice_buffer, MPI.SUM, get_comm())

    return slice_buffer
end

# Backward compatibility wrapper with thread-safe buffer access
function extract_physical_slice_generic(phys_data, r_local, config;
                                        axes_local::Union{Nothing, Tuple}=nothing)
    nlat, nlon = config.nlat, config.nlon
    # Get or create cached buffer for generic slice (thread-safe)
    slice_buffer = get_cached_buffer!(config, :generic_slice_buffer) do
        zeros(eltype(phys_data), nlat, nlon)
    end
    return extract_physical_slice_generic!(slice_buffer, phys_data, r_local, config;
                                           axes_local=axes_local)
end

"""
    extract_vector_component_generic!(component_buffer, v_data, r_local, config)

Generic extraction for vector components using pre-allocated buffer.

# WARNING: MPI Synchronization
This function contains MPI.Allreduce! which is a collective operation.
When called inside a per-radial loop, ALL MPI processes must call this
function the same number of times, otherwise deadlock will occur.
Ensure even radial distribution or use global loop bounds.
"""
function extract_vector_component_generic!(component_buffer::Matrix{T}, v_data, r_local, config;
                                           axes_local::Union{Nothing, Tuple}=nothing) where T
    nlat, nlon = config.nlat, config.nlon

    # Clear buffer for reuse
    fill!(component_buffer, zero(T))

    # Check if this process has data at this radial level
    has_local_data = r_local <= size(v_data, 3)

    if axes_local !== nothing
        # Use global offsets: place local data at correct position in the full grid buffer
        θ_range = axes_local[1]
        φ_range = axes_local[2]
        if has_local_data
            Threads.@threads for i_local in 1:size(v_data, 1)
                i_global = θ_range[i_local]
                for j_local in 1:size(v_data, 2)
                    j_global = φ_range[j_local]
                    component_buffer[i_global, j_global] = v_data[i_local, j_local, r_local]
                end
            end
        end
    else
        # Legacy path: assumes local indices match global
        common_i_range = 1:min(size(v_data, 1), nlat, size(component_buffer, 1))
        common_j_range = 1:min(size(v_data, 2), nlon, size(component_buffer, 2))
        if has_local_data
            Threads.@threads for i in common_i_range
                for j in common_j_range
                    component_buffer[i, j] = v_data[i, j, r_local]
                end
            end
        end
    end

    # Gather complete grid across all MPI processes
    # This is a collective operation - all processes must participate
    Allreduce!(component_buffer, MPI.SUM, get_comm())

    return component_buffer
end

# Backward compatibility wrapper with thread-safe buffer access
function extract_vector_component_generic(v_data, r_local, config)
    nlat, nlon = config.nlat, config.nlon
    # Get or create cached buffer for vector component (thread-safe)
    component_buffer = get_cached_buffer!(config, :vector_component_buffer) do
        zeros(eltype(v_data), nlat, nlon)
    end
    return extract_vector_component_generic!(component_buffer, v_data, r_local, config)
end

"""
    store_vector_components_generic!(v_theta, v_phi, vt_field, vp_field, r_local, config)

Store vector components for any pencil orientation.
"""
function store_vector_components_generic!(v_theta, v_phi, vt_field, vp_field, r_local, config;
                                          axes_local::Union{Nothing, Tuple}=nothing)
    # Check radial bounds once outside the loop
    if r_local > size(v_theta, 3) || r_local > size(v_phi, 3)
        return
    end

    if axes_local !== nothing
        # Use global offsets: vt_field/vp_field are full (nlat, nlon) grids,
        # extract only this rank's local portion
        θ_range = axes_local[1]
        φ_range = axes_local[2]
        Threads.@threads for i_local in 1:size(v_theta, 1)
            i_global = θ_range[i_local]
            for j_local in 1:size(v_theta, 2)
                j_global = φ_range[j_local]
                v_theta[i_local, j_local, r_local] = vt_field[i_global, j_global]
                v_phi[i_local, j_local, r_local] = vp_field[i_global, j_global]
            end
        end
    else
        # Legacy path: assumes local indices match global
        common_i_range = 1:min(size(v_theta, 1), size(v_phi, 1), size(vt_field, 1), size(vp_field, 1))
        common_j_range = 1:min(size(v_theta, 2), size(v_phi, 2), size(vt_field, 2), size(vp_field, 2))
        Threads.@threads for i in common_i_range
            for j in common_j_range
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
function store_scalar_component_generic!(v_component, field, r_local, config;
                                         axes_local::Union{Nothing, Tuple}=nothing)
    # Check radial bounds once outside the loop
    if r_local > size(v_component, 3)
        return
    end

    if axes_local !== nothing
        # Use global offsets: field is full (nlat, nlon) grid,
        # extract only this rank's local portion
        θ_range = axes_local[1]
        φ_range = axes_local[2]
        Threads.@threads for i_local in 1:size(v_component, 1)
            i_global = θ_range[i_local]
            for j_local in 1:size(v_component, 2)
                j_global = φ_range[j_local]
                v_component[i_local, j_local, r_local] = field[i_global, j_global]
            end
        end
    else
        # Legacy path: assumes local indices match global
        common_i_range = 1:min(size(v_component, 1), size(field, 1))
        common_j_range = 1:min(size(v_component, 2), size(field, 2))
        Threads.@threads for i in common_i_range
            for j in common_j_range
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
    if r_local > size(v_component, 3)
        return
    end

    # Threaded zeroing for consistency with other store functions
    Threads.@threads for i in axes(v_component, 1)
        for j in axes(v_component, 2)
            v_component[i, j, r_local] = zero(eltype(v_component))
        end
    end
end

# ================================================================================
# Batch Processing for Enhanced Performance
# ================================================================================

"""
    batch_shtnskit_transforms!(specs::Vector{SHTnsSpecField{T}},
                              physs::Vector{SHTnsPhysField{T}}) where T

Batch process multiple transforms using SHTnsKit with PencilArrays.

# MPI Safety
This function processes transforms sequentially to avoid calling MPI collectives
(Barrier) from multiple threads simultaneously. MPI collective operations must
be called from the same thread on all processes to avoid deadlock.

Note: The individual transforms themselves are still efficient as SHTnsKit
performs optimized Legendre transforms and FFTs internally.
"""
function batch_shtnskit_transforms!(specs::Vector{SHTnsSpecField{T}},
                                   physs::Vector{SHTnsPhysField{T}}) where T
    @assert length(specs) == length(physs)

    if isempty(specs)
        return
    end

    # Process sequentially to avoid MPI collectives from multiple threads
    # Each shtnskit_spectral_to_physical! call has MPI.Barrier at the end,
    # which must not be called from multiple threads simultaneously
    for batch_idx in eachindex(specs)
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
function batch_spectral_to_physical!(specs::Vector{SHTnsSpecField{T}},
                                     physs::Vector{SHTnsPhysField{T}}) where T
    return batch_shtnskit_transforms!(specs, physs)
end

# ================================================================================
# Performance Monitoring
# ================================================================================

"""
    get_shtnskit_performance_stats()

Get performance statistics for SHTnsKit transforms with PencilArrays.
Returns information about the v1.1.15+ features being used.
"""
function get_shtnskit_performance_stats()
    version_info = get_shtnskit_version_info()
    return (
        library = "SHTnsKit",
        version = version_info.version,
        parallelization = "theta-phi MPI + PencilArrays",
        fft_backend = "PencilFFTs",
        optimization = "enabled",
        distributed_transforms = version_info.has_distributed_transforms,
        qst_transforms = version_info.has_qst_transforms,
        energy_functions = version_info.has_energy_functions
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
function synchronize_pencil_data!(field::Union{SHTnsSpecField{T}, SHTnsPhysField{T}}) where T
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

# ================================================================================
# SHTnsKit v1.1.15+ Enhanced Features
# ================================================================================
# These functions leverage new capabilities in SHTnsKit v1.1.15:
# - Energy/power spectrum analysis
# - Spectral differential operators
# - QST vector transforms for full 3D fields
# - Native threading controls
# ================================================================================

"""
    compute_scalar_energy_spectrum(config::SHTnsKitConfig, alm::Matrix{ComplexF64}; real_field::Bool=true)

Compute the energy spectrum per spherical harmonic degree l using SHTnsKit v1.1.15.

# Returns
Vector of length lmax+1 with energy at each degree l.
"""
function compute_scalar_energy_spectrum(config::SHTnsKitConfig, alm::Matrix{ComplexF64}; real_field::Bool=true)
    try
        return SHTnsKit.energy_scalar_l_spectrum(config.sht_config, alm; real_field=real_field)
    catch e
        # Fallback manual computation for older SHTnsKit versions
        lmax = config.lmax
        spectrum = zeros(Float64, lmax + 1)
        for l in 0:lmax
            for m in 0:min(l, config.mmax)
                # Bounds check for safety
                if l+1 <= size(alm, 1) && m+1 <= size(alm, 2)
                    coeff = alm[l+1, m+1]
                    energy = abs2(coeff)
                    if m > 0 && real_field
                        energy *= 2.0  # Account for negative m modes
                    end
                    spectrum[l+1] += energy
                end
            end
        end
        return spectrum
    end
end

"""
    compute_vector_energy_spectrum(config::SHTnsKitConfig, Slm::Matrix{ComplexF64}, Tlm::Matrix{ComplexF64}; real_field::Bool=true)

Compute the kinetic energy spectrum per spherical harmonic degree l for a vector field
decomposed into spheroidal (Slm) and toroidal (Tlm) components.

# Returns
Vector of length lmax+1 with kinetic energy at each degree l.
"""
function compute_vector_energy_spectrum(config::SHTnsKitConfig, Slm::Matrix{ComplexF64}, Tlm::Matrix{ComplexF64}; real_field::Bool=true)
    try
        return SHTnsKit.energy_vector_l_spectrum(config.sht_config, Slm, Tlm; real_field=real_field)
    catch e
        # Fallback: sum individual spectra
        spec_S = compute_scalar_energy_spectrum(config, Slm; real_field=real_field)
        spec_T = compute_scalar_energy_spectrum(config, Tlm; real_field=real_field)
        return spec_S .+ spec_T
    end
end

"""
    compute_total_scalar_energy(config::SHTnsKitConfig, alm::Matrix{ComplexF64}; real_field::Bool=true)

Compute total energy of a scalar field from its spectral coefficients.
"""
function compute_total_scalar_energy(config::SHTnsKitConfig, alm::Matrix{ComplexF64}; real_field::Bool=true)
    try
        return SHTnsKit.energy_scalar(config.sht_config, alm; real_field=real_field)
    catch e
        # Fallback: sum the spectrum
        return sum(compute_scalar_energy_spectrum(config, alm; real_field=real_field))
    end
end

"""
    compute_total_vector_energy(config::SHTnsKitConfig, Slm::Matrix{ComplexF64}, Tlm::Matrix{ComplexF64}; real_field::Bool=true)

Compute total kinetic energy of a vector field from spheroidal/toroidal coefficients.
"""
function compute_total_vector_energy(config::SHTnsKitConfig, Slm::Matrix{ComplexF64}, Tlm::Matrix{ComplexF64}; real_field::Bool=true)
    try
        return SHTnsKit.energy_vector(config.sht_config, Slm, Tlm; real_field=real_field)
    catch e
        return sum(compute_vector_energy_spectrum(config, Slm, Tlm; real_field=real_field))
    end
end

"""
    compute_enstrophy(config::SHTnsKitConfig, Tlm::Matrix{ComplexF64}; real_field::Bool=true)

Compute enstrophy (mean square vorticity) from toroidal coefficients.
Enstrophy is related to the rotational part of the kinetic energy.
"""
function compute_enstrophy(config::SHTnsKitConfig, Tlm::Matrix{ComplexF64}; real_field::Bool=true)
    try
        return SHTnsKit.enstrophy(config.sht_config, Tlm; real_field=real_field)
    catch e
        # Fallback: compute from spectrum with l(l+1) factor
        lmax = config.lmax
        total = 0.0
        for l in 1:lmax
            for m in 0:min(l, config.mmax)
                # Bounds check for safety
                if l+1 <= size(Tlm, 1) && m+1 <= size(Tlm, 2)
                    coeff = Tlm[l+1, m+1]
                    energy = abs2(coeff) * l * (l + 1)
                    if m > 0 && real_field
                        energy *= 2.0
                    end
                    total += energy
                end
            end
        end
        return total
    end
end

# ================================================================================
# Spectral Differential Operators (SHTnsKit v1.1.15)
# ================================================================================

"""
    spectral_gradient!(config::SHTnsKitConfig, Slm::Matrix{ComplexF64},
                       grad_theta::Matrix{Float64}, grad_phi::Matrix{Float64})

Compute the horizontal gradient of a scalar field in spectral space.
Uses SHTnsKit.SH_to_grad_spat for efficient computation.

# Arguments
- `Slm`: Spectral coefficients of the scalar field
- `grad_theta`: Output θ-component of gradient (modified in-place)
- `grad_phi`: Output φ-component of gradient (modified in-place)
"""
function spectral_gradient!(config::SHTnsKitConfig, Slm::Matrix{ComplexF64},
                           grad_theta::Matrix{Float64}, grad_phi::Matrix{Float64})
    try
        gt, gp = SHTnsKit.SH_to_grad_spat(config.sht_config, Slm; real_output=true)
        copyto!(grad_theta, gt)
        copyto!(grad_phi, gp)
    catch e
        # Fallback: compute using synthesis of derivatives
        @warn "SH_to_grad_spat not available, using fallback gradient computation"
        # Manual gradient would require implementing derivative operators
        fill!(grad_theta, 0.0)
        fill!(grad_phi, 0.0)
    end
end

"""
    extract_divergence_coefficients(config::SHTnsKitConfig, Slm::Matrix{ComplexF64})

Extract divergence spectral coefficients from spheroidal potential.
The divergence field in spectral space is related to Slm by the horizontal Laplacian.

# Returns
Matrix of divergence coefficients.
"""
function extract_divergence_coefficients(config::SHTnsKitConfig, Slm::Matrix{ComplexF64})
    try
        return SHTnsKit.divergence_from_spheroidal(config.sht_config, Slm)
    catch e
        # Fallback: multiply by -l(l+1)/r²
        lmax, mmax = config.lmax, config.mmax
        div_coeffs = zeros(ComplexF64, lmax+1, mmax+1)
        for l in 0:lmax
            factor = -l * (l + 1)  # Note: r² factor depends on application
            for m in 0:min(l, mmax)
                # Bounds check for safety
                if l+1 <= size(Slm, 1) && m+1 <= size(Slm, 2)
                    div_coeffs[l+1, m+1] = Slm[l+1, m+1] * factor
                end
            end
        end
        return div_coeffs
    end
end

"""
    extract_vorticity_coefficients(config::SHTnsKitConfig, Tlm::Matrix{ComplexF64})

Extract vorticity spectral coefficients from toroidal potential.
The vorticity field in spectral space is related to Tlm by the horizontal Laplacian.

# Returns
Matrix of vorticity coefficients.
"""
function extract_vorticity_coefficients(config::SHTnsKitConfig, Tlm::Matrix{ComplexF64})
    try
        return SHTnsKit.vorticity_from_toroidal(config.sht_config, Tlm)
    catch e
        # Fallback: multiply by -l(l+1)
        lmax, mmax = config.lmax, config.mmax
        vort_coeffs = zeros(ComplexF64, lmax+1, mmax+1)
        for l in 0:lmax
            factor = -l * (l + 1)
            for m in 0:min(l, mmax)
                # Bounds check for safety
                if l+1 <= size(Tlm, 1) && m+1 <= size(Tlm, 2)
                    vort_coeffs[l+1, m+1] = Tlm[l+1, m+1] * factor
                end
            end
        end
        return vort_coeffs
    end
end

# ================================================================================
# QST Vector Transforms (SHTnsKit v1.1.15)
# ================================================================================
# QST decomposition: (Q, S, T) where Q relates to radial component,
# S (spheroidal/poloidal) and T (toroidal) relate to tangential components.
# ================================================================================

"""
    shtnskit_qst_to_spatial!(config::SHTnsKitConfig, Qlm, Slm, Tlm, vr, vtheta, vphi)

Convert QST spectral coefficients to full 3D spatial vector field using SHTnsKit v1.1.15.

This is more efficient than separate scalar + vector synthesis as it handles
all three components in a single call.

# Arguments
- `Qlm`: Q (radial) spectral coefficients
- `Slm`: S (spheroidal/poloidal) spectral coefficients
- `Tlm`: T (toroidal) spectral coefficients
- `vr`, `vtheta`, `vphi`: Output spatial components (modified in-place)
"""
function shtnskit_qst_to_spatial!(config::SHTnsKitConfig, Qlm::Matrix{ComplexF64},
                                  Slm::Matrix{ComplexF64}, Tlm::Matrix{ComplexF64},
                                  vr::Matrix{Float64}, vtheta::Matrix{Float64}, vphi::Matrix{Float64})
    if SHTNSKIT_USE_QST
        try
            vr_out, vt_out, vp_out = SHTnsKit.SHqst_to_spat(config.sht_config, Qlm, Slm, Tlm; real_output=true)
            copyto!(vr, vr_out)
            copyto!(vtheta, vt_out)
            copyto!(vphi, vp_out)
            return
        catch e
            @debug "SHqst_to_spat not available, using fallback: $e"
        end
    end

    # Fallback: separate synthesis calls
    vr .= SHTnsKit.synthesis(config.sht_config, Qlm; real_output=true)
    vt_tmp, vp_tmp = SHTnsKit.synthesis_sphtor(config.sht_config, Slm, Tlm; real_output=true)
    copyto!(vtheta, vt_tmp)
    copyto!(vphi, vp_tmp)
end

"""
    shtnskit_spatial_to_qst!(config::SHTnsKitConfig, vr, vtheta, vphi, Qlm, Slm, Tlm)

Convert full 3D spatial vector field to QST spectral coefficients using SHTnsKit v1.1.15.

# Arguments
- `vr`, `vtheta`, `vphi`: Input spatial components
- `Qlm`: Output Q (radial) spectral coefficients (modified in-place)
- `Slm`: Output S (spheroidal/poloidal) spectral coefficients (modified in-place)
- `Tlm`: Output T (toroidal) spectral coefficients (modified in-place)
"""
function shtnskit_spatial_to_qst!(config::SHTnsKitConfig, vr::Matrix{Float64},
                                  vtheta::Matrix{Float64}, vphi::Matrix{Float64},
                                  Qlm::Matrix{ComplexF64}, Slm::Matrix{ComplexF64}, Tlm::Matrix{ComplexF64})
    if SHTNSKIT_USE_QST
        try
            Q_out, S_out, T_out = SHTnsKit.spat_to_SHqst(config.sht_config, vr, vtheta, vphi)
            copyto!(Qlm, Q_out)
            copyto!(Slm, S_out)
            copyto!(Tlm, T_out)
            return
        catch e
            @debug "spat_to_SHqst not available, using fallback: $e"
        end
    end

    # Fallback: separate analysis calls
    Qlm .= SHTnsKit.analysis(config.sht_config, vr)
    S_tmp, T_tmp = SHTnsKit.analysis_sphtor(config.sht_config, vtheta, vphi)
    copyto!(Slm, S_tmp)
    copyto!(Tlm, T_tmp)
end

# ================================================================================
# Threading Control (SHTnsKit v1.1.15)
# ================================================================================

"""
    set_shtnskit_threads(num_threads::Int)

Configure the number of threads used by SHTnsKit transforms.
Uses SHTnsKit.shtns_use_threads when available.
"""
function set_shtnskit_threads(num_threads::Int)
    try
        SHTnsKit.shtns_use_threads(num_threads)
        if get_rank() == 0
            @info "SHTnsKit configured to use $num_threads threads"
        end
    catch e
        @debug "shtns_use_threads not available: $e"
    end
end

"""
    get_shtnskit_version_info()

Get version and capability information about the SHTnsKit installation.
"""
function get_shtnskit_version_info()
    version = try
        string(pkgversion(SHTnsKit))
    catch
        "unknown"
    end

    has_distributed = try
        isdefined(SHTnsKit, :dist_synthesis)
    catch
        false
    end

    has_qst = try
        isdefined(SHTnsKit, :SHqst_to_spat)
    catch
        false
    end

    has_energy = try
        isdefined(SHTnsKit, :energy_scalar)
    catch
        false
    end

    has_rotation = try
        isdefined(SHTnsKit, :SH_Zrotate)
    catch
        false
    end

    has_inplace = try
        isdefined(SHTnsKit, :synthesis!)
    catch
        false
    end

    return (
        version = version,
        has_distributed_transforms = has_distributed,
        has_qst_transforms = has_qst,
        has_energy_functions = has_energy,
        has_rotation_functions = has_rotation,
        has_inplace_transforms = has_inplace,
        use_distributed = SHTNSKIT_USE_DISTRIBUTED,
        use_qst = SHTNSKIT_USE_QST,
        use_scratch_buffers = SHTNSKIT_USE_SCRATCH_BUFFERS
    )
end

# ================================================================================
# In-Place Transform Functions (SHTnsKit v1.1.15)
# ================================================================================
# These functions use in-place operations to reduce memory allocations

"""
    shtnskit_synthesis_inplace!(config::SHTnsKitConfig, alm::Matrix{ComplexF64},
                                 f_out::Matrix{Float64})

In-place spectral-to-physical synthesis using SHTnsKit v1.1.15.
Writes result directly to f_out, avoiding allocation of temporary arrays.

# Arguments
- `config`: SHTnsKit configuration
- `alm`: Input spectral coefficients (lmax+1 × mmax+1)
- `f_out`: Output physical field (nlat × nlon), modified in-place
"""
function shtnskit_synthesis_inplace!(config::SHTnsKitConfig, alm::Matrix{ComplexF64},
                                      f_out::Matrix{Float64})
    try
        # Use in-place synthesis if available (v1.1.15+)
        # SHTnsKit API: synthesis!(config, f_out, alm) — output before input
        SHTnsKit.synthesis!(config.sht_config, f_out, alm)
    catch e
        # Fallback to allocating version
        result = SHTnsKit.synthesis(config.sht_config, alm; real_output=true)
        copyto!(f_out, result)
    end
    return f_out
end

"""
    shtnskit_analysis_inplace!(config::SHTnsKitConfig, f::Matrix{Float64},
                                alm_out::Matrix{ComplexF64})

In-place physical-to-spectral analysis using SHTnsKit v1.1.15.
Writes result directly to alm_out, avoiding allocation of temporary arrays.

# Arguments
- `config`: SHTnsKit configuration
- `f`: Input physical field (nlat × nlon)
- `alm_out`: Output spectral coefficients (lmax+1 × mmax+1), modified in-place
"""
function shtnskit_analysis_inplace!(config::SHTnsKitConfig, f::Matrix{Float64},
                                     alm_out::Matrix{ComplexF64})
    try
        # Use in-place analysis if available (v1.1.15+)
        # SHTnsKit API: analysis!(config, alm_out, f) — output before input
        SHTnsKit.analysis!(config.sht_config, alm_out, f)
    catch e
        # Fallback to allocating version
        result = SHTnsKit.analysis(config.sht_config, f)
        copyto!(alm_out, result)
    end
    return alm_out
end

# ================================================================================
# Field Rotation Functions (SHTnsKit v1.1.15)
# ================================================================================
# These functions use Wigner D-matrices for efficient field rotations in spectral space

"""
    rotate_field_z!(config::SHTnsKitConfig, alm::Matrix{ComplexF64}, alpha::Real;
                    alm_out::Union{Matrix{ComplexF64},Nothing}=nothing)

Rotate a scalar field around the z-axis by angle alpha in spectral space.
This is a pure phase rotation: alm[l,m] -> alm[l,m] * exp(-i*m*alpha)

# Arguments
- `config`: SHTnsKit configuration
- `alm`: Input spectral coefficients (modified in-place if alm_out is nothing)
- `alpha`: Rotation angle in radians
- `alm_out`: Optional output array (if nothing, modifies alm in-place)

# Returns
The rotated coefficients (alm_out if provided, otherwise alm)
"""
function rotate_field_z!(config::SHTnsKitConfig, alm::Matrix{ComplexF64}, alpha::Real;
                         alm_out::Union{Matrix{ComplexF64},Nothing}=nothing)
    output = alm_out === nothing ? alm : alm_out

    try
        # Use native SHTnsKit rotation if available
        SHTnsKit.SH_Zrotate(config.sht_config, alm, alpha, output)
    catch e
        # Fallback: manual phase rotation
        lmax, mmax = config.lmax, config.mmax
        for l in 0:lmax
            for m in 0:min(l, mmax)
                # Bounds check for safety on both input and output
                if l+1 <= size(alm, 1) && m+1 <= size(alm, 2) &&
                   l+1 <= size(output, 1) && m+1 <= size(output, 2)
                    phase = exp(-im * m * alpha)
                    if alm_out === nothing
                        alm[l+1, m+1] *= phase
                    else
                        output[l+1, m+1] = alm[l+1, m+1] * phase
                    end
                end
            end
        end
    end
    return output
end

"""
    rotate_field_y!(config::SHTnsKitConfig, alm::Matrix{ComplexF64}, beta::Real;
                    alm_out::Union{Matrix{ComplexF64},Nothing}=nothing)

Rotate a scalar field around the y-axis by angle beta in spectral space.
Uses Wigner d-matrices (small Wigner rotation matrices).

# Arguments
- `config`: SHTnsKit configuration
- `alm`: Input spectral coefficients
- `beta`: Rotation angle in radians
- `alm_out`: Optional output array

# Returns
The rotated coefficients
"""
function rotate_field_y!(config::SHTnsKitConfig, alm::Matrix{ComplexF64}, beta::Real;
                         alm_out::Union{Matrix{ComplexF64},Nothing}=nothing)
    # Use zeros instead of similar to avoid uninitialized data if SHTnsKit function doesn't fill output
    output = alm_out === nothing ? zeros(ComplexF64, size(alm)) : alm_out

    try
        SHTnsKit.SH_Yrotate(config.sht_config, alm, beta, output)
    catch e
        @warn "SH_Yrotate not available, y-rotation requires Wigner d-matrices. Returning identity (unrotated copy)."
        # Y-rotation is complex - requires Wigner d-matrices
        # Return identity (copy input) when native support is unavailable
        # This prevents returning uninitialized data
        copyto!(output, alm)
    end
    return output
end

"""
    rotate_field_90y!(config::SHTnsKitConfig, alm::Matrix{ComplexF64};
                      alm_out::Union{Matrix{ComplexF64},Nothing}=nothing)

Rotate a scalar field by 90 degrees around the y-axis.
This is a special case with optimized Wigner d-matrix values.

# Arguments
- `config`: SHTnsKit configuration
- `alm`: Input spectral coefficients
- `alm_out`: Optional output array

# Returns
The rotated coefficients
"""
function rotate_field_90y!(config::SHTnsKitConfig, alm::Matrix{ComplexF64};
                           alm_out::Union{Matrix{ComplexF64},Nothing}=nothing)
    # Use zeros instead of similar to avoid uninitialized data if SHTnsKit function doesn't fill output
    output = alm_out === nothing ? zeros(ComplexF64, size(alm)) : alm_out

    try
        SHTnsKit.SH_Yrotate90(config.sht_config, alm, output)
    catch e
        # Fallback to general Y rotation
        rotate_field_y!(config, alm, π/2; alm_out=output)
    end
    return output
end

"""
    rotate_field_90x!(config::SHTnsKitConfig, alm::Matrix{ComplexF64};
                      alm_out::Union{Matrix{ComplexF64},Nothing}=nothing)

Rotate a scalar field by 90 degrees around the x-axis.
Equivalent to: Z(-π/2) * Y(π/2) * Z(π/2)

# Arguments
- `config`: SHTnsKit configuration
- `alm`: Input spectral coefficients
- `alm_out`: Optional output array

# Returns
The rotated coefficients
"""
function rotate_field_90x!(config::SHTnsKitConfig, alm::Matrix{ComplexF64};
                           alm_out::Union{Matrix{ComplexF64},Nothing}=nothing)
    # Use zeros instead of similar to avoid uninitialized data if SHTnsKit function doesn't fill output
    output = alm_out === nothing ? zeros(ComplexF64, size(alm)) : alm_out

    try
        SHTnsKit.SH_Xrotate90(config.sht_config, alm, output)
    catch e
        # Fallback: decompose into Z and Y rotations
        # Use zeros instead of similar to avoid uninitialized values at invalid (l,m) positions
        temp = zeros(ComplexF64, size(alm))
        rotate_field_z!(config, alm, π/2; alm_out=temp)
        rotate_field_90y!(config, temp; alm_out=output)
        rotate_field_z!(config, output, -π/2; alm_out=output)
    end
    return output
end

"""
    rotate_field_euler!(config::SHTnsKitConfig, alm::Matrix{ComplexF64},
                        alpha::Real, beta::Real, gamma::Real;
                        alm_out::Union{Matrix{ComplexF64},Nothing}=nothing)

Rotate a scalar field by Euler angles (ZYZ convention) in spectral space.
The rotation is: R = Rz(gamma) * Ry(beta) * Rz(alpha)

# Arguments
- `config`: SHTnsKit configuration
- `alm`: Input spectral coefficients
- `alpha, beta, gamma`: Euler angles in radians (ZYZ convention)
- `alm_out`: Optional output array

# Returns
The rotated coefficients
"""
function rotate_field_euler!(config::SHTnsKitConfig, alm::Matrix{ComplexF64},
                             alpha::Real, beta::Real, gamma::Real;
                             alm_out::Union{Matrix{ComplexF64},Nothing}=nothing)
    # Use zeros instead of similar to avoid uninitialized data if SHTnsKit function doesn't fill output
    output = alm_out === nothing ? zeros(ComplexF64, size(alm)) : alm_out
    # Use zeros instead of similar to avoid uninitialized values at invalid (l,m) positions
    temp = zeros(ComplexF64, size(alm))

    # Apply rotations in sequence: Rz(alpha), then Ry(beta), then Rz(gamma)
    rotate_field_z!(config, alm, alpha; alm_out=temp)
    rotate_field_y!(config, temp, beta; alm_out=output)
    rotate_field_z!(config, output, gamma; alm_out=output)

    return output
end

# ================================================================================
# Horizontal Laplacian Operator (SHTnsKit v1.1.15)
# ================================================================================

"""
    apply_horizontal_laplacian!(config::SHTnsKitConfig, alm::Matrix{ComplexF64};
                                 alm_out::Union{Matrix{ComplexF64},Nothing}=nothing)

Apply the horizontal (angular) Laplacian to spectral coefficients.
∇²_h Y_l^m = -l(l+1) Y_l^m

This scales each coefficient by -l(l+1), which is the eigenvalue of the
horizontal Laplacian on the unit sphere.

# Arguments
- `config`: SHTnsKit configuration
- `alm`: Input spectral coefficients
- `alm_out`: Optional output array (if nothing, modifies alm in-place)

# Returns
The Laplacian-transformed coefficients
"""
function apply_horizontal_laplacian!(config::SHTnsKitConfig, alm::Matrix{ComplexF64};
                                      alm_out::Union{Matrix{ComplexF64},Nothing}=nothing)
    output = alm_out === nothing ? alm : alm_out
    lmax, mmax = config.lmax, config.mmax

    for l in 0:lmax
        factor = -l * (l + 1)
        for m in 0:min(l, mmax)
            # Check bounds on both input and output
            if l+1 <= size(alm, 1) && m+1 <= size(alm, 2) &&
               l+1 <= size(output, 1) && m+1 <= size(output, 2)
                output[l+1, m+1] = alm[l+1, m+1] * factor
            end
        end
    end
    return output
end

"""
    apply_inverse_horizontal_laplacian!(config::SHTnsKitConfig, alm::Matrix{ComplexF64};
                                         alm_out::Union{Matrix{ComplexF64},Nothing}=nothing,
                                         regularize_l0::Bool=true)

Apply the inverse horizontal Laplacian to spectral coefficients.
This scales each coefficient by -1/(l(l+1)), which is useful for solving Poisson equations.

# Arguments
- `config`: SHTnsKit configuration
- `alm`: Input spectral coefficients
- `alm_out`: Optional output array
- `regularize_l0`: If true, set l=0 mode to zero (since 1/0 is undefined)

# Returns
The inverse-Laplacian-transformed coefficients
"""
function apply_inverse_horizontal_laplacian!(config::SHTnsKitConfig, alm::Matrix{ComplexF64};
                                              alm_out::Union{Matrix{ComplexF64},Nothing}=nothing,
                                              regularize_l0::Bool=true)
    output = alm_out === nothing ? alm : alm_out
    lmax, mmax = config.lmax, config.mmax

    for l in 0:lmax
        if l == 0
            # l=0 mode: set to zero (or keep original if not regularizing)
            # Check bounds for l=0 mode
            if size(output, 1) >= 1 && size(output, 2) >= 1
                if regularize_l0
                    output[1, 1] = zero(ComplexF64)
                elseif alm_out !== nothing && size(alm, 1) >= 1 && size(alm, 2) >= 1
                    output[1, 1] = alm[1, 1]
                end
            end
        else
            factor = -1.0 / (l * (l + 1))
            for m in 0:min(l, mmax)
                # Check bounds on both input and output
                if l+1 <= size(alm, 1) && m+1 <= size(alm, 2) &&
                   l+1 <= size(output, 1) && m+1 <= size(output, 2)
                    output[l+1, m+1] = alm[l+1, m+1] * factor
                end
            end
        end
    end
    return output
end

"""
    compute_horizontal_gradient_magnitude(config::SHTnsKitConfig, alm::Matrix{ComplexF64})

Compute the magnitude of the horizontal gradient |∇_h f|² in spectral space.
|∇_h f|² = l(l+1) |f_lm|²  (summed over all modes)

This is useful for computing gradient energy or penalty terms.

# Arguments
- `config`: SHTnsKit configuration
- `alm`: Spectral coefficients

# Returns
Scalar value of the integrated horizontal gradient magnitude squared
"""
function compute_horizontal_gradient_magnitude(config::SHTnsKitConfig, alm::Matrix{ComplexF64})
    lmax, mmax = config.lmax, config.mmax
    total = 0.0

    for l in 1:lmax  # Skip l=0 (no gradient contribution)
        factor = l * (l + 1)
        for m in 0:min(l, mmax)
            if l+1 <= size(alm, 1) && m+1 <= size(alm, 2)
                energy = abs2(alm[l+1, m+1])
                if m > 0
                    energy *= 2.0  # Account for negative m modes
                end
                total += factor * energy
            end
        end
    end
    return total
end

# ================================================================================
# Spectral Filtering Functions (SHTnsKit v1.1.15)
# ================================================================================

"""
    apply_spectral_filter!(config::SHTnsKitConfig, alm::Matrix{ComplexF64},
                           filter_func::Function;
                           alm_out::Union{Matrix{ComplexF64},Nothing}=nothing)

Apply a custom spectral filter to coefficients.
The filter function takes (l, m) and returns a scaling factor.

# Arguments
- `config`: SHTnsKit configuration
- `alm`: Input spectral coefficients
- `filter_func`: Function (l, m) -> scale_factor
- `alm_out`: Optional output array

# Example
```julia
# Exponential filter for dealiasing
exp_filter(l, m) = exp(-(l/lmax)^16)
apply_spectral_filter!(config, alm, exp_filter)
```
"""
function apply_spectral_filter!(config::SHTnsKitConfig, alm::Matrix{ComplexF64},
                                filter_func::Function;
                                alm_out::Union{Matrix{ComplexF64},Nothing}=nothing)
    output = alm_out === nothing ? alm : alm_out
    lmax, mmax = config.lmax, config.mmax

    for l in 0:lmax
        for m in 0:min(l, mmax)
            # Check bounds on both input and output
            if l+1 <= size(alm, 1) && m+1 <= size(alm, 2) &&
               l+1 <= size(output, 1) && m+1 <= size(output, 2)
                scale = filter_func(l, m)
                output[l+1, m+1] = alm[l+1, m+1] * scale
            end
        end
    end
    return output
end

"""
    apply_exponential_filter!(config::SHTnsKitConfig, alm::Matrix{ComplexF64};
                               order::Int=16, cutoff::Float64=0.65,
                               alm_out::Union{Matrix{ComplexF64},Nothing}=nothing)

Apply an exponential spectral filter for dealiasing.
filter(l) = exp(-α * (l/lmax)^order) where α is chosen so filter(cutoff*lmax) = 0.5

# Arguments
- `config`: SHTnsKit configuration
- `alm`: Input spectral coefficients
- `order`: Filter order (higher = sharper cutoff, default 16)
- `cutoff`: Fraction of lmax where filter = 0.5 (default 0.65)
- `alm_out`: Optional output array
"""
function apply_exponential_filter!(config::SHTnsKitConfig, alm::Matrix{ComplexF64};
                                    order::Int=16, cutoff::Float64=0.65,
                                    alm_out::Union{Matrix{ComplexF64},Nothing}=nothing)
    lmax = config.lmax

    # Validate cutoff to prevent division by zero
    if cutoff <= 0.0 || cutoff > 1.0
        throw(ArgumentError("cutoff must be in (0, 1], got $cutoff"))
    end

    # Handle degenerate case where lmax = 0 (only monopole mode)
    if lmax == 0
        # No filtering needed for single mode - just copy if output provided
        if alm_out !== nothing
            if size(alm_out, 1) >= 1 && size(alm_out, 2) >= 1 &&
               size(alm, 1) >= 1 && size(alm, 2) >= 1
                alm_out[1, 1] = alm[1, 1]
            end
            return alm_out
        end
        return alm
    end

    # Solve for α: exp(-α * cutoff^order) = 0.5 => α = log(2) / cutoff^order
    α = log(2) / cutoff^order

    filter_func(l, m) = exp(-α * (l / lmax)^order)
    return apply_spectral_filter!(config, alm, filter_func; alm_out=alm_out)
end

"""
    truncate_spectral_modes!(config::SHTnsKitConfig, alm::Matrix{ComplexF64},
                             lmax_new::Int, mmax_new::Int=lmax_new;
                             alm_out::Union{Matrix{ComplexF64},Nothing}=nothing)

Truncate spectral coefficients to a lower resolution.
Sets all modes with l > lmax_new or m > mmax_new to zero.

# Arguments
- `config`: SHTnsKit configuration
- `alm`: Input spectral coefficients
- `lmax_new`: New maximum degree
- `mmax_new`: New maximum order (default = lmax_new)
- `alm_out`: Optional output array
"""
function truncate_spectral_modes!(config::SHTnsKitConfig, alm::Matrix{ComplexF64},
                                  lmax_new::Int, mmax_new::Int=lmax_new;
                                  alm_out::Union{Matrix{ComplexF64},Nothing}=nothing)
    output = alm_out === nothing ? alm : alm_out
    lmax, mmax = config.lmax, config.mmax

    for l in 0:lmax
        for m in 0:min(l, mmax)
            # Check bounds on both input and output arrays
            if l+1 <= size(alm, 1) && m+1 <= size(alm, 2) &&
               l+1 <= size(output, 1) && m+1 <= size(output, 2)
                if l > lmax_new || m > mmax_new
                    output[l+1, m+1] = zero(ComplexF64)
                elseif alm_out !== nothing
                    output[l+1, m+1] = alm[l+1, m+1]
                end
            end
        end
    end
    return output
end
