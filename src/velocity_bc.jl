# ================================================================================
# Velocity Boundary Conditions (Workspace-based, zero-allocation)
# ================================================================================
#
# This file contains workspace-based versions of velocity BC functions that
# avoid allocating temporary arrays for each mode.
#
# Performance improvement: ~10-100x faster for BC application depending on nlm
#
# Usage: Create and register workspace before time loop:
#   ws = create_velocity_workspace(Float64, nr)
#   set_velocity_workspace!(ws)
#
# The BC functions will automatically use the workspace if available.
#
# Included by: src/velocity.jl

"""
    apply_velocity_flux_bc_direct_ws!(spec_real, spec_imag, local_lm, lm_idx,
                                      apply_inner, apply_outer,
                                      domain, r_range, ws, tid)

Workspace-based version of direct flux BC (zero allocation).
Uses pre-allocated buffers from workspace `ws` for thread `tid`.
"""
function apply_velocity_flux_bc_direct_ws!(spec_real, spec_imag, local_lm, lm_idx,
                                          apply_inner, apply_outer,
                                          domain, r_range, ws::VelocityWorkspace{T}, tid::Int) where T
    nr = domain.N

    # Use pre-allocated buffers from workspace (no allocation!)
    profile_real = ws.bc_profile_real[tid]
    profile_imag = ws.bc_profile_imag[tid]

    # Clear buffers
    fill!(profile_real, zero(T))
    fill!(profile_imag, zero(T))

    # Extract radial profile
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(spec_real, 3)
            profile_real[r_idx] = spec_real[local_lm, 1, local_r]
            profile_imag[r_idx] = spec_imag[local_lm, 1, local_r]
        end
    end

    # MPI gather
    comm = BoundaryConditions.get_comm()
    if comm !== nothing && Comm_size(comm) > 1
        Allreduce!(profile_real, SUM, comm)
        Allreduce!(profile_imag, SUM, comm)
    end

    # Apply BC: ∂T/∂r = 0
    if apply_inner
        @inbounds profile_real[1] = profile_real[2]
        if any(x -> abs(x) > 1e-12, profile_imag)
            @inbounds profile_imag[1] = profile_imag[2]
        end
    end

    if apply_outer
        @inbounds profile_real[nr] = profile_real[nr-1]
        if any(x -> abs(x) > 1e-12, profile_imag)
            @inbounds profile_imag[nr] = profile_imag[nr-1]
        end
    end

    # Store back
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(spec_real, 3)
            spec_real[local_lm, 1, local_r] = profile_real[r_idx]
            spec_imag[local_lm, 1, local_r] = profile_imag[r_idx]
        end
    end
end

"""
    apply_velocity_flux_bc_physical_stress_ws!(spec_real, spec_imag, local_lm, lm_idx,
                                               apply_inner, apply_outer,
                                               domain, r_range, ws, tid)

Workspace-based version of physical stress BC (zero allocation).
Enforces ∂T/∂r = T/r for proper stress-free boundaries.
"""
function apply_velocity_flux_bc_physical_stress_ws!(spec_real, spec_imag, local_lm, lm_idx,
                                                    apply_inner, apply_outer,
                                                    domain, r_range, ws::VelocityWorkspace{T}, tid::Int) where T
    nr = domain.N
    r = domain.r[:, 4]

    # Use pre-allocated buffers (no allocation!)
    profile_real = ws.bc_profile_real[tid]
    profile_imag = ws.bc_profile_imag[tid]

    # Clear buffers
    fill!(profile_real, zero(T))
    fill!(profile_imag, zero(T))

    # Extract radial profile
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(spec_real, 3)
            profile_real[r_idx] = spec_real[local_lm, 1, local_r]
            profile_imag[r_idx] = spec_imag[local_lm, 1, local_r]
        end
    end

    # MPI gather
    comm = BoundaryConditions.get_comm()
    if comm !== nothing && Comm_size(comm) > 1
        Allreduce!(profile_real, SUM, comm)
        Allreduce!(profile_imag, SUM, comm)
    end

    # Apply BC: ∂T/∂r = T/r (physically correct for stress-free)
    if apply_inner
        Δr = r[2] - r[1]
        scaling_factor = 1.0 / (1.0 + Δr / r[1])
        @inbounds profile_real[1] = profile_real[2] * scaling_factor

        if any(x -> abs(x) > 1e-12, profile_imag)
            @inbounds profile_imag[1] = profile_imag[2] * scaling_factor
        end
    end

    if apply_outer
        Δr = r[nr] - r[nr-1]
        scaling_factor = 1.0 / (1.0 - Δr / r[nr])
        @inbounds profile_real[nr] = profile_real[nr-1] * scaling_factor

        if any(x -> abs(x) > 1e-12, profile_imag)
            @inbounds profile_imag[nr] = profile_imag[nr-1] * scaling_factor
        end
    end

    # Store back
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(spec_real, 3)
            spec_real[local_lm, 1, local_r] = profile_real[r_idx]
            spec_imag[local_lm, 1, local_r] = profile_imag[r_idx]
        end
    end
end

"""
    apply_velocity_flux_bc_tau_ws!(spec_real, spec_imag, local_lm, lm_idx,
                                   apply_inner, apply_outer, dr_matrix,
                                   domain, r_range, ws, tid)

Workspace-based version of tau method flux BC (zero allocation).
Uses pre-allocated buffers for profiles, derivatives, and corrections.
"""
function apply_velocity_flux_bc_tau_ws!(spec_real, spec_imag, local_lm, lm_idx,
                                       apply_inner, apply_outer,
                                       dr_matrix::BandedMatrix, domain, r_range,
                                       ws::VelocityWorkspace{T}, tid::Int) where T
    nr = domain.N

    # Use pre-allocated buffers (no allocation!)
    profile_real = ws.bc_profile_real[tid]
    profile_imag = ws.bc_profile_imag[tid]
    dprofile_real = ws.bc_dprofile_real[tid]
    dprofile_imag = ws.bc_dprofile_imag[tid]
    correction = ws.bc_correction[tid]

    # Clear buffers
    fill!(profile_real, zero(T))
    fill!(profile_imag, zero(T))

    # Extract radial profile
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(spec_real, 3)
            profile_real[r_idx] = spec_real[local_lm, 1, local_r]
            profile_imag[r_idx] = spec_imag[local_lm, 1, local_r]
        end
    end

    # MPI gather
    comm = BoundaryConditions.get_comm()
    if comm !== nothing && Comm_size(comm) > 1
        Allreduce!(profile_real, SUM, comm)
        Allreduce!(profile_imag, SUM, comm)
    end

    # Prescribed flux for stress-free
    flux_inner = T(0)
    flux_outer = T(0)

    # Compute current fluxes
    apply_derivative_matrix!(dprofile_real, dr_matrix, profile_real)
    current_flux_inner = dprofile_real[1]
    current_flux_outer = dprofile_real[nr]

    # Apply tau corrections (in-place, no allocation!)
    if apply_inner && apply_outer
        compute_tau_correction_both_boundaries!(correction,
                                               flux_inner - current_flux_inner,
                                               flux_outer - current_flux_outer,
                                               domain)
        @inbounds for i in 1:nr
            profile_real[i] += correction[i]
        end

        if any(x -> abs(x) > 1e-12, profile_imag)
            apply_derivative_matrix!(dprofile_imag, dr_matrix, profile_imag)
            current_flux_inner_imag = dprofile_imag[1]
            current_flux_outer_imag = dprofile_imag[nr]
            compute_tau_correction_both_boundaries!(correction,
                                                   -current_flux_inner_imag,
                                                   -current_flux_outer_imag,
                                                   domain)
            @inbounds for i in 1:nr
                profile_imag[i] += correction[i]
            end
        end

    elseif apply_inner
        compute_tau_correction_inner_boundary!(correction,
                                              flux_inner - current_flux_inner,
                                              domain)
        @inbounds for i in 1:nr
            profile_real[i] += correction[i]
        end

        if any(x -> abs(x) > 1e-12, profile_imag)
            apply_derivative_matrix!(dprofile_imag, dr_matrix, profile_imag)
            current_flux_inner_imag = dprofile_imag[1]
            compute_tau_correction_inner_boundary!(correction,
                                                  -current_flux_inner_imag,
                                                  domain)
            @inbounds for i in 1:nr
                profile_imag[i] += correction[i]
            end
        end

    elseif apply_outer
        compute_tau_correction_outer_boundary!(correction,
                                              flux_outer - current_flux_outer,
                                              domain)
        @inbounds for i in 1:nr
            profile_real[i] += correction[i]
        end

        if any(x -> abs(x) > 1e-12, profile_imag)
            apply_derivative_matrix!(dprofile_imag, dr_matrix, profile_imag)
            current_flux_outer_imag = dprofile_imag[nr]
            compute_tau_correction_outer_boundary!(correction,
                                                  -current_flux_outer_imag,
                                                  domain)
            @inbounds for i in 1:nr
                profile_imag[i] += correction[i]
            end
        end
    end

    # Store corrected profile back
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(spec_real, 3)
            spec_real[local_lm, 1, local_r] = profile_real[r_idx]
            spec_imag[local_lm, 1, local_r] = profile_imag[r_idx]
        end
    end
end
