# ================================================================================
# Timestepping Module with SHTns
# ================================================================================
#
# This module implements implicit-explicit (IMEX) time-stepping schemes for
# geodynamo simulations using spectral methods with SHTnsKit.
#
# ================================================================================
# MPI PARALLELIZATION - CRITICAL DESIGN PATTERN
# ================================================================================
#
# The spectral data is distributed across MPI processes using PencilArrays.
# Each process owns a subset of (l,m) modes defined by `lm_range`.
#
# PROBLEM: Different processes may own different numbers of modes.
#   - Process 0: lm_range = 1:50   (50 modes)
#   - Process 1: lm_range = 51:100 (50 modes)
#   - Process 2: lm_range = 101:145 (45 modes)  <-- fewer modes!
#
# If we use `for lm_idx in lm_range` with MPI collectives inside, processes
# will call Allreduce different numbers of times → DEADLOCK!
#
# SOLUTION: Use GLOBAL loop bounds with ownership check:
#
#   ```julia
#   nlm_total = u.nlm  # Total number of modes (same for all processes)
#
#   for lm_idx in 1:nlm_total  # ALL processes iterate same number of times
#       owns_mode = lm_idx in lm_range  # Check ownership
#
#       # Allocate buffers (all processes)
#       ur = zeros(T, nr)
#
#       # Fill data only if this process owns the mode
#       if owns_mode
#           ll = lm_idx - first(lm_range) + 1  # Local index
#           for r in r_range
#               ur[r] = u_real[ll, 1, local_r]
#           end
#       end
#
#       # ALL processes call Allreduce together (collective operation)
#       if multi
#           Allreduce!(ur, MPI.SUM, comm)  # Gathers data from owning process
#       end
#
#       # ... perform computation on gathered data ...
#
#       # Scatter result back only if this process owns the mode
#       if owns_mode
#           for r in r_range
#               u_real[ll, 1, local_r] = ur_new[r]
#           end
#       end
#   end
#   ```
#
# KEY INVARIANT: MPI collectives (Allreduce, Allgather, Barrier) must be called
# the same number of times by ALL processes. The global loop bounds ensure this.
#
# DEBUGGING MPI ISSUES:
#   1. If simulation hangs, likely MPI deadlock from unbalanced collective calls
#   2. Check that all loops containing MPI collectives use global bounds
#   3. Use `MPI.Comm_rank(comm)` to identify which process is stuck
#   4. Enable MPI_DEBUG environment variable for detailed tracing
#
# ================================================================================
# TIME-STEPPING SCHEMES
# ================================================================================
#
# EAB2 (Exponential Adams-Bashforth 2nd order):
#   - Treats diffusion implicitly using matrix exponential
#   - Uses Krylov subspace methods for exp(A) and φ₁(A) actions
#   - Nonlinear terms treated explicitly with AB2 extrapolation
#
# CNAB2 (Crank-Nicolson Adams-Bashforth 2nd order):
#   - Diffusion: Crank-Nicolson (implicit, 2nd order)
#   - Nonlinear: Adams-Bashforth (explicit, 2nd order)
#   - Requires solving tridiagonal systems per (l,m) mode
#
# ERK2 (Explicit Runge-Kutta 2nd order):
#   - Fully explicit scheme (for testing/comparison)
#   - CFL-limited timestep required
#
# ================================================================================

using MPI
using LinearAlgebra
using Dates
using JLD2
    
# Timestepping state
mutable struct TimestepState
    time::Float64
    dt::Float64
    step::Int
    iteration::Int
    error::Float64
    converged::Bool
end

# ================================================================================
# NaN/Inf Detection Framework
# ================================================================================

"""
    NaNDetectionConfig

Configuration for NaN/Inf detection during simulation.
"""
struct NaNDetectionConfig
    enabled::Bool
    check_every_n_steps::Int
    abort_on_nan::Bool
    verbose::Bool
end

# Default: check every step, abort on NaN, verbose warnings
const DEFAULT_NAN_CONFIG = NaNDetectionConfig(true, 1, true, true)

"""
    check_field_for_nan(field_data::AbstractArray, field_name::String, config::NaNDetectionConfig, step::Int)

Check a field for NaN or Inf values. Returns (has_nan, has_inf, nan_count, inf_count).
"""
function check_field_for_nan(field_data::AbstractArray, field_name::String,
                              config::NaNDetectionConfig, step::Int)
    if !config.enabled || step % config.check_every_n_steps != 0
        return (false, false, 0, 0)
    end

    nan_count = count(isnan, field_data)
    inf_count = count(isinf, field_data)
    has_nan = nan_count > 0
    has_inf = inf_count > 0

    if (has_nan || has_inf) && config.verbose && get_rank() == 0
        @warn "Numerical issue detected in $field_name at step $step" nan_count inf_count
    end

    return (has_nan, has_inf, nan_count, inf_count)
end

"""
    check_spectral_field_for_nan(field::SHTnsSpecField, field_name::String,
                                  config::NaNDetectionConfig, step::Int)

Check both real and imaginary parts of a spectral field.
"""
function check_spectral_field_for_nan(field, field_name::String,
                                       config::NaNDetectionConfig, step::Int)
    # Check real part
    has_nan_r, has_inf_r, nan_r, inf_r = check_field_for_nan(
        parent(field.data_real), "$(field_name)_real", config, step)

    # Check imaginary part
    has_nan_i, has_inf_i, nan_i, inf_i = check_field_for_nan(
        parent(field.data_imag), "$(field_name)_imag", config, step)

    return (has_nan_r || has_nan_i, has_inf_r || has_inf_i,
            nan_r + nan_i, inf_r + inf_i)
end

"""
    check_simulation_state_for_nan(state::SimulationState, step::Int;
                                     config::NaNDetectionConfig=DEFAULT_NAN_CONFIG)

Comprehensive NaN/Inf check across all simulation fields.
Returns true if any NaN/Inf detected.
"""
function check_simulation_state_for_nan(state, step::Int;
                                         config::NaNDetectionConfig=DEFAULT_NAN_CONFIG)
    if !config.enabled || step % config.check_every_n_steps != 0
        return false
    end

    any_issue = false

    # Check velocity fields
    has_nan, has_inf, _, _ = check_spectral_field_for_nan(
        state.velocity.𝒯, "velocity_toroidal", config, step)
    any_issue |= (has_nan || has_inf)

    has_nan, has_inf, _, _ = check_spectral_field_for_nan(
        state.velocity.𝒫, "velocity_poloidal", config, step)
    any_issue |= (has_nan || has_inf)

    # Check magnetic fields
    has_nan, has_inf, _, _ = check_spectral_field_for_nan(
        state.magnetic.𝒯, "magnetic_toroidal", config, step)
    any_issue |= (has_nan || has_inf)

    has_nan, has_inf, _, _ = check_spectral_field_for_nan(
        state.magnetic.𝒫, "magnetic_poloidal", config, step)
    any_issue |= (has_nan || has_inf)

    # Check temperature
    has_nan, has_inf, _, _ = check_spectral_field_for_nan(
        state.temperature.spectral, "temperature", config, step)
    any_issue |= (has_nan || has_inf)

    # Check composition if present
    if state.composition !== nothing
        has_nan, has_inf, _, _ = check_spectral_field_for_nan(
            state.composition.spectral, "composition", config, step)
        any_issue |= (has_nan || has_inf)
    end

    if any_issue && config.abort_on_nan
        error("NaN or Inf detected in simulation fields at step $step. " *
              "Simulation aborted to prevent invalid results. " *
              "Check:\n" *
              "  1. Timestep size (may be too large for stability)\n" *
              "  2. Boundary conditions (check for invalid values)\n" *
              "  3. Initial conditions (check for NaN/Inf in IC)\n" *
              "  4. Physical parameters (negative diffusivities, etc.)\n" *
              "Set config.abort_on_nan=false to continue despite NaN.")
    end

    return any_issue
end

# ================================
# Exponential AB2 (EAB2) Utilities
# ================================

struct ETDCache{T}
    dt::Float64
    l_values::Vector{Int}
    E::Vector{Matrix{T}}      # exp(dt A_l) per l
    phi1::Vector{Matrix{T}}   # phi1(dt A_l) per l
end

"""
    EAB2ALUCacheEntry{T}

Type-stable cache entry for EAB2 (Exponential Adams-Bashforth 2) method.
Stores banded matrices and their LU factorizations for each spherical harmonic degree l.
"""
struct EAB2ALUCacheEntry{T}
    ν::Float64  # Diffusivity coefficient
    nr::Int     # Number of radial points
    map::Dict{Int, Tuple{BandedMatrix{T}, BandedLU{T}}}  # l -> (A_banded, LU(A_banded))
end

"""
    create_etd_cache(config, domain, diffusivity, dt) -> ETDCache

Build per-l exponential cache for the linear operator A_l = diffusivity*(d²/dr² + (2/r)d/dr − l(l+1)/r²).
Computes exp(dt A_l) and phi1(dt A_l) via dense methods. Single-rank recommended.
"""
function create_etd_cache(::Type{T}, config::SHTnsKitConfig, domain::RadialDomain,
                          diffusivity::Float64, dt::Float64) where T
    lap = create_radial_laplacian(domain)
    nr = domain.N
    r_inv2 = @views domain.r[1:nr, 2]
    lvals = unique(config.l_values)
    E = Matrix{T}[]
    PHI1 = Matrix{T}[]
    for l in lvals
        # Build banded for A = ν*(d² + (2/r)d − l(l+1)/r²)
        Adata = diffusivity .* lap.data
        # Convert to dense and subtract l(l+1)/r² on diagonal
        Adense = banded_to_dense(BandedMatrix(Adata, i_KL, nr))
        lfac = Float64(l * (l + 1))
        @inbounds for n in 1:nr
            Adense[n, n] -= diffusivity * lfac * r_inv2[n]
        end
        # exp(dt A)
        Adt = dt .* Adense
        E_l = exp(Adt)
        push!(E, Matrix{T}(E_l))
        # phi1(dt A) = A^{-1} * (exp(dt A) − I) / dt
        F = (E_l - I) / dt
        fac = lu(Adense)
        phi1_l = fac \ F
        push!(PHI1, Matrix{T}(phi1_l))
    end
    return ETDCache{T}(dt, lvals, E, PHI1)
end

"""
    build_banded_A(T, domain, diffusivity, l) -> BandedMatrix{T}

Construct banded A = ν*(d²/dr² + (2/r)d/dr − l(l+1)/r²) in banded storage.
"""
function build_banded_A(::Type{T}, domain::RadialDomain, diffusivity::Float64, l::Int) where T
    lap = create_radial_laplacian(domain)
    data = diffusivity .* lap.data
    nr = domain.N
    r_inv2 = @views domain.r[1:nr, 2]
    lfac = Float64(l * (l + 1))
    @inbounds for n in 1:nr
        data[i_KL + 1, n] -= diffusivity * lfac * r_inv2[n]
    end
    return BandedMatrix{T}(Matrix{T}(data), i_KL, nr)
end

"""
    apply_banded_full!(out, B, v)

Apply banded matrix to full vector.
"""
function apply_banded_full!(out::Vector{T}, B::BandedMatrix{T}, v::Vector{T}) where T
    fill!(out, zero(T))
    N = B.size; bw = B.bandwidth
    @inbounds for j in 1:N
        for i in max(1, j - bw):min(N, j + bw)
            row = bw + 1 + i - j
            if 1 <= row <= 2*bw+1
                out[i] += B.data[row, j] * v[j]
            end
        end
    end
    return out
end

"""
    exp_action_krylov(Aop!, v, dt; m=20, tol=1e-8) -> y ≈ exp(dt A) v

Simple Arnoldi-based approximation of the exponential action.
"""
function exp_action_krylov(Aop!, v::Vector{T}, dt::Float64; m::Int=20, tol::Float64=1e-8) where T
    n = length(v)

    # Input validation
    if n == 0 || !all(isfinite.(v))
        return zeros(T, n)
    end

    V = Matrix{T}(undef, n, m)
    H = zeros(T, m, m)
    beta = norm(v)
    if beta == 0
        return zeros(T, n)
    end

    # Check for very small timestep
    if abs(dt) < eps(T) * 10
        return copy(v)  # exp(0*A) * v = v
    end

    V[:, 1] = v / beta
    w = similar(v)
    kmax = m

    for j in 1:m
        Aop!(w, view(V, :, j))

        # Check for NaN/Inf in operator result
        if !all(isfinite.(w))
            @warn "Non-finite values from operator in Krylov iteration $j"
            kmax = max(1, j-1)
            break
        end

        for i in 1:j
            H[i, j] = dot(view(V, :, i), w)
            @. w = w - H[i, j] * V[:, i]
        end

        if j < m
            H[j+1, j] = norm(w)
            if H[j+1, j] < eps(T) * 100  # More robust zero check
                kmax = j
                break
            end
            V[:, j+1] = w / H[j+1, j]

            # Adaptive residual-based stopping criterion with stability check
            try
                Hred_j = dt .* @view H[1:j, 1:j]

                # Check condition number of H submatrix
                if j > 1 && cond(Hred_j) > 1e12
                    @warn "Ill-conditioned Hessenberg matrix, stopping Krylov at iteration $j"
                    kmax = j
                    break
                end

                e1 = zeros(T, j); e1[1] = one(T)
                y_small_j = exp(Hred_j) * (beta .* e1)

                if !all(isfinite.(y_small_j))
                    @warn "Non-finite exponential result, stopping Krylov at iteration $j"
                    kmax = j
                    break
                end

                res_est = abs(H[j+1, j]) * abs(j > 0 ? y_small_j[end] : beta)
                if res_est <= tol * norm(y_small_j)
                    kmax = j
                    break
                end
            catch e
                @warn "Error in Krylov convergence check: $e, stopping at iteration $j"
                kmax = j
                break
            end
        end
    end

    # Final computation with error handling
    try
        Hred = dt .* H[1:kmax, 1:kmax]
        e1 = zeros(T, kmax); e1[1] = one(T)
        y_small = exp(Hred) * (beta .* e1)

        if !all(isfinite.(y_small))
            @warn "Non-finite result in final Krylov computation, using first-order approximation"
            # Fallback to first-order: exp(dt*A)*v ≈ v + dt*A*v
            result = copy(v)
            Aop!(w, v)
            result .+= dt .* w
            return result
        end

        result = V[:, 1:kmax] * y_small

        if !all(isfinite.(result))
            @warn "Non-finite final result in Krylov, using first-order approximation"
            result = copy(v)
            Aop!(w, v)
            result .+= dt .* w
        end

        return result
    catch e
        @warn "Error in final Krylov computation: $e, using first-order approximation"
        result = copy(v)
        Aop!(w, v)
        result .+= dt .* w
        return result
    end
end

"""
    phi1_action_krylov(BA, LU_A, v, dt; m=20, tol=1e-8) -> y ≈ φ1(dt A) v

Compute φ1(dt A) v = A^{-1}[(exp(dt A) − I) v]/dt using Krylov exp(action) and banded solve.
"""
function phi1_action_krylov(Aop!, A_lu::BandedLU{T}, v::Vector{T}, dt::Float64; m::Int=20, tol::Float64=1e-8) where T
    # Check for zero input
    if norm(v) < eps(T) * 100
        return zeros(T, length(v))
    end

    # Compute exp(dt*A) * v
    ev = exp_action_krylov(Aop!, v, dt; m, tol)
    c = ev .- v

    # Check if dt is very small - use series expansion
    if dt < 1e-8
        # φ1(dt*A) * v ≈ v + (dt/2)*A*v for small dt
        Av = similar(v)
        Aop!(Av, v)
        return v .+ (dt/2) .* Av
    end

    # Solve A * x = c
    x = copy(c)
    try
        solve_banded!(x, A_lu, c)
        @. x = x / dt

        # Validate result
        if !all(isfinite.(x))
            @warn "Non-finite result in phi1_action_krylov, using fallback"
            # Fallback to series expansion
            Av = similar(v)
            Aop!(Av, v)
            return v .+ (dt/2) .* Av
        end

        return x
    catch e
        @warn "Banded solve failed in phi1_action_krylov: $e, using fallback"
        # Fallback to series expansion
        Av = similar(v)
        Aop!(Av, v)
        return v .+ (dt/2) .* Av
    end
end

"""
    eab2_update_krylov!(u, nl, nl_prev, domain, diffusivity, config, dt; m=20, tol=1e-8)

EAB2 update using Krylov exp/φ1 actions and banded LU for φ1.

# MPI Safety
Uses global loop bounds (1:nlm) to ensure all processes call Allreduce
the same number of times, preventing deadlock with uneven lm distribution.
"""
function eab2_update_krylov!(u::SHTnsSpecField{T}, nl::SHTnsSpecField{T},
                             nl_prev::SHTnsSpecField{T}, domain::RadialDomain,
                             diffusivity::Float64, config::SHTnsKitConfig,
                             dt::Float64; m::Int=20, tol::Float64=1e-8) where T
    u_real = parent(u.data_real); u_imag = parent(u.data_imag)
    n_real = parent(nl.data_real); n_imag = parent(nl.data_imag)
    p_real = parent(nl_prev.data_real); p_imag = parent(nl_prev.data_imag)
    lm_range = get_local_range(u.pencil, 1)
    r_range  = get_local_range(u.pencil, 3)
    nr = domain.N
    comm = get_comm()
    multi = MPI.Comm_size(comm) > 1
    nlm_total = u.nlm

    # Use GLOBAL loop bounds to ensure all processes call Allreduce same number of times
    for lm_idx in 1:nlm_total
        # Check if this process owns this lm mode
        owns_mode = lm_idx in lm_range

        l = config.l_values[lm_idx]
        A_banded = build_banded_A(T, domain, diffusivity, l)
        A_lu = factorize_banded(A_banded)

        # Assembled full vectors - all processes allocate
        ur = zeros(T, nr); ui = zeros(T, nr)
        nrn = zeros(T, nr); nin = zeros(T, nr)

        # Only fill if this process owns the mode
        if owns_mode
            ll = lm_idx - first(lm_range) + 1
            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    ur[r] = u_real[ll,1,lr]; ui[r] = u_imag[ll,1,lr]
                    nrn[r] = (3/2)*n_real[ll,1,lr] - (1/2)*p_real[ll,1,lr]
                    nin[r] = (3/2)*n_imag[ll,1,lr] - (1/2)*p_imag[ll,1,lr]
                end
            end
        end

        # ALL processes call Allreduce together (collective operation)
        if multi
            Allreduce!(ur, MPI.SUM, comm)
            Allreduce!(ui, MPI.SUM, comm)
            Allreduce!(nrn, MPI.SUM, comm)
            Allreduce!(nin, MPI.SUM, comm)
        end

        # Define Aop! using banded apply
        function Aop!(out, v)
            apply_banded_full!(out, A_banded, v)
            return nothing
        end

        # Real
        ur_new = exp_action_krylov(Aop!, ur, dt; m, tol)
        add_r = phi1_action_krylov(Aop!, A_lu, nrn, dt; m, tol)
        @. ur_new = ur_new + dt * add_r

        # Imag
        ui_new = exp_action_krylov(Aop!, ui, dt; m, tol)
        add_i = phi1_action_krylov(Aop!, A_lu, nin, dt; m, tol)
        @. ui_new = ui_new + dt * add_i

        # Scatter back only if this process owns the mode
        if owns_mode
            ll = lm_idx - first(lm_range) + 1
            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    u_real[ll,1,lr] = ur_new[r]
                    u_imag[ll,1,lr] = ui_new[r]
                end
            end
        end
    end
    return u
end

"""
    get_eab2_alu_cache!(caches, key, ν, T, domain) -> Dict{Int,Tuple{BandedMatrix{T},BandedLU{T}}}

Retrieve or initialize a cache mapping l -> (A_banded, LU(A_banded)) for EAB2.
Reinitializes if ν or nr changed.
Type-stable version using EAB2ALUCacheEntry.
"""
function get_eab2_alu_cache!(caches::Dict{Symbol, EAB2ALUCacheEntry{T}}, key::Symbol, ν::Float64, ::Type{T}, domain::RadialDomain) where T
    entry = get(caches, key, nothing)
    nr = domain.N
    if entry === nothing || entry.ν != ν || entry.nr != nr
        entry = EAB2ALUCacheEntry{T}(ν, nr, Dict{Int, Tuple{BandedMatrix{T}, BandedLU{T}}}())
        caches[key] = entry
    end
    return entry.map
end

"""
    eab2_update_krylov_cached!(u, nl, nl_prev, alu_map, domain, ν, config, dt; m=20, tol=1e-8)

Same as eab2_update_krylov!, but reuses cached banded A and LU per l.

# MPI Safety
Uses global loop bounds (1:nlm) to ensure all processes call Allreduce
the same number of times, preventing deadlock with uneven lm distribution.
"""
function eab2_update_krylov_cached!(u::SHTnsSpecField{T}, nl::SHTnsSpecField{T},
                                    nl_prev::SHTnsSpecField{T}, alu_map::Dict{Int, Tuple{BandedMatrix{T}, BandedLU{T}}},
                                    domain::RadialDomain, diffusivity::Float64, config::SHTnsKitConfig,
                                    dt::Float64; m::Int=20, tol::Float64=1e-8) where T
    u_real = parent(u.data_real); u_imag = parent(u.data_imag)
    n_real = parent(nl.data_real); n_imag = parent(nl.data_imag)
    p_real = parent(nl_prev.data_real); p_imag = parent(nl_prev.data_imag)
    lm_range = get_local_range(u.pencil, 1)
    r_range  = get_local_range(u.pencil, 3)
    nr = domain.N
    comm = get_comm()
    multi = MPI.Comm_size(comm) > 1
    nlm_total = u.nlm

    # Use GLOBAL loop bounds to ensure all processes call Allreduce same number of times
    for lm_idx in 1:nlm_total
        # Check if this process owns this lm mode
        owns_mode = lm_idx in lm_range

        l = config.l_values[lm_idx]
        # get or build A and LU for this l
        tup = get(alu_map, l, nothing)
        if tup === nothing
            A_banded = build_banded_A(T, domain, diffusivity, l)
            A_lu = factorize_banded(A_banded)
            tup = (A_banded, A_lu)
            alu_map[l] = tup
        end
        A_banded, A_lu = tup

        # Assembled full vectors - all processes allocate
        ur = zeros(T, nr); ui = zeros(T, nr)
        nrn = zeros(T, nr); nin = zeros(T, nr)

        # Only fill if this process owns the mode
        if owns_mode
            ll = lm_idx - first(lm_range) + 1
            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    ur[r] = u_real[ll,1,lr]; ui[r] = u_imag[ll,1,lr]
                    nrn[r] = (3/2)*n_real[ll,1,lr] - (1/2)*p_real[ll,1,lr]
                    nin[r] = (3/2)*n_imag[ll,1,lr] - (1/2)*p_imag[ll,1,lr]
                end
            end
        end

        # ALL processes call Allreduce together (collective operation)
        if multi
            Allreduce!(ur, MPI.SUM, comm)
            Allreduce!(ui, MPI.SUM, comm)
            Allreduce!(nrn, MPI.SUM, comm)
            Allreduce!(nin, MPI.SUM, comm)
        end

        # Define Aop! using banded apply
        function Aop!(out, v)
            apply_banded_full!(out, A_banded, v)
            return nothing
        end

        ur_new = exp_action_krylov(Aop!, ur, dt; m, tol)
        add_r = phi1_action_krylov(Aop!, A_lu, nrn, dt; m, tol)
        @. ur_new = ur_new + dt * add_r
        ui_new = exp_action_krylov(Aop!, ui, dt; m, tol)
        add_i = phi1_action_krylov(Aop!, A_lu, nin, dt; m, tol)
        @. ui_new = ui_new + dt * add_i

        # Scatter back only if this process owns the mode
        if owns_mode
            ll = lm_idx - first(lm_range) + 1
            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    u_real[ll,1,lr] = ur_new[r]
                    u_imag[ll,1,lr] = ui_new[r]
                end
            end
        end
    end
    return u
end
"""
    eab2_update!(u, nl, nl_prev, etd, config)

Apply EAB2 update per (l,m): u^{n+1} = E u^n + dt*phi1*(3/2 nl^n − 1/2 nl^{n−1}).

# MPI Safety
Uses global loop bounds (1:nlm) to ensure all processes call Allreduce
the same number of times, preventing deadlock with uneven lm distribution.
"""
function eab2_update!(u::SHTnsSpecField{T}, nl::SHTnsSpecField{T},
                      nl_prev::SHTnsSpecField{T}, etd::ETDCache{T}, config::SHTnsKitConfig,
                      dt::Float64) where T
    u_real = parent(u.data_real); u_imag = parent(u.data_imag)
    n_real = parent(nl.data_real); n_imag = parent(nl.data_imag)
    p_real = parent(nl_prev.data_real); p_imag = parent(nl_prev.data_imag)
    lm_range = get_local_range(u.pencil, 1)
    r_range  = get_local_range(u.pencil, 3)
    nr_full = size(etd.E[1], 1)
    comm = get_comm()
    multi = MPI.Comm_size(comm) > 1
    nlm_total = u.nlm
    linear_r_work = zeros(T, nr_full)
    linear_i_work = similar(linear_r_work)
    phi_tmp = similar(linear_r_work)

    # Use GLOBAL loop bounds to ensure all processes call Allreduce same number of times
    for lm_idx in 1:nlm_total
        # Check if this process owns this lm mode
        owns_mode = lm_idx in lm_range

        l = config.l_values[lm_idx]
        lpos = findfirst(==(l), etd.l_values)
        E = etd.E[lpos]
        P1 = etd.phi1[lpos]

        # Assemble full radial vectors - all processes allocate
        ur = zeros(T, nr_full); ui = zeros(T, nr_full)
        nrn = zeros(T, nr_full); nin = zeros(T, nr_full)

        # Only fill if this process owns the mode
        if owns_mode
            ll = lm_idx - first(lm_range) + 1
            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    ur[r] = u_real[ll, 1, lr]
                    ui[r] = u_imag[ll, 1, lr]
                    nrn[r] = (3/2)*n_real[ll,1,lr] - (1/2)*p_real[ll,1,lr]
                    nin[r] = (3/2)*n_imag[ll,1,lr] - (1/2)*p_imag[ll,1,lr]
                end
            end
        end

        # ALL processes call Allreduce together (collective operation)
        if multi
            Allreduce!(ur, MPI.SUM, comm)
            Allreduce!(ui, MPI.SUM, comm)
            Allreduce!(nrn, MPI.SUM, comm)
            Allreduce!(nin, MPI.SUM, comm)
        end

        mul!(linear_r_work, E, ur)
        mul!(phi_tmp, P1, nrn)
        @. linear_r_work = linear_r_work + dt * phi_tmp

        mul!(linear_i_work, E, ui)
        mul!(phi_tmp, P1, nin)
        @. linear_i_work = linear_i_work + dt * phi_tmp

        # Scatter back only if this process owns the mode
        if owns_mode
            ll = lm_idx - first(lm_range) + 1
            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    u_real[ll,1,lr] = linear_r_work[r]
                    u_imag[ll,1,lr] = linear_i_work[r]
                end
            end
        end
    end
    return u
end
# Implicit matrices for each spherical harmonic mode (SHTns version)
struct SHTnsImplicitMatrices{T}
    system_matrices::Vector{BandedMatrix{T}}  # (1/Δt)I − θ·L per l
    factorizations::Vector{BandedLU{T}}       # Banded LU factorizations
    linear_matrices::Vector{BandedMatrix{T}}  # Linear operator L per l (scaled by diffusivity)
    l_values::Vector{Int}                     # l values for indexing
    lookup::Dict{Int,Int}                     # Map l → index into vectors above
    theta::Float64                            # Crank–Nicolson weight θ
end

function create_shtns_timestepping_matrices(config::SHTnsKitConfig,
                                            domain::RadialDomain,
                                            diffusivity::Float64,
                                            dt::Float64;
                                            theta::Float64=d_implicit,
                                            T::Type{<:Number}=Float64)
    unique_l = unique(config.l_values)
    laplacian = create_radial_laplacian(domain)
    r_inv_sq = @views domain.r[1:domain.N, 2]

    base_data = T.(diffusivity .* laplacian.data)
    system_matrices = Vector{BandedMatrix{T}}(undef, length(unique_l))
    linear_matrices = Vector{BandedMatrix{T}}(undef, length(unique_l))
    factorizations = Vector{BandedLU{T}}(undef, length(unique_l))
    l_values = Vector{Int}(undef, length(unique_l))
    lookup = Dict{Int,Int}()

    inv_dt = T(1 / dt)
    θ_T = T(theta)
    minus_θ = -θ_T

    for (idx, l) in enumerate(unique_l)
        l_values[idx] = l
        lookup[l] = idx

        linear_data = copy(base_data)
        l_factor = Float64(l * (l + 1))
        @inbounds for n in 1:domain.N
            linear_data[i_KL + 1, n] -= T(diffusivity * l_factor * r_inv_sq[n])
        end

        linear_matrix = BandedMatrix{T}(copy(linear_data), i_KL, domain.N)

        system_data = copy(linear_data)
        system_data .*= minus_θ
        system_data[i_KL + 1, :] .+= inv_dt
        system_matrix = BandedMatrix{T}(system_data, i_KL, domain.N)

        system_matrices[idx] = system_matrix
        linear_matrices[idx] = linear_matrix
        factorizations[idx] = factorize_banded(system_matrix)
    end

    return SHTnsImplicitMatrices{T}(system_matrices, factorizations,
                                    linear_matrices, l_values, lookup, theta)
end

# Velocity-specific matrix construction with embedded BCs is in src/bcs/velocity_bc.jl

function banded_to_dense(matrix::BandedMatrix{T}) where T
    # Convert banded matrix to dense for LU factorization
    N = matrix.size
    bandwidth = matrix.bandwidth
    dense = zeros(T, N, N)
    
    for j in 1:N
        for i in max(1, j - bandwidth):min(N, j + bandwidth)
            band_row = bandwidth + 1 + i - j
            dense[i, j] = matrix.data[band_row, j]
        end
    end
    
    return dense
end

function apply_explicit_operator!(output::SHTnsSpecField{T},
                                  input::SHTnsSpecField{T},
                                  nonlinear::SHTnsSpecField{T},
                                  domain::RadialDomain,
                                  diffusivity::Float64,
                                  dt::Float64;
                                  nl_prev::Union{SHTnsSpecField{T},Nothing}=nothing,
                                  matrices::Union{SHTnsImplicitMatrices{T},Nothing}=nothing) where T

    if nl_prev !== nothing && matrices !== nothing
        build_rhs_cnab2!(output, input, nonlinear, nl_prev, dt, matrices)
        return output
    end

    # Fallback: backward Euler style explicit operator without linear correction.
    out_real = parent(output.data_real)
    out_imag = parent(output.data_imag)
    in_real  = parent(input.data_real)
    in_imag  = parent(input.data_imag)
    nl_real  = parent(nonlinear.data_real)
    nl_imag  = parent(nonlinear.data_imag)

    lm_range = get_local_range(input.pencil, 1)
    r_range  = get_local_range(input.pencil, 3)

    @inbounds for lm_idx in lm_range
        if lm_idx <= input.nlm
            local_lm = lm_idx - first(lm_range) + 1
            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_lm <= size(out_real, 1) && local_r <= size(out_real, 3)
                    out_real[local_lm, 1, local_r] = in_real[local_lm, 1, local_r] / dt +
                                                     nl_real[local_lm, 1, local_r]
                    out_imag[local_lm, 1, local_r] = in_imag[local_lm, 1, local_r] / dt +
                                                     nl_imag[local_lm, 1, local_r]
                end
            end
        end
    end

    return output
end

"""
    build_rhs_cnab2!(rhs, un, nl, nl_prev, dt, matrices)

Build RHS for CNAB2 IMEX: rhs = un/dt + (1-θ)·L·un + (3/2)·nl − (1/2)·nl_prev,
where θ = matrices.theta and L is the diffusivity-scaled linear operator.

# MPI Safety
Uses global loop bounds (1:nlm) to ensure all processes call Allreduce
the same number of times, preventing deadlock with uneven lm distribution.
"""
function build_rhs_cnab2!(rhs::SHTnsSpecField{T}, un::SHTnsSpecField{T},
                          nl::SHTnsSpecField{T}, nl_prev::SHTnsSpecField{T},
                          dt::Float64, matrices::SHTnsImplicitMatrices{T}) where T
    r_real = parent(rhs.data_real); r_imag = parent(rhs.data_imag)
    u_real = parent(un.data_real);  u_imag = parent(un.data_imag)
    n_real = parent(nl.data_real);  n_imag = parent(nl.data_imag)
    p_real = parent(nl_prev.data_real); p_imag = parent(nl_prev.data_imag)

    lm_range = get_local_range(un.pencil, 1)
    r_range  = get_local_range(un.pencil, 3)

    inv_dt = T(1 / dt)
    three_halves = T(1.5)
    half = T(0.5)

    θ_T = T(matrices.theta)
    linear_weight = one(T) - θ_T
    add_linear = !iszero(linear_weight)

    nr_global = add_linear ? matrices.system_matrices[1].size : 0
    ur = add_linear ? zeros(T, nr_global) : T[]
    ui = add_linear ? zeros(T, nr_global) : T[]
    lin_r = add_linear ? zeros(T, nr_global) : T[]
    lin_i = add_linear ? zeros(T, nr_global) : T[]

    comm = get_comm()
    multi = MPI.Comm_size(comm) > 1
    nlm_total = un.nlm

    # Use GLOBAL loop bounds to ensure all processes call Allreduce same number of times
    @inbounds for lm_idx in 1:nlm_total
        # Check if this process owns this lm mode
        owns_mode = lm_idx in lm_range

        l = un.config.l_values[lm_idx]
        idx = add_linear ? get(matrices.lookup, l, nothing) : nothing

        if add_linear
            idx === nothing && error("Missing implicit matrix for l=$l")
            fill!(ur, zero(T)); fill!(ui, zero(T))

            # Only fill if this process owns the mode
            if owns_mode
                ll = lm_idx - first(lm_range) + 1
                if ll <= size(r_real, 1)
                    for r in r_range
                        lr = r - first(r_range) + 1
                        if lr <= size(u_real, 3)
                            ur[r] = u_real[ll, 1, lr]
                            ui[r] = u_imag[ll, 1, lr]
                        end
                    end
                end
            end

            # ALL processes call Allreduce together (collective operation)
            if multi
                Allreduce!(ur, MPI.SUM, comm)
                Allreduce!(ui, MPI.SUM, comm)
            end

            fill!(lin_r, zero(T)); fill!(lin_i, zero(T))
            apply_banded_full!(lin_r, matrices.linear_matrices[idx], ur)
            apply_banded_full!(lin_i, matrices.linear_matrices[idx], ui)
        end

        # Only update output if this process owns the mode
        if owns_mode
            ll = lm_idx - first(lm_range) + 1
            if ll <= size(r_real, 1)
                for r in r_range
                    lr = r - first(r_range) + 1
                    lr > size(r_real, 3) && continue

                    value_real = inv_dt * u_real[ll, 1, lr] +
                                 three_halves * n_real[ll, 1, lr] -
                                 half * p_real[ll, 1, lr]
                    value_imag = inv_dt * u_imag[ll, 1, lr] +
                                 three_halves * n_imag[ll, 1, lr] -
                                 half * p_imag[ll, 1, lr]

                    if add_linear
                        value_real += linear_weight * lin_r[r]
                        value_imag += linear_weight * lin_i[r]
                    end

                    r_real[ll, 1, lr] = value_real
                    r_imag[ll, 1, lr] = value_imag
                end
            end
        end
    end

    return rhs
end

function solve_implicit_step!(solution::SHTnsSpecField{T},
                              rhs::SHTnsSpecField{T},
                              matrices::SHTnsImplicitMatrices{T}) where T
    sol_real = parent(solution.data_real)
    sol_imag = parent(solution.data_imag)
    rhs_real = parent(rhs.data_real)
    rhs_imag = parent(rhs.data_imag)

    lm_range = get_local_range(solution.pencil, 1)
    nr = matrices.system_matrices[1].size  # Full radial size (local = global for spectral)

    # Allocate buffers for the radial profile
    tmp_r = Vector{T}(undef, nr)
    tmp_i = Vector{T}(undef, nr)

    # Loop over local lm modes only (radial is fully local, matching DD_2DCODE)
    @inbounds for lm_idx in lm_range
        local_lm = lm_idx - first(lm_range) + 1

        l = solution.config.l_values[lm_idx]
        idx = get(matrices.lookup, l, nothing)
        idx === nothing && continue

        # Copy RHS radial profile to work buffer
        for ir in 1:nr
            tmp_r[ir] = rhs_real[local_lm, 1, ir]
            tmp_i[ir] = rhs_imag[local_lm, 1, ir]
        end

        # Solve the banded system (matching Fortran tim_invX)
        solve_banded!(tmp_r, matrices.factorizations[idx], tmp_r)
        solve_banded!(tmp_i, matrices.factorizations[idx], tmp_i)

        # Store solution back
        for ir in 1:nr
            sol_real[local_lm, 1, ir] = tmp_r[ir]
            sol_imag[local_lm, 1, ir] = tmp_i[ir]
        end
    end

    return solution
end


function compute_timestep_error(new_field::SHTnsSpecField{T}, 
                               old_field::SHTnsSpecField{T}) where T
    error = zero(Float64)
    
    # Get local data
    new_real = parent(new_field.data_real)
    new_imag = parent(new_field.data_imag)
    old_real = parent(old_field.data_real)
    old_imag = parent(old_field.data_imag)
    
    # Compute local error with bounds checking for PencilArrays
    @inbounds for idx in eachindex(new_real, old_real)
        diff_real = new_real[idx] - old_real[idx]
        diff_imag = new_imag[idx] - old_imag[idx]
        error += diff_real^2 + diff_imag^2
    end
    
    # Global reduction across all MPI processes
    global_error = Allreduce(error, MPI.SUM, get_comm())
    return sqrt(global_error)
end

"""
    synchronize_pencil_transforms!(field::SHTnsSpecField{T}) where T

Ensure all pending PencilFFTs operations are completed and data is consistent across processes.
"""
function synchronize_pencil_transforms!(field::SHTnsSpecField{T}) where T
    # Synchronize data across pencil decomposition
    MPI.Barrier(get_comm())
    return field
end

"""
    validate_mpi_consistency!(field::SHTnsSpecField{T}) where T

Check that spectral field data is valid (no NaN/Inf) across all MPI processes.
Returns the field after validation. Warns if any process has invalid data.
"""
function validate_mpi_consistency!(field::SHTnsSpecField{T}) where T
    comm = get_comm()
    rank = get_rank()

    # Check local data for NaN/Inf
    real_data = parent(field.data_real)
    imag_data = parent(field.data_imag)

    local_nan_count = count(isnan, real_data) + count(isnan, imag_data)
    local_inf_count = count(isinf, real_data) + count(isinf, imag_data)
    local_has_issues = (local_nan_count > 0 || local_inf_count > 0) ? 1 : 0

    # Check if any process has issues
    global_has_issues = Allreduce(local_has_issues, MPI.MAX, comm)

    if global_has_issues > 0
        # Gather counts from all processes for detailed reporting
        total_nan = Allreduce(local_nan_count, MPI.SUM, comm)
        total_inf = Allreduce(local_inf_count, MPI.SUM, comm)

        if rank == 0
            @warn "MPI data validation failed: $total_nan NaN values, $total_inf Inf values across all processes"
        end
    end

    return field
end


# ================================================================================
# Exponential 2nd Order Runge-Kutta (ERK2) Implementation
# ================================================================================

# ---- ERK2 Boundary Condition Specification ----

"""
    ERK2BoundarySide{T}

Encodes how to enforce a boundary condition at one boundary (inner or outer)
for the ERK2 exponential integrator.

All BC types are expressed as a linear constraint:
    sum_j stencil[j] * u[j] + l_correction * u[b] = value

which is solved for u[b]:
    u[b] = (value - sum_{j≠b} stencil[j] * u[j]) / (stencil[b] + l_correction)

BC types and their encoding:
  - :dirichlet         → stencil = delta_{b}, value = BC_val  (u[b] = BC_val)
  - :neumann           → stencil = d1[b,:], value = q         (du/dr = q)
  - :stress_free_tor   → stencil = d1[b,:], l_correction = -1/r_b  (dT/dr - T/r = 0)
  - :noslip_pol        → stencil = d1[b,:], value = 0         (dP/dr = 0)
  - :stress_free_pol   → stencil = d2[b,:], value = 0         (d²P/dr² = 0)
  - :insulating_inner  → stencil = d1[b,:], l_correction = -l/r_b  (d/dr - l/r)P = 0
  - :insulating_outer  → stencil = d1[b,:], l_correction = +(l+1)/r_b  (d/dr + (l+1)/r)P = 0
"""
struct ERK2BoundarySide{T}
    type::Symbol              # BC type identifier
    value::T                  # Prescribed RHS value (BC_val for Dirichlet, flux for Neumann)
    stencil::Vector{T}        # Dense derivative stencil row (length nr)
    r_inv::T                  # 1/r at this boundary
    l_sign::T                 # Sign/multiplier for l-dependent correction (+1, -1, or 0)
    use_l_correction::Bool    # Whether correction is l-dependent
    fixed_correction::T       # Fixed additive correction to diagonal (e.g., -1/r for stress-free tor)
    l0_dirichlet::Bool        # Override to Dirichlet for l=0 (avoids underdetermined NN systems)
end

"""
    ERK2BoundarySpec{T}

Complete boundary condition specification for both boundaries.

`inner_mode_values` and `outer_mode_values` are optional per-mode value overrides
(indexed by lm_idx). When provided, they override `bc_side.value` for specific modes.
Used e.g. for rotating inner core: T(l=1,m=0) = rot_omega * r_inner.
"""
struct ERK2BoundarySpec{T}
    inner::ERK2BoundarySide{T}
    outer::ERK2BoundarySide{T}
    inner_mode_values::Union{Nothing, Vector{T}}
    outer_mode_values::Union{Nothing, Vector{T}}
end

# Convenience constructor without per-mode values
function ERK2BoundarySpec{T}(inner::ERK2BoundarySide{T}, outer::ERK2BoundarySide{T}) where T
    return ERK2BoundarySpec{T}(inner, outer, nothing, nothing)
end

"""
    enforce_erk2_bc!(result, bc_side, boundary_idx, l, nr)

Enforce a boundary condition on the result vector at the given boundary index.
Modifies `result[boundary_idx]` to satisfy the linear constraint encoded in `bc_side`.
"""
function enforce_erk2_bc!(result::AbstractVector{T}, bc_side::ERK2BoundarySide{T},
                          boundary_idx::Int, l::Int, nr::Int;
                          value_override::Union{T, Nothing}=nothing) where T
    b = boundary_idx
    effective_value = value_override !== nothing ? value_override : bc_side.value

    # l=0 override: use Dirichlet to avoid underdetermined NN systems
    # (matches CNAB2 treatment: pin value at inner boundary for l=0 when both are Neumann)
    if bc_side.l0_dirichlet && l == 0
        result[b] = effective_value
        return
    end

    # Dirichlet: just set the value directly
    if bc_side.type === :dirichlet
        result[b] = effective_value
        return
    end

    # Compute the effective diagonal coefficient at boundary:
    # self_coeff = stencil[b] + fixed_correction + l_sign * l * r_inv (if l-dependent)
    self_coeff = bc_side.stencil[b] + bc_side.fixed_correction
    if bc_side.use_l_correction
        self_coeff += bc_side.l_sign * T(l) * bc_side.r_inv
    end

    # Compute the off-diagonal contribution: sum_{j≠b} stencil[j] * result[j]
    off_diag_sum = zero(T)
    @inbounds for j in 1:nr
        if j != b
            off_diag_sum += bc_side.stencil[j] * result[j]
        end
    end

    # Solve: self_coeff * u[b] + off_diag_sum = value
    # → u[b] = (value - off_diag_sum) / self_coeff
    if abs(self_coeff) > eps(T) * T(100)
        result[b] = (effective_value - off_diag_sum) / self_coeff
    else
        # Degenerate case (e.g., l=0 with insulating BC): fall back to value
        result[b] = effective_value
    end
end

"""
    create_dirichlet_bc(T, nr, value) -> ERK2BoundarySide{T}

Create a Dirichlet BC side (u = value at boundary).
"""
function create_dirichlet_bc(::Type{T}, nr::Int, value::T=zero(T)) where T
    stencil = zeros(T, nr)
    return ERK2BoundarySide{T}(:dirichlet, value, stencil, zero(T), zero(T), false, zero(T), false)
end

"""
    create_neumann_bc(T, d1_row, value) -> ERK2BoundarySide{T}

Create a Neumann BC side (du/dr = value at boundary).
"""
function create_neumann_bc(::Type{T}, d1_row::Vector{T}, value::T=zero(T); l0_dirichlet::Bool=false) where T
    nr = length(d1_row)
    return ERK2BoundarySide{T}(:neumann, value, copy(d1_row), zero(T), zero(T), false, zero(T), l0_dirichlet)
end

"""
    create_stress_free_tor_bc(T, d1_row, r_inv) -> ERK2BoundarySide{T}

Create a stress-free toroidal BC side: dT/dr - T/r = 0.
"""
function create_stress_free_tor_bc(::Type{T}, d1_row::Vector{T}, r_inv::T) where T
    return ERK2BoundarySide{T}(:stress_free_tor, zero(T), copy(d1_row), r_inv, zero(T), false, -r_inv, false)
end

"""
    create_noslip_pol_bc(T, d1_row) -> ERK2BoundarySide{T}

Create a no-slip poloidal BC side: dP/dr = 0.
"""
function create_noslip_pol_bc(::Type{T}, d1_row::Vector{T}) where T
    return ERK2BoundarySide{T}(:noslip_pol, zero(T), copy(d1_row), zero(T), zero(T), false, zero(T), false)
end

"""
    create_stress_free_pol_bc(T, d2_row) -> ERK2BoundarySide{T}

Create a stress-free poloidal BC side: d²P/dr² = 0.
"""
function create_stress_free_pol_bc(::Type{T}, d2_row::Vector{T}) where T
    return ERK2BoundarySide{T}(:stress_free_pol, zero(T), copy(d2_row), zero(T), zero(T), false, zero(T), false)
end

"""
    create_insulating_inner_bc(T, d1_row, r_inv) -> ERK2BoundarySide{T}

Create an insulating magnetic poloidal inner BC: (d/dr - l/r)P = 0.
The l-factor is applied dynamically during enforcement.
"""
function create_insulating_inner_bc(::Type{T}, d1_row::Vector{T}, r_inv::T) where T
    return ERK2BoundarySide{T}(:insulating_inner, zero(T), copy(d1_row), r_inv, -one(T), true, zero(T), false)
end

"""
    create_insulating_outer_bc(T, d1_row, r_inv) -> ERK2BoundarySide{T}

Create an insulating magnetic poloidal outer BC: (d/dr + (l+1)/r)P = 0.
The (l+1) factor is split: fixed_correction = +1/r (the "+1" part),
and l_sign = +1 with use_l_correction = true (the "l" part).
Total correction = l_sign*l*r_inv + fixed_correction = (l+1)/r.
"""
function create_insulating_outer_bc(::Type{T}, d1_row::Vector{T}, r_inv::T) where T
    return ERK2BoundarySide{T}(:insulating_outer, zero(T), copy(d1_row), r_inv, one(T), true, r_inv, false)
end

# ================================================================================
# ERK2 INFLUENCE MATRIX FOR POLOIDAL VELOCITY
# ================================================================================
# The influence matrix method enforces P = 0 at boundaries while using derivative
# BCs (∂P/∂r = 0 for no-slip, ∂²P/∂r² = 0 for stress-free) in the exponential operator.
#
# Algorithm (matching DD_2DCODE vel_matrices):
# 1. Create "Green's function" problem with Dirichlet BCs at boundaries
# 2. Set RHS = [1,0,...,0] for inner Green's function, [0,...,0,1] for outer
# 3. Solve with Dirichlet system → get Green's function responses G₁(r), G₂(r)
# 4. Apply these through the derivative BC system → get final Green's functions
# 5. Construct 2×2 influence matrix: invG[i,j] = Gⱼ at boundary i
# 6. Invert the influence matrix
# 7. During timestepping: compute correction = invG * [P(r_inner), P(r_outer)]
#    and subtract Green's function * correction from the solution
# ================================================================================

"""
    ERK2InfluenceMatrix{T}

Precomputed influence matrix data for enforcing P = 0 at boundaries while
using derivative boundary conditions in the exponential operator.

This matches the Fortran DD_2DCODE influence matrix method for poloidal velocity.
"""
struct ERK2InfluenceMatrix{T}
    # Green's functions: Gre[:, 1] = response to inner BC, Gre[:, 2] = response to outer BC
    Gre::Matrix{T}           # (nr, 2) Green's function radial profiles
    invG::Matrix{T}          # (2, 2) inverse influence matrix
    l::Int                   # spherical harmonic degree
end

"""
    create_velocity_poloidal_influence_matrices(T, config, domain, diffusivity, dt, i_vel_bc; theta)

Create influence matrices for poloidal velocity to enforce P = 0 at boundaries
while using derivative BCs (∂P/∂r = 0 or ∂²P/∂r² = 0) in the implicit/exponential system.

This matches DD_2DCODE's vel_matrices routine.
"""
function create_velocity_poloidal_influence_matrices(::Type{T},
                                                      config::SHTnsKitConfig,
                                                      domain::RadialDomain,
                                                      diffusivity::Float64,
                                                      dt::Float64,
                                                      i_vel_bc::Int;
                                                      theta::Float64=d_implicit) where T
    unique_l = unique(config.l_values)
    nr = domain.N
    bw = i_KL

    # Create derivative matrices
    d1 = create_derivative_matrix(T, 1, domain)
    d2 = create_derivative_matrix(T, 2, domain)
    laplacian = create_radial_laplacian(domain)
    r_inv_sq = @views domain.r[1:nr, 2]

    # Diffusion operator base
    base_data = T.(diffusivity .* laplacian.data)

    influence_matrices = Dict{Int, ERK2InfluenceMatrix{T}}()

    for l in unique_l
        l == 0 && continue  # l=0 has no poloidal component

        # Build the diffusion operator for this l: A_l = ν * (∇² - l(l+1)/r²)
        A_data = copy(base_data)
        l_factor = Float64(l * (l + 1))
        @inbounds for n in 1:nr
            A_data[bw + 1, n] -= T(diffusivity * l_factor * r_inv_sq[n])
        end

        # ========================================
        # Step 1: Create Green's function system with Dirichlet BCs
        # ========================================
        # System matrix: X = (1/dt)I - θ*A
        inv_dt = T(1 / dt)
        θ_T = T(theta)

        Xgre_data = copy(A_data)
        Xgre_data .*= -θ_T
        Xgre_data[bw + 1, :] .+= inv_dt

        # Zero boundary rows
        @inbounds for j in 1:(1 + bw)
            Xgre_data[bw + 1 + 1 - j, j] = zero(T)
        end
        @inbounds for j in (nr - bw):nr
            Xgre_data[bw + 1 + nr - j, j] = zero(T)
        end

        # Dirichlet BCs (identity rows) at both boundaries
        Xgre_data[bw + 1, 1] = one(T)
        Xgre_data[bw + 1, nr] = one(T)

        Xgre = BandedMatrix{T}(Xgre_data, bw, nr)
        Xgre_lu = factorize_banded(Xgre)

        # ========================================
        # Step 2: Create physical BC system (derivative BCs)
        # ========================================
        Xpol_data = copy(A_data)
        Xpol_data .*= -θ_T
        Xpol_data[bw + 1, :] .+= inv_dt

        # Zero boundary rows
        @inbounds for j in 1:(1 + bw)
            Xpol_data[bw + 1 + 1 - j, j] = zero(T)
        end
        @inbounds for j in (nr - bw):nr
            Xpol_data[bw + 1 + nr - j, j] = zero(T)
        end

        # Apply poloidal derivative BCs (matching DD_2DCODE vel_bc_Pol)
        # Inner boundary
        if i_vel_bc == 1 || i_vel_bc == 2  # No-slip: ∂P/∂r = 0
            @inbounds for j in 1:(1 + bw)
                Xpol_data[bw + 1 + 1 - j, j] = d1.data[bw + 1 + 1 - j, j]
            end
        else  # Stress-free: ∂²P/∂r² = 0
            @inbounds for j in 1:(1 + bw)
                Xpol_data[bw + 1 + 1 - j, j] = d2.data[bw + 1 + 1 - j, j]
            end
        end

        # Outer boundary
        if i_vel_bc == 1 || i_vel_bc == 3  # No-slip: ∂P/∂r = 0
            @inbounds for j in (nr - bw):nr
                Xpol_data[bw + 1 + nr - j, j] = d1.data[bw + 1 + nr - j, j]
            end
        else  # Stress-free: ∂²P/∂r² = 0
            @inbounds for j in (nr - bw):nr
                Xpol_data[bw + 1 + nr - j, j] = d2.data[bw + 1 + nr - j, j]
            end
        end

        Xpol = BandedMatrix{T}(Xpol_data, bw, nr)
        Xpol_lu = factorize_banded(Xpol)

        # ========================================
        # Step 3: Compute Green's functions
        # ========================================
        Gre = zeros(T, nr, 2)

        # Green's function for inner boundary perturbation
        rhs = zeros(T, nr)
        rhs[1] = one(T)   # Unit perturbation at inner
        solve_banded!(rhs, Xgre_lu, rhs)  # Solve Dirichlet problem

        # Zero the BCs (matching Fortran tim_zerobc)
        rhs[1] = zero(T)
        rhs[nr] = zero(T)

        # Solve with physical (derivative) BCs
        solve_banded!(rhs, Xpol_lu, rhs)
        Gre[:, 1] = rhs

        # Green's function for outer boundary perturbation
        fill!(rhs, zero(T))
        rhs[nr] = one(T)  # Unit perturbation at outer
        solve_banded!(rhs, Xgre_lu, rhs)  # Solve Dirichlet problem

        # Zero the BCs
        rhs[1] = zero(T)
        rhs[nr] = zero(T)

        # Solve with physical BCs
        solve_banded!(rhs, Xpol_lu, rhs)
        Gre[:, 2] = rhs

        # ========================================
        # Step 4: Build and invert influence matrix
        # ========================================
        invG = zeros(T, 2, 2)
        invG[1, 1] = Gre[1, 1]   # Inner response to inner perturbation
        invG[1, 2] = Gre[1, 2]   # Inner response to outer perturbation
        invG[2, 1] = Gre[nr, 1]  # Outer response to inner perturbation
        invG[2, 2] = Gre[nr, 2]  # Outer response to outer perturbation

        # Invert the 2×2 matrix
        det = invG[1, 1] * invG[2, 2] - invG[1, 2] * invG[2, 1]
        if abs(det) > eps(T) * T(100)
            inv_det = one(T) / det
            tmp = invG[1, 1]
            invG[1, 1] = invG[2, 2] * inv_det
            invG[2, 2] = tmp * inv_det
            invG[1, 2] = -invG[1, 2] * inv_det
            invG[2, 1] = -invG[2, 1] * inv_det
        else
            # Singular matrix - use identity (no correction)
            invG[1, 1] = one(T)
            invG[2, 2] = one(T)
            invG[1, 2] = zero(T)
            invG[2, 1] = zero(T)
        end

        influence_matrices[l] = ERK2InfluenceMatrix{T}(Gre, invG, l)
    end

    return influence_matrices
end

"""
    apply_influence_matrix_correction!(result, influence, bc_inner_val, bc_outer_val)

Apply influence matrix correction to enforce P = 0 at boundaries.
This subtracts the Green's function response that would give non-zero boundary values.

After this correction, result[1] ≈ bc_inner_val and result[nr] ≈ bc_outer_val
(typically both zero for no-penetration).
"""
function apply_influence_matrix_correction!(result::AbstractVector{T},
                                             influence::ERK2InfluenceMatrix{T},
                                             bc_inner_val::T=zero(T),
                                             bc_outer_val::T=zero(T)) where T
    nr = length(result)

    # Current boundary values (before correction)
    P_inner = result[1]
    P_outer = result[nr]

    # Deviation from desired values
    delta_inner = P_inner - bc_inner_val
    delta_outer = P_outer - bc_outer_val

    # Compute correction coefficients: c = invG * [delta_inner; delta_outer]
    c1 = influence.invG[1, 1] * delta_inner + influence.invG[1, 2] * delta_outer
    c2 = influence.invG[2, 1] * delta_inner + influence.invG[2, 2] * delta_outer

    # Subtract Green's function response: result -= c1*G1 + c2*G2
    @inbounds for i in 1:nr
        result[i] -= c1 * influence.Gre[i, 1] + c2 * influence.Gre[i, 2]
    end

    return result
end

"""
    extract_dense_row(banded::BandedMatrix{T}, row::Int) -> Vector{T}

Extract a full dense row from a banded matrix.
"""
function extract_dense_row(data::Matrix{T}, bandwidth::Int, nr::Int, row::Int) where T
    result = zeros(T, nr)
    @inbounds for j in max(1, row - bandwidth):min(nr, row + bandwidth)
        band_idx = bandwidth + 1 + row - j
        if 1 <= band_idx <= 2*bandwidth + 1
            result[j] = data[band_idx, j]
        end
    end
    return result
end

"""
    ERK2Cache{T}

Cached data structure for Exponential 2nd Order Runge-Kutta method.
Stores precomputed matrix exponentials and φ functions for each spherical harmonic mode.
"""
struct ERK2Cache{T}
    dt::Float64
    diffusivity::Float64
    nr::Int
    l_values::Vector{Int}

    # Matrix exponentials: exp(dt/2 * A_l) and exp(dt * A_l)
    E_half::Vector{Matrix{T}}     # exp(dt/2 * A_l) per l
    E_full::Vector{Matrix{T}}     # exp(dt * A_l) per l

    # φ functions for ERK2 (both φ1 and φ2 needed for correct formula)
    phi1_half::Vector{Matrix{T}}  # φ1(dt/2 * A_l) per l
    phi1_full::Vector{Matrix{T}}  # φ1(dt * A_l) per l
    phi2_full::Vector{Matrix{T}}  # φ2(dt * A_l) per l

    # Krylov method parameters
    use_krylov::Bool
    krylov_m::Int
    krylov_tol::Float64

    # MPI-aware caching for distributed operations
    mpi_consistent::Bool
end

# Backward-compatible constructor for legacy cache bundles (pre-diffusivity metadata)
function ERK2Cache{T}(dt::Float64, l_values::Vector{Int},
                      E_half::Vector{Matrix{T}}, E_full::Vector{Matrix{T}},
                      phi1_half::Vector{Matrix{T}}, phi1_full::Vector{Matrix{T}},
                      phi2_full::Vector{Matrix{T}}, use_krylov::Bool,
                      krylov_m::Int, krylov_tol::Float64, mpi_consistent::Bool) where T
    nr = isempty(E_half) ? 0 : size(E_half[1], 1)
    return ERK2Cache{T}(dt, NaN, nr, l_values, E_half, E_full,
                        phi1_half, phi1_full, phi2_full,
                        use_krylov, krylov_m, krylov_tol, mpi_consistent)
end

ERK2Cache(args...) = ERK2Cache{Float64}(args...)

const ERK2_DIAGNOSTICS_ENABLED = Ref(false)
const ERK2_DIAGNOSTICS_INTERVAL = Ref(1)

function set_erk2_diagnostics_interval!(interval::Int)
    interval <= 0 && error("ERK2 diagnostics interval must be positive, got $interval")
    ERK2_DIAGNOSTICS_INTERVAL[] = interval
    return interval
end

function enable_erk2_diagnostics!(; interval::Int=ERK2_DIAGNOSTICS_INTERVAL[])
    set_erk2_diagnostics_interval!(interval)
    ERK2_DIAGNOSTICS_ENABLED[] = true
    return nothing
end

disable_erk2_diagnostics!() = (ERK2_DIAGNOSTICS_ENABLED[] = false; nothing)
erk2_diagnostics_enabled() = ERK2_DIAGNOSTICS_ENABLED[]
erk2_diagnostics_interval() = ERK2_DIAGNOSTICS_INTERVAL[]

let env_val = get(ENV, "GEODYNAMO_ERK2_DIAGNOSTICS", nothing)
    if env_val !== nothing
        enable = startswith(lowercase(strip(env_val)), "t") ||
                 lowercase(strip(env_val)) in ("1", "yes", "on")
        if enable
            interval_val = get(ENV, "GEODYNAMO_ERK2_DIAGNOSTICS_INTERVAL", "")
            interval = try
                isempty(interval_val) ? 1 : parse(Int, strip(interval_val))
            catch
                1
            end
            try
                enable_erk2_diagnostics!(interval=interval)
            catch e
                @warn "Failed to enable ERK2 diagnostics from environment: $e"
            end
        end
    end
end

"""
    create_erk2_cache(config, domain, diffusivity, dt; use_krylov=false, m=20, tol=1e-8, bc_spec=nothing)

Create ERK2 cache with precomputed matrix functions for all spherical harmonic modes.

Boundary rows of A are zeroed only for l=0 (where the spherical Laplacian has a null
space, making A singular). For l≥1, the full operator is retained (non-singular due to
the -l(l+1)/r² term), enabling accurate LU-based phi1/phi2 computation and O(h³)
enforcement corrections per step.
"""
function create_erk2_cache(::Type{T}, config::SHTnsKitConfig, domain::RadialDomain,
                          diffusivity::Float64, dt::Float64;
                          use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8,
                          bc_spec::Union{ERK2BoundarySpec{T}, Nothing}=nothing) where T
    
    lap = create_radial_laplacian(domain)
    nr = domain.N
    r_inv2 = @views domain.r[1:nr, 2]
    lvals = unique(config.l_values)
    
    E_half = Matrix{T}[]
    E_full = Matrix{T}[]
    phi1_half = Matrix{T}[]
    phi1_full = Matrix{T}[]
    phi2_full = Matrix{T}[]
    
    if get_rank() == 0
        @info "Creating ERK2 cache for $(length(lvals)) l-modes with $(use_krylov ? "Krylov" : "dense") methods"
    end
    
    for l in lvals
        # Build A_l = diffusivity * (d²/dr² + (2/r)d/dr - l(l+1)/r²)
        Adata = diffusivity .* lap.data
        Adense = banded_to_dense(BandedMatrix(Adata, i_KL, nr))
        lfac = Float64(l * (l + 1))

        @inbounds for n in 1:nr
            Adense[n, n] -= diffusivity * lfac * r_inv2[n]
        end

        # Zero boundary rows ONLY for l=0 where the spherical Laplacian
        # (d²/dr² + 2/r d/dr) has a null space (constant function), making A singular.
        # For l≥1, the -l(l+1)/r² term ensures A is non-singular, so the LU-based
        # phi1/phi2 computation works correctly and gives O(h³) enforcement corrections.
        # Zeroing for l≥1 would make A singular, forcing fallback to the Taylor series
        # and degrading boundary accuracy from O(h³) to O(h) per step.
        if l == 0
            Adense[1, :] .= zero(Float64)
            Adense[nr, :] .= zero(Float64)
        end

        if use_krylov
            # For large problems, we'll use Krylov methods during timestepping
            # Store only the operator for action-based computation
            push!(E_half, Adense)  # Store A for later Krylov action
            push!(E_full, Adense)
            push!(phi1_half, Adense)
            push!(phi1_full, Adense)
            push!(phi2_full, Adense)
        else
            # Dense computation of matrix functions
            Adt_half = (dt/2) .* Adense
            Adt_full = dt .* Adense
            
            # Compute exp(dt/2 * A) and exp(dt * A)
            E_half_l = exp(Adt_half)
            E_full_l = exp(Adt_full)
            push!(E_half, Matrix{T}(E_half_l))
            push!(E_full, Matrix{T}(E_full_l))
            
            # Compute φ1 functions: φ1(z) = (exp(z) - I) / z
            phi1_half_l = compute_phi1_function(Adt_half, E_half_l)
            phi1_full_l = compute_phi1_function(Adt_full, E_full_l)
            push!(phi1_half, Matrix{T}(phi1_half_l))
            push!(phi1_full, Matrix{T}(phi1_full_l))

            # Compute φ2 function: φ2(z) = (exp(z) - I - z) / z²
            phi2_full_l = compute_phi2_function(Adt_full, E_full_l; l=l)
            push!(phi2_full, Matrix{T}(phi2_full_l))
            
        end
    end
    
    # Ensure MPI consistency
    MPI.Barrier(get_comm())

    return ERK2Cache{T}(dt, diffusivity, nr, lvals, E_half, E_full, phi1_half, phi1_full,
                       phi2_full, use_krylov, m, tol, true)
end

"""
    create_erk2_cache_magnetic_poloidal(T, config, domain, diffusivity, dt; use_krylov=false, m=20, tol=1e-8)

Create ERK2 cache for magnetic poloidal field with insulating BCs embedded in the matrix A.

This matches DD_2DCODE's approach where the matrix has:
- Inner boundary row: (∂/∂r - l/r) operator → insulating interior
- Outer boundary row: (∂/∂r + (l+1)/r) operator → insulating exterior

Embedding BCs in A ensures exp(dt*A) automatically preserves solutions satisfying the
insulating constraints, rather than requiring post-evolution correction.
"""
function create_erk2_cache_magnetic_poloidal(::Type{T}, config::SHTnsKitConfig, domain::RadialDomain,
                                              diffusivity::Float64, dt::Float64;
                                              use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T
    lap = create_radial_laplacian(domain)
    d1 = create_derivative_matrix(T, 1, domain)
    nr = domain.N
    bw = i_KL
    r_inv2 = @views domain.r[1:nr, 2]
    r_inv = @views domain.r[1:nr, 3]  # 1/r values
    lvals = unique(config.l_values)

    E_half = Matrix{T}[]
    E_full = Matrix{T}[]
    phi1_half = Matrix{T}[]
    phi1_full = Matrix{T}[]
    phi2_full = Matrix{T}[]

    if get_rank() == 0
        @info "Creating ERK2 cache for magnetic poloidal with insulating BCs embedded"
    end

    for l in lvals
        # Build A_l = diffusivity * (d²/dr² + (2/r)d/dr - l(l+1)/r²)
        Adata = diffusivity .* lap.data
        Adense = banded_to_dense(BandedMatrix(Adata, bw, nr))
        lfac = Float64(l * (l + 1))

        @inbounds for n in 1:nr
            Adense[n, n] -= diffusivity * lfac * r_inv2[n]
        end

        # Zero boundary rows before embedding BCs
        Adense[1, :] .= zero(T)
        Adense[nr, :] .= zero(T)

        # Embed insulating BC at inner boundary: (∂/∂r - l/r)P = 0
        # Copy first derivative row and subtract l/r on diagonal
        for j in max(1, 1 - bw):min(nr, 1 + bw)
            band_idx = bw + 1 + 1 - j
            if 1 <= band_idx <= 2 * bw + 1
                Adense[1, j] = T(d1.data[band_idx, j])
            end
        end
        Adense[1, 1] -= T(l) * r_inv[1]

        # Embed insulating BC at outer boundary: (∂/∂r + (l+1)/r)P = 0
        # Copy first derivative row and add (l+1)/r on diagonal
        for j in max(1, nr - bw):min(nr, nr + bw)
            band_idx = bw + 1 + nr - j
            if 1 <= band_idx <= 2 * bw + 1
                Adense[nr, j] = T(d1.data[band_idx, j])
            end
        end
        Adense[nr, nr] += T(l + 1) * r_inv[nr]

        if use_krylov
            push!(E_half, Adense)
            push!(E_full, Adense)
            push!(phi1_half, Adense)
            push!(phi1_full, Adense)
            push!(phi2_full, Adense)
        else
            # Dense computation of matrix functions
            Adt_half = (dt / 2) .* Adense
            Adt_full = dt .* Adense

            # Compute exp(dt/2 * A) and exp(dt * A)
            E_half_l = exp(Adt_half)
            E_full_l = exp(Adt_full)
            push!(E_half, Matrix{T}(E_half_l))
            push!(E_full, Matrix{T}(E_full_l))

            # Compute φ1 functions
            phi1_half_l = compute_phi1_function(Adt_half, E_half_l)
            phi1_full_l = compute_phi1_function(Adt_full, E_full_l)
            push!(phi1_half, Matrix{T}(phi1_half_l))
            push!(phi1_full, Matrix{T}(phi1_full_l))

            # Compute φ2 function
            phi2_full_l = compute_phi2_function(Adt_full, E_full_l; l=l)
            push!(phi2_full, Matrix{T}(phi2_full_l))
        end
    end

    MPI.Barrier(get_comm())

    return ERK2Cache{T}(dt, diffusivity, nr, lvals, E_half, E_full, phi1_half, phi1_full,
                        phi2_full, use_krylov, m, tol, true)
end

"""
    create_erk2_cache_magnetic_toroidal(T, config, domain, diffusivity, dt; use_krylov=false, m=20, tol=1e-8)

Create ERK2 cache for magnetic toroidal field with insulating BCs embedded in the matrix A.

This matches DD_2DCODE's approach where the matrix has:
- Inner boundary row: identity → BT = 0
- Outer boundary row: identity → BT = 0

For Dirichlet BCs (BT = 0), we zero the boundary rows except for the diagonal (identity),
which makes exp(dt*A)|_boundary = identity, preserving BT = 0.
"""
function create_erk2_cache_magnetic_toroidal(::Type{T}, config::SHTnsKitConfig, domain::RadialDomain,
                                              diffusivity::Float64, dt::Float64;
                                              use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T
    lap = create_radial_laplacian(domain)
    nr = domain.N
    bw = i_KL
    r_inv2 = @views domain.r[1:nr, 2]
    lvals = unique(config.l_values)

    E_half = Matrix{T}[]
    E_full = Matrix{T}[]
    phi1_half = Matrix{T}[]
    phi1_full = Matrix{T}[]
    phi2_full = Matrix{T}[]

    if get_rank() == 0
        @info "Creating ERK2 cache for magnetic toroidal with Dirichlet BCs embedded"
    end

    for l in lvals
        # Build A_l = diffusivity * (d²/dr² + (2/r)d/dr - l(l+1)/r²)
        Adata = diffusivity .* lap.data
        Adense = banded_to_dense(BandedMatrix(Adata, bw, nr))
        lfac = Float64(l * (l + 1))

        @inbounds for n in 1:nr
            Adense[n, n] -= diffusivity * lfac * r_inv2[n]
        end

        # Zero boundary rows (Dirichlet: BT = 0)
        # This makes exp(dt*A)|_boundary = identity, preserving BT = 0
        Adense[1, :] .= zero(T)
        Adense[nr, :] .= zero(T)

        if use_krylov
            push!(E_half, Adense)
            push!(E_full, Adense)
            push!(phi1_half, Adense)
            push!(phi1_full, Adense)
            push!(phi2_full, Adense)
        else
            Adt_half = (dt / 2) .* Adense
            Adt_full = dt .* Adense

            E_half_l = exp(Adt_half)
            E_full_l = exp(Adt_full)
            push!(E_half, Matrix{T}(E_half_l))
            push!(E_full, Matrix{T}(E_full_l))

            phi1_half_l = compute_phi1_function(Adt_half, E_half_l)
            phi1_full_l = compute_phi1_function(Adt_full, E_full_l)
            push!(phi1_half, Matrix{T}(phi1_half_l))
            push!(phi1_full, Matrix{T}(phi1_full_l))

            phi2_full_l = compute_phi2_function(Adt_full, E_full_l; l=l)
            push!(phi2_full, Matrix{T}(phi2_full_l))
        end
    end

    MPI.Barrier(get_comm())

    return ERK2Cache{T}(dt, diffusivity, nr, lvals, E_half, E_full, phi1_half, phi1_full,
                        phi2_full, use_krylov, m, tol, true)
end

"""
    create_erk2_cache_scalar(T, config, domain, diffusivity, dt, i_bc; use_krylov=false, m=20, tol=1e-8)

Create ERK2 cache for scalar fields (temperature or composition) with boundary rows
zeroed in the matrix A.

Boundary condition types (matching DD_2DCODE tmp_bc_T / cmp_bc_C):
- i_bc = 1: Dirichlet-Dirichlet (fixed value both boundaries)
- i_bc = 2: Dirichlet-Neumann (fixed value inner, fixed flux outer)
- i_bc = 3: Neumann-Dirichlet (fixed flux inner, fixed value outer)
- i_bc = 4: Neumann-Neumann (fixed flux both, l=0 inner uses Dirichlet)

For exponential integrators, boundary rows are always zeroed:
- exp(dt*A)[boundary,:] = [1, 0, ..., 0] → preserves boundary value
- Actual BC enforcement (Dirichlet/Neumann) is done via post-processing
  in erk2_prepare_field! and erk2_finalize_field! using enforce_erk2_bc!
"""
function create_erk2_cache_scalar(::Type{T}, config::SHTnsKitConfig, domain::RadialDomain,
                                   diffusivity::Float64, dt::Float64, i_bc::Int;
                                   use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T
    lap = create_radial_laplacian(domain)
    d1 = create_derivative_matrix(T, 1, domain)
    nr = domain.N
    bw = i_KL
    r_inv2 = @views domain.r[1:nr, 2]
    lvals = unique(config.l_values)

    E_half = Matrix{T}[]
    E_full = Matrix{T}[]
    phi1_half = Matrix{T}[]
    phi1_full = Matrix{T}[]
    phi2_full = Matrix{T}[]

    bc_desc = ["DD", "DN", "ND", "NN"][clamp(i_bc, 1, 4)]
    if get_rank() == 0
        @info "Creating ERK2 cache for scalar field with embedded BCs (type=$bc_desc, ν=$diffusivity)"
    end

    for l in lvals
        # Build A_l = diffusivity * (d²/dr² + (2/r)d/dr - l(l+1)/r²)
        Adata = diffusivity .* lap.data
        Adense = banded_to_dense(BandedMatrix(Adata, bw, nr))
        lfac = Float64(l * (l + 1))

        @inbounds for n in 1:nr
            Adense[n, n] -= diffusivity * lfac * r_inv2[n]
        end

        # Zero boundary rows for exponential integrator
        # For exp(dt*A), zeroing the boundary row means:
        #   exp(dt*A)[boundary,:] = [1, 0, 0, ..., 0] (or [..., 0, 0, 1])
        # This preserves the boundary value during exponential evolution.
        # Actual BC enforcement is done via post-processing (enforce_erk2_bc!).
        #
        # This differs from implicit schemes where:
        #   Dirichlet → identity row (diagonal=1)
        #   Neumann → derivative row
        # For exponential integrators, we always zero boundary rows and rely
        # on post-processing to enforce the constraint.
        Adense[1, :] .= zero(T)
        Adense[nr, :] .= zero(T)

        # Note: BC type (Dirichlet/Neumann) and special l=0 handling for i_bc=4
        # are handled by the bc_spec post-processing in erk2_prepare_field! and
        # erk2_finalize_field! via enforce_erk2_bc!.

        if use_krylov
            push!(E_half, Adense)
            push!(E_full, Adense)
            push!(phi1_half, Adense)
            push!(phi1_full, Adense)
            push!(phi2_full, Adense)
        else
            Adt_half = (dt / 2) .* Adense
            Adt_full = dt .* Adense

            E_half_l = exp(Adt_half)
            E_full_l = exp(Adt_full)
            push!(E_half, Matrix{T}(E_half_l))
            push!(E_full, Matrix{T}(E_full_l))

            phi1_half_l = compute_phi1_function(Adt_half, E_half_l)
            phi1_full_l = compute_phi1_function(Adt_full, E_full_l)
            push!(phi1_half, Matrix{T}(phi1_half_l))
            push!(phi1_full, Matrix{T}(phi1_full_l))

            phi2_full_l = compute_phi2_function(Adt_full, E_full_l; l=l)
            push!(phi2_full, Matrix{T}(phi2_full_l))
        end
    end

    MPI.Barrier(get_comm())

    return ERK2Cache{T}(dt, diffusivity, nr, lvals, E_half, E_full, phi1_half, phi1_full,
                        phi2_full, use_krylov, m, tol, true)
end

# Convenience aliases for temperature and composition
"""
    create_erk2_cache_temperature(T, config, domain, diffusivity, dt, i_tmp_bc; kwargs...)

Create ERK2 cache for temperature field with embedded BCs.
Wrapper around create_erk2_cache_scalar with temperature-specific defaults.
"""
create_erk2_cache_temperature(::Type{T}, config::SHTnsKitConfig, domain::RadialDomain,
                               diffusivity::Float64, dt::Float64, i_tmp_bc::Int;
                               use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T =
    create_erk2_cache_scalar(T, config, domain, diffusivity, dt, i_tmp_bc;
                              use_krylov=use_krylov, m=m, tol=tol)

"""
    create_erk2_cache_composition(T, config, domain, diffusivity, dt, i_cmp_bc; kwargs...)

Create ERK2 cache for composition field with embedded BCs.
Wrapper around create_erk2_cache_scalar with composition-specific defaults.
"""
create_erk2_cache_composition(::Type{T}, config::SHTnsKitConfig, domain::RadialDomain,
                               diffusivity::Float64, dt::Float64, i_cmp_bc::Int;
                               use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T =
    create_erk2_cache_scalar(T, config, domain, diffusivity, dt, i_cmp_bc;
                              use_krylov=use_krylov, m=m, tol=tol)

"""
    compute_phi1_function(A, expA)

Compute φ1(A) = (exp(A) - I) / A efficiently with comprehensive error handling.
Uses series expansion for small ||A|| to avoid numerical issues.
"""
function compute_phi1_function(A::Matrix{T}, expA::Matrix{T}) where T
    nr = size(A, 1)
    I_mat = Matrix{T}(I, nr, nr)

    # Check for NaN or Inf in inputs
    if !all(isfinite.(A)) || !all(isfinite.(expA))
        @warn "Non-finite values detected in φ1 computation, using identity approximation"
        return I_mat
    end

    # Check if A is close to zero matrix - use series expansion
    A_norm = opnorm(A)
    if A_norm < 1e-2
        # Use Taylor series: φ1(A) = Σ(k=0 to ∞) A^k/(k+1)! = I/1! + A/2! + A²/3! + A³/4! + ...
        result = copy(I_mat)  # k=0: A⁰/1! = I/1!
        A_power = copy(I_mat)
        for k in 1:15  # Use enough terms for good accuracy
            A_power = A_power * A  # A^k
            factorial_k_plus_1 = factorial(k + 1)  # (k+1)!
            term = A_power / factorial_k_plus_1
            result += term
            if opnorm(term) < eps(T) * 100
                break
            end
        end
        return result
    end

    # For larger A, use φ1(A) = (exp(A) - I) / A
    diff = expA - I_mat

    # Use lu factorization for stable division by A
    try
        lu_A = lu(A)

        # Check condition number
        if rcond(lu_A) < sqrt(eps(T))
            @warn "Ill-conditioned matrix in φ1 computation (rcond = $(rcond(lu_A))), using series expansion"
            # Fall back to series expansion: φ1(A) = Σ(k=0 to ∞) A^k/(k+1)!
            result = copy(I_mat)  # k=0: A⁰/1!
            A_power = copy(I_mat)
            for k in 1:15
                A_power = A_power * A  # A^k
                factorial_k_plus_1 = factorial(k + 1)  # (k+1)!
                term = A_power / factorial_k_plus_1
                result += term
                if opnorm(term) < eps(T) * 100
                    break
                end
            end
            return result
        else
            result = lu_A \ diff
        end

        # Validate result
        if !all(isfinite.(result))
            @warn "Non-finite result in φ1 computation, falling back to series expansion"
            result = I_mat + A/2
            A_power = A * A
            factorial = 6
            for k in 2:15
                term = A_power / factorial
                result += term
                if opnorm(term) < eps(T) * 100
                    break
                end
                A_power = A_power * A
                factorial *= (k + 2)
            end
        end

        return result

    catch e
        @warn "LU factorization failed in φ1 computation: $e, using series expansion"
        try
            # Fall back to series expansion: φ1(A) = Σ(k=0 to ∞) A^k/(k+1)!
            result = copy(I_mat)  # k=0: A⁰/1!
            A_power = copy(I_mat)
            for k in 1:15
                A_power = A_power * A  # A^k
                factorial_k_plus_1 = factorial(k + 1)  # (k+1)!
                term = A_power / factorial_k_plus_1
                result += term
                if opnorm(term) < eps(T) * 100
                    break
                end
            end
            return result
        catch e2
            @error "Complete failure in φ1 computation: $e2, returning identity"
            return I_mat
        end
    end
end

"""
    Phi2ConditioningMonitor

Global structure for monitoring φ₂ function conditioning during ERK2 integration.
Tracks worst conditioning, series expansion usage, and numerical issues.
"""
mutable struct Phi2ConditioningMonitor
    worst_rcond::Float64
    worst_l::Int
    series_expansion_count::Int
    lu_failure_count::Int
    nonfinite_count::Int
    last_report_step::Int
    enable_monitoring::Bool
end

# Global monitor instance
const PHI2_MONITOR = Phi2ConditioningMonitor(1.0, 0, 0, 0, 0, 0, true)

"""
    reset_phi2_monitor!()

Reset φ₂ conditioning monitor statistics.
"""
function reset_phi2_monitor!()
    PHI2_MONITOR.worst_rcond = 1.0
    PHI2_MONITOR.worst_l = 0
    PHI2_MONITOR.series_expansion_count = 0
    PHI2_MONITOR.lu_failure_count = 0
    PHI2_MONITOR.nonfinite_count = 0
    PHI2_MONITOR.last_report_step = 0
end

"""
    report_phi2_conditioning(step::Int; interval::Int=100)

Report φ₂ conditioning statistics periodically during simulation.
"""
function report_phi2_conditioning(step::Int; interval::Int=100)
    if !PHI2_MONITOR.enable_monitoring
        return
    end

    if step - PHI2_MONITOR.last_report_step >= interval
        if get_rank() == 0
            @info """
            ╔══════════════════════════════════════════════════════════╗
            ║            φ₂ Conditioning Report (Step $step)
            ╠══════════════════════════════════════════════════════════╣
            ║ Worst rcond:             $(PHI2_MONITOR.worst_rcond)
            ║ Worst mode (l):          $(PHI2_MONITOR.worst_l)
            ║ Series expansion used:   $(PHI2_MONITOR.series_expansion_count) times
            ║ LU failures:             $(PHI2_MONITOR.lu_failure_count) times
            ║ Non-finite values:       $(PHI2_MONITOR.nonfinite_count) times
            ╚══════════════════════════════════════════════════════════╝
            """
        end
        PHI2_MONITOR.last_report_step = step
        # Reset counters for next interval
        reset_phi2_monitor!()
    end
end

"""
    compute_phi2_function(A, expA; l=0)

Compute φ2(A) = (exp(A) - I - A) / A² efficiently with comprehensive error handling.
Uses series expansion for small ||A|| to avoid numerical issues.
Tracks conditioning statistics when monitoring is enabled.
"""
function compute_phi2_function(A::Matrix{T}, expA::Matrix{T}; l::Int=0) where T
    nr = size(A, 1)
    I_mat = Matrix{T}(I, nr, nr)

    # Check for NaN or Inf in inputs
    if !all(isfinite.(A)) || !all(isfinite.(expA))
        if PHI2_MONITOR.enable_monitoring
            PHI2_MONITOR.nonfinite_count += 1
        end
        @warn "Non-finite values detected in φ2 computation (l=$l), using zero approximation"
        return zeros(T, nr, nr)
    end

    # Check if A is close to zero matrix - use series expansion
    A_norm = opnorm(A)
    if A_norm < 1e-2
        # Use Taylor series: φ2(A) = Σ(k=0 to ∞) A^k/(k+2)! = I/2! + A/3! + A²/4! + A³/5! + ...
        result = I_mat / 2  # k=0: A⁰/2! = I/2
        A_power = copy(A)   # k=1: A¹/3!
        factorial = 6       # 3! = 6
        result += A_power / factorial
        
        for k in 2:15  # Use enough terms for good accuracy
            A_power = A_power * A  # A^k
            factorial *= (k + 2)   # (k+2)! 
            term = A_power / factorial
            result += term
            if opnorm(term) < eps(T) * 100
                break
            end
        end
        return result
    end

    # For larger A, use φ2(A) = (exp(A) - I - A) / A²
    diff = expA - I_mat - A

    # Need to solve A² * result = diff
    # This is equivalent to solving A * (A * result) = diff
    try
        lu_A = lu(A)

        # Check condition number and track worst conditioning
        rcond_val = rcond(lu_A)
        if PHI2_MONITOR.enable_monitoring
            if rcond_val < PHI2_MONITOR.worst_rcond
                PHI2_MONITOR.worst_rcond = rcond_val
                PHI2_MONITOR.worst_l = l
            end
        end

        if rcond_val < sqrt(eps(T))
            if PHI2_MONITOR.enable_monitoring
                PHI2_MONITOR.series_expansion_count += 1
            end
            @warn "Ill-conditioned matrix in φ2 computation (l=$l, rcond=$rcond_val), using series expansion"
            # Fall back to series expansion: φ2(A) = Σ(k=0 to ∞) A^k/(k+2)!
            result = I_mat / 2  # k=0: A⁰/2! = I/2
            A_power = copy(A)   # k=1: A¹/3!
            factorial = 6       # 3! = 6
            result += A_power / factorial
            
            for k in 2:15
                A_power = A_power * A  # A^k
                factorial *= (k + 2)   # (k+2)!
                term = A_power / factorial
                result += term
                if opnorm(term) < eps(T) * 100
                    break
                end
            end
            return result
        else
            # Solve A * temp = diff, then A * result = temp
            temp = lu_A \ diff
            result = lu_A \ temp
        end

        # Validate result
        if !all(isfinite.(result))
            @warn "Non-finite result in φ2 computation, falling back to series expansion"
            result = I_mat / 2 + A / 6
            A_power = A * A
            factorial = 24
            for k in 2:15
                term = A_power / factorial
                result += term
                if opnorm(term) < eps(T) * 100
                    break
                end
                A_power = A_power * A
                factorial *= (k + 3)
            end
        end

        return result

    catch e
        if PHI2_MONITOR.enable_monitoring
            PHI2_MONITOR.lu_failure_count += 1
        end
        @warn "LU factorization failed in φ2 computation (l=$l): $e, using series expansion"
        try
            # Fall back to series expansion: φ2(A) = Σ(k=0 to ∞) A^k/(k+2)!
            result = I_mat / 2  # k=0: A⁰/2! = I/2
            A_power = copy(A)   # k=1: A¹/3!
            factorial = 6       # 3! = 6
            result += A_power / factorial
            
            for k in 2:15
                A_power = A_power * A  # A^k
                factorial *= (k + 2)   # (k+2)!
                term = A_power / factorial
                result += term
                if opnorm(term) < eps(T) * 100
                    break
                end
            end
            return result
        catch e2
            @error "Complete failure in φ2 computation: $e2, returning zero matrix"
            return zeros(T, nr, nr)
        end
    end
end


"""
    get_erk2_cache!(caches, key, diffusivity, config, domain, dt; use_krylov=false)

Retrieve or create ERK2 cache with automatic invalidation when parameters change.
Type-stable version using concrete ERK2Cache{T} type.
"""
function get_erk2_cache!(caches::Dict{Symbol, ERK2Cache{T}}, key::Symbol, diffusivity::Float64,
                        ::Type{T}, config::SHTnsKitConfig, domain::RadialDomain, dt::Float64;
                        use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8,
                        bc_spec::Union{ERK2BoundarySpec{T}, Nothing}=nothing) where T
    nr = domain.N
    cache = get(caches, key, nothing)

    needs_rebuild = cache === nothing ||
                    cache.diffusivity != diffusivity ||
                    cache.nr != nr ||
                    cache.dt != dt ||
                    cache.use_krylov != use_krylov ||
                    !cache.mpi_consistent

    if needs_rebuild
        if get_rank() == 0
            @info "Creating new ERK2 cache for $key (ν=$diffusivity, nr=$nr, dt=$dt)"
        end

        cache = create_erk2_cache(T, config, domain, diffusivity, dt;
                                  use_krylov=use_krylov, m=m, tol=tol, bc_spec=bc_spec)
    end

    caches[key] = cache
    return cache::ERK2Cache{T}  # Type assertion for compiler optimization
end

"""
    get_erk2_magnetic_toroidal_cache!(caches, diffusivity, T, config, domain, dt; use_krylov=false, m=20, tol=1e-8)

Retrieve or create ERK2 cache for magnetic toroidal field with embedded insulating BCs.
Uses Dirichlet BCs (BT = 0) at both boundaries, matching DD_2DCODE's mag_bc_Tor.
"""
function get_erk2_magnetic_toroidal_cache!(caches::Dict{Symbol, ERK2Cache{T}}, diffusivity::Float64,
                                           ::Type{T}, config::SHTnsKitConfig, domain::RadialDomain, dt::Float64;
                                           use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T
    key = :magnetic_toroidal_embedded
    nr = domain.N
    cache = get(caches, key, nothing)

    needs_rebuild = cache === nothing ||
                    cache.diffusivity != diffusivity ||
                    cache.nr != nr ||
                    cache.dt != dt ||
                    cache.use_krylov != use_krylov ||
                    !cache.mpi_consistent

    if needs_rebuild
        if get_rank() == 0
            @info "Creating magnetic toroidal ERK2 cache with embedded insulating BCs (ν=$diffusivity, nr=$nr, dt=$dt)"
        end
        cache = create_erk2_cache_magnetic_toroidal(T, config, domain, diffusivity, dt;
                                                     use_krylov=use_krylov, m=m, tol=tol)
    end

    caches[key] = cache
    return cache::ERK2Cache{T}
end

"""
    get_erk2_magnetic_poloidal_cache!(caches, diffusivity, T, config, domain, dt; use_krylov=false, m=20, tol=1e-8)

Retrieve or create ERK2 cache for magnetic poloidal field with embedded insulating BCs.
Uses l-dependent insulating BCs matching DD_2DCODE's mag_bc_Pol:
- Inner boundary: (∂/∂r - l/r)P = 0
- Outer boundary: (∂/∂r + (l+1)/r)P = 0
"""
function get_erk2_magnetic_poloidal_cache!(caches::Dict{Symbol, ERK2Cache{T}}, diffusivity::Float64,
                                           ::Type{T}, config::SHTnsKitConfig, domain::RadialDomain, dt::Float64;
                                           use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T
    key = :magnetic_poloidal_embedded
    nr = domain.N
    cache = get(caches, key, nothing)

    needs_rebuild = cache === nothing ||
                    cache.diffusivity != diffusivity ||
                    cache.nr != nr ||
                    cache.dt != dt ||
                    cache.use_krylov != use_krylov ||
                    !cache.mpi_consistent

    if needs_rebuild
        if get_rank() == 0
            @info "Creating magnetic poloidal ERK2 cache with embedded insulating BCs (ν=$diffusivity, nr=$nr, dt=$dt)"
        end
        cache = create_erk2_cache_magnetic_poloidal(T, config, domain, diffusivity, dt;
                                                     use_krylov=use_krylov, m=m, tol=tol)
    end

    caches[key] = cache
    return cache::ERK2Cache{T}
end

"""
    get_erk2_temperature_cache!(caches, diffusivity, T, config, domain, dt, i_tmp_bc; use_krylov=false, m=20, tol=1e-8)

Retrieve or create ERK2 cache for temperature field with embedded BCs.
Uses BC type specified by i_tmp_bc (1=DD, 2=DN, 3=ND, 4=NN), matching DD_2DCODE's tmp_bc_T.
"""
function get_erk2_temperature_cache!(caches::Dict{Symbol, ERK2Cache{T}}, diffusivity::Float64,
                                      ::Type{T}, config::SHTnsKitConfig, domain::RadialDomain,
                                      dt::Float64, i_tmp_bc::Int;
                                      use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T
    key = Symbol(:temperature_embedded_bc, i_tmp_bc)
    nr = domain.N
    cache = get(caches, key, nothing)

    needs_rebuild = cache === nothing ||
                    cache.diffusivity != diffusivity ||
                    cache.nr != nr ||
                    cache.dt != dt ||
                    cache.use_krylov != use_krylov ||
                    !cache.mpi_consistent

    if needs_rebuild
        bc_desc = ["DD", "DN", "ND", "NN"][clamp(i_tmp_bc, 1, 4)]
        if get_rank() == 0
            @info "Creating temperature ERK2 cache with embedded BCs (type=$bc_desc, ν=$diffusivity, nr=$nr, dt=$dt)"
        end
        cache = create_erk2_cache_temperature(T, config, domain, diffusivity, dt, i_tmp_bc;
                                               use_krylov=use_krylov, m=m, tol=tol)
    end

    caches[key] = cache
    return cache::ERK2Cache{T}
end

"""
    get_erk2_composition_cache!(caches, diffusivity, T, config, domain, dt, i_cmp_bc; use_krylov=false, m=20, tol=1e-8)

Retrieve or create ERK2 cache for composition field with embedded BCs.
Uses BC type specified by i_cmp_bc (1=DD, 2=DN, 3=ND, 4=NN), matching DD_2DCODE's cmp_bc_C.
"""
function get_erk2_composition_cache!(caches::Dict{Symbol, ERK2Cache{T}}, diffusivity::Float64,
                                      ::Type{T}, config::SHTnsKitConfig, domain::RadialDomain,
                                      dt::Float64, i_cmp_bc::Int;
                                      use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T
    key = Symbol(:composition_embedded_bc, i_cmp_bc)
    nr = domain.N
    cache = get(caches, key, nothing)

    needs_rebuild = cache === nothing ||
                    cache.diffusivity != diffusivity ||
                    cache.nr != nr ||
                    cache.dt != dt ||
                    cache.use_krylov != use_krylov ||
                    !cache.mpi_consistent

    if needs_rebuild
        bc_desc = ["DD", "DN", "ND", "NN"][clamp(i_cmp_bc, 1, 4)]
        if get_rank() == 0
            @info "Creating composition ERK2 cache with embedded BCs (type=$bc_desc, ν=$diffusivity, nr=$nr, dt=$dt)"
        end
        cache = create_erk2_cache_composition(T, config, domain, diffusivity, dt, i_cmp_bc;
                                               use_krylov=use_krylov, m=m, tol=tol)
    end

    caches[key] = cache
    return cache::ERK2Cache{T}
end

function _normalize_erk2_cache_entry(entry)
    if entry isa ERK2Cache
        return entry
    elseif entry isa Dict
        cache = get(entry, :cache, nothing)
        return cache isa ERK2Cache ? cache : nothing
    elseif entry === nothing
        return nothing
    else
        return nothing
    end
end

"""
    save_erk2_cache_bundle(path, caches; metadata=Dict())

Persist a dictionary of ERK2 caches to disk along with optional metadata.
"""
function save_erk2_cache_bundle(path::AbstractString,
                                caches::Dict{Symbol,<:ERK2Cache};
                                metadata::Dict{String,Any}=Dict{String,Any}())
    meta = Dict{String,Any}(metadata)
    meta["created_at"] = get(meta, "created_at", string(now()))
    jldopen(path, "w") do file
        file["caches"] = caches
        file["metadata"] = meta
    end
    return path
end

function save_erk2_cache_bundle(path::AbstractString,
                                caches::Dict{Symbol,Any};
                                metadata::Dict{String,Any}=Dict{String,Any}())
    bundle = Dict{Symbol,ERK2Cache}()
    for (key, value) in caches
        cache = _normalize_erk2_cache_entry(value)
        cache === nothing && continue
        bundle[key] = cache
    end
    return save_erk2_cache_bundle(path, bundle; metadata=metadata)
end

"""
    load_erk2_cache_bundle(path) -> (caches, metadata)

Load ERK2 caches and associated metadata from disk.
"""
function load_erk2_cache_bundle(path::AbstractString)
    caches = Dict{Symbol,Any}()
    metadata = Dict{String,Any}()
    jldopen(path, "r") do file
        caches = Dict{Symbol,Any}(file["caches"])
        metadata = haskey(file, "metadata") ? Dict{String,Any}(file["metadata"]) : Dict{String,Any}()
    end
    return caches, metadata
end

"""
    install_erk2_cache_bundle!(target, bundle)

Copy ERK2 caches from `bundle` into the target cache dictionary.
"""
function install_erk2_cache_bundle!(target::Dict{Symbol,Any},
                                    bundle::Dict{Symbol,<:Any})
    for (key, value) in bundle
        cache = _normalize_erk2_cache_entry(value)
        cache === nothing && continue
        target[key] = cache
    end
    return target
end

"""
    load_erk2_cache_bundle!(target, path) -> metadata

Load caches from `path` and install them into `target`, returning metadata.
"""
function load_erk2_cache_bundle!(target::Dict{Symbol,Any}, path::AbstractString)
    bundle, metadata = load_erk2_cache_bundle(path)
    install_erk2_cache_bundle!(target, bundle)
    return metadata
end

# ================================================================================
# ERK2 helper utilities for staged evaluation
# ================================================================================

struct ERK2FieldBuffers{T}
    linear_real::Array{T,3}
    linear_imag::Array{T,3}
    k1_real::Array{T,3}
    k1_imag::Array{T,3}
    stage_real::Array{T,3}
    stage_imag::Array{T,3}
    n_current_real::Array{T,3}
    n_current_imag::Array{T,3}
    stage_nl_real::Array{T,3}
    stage_nl_imag::Array{T,3}
    cache_lookup::Dict{Int,Int}
    nr::Int
end

function ERK2FieldBuffers(u::SHTnsSpecField{T}, nl::SHTnsSpecField{T}, cache::ERK2Cache{T}) where T
    isempty(cache.E_full) && error("ERK2 cache has no precomputed matrices")
    real_data = parent(u.data_real)
    imag_data = parent(u.data_imag)
    nl_real = parent(nl.data_real)
    nl_imag = parent(nl.data_imag)
    cache_lookup = Dict{Int,Int}(l => idx for (idx, l) in enumerate(cache.l_values))
    nr = size(cache.E_full[1], 1)
    return ERK2FieldBuffers{T}(
        similar(real_data), similar(imag_data),
        similar(real_data), similar(imag_data),
        similar(real_data), similar(imag_data),
        similar(nl_real), similar(nl_imag),
        similar(nl_real), similar(nl_imag),
        cache_lookup, nr
    )
end

"""
    erk2_prepare_field!(buffers, u, nl, cache, config, dt)

Prepare ERK2 first stage by computing linear evolution and k1 terms.

# MPI Safety
Uses global loop bounds (1:nlm) to ensure all processes call Allreduce
the same number of times, preventing deadlock with uneven lm distribution.
"""
function erk2_prepare_field!(buffers::ERK2FieldBuffers{T}, u::SHTnsSpecField{T},
                             nl::SHTnsSpecField{T}, cache::ERK2Cache{T},
                             config::SHTnsKitConfig, dt::Float64;
                             bc_spec::Union{ERK2BoundarySpec{T}, Nothing}=nothing) where T
    cache.use_krylov && error("Krylov-based ERK2 caches are not supported in staged integration")

    u_real = parent(u.data_real)
    u_imag = parent(u.data_imag)
    nl_real = parent(nl.data_real)
    nl_imag = parent(nl.data_imag)

    copyto!(buffers.n_current_real, nl_real)
    copyto!(buffers.n_current_imag, nl_imag)

    lm_range = get_local_range(u.pencil, 1)
    r_range = get_local_range(u.pencil, 3)

    nr = buffers.nr
    ur = zeros(T, nr)
    ui = similar(ur)
    nr_vec = similar(ur)
    ni_vec = similar(ur)
    linear_tmp = similar(ur)
    k1_tmp = similar(ur)
    stage_tmp = similar(ur)
    stage_phi_tmp = similar(ur)
    half_dt = T(dt) / T(2)

    comm = get_comm()
    multi = MPI.Comm_size(comm) > 1
    nlm_total = u.nlm

    linear_real = buffers.linear_real
    linear_imag = buffers.linear_imag
    k1_real = buffers.k1_real
    k1_imag = buffers.k1_imag
    stage_real = buffers.stage_real
    stage_imag = buffers.stage_imag

    # Use GLOBAL loop bounds to ensure all processes call Allreduce same number of times
    for lm_idx in 1:nlm_total
        # Check if this process owns this lm mode
        owns_mode = lm_idx in lm_range

        l = config.l_values[lm_idx]
        cache_idx = get(buffers.cache_lookup, l, nothing)
        cache_idx === nothing && error("Missing ERK2 cache entry for l=$l")

        E_full = cache.E_full[cache_idx]
        E_half = cache.E_half[cache_idx]
        phi1_full = cache.phi1_full[cache_idx]
        phi1_half = cache.phi1_half[cache_idx]

        fill!(ur, zero(T)); fill!(ui, zero(T))
        fill!(nr_vec, zero(T)); fill!(ni_vec, zero(T))

        # Only fill if this process owns the mode
        if owns_mode
            ll = lm_idx - first(lm_range) + 1

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    ur[r] = u_real[ll, 1, lr]
                    ui[r] = u_imag[ll, 1, lr]
                    nr_vec[r] = buffers.n_current_real[ll, 1, lr]
                    ni_vec[r] = buffers.n_current_imag[ll, 1, lr]
                end
            end
        end

        # ALL processes call Allreduce together (collective operation)
        if multi
            Allreduce!(ur, MPI.SUM, comm)
            Allreduce!(ui, MPI.SUM, comm)
            Allreduce!(nr_vec, MPI.SUM, comm)
            Allreduce!(ni_vec, MPI.SUM, comm)
        end

        mul!(linear_tmp, E_full, ur)
        mul!(k1_tmp, phi1_full, nr_vec)

        # Scatter back only if this process owns the mode
        if owns_mode
            ll = lm_idx - first(lm_range) + 1

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    linear_real[ll, 1, lr] = linear_tmp[r]
                    k1_real[ll, 1, lr] = k1_tmp[r]
                end
            end
        end

        mul!(stage_tmp, E_half, ur)
        mul!(stage_phi_tmp, phi1_half, nr_vec)
        @. stage_tmp = stage_tmp + half_dt * stage_phi_tmp
        # Enforce boundary conditions (real part)
        if bc_spec !== nothing
            inner_val = bc_spec.inner_mode_values !== nothing && lm_idx <= length(bc_spec.inner_mode_values) ?
                        bc_spec.inner_mode_values[lm_idx] : nothing
            outer_val = bc_spec.outer_mode_values !== nothing && lm_idx <= length(bc_spec.outer_mode_values) ?
                        bc_spec.outer_mode_values[lm_idx] : nothing
            enforce_erk2_bc!(stage_tmp, bc_spec.inner, 1, l, nr; value_override=inner_val)
            enforce_erk2_bc!(stage_tmp, bc_spec.outer, nr, l, nr; value_override=outer_val)
        else
            stage_tmp[1] = zero(T)
            stage_tmp[nr] = zero(T)
        end

        # Scatter stage results only if this process owns the mode
        if owns_mode
            ll = lm_idx - first(lm_range) + 1

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    stage_real[ll, 1, lr] = stage_tmp[r]
                end
            end
        end

        mul!(linear_tmp, E_full, ui)
        mul!(k1_tmp, phi1_full, ni_vec)

        # Scatter imaginary linear/k1 results only if this process owns the mode
        if owns_mode
            ll = lm_idx - first(lm_range) + 1

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_imag, 3)
                    linear_imag[ll, 1, lr] = linear_tmp[r]
                    k1_imag[ll, 1, lr] = k1_tmp[r]
                end
            end
        end

        mul!(stage_tmp, E_half, ui)
        mul!(stage_phi_tmp, phi1_half, ni_vec)
        @. stage_tmp = stage_tmp + half_dt * stage_phi_tmp
        # Enforce boundary conditions
        if bc_spec !== nothing
            enforce_erk2_bc!(stage_tmp, bc_spec.inner, 1, l, nr)
            enforce_erk2_bc!(stage_tmp, bc_spec.outer, nr, l, nr)
        else
            stage_tmp[1] = zero(T)
            stage_tmp[nr] = zero(T)
        end

        # Scatter imaginary stage results only if this process owns the mode
        if owns_mode
            ll = lm_idx - first(lm_range) + 1

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_imag, 3)
                    stage_imag[ll, 1, lr] = stage_tmp[r]
                end
            end
        end
    end

    return buffers
end

erk2_apply_stage!(buffers::ERK2FieldBuffers{T}, u::SHTnsSpecField{T}) where T = begin
    parent(u.data_real) .= buffers.stage_real
    parent(u.data_imag) .= buffers.stage_imag
    return u
end

erk2_store_stage_nonlinear!(buffers::ERK2FieldBuffers{T}, nl::SHTnsSpecField{T}) where T = begin
    parent_nl_real = parent(nl.data_real)
    parent_nl_imag = parent(nl.data_imag)
    copyto!(buffers.stage_nl_real, parent_nl_real)
    copyto!(buffers.stage_nl_imag, parent_nl_imag)
    return buffers
end

"""
    erk2_finalize_field!(buffers, u, cache, config, dt)

Finalize ERK2 second stage by applying phi2 correction.

# MPI Safety
Uses global loop bounds (1:nlm) to ensure all processes call Allreduce
the same number of times, preventing deadlock with uneven lm distribution.
"""
function erk2_finalize_field!(buffers::ERK2FieldBuffers{T}, u::SHTnsSpecField{T},
                              cache::ERK2Cache{T}, config::SHTnsKitConfig, dt::Float64;
                              bc_spec::Union{ERK2BoundarySpec{T}, Nothing}=nothing) where T
    cache.use_krylov && error("Krylov-based ERK2 caches are not supported in staged integration")

    u_real = parent(u.data_real)
    u_imag = parent(u.data_imag)

    lm_range = get_local_range(u.pencil, 1)
    r_range = get_local_range(u.pencil, 3)

    nr = buffers.nr
    tmp_linear = zeros(T, nr)
    tmp_k1 = similar(tmp_linear)
    tmp_Nn = similar(tmp_linear)
    tmp_stage = similar(tmp_linear)
    delta = similar(tmp_linear)
    correction = similar(tmp_linear)
    result = similar(tmp_linear)

    comm = get_comm()
    multi = MPI.Comm_size(comm) > 1
    nlm_total = u.nlm

    # Use GLOBAL loop bounds to ensure all processes call Allreduce same number of times
    for lm_idx in 1:nlm_total
        # Check if this process owns this lm mode
        owns_mode = lm_idx in lm_range

        l = config.l_values[lm_idx]
        cache_idx = get(buffers.cache_lookup, l, nothing)
        cache_idx === nothing && error("Missing ERK2 cache entry for l=$l")
        phi2 = cache.phi2_full[cache_idx]

        # Initialize buffers - all processes
        fill!(tmp_linear, zero(T))
        fill!(tmp_k1, zero(T))
        fill!(tmp_Nn, zero(T))
        fill!(tmp_stage, zero(T))

        # Only fill if this process owns the mode (REAL part)
        if owns_mode
            ll = lm_idx - first(lm_range) + 1

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    tmp_linear[r] = buffers.linear_real[ll, 1, lr]
                    tmp_k1[r] = buffers.k1_real[ll, 1, lr]
                    tmp_Nn[r] = buffers.n_current_real[ll, 1, lr]
                    tmp_stage[r] = buffers.stage_nl_real[ll, 1, lr]
                end
            end
        end

        # ALL processes call Allreduce together (collective operation)
        if multi
            Allreduce!(tmp_linear, MPI.SUM, comm)
            Allreduce!(tmp_k1, MPI.SUM, comm)
            Allreduce!(tmp_Nn, MPI.SUM, comm)
            Allreduce!(tmp_stage, MPI.SUM, comm)
        end

        # ERK2 final formula (Hochbruck & Ostermann 2010, c₂ = 1/2):
        # u^{n+1} = exp(hA)·u + h·φ₁(hA)·N + (h/c₂)·φ₂(hA)·(N_stage - N)
        # With c₂ = 1/2, the correction coefficient is 1/c₂ = 2.
        delta .= tmp_stage
        @. delta = delta - tmp_Nn
        mul!(correction, phi2, delta)
        @. result = tmp_linear + dt * tmp_k1 + T(2) * dt * correction
        # Enforce boundary conditions (real part)
        if bc_spec !== nothing
            inner_val = bc_spec.inner_mode_values !== nothing && lm_idx <= length(bc_spec.inner_mode_values) ?
                        bc_spec.inner_mode_values[lm_idx] : nothing
            outer_val = bc_spec.outer_mode_values !== nothing && lm_idx <= length(bc_spec.outer_mode_values) ?
                        bc_spec.outer_mode_values[lm_idx] : nothing
            enforce_erk2_bc!(result, bc_spec.inner, 1, l, nr; value_override=inner_val)
            enforce_erk2_bc!(result, bc_spec.outer, nr, l, nr; value_override=outer_val)
        else
            result[1] = zero(T)
            result[nr] = zero(T)
        end

        # Scatter back only if this process owns the mode (REAL part)
        if owns_mode
            ll = lm_idx - first(lm_range) + 1

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    u_real[ll, 1, lr] = result[r]
                end
            end
        end

        # Reset buffers for imaginary part - all processes
        fill!(tmp_linear, zero(T))
        fill!(tmp_k1, zero(T))
        fill!(tmp_Nn, zero(T))
        fill!(tmp_stage, zero(T))

        # Only fill if this process owns the mode (IMAG part)
        if owns_mode
            ll = lm_idx - first(lm_range) + 1

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_imag, 3)
                    tmp_linear[r] = buffers.linear_imag[ll, 1, lr]
                    tmp_k1[r] = buffers.k1_imag[ll, 1, lr]
                    tmp_Nn[r] = buffers.n_current_imag[ll, 1, lr]
                    tmp_stage[r] = buffers.stage_nl_imag[ll, 1, lr]
                end
            end
        end

        # ALL processes call Allreduce together (collective operation)
        if multi
            Allreduce!(tmp_linear, MPI.SUM, comm)
            Allreduce!(tmp_k1, MPI.SUM, comm)
            Allreduce!(tmp_Nn, MPI.SUM, comm)
            Allreduce!(tmp_stage, MPI.SUM, comm)
        end

        delta .= tmp_stage
        @. delta = delta - tmp_Nn
        mul!(correction, phi2, delta)
        @. result = tmp_linear + dt * tmp_k1 + T(2) * dt * correction
        # Enforce boundary conditions
        if bc_spec !== nothing
            enforce_erk2_bc!(result, bc_spec.inner, 1, l, nr)
            enforce_erk2_bc!(result, bc_spec.outer, nr, l, nr)
        else
            result[1] = zero(T)
            result[nr] = zero(T)
        end

        # Scatter back only if this process owns the mode (IMAG part)
        if owns_mode
            ll = lm_idx - first(lm_range) + 1

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_imag, 3)
                    u_imag[ll, 1, lr] = result[r]
                end
            end
        end
    end

    synchronize_pencil_transforms!(u)
    return u
end

"""
    erk2_stage_residual_stats(buffers) -> NamedTuple

Compute diagnostic statistics for the difference between stage nonlinear terms
and the base-step nonlinear terms.
"""
function erk2_stage_residual_stats(buffers::ERK2FieldBuffers{T}) where T
    stage_real = buffers.stage_nl_real
    stage_imag = buffers.stage_nl_imag
    base_real = buffers.n_current_real
    base_imag = buffers.n_current_imag

    local_max = zero(T)
    local_sum = zero(T)

    @inbounds for idx in eachindex(stage_real)
        diff = stage_real[idx] - base_real[idx]
        mag = abs(diff)
        mag > local_max && (local_max = mag)
        local_sum += abs2(diff)
    end

    @inbounds for idx in eachindex(stage_imag)
        diff = stage_imag[idx] - base_imag[idx]
        mag = abs(diff)
        mag > local_max && (local_max = mag)
        local_sum += abs2(diff)
    end

    if MPI.Initialized() && MPI.Comm_size(get_comm()) > 1
        comm = get_comm()
        global_max = MPI.allreduce(local_max, MPI.MAX, comm)
        global_sum = MPI.allreduce(local_sum, MPI.SUM, comm)
    else
        global_max = local_max
        global_sum = local_sum
    end

    return (max=global_max, l2=sqrt(global_sum))
end

"""
    maybe_log_erk2_stage_residual!(label, buffers, step)

Emit a diagnostic log entry when ERK2 diagnostics are enabled.
"""
function maybe_log_erk2_stage_residual!(label::Symbol, buffers::ERK2FieldBuffers, step::Int)
    ERK2_DIAGNOSTICS_ENABLED[] || return nothing
    interval = ERK2_DIAGNOSTICS_INTERVAL[]
    interval <= 0 && return nothing
    (step % interval == 0) || return nothing

    stats = erk2_stage_residual_stats(buffers)
    @info "ERK2 stage residual" field=label step=step max_residual=stats.max l2_residual=stats.l2
    return stats
end

# Exports are handled by main module
