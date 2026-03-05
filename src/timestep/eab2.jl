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

    # Small dt: series expansion avoids catastrophic cancellation in (exp(dt A)v - v)
    # and skips the expensive Krylov computation entirely
    if dt < 1e-8
        # φ1(dt*A) * v ≈ v + (dt/2)*A*v for small dt
        Av = similar(v)
        Aop!(Av, v)
        return v .+ (dt/2) .* Av
    end

    # Compute exp(dt*A) * v
    ev = exp_action_krylov(Aop!, v, dt; m, tol)
    c = ev .- v

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

    # Pre-allocate work arrays outside the loop to avoid per-mode GC pressure
    ur = zeros(T, nr); ui = zeros(T, nr)
    nrn = zeros(T, nr); nin = zeros(T, nr)

    # Use GLOBAL loop bounds to ensure all processes call Allreduce same number of times
    for lm_idx in 1:nlm_total
        # Check if this process owns this lm mode
        owns_mode = lm_idx in lm_range

        l = config.l_values[lm_idx]
        A_banded = build_banded_A(T, domain, diffusivity, l)
        A_lu = factorize_banded(A_banded)

        # Reset work arrays for this iteration
        fill!(ur, zero(T)); fill!(ui, zero(T))
        fill!(nrn, zero(T)); fill!(nin, zero(T))

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
    eab2_update_krylov_cached!(u, nl, nl_prev, alu_map, domain, ν, config, dt; m=20, tol=1e-8, mass_coeff=1.0)

Same as eab2_update_krylov!, but reuses cached banded A and LU per l.

# Mass Coefficient
For equations of the form `c * du/dt = ν*L*u + NL` (e.g. velocity with c=d_E),
pass `mass_coeff=c`. The operator becomes `A = (ν/c)*L` and NL is scaled by `1/c`.

# MPI Safety
Uses global loop bounds (1:nlm) to ensure all processes call Allreduce
the same number of times, preventing deadlock with uneven lm distribution.
"""
function eab2_update_krylov_cached!(u::SHTnsSpecField{T}, nl::SHTnsSpecField{T},
                                    nl_prev::SHTnsSpecField{T}, alu_map::Dict{Int, Tuple{BandedMatrix{T}, BandedLU{T}}},
                                    domain::RadialDomain, diffusivity::Float64, config::SHTnsKitConfig,
                                    dt::Float64; m::Int=20, tol::Float64=1e-8, mass_coeff::Float64=1.0) where T
    u_real = parent(u.data_real); u_imag = parent(u.data_imag)
    n_real = parent(nl.data_real); n_imag = parent(nl.data_imag)
    p_real = parent(nl_prev.data_real); p_imag = parent(nl_prev.data_imag)
    lm_range = get_local_range(u.pencil, 1)
    r_range  = get_local_range(u.pencil, 3)
    nr = domain.N
    comm = get_comm()
    multi = MPI.Comm_size(comm) > 1
    nlm_total = u.nlm

    # Pre-allocate work arrays outside the loop to avoid per-mode GC pressure
    ur = zeros(T, nr); ui = zeros(T, nr)
    nrn = zeros(T, nr); nin = zeros(T, nr)

    # Use GLOBAL loop bounds to ensure all processes call Allreduce same number of times
    for lm_idx in 1:nlm_total
        # Check if this process owns this lm mode
        owns_mode = lm_idx in lm_range

        l = config.l_values[lm_idx]
        # get or build A and LU for this l
        # For equations c*du/dt = ν*L*u + NL, the effective operator is A = (ν/c)*L
        tup = get(alu_map, l, nothing)
        if tup === nothing
            A_banded = build_banded_A(T, domain, diffusivity / mass_coeff, l)
            A_lu = factorize_banded(A_banded)
            tup = (A_banded, A_lu)
            alu_map[l] = tup
        end
        A_banded, A_lu = tup

        # Reset work arrays for this iteration
        fill!(ur, zero(T)); fill!(ui, zero(T))
        fill!(nrn, zero(T)); fill!(nin, zero(T))

        # Scale factor for nonlinear terms: 1/mass_coeff
        inv_mc = T(1.0 / mass_coeff)

        # Only fill if this process owns the mode
        if owns_mode
            ll = lm_idx - first(lm_range) + 1
            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    ur[r] = u_real[ll,1,lr]; ui[r] = u_imag[ll,1,lr]
                    nrn[r] = inv_mc * ((3/2)*n_real[ll,1,lr] - (1/2)*p_real[ll,1,lr])
                    nin[r] = inv_mc * ((3/2)*n_imag[ll,1,lr] - (1/2)*p_imag[ll,1,lr])
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

    # Pre-allocate work arrays outside the loop to avoid per-mode GC pressure
    ur = zeros(T, nr_full); ui = zeros(T, nr_full)
    nrn = zeros(T, nr_full); nin = zeros(T, nr_full)

    # Use GLOBAL loop bounds to ensure all processes call Allreduce same number of times
    for lm_idx in 1:nlm_total
        # Check if this process owns this lm mode
        owns_mode = lm_idx in lm_range

        l = config.l_values[lm_idx]
        lpos = findfirst(==(l), etd.l_values)
        if lpos === nothing
            error("ETD cache missing l=$l. The cache may be stale after a resolution change; " *
                  "delete the cache file and restart.")
        end
        E = etd.E[lpos]
        P1 = etd.phi1[lpos]

        # Reset work arrays for this iteration
        fill!(ur, zero(T)); fill!(ui, zero(T))
        fill!(nrn, zero(T)); fill!(nin, zero(T))

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
