# ================================================================================
# Timestepping Module with SHTns
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
"""
function eab2_update_krylov!(u::SHTnsSpectralField{T}, nl::SHTnsSpectralField{T},
                             nl_prev::SHTnsSpectralField{T}, domain::RadialDomain,
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
    for lm_idx in lm_range
        if lm_idx <= u.nlm
            l = config.l_values[lm_idx]
            ll = lm_idx - first(lm_range) + 1
            A_banded = build_banded_A(T, domain, diffusivity, l)
            A_lu = factorize_banded(A_banded)
            # Assembled full vectors
            ur = zeros(T, nr); ui = zeros(T, nr)
            nrn = zeros(T, nr); nin = zeros(T, nr)
            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    ur[r] = u_real[ll,1,lr]; ui[r] = u_imag[ll,1,lr]
                    nrn[r] = (3/2)*n_real[ll,1,lr] - (1/2)*p_real[ll,1,lr]
                    nin[r] = (3/2)*n_imag[ll,1,lr] - (1/2)*p_imag[ll,1,lr]
                end
            end
            if multi
                MPI.Allreduce!(ur, MPI.SUM, comm)
                MPI.Allreduce!(ui, MPI.SUM, comm)
                MPI.Allreduce!(nrn, MPI.SUM, comm)
                MPI.Allreduce!(nin, MPI.SUM, comm)
            end
            # Define Aop! using banded apply
            tmp = zeros(T, nr)
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
            # Scatter back
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
"""
function get_eab2_alu_cache!(caches::Dict{Symbol,Any}, key::Symbol, ν::Float64, ::Type{T}, domain::RadialDomain) where T
    entry = get(caches, key, nothing)
    nr = domain.N
    if entry === nothing || entry[:ν] != ν || entry[:nr] != nr
        entry = Dict{Symbol,Any}(:ν => ν, :nr => nr, :map => Dict{Int, Tuple{BandedMatrix{T}, BandedLU{T}}}())
        caches[key] = entry
    end
    return entry[:map]
end

"""
    eab2_update_krylov_cached!(u, nl, nl_prev, alu_map, domain, ν, config, dt; m=20, tol=1e-8)

Same as eab2_update_krylov!, but reuses cached banded A and LU per l.
"""
function eab2_update_krylov_cached!(u::SHTnsSpectralField{T}, nl::SHTnsSpectralField{T},
                                    nl_prev::SHTnsSpectralField{T}, alu_map::Dict{Int, Tuple{BandedMatrix{T}, BandedLU{T}}},
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
    for lm_idx in lm_range
        if lm_idx <= u.nlm
            l = config.l_values[lm_idx]
            ll = lm_idx - first(lm_range) + 1
            # get or build A and LU for this l
            tup = get(alu_map, l, nothing)
            if tup === nothing
                A_banded = build_banded_A(T, domain, diffusivity, l)
                A_lu = factorize_banded(A_banded)
                tup = (A_banded, A_lu)
                alu_map[l] = tup
            end
            A_banded, A_lu = tup
            # Assembled full vectors
            ur = zeros(T, nr); ui = zeros(T, nr)
            nrn = zeros(T, nr); nin = zeros(T, nr)
            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    ur[r] = u_real[ll,1,lr]; ui[r] = u_imag[ll,1,lr]
                    nrn[r] = (3/2)*n_real[ll,1,lr] - (1/2)*p_real[ll,1,lr]
                    nin[r] = (3/2)*n_imag[ll,1,lr] - (1/2)*p_imag[ll,1,lr]
                end
            end
            if multi
                MPI.Allreduce!(ur, MPI.SUM, comm)
                MPI.Allreduce!(ui, MPI.SUM, comm)
                MPI.Allreduce!(nrn, MPI.SUM, comm)
                MPI.Allreduce!(nin, MPI.SUM, comm)
            end
            # Define Aop! using banded apply
            tmp = zeros(T, nr)
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
            # Scatter back
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
"""
function eab2_update!(u::SHTnsSpectralField{T}, nl::SHTnsSpectralField{T},
                      nl_prev::SHTnsSpectralField{T}, etd::ETDCache{T}, config::SHTnsKitConfig,
                      dt::Float64) where T
    u_real = parent(u.data_real); u_imag = parent(u.data_imag)
    n_real = parent(nl.data_real); n_imag = parent(nl.data_imag)
    p_real = parent(nl_prev.data_real); p_imag = parent(nl_prev.data_imag)
    lm_range = get_local_range(u.pencil, 1)
    r_range  = get_local_range(u.pencil, 3)
    nr_full = size(etd.E[1], 1)
    comm = get_comm()
    multi = MPI.Comm_size(comm) > 1
    linear_r_work = zeros(T, nr_full)
    linear_i_work = similar(linear_r_work)
    phi_tmp = similar(linear_r_work)
    # Build map from lm_idx to l index in etd
    for lm_idx in lm_range
        if lm_idx <= u.nlm
            l = config.l_values[lm_idx]
            lpos = findfirst(==(l), etd.l_values)
            E = etd.E[lpos]
            P1 = etd.phi1[lpos]
            ll = lm_idx - first(lm_range) + 1
            # Assemble full radial vectors
            ur = zeros(T, nr_full); ui = zeros(T, nr_full)
            nrn = zeros(T, nr_full); nin = zeros(T, nr_full)
            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    ur[r] = u_real[ll, 1, lr]
                    ui[r] = u_imag[ll, 1, lr]
                    nrn[r] = (3/2)*n_real[ll,1,lr] - (1/2)*p_real[ll,1,lr]
                    nin[r] = (3/2)*n_imag[ll,1,lr] - (1/2)*p_imag[ll,1,lr]
                end
            end
            if multi
                MPI.Allreduce!(ur, MPI.SUM, comm)
                MPI.Allreduce!(ui, MPI.SUM, comm)
                MPI.Allreduce!(nrn, MPI.SUM, comm)
                MPI.Allreduce!(nin, MPI.SUM, comm)
            end
            mul!(linear_r_work, E, ur)
            mul!(phi_tmp, P1, nrn)
            @. linear_r_work = linear_r_work + dt * phi_tmp

            mul!(linear_i_work, E, ui)
            mul!(phi_tmp, P1, nin)
            @. linear_i_work = linear_i_work + dt * phi_tmp
            # Scatter back to local slab
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

function apply_explicit_operator!(output::SHTnsSpectralField{T},
                                  input::SHTnsSpectralField{T},
                                  nonlinear::SHTnsSpectralField{T},
                                  domain::RadialDomain,
                                  diffusivity::Float64,
                                  dt::Float64;
                                  nl_prev::Union{SHTnsSpectralField{T},Nothing}=nothing,
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
"""
function build_rhs_cnab2!(rhs::SHTnsSpectralField{T}, un::SHTnsSpectralField{T},
                          nl::SHTnsSpectralField{T}, nl_prev::SHTnsSpectralField{T},
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

    @inbounds for lm_idx in lm_range
        if lm_idx <= un.nlm
            l = un.config.l_values[lm_idx]
            ll = lm_idx - first(lm_range) + 1
            ll > size(r_real, 1) && continue
            idx = add_linear ? get(matrices.lookup, l, nothing) : nothing

            if add_linear
                idx === nothing && error("Missing implicit matrix for l=$l")
                fill!(ur, zero(T)); fill!(ui, zero(T))

                for r in r_range
                    lr = r - first(r_range) + 1
                    if lr <= size(u_real, 3)
                        ur[r] = u_real[ll, 1, lr]
                        ui[r] = u_imag[ll, 1, lr]
                    end
                end

                if multi
                    MPI.Allreduce!(ur, MPI.SUM, comm)
                    MPI.Allreduce!(ui, MPI.SUM, comm)
                end

                fill!(lin_r, zero(T)); fill!(lin_i, zero(T))
                apply_banded_full!(lin_r, matrices.linear_matrices[idx], ur)
                apply_banded_full!(lin_i, matrices.linear_matrices[idx], ui)
            end

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

    return rhs
end

function solve_implicit_step!(solution::SHTnsSpectralField{T},
                              rhs::SHTnsSpectralField{T},
                              matrices::SHTnsImplicitMatrices{T}) where T
    sol_real = parent(solution.data_real)
    sol_imag = parent(solution.data_imag)
    rhs_real = parent(rhs.data_real)
    rhs_imag = parent(rhs.data_imag)

    lm_range = get_local_range(solution.pencil, 1)
    nr = size(rhs_real, 3)
    tmp_r = Vector{T}(undef, nr)
    tmp_i = Vector{T}(undef, nr)

    for lm_idx in lm_range
        if lm_idx <= solution.nlm
            l = solution.config.l_values[lm_idx]
            idx = get(matrices.lookup, l, nothing)
            idx === nothing && continue

            local_lm = lm_idx - first(lm_range) + 1
            local_lm <= size(rhs_real, 1) || continue

            @inbounds for k in 1:nr
                tmp_r[k] = rhs_real[local_lm, 1, k]
                tmp_i[k] = rhs_imag[local_lm, 1, k]
            end

            solve_banded!(tmp_r, matrices.factorizations[idx], tmp_r)
            solve_banded!(tmp_i, matrices.factorizations[idx], tmp_i)

            @inbounds for k in 1:nr
                sol_real[local_lm, 1, k] = tmp_r[k]
                sol_imag[local_lm, 1, k] = tmp_i[k]
            end
        end
    end

    return solution
end


function compute_timestep_error(new_field::SHTnsSpectralField{T}, 
                               old_field::SHTnsSpectralField{T}) where T
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
    global_error = MPI.Allreduce(error, MPI.SUM, get_comm())
    return sqrt(global_error)
end

"""
    synchronize_pencil_transforms!(field::SHTnsSpectralField{T}) where T

Ensure all pending PencilFFTs operations are completed and data is consistent across processes.
"""
function synchronize_pencil_transforms!(field::SHTnsSpectralField{T}) where T
    # Synchronize data across pencil decomposition
    MPI.Barrier(get_comm())
    return field
end

"""
    validate_mpi_consistency!(field::SHTnsSpectralField{T}) where T

Check that spectral field data is consistent across MPI processes after time stepping.
"""
function validate_mpi_consistency!(field::SHTnsSpectralField{T}) where T
    comm = get_comm()
    rank = get_rank()
    nprocs = get_nprocs()
    
    if nprocs > 1
        # Check a few sample values for consistency
        real_data = parent(field.data_real)
        imag_data = parent(field.data_imag)
        
        # Sample first few elements
        n_samples = min(5, length(real_data))
        local_samples_real = Vector{T}(undef, n_samples)
        local_samples_imag = Vector{T}(undef, n_samples)
        
        @inbounds for i in 1:n_samples
            local_samples_real[i] = real_data[i]
            local_samples_imag[i] = imag_data[i]
        end
        
        # Gather samples from all processes
        all_samples_real = MPI.Allgather(local_samples_real, comm)
        all_samples_imag = MPI.Allgather(local_samples_imag, comm)
        
        # Check consistency on rank 0
        if rank == 0
            max_diff_real = zero(T)
            max_diff_imag = zero(T)
            
            for proc in 2:nprocs
                for i in 1:n_samples
                    diff_real = abs(all_samples_real[(proc-1)*n_samples + i] - all_samples_real[i])
                    diff_imag = abs(all_samples_imag[(proc-1)*n_samples + i] - all_samples_imag[i])
                    max_diff_real = max(max_diff_real, diff_real)
                    max_diff_imag = max(max_diff_imag, diff_imag)
                end
            end
            
            # Warn if inconsistency detected
            if max_diff_real > 1e-12 || max_diff_imag > 1e-12
                @warn "MPI data inconsistency detected: max_diff_real=$max_diff_real, max_diff_imag=$max_diff_imag"
            end
        end
    end
    
    return field
end


# ================================================================================
# Exponential 2nd Order Runge-Kutta (ERK2) Implementation
# ================================================================================

"""
    ERK2Cache{T}

Cached data structure for Exponential 2nd Order Runge-Kutta method.
Stores precomputed matrix exponentials and φ functions for each spherical harmonic mode.
"""
struct ERK2Cache{T}
    dt::Float64
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
    create_erk2_cache(config, domain, diffusivity, dt; use_krylov=false, m=20, tol=1e-8)

Create ERK2 cache with precomputed matrix functions for all spherical harmonic modes.
"""
function create_erk2_cache(::Type{T}, config::SHTnsKitConfig, domain::RadialDomain,
                          diffusivity::Float64, dt::Float64;
                          use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T
    
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
            phi2_full_l = compute_phi2_function(Adt_full, E_full_l)
            push!(phi2_full, Matrix{T}(phi2_full_l))
            
        end
    end
    
    # Ensure MPI consistency
    MPI.Barrier(get_comm())
    
    return ERK2Cache{T}(dt, lvals, E_half, E_full, phi1_half, phi1_full, phi2_full,
                       use_krylov, m, tol, true)
end

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
    compute_phi2_function(A, expA)

Compute φ2(A) = (exp(A) - I - A) / A² efficiently with comprehensive error handling.
Uses series expansion for small ||A|| to avoid numerical issues.
"""
function compute_phi2_function(A::Matrix{T}, expA::Matrix{T}) where T
    nr = size(A, 1)
    I_mat = Matrix{T}(I, nr, nr)

    # Check for NaN or Inf in inputs
    if !all(isfinite.(A)) || !all(isfinite.(expA))
        @warn "Non-finite values detected in φ2 computation, using zero approximation"
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

        # Check condition number
        rcond_val = rcond(lu_A)
        if rcond_val < sqrt(eps(T))
            @warn "Ill-conditioned matrix in φ2 computation (rcond = $rcond_val), using series expansion"
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
        @warn "LU factorization failed in φ2 computation: $e, using series expansion"
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
"""
function get_erk2_cache!(caches::Dict{Symbol,Any}, key::Symbol, diffusivity::Float64,
                        ::Type{T}, config::SHTnsKitConfig, domain::RadialDomain, dt::Float64;
                        use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T
    
    entry = get(caches, key, nothing)
    nr = domain.N
    
    # Check if cache needs to be rebuilt
    if entry === nothing || 
       get(entry, :diffusivity, nothing) != diffusivity ||
       get(entry, :nr, nothing) != nr ||
       get(entry, :dt, nothing) != dt ||
       !get(entry, :mpi_consistent, false)
        
        if get_rank() == 0
            @info "Creating new ERK2 cache for $key (ν=$diffusivity, nr=$nr, dt=$dt)"
        end
        
        cache = create_erk2_cache(T, config, domain, diffusivity, dt; 
                                use_krylov, m, tol)
        
        entry = Dict{Symbol,Any}(
            :cache => cache,
            :diffusivity => diffusivity,
            :nr => nr,
            :dt => dt,
            :mpi_consistent => true
        )
        caches[key] = entry
    end
    
    return entry[:cache]
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
                                    bundle::Dict{Symbol,<:ERK2Cache})
    for (key, cache) in bundle
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

function ERK2FieldBuffers(u::SHTnsSpectralField{T}, nl::SHTnsSpectralField{T}, cache::ERK2Cache{T}) where T
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

function erk2_prepare_field!(buffers::ERK2FieldBuffers{T}, u::SHTnsSpectralField{T},
                             nl::SHTnsSpectralField{T}, cache::ERK2Cache{T},
                             config::SHTnsKitConfig, dt::Float64) where T
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

    linear_real = buffers.linear_real
    linear_imag = buffers.linear_imag
    k1_real = buffers.k1_real
    k1_imag = buffers.k1_imag
    stage_real = buffers.stage_real
    stage_imag = buffers.stage_imag

    for lm_idx in lm_range
        if lm_idx <= u.nlm
            l = config.l_values[lm_idx]
            cache_idx = get(buffers.cache_lookup, l, nothing)
            cache_idx === nothing && error("Missing ERK2 cache entry for l=$l")

            E_full = cache.E_full[cache_idx]
            E_half = cache.E_half[cache_idx]
            phi1_full = cache.phi1_full[cache_idx]
            phi1_half = cache.phi1_half[cache_idx]

            fill!(ur, zero(T)); fill!(ui, zero(T))
            fill!(nr_vec, zero(T)); fill!(ni_vec, zero(T))

            ll = lm_idx - first(lm_range) + 1

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    ur[lr] = u_real[ll, 1, lr]
                    ui[lr] = u_imag[ll, 1, lr]
                    nr_vec[lr] = buffers.n_current_real[ll, 1, lr]
                    ni_vec[lr] = buffers.n_current_imag[ll, 1, lr]
                end
            end

            if multi
                MPI.Allreduce!(ur, MPI.SUM, comm)
                MPI.Allreduce!(ui, MPI.SUM, comm)
                MPI.Allreduce!(nr_vec, MPI.SUM, comm)
                MPI.Allreduce!(ni_vec, MPI.SUM, comm)
            end

            mul!(linear_tmp, E_full, ur)
            mul!(k1_tmp, phi1_full, nr_vec)

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    linear_real[ll, 1, lr] = linear_tmp[lr]
                    k1_real[ll, 1, lr] = k1_tmp[lr]
                end
            end

            mul!(stage_tmp, E_half, ur)
            mul!(stage_phi_tmp, phi1_half, nr_vec)
            @. stage_tmp = stage_tmp + half_dt * stage_phi_tmp

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    stage_real[ll, 1, lr] = stage_tmp[lr]
                end
            end

            mul!(linear_tmp, E_full, ui)
            mul!(k1_tmp, phi1_full, ni_vec)

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_imag, 3)
                    linear_imag[ll, 1, lr] = linear_tmp[lr]
                    k1_imag[ll, 1, lr] = k1_tmp[lr]
                end
            end

            mul!(stage_tmp, E_half, ui)
            mul!(stage_phi_tmp, phi1_half, ni_vec)
            @. stage_tmp = stage_tmp + half_dt * stage_phi_tmp

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_imag, 3)
                    stage_imag[ll, 1, lr] = stage_tmp[lr]
                end
            end
        end
    end

    return buffers
end

erk2_apply_stage!(buffers::ERK2FieldBuffers{T}, u::SHTnsSpectralField{T}) where T = begin
    parent(u.data_real) .= buffers.stage_real
    parent(u.data_imag) .= buffers.stage_imag
    return u
end

erk2_store_stage_nonlinear!(buffers::ERK2FieldBuffers{T}, nl::SHTnsSpectralField{T}) where T = begin
    parent_nl_real = parent(nl.data_real)
    parent_nl_imag = parent(nl.data_imag)
    copyto!(buffers.stage_nl_real, parent_nl_real)
    copyto!(buffers.stage_nl_imag, parent_nl_imag)
    return buffers
end

function erk2_finalize_field!(buffers::ERK2FieldBuffers{T}, u::SHTnsSpectralField{T},
                              cache::ERK2Cache{T}, config::SHTnsKitConfig, dt::Float64) where T
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

    for lm_idx in lm_range
        if lm_idx <= u.nlm
            l = config.l_values[lm_idx]
            cache_idx = get(buffers.cache_lookup, l, nothing)
            cache_idx === nothing && error("Missing ERK2 cache entry for l=$l")
            phi2 = cache.phi2_full[cache_idx]
            ll = lm_idx - first(lm_range) + 1

            fill!(tmp_linear, zero(T))
            fill!(tmp_k1, zero(T))
            fill!(tmp_Nn, zero(T))
            fill!(tmp_stage, zero(T))

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    tmp_linear[lr] = buffers.linear_real[ll, 1, lr]
                    tmp_k1[lr] = buffers.k1_real[ll, 1, lr]
                    tmp_Nn[lr] = buffers.n_current_real[ll, 1, lr]
                    tmp_stage[lr] = buffers.stage_nl_real[ll, 1, lr]
                end
            end

            if multi
                MPI.Allreduce!(tmp_linear, MPI.SUM, comm)
                MPI.Allreduce!(tmp_k1, MPI.SUM, comm)
                MPI.Allreduce!(tmp_Nn, MPI.SUM, comm)
                MPI.Allreduce!(tmp_stage, MPI.SUM, comm)
            end

            delta .= tmp_stage
            @. delta = delta - tmp_Nn
            mul!(correction, phi2, delta)
            @. result = tmp_linear + dt * (tmp_k1 + correction)

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    u_real[ll, 1, lr] = result[lr]
                end
            end

            fill!(tmp_linear, zero(T))
            fill!(tmp_k1, zero(T))
            fill!(tmp_Nn, zero(T))
            fill!(tmp_stage, zero(T))

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_imag, 3)
                    tmp_linear[lr] = buffers.linear_imag[ll, 1, lr]
                    tmp_k1[lr] = buffers.k1_imag[ll, 1, lr]
                    tmp_Nn[lr] = buffers.n_current_imag[ll, 1, lr]
                    tmp_stage[lr] = buffers.stage_nl_imag[ll, 1, lr]
                end
            end

            if multi
                MPI.Allreduce!(tmp_linear, MPI.SUM, comm)
                MPI.Allreduce!(tmp_k1, MPI.SUM, comm)
                MPI.Allreduce!(tmp_Nn, MPI.SUM, comm)
                MPI.Allreduce!(tmp_stage, MPI.SUM, comm)
            end

            delta .= tmp_stage
            @. delta = delta - tmp_Nn
            mul!(correction, phi2, delta)
            @. result = tmp_linear + dt * (tmp_k1 + correction)

            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_imag, 3)
                    u_imag[ll, 1, lr] = result[lr]
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
