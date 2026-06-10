const LA = SOLVER_LINEAR_ALGEBRA
const SolverMPI = SOLVER_MPI

const SOLVER_ERK2_DIAGNOSTICS_ENABLED = Ref(false)
const SOLVER_ERK2_DIAGNOSTICS_INTERVAL = Ref(100)
const SOLVER_ENABLE_TIMING = Ref(false)

struct SolverModeIndexCacheKey
    lmax::Int
    mmax::Int
    nlm::Int
    l_values_hash::UInt
    m_values_hash::UInt
end

@inline function mode_index_cache_key(config::SHTnsConfigType)
    return SolverModeIndexCacheKey(
        config.lmax,
        config.mmax,
        config.nlm,
        hash(config.l_values),
        hash(config.m_values)
    )
end

const SOLVER_MODE_INDEX_CACHE = Dict{SolverModeIndexCacheKey, Dict{Tuple{Int, Int}, Int}}()
const SOLVER_MODE_INDEX_CACHE_LOCK = ReentrantLock()

@inline local_range(pencil, dim::Int) = pencil.axes_local[dim]
@inline function mpi_comm()
    backend_ensure_mpi!()
    return SOLVER_BACKEND_MPI.COMM_WORLD
end
@inline mpi_rank(comm = mpi_comm()) = SolverMPI.Comm_rank(comm)
@inline mpi_initialized() = SolverMPI.Initialized()
@inline mpi_comm_size(comm = mpi_comm()) = SolverMPI.Comm_size(comm)
@inline mpi_wtime() = SolverMPI.Wtime()
@inline timing_enabled() = SOLVER_ENABLE_TIMING[]
@inline solver_buffer_cache_lock() = SOLVER_SHARED_BUFFER_CACHE_LOCK

@inline function set_timing_enabled!(enabled::Bool)
    SOLVER_ENABLE_TIMING[] = enabled
    return nothing
end

@inline function set_erk2_diagnostics!(enabled::Bool, interval::Integer)
    SOLVER_ERK2_DIAGNOSTICS_ENABLED[] = enabled
    SOLVER_ERK2_DIAGNOSTICS_INTERVAL[] = Int(interval)
    return nothing
end

function build_mode_index_table(config)
    table = Dict{Tuple{Int, Int}, Int}()
    sizehint!(table, config.nlm)
    @inbounds for idx in 1:config.nlm
        table[(config.l_values[idx], config.m_values[idx])] = idx
    end
    return table
end

function mode_index(config, l::Int, m::Int)
    key = mode_index_cache_key(config)
    table = get(SOLVER_MODE_INDEX_CACHE, key, nothing)
    if table === nothing
        lock(SOLVER_MODE_INDEX_CACHE_LOCK) do
            table = get(SOLVER_MODE_INDEX_CACHE, key, nothing)
            if table === nothing
                table = build_mode_index_table(config)
                SOLVER_MODE_INDEX_CACHE[key] = table
            end
        end
        table = SOLVER_MODE_INDEX_CACHE[key]
    end
    return get(table, (l, m), 0)
end

# These macros centralize the pencil-ownership guard used in many spectral
# loops: a rank iterates candidate global mode ids, maps owned modes to its
# local storage slot, and skips modes outside the rank's rectangular pencil.
macro solver_local_spectral_modes(
        lm_var, slot_var, lm_range, config, limit_data, storage_data, body)
    lm = esc(lm_var)
    slot = esc(slot_var)
    return quote
        for $lm in $(esc(lm_range))
            if $lm > length($(esc(limit_data)))
                continue
            end
            $slot = local_spectral_storage_slot($(esc(config)), $lm)
            $slot === nothing && continue
            if $slot[1] > size($(esc(storage_data)), 1) ||
               $slot[2] > size($(esc(storage_data)), 2)
                continue
            end
            $(esc(body))
        end
    end
end

macro solver_threaded_local_spectral_modes(
        lm_var, slot_var, lm_range, config, limit_data, storage_data, body)
    lm = esc(lm_var)
    slot = esc(slot_var)
    return quote
        Threads.@threads for $lm in $(esc(lm_range))
            if $lm > length($(esc(limit_data)))
                continue
            end
            $slot = local_spectral_storage_slot($(esc(config)), $lm)
            $slot === nothing && continue
            if $slot[1] > size($(esc(storage_data)), 1) ||
               $slot[2] > size($(esc(storage_data)), 2)
                continue
            end
            $(esc(body))
        end
    end
end

"""
    get_bc_vectors(field)

Return mode-indexed scalar boundary vectors for timestep solves.

When a spectral BC file has been loaded, the interpolation cache supplies real
and imaginary values for each mode. Otherwise the solver falls back to the
field's parameter-derived `boundary_values`, which contains real values only.
"""
function get_bc_vectors(field)
    cache = field.boundary_interpolation_cache
    if cache isa bcs.BoundaryInterpolationCache
        bc_real = cache.bc_real
        bc_imag = cache.bc_imag
        if cache.bc_loaded && bc_real !== nothing && bc_imag !== nothing
            return (
                inner_real = view(bc_real, 1, :),
                outer_real = view(bc_real, 2, :),
                inner_imag = view(bc_imag, 1, :),
                outer_imag = view(bc_imag, 2, :)
            )
        end
    end

    if hasfield(typeof(field), :boundary_values)
        return (
            inner_real = view(field.boundary_values, 1, :),
            outer_real = view(field.boundary_values, 2, :),
            inner_imag = nothing,
            outer_imag = nothing
        )
    end

    return (
        inner_real = nothing,
        outer_real = nothing,
        inner_imag = nothing,
        outer_imag = nothing
    )
end

@inline function mpi_barrier!(comm = mpi_comm())
    MPI.Barrier(comm)
    return nothing
end

@inline function allreduce_sum_in_place!(buffer, comm = mpi_comm())
    MPI.Allreduce!(buffer, MPI.SUM, comm)
    return buffer
end

@inline function allreduce_sum_buffers!(sendbuf, recvbuf, comm = mpi_comm())
    sendbuf === recvbuf || copyto!(recvbuf, sendbuf)
    MPI.Allreduce!(recvbuf, MPI.SUM, comm)
    return recvbuf
end

@inline function allreduce_sum(value, comm = mpi_comm())
    return MPI.Allreduce(value, +, comm)
end

@inline function allreduce_max(value, comm = mpi_comm())
    return MPI.Allreduce(value, MPI.MAX, comm)
end

@inline function domain_bandwidth(domain::RadialDomainType)
    if !isempty(domain.dr_matrices)
        return (size(domain.dr_matrices[1], 1) - 1) ÷ 2
    elseif !isempty(domain.radial_laplacian)
        return (size(domain.radial_laplacian, 1) - 1) ÷ 2
    end
    error("Cannot infer solver radial stencil bandwidth from domain")
end

function build_radial_derivative_matrix(
        ::Type{T},
        order::Int,
        domain::RadialDomainType
) where {T}
    N = domain.N
    bandwidth = domain_bandwidth(domain)
    calc_T = promote_type(T, eltype(domain.r))
    data = zeros(T, 2 * bandwidth + 1, N)

    for n in 1:N
        left = max(1, n - bandwidth)
        right = min(N, n + bandwidth)
        stencil_size = right - left + 1

        V = ones(calc_T, stencil_size, stencil_size)
        points = calc_T.(domain.r[left:right, 4])
        center = calc_T(domain.r[n, 4])

        for j in 2:stencil_size
            for i in 1:stencil_size
                V[i, j] = V[i, j - 1] * (points[i] - center)
            end
        end

        rhs = zeros(calc_T, stencil_size)
        if order + 1 > stencil_size
            error(
                "Insufficient stencil size ($stencil_size) for solver derivative order $order at grid point $n. " *
                "Need at least $(order + 1) points.",
            )
        end
        rhs[order + 1] = calc_T(factorial(order))

        coeffs = try
            transpose(V) \ rhs
        catch e
            error(
                "Failed to solve solver Vandermonde system at grid point $n. " *
                "Original error: $e",
            )
        end

        for (i, idx) in enumerate(left:right)
            band_row = bandwidth + 1 + n - idx
            if 1 <= band_row <= 2 * bandwidth + 1
                data[band_row, idx] = T(coeffs[i])
            end
        end
    end

    return BandedOperator{T}(data, bandwidth, N)
end

@inline build_radial_derivative_matrix(order::Int,
    domain::RadialDomainType) = build_radial_derivative_matrix(eltype(domain.r), order, domain)

function extract_dense_row(data::AbstractMatrix{T}, bandwidth::Int, nr::Int, row::Int) where {T}
    result = zeros(T, nr)
    @inbounds for j in max(1, row - bandwidth):min(nr, row + bandwidth)
        band_idx = bandwidth + 1 + row - j
        if 1 <= band_idx <= 2 * bandwidth + 1
            result[j] = data[band_idx, j]
        end
    end
    return result
end

function build_radial_laplacian(::Type{T}, domain::RadialDomainType) where {T}
    d2_matrix = build_radial_derivative_matrix(T, 2, domain)
    d1_matrix = build_radial_derivative_matrix(T, 1, domain)
    bandwidth = d2_matrix.bandwidth
    laplacian_data = copy(d2_matrix.data)

    for n in 1:domain.N
        r_inv = T(domain.r[n, 3])
        for j in max(1, n - bandwidth):min(domain.N, n + bandwidth)
            band_row = bandwidth + 1 + n - j
            laplacian_data[band_row, j] += T(2) * r_inv * d1_matrix.data[band_row, j]
        end
    end

    return BandedOperator{T}(laplacian_data, bandwidth, domain.N)
end

@inline build_radial_laplacian(domain::RadialDomainType) = build_radial_laplacian(eltype(domain.r), domain)

function krylov_exp_action(
        Aop!,
        v::Vector{T},
        dt::Float64;
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    n = length(v)

    if n == 0 || !all(isfinite, v)
        return zeros(T, n)
    end

    V = Matrix{T}(undef, n, m)
    H = zeros(T, m, m)
    beta = LA.norm(v)
    if beta == zero(T)
        return zeros(T, n)
    end

    if abs(dt) < eps(T) * 10
        return copy(v)
    end

    V[:, 1] = v / beta
    w = similar(v)
    kmax = m

    for j in 1:m
        Aop!(w, view(V, :, j))

        if !all(isfinite, w)
            @warn "Non-finite values from solver operator in Krylov iteration $j"
            kmax = max(1, j - 1)
            break
        end

        for i in 1:j
            H[i, j] = LA.dot(view(V, :, i), w)
            # View, not a copy, of column i: matches the workspace `!` variant and
            # avoids an O(n) allocation per inner Arnoldi iteration on this
            # (no-workspace fallback / `exp_action_krylov` utility) path.
            @views @. w = w - H[i, j] * V[:, i]
        end

        if j < m
            H[j + 1, j] = LA.norm(w)
            if H[j + 1, j] < series_tol(T)
                kmax = j
                break
            end
            V[:, j + 1] = w / H[j + 1, j]

            try
                Hred_j = dt .* @view H[1:j, 1:j]
                if j > 1 && LA.cond(Hred_j) > 1e12
                    @warn "Ill-conditioned solver Hessenberg matrix, stopping Krylov at iteration $j"
                    kmax = j
                    break
                end

                e1 = zeros(T, j)
                e1[1] = one(T)
                y_small_j = exp(Hred_j) * (beta .* e1)

                if !all(isfinite, y_small_j)
                    @warn "Non-finite solver exponential result, stopping Krylov at iteration $j"
                    kmax = j
                    break
                end

                res_est = abs(H[j + 1, j]) * abs(j > 0 ? y_small_j[end] : beta)
                if res_est <= tol * LA.norm(y_small_j)
                    kmax = j
                    break
                end
            catch e
                @warn "Error in solver Krylov convergence check: $e, stopping at iteration $j"
                kmax = j
                break
            end
        end
    end

    try
        Hred = dt .* H[1:kmax, 1:kmax]
        e1 = zeros(T, kmax)
        e1[1] = one(T)
        y_small = exp(Hred) * (beta .* e1)

        if !all(isfinite, y_small)
            error(
                "Non-finite result in final solver Krylov computation. " *
                "Consider reducing dt or increasing Krylov subspace dimension m.",
            )
        end

        result = V[:, 1:kmax] * y_small
        if !all(isfinite, result)
            error(
                "Non-finite final result in solver Krylov exponential action. " *
                "Consider reducing dt or increasing Krylov subspace dimension m.",
            )
        end

        return result
    catch e
        e isa ErrorException && rethrow(e)
        error("Error in final solver Krylov computation: $e")
    end
end

function krylov_exp_action!(
        dest::Vector{T},
        Aop!,
        v::Vector{T},
        dt::Float64,
        work::SolverKrylovWork{T};
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    n = length(v)
    ensure_krylov_work!(work, n, m)

    if n == 0 || !all(isfinite, v)
        fill!(dest, zero(T))
        return dest
    end

    beta = LA.norm(v)
    if beta == zero(T)
        fill!(dest, zero(T))
        return dest
    end

    if abs(dt) < eps(T) * 10
        copyto!(dest, v)
        return dest
    end

    V = work.V
    H = work.H
    Hred = work.Hred
    w = work.w
    rhs = work.rhs

    fill!(H, zero(T))
    inv_beta = inv(beta)
    @inbounds @simd for i in 1:n
        V[i, 1] = v[i] * inv_beta
    end

    kmax = m
    @inbounds for j in 1:m
        Aop!(w, @view V[:, j])

        if !all(isfinite, w)
            @warn "Non-finite values from solver operator in Krylov iteration $j"
            kmax = max(1, j - 1)
            break
        end

        for i in 1:j
            hij = zero(T)
            @simd for r in 1:n
                hij += V[r, i] * w[r]
            end
            H[i, j] = hij
            @simd for r in 1:n
                w[r] -= hij * V[r, i]
            end
        end

        if j < m
            hnext = LA.norm(w)
            H[j + 1, j] = hnext
            if hnext < series_tol(T)
                kmax = j
                break
            end
            inv_hnext = inv(hnext)
            @simd for r in 1:n
                V[r, j + 1] = w[r] * inv_hnext
            end
        end
    end

    @inbounds for j in 1:kmax
        rhs[j] = j == 1 ? beta : zero(T)
        for i in 1:kmax
            Hred[i, j] = dt * H[i, j]
        end
    end

    y_small = exp(@view Hred[1:kmax, 1:kmax]) * @view rhs[1:kmax]
    if !all(isfinite, y_small)
        error(
            "Non-finite result in final solver Krylov computation. " *
            "Consider reducing dt or increasing Krylov subspace dimension m.",
        )
    end

    fill!(dest, zero(T))
    @inbounds for j in 1:kmax
        yj = y_small[j]
        @simd for r in 1:n
            dest[r] += V[r, j] * yj
        end
    end

    if !all(isfinite, dest)
        error(
            "Non-finite final result in solver Krylov exponential action. " *
            "Consider reducing dt or increasing Krylov subspace dimension m.",
        )
    end

    return dest
end

@inline function get_sht_plan(buffers::SHTnsBuffers)
    return buffers.sht_plan::Union{SHTnsKit.SHTPlan, Nothing}
end

@inline function transform_arch(config)
    device = config._buffers.transform_device
    device isa AbstractArchitecture && return device
    return SHTnsKit.is_gpu_config(config.sht_config) ? GPU(device) : CPU()
end

@inline uses_gpu(config) = !(transform_arch(config) isa CPU)

function fill_vector_coeff_buffer!(coeffs_buffer, spec_real, spec_imag, r_local, config)
    if uses_gpu(config) && solver_gpu_device() !== :cuda
        return solver_gpu_fill_vector_coeff_buffer(
            coeffs_buffer,
            spec_real,
            spec_imag,
            r_local,
            config
        )
    end
    return cpu_fill_vector_coeff_buffer!(
        coeffs_buffer,
        spec_real,
        spec_imag,
        r_local,
        config
    )
end

@inline function cpu_fill_vector_coeff_buffer!(
        coeffs_buffer, spec_real, spec_imag, r_local, config)
    return cpu_fill_scalar_coeff_buffer!(
        coeffs_buffer, spec_real, spec_imag, r_local, config)
end

function store_vector_coefficients!(spec_real, spec_imag, coeffs_matrix, r_local, config)
    if uses_gpu(config) && solver_gpu_device() !== :cuda
        return solver_gpu_store_vector_coefficients(
            spec_real,
            spec_imag,
            coeffs_matrix,
            r_local,
            config
        )
    end
    return cpu_store_vector_coefficients!(
        spec_real,
        spec_imag,
        coeffs_matrix,
        r_local,
        config
    )
end

function cpu_store_vector_coefficients!(
        spec_real, spec_imag, coeffs_matrix, r_local, config)
    matrix_lmax = size(coeffs_matrix, 1) - 1
    matrix_mmax = size(coeffs_matrix, 2) - 1

    local_modes = local_spectral_mode_indices(config)

    for lm_idx in local_modes
        slot = local_spectral_storage_slot(config, lm_idx)
        slot === nothing && continue
        l = config.l_values[lm_idx]
        m = config.m_values[lm_idx]
        if r_local <= size(spec_real, 3) && r_local <= size(spec_imag, 3)
            if l <= matrix_lmax && m <= matrix_mmax
                coeff = coeffs_matrix[l + 1, m + 1]
                set_local_spectral_value!(spec_real, slot, r_local, real(coeff))
                set_local_spectral_value!(spec_imag, slot, r_local, m == 0 ? 0.0 :
                                                                    imag(coeff))
            else
                set_local_spectral_value!(spec_real, slot, r_local, 0.0)
                set_local_spectral_value!(spec_imag, slot, r_local, 0.0)
            end
        end
    end

    return spec_real, spec_imag
end

function extract_vector_component!(
        component_buffer::Matrix{T},
        v_data,
        r_local,
        config;
        axes_local::Union{Nothing, Tuple} = nothing
) where {T}
    if uses_gpu(config) && solver_gpu_device() !== :cuda
        return solver_gpu_extract_vector_component(
            component_buffer,
            v_data,
            r_local,
            config;
            axes_local = axes_local
        )
    end
    return cpu_extract_vector_component!(
        component_buffer,
        v_data,
        r_local,
        config;
        axes_local = axes_local
    )
end

function cpu_extract_vector_component!(
        component_buffer::Matrix{T},
        v_data,
        r_local,
        config;
        axes_local::Union{Nothing, Tuple} = nothing
) where {T}
    nlat, nlon = config.nlat, config.nlon
    fill!(component_buffer, zero(T))
    has_local_data = r_local <= size(v_data, 3)

    if axes_local !== nothing
        θ_range = axes_local[1]
        φ_range = axes_local[2]
        if has_local_data
            for i_local in 1:size(v_data, 1)
                i_global = θ_range[i_local]
                for j_local in 1:size(v_data, 2)
                    j_global = φ_range[j_local]
                    component_buffer[i_global, j_global] = v_data[i_local, j_local, r_local]
                end
            end
        end
    else
        common_i_range = 1:min(size(v_data, 1), nlat, size(component_buffer, 1))
        common_j_range = 1:min(size(v_data, 2), nlon, size(component_buffer, 2))
        if has_local_data
            for i in common_i_range
                for j in common_j_range
                    component_buffer[i, j] = v_data[i, j, r_local]
                end
            end
        end
    end

    allreduce_sum_in_place!(component_buffer)
    return component_buffer
end

function store_vector_components!(
        v_theta,
        v_phi,
        vt_field,
        vp_field,
        r_local,
        config;
        axes_local::Union{Nothing, Tuple} = nothing
)
    if uses_gpu(config) && solver_gpu_device() !== :cuda
        return solver_gpu_store_vector_components(
            v_theta,
            v_phi,
            vt_field,
            vp_field,
            r_local,
            config;
            axes_local = axes_local
        )
    end
    return cpu_store_vector_components!(
        v_theta,
        v_phi,
        vt_field,
        vp_field,
        r_local,
        config;
        axes_local = axes_local
    )
end

function cpu_store_vector_components!(
        v_theta,
        v_phi,
        vt_field,
        vp_field,
        r_local,
        config;
        axes_local::Union{Nothing, Tuple} = nothing
)
    if r_local > size(v_theta, 3) || r_local > size(v_phi, 3)
        return v_theta, v_phi
    end

    if axes_local !== nothing
        θ_range = axes_local[1]
        φ_range = axes_local[2]
        for i_local in 1:size(v_theta, 1)
            i_global = θ_range[i_local]
            for j_local in 1:size(v_theta, 2)
                j_global = φ_range[j_local]
                v_theta[i_local, j_local, r_local] = vt_field[i_global, j_global]
                v_phi[i_local, j_local, r_local] = vp_field[i_global, j_global]
            end
        end
    else
        common_i_range = 1:min(size(v_theta, 1), size(v_phi, 1), size(vt_field, 1), size(vp_field, 1))
        common_j_range = 1:min(size(v_theta, 2), size(v_phi, 2), size(vt_field, 2), size(vp_field, 2))
        for i in common_i_range
            for j in common_j_range
                v_theta[i, j, r_local] = vt_field[i, j]
                v_phi[i, j, r_local] = vp_field[i, j]
            end
        end
    end

    return v_theta, v_phi
end

function store_scalar_component!(
        v_component,
        field,
        r_local,
        config;
        axes_local::Union{Nothing, Tuple} = nothing
)
    if r_local > size(v_component, 3)
        return v_component
    end

    if axes_local !== nothing
        θ_range = axes_local[1]
        φ_range = axes_local[2]
        for i_local in 1:size(v_component, 1)
            i_global = θ_range[i_local]
            for j_local in 1:size(v_component, 2)
                j_global = φ_range[j_local]
                v_component[i_local, j_local, r_local] = field[i_global, j_global]
            end
        end
    else
        common_i_range = 1:min(size(v_component, 1), size(field, 1))
        common_j_range = 1:min(size(v_component, 2), size(field, 2))
        for i in common_i_range
            for j in common_j_range
                v_component[i, j, r_local] = field[i, j]
            end
        end
    end

    return v_component
end

function store_zero_component!(v_component, r_local, config)
    if r_local > size(v_component, 3)
        return v_component
    end

    for i in axes(v_component, 1)
        for j in axes(v_component, 2)
            v_component[i, j, r_local] = zero(eltype(v_component))
        end
    end

    return v_component
end

# ---------------------------------------------------------------------------
# Phase-3 copy-loop function barriers (also used by nonlinear.jl scalar drivers)
#
# `_scalar_scratch` / `_vector_scratch` return from an `IdDict{Any,Any}`, so
# every field read from the returned NamedTuple is typed `::Any`.  Indexing an
# `::Any` array boxes every element access (~2.3 MB/call for scalar s2p at
# nr=32).  Routing the copy loops through these concrete-typed helpers lets
# Julia specialise on the real runtime array types and eliminate the boxing.
# No type annotation is needed on the arguments: Julia infers the concrete
# types from the call site.  numerics.jl is included before nonlinear.jl so
# the definitions live here and are visible to both files.
# ---------------------------------------------------------------------------

"""    _copy_spatial_to_physical!(pd, fp)
Copy `fp[j, i, k]` → `pd[i, j, k]` (fspatial/Vt/Vp → phys, synthesis direction).
`pd` is (nlat, nlon, nlev_pd), `fp` is (nlon, nlat, nlev_fp).
"""
@inline function _copy_spatial_to_physical!(pd, fp)
    nθ = size(pd, 1); nφ = size(pd, 2); nlev = min(size(fp, 3), size(pd, 3))
    @inbounds for k in 1:nlev, j in 1:nφ, i in 1:nθ
        pd[i, j, k] = fp[j, i, k]
    end
    return nothing
end

"""    _copy_spatial2_to_physical2!(pd1, pd2, fp1, fp2)
Copy two spatial arrays to two physical arrays in one pass (Vt+Vp vector synthesis).
"""
@inline function _copy_spatial2_to_physical2!(pd1, pd2, fp1, fp2)
    nθ = size(pd1, 1); nφ = size(pd1, 2); nlev = min(size(fp1, 3), size(pd1, 3))
    @inbounds for k in 1:nlev, j in 1:nφ, i in 1:nθ
        pd1[i, j, k] = fp1[j, i, k]
        pd2[i, j, k] = fp2[j, i, k]
    end
    return nothing
end

"""    _copy_physical_to_spatial!(fp, pd)
Copy `pd[i, j, k]` → `fp[j, i, k]` (phys → fspatial/Vt/Vp, analysis direction).
`pd` is (nlat, nlon, nlev_pd), `fp` is (nlon, nlat, nlev_fp).
"""
@inline function _copy_physical_to_spatial!(fp, pd)
    nθ = size(pd, 1); nφ = size(pd, 2); nlev = min(size(fp, 3), size(pd, 3))
    @inbounds for k in 1:nlev, j in 1:nφ, i in 1:nθ
        fp[j, i, k] = pd[i, j, k]
    end
    return nothing
end

"""    _copy_physical2_to_spatial2!(fp1, fp2, pd1, pd2)
Copy two physical arrays to two spatial arrays in one pass (Vt+Vp vector analysis).
"""
@inline function _copy_physical2_to_spatial2!(fp1, fp2, pd1, pd2)
    nθ = size(pd1, 1); nφ = size(pd1, 2); nlev = min(size(fp1, 3), size(pd1, 3))
    @inbounds for k in 1:nlev, j in 1:nφ, i in 1:nθ
        fp1[j, i, k] = pd1[i, j, k]
        fp2[j, i, k] = pd2[i, j, k]
    end
    return nothing
end

# Single solenoidal radial convention: u_r = l(l+1)·P/r², shared by the solver
# and MIE paths since the Stage-2 unification (the solver previously used
# l(l+1)/r and the MIE path l(l+1)/r² — two inconsistent conventions).
_solenoidal_vr_factor(l, r_val) = l * (l + 1) / (r_val * r_val)

function vector_spectral_to_physical!(
        toroidal::SpectralFieldType{T},
        poloidal::SpectralFieldType{T},
        vector_field::VectorFieldType{T};
        domain::Union{RadialDomainType, Nothing} = nothing,
        raw_spheroidal::Bool = false
) where {T}
    config = toroidal.config
    plan   = get_disttranspose_plan(config)
    vector_spectral_to_physical_disttranspose!(
        config, plan, toroidal, poloidal, vector_field, domain,
        _solenoidal_vr_factor; raw_spheroidal = raw_spheroidal)
    return vector_field
end

# Function barrier for the v_r poloidal-coefficient fill.  `Vap`/`Sp` are
# `parent(Vr_alm)`/`parent(Slm)` taken from the `::Any` IdDict scratch cache;
# running this triple loop inline boxed every scalar write (~tens of thousands
# of allocations per step — the dominant hot-path allocator).  Passing the
# arrays as ordinary arguments specializes the loop on their concrete types, so
# the writes don't box.  Same pattern as `_copy_spatial_to_physical!`.
function _fill_vr_alm!(Vap, Sp, m_local, r_range_ph, domain, lmax::Int, mmax::Int,
        vr_factor::F) where {F}
    fill!(Vap, zero(eltype(Vap)))
    rN = domain.r[domain.N, 4]
    @inbounds for (lev, r_glob) in enumerate(r_range_ph)
        lev > size(Vap, 3) && continue
        (1 <= r_glob <= domain.N) || continue
        r_val = domain.r[r_glob, 4]
        r_val > eps(Float64) * rN || continue
        for (mi, m) in enumerate(m_local)
            (0 <= m <= mmax) || continue
            for l in max(m, 0):lmax
                Vap[l + 1, mi, lev] = Sp[l + 1, mi, lev] * vr_factor(l, r_val)
            end
        end
    end
    return nothing
end

# Shared Phase-3 vector synthesis used by both the solver path (numerics.jl) and
# the non-solver path (fields/transforms.jl).
#
# Solenoidal convention (Stage 2): the stored poloidal potential P synthesizes
#   u_r = l(l+1)·P/r²   (vr_factor = _solenoidal_vr_factor for both paths)
#   tangential spheroidal scalar S = (∂_r P)/r
# (derived from u = ∇×∇×(P·Y·r̂): B_θ = (P'/r)·∂θY; then
#  ∇·u = λP'/r² − λ(P'/r)/r ≡ 0, divergence-free by construction).
#
# `raw_spheroidal = true` skips the derivative coupling and feeds the stored
# coefficients to the sphtor synthesis verbatim (and zero-fills v_r unless a
# domain is supplied). That is the tangential-basis synthesis primitive
# (∇₁S + r̂×∇₁T) used for exact spectral angular differentiation by the
# force-projection reference tests; it is NOT a velocity synthesis. The same
# raw behavior applies when `domain === nothing` (no radii to differentiate).
function vector_spectral_to_physical_disttranspose!(
        config, plan,
        toroidal, poloidal, vector_field, domain, vr_factor;
        raw_spheroidal::Bool = false)
    lmax, mmax = config.lmax, config.mmax

    sc  = _vector_scratch(config, plan)
    Slm = sc.Slm   # spheroidal/poloidal
    Tlm = sc.Tlm   # toroidal
    Vt  = sc.Vt
    Vp  = sc.Vp
    Vr  = sc.Vr
    Vr_alm = sc.Vr_alm
    solve  = sc.solve

    # 1. spec storage → Slm / Tlm via the m-axis bridge + r_comm l-transpose.
    #    Slm holds the stored POLOIDAL potential P at this point.
    spec_storage_to_solve!(config, solve, parent(poloidal.data_real),
                           parent(poloidal.data_imag), plan)
    from_spec_solve!(config, Slm, solve, plan)
    spec_storage_to_solve!(config, solve, parent(toroidal.data_real),
                           parent(toroidal.data_imag), plan)
    from_spec_solve!(config, Tlm, solve, plan)

    solenoidal = !raw_spheroidal && domain !== nothing
    r_range_ph = PencilArrays.range_local(vector_field.θ_component.pencil)[3]

    if solenoidal
        length(r_range_ph) == domain.N || error(
            "solenoidal vector synthesis requires the radial axis fully local " *
            "(got $(length(r_range_ph)) of $(domain.N) levels); " *
            "r-distributed support is a Stage-2 follow-up")
        # 2a. v_r coefficients from the ORIGINAL P (before Slm is mutated to S).
        _fill_vr_alm!(parent(Vr_alm), parent(Slm), plan.m_local, r_range_ph,
            domain, lmax, mmax, vr_factor)
        # 2b. In-place spheroidal coupling: Slm ← S = (∂_r P)/r.
        _spheroidal_from_poloidal!(parent(Slm), plan.m_local, r_range_ph,
            domain, lmax, mmax, create_derivative_matrix(Float64, 1, domain))
    end

    # 3. Distributed sphtor synthesis: (Slm, Tlm) → (Vt, Vp), batched over nr_local.
    SHTnsKit.dist_synthesis_sphtor!(plan, Vt, Vp, Slm, Tlm)

    _VECTOR_DISTTRANSPOSE_COUNT[] += 1

    # 4. Copy Vt/Vp (nlon, nlat_local, lev) → v_θ/v_φ (nlat_local, nlon, nr_local)
    #    with the first-two-axis transpose; lev ↔ r_local align per rank.
    #    Function barrier: sc is ::Any from IdDict cache; parent(Vt/Vp) would be
    #    ::Any and box every element.  Use the concrete-typed helper.
    v_theta = parent(vector_field.θ_component.data)
    v_phi   = parent(vector_field.φ_component.data)
    v_r     = parent(vector_field.r_component.data)
    _copy_spatial2_to_physical2!(v_theta, v_phi, parent(Vt), parent(Vp))

    # 5. Radial component v_r = l(l+1)·P/r² via scalar dist_synthesis! (filled
    #    in 2a from the original P). In raw mode with a domain, fill from the
    #    raw coefficients (legacy behavior); without a domain, zero-fill.
    if solenoidal
        SHTnsKit.dist_synthesis!(plan, Vr, Vr_alm)
        # Function barrier: Vr is ::Any from sc (IdDict); parent(Vr) would be ::Any.
        _copy_spatial_to_physical!(v_r, parent(Vr))
    elseif domain !== nothing
        _fill_vr_alm!(parent(Vr_alm), parent(Slm), plan.m_local, r_range_ph,
            domain, lmax, mmax, vr_factor)
        SHTnsKit.dist_synthesis!(plan, Vr, Vr_alm)
        _copy_spatial_to_physical!(v_r, parent(Vr))
    else
        fill!(v_r, zero(eltype(v_r)))
    end

    return vector_field
end

# In-place spheroidal coupling for the solenoidal synthesis: per (l,m) mode,
# S = (∂_r P)/r over the (fully local) radial lev-axis of the Alm-layout
# array `Sp` (l_index, m_local, lev). Complex profiles are differentiated as
# separate real/imaginary passes through the banded D1. Concrete-typed function
# barrier — `Sp` comes from the ::Any scratch (see _fill_vr_alm!).
function _spheroidal_from_poloidal!(Sp, m_local, r_range_ph, domain,
        lmax::Int, mmax::Int, D1)
    nrl = length(r_range_ph)
    P_re = Vector{Float64}(undef, nrl)
    P_im = Vector{Float64}(undef, nrl)
    d_re = Vector{Float64}(undef, nrl)
    d_im = Vector{Float64}(undef, nrl)
    @inbounds for (mi, m) in enumerate(m_local)
        (0 <= m <= mmax) || continue
        for l in max(m, 0):lmax
            l + 1 <= size(Sp, 1) || continue
            for lev in 1:nrl
                c = Sp[l + 1, mi, lev]
                P_re[lev] = real(c)
                P_im[lev] = imag(c)
            end
            mul!(d_re, D1, P_re)
            mul!(d_im, D1, P_im)
            for (lev, r_glob) in enumerate(r_range_ph)
                r = domain.r[r_glob, 4]
                Sp[l + 1, mi, lev] = complex(d_re[lev] / r, d_im[lev] / r)
            end
        end
    end
    return nothing
end

function vector_physical_to_spectral!(
        vector_field::VectorFieldType{T},
        toroidal::SpectralFieldType{T},
        poloidal::SpectralFieldType{T};
        domain::Union{RadialDomainType, Nothing} = nothing,
        verify_solenoidal::Bool = false,
        raw_spheroidal::Bool = false
) where {T}
    config = toroidal.config
    plan   = get_disttranspose_plan(config)
    # Tangential sphtor analysis: toroidal ← T-scalar, poloidal ← S-scalar.
    vector_physical_to_spectral_disttranspose!(
        config, plan, vector_field, toroidal, poloidal)

    if !raw_spheroidal && domain !== nothing
        # Solenoidal convention (Stage 2): the poloidal potential is recovered
        # from the RADIAL component, P = r²·Q/(l(l+1)), the exact inverse of
        # the synthesis u_r = l(l+1)·P/r². (The raw S-scalar path above is the
        # tangential-basis analysis used for force/QST decompositions.)
        scalar_physical_to_spectral!(vector_field.r_component, poloidal)
        _poloidal_from_radial_q!(parent(poloidal.data_real),
            parent(poloidal.data_imag), config, domain)
    end

    return toroidal, poloidal
end

# In-place P = r²·Q/(l(l+1)) on the spectral STORAGE arrays (Q was written by
# the scalar analysis of u_r); l = 0 carries no poloidal content.
function _poloidal_from_radial_q!(qr, qi, config, domain)
    r_range = local_range(config.pencils.spec, 3)
    @inbounds for lm in 1:config.nlm
        slot = local_spectral_storage_slot(config, lm)
        slot === nothing && continue
        l = config.l_values[lm]
        if l == 0
            for r_idx in r_range
                lr = r_idx - first(r_range) + 1
                set_local_spectral_value!(qr, slot, lr, 0.0)
                set_local_spectral_value!(qi, slot, lr, 0.0)
            end
            continue
        end
        invλ = 1.0 / (l * (l + 1))
        for r_idx in r_range
            lr = r_idx - first(r_range) + 1
            (1 <= r_idx <= domain.N) || continue
            r2 = domain.r[r_idx, 4]^2
            set_local_spectral_value!(qr, slot, lr,
                r2 * invλ * local_spectral_value(qr, slot, lr))
            set_local_spectral_value!(qi, slot, lr,
                r2 * invλ * local_spectral_value(qi, slot, lr))
        end
    end
    return nothing
end

# Shared Phase-3 vector analysis used by both the solver path (numerics.jl) and
# the non-solver path (fields/transforms.jl).  (v_θ,v_φ) → (poloidal, toroidal)
# via dist_analysis_sphtor!; v_r is NOT consumed (it is redundant for a
# solenoidal field — the toroidal-poloidal decomposition is recovered from the
# tangential components alone, matching the historical 2-component analysis).
function vector_physical_to_spectral_disttranspose!(
        config, plan, vector_field, toroidal, poloidal)
    sc  = _vector_scratch(config, plan)
    Slm = sc.Slm   # spheroidal/poloidal
    Tlm = sc.Tlm   # toroidal
    Vt  = sc.Vt
    Vp  = sc.Vp
    solve = sc.solve

    # 1. Copy v_θ/v_φ (nlat_local, nlon, nr_local) → Vt/Vp (nlon, nlat_local, lev)
    #    with the first-two-axis transpose; lev ↔ r_local align per rank.
    #    Function barrier: sc is ::Any from IdDict cache; parent(Vt/Vp) would be
    #    ::Any and box every element.  Use the concrete-typed helper.
    v_theta = parent(vector_field.θ_component.data)
    v_phi   = parent(vector_field.φ_component.data)
    _copy_physical2_to_spatial2!(parent(Vt), parent(Vp), v_theta, v_phi)

    # 2. Distributed sphtor analysis: (Vt, Vp) → (Slm=poloidal, Tlm=toroidal).
    SHTnsKit.dist_analysis_sphtor!(plan, Slm, Tlm, Vt, Vp)

    _VECTOR_DISTTRANSPOSE_COUNT[] += 1

    # 3. Alm → solve → spec storage via r_comm l-transpose + m-axis bridge, for
    #    BOTH spectral fields (sequential: solve scratch fully consumed each time).
    solve = GeoDynamo.to_spec_solve(config, Slm, plan)
    solve_to_spec_storage!(config, parent(poloidal.data_real),
                           parent(poloidal.data_imag), solve, plan)
    solve = GeoDynamo.to_spec_solve(config, Tlm, plan)
    solve_to_spec_storage!(config, parent(toroidal.data_real),
                           parent(toroidal.data_imag), solve, plan)

    return toroidal, poloidal
end

@inline function reset_velocity_work_arrays!(velocity_fields)
    T = eltype(parent(velocity_fields.work_tor.data_real))
    z = zero(T)
    fill!(parent(velocity_fields.work_tor.data_real), z)
    fill!(parent(velocity_fields.work_tor.data_imag), z)
    fill!(parent(velocity_fields.work_pol.data_real), z)
    fill!(parent(velocity_fields.work_pol.data_imag), z)
    fill!(parent(velocity_fields.work_physical.r_component.data), z)
    fill!(parent(velocity_fields.work_physical.θ_component.data), z)
    fill!(parent(velocity_fields.work_physical.φ_component.data), z)
    fill!(parent(velocity_fields.advection_physical.r_component.data), z)
    fill!(parent(velocity_fields.advection_physical.θ_component.data), z)
    fill!(parent(velocity_fields.advection_physical.φ_component.data), z)
    fill!(parent(velocity_fields.ζᵀ.data_real), z)
    fill!(parent(velocity_fields.ζᵀ.data_imag), z)
    fill!(parent(velocity_fields.ζᴾ.data_real), z)
    fill!(parent(velocity_fields.ζᴾ.data_imag), z)
    return velocity_fields
end

function refresh_velocity_physical_fields!(velocity_fields, domain)
    vector_spectral_to_physical!(
        velocity_fields.toroidal,
        velocity_fields.poloidal,
        velocity_fields.velocity;
        domain
    )
    return velocity_fields.velocity
end

function refresh_vorticity_physical_fields!(velocity_fields, domain)
    compute_vorticity_spectral!(velocity_fields, domain)
    vector_spectral_to_physical!(
        velocity_fields.ζᵀ,
        velocity_fields.ζᴾ,
        velocity_fields.vorticity;
        domain
    )
    return velocity_fields.vorticity
end

function solver_extract_local_radial_profile!(
        profile::Vector{T},
        data::AbstractArray{T, 3},
        slot::CartesianIndex{2},
        nr::Int,
        r_range
) where {T}
    fill!(profile, zero(T))
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(data, 3) && r_idx <= nr && r_idx <= length(profile)
            profile[r_idx] = local_spectral_value(data, slot, local_r)
        end
    end
    return profile
end

function apply_radial_derivative!(
        output::Vector{T},
        matrix,
        input::Vector{T}
) where {T}
    N = matrix.size
    bandwidth = matrix.bandwidth
    fill!(output, zero(T))

    @inbounds for j in 1:N
        for i in max(1, j - bandwidth):min(N, j + bandwidth)
            band_row = bandwidth + 1 + i - j
            if 1 <= band_row <= 2 * bandwidth + 1
                output[i] += matrix.data[band_row, j] * input[j]
            end
        end
    end

    return output
end

function compute_vorticity_spectral!(
        velocity_fields::VelocityFieldsType{T},
        domain
) where {T}
    u_tor_real = parent(velocity_fields.toroidal.data_real)
    u_tor_imag = parent(velocity_fields.toroidal.data_imag)
    u_pol_real = parent(velocity_fields.poloidal.data_real)
    u_pol_imag = parent(velocity_fields.poloidal.data_imag)

    ζ_tor_real = parent(velocity_fields.ζᵀ.data_real)
    ζ_tor_imag = parent(velocity_fields.ζᵀ.data_imag)
    ζ_pol_real = parent(velocity_fields.ζᴾ.data_real)
    ζ_pol_imag = parent(velocity_fields.ζᴾ.data_imag)

    config = velocity_fields.toroidal.config
    lm_range = local_spectral_mode_indices(config)
    r_range = local_range(config.pencils.spec, 3)
    nr = domain.N

    nthreads = max(1, Threads.nthreads(), Threads.maxthreadid())
    # Field-owned scratch (never a module global) — see velocity/field.jl.
    workspace = _get_or_build_velocity_workspace!(velocity_fields, nr, nthreads)

    pol_profile_real_bufs = workspace.Pᴾ_profile_real
    pol_profile_imag_bufs = workspace.Pᴾ_profile_imag
    tor_profile_real_bufs = workspace.Tᵀ_profile_real
    tor_profile_imag_bufs = workspace.Tᵀ_profile_imag
    dpol_dr_real_bufs = workspace.∂ᵣpoloidal_real
    dpol_dr_imag_bufs = workspace.∂ᵣpoloidal_imag
    d2pol_dr2_real_bufs = workspace.∂ᵣᵣpoloidal_real
    d2pol_dr2_imag_bufs = workspace.∂ᵣᵣpoloidal_imag

    @solver_threaded_local_spectral_modes lm_idx slot lm_range config velocity_fields.l_factors u_pol_real begin
        tid = Threads.threadid()
        l_factor = velocity_fields.l_factors[lm_idx]
        pol_profile_real = pol_profile_real_bufs[tid]
        pol_profile_imag = pol_profile_imag_bufs[tid]
        tor_profile_real = tor_profile_real_bufs[tid]
        tor_profile_imag = tor_profile_imag_bufs[tid]
        dpol_dr_real = dpol_dr_real_bufs[tid]
        dpol_dr_imag = dpol_dr_imag_bufs[tid]
        d2pol_dr2_real = d2pol_dr2_real_bufs[tid]
        d2pol_dr2_imag = d2pol_dr2_imag_bufs[tid]

        solver_extract_local_radial_profile!(
            pol_profile_real, u_pol_real, slot, nr, r_range)
        solver_extract_local_radial_profile!(
            pol_profile_imag, u_pol_imag, slot, nr, r_range)
        solver_extract_local_radial_profile!(
            tor_profile_real, u_tor_real, slot, nr, r_range)
        solver_extract_local_radial_profile!(
            tor_profile_imag, u_tor_imag, slot, nr, r_range)

        apply_radial_derivative!(dpol_dr_real, velocity_fields.∂r, pol_profile_real)
        apply_radial_derivative!(dpol_dr_imag, velocity_fields.∂r, pol_profile_imag)
        apply_radial_derivative!(d2pol_dr2_real, velocity_fields.∂²r, pol_profile_real)
        apply_radial_derivative!(d2pol_dr2_imag, velocity_fields.∂²r, pol_profile_imag)

        r_first = first(r_range)
        r_last = min(last(r_range), nr)
        if r_last < r_first
            continue
        end

        @inbounds @simd for r_idx in r_first:r_last
            local_r = r_idx - r_first + 1
            if local_r <= size(ζ_tor_real, 3)
                r = domain.r[r_idx, 4]
                if r == 0.0
                    set_local_spectral_value!(ζ_tor_real, slot, local_r, zero(T))
                    set_local_spectral_value!(ζ_tor_imag, slot, local_r, zero(T))
                    set_local_spectral_value!(ζ_pol_real, slot, local_r, zero(T))
                    set_local_spectral_value!(ζ_pol_imag, slot, local_r, zero(T))
                else
                    r_inv = domain.r[r_idx, 3]
                    r_inv2 = domain.r[r_idx, 2]
                    # Solenoidal convention (Stage 2): ω = ∇×u has stored
                    # potentials T_ω = (P'' − λP/r²)/r and P_ω = −r·T
                    # (verified against the Stage-1 curl projections).
                    set_local_spectral_value!(ζ_tor_real,
                        slot,
                        local_r,
                        r_inv * (
                            d2pol_dr2_real[r_idx]
                            -
                            l_factor * r_inv2 * pol_profile_real[r_idx]
                        ))
                    set_local_spectral_value!(ζ_tor_imag,
                        slot,
                        local_r,
                        r_inv * (
                            d2pol_dr2_imag[r_idx]
                            -
                            l_factor * r_inv2 * pol_profile_imag[r_idx]
                        ))
                    set_local_spectral_value!(ζ_pol_real, slot, local_r,
                        -r * tor_profile_real[r_idx])
                    set_local_spectral_value!(ζ_pol_imag, slot, local_r,
                        -r * tor_profile_imag[r_idx])
                end
            end
        end
    end

    return velocity_fields
end

function compute_velocity_body_forces!(
        velocity_fields::VelocityFieldsType{T},
        temperature_field,
        composition_field,
        magnetic_field,
        domain,
        params::SolverParameters
) where {T}
    E = params.Ek
    Pm = params.Pm
    Pr = params.Pr
    Sc = params.Sc
    Ra = params.Ra
    RaC = params.RaC

    v_r = parent(velocity_fields.velocity.r_component.data)
    v_θ = parent(velocity_fields.velocity.θ_component.data)
    v_φ = parent(velocity_fields.velocity.φ_component.data)

    ζ_r = parent(velocity_fields.vorticity.r_component.data)
    ζ_θ = parent(velocity_fields.vorticity.θ_component.data)
    ζ_φ = parent(velocity_fields.vorticity.φ_component.data)

    adv_r = parent(velocity_fields.advection_physical.r_component.data)
    adv_θ = parent(velocity_fields.advection_physical.θ_component.data)
    adv_φ = parent(velocity_fields.advection_physical.φ_component.data)

    config = velocity_fields.velocity.r_component.config
    local_size = size(v_r)
    r_range = local_range(config.pencils.r, 3)
    θ_range = local_range(config.pencils.r, 1)

    adv_coeff = T(E)
    @inbounds Threads.@threads for k in 1:local_size[3]
        r_idx = k + first(r_range) - 1
        if r_idx <= domain.N
            r = domain.r[r_idx, 4]
        else
            r = 1.0
        end

        for j in 1:local_size[2]
            @simd for i in 1:local_size[1]
                theta_idx_global = θ_range[i]
                sin_theta = velocity_fields.coriolis_factors[1, theta_idx_global]
                cos_theta = velocity_fields.coriolis_factors[2, theta_idx_global]
                linear_idx = i + (j - 1) * local_size[1] +
                             (k - 1) * local_size[1] * local_size[2]

                if linear_idx <= length(v_r)
                    u_r = v_r[linear_idx]
                    u_θ = v_θ[linear_idx]
                    u_φ = v_φ[linear_idx]

                    ω_r = ζ_r[linear_idx]
                    ω_θ = ζ_θ[linear_idx]
                    ω_φ = ζ_φ[linear_idx]

                    adv_r_val = adv_coeff * (u_θ * ω_φ - u_φ * ω_θ)
                    adv_θ_val = adv_coeff * (u_φ * ω_r - u_r * ω_φ)
                    adv_φ_val = adv_coeff * (u_r * ω_θ - u_θ * ω_r)

                    zhat_cross_r = -sin_theta * u_φ
                    zhat_cross_θ = -cos_theta * u_φ
                    zhat_cross_φ = cos_theta * u_θ + sin_theta * u_r

                    adv_r[linear_idx] = adv_r_val - zhat_cross_r
                    adv_θ[linear_idx] = adv_θ_val - zhat_cross_θ
                    adv_φ[linear_idx] = adv_φ_val - zhat_cross_φ
                end
            end
        end
    end

    if temperature_field !== nothing
        solver_add_thermal_buoyancy_force!(adv_r, temperature_field, (Pm / Pr) * Ra, domain)
    end

    if composition_field !== nothing
        add_compositional_buoyancy_force!(adv_r, composition_field, (Pm / Sc) * RaC, domain)
    end

    if magnetic_field !== nothing
        solver_add_lorentz_force!(velocity_fields, magnetic_field, Pm)
    end

    return velocity_fields
end

function scalar_field_data_and_config(field)
    if hasproperty(field, :data) && hasproperty(field, :config)
        return parent(getproperty(field, :data)), getproperty(field, :config)
    elseif hasproperty(field, :temperature)
        temperature = getproperty(field, :temperature)
        return parent(temperature.data), temperature.config
    elseif hasproperty(field, :composition)
        composition = getproperty(field, :composition)
        return parent(composition.data), composition.config
    end

    error("Unsupported solver scalar field container: $(typeof(field))")
end

function solver_add_thermal_buoyancy_force!(
        force_r::AbstractArray{T, 3},
        scalar_field,
        factor::Float64,
        domain
) where {T}
    iszero(factor) && return force_r
    scalar_data, config = scalar_field_data_and_config(scalar_field)
    r_range = local_range(config.pencils.r, 3)
    local_size = size(force_r)

    @inbounds Threads.@threads for k in 1:local_size[3]
        r_idx = k + first(r_range) - 1
        r = r_idx <= domain.N ? domain.r[r_idx, 4] : 1.0
        factor_r = factor * r
        for j in 1:local_size[2]
            @simd for i in 1:local_size[1]
                linear_idx = i + (j - 1) * local_size[1] +
                             (k - 1) * local_size[1] * local_size[2]
                if linear_idx <= length(scalar_data)
                    force_r[linear_idx] += factor_r * scalar_data[linear_idx]
                end
            end
        end
    end

    return force_r
end

function add_compositional_buoyancy_force!(
        force_r::AbstractArray{T, 3},
        composition_field,
        factor::Float64,
        domain
) where {T}
    iszero(factor) && return force_r
    composition_data, config = scalar_field_data_and_config(composition_field)
    r_range = local_range(config.pencils.r, 3)
    local_size = size(force_r)

    @inbounds Threads.@threads for k in 1:local_size[3]
        r_idx = k + first(r_range) - 1
        r = r_idx <= domain.N ? domain.r[r_idx, 4] : 1.0
        factor_r = factor * r
        for j in 1:local_size[2]
            @simd for i in 1:local_size[1]
                linear_idx = i + (j - 1) * local_size[1] +
                             (k - 1) * local_size[1] * local_size[2]
                if linear_idx <= length(composition_data)
                    force_r[linear_idx] += factor_r * composition_data[linear_idx]
                end
            end
        end
    end

    return force_r
end

function solver_add_lorentz_force!(velocity_fields, magnetic_field, Pm::Float64)
    j_r = parent(magnetic_field.current.r_component.data)
    j_θ = parent(magnetic_field.current.θ_component.data)
    j_φ = parent(magnetic_field.current.φ_component.data)

    B_r = parent(magnetic_field.magnetic.r_component.data)
    B_θ = parent(magnetic_field.magnetic.θ_component.data)
    B_φ = parent(magnetic_field.magnetic.φ_component.data)

    adv_r = parent(velocity_fields.advection_physical.r_component.data)
    adv_θ = parent(velocity_fields.advection_physical.θ_component.data)
    adv_φ = parent(velocity_fields.advection_physical.φ_component.data)

    lorentz_coeff = eltype(adv_r)(1.0 / Pm)
    @inbounds @simd for idx in eachindex(j_r)
        if idx <= length(B_r)
            adv_r[idx] += lorentz_coeff * (j_θ[idx] * B_φ[idx] - j_φ[idx] * B_θ[idx])
            adv_θ[idx] += lorentz_coeff * (j_φ[idx] * B_r[idx] - j_r[idx] * B_φ[idx])
            adv_φ[idx] += lorentz_coeff * (j_r[idx] * B_θ[idx] - j_θ[idx] * B_r[idx])
        end
    end

    return velocity_fields
end

@inline function reset_magnetic_work_arrays!(magnetic_fields)
    T = eltype(parent(magnetic_fields.work_tor.data_real))
    z = zero(T)
    fill!(parent(magnetic_fields.work_tor.data_real), z)
    fill!(parent(magnetic_fields.work_tor.data_imag), z)
    fill!(parent(magnetic_fields.work_pol.data_real), z)
    fill!(parent(magnetic_fields.work_pol.data_imag), z)
    fill!(parent(magnetic_fields.work_physical.r_component.data), z)
    fill!(parent(magnetic_fields.work_physical.θ_component.data), z)
    fill!(parent(magnetic_fields.work_physical.φ_component.data), z)
    fill!(parent(magnetic_fields.induction_physical.r_component.data), z)
    fill!(parent(magnetic_fields.induction_physical.θ_component.data), z)
    fill!(parent(magnetic_fields.induction_physical.φ_component.data), z)
    return magnetic_fields
end

function refresh_magnetic_physical_fields!(magnetic_fields, outer_domain)
    vector_spectral_to_physical!(
        magnetic_fields.toroidal,
        magnetic_fields.poloidal,
        magnetic_fields.magnetic;
        domain = outer_domain
    )
    return magnetic_fields.magnetic
end

function refresh_current_physical_fields!(magnetic_fields, outer_domain)
    solver_compute_current_density_spectral!(magnetic_fields, outer_domain)
    vector_spectral_to_physical!(
        magnetic_fields.work_tor,
        magnetic_fields.work_pol,
        magnetic_fields.current;
        domain = outer_domain
    )
    return magnetic_fields.current
end

function spectral_curl_torpol!(
        dst_tor_r,
        dst_tor_i,
        dst_pol_r,
        dst_pol_i,
        src_tor_r,
        src_tor_i,
        src_pol_r,
        src_pol_i,
        l_factors,
        d1_matrix,
        d²_matrix,
        domain::RadialDomainType,
        config::SHTnsConfigType,
        ::Type{T};
        _work::Union{Nothing, NTuple{6, Vector{T}}} = nothing
) where {T}
    lm_range = local_spectral_mode_indices(config)
    r_range = local_range(config.pencils.spec, 3)
    nr = domain.N

    if _work !== nothing
        Pᴾ_profile_real, Pᴾ_profile_imag, dᴾ_dr_real, dᴾ_dr_imag, d²ᴾ_dr²_real,
        d²ᴾ_dr²_imag = _work
    else
        Pᴾ_profile_real = zeros(T, nr)
        Pᴾ_profile_imag = zeros(T, nr)
        dᴾ_dr_real = zeros(T, nr)
        dᴾ_dr_imag = zeros(T, nr)
        d²ᴾ_dr²_real = zeros(T, nr)
        d²ᴾ_dr²_imag = zeros(T, nr)
    end

    @solver_local_spectral_modes lm_idx slot lm_range config l_factors src_pol_r begin
        l_factor = l_factors[lm_idx]

        fill!(Pᴾ_profile_real, zero(T))
        fill!(Pᴾ_profile_imag, zero(T))
        gather_local_radial_profile!(Pᴾ_profile_real, Pᴾ_profile_imag,
            src_pol_r, src_pol_i, slot, r_range)

        apply_radial_derivative!(dᴾ_dr_real, d1_matrix, Pᴾ_profile_real)
        apply_radial_derivative!(dᴾ_dr_imag, d1_matrix, Pᴾ_profile_imag)
        apply_radial_derivative!(d²ᴾ_dr²_real, d²_matrix, Pᴾ_profile_real)
        apply_radial_derivative!(d²ᴾ_dr²_imag, d²_matrix, Pᴾ_profile_imag)

        r_first = first(r_range)
        r_last = min(last(r_range), nr)
        r_last < r_first && continue

        @inbounds @simd for r_idx in r_first:r_last
            local_r = r_idx - r_first + 1
            if local_r <= size(dst_tor_r, 3)
                r_val = domain.r[r_idx, 4]
                if r_val == 0.0
                    set_local_spectral_value!(dst_tor_r, slot, local_r, zero(T))
                    set_local_spectral_value!(dst_tor_i, slot, local_r, zero(T))
                    set_local_spectral_value!(dst_pol_r, slot, local_r, zero(T))
                    set_local_spectral_value!(dst_pol_i, slot, local_r, zero(T))
                else
                    r⁻¹ = domain.r[r_idx, 3]
                    r⁻² = domain.r[r_idx, 2]
                    set_local_spectral_value!(dst_tor_r,
                        slot,
                        local_r,
                        (
                            l_factor * r⁻² * Pᴾ_profile_real[r_idx] -
                            d²ᴾ_dr²_real[r_idx] -
                            2.0 * r⁻¹ * dᴾ_dr_real[r_idx]
                        ))
                    set_local_spectral_value!(dst_tor_i,
                        slot,
                        local_r,
                        (
                            l_factor * r⁻² * Pᴾ_profile_imag[r_idx] -
                            d²ᴾ_dr²_imag[r_idx] -
                            2.0 * r⁻¹ * dᴾ_dr_imag[r_idx]
                        ))
                    set_local_spectral_value!(dst_pol_r, slot, local_r,
                        -l_factor * r⁻² * local_spectral_value(src_tor_r, slot, local_r))
                    set_local_spectral_value!(dst_pol_i, slot, local_r,
                        -l_factor * r⁻² * local_spectral_value(src_tor_i, slot, local_r))
                end
            end
        end
    end

    return dst_tor_r, dst_tor_i, dst_pol_r, dst_pol_i
end

function solver_compute_current_density_spectral!(magnetic_fields, outer_domain)
    T = eltype(parent(magnetic_fields.work_tor.data_real))
    spectral_curl_torpol!(
        parent(magnetic_fields.work_tor.data_real), parent(magnetic_fields.work_tor.data_imag),
        parent(magnetic_fields.work_pol.data_real), parent(magnetic_fields.work_pol.data_imag),
        parent(magnetic_fields.toroidal.data_real), parent(magnetic_fields.toroidal.data_imag),
        parent(magnetic_fields.poloidal.data_real), parent(magnetic_fields.poloidal.data_imag),
        magnetic_fields.l_factors,
        magnetic_fields.∂r,
        magnetic_fields.∂²r,
        outer_domain,
        magnetic_fields.toroidal.config,
        T;
        _work = magnetic_fields.curl_work
    )
    return magnetic_fields
end

function apply_induction_nonlinear!(
        magnetic_fields,
        velocity_fields;
        geometry::Symbol
)
    solver_compute_velocity_cross_magnetic!(magnetic_fields, velocity_fields)
    if geometry === :ball
        # Ball geometry: still on the legacy potential-style curl (deferred per
        # the double-curl spec; regularity conditions differ at r→0).
        solver_ball_vector_analysis!(
            magnetic_fields.induction_physical,
            magnetic_fields.work_tor,
            magnetic_fields.work_pol
        )
        solver_compute_curl_of_induction!(magnetic_fields)
    else
        # Stage-4 solenoidal convention. E = u×B is NOT solenoidal; the curl's
        # stored potentials follow from the verified projection identities:
        #   P_{∇×E} = −r·T_E              (T_E = raw toroidal sphtor scalar)
        #   T_{∇×E} = −(1/r)·(Q_E − ∂_r(r·S_E))
        # with Q_E from the radial component. The old path treated (T_E, S_E)
        # as potentials and never consumed Q_E — the same dropped-radial flaw
        # the momentum equation had.
        vector_physical_to_spectral!(
            magnetic_fields.induction_physical,
            magnetic_fields.work_tor,
            magnetic_fields.work_pol;
            raw_spheroidal = true
        )
        scalar_physical_to_spectral!(
            magnetic_fields.induction_physical.r_component,
            magnetic_fields.nl_toroidal     # Q_E scratch; combined in place below
        )
        _induction_curl_potentials!(magnetic_fields)
    end
    return magnetic_fields
end

# nl_toroidal holds Q_E on entry (scratch); work_tor/work_pol hold (T_E, S_E).
# Writes nl_toroidal ← T_{∇×E} = −(1/r)(Q_E − ∂_r(r·S_E)) and
# nl_poloidal ← P_{∇×E} = −r·T_E, per mode over the (r-local) radial axis.
function _induction_curl_potentials!(magnetic_fields)
    cfg = magnetic_fields.nl_toroidal.config
    domain = magnetic_fields.outer_domain
    T = eltype(parent(magnetic_fields.nl_toroidal.data_real))
    D1 = create_derivative_matrix(T, 1, domain)
    nr = domain.N
    rS = Vector{T}(undef, nr)
    drS = Vector{T}(undef, nr)
    r_range = local_range(magnetic_fields.nl_toroidal.pencil, 3)
    length(r_range) == nr || error(
        "induction curl projection requires the radial axis fully local " *
        "(got $(length(r_range)) of $nr levels); r-distributed support is a follow-up")

    for (q_arr, s_arr, t_arr, npol_arr) in (
        (parent(magnetic_fields.nl_toroidal.data_real),
         parent(magnetic_fields.work_pol.data_real),
         parent(magnetic_fields.work_tor.data_real),
         parent(magnetic_fields.nl_poloidal.data_real)),
        (parent(magnetic_fields.nl_toroidal.data_imag),
         parent(magnetic_fields.work_pol.data_imag),
         parent(magnetic_fields.work_tor.data_imag),
         parent(magnetic_fields.nl_poloidal.data_imag)),
    )
        @inbounds for lm in 1:cfg.nlm
            slot = local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            for r_idx in 1:nr
                rS[r_idx] = T(domain.r[r_idx, 4]) *
                            local_spectral_value(s_arr, slot, r_idx)
            end
            mul!(drS, D1, rS)
            for r_idx in 1:nr
                r = T(domain.r[r_idx, 4])
                q = local_spectral_value(q_arr, slot, r_idx)
                t = local_spectral_value(t_arr, slot, r_idx)
                set_local_spectral_value!(q_arr, slot, r_idx,
                    -(q - drS[r_idx]) / r)        # nl_toroidal ← T_{∇×E}
                set_local_spectral_value!(npol_arr, slot, r_idx,
                    -r * t)                        # nl_poloidal ← P_{∇×E}
            end
        end
    end
    return magnetic_fields
end

function apply_inner_core_rotation!(magnetic_fields, rotation_rate)
    ic_tor_real = parent(magnetic_fields.toroidal_ic.data_real)
    ic_tor_imag = parent(magnetic_fields.toroidal_ic.data_imag)
    ic_pol_real = parent(magnetic_fields.poloidal_ic.data_real)
    ic_pol_imag = parent(magnetic_fields.poloidal_ic.data_imag)

    nl_tor_real = parent(magnetic_fields.nl_toroidal.data_real)
    nl_tor_imag = parent(magnetic_fields.nl_toroidal.data_imag)
    nl_pol_real = parent(magnetic_fields.nl_poloidal.data_real)
    nl_pol_imag = parent(magnetic_fields.nl_poloidal.data_imag)

    lm_range = local_range(magnetic_fields.toroidal_ic.pencil, 1)
    r_range = local_range(magnetic_fields.toroidal_ic.pencil, 3)
    rotation_factor = rotation_rate

    @inbounds for lm_idx in lm_range
        if lm_idx <= magnetic_fields.toroidal_ic.nlm
            slot = local_spectral_storage_slot(magnetic_fields.toroidal.config, lm_idx)
            slot === nothing && continue
            m = magnetic_fields.toroidal.config.m_values[lm_idx]
            if m != 0 && 1 in r_range
                local_r = 1 - first(r_range) + 1
                if local_r <= size(nl_tor_real, 3)
                    coupling_factor = rotation_factor * Float64(m)
                    set_local_spectral_value!(
                        nl_tor_real, slot, local_r,
                        local_spectral_value(nl_tor_real, slot, local_r) +
                        coupling_factor * local_spectral_value(ic_pol_imag, slot, local_r)
                    )
                    set_local_spectral_value!(
                        nl_tor_imag, slot, local_r,
                        local_spectral_value(nl_tor_imag, slot, local_r) -
                        coupling_factor * local_spectral_value(ic_pol_real, slot, local_r)
                    )
                    set_local_spectral_value!(
                        nl_pol_real, slot, local_r,
                        local_spectral_value(nl_pol_real, slot, local_r) -
                        coupling_factor * local_spectral_value(ic_tor_imag, slot, local_r)
                    )
                    set_local_spectral_value!(
                        nl_pol_imag, slot, local_r,
                        local_spectral_value(nl_pol_imag, slot, local_r) +
                        coupling_factor * local_spectral_value(ic_tor_real, slot, local_r)
                    )
                end
            end
        end
    end
    return magnetic_fields
end

function solver_compute_velocity_cross_magnetic!(magnetic_fields, velocity_fields)
    u_r = parent(velocity_fields.velocity.r_component.data)
    u_θ = parent(velocity_fields.velocity.θ_component.data)
    u_φ = parent(velocity_fields.velocity.φ_component.data)

    B_r = parent(magnetic_fields.magnetic.r_component.data)
    B_θ = parent(magnetic_fields.magnetic.θ_component.data)
    B_φ = parent(magnetic_fields.magnetic.φ_component.data)

    uB_r = parent(magnetic_fields.induction_physical.r_component.data)
    uB_θ = parent(magnetic_fields.induction_physical.θ_component.data)
    uB_φ = parent(magnetic_fields.induction_physical.φ_component.data)

    @assert length(u_r) == length(B_r) "Velocity and magnetic field arrays must have the same size"
    @inbounds @simd for idx in eachindex(u_r)
        uB_r[idx] = u_θ[idx] * B_φ[idx] - u_φ[idx] * B_θ[idx]
        uB_θ[idx] = u_φ[idx] * B_r[idx] - u_r[idx] * B_φ[idx]
        uB_φ[idx] = u_r[idx] * B_θ[idx] - u_θ[idx] * B_r[idx]
    end
    return magnetic_fields
end

function solver_compute_curl_of_induction!(magnetic_fields)
    T = eltype(parent(magnetic_fields.nl_toroidal.data_real))
    spectral_curl_torpol!(
        parent(magnetic_fields.nl_toroidal.data_real), parent(magnetic_fields.nl_toroidal.data_imag),
        parent(magnetic_fields.nl_poloidal.data_real), parent(magnetic_fields.nl_poloidal.data_imag),
        parent(magnetic_fields.work_tor.data_real), parent(magnetic_fields.work_tor.data_imag),
        parent(magnetic_fields.work_pol.data_real), parent(magnetic_fields.work_pol.data_imag),
        magnetic_fields.l_factors,
        magnetic_fields.∂r,
        magnetic_fields.∂²r,
        magnetic_fields.outer_domain,
        magnetic_fields.toroidal.config,
        T;
        _work = magnetic_fields.curl_work
    )
    return magnetic_fields
end

function solver_enforce_ball_vector_regularity!(
        tor_spec::SpectralFieldType,
        pol_spec::SpectralFieldType
)
    cfg = tor_spec.config
    lm_range = local_spectral_mode_indices(cfg)
    r_range = local_range(cfg.pencils.spec, 3)

    if !(1 in r_range)
        return tor_spec, pol_spec
    end

    r_local_idx = 1 - first(r_range) + 1

    for spec in (tor_spec, pol_spec)
        spec_real = parent(spec.data_real)
        spec_imag = parent(spec.data_imag)
        T = eltype(spec_real)

        @inbounds for lm_idx in lm_range
            if lm_idx <= cfg.nlm
                slot = local_spectral_storage_slot(cfg, lm_idx)
                slot === nothing && continue
                l = cfg.l_values[lm_idx]
                if l >= 1
                    set_local_spectral_value!(spec_real, slot, r_local_idx, zero(T))
                    set_local_spectral_value!(spec_imag, slot, r_local_idx, zero(T))
                end
            end
        end
    end

    return tor_spec, pol_spec
end

function ball_vector_physical_to_spectral!(vector_field, toroidal, poloidal)
    vector_physical_to_spectral!(vector_field, toroidal, poloidal)
    solver_enforce_ball_vector_regularity!(toroidal, poloidal)
    return toroidal, poloidal
end

function solver_ball_vector_analysis!(vector_field, toroidal, poloidal)
    return ball_vector_physical_to_spectral!(vector_field, toroidal, poloidal)
end

@inline solver_band_row(i::Int, j::Int, bw::Int) = bw + 1 + i - j

function solver_banded_to_dense(
        matrix::Union{OldBandedMatrix{T}, BandedOperator{T}},
) where {T}
    n = matrix.size
    bandwidth = matrix.bandwidth
    dense = zeros(T, n, n)

    for j in 1:n
        for i in max(1, j - bandwidth):min(n, j + bandwidth)
            band_row = bandwidth + 1 + i - j
            dense[i, j] = matrix.data[band_row, j]
        end
    end

    return dense
end

function solver_factorize_banded(A::BandedOperator{T}) where {T}
    n = A.size
    bw = A.bandwidth
    lu = copy(A.data)

    @inbounds for k in 1:(n - 1)
        piv_row = solver_band_row(k, k, bw)
        if !(1 <= piv_row <= 2 * bw + 1)
            continue
        end
        piv = lu[piv_row, k]
        tol = pivot_tol(T)
        if abs(piv) < tol
            error(
                "Singular matrix detected during solver LU factorization at pivot $k. " *
                "Pivot value = $piv, below tolerance $tol.",
            )
        end

        i_max = min(n, k + bw)
        for i in (k + 1):i_max
            row = solver_band_row(i, k, bw)
            if 1 <= row <= 2 * bw + 1
                L = lu[row, k] / piv
                lu[row, k] = L
                j_max = min(n, k + bw)
                for j in (k + 1):j_max
                    col = solver_band_row(i, j, bw)
                    if 1 <= col <= 2 * bw + 1
                        urow = solver_band_row(k, j, bw)
                        if 1 <= urow <= 2 * bw + 1
                            lu[col, j] -= L * lu[urow, j]
                        end
                    end
                end
            end
        end
    end

    return BandedFactorization{T}(lu, bw, n)
end

function solve_banded!(
        x::Vector{T},
        lu::Union{OldBandedLU{T}, BandedFactorization{T}},
        b::Vector{T}
) where {T}
    n = lu.size
    bw = lu.bandwidth

    @inbounds for i in 1:n
        s = zero(T)
        j_min = max(1, i - bw)
        for j in j_min:(i - 1)
            row = solver_band_row(i, j, bw)
            if 1 <= row <= 2 * bw + 1
                s += lu.lu[row, j] * x[j]
            end
        end
        x[i] = b[i] - s
    end

    @inbounds for i in n:-1:1
        s = zero(T)
        j_max = min(n, i + bw)
        for j in (i + 1):j_max
            row = solver_band_row(i, j, bw)
            if 1 <= row <= 2 * bw + 1
                s += lu.lu[row, j] * x[j]
            end
        end
        diag_row = solver_band_row(i, i, bw)
        diag_val = lu.lu[diag_row, i]
        tol = pivot_tol(T)
        if abs(diag_val) < tol
            error(
                "Zero diagonal detected during solver back substitution at row $i. " *
                "Diagonal value = $diag_val, below tolerance $tol.",
            )
        end
        x[i] = (x[i] - s) / diag_val
    end

    return x
end

@eval GeoDynamo begin
    """
        apply_banded_full!(out, B, v)

    Apply a banded radial operator to a full vector without materializing the
    dense matrix.

    This is the single implementation used by both the solver internals and the
    low-level public API.
    """
    function apply_banded_full!(
            out::Vector{T},
            B::Union{$(OldBandedMatrix){T}, $(BandedOperator){T}},
            v::AbstractVector{T}
    ) where {T}
        fill!(out, zero(T))
        n = B.size
        bw = B.bandwidth
        @inbounds for j in 1:n
            for i in max(1, j - bw):min(n, j + bw)
                row = solver_band_row(i, j, bw)
                if 1 <= row <= 2 * bw + 1
                    out[i] += B.data[row, j] * v[j]
                end
            end
        end
        return out
    end
end

function solver_build_banded_A(
        ::Type{T},
        domain::RadialDomainType,
        diffusivity::Float64,
        l::Int
) where {T}
    lap = build_radial_laplacian(domain)
    data = diffusivity .* lap.data
    nr = domain.N
    r_inv2 = @views domain.r[1:nr, 2]
    l_factor = Float64(l * (l + 1))
    bw = lap.bandwidth
    @inbounds for n in 1:nr
        data[bw + 1, n] -= diffusivity * l_factor * r_inv2[n]
    end
    return BandedOperator{T}(Matrix{T}(data), bw, nr)
end

function solver_phi1_action_krylov(
        Aop!,
        A_lu::Union{OldBandedLU{T}, BandedFactorization{T}},
        v::Vector{T},
        dt::Float64;
        m::Int = 20,
        tol::Float64 = 1e-8
) where {T}
    if LA.norm(v) < series_tol(T)
        return zeros(T, length(v))
    end

    if dt < 1e-8
        Av = similar(v)
        Aop!(Av, v)
        local_scale = abs(dt) * LA.norm(Av) / max(LA.norm(v), eps(real(T)))
        if local_scale < sqrt(eps(real(T)))
            return v .+ (dt / 2) .* Av
        end
    end

    ev = krylov_exp_action(Aop!, v, dt; m, tol)
    c = ev .- v
    x = copy(c)

    try
        solve_banded!(x, A_lu, c)
        @. x = x / dt

        if !all(isfinite, x)
            error(
                "Non-finite result in solver_phi1_action_krylov. " *
                "Consider reducing dt or checking the banded operator conditioning.",
            )
        end

        return x
    catch e
        e isa ErrorException && rethrow(e)
        error("Banded solve failed in solver_phi1_action_krylov: $e")
    end
end

function phi1_action_krylov!(
        dest::Vector{T},
        Aop!,
        A_lu::Union{OldBandedLU{T}, BandedFactorization{T}},
        v::Vector{T},
        dt::Float64;
        m::Int = 20,
        tol::Float64 = 1e-8,
        work::Union{SolverKrylovWork{T}, Nothing} = nothing
) where {T}
    if work === nothing
        copyto!(dest, solver_phi1_action_krylov(Aop!, A_lu, v, dt; m, tol))
        return dest
    end

    ensure_krylov_work!(work, length(v), m)

    if LA.norm(v) < series_tol(T)
        fill!(dest, zero(T))
        return dest
    end

    if dt < 1e-8
        Aop!(work.w, v)
        local_scale = abs(dt) * LA.norm(work.w) / max(LA.norm(v), eps(real(T)))
        if local_scale < sqrt(eps(real(T)))
            @. dest = v + (dt / 2) * work.w
            return dest
        end
    end

    krylov_exp_action!(work.tmp, Aop!, v, dt, work; m, tol)
    @. work.c = work.tmp - v

    try
        solve_banded!(dest, A_lu, work.c)
        @. dest = dest / dt

        if !all(isfinite, dest)
            error(
                "Non-finite result in solver_phi1_action_krylov. " *
                "Consider reducing dt or checking the banded operator conditioning.",
            )
        end
    catch e
        e isa ErrorException && rethrow(e)
        error("Banded solve failed in solver_phi1_action_krylov: $e")
    end

    return dest
end

@eval GeoDynamo begin
    """
        exp_action_krylov(Aop!, v, dt; m=20, tol=1e-8)

    Compute `exp(dt*A)v` with an Arnoldi/Krylov approximation.

    The high-level timestep drivers call `exp_action_krylov` directly,
    but this root-level entry point remains available as a reusable numerical
    utility.
    """
    function exp_action_krylov(Aop!, v, dt; m::Int = 20, tol::Float64 = 1e-8)
        return krylov_exp_action(Aop!, v, dt; m, tol)
    end

    function exp_action_krylov!(
            dest::Vector{T},
            Aop!,
            v::Vector{T},
            dt;
            m::Int = 20,
            tol::Float64 = 1e-8,
            work::Union{SolverKrylovWork{T}, Nothing} = nothing
    ) where {T}
        if work === nothing
            copyto!(dest, krylov_exp_action(Aop!, v, dt; m, tol))
        else
            krylov_exp_action!(dest, Aop!, v, dt, work; m, tol)
        end
        return dest
    end
end

function solver_synchronize_pencil_transforms!(field::SpectralFieldType{T}) where {T}
    mpi_barrier!()
    field
end

function rcond_estimate(lu_A, A::Matrix{T}) where {T}
    anorm = LA.opnorm(A, 1)
    if anorm == zero(T)
        return one(T)
    end
    try
        return LA.LAPACK.gecon!('1', copy(lu_A.factors), anorm)
    catch
        c = LA.cond(A, 1)
        return isfinite(c) && c > zero(T) ? one(T) / c : zero(T)
    end
end

function phi1_series(A::Matrix{T}) where {T}
    nr = size(A, 1)
    result = Matrix{T}(LA.I, nr, nr)
    A_power = copy(result)
    for k in 1:15
        A_power = A_power * A
        term = A_power / factorial(k + 1)
        result += term
        if LA.opnorm(term) < series_tol(T)
            break
        end
    end
    return result
end

function phi2_series(A::Matrix{T}) where {T}
    nr = size(A, 1)
    result = Matrix{T}(LA.I, nr, nr) / 2
    A_power = copy(A)
    factorial_term = 6
    result += A_power / factorial_term

    for k in 2:15
        A_power = A_power * A
        factorial_term *= (k + 2)
        term = A_power / factorial_term
        result += term
        if LA.opnorm(term) < series_tol(T)
            break
        end
    end

    return result
end

function solver_compute_phi1_function(A::Matrix{T}, expA::Matrix{T}) where {T}
    nr = size(A, 1)
    I_mat = Matrix{T}(LA.I, nr, nr)

    if !all(isfinite, A) || !all(isfinite, expA)
        throw(ArgumentError("Non-finite values detected in solver φ1 computation"))
    end

    if LA.opnorm(A) < sqrt(eps(T))
        return phi1_series(A)
    end

    diff = expA - I_mat

    try
        lu_A = LA.lu(A)
        rc = rcond_estimate(lu_A, A)
        if rc < rcond_fallback_tol(T)
            @debug "Ill-conditioned matrix in solver φ1 computation (rcond = $rc), using series expansion"
            return phi1_series(A)
        end

        result = lu_A \ diff
        if !all(isfinite, result)
            @warn "Non-finite result in solver φ1 computation, falling back to series expansion"
            return phi1_series(A)
        end
        return result
    catch e
        @debug "LU factorization failed in solver φ1 computation: $e, using series expansion"
        try
            return phi1_series(A)
        catch e2
            @error "Complete failure in solver φ1 computation: $e2, returning identity"
            return I_mat
        end
    end
end

mutable struct SolverPhi2ConditioningMonitor
    worst_rcond::Float64
    worst_l::Int
    series_expansion_count::Int
    lu_failure_count::Int
    nonfinite_count::Int
    last_report_step::Int
    enable_monitoring::Bool
end

const SOLVER_PHI2_MONITOR = SolverPhi2ConditioningMonitor(1.0, 0, 0, 0, 0, 0, true)

function reset_solver_phi2_monitor!()
    SOLVER_PHI2_MONITOR.worst_rcond = 1.0
    SOLVER_PHI2_MONITOR.worst_l = 0
    SOLVER_PHI2_MONITOR.series_expansion_count = 0
    SOLVER_PHI2_MONITOR.lu_failure_count = 0
    SOLVER_PHI2_MONITOR.nonfinite_count = 0
    SOLVER_PHI2_MONITOR.last_report_step = 0
    return nothing
end

function report_solver_phi2_conditioning(step::Int; interval::Int = 100)
    SOLVER_PHI2_MONITOR.enable_monitoring || return nothing

    if step - SOLVER_PHI2_MONITOR.last_report_step >= interval
        if mpi_rank() == 0
            @info """
            ╔══════════════════════════════════════════════════════════╗
            ║        Solver φ₂ Conditioning Report (Step $step)
            ╠══════════════════════════════════════════════════════════╣
            ║ Worst rcond:             $(SOLVER_PHI2_MONITOR.worst_rcond)
            ║ Worst mode (l):          $(SOLVER_PHI2_MONITOR.worst_l)
            ║ Series expansion used:   $(SOLVER_PHI2_MONITOR.series_expansion_count) times
            ║ LU failures:             $(SOLVER_PHI2_MONITOR.lu_failure_count) times
            ║ Non-finite values:       $(SOLVER_PHI2_MONITOR.nonfinite_count) times
            ╚══════════════════════════════════════════════════════════╝
            """
        end
        reset_solver_phi2_monitor!()
        SOLVER_PHI2_MONITOR.last_report_step = step
    end

    return nothing
end

function solver_compute_phi2_function(A::Matrix{T}, expA::Matrix{T}; l::Int = 0) where {T}
    nr = size(A, 1)
    I_mat = Matrix{T}(LA.I, nr, nr)

    if !all(isfinite, A) || !all(isfinite, expA)
        SOLVER_PHI2_MONITOR.enable_monitoring && (SOLVER_PHI2_MONITOR.nonfinite_count += 1)
        throw(ArgumentError("Non-finite values detected in solver φ2 computation (l=$l)"))
    end

    if LA.opnorm(A) < sqrt(eps(T))
        return phi2_series(A)
    end

    diff = expA - I_mat - A

    try
        lu_A = LA.lu(A)
        rcond_val = rcond_estimate(lu_A, A)
        if SOLVER_PHI2_MONITOR.enable_monitoring &&
           rcond_val < SOLVER_PHI2_MONITOR.worst_rcond
            SOLVER_PHI2_MONITOR.worst_rcond = rcond_val
            SOLVER_PHI2_MONITOR.worst_l = l
        end

        if rcond_val < rcond_fallback_tol(T)
            if SOLVER_PHI2_MONITOR.enable_monitoring
                SOLVER_PHI2_MONITOR.series_expansion_count += 1
            end
            @debug "Ill-conditioned matrix in solver φ2 computation (l=$l, rcond=$rcond_val), using series expansion"
            return phi2_series(A)
        end

        temp = lu_A \ diff
        result = lu_A \ temp

        if !all(isfinite, result)
            @warn "Non-finite result in solver φ2 computation, falling back to series expansion"
            return phi2_series(A)
        end

        return result
    catch e
        if SOLVER_PHI2_MONITOR.enable_monitoring
            SOLVER_PHI2_MONITOR.lu_failure_count += 1
        end
        @debug "LU factorization failed in solver φ2 computation (l=$l): $e, using series expansion"
        try
            return phi2_series(A)
        catch e2
            @error "Complete failure in solver φ2 computation: $e2, returning zero matrix"
            return zeros(T, nr, nr)
        end
    end
end
