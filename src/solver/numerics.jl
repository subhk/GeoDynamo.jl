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

@inline function solver_mode_index_cache_key(config::SHTnsConfigType)
    return SolverModeIndexCacheKey(
        config.lmax,
        config.mmax,
        config.nlm,
        hash(config.l_values),
        hash(config.m_values),
    )
end

const SOLVER_MODE_INDEX_CACHE = Dict{SolverModeIndexCacheKey, Dict{Tuple{Int,Int}, Int}}()
const SOLVER_MODE_INDEX_CACHE_LOCK = ReentrantLock()

@inline local_range(pencil, dim::Int) = pencil.axes_local[dim]
@inline function mpi_comm()
    solver_backend_ensure_mpi!()
    return SOLVER_BACKEND_MPI.COMM_WORLD
end
@inline mpi_rank(comm=mpi_comm()) = SolverMPI.Comm_rank(comm)
@inline mpi_initialized() = SolverMPI.Initialized()
@inline mpi_comm_size(comm=mpi_comm()) = SolverMPI.Comm_size(comm)
@inline mpi_wtime() = SolverMPI.Wtime()
@inline timing_enabled() = SOLVER_ENABLE_TIMING[]
@inline solver_buffer_cache_lock() = SOLVER_SHARED_BUFFER_CACHE_LOCK

@inline function solver_set_timing_enabled!(enabled::Bool)
    SOLVER_ENABLE_TIMING[] = enabled
    return nothing
end

@inline function solver_set_erk2_diagnostics!(enabled::Bool, interval::Integer)
    SOLVER_ERK2_DIAGNOSTICS_ENABLED[] = enabled
    SOLVER_ERK2_DIAGNOSTICS_INTERVAL[] = Int(interval)
    return nothing
end

function build_mode_index_table(config)
    table = Dict{Tuple{Int,Int}, Int}()
    sizehint!(table, config.nlm)
    @inbounds for idx in 1:config.nlm
        table[(config.l_values[idx], config.m_values[idx])] = idx
    end
    return table
end

function mode_index(config, l::Int, m::Int)
    key = solver_mode_index_cache_key(config)
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

macro solver_local_spectral_modes(lm_var, slot_var, lm_range, config, limit_data, storage_data, body)
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

macro solver_threaded_local_spectral_modes(lm_var, slot_var, lm_range, config, limit_data, storage_data, body)
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

function get_bc_vectors(field)
    cache = field.boundary_interpolation_cache
    if !get(cache, "bc_loaded", false)
        return (
            inner_real=nothing,
            outer_real=nothing,
            inner_imag=nothing,
            outer_imag=nothing,
        )
    end

    bc_real = cache["bc_real"]::Matrix
    bc_imag = cache["bc_imag"]::Matrix

    return (
        inner_real=view(bc_real, 1, :),
        outer_real=view(bc_real, 2, :),
        inner_imag=view(bc_imag, 1, :),
        outer_imag=view(bc_imag, 2, :),
    )
end

@inline function mpi_barrier!(comm=mpi_comm())
    MPI.Barrier(comm)
    return nothing
end

@inline function allreduce_sum_in_place!(buffer, comm=mpi_comm())
    MPI.Allreduce!(buffer, MPI.SUM, comm)
    return buffer
end

@inline function allreduce_sum_buffers!(sendbuf, recvbuf, comm=mpi_comm())
    sendbuf === recvbuf || copyto!(recvbuf, sendbuf)
    MPI.Allreduce!(recvbuf, MPI.SUM, comm)
    return recvbuf
end

@inline function allreduce_sum(value, comm=mpi_comm())
    return MPI.Allreduce(value, +, comm)
end

@inline function allreduce_max(value, comm=mpi_comm())
    return MPI.Allreduce(value, MPI.MAX, comm)
end

@inline function solver_domain_bandwidth(domain::RadialDomainType)
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
    domain::RadialDomainType,
) where {T}
    N = domain.N
    bandwidth = solver_domain_bandwidth(domain)
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

@inline build_radial_derivative_matrix(order::Int, domain::RadialDomainType) =
    build_radial_derivative_matrix(eltype(domain.r), order, domain)

function solver_extract_dense_row(data::AbstractMatrix{T}, bandwidth::Int, nr::Int, row::Int) where T
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

@inline build_radial_laplacian(domain::RadialDomainType) =
    build_radial_laplacian(eltype(domain.r), domain)

function krylov_exp_action(
    Aop!,
    v::Vector{T},
    dt::Float64;
    m::Int=20,
    tol::Float64=1e-8,
) where T
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
            @. w = w - H[i, j] * V[:, i]
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

@inline function get_sht_plan(buffers::SHTnsBuffers)
    return buffers.sht_plan::Union{SHTnsKit.SHTPlan, Nothing}
end

@inline function get_synth_out(buffers::SHTnsBuffers)
    return buffers.synth_out::Union{Matrix{Float64}, Nothing}
end

@inline function get_anal_out(buffers::SHTnsBuffers)
    return buffers.anal_out::Union{Matrix{ComplexF64}, Nothing}
end

@inline function get_vt_out(buffers::SHTnsBuffers, ::Type{T}) where {T<:AbstractFloat}
    return buffers.vt_out::Union{Matrix{Float64}, Nothing}
end

@inline function get_vp_out(buffers::SHTnsBuffers, ::Type{T}) where {T<:AbstractFloat}
    return buffers.vp_out::Union{Matrix{Float64}, Nothing}
end

@inline function get_slm_out(buffers::SHTnsBuffers, ::Type{T}) where {T<:AbstractFloat}
    return buffers.slm_out::Union{Matrix{ComplexF64}, Nothing}
end

@inline function get_tlm_out(buffers::SHTnsBuffers, ::Type{T}) where {T<:AbstractFloat}
    return buffers.tlm_out::Union{Matrix{ComplexF64}, Nothing}
end

@inline function solver_transform_arch(config)
    device = config._buffers.transform_device
    device isa AbstractArchitecture && return device
    return SHTnsKit.is_gpu_config(config.sht_config) ? GPU(device) : CPU()
end

@inline uses_gpu(config) = !(solver_transform_arch(config) isa CPU)

@inline function sht_synthesis!(plan, synth_out, coeffs_matrix)
    SHTnsKit.synthesis!(plan, synth_out, coeffs_matrix; real_output=true)
    return synth_out
end

@inline function sht_synthesis(config, coeffs_matrix)
    if uses_gpu(config)
        if solver_gpu_device() === :cuda
            return SHTnsKit.gpu_synthesis_safe(
                config.sht_config,
                coeffs_matrix;
                device=SHTnsKit.CUDA_DEVICE,
                real_output=true,
            )
        end
        return solver_gpu_scalar_synthesis(config.sht_config, coeffs_matrix; real_output=true)
    end
    return SHTnsKit.synthesis(config.sht_config, coeffs_matrix; real_output=true)
end

@inline function sht_analysis!(plan, anal_out, phys_slice)
    SHTnsKit.analysis!(plan, anal_out, phys_slice)
    return anal_out
end

@inline function sht_analysis(config, phys_slice)
    if uses_gpu(config)
        if solver_gpu_device() === :cuda
            return SHTnsKit.gpu_analysis_safe(
                config.sht_config,
                phys_slice;
                device=SHTnsKit.CUDA_DEVICE,
                real_output=true,
            )
        end
        return solver_gpu_scalar_analysis(config.sht_config, phys_slice; real_output=true)
    end
    return SHTnsKit.analysis(config.sht_config, phys_slice)
end

@inline function sht_vector_synthesis!(
    plan,
    vt_out,
    vp_out,
    pol_coeffs,
    tor_coeffs,
)
    SHTnsKit.synthesis_sphtor!(plan, vt_out, vp_out, pol_coeffs, tor_coeffs; real_output=true)
    return vt_out, vp_out
end

@inline function sht_vector_synthesis(config, pol_coeffs, tor_coeffs)
    if uses_gpu(config)
        if solver_gpu_device() === :cuda
            return SHTnsKit.gpu_synthesis_sphtor(
                config.sht_config,
                pol_coeffs,
                tor_coeffs;
                device=SHTnsKit.CUDA_DEVICE,
                real_output=true,
            )
        end
        return solver_gpu_vector_synthesis(
            config.sht_config,
            pol_coeffs,
            tor_coeffs;
            real_output=true,
        )
    end
    return SHTnsKit.synthesis_sphtor(config.sht_config, pol_coeffs, tor_coeffs; real_output=true)
end

@inline function sht_vector_analysis!(
    plan,
    slm_out,
    tlm_out,
    vt_field,
    vp_field,
)
    SHTnsKit.analysis_sphtor!(plan, slm_out, tlm_out, vt_field, vp_field)
    return slm_out, tlm_out
end

@inline function sht_vector_analysis(config, vt_field, vp_field)
    if uses_gpu(config)
        if solver_gpu_device() === :cuda
            return SHTnsKit.gpu_analysis_sphtor(
                config.sht_config,
                vt_field,
                vp_field;
                device=SHTnsKit.CUDA_DEVICE,
            )
        end
        return solver_gpu_vector_analysis(config.sht_config, vt_field, vp_field)
    end
    return SHTnsKit.analysis_sphtor(config.sht_config, vt_field, vp_field)
end

function collect_vector_coefficients(
    spec1_real,
    spec1_imag,
    spec2_real,
    spec2_imag,
    r_local,
    config,
)
    lmax, mmax = config.lmax, config.mmax

    coeffs_buffer1 = solver_get_cached_buffer!(config, :solver_vector_coeffs_buffer_1) do
        solver_workspace_zeros(config, ComplexF64, lmax + 1, mmax + 1)
    end::Matrix{ComplexF64}
    coeffs_buffer2 = solver_get_cached_buffer!(config, :solver_vector_coeffs_buffer_2) do
        solver_workspace_zeros(config, ComplexF64, lmax + 1, mmax + 1)
    end::Matrix{ComplexF64}

    fill_vector_coeff_buffer!(coeffs_buffer1, spec1_real, spec1_imag, r_local, config)
    fill_vector_coeff_buffer!(coeffs_buffer2, spec2_real, spec2_imag, r_local, config)

    coeffs_gathered1 = solver_get_cached_buffer!(config, :solver_vector_coeffs_gathered_1) do
        solver_workspace_zeros(config, ComplexF64, lmax + 1, mmax + 1)
    end::Matrix{ComplexF64}
    coeffs_gathered2 = solver_get_cached_buffer!(config, :solver_vector_coeffs_gathered_2) do
        solver_workspace_zeros(config, ComplexF64, lmax + 1, mmax + 1)
    end::Matrix{ComplexF64}

    allreduce_sum!(coeffs_buffer1, coeffs_gathered1)
    allreduce_sum!(coeffs_buffer2, coeffs_gathered2)

    return coeffs_gathered1, coeffs_gathered2
end

function fill_vector_coeff_buffer!(coeffs_buffer, spec_real, spec_imag, r_local, config)
    if uses_gpu(config) && solver_gpu_device() !== :cuda
        return solver_gpu_fill_vector_coeff_buffer(
            coeffs_buffer,
            spec_real,
            spec_imag,
            r_local,
            config,
        )
    end
    return cpu_fill_vector_coeff_buffer!(
        coeffs_buffer,
        spec_real,
        spec_imag,
        r_local,
        config,
    )
end

@inline function cpu_fill_vector_coeff_buffer!(coeffs_buffer, spec_real, spec_imag, r_local, config)
    return solver_cpu_fill_scalar_coeff_buffer!(coeffs_buffer, spec_real, spec_imag, r_local, config)
end

function store_vector_coefficients!(spec_real, spec_imag, coeffs_matrix, r_local, config)
    if uses_gpu(config) && solver_gpu_device() !== :cuda
        return solver_gpu_store_vector_coefficients(
            spec_real,
            spec_imag,
            coeffs_matrix,
            r_local,
            config,
        )
    end
    return cpu_store_vector_coefficients!(
        spec_real,
        spec_imag,
        coeffs_matrix,
        r_local,
        config,
    )
end

function cpu_store_vector_coefficients!(spec_real, spec_imag, coeffs_matrix, r_local, config)
    matrix_lmax = size(coeffs_matrix, 1) - 1
    matrix_mmax = size(coeffs_matrix, 2) - 1

    local_modes = local_spectral_mode_indices(config)

    Threads.@threads for lm_idx in local_modes
        slot = local_spectral_storage_slot(config, lm_idx)
        slot === nothing && continue
        l = config.l_values[lm_idx]
        m = config.m_values[lm_idx]
        if r_local <= size(spec_real, 3) && r_local <= size(spec_imag, 3)
            if l <= matrix_lmax && m <= matrix_mmax
                coeff = coeffs_matrix[l + 1, m + 1]
                set_local_spectral_value!(spec_real, slot, r_local, real(coeff))
                set_local_spectral_value!(spec_imag, slot, r_local, m == 0 ? 0.0 : imag(coeff))
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
    axes_local::Union{Nothing,Tuple}=nothing,
) where T
    if uses_gpu(config) && solver_gpu_device() !== :cuda
        return solver_gpu_extract_vector_component(
            component_buffer,
            v_data,
            r_local,
            config;
            axes_local=axes_local,
        )
    end
    return cpu_extract_vector_component!(
        component_buffer,
        v_data,
        r_local,
        config;
        axes_local=axes_local,
    )
end

function cpu_extract_vector_component!(
    component_buffer::Matrix{T},
    v_data,
    r_local,
    config;
    axes_local::Union{Nothing,Tuple}=nothing,
) where T
    nlat, nlon = config.nlat, config.nlon
    fill!(component_buffer, zero(T))
    has_local_data = r_local <= size(v_data, 3)

    if axes_local !== nothing
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
    axes_local::Union{Nothing,Tuple}=nothing,
)
    if uses_gpu(config) && solver_gpu_device() !== :cuda
        return solver_gpu_store_vector_components(
            v_theta,
            v_phi,
            vt_field,
            vp_field,
            r_local,
            config;
            axes_local=axes_local,
        )
    end
    return cpu_store_vector_components!(
        v_theta,
        v_phi,
        vt_field,
        vp_field,
        r_local,
        config;
        axes_local=axes_local,
    )
end

function cpu_store_vector_components!(
    v_theta,
    v_phi,
    vt_field,
    vp_field,
    r_local,
    config;
    axes_local::Union{Nothing,Tuple}=nothing,
)
    if r_local > size(v_theta, 3) || r_local > size(v_phi, 3)
        return v_theta, v_phi
    end

    if axes_local !== nothing
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
        common_i_range = 1:min(size(v_theta, 1), size(v_phi, 1), size(vt_field, 1), size(vp_field, 1))
        common_j_range = 1:min(size(v_theta, 2), size(v_phi, 2), size(vt_field, 2), size(vp_field, 2))
        Threads.@threads for i in common_i_range
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
    axes_local::Union{Nothing,Tuple}=nothing,
)
    if r_local > size(v_component, 3)
        return v_component
    end

    if axes_local !== nothing
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
        common_i_range = 1:min(size(v_component, 1), size(field, 1))
        common_j_range = 1:min(size(v_component, 2), size(field, 2))
        Threads.@threads for i in common_i_range
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

    Threads.@threads for i in axes(v_component, 1)
        for j in axes(v_component, 2)
            v_component[i, j, r_local] = zero(eltype(v_component))
        end
    end

    return v_component
end

function vector_spectral_to_physical!(
    toroidal::SpectralFieldType{T},
    poloidal::SpectralFieldType{T},
    vector_field::VectorFieldType{T};
    domain::Union{RadialDomainType,Nothing}=nothing,
) where T
    config = toroidal.config

    tor_real = parent(toroidal.data_real)
    tor_imag = parent(toroidal.data_imag)
    pol_real = parent(poloidal.data_real)
    pol_imag = parent(poloidal.data_imag)

    v_r = parent(vector_field.r_component.data)
    v_theta = parent(vector_field.θ_component.data)
    v_phi = parent(vector_field.φ_component.data)

    r_range = local_range(poloidal.pencil, 3)
    plan = get_sht_plan(config._buffers)
    vt_out = get_vt_out(config._buffers, T)
    vp_out = get_vp_out(config._buffers, T)
    synth_out = get_synth_out(config._buffers)

    @assert size(v_r, 3) == size(tor_real, 3) "Radial dimension mismatch: physical=$(size(v_r,3)) vs spectral=$(size(tor_real,3)). Vector SH transforms require radial to be local."

    phys_axes_local = vector_field.r_component.pencil.axes_local

    for r_local in axes(tor_real, 3)
        tor_coeffs, pol_coeffs = collect_vector_coefficients(
            tor_real,
            tor_imag,
            pol_real,
            pol_imag,
            r_local,
            config,
        )

        if plan !== nothing && vt_out !== nothing && vp_out !== nothing
            sht_vector_synthesis!(plan, vt_out, vp_out, pol_coeffs, tor_coeffs)
            store_vector_components!(
                v_theta,
                v_phi,
                vt_out,
                vp_out,
                r_local,
                config;
                axes_local=phys_axes_local,
            )
        else
            vt_field, vp_field = sht_vector_synthesis(config, pol_coeffs, tor_coeffs)
            store_vector_components!(
                v_theta,
                v_phi,
                vt_field,
                vp_field,
                r_local,
                config;
                axes_local=phys_axes_local,
            )
        end

        if domain !== nothing
            r_idx_global = r_local + first(r_range) - 1
            if 1 <= r_idx_global <= domain.N
                r_val = domain.r[r_idx_global, 4]
                if r_val > eps(Float64) * domain.r[domain.N, 4]
                    lmax, mmax = config.lmax, config.mmax
                    pol_rad_coeffs = solver_get_cached_buffer!(config, :solver_pol_rad_coeffs_buffer) do
                        solver_workspace_zeros(config, ComplexF64, lmax + 1, mmax + 1)
                    end::Matrix{ComplexF64}
                    fill!(pol_rad_coeffs, zero(ComplexF64))

                    for l in 0:lmax
                        l_factor = l * (l + 1) / r_val
                        for m in 0:min(l, mmax)
                            pol_rad_coeffs[l + 1, m + 1] = pol_coeffs[l + 1, m + 1] * l_factor
                        end
                    end

                    if plan !== nothing && synth_out !== nothing
                        sht_synthesis!(plan, synth_out, pol_rad_coeffs)
                        store_scalar_component!(
                            v_r,
                            synth_out,
                            r_local,
                            config;
                            axes_local=phys_axes_local,
                        )
                    else
                        vr_field = sht_synthesis(config, pol_rad_coeffs)
                        store_scalar_component!(
                            v_r,
                            vr_field,
                            r_local,
                            config;
                            axes_local=phys_axes_local,
                        )
                    end
                else
                    store_zero_component!(v_r, r_local, config)
                end
            end
        else
            store_zero_component!(v_r, r_local, config)
        end
    end

    return vector_field
end

function vector_physical_to_spectral!(
    vector_field::VectorFieldType{T},
    toroidal::SpectralFieldType{T},
    poloidal::SpectralFieldType{T};
    domain::Union{RadialDomainType,Nothing}=nothing,
    verify_solenoidal::Bool=false,
) where T
    config = toroidal.config

    v_theta = parent(vector_field.θ_component.data)
    v_phi = parent(vector_field.φ_component.data)
    tor_real = parent(toroidal.data_real)
    tor_imag = parent(toroidal.data_imag)
    pol_real = parent(poloidal.data_real)
    pol_imag = parent(poloidal.data_imag)

    plan = get_sht_plan(config._buffers)
    slm_out = get_slm_out(config._buffers, T)
    tlm_out = get_tlm_out(config._buffers, T)
    phys_axes_local = vector_field.r_component.pencil.axes_local

    for r_local in axes(v_theta, 3)
        nlat, nlon = config.nlat, config.nlon
        vt_buffer = solver_get_cached_buffer!(config, :solver_vector_component_buffer_vt) do
            solver_workspace_zeros(config, eltype(v_theta), nlat, nlon)
        end::Matrix{Float64}
        vp_buffer = solver_get_cached_buffer!(config, :solver_vector_component_buffer_vp) do
            solver_workspace_zeros(config, eltype(v_phi), nlat, nlon)
        end::Matrix{Float64}

        vt_field = extract_vector_component!(
            vt_buffer,
            v_theta,
            r_local,
            config;
            axes_local=phys_axes_local,
        )
        vp_field = extract_vector_component!(
            vp_buffer,
            v_phi,
            r_local,
            config;
            axes_local=phys_axes_local,
        )

        if plan !== nothing && slm_out !== nothing && tlm_out !== nothing
            sht_vector_analysis!(plan, slm_out, tlm_out, vt_field, vp_field)
            store_vector_coefficients!(pol_real, pol_imag, slm_out, r_local, config)
            store_vector_coefficients!(tor_real, tor_imag, tlm_out, r_local, config)
        else
            pol_coeffs, tor_coeffs = sht_vector_analysis(config, vt_field, vp_field)
            store_vector_coefficients!(pol_real, pol_imag, pol_coeffs, r_local, config)
            store_vector_coefficients!(tor_real, tor_imag, tor_coeffs, r_local, config)
        end
    end

    return toroidal, poloidal
end

@inline function solver_reset_velocity_work_arrays!(velocity_fields)
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

function solver_refresh_velocity_physical_fields!(velocity_fields, domain)
    vector_spectral_to_physical!(
        velocity_fields.𝒯,
        velocity_fields.𝒫,
        velocity_fields.velocity;
        domain,
    )
    return velocity_fields.velocity
end

function solver_refresh_vorticity_physical_fields!(velocity_fields, domain)
    solver_compute_vorticity_spectral!(velocity_fields, domain)
    vector_spectral_to_physical!(
        velocity_fields.ζᵀ,
        velocity_fields.ζᴾ,
        velocity_fields.vorticity;
        domain,
    )
    return velocity_fields.vorticity
end

function solver_extract_local_radial_profile!(
    profile::Vector{T},
    data::AbstractArray{T,3},
    slot::CartesianIndex{2},
    nr::Int,
    r_range,
) where T
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
    input::Vector{T},
) where T
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

function solver_compute_vorticity_spectral!(
    velocity_fields::VelocityFieldsType{T},
    domain,
) where T
    u_tor_real = parent(velocity_fields.𝒯.data_real)
    u_tor_imag = parent(velocity_fields.𝒯.data_imag)
    u_pol_real = parent(velocity_fields.𝒫.data_real)
    u_pol_imag = parent(velocity_fields.𝒫.data_imag)

    ζ_tor_real = parent(velocity_fields.ζᵀ.data_real)
    ζ_tor_imag = parent(velocity_fields.ζᵀ.data_imag)
    ζ_pol_real = parent(velocity_fields.ζᴾ.data_real)
    ζ_pol_imag = parent(velocity_fields.ζᴾ.data_imag)

    config = velocity_fields.𝒯.config
    lm_range = local_spectral_mode_indices(config)
    r_range = local_range(config.pencils.spec, 3)
    nr = domain.N

    nthreads = max(1, Threads.nthreads(), Threads.maxthreadid())
    workspace = get_velocity_workspace(T)
    if workspace === nothing ||
       length(workspace.Pᴾ_profile_real) < nthreads ||
       length(workspace.Pᴾ_profile_real[1]) != nr
        workspace = create_velocity_workspace(T, nr, nthreads)
        set_velocity_workspace!(workspace)
    end

    pol_profile_real_bufs = workspace.Pᴾ_profile_real
    pol_profile_imag_bufs = workspace.Pᴾ_profile_imag
    tor_profile_real_bufs = workspace.Tᵀ_profile_real
    tor_profile_imag_bufs = workspace.Tᵀ_profile_imag
    dpol_dr_real_bufs = workspace.∂ᵣ𝒫_real
    dpol_dr_imag_bufs = workspace.∂ᵣ𝒫_imag
    d2pol_dr2_real_bufs = workspace.∂ᵣᵣ𝒫_real
    d2pol_dr2_imag_bufs = workspace.∂ᵣᵣ𝒫_imag

    @solver_threaded_local_spectral_modes lm_idx slot lm_range config velocity_fields.ℓ_factors u_pol_real begin
        tid = Threads.threadid()
        ℓ_factor = velocity_fields.ℓ_factors[lm_idx]
        pol_profile_real = pol_profile_real_bufs[tid]
        pol_profile_imag = pol_profile_imag_bufs[tid]
        tor_profile_real = tor_profile_real_bufs[tid]
        tor_profile_imag = tor_profile_imag_bufs[tid]
        dpol_dr_real = dpol_dr_real_bufs[tid]
        dpol_dr_imag = dpol_dr_imag_bufs[tid]
        d2pol_dr2_real = d2pol_dr2_real_bufs[tid]
        d2pol_dr2_imag = d2pol_dr2_imag_bufs[tid]

        solver_extract_local_radial_profile!(pol_profile_real, u_pol_real, slot, nr, r_range)
        solver_extract_local_radial_profile!(pol_profile_imag, u_pol_imag, slot, nr, r_range)
        solver_extract_local_radial_profile!(tor_profile_real, u_tor_real, slot, nr, r_range)
        solver_extract_local_radial_profile!(tor_profile_imag, u_tor_imag, slot, nr, r_range)

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
                    set_local_spectral_value!(ζ_tor_real, slot, local_r, (
                        ℓ_factor * r_inv2 * pol_profile_real[r_idx]
                        - d2pol_dr2_real[r_idx]
                        - 2.0 * r_inv * dpol_dr_real[r_idx]
                    ))
                    set_local_spectral_value!(ζ_tor_imag, slot, local_r, (
                        ℓ_factor * r_inv2 * pol_profile_imag[r_idx]
                        - d2pol_dr2_imag[r_idx]
                        - 2.0 * r_inv * dpol_dr_imag[r_idx]
                    ))
                    set_local_spectral_value!(ζ_pol_real, slot, local_r,
                                              -ℓ_factor * r_inv2 * tor_profile_real[r_idx])
                    set_local_spectral_value!(ζ_pol_imag, slot, local_r,
                                              -ℓ_factor * r_inv2 * tor_profile_imag[r_idx])
                end
            end
        end
    end

    return velocity_fields
end

function solver_compute_velocity_body_forces!(
    velocity_fields::VelocityFieldsType{T},
    temperature_field,
    composition_field,
    magnetic_field,
    domain,
    params::SolverParameters,
) where T
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
                linear_idx = i + (j - 1) * local_size[1] + (k - 1) * local_size[1] * local_size[2]

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
        solver_add_compositional_buoyancy_force!(adv_r, composition_field, (Pm / Sc) * RaC, domain)
    end

    if magnetic_field !== nothing
        solver_add_lorentz_force!(velocity_fields, magnetic_field, Pm)
    end

    return velocity_fields
end

function solver_scalar_field_data_and_config(field)
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
    force_r::AbstractArray{T,3},
    scalar_field,
    factor::Float64,
    domain,
) where T
    iszero(factor) && return force_r
    scalar_data, config = solver_scalar_field_data_and_config(scalar_field)
    r_range = local_range(config.pencils.r, 3)
    local_size = size(force_r)

    @inbounds Threads.@threads for k in 1:local_size[3]
        r_idx = k + first(r_range) - 1
        r = r_idx <= domain.N ? domain.r[r_idx, 4] : 1.0
        factor_r = factor * r
        for j in 1:local_size[2]
            @simd for i in 1:local_size[1]
                linear_idx = i + (j - 1) * local_size[1] + (k - 1) * local_size[1] * local_size[2]
                if linear_idx <= length(scalar_data)
                    force_r[linear_idx] += factor_r * scalar_data[linear_idx]
                end
            end
        end
    end

    return force_r
end

function solver_add_compositional_buoyancy_force!(
    force_r::AbstractArray{T,3},
    composition_field,
    factor::Float64,
    domain,
) where T
    iszero(factor) && return force_r
    composition_data, config = solver_scalar_field_data_and_config(composition_field)
    r_range = local_range(config.pencils.r, 3)
    local_size = size(force_r)

    @inbounds Threads.@threads for k in 1:local_size[3]
        r_idx = k + first(r_range) - 1
        r = r_idx <= domain.N ? domain.r[r_idx, 4] : 1.0
        factor_r = factor * r
        for j in 1:local_size[2]
            @simd for i in 1:local_size[1]
                linear_idx = i + (j - 1) * local_size[1] + (k - 1) * local_size[1] * local_size[2]
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

@inline function solver_reset_magnetic_work_arrays!(magnetic_fields)
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

function solver_refresh_magnetic_physical_fields!(magnetic_fields, outer_domain)
    vector_spectral_to_physical!(
        magnetic_fields.𝒯,
        magnetic_fields.𝒫,
        magnetic_fields.magnetic;
        domain=outer_domain,
    )
    return magnetic_fields.magnetic
end

function solver_refresh_current_physical_fields!(magnetic_fields, outer_domain)
    solver_compute_current_density_spectral!(magnetic_fields, outer_domain)
    vector_spectral_to_physical!(
        magnetic_fields.work_tor,
        magnetic_fields.work_pol,
        magnetic_fields.current;
        domain=outer_domain,
    )
    return magnetic_fields.current
end

function solver_spectral_curl_torpol!(
    dst_tor_r,
    dst_tor_i,
    dst_pol_r,
    dst_pol_i,
    src_tor_r,
    src_tor_i,
    src_pol_r,
    src_pol_i,
    ℓ_factors,
    d1_matrix,
    d²_matrix,
    domain::RadialDomainType,
    config::SHTnsConfigType,
    ::Type{T};
    _work::Union{Nothing,NTuple{6,Vector{T}}}=nothing,
) where T
    lm_range = local_spectral_mode_indices(config)
    r_range = local_range(config.pencils.spec, 3)
    nr = domain.N

    if _work !== nothing
        Pᴾ_profile_real, Pᴾ_profile_imag, dᴾ_dr_real, dᴾ_dr_imag, d²ᴾ_dr²_real, d²ᴾ_dr²_imag = _work
    else
        Pᴾ_profile_real = zeros(T, nr)
        Pᴾ_profile_imag = zeros(T, nr)
        dᴾ_dr_real = zeros(T, nr)
        dᴾ_dr_imag = zeros(T, nr)
        d²ᴾ_dr²_real = zeros(T, nr)
        d²ᴾ_dr²_imag = zeros(T, nr)
    end

    @solver_local_spectral_modes lm_idx slot lm_range config ℓ_factors src_pol_r begin
        ℓ_factor = ℓ_factors[lm_idx]

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
                    set_local_spectral_value!(dst_tor_r, slot, local_r, (
                        ℓ_factor * r⁻² * Pᴾ_profile_real[r_idx] -
                        d²ᴾ_dr²_real[r_idx] -
                        2.0 * r⁻¹ * dᴾ_dr_real[r_idx]
                    ))
                    set_local_spectral_value!(dst_tor_i, slot, local_r, (
                        ℓ_factor * r⁻² * Pᴾ_profile_imag[r_idx] -
                        d²ᴾ_dr²_imag[r_idx] -
                        2.0 * r⁻¹ * dᴾ_dr_imag[r_idx]
                    ))
                    set_local_spectral_value!(dst_pol_r, slot, local_r,
                                              -ℓ_factor * r⁻² * local_spectral_value(src_tor_r, slot, local_r))
                    set_local_spectral_value!(dst_pol_i, slot, local_r,
                                              -ℓ_factor * r⁻² * local_spectral_value(src_tor_i, slot, local_r))
                end
            end
        end
    end

    return dst_tor_r, dst_tor_i, dst_pol_r, dst_pol_i
end

function solver_compute_current_density_spectral!(magnetic_fields, outer_domain)
    T = eltype(parent(magnetic_fields.work_tor.data_real))
    solver_spectral_curl_torpol!(
        parent(magnetic_fields.work_tor.data_real), parent(magnetic_fields.work_tor.data_imag),
        parent(magnetic_fields.work_pol.data_real), parent(magnetic_fields.work_pol.data_imag),
        parent(magnetic_fields.𝒯.data_real), parent(magnetic_fields.𝒯.data_imag),
        parent(magnetic_fields.𝒫.data_real), parent(magnetic_fields.𝒫.data_imag),
        magnetic_fields.ℓ_factors,
        magnetic_fields.∂r,
        magnetic_fields.∂²r,
        outer_domain,
        magnetic_fields.𝒯.config,
        T,
    )
    return magnetic_fields
end

function solver_apply_induction_nonlinear!(
    magnetic_fields,
    velocity_fields;
    geometry::Symbol,
)
    solver_compute_velocity_cross_magnetic!(magnetic_fields, velocity_fields)
    if geometry === :ball
        solver_ball_vector_analysis!(
            magnetic_fields.induction_physical,
            magnetic_fields.work_tor,
            magnetic_fields.work_pol,
        )
    else
        vector_physical_to_spectral!(
            magnetic_fields.induction_physical,
            magnetic_fields.work_tor,
            magnetic_fields.work_pol,
        )
    end
    solver_compute_curl_of_induction!(magnetic_fields)
    return magnetic_fields
end

function solver_apply_inner_core_rotation!(magnetic_fields, rotation_rate)
    ic_tor_real = parent(magnetic_fields.𝒯ⁱᶜ.data_real)
    ic_tor_imag = parent(magnetic_fields.𝒯ⁱᶜ.data_imag)
    ic_pol_real = parent(magnetic_fields.𝒫ⁱᶜ.data_real)
    ic_pol_imag = parent(magnetic_fields.𝒫ⁱᶜ.data_imag)

    nl_tor_real = parent(magnetic_fields.nlᵀ.data_real)
    nl_tor_imag = parent(magnetic_fields.nlᵀ.data_imag)
    nl_pol_real = parent(magnetic_fields.nlᴾ.data_real)
    nl_pol_imag = parent(magnetic_fields.nlᴾ.data_imag)

    lm_range = local_range(magnetic_fields.𝒯ⁱᶜ.pencil, 1)
    r_range = local_range(magnetic_fields.𝒯ⁱᶜ.pencil, 3)
    rotation_factor = rotation_rate

    @inbounds for lm_idx in lm_range
        if lm_idx <= magnetic_fields.𝒯ⁱᶜ.nlm
            slot = local_spectral_storage_slot(magnetic_fields.𝒯.config, lm_idx)
            slot === nothing && continue
            m = magnetic_fields.𝒯.config.m_values[lm_idx]
            if m != 0 && 1 in r_range
                local_r = 1 - first(r_range) + 1
                if local_r <= size(nl_tor_real, 3)
                    coupling_factor = rotation_factor * Float64(m)
                    set_local_spectral_value!(
                        nl_tor_real, slot, local_r,
                        local_spectral_value(nl_tor_real, slot, local_r) +
                        coupling_factor * local_spectral_value(ic_pol_imag, slot, local_r),
                    )
                    set_local_spectral_value!(
                        nl_tor_imag, slot, local_r,
                        local_spectral_value(nl_tor_imag, slot, local_r) -
                        coupling_factor * local_spectral_value(ic_pol_real, slot, local_r),
                    )
                    set_local_spectral_value!(
                        nl_pol_real, slot, local_r,
                        local_spectral_value(nl_pol_real, slot, local_r) -
                        coupling_factor * local_spectral_value(ic_tor_imag, slot, local_r),
                    )
                    set_local_spectral_value!(
                        nl_pol_imag, slot, local_r,
                        local_spectral_value(nl_pol_imag, slot, local_r) +
                        coupling_factor * local_spectral_value(ic_tor_real, slot, local_r),
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
    T = eltype(parent(magnetic_fields.nlᵀ.data_real))
    solver_spectral_curl_torpol!(
        parent(magnetic_fields.nlᵀ.data_real), parent(magnetic_fields.nlᵀ.data_imag),
        parent(magnetic_fields.nlᴾ.data_real), parent(magnetic_fields.nlᴾ.data_imag),
        parent(magnetic_fields.work_tor.data_real), parent(magnetic_fields.work_tor.data_imag),
        parent(magnetic_fields.work_pol.data_real), parent(magnetic_fields.work_pol.data_imag),
        magnetic_fields.ℓ_factors,
        magnetic_fields.∂r,
        magnetic_fields.∂²r,
        magnetic_fields.outer_domain,
        magnetic_fields.𝒯.config,
        T,
    )
    return magnetic_fields
end

function solver_enforce_ball_vector_regularity!(
    tor_spec::SpectralFieldType,
    pol_spec::SpectralFieldType,
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

function solver_ball_vector_physical_to_spectral!(vector_field, toroidal, poloidal)
    vector_physical_to_spectral!(vector_field, toroidal, poloidal)
    solver_enforce_ball_vector_regularity!(toroidal, poloidal)
    return toroidal, poloidal
end

function solver_ball_vector_analysis!(vector_field, toroidal, poloidal)
    return solver_ball_vector_physical_to_spectral!(vector_field, toroidal, poloidal)
end

@inline solver_band_row(i::Int, j::Int, bw::Int) = bw + 1 + i - j

function solver_banded_to_dense(
    matrix::Union{OldBandedMatrix{T}, BandedOperator{T}},
) where T
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

function solver_factorize_banded(A::BandedOperator{T}) where T
    n = A.size
    bw = A.bandwidth
    lu = copy(A.data)

    @inbounds for k in 1:n-1
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
        for i in k+1:i_max
            row = solver_band_row(i, k, bw)
            if 1 <= row <= 2 * bw + 1
                L = lu[row, k] / piv
                lu[row, k] = L
                j_max = min(n, k + bw)
                for j in k+1:j_max
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
    b::Vector{T},
) where T
    n = lu.size
    bw = lu.bandwidth

    @inbounds for i in 1:n
        s = zero(T)
        j_min = max(1, i - bw)
        for j in j_min:i-1
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
        for j in i+1:j_max
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
        v::Vector{T},
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
    l::Int,
) where T
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
    m::Int=20,
    tol::Float64=1e-8,
) where T
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

        if !all(isfinite.(x))
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

@eval GeoDynamo begin
    """
        exp_action_krylov(Aop!, v, dt; m=20, tol=1e-8)

    Compute `exp(dt*A)v` with an Arnoldi/Krylov approximation.

    The high-level timestep drivers call `exp_action_krylov` directly,
    but this root-level entry point remains available as a reusable numerical
    utility.
    """
    function exp_action_krylov(Aop!, v, dt; m::Int=20, tol::Float64=1e-8)
        return krylov_exp_action(Aop!, v, dt; m, tol)
    end
end

solver_synchronize_pencil_transforms!(field::SpectralFieldType{T}) where {T} = begin
    mpi_barrier!()
    field
end

function solver_rcond_estimate(lu_A, A::Matrix{T}) where T
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

function solver_phi1_series(A::Matrix{T}) where T
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

function solver_phi2_series(A::Matrix{T}) where T
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

function solver_compute_phi1_function(A::Matrix{T}, expA::Matrix{T}) where T
    nr = size(A, 1)
    I_mat = Matrix{T}(LA.I, nr, nr)

    if !all(isfinite, A) || !all(isfinite, expA)
        throw(ArgumentError("Non-finite values detected in solver φ1 computation"))
    end

    if LA.opnorm(A) < sqrt(eps(T))
        return solver_phi1_series(A)
    end

    diff = expA - I_mat

    try
        lu_A = LA.lu(A)
        rc = solver_rcond_estimate(lu_A, A)
        if rc < rcond_fallback_tol(T)
            @debug "Ill-conditioned matrix in solver φ1 computation (rcond = $rc), using series expansion"
            return solver_phi1_series(A)
        end

        result = lu_A \ diff
        if !all(isfinite, result)
            @warn "Non-finite result in solver φ1 computation, falling back to series expansion"
            return solver_phi1_series(A)
        end
        return result
    catch e
        @debug "LU factorization failed in solver φ1 computation: $e, using series expansion"
        try
            return solver_phi1_series(A)
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

function report_solver_phi2_conditioning(step::Int; interval::Int=100)
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

function solver_compute_phi2_function(A::Matrix{T}, expA::Matrix{T}; l::Int=0) where T
    nr = size(A, 1)
    I_mat = Matrix{T}(LA.I, nr, nr)

    if !all(isfinite, A) || !all(isfinite, expA)
        SOLVER_PHI2_MONITOR.enable_monitoring && (SOLVER_PHI2_MONITOR.nonfinite_count += 1)
        throw(ArgumentError("Non-finite values detected in solver φ2 computation (l=$l)"))
    end

    if LA.opnorm(A) < sqrt(eps(T))
        return solver_phi2_series(A)
    end

    diff = expA - I_mat - A

    try
        lu_A = LA.lu(A)
        rcond_val = solver_rcond_estimate(lu_A, A)
        if SOLVER_PHI2_MONITOR.enable_monitoring && rcond_val < SOLVER_PHI2_MONITOR.worst_rcond
            SOLVER_PHI2_MONITOR.worst_rcond = rcond_val
            SOLVER_PHI2_MONITOR.worst_l = l
        end

        if rcond_val < rcond_fallback_tol(T)
            if SOLVER_PHI2_MONITOR.enable_monitoring
                SOLVER_PHI2_MONITOR.series_expansion_count += 1
            end
            @debug "Ill-conditioned matrix in solver φ2 computation (l=$l, rcond=$rcond_val), using series expansion"
            return solver_phi2_series(A)
        end

        temp = lu_A \ diff
        result = lu_A \ temp

        if !all(isfinite, result)
            @warn "Non-finite result in solver φ2 computation, falling back to series expansion"
            return solver_phi2_series(A)
        end

        return result
    catch e
        if SOLVER_PHI2_MONITOR.enable_monitoring
            SOLVER_PHI2_MONITOR.lu_failure_count += 1
        end
        @debug "LU factorization failed in solver φ2 computation (l=$l): $e, using series expansion"
        try
            return solver_phi2_series(A)
        catch e2
            @error "Complete failure in solver φ2 computation: $e2, returning zero matrix"
            return zeros(T, nr, nr)
        end
    end
end
