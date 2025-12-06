# ================================================================================
# SHTnsKit Spherical Harmonic Transforms with PencilArrays Integration
# ================================================================================
#
# This module implements spherical harmonic transforms using SHTnsKit.jl
# with MPI parallelization across theta and phi directions using PencilArrays
# and efficient FFTs using PencilFFTs
#

using SHTnsKit
using PencilArrays
using PencilFFTs
using FFTW
using LinearAlgebra
using Base.Threads

@inline function _shtns_make_transpose(pair)
    src = first(pair)
    dest = last(pair)

    # Create temporary arrays for planning (use Float64 as a generic type)
    src_array = PencilArray{Float64}(undef, src)
    dest_array = PencilArray{Float64}(undef, dest)

    # Create the transposition plan
    return PencilArrays.Transpositions.Transposition(dest_array, src_array)
end

# Simple heuristic for number of simultaneously allocated fields (for memory estimate)
estimate_field_count() = 6

# Default grid size getters that work during precompilation
# These provide fallback values when the parameter system isn't loaded
function get_default_nlat()
    # Try to use parameter system if available, otherwise use reasonable default
    try
        params = get_parameters()
        return params.i_Th
    catch
        return 64  # Compatible with most SHTnsKit configurations
    end
end

function get_default_nlon()
    # Try to use parameter system if available, otherwise use reasonable default
    try
        params = get_parameters()
        return params.i_Ph
    catch
        return 128  # Power of 2 for efficient FFTs
    end
end

# ================================================================================
# SHTnsKit Configuration Structure
# ================================================================================

# Forward declaration for fields.jl
abstract type AbstractSHTnsConfig end

struct SHTnsKitConfig <: AbstractSHTnsConfig
    # SHTnsKit configuration
    sht_config::SHTnsKit.SHTConfig

    # Grid parameters
    nlat::Int
    nlon::Int
    lmax::Int
    mmax::Int
    nlm::Int

    # PencilArrays decomposition for parallelization
    pencils::NamedTuple

    # PencilFFTs plans for efficient phi-direction FFTs
    fft_plans::Dict{Symbol, Any}

    # Transpose plans for pencil reorientations
    transpose_plans::Dict{Symbol, Any}

    # Memory estimate
    memory_estimate::String

    # Convenience fields for compatibility with legacy code paths
    l_values::Vector{Int}
    m_values::Vector{Int}
    theta_grid::Vector{Float64}
    phi_grid::Vector{Float64}
    gauss_weights::Vector{Float64}

    # Memory-efficient buffer cache for reuse
    _buffer_cache::Dict{Symbol, Any}
end

"""
    create_shtnskit_config(; lmax::Int, mmax::Int=lmax, nlat::Int=lmax+2,
                           nlon::Int=max(2*lmax+1, 4), optimize_decomp::Bool=true) -> SHTnsKitConfig

Create SHTnsKit configuration with MPI parallelization using PencilArrays.
This creates proper integration with ../SHTnsKit.jl, PencilArrays, and PencilFFTs.
"""
function create_shtnskit_config(; lmax::Int, mmax::Int=lmax,
                               nlat::Int=max(lmax+2, get_default_nlat()),
                               nlon::Int=max(2*lmax+1, 4, get_default_nlon()),
                               nr::Int=i_N,
                               optimize_decomp::Bool=true)

    # Create SHTnsKit configuration using the local ../SHTnsKit.jl
    sht_config = SHTnsKit.create_gauss_config(lmax, nlat;
                                            mmax=mmax,
                                            nlon=nlon,
                                            norm=:orthonormal)

    # Disable precomputed Legendre polynomial tables to avoid version-dependent
    # dimension mismatches between SHTnsKit's table creation and transform code.
    # The on-the-fly Plm computation is reliable and the performance impact is
    # minimal for typical problem sizes.
    SHTnsKit.disable_plm_tables!(sht_config)

    # Get MPI communicator
    comm = get_comm()
    nprocs = get_nprocs()

    # Create pencil decomposition for parallel theta-phi transforms
    pencils = create_pencil_decomposition_shtnskit(nlat, nlon, nr, sht_config, comm, optimize_decomp)

    # Create PencilFFTs plans for efficient phi-direction transforms
    fft_plans = create_pencil_fft_plans(pencils, (nlat, nlon, nr))

    # Create transpose plans between different pencil orientations
    transpose_plans = create_shtnskit_transpose_plans(pencils)

    # Estimate memory usage
    field_count = estimate_field_count()
    memory_mb = estimate_memory_usage_shtnskit(nlat, nlon, lmax, field_count, Float64)
    memory_estimate = "$(round(memory_mb, digits=1)) MB"

    nlm = sht_config.nlm

    # Populate compatibility grids and index arrays
    theta_grid = try
        Vector{Float64}(SHTnsKit.grid_latitudes(sht_config))
    catch
        range(-pi/2, stop=pi/2, length=nlat) |> collect |> Vector{Float64}
    end
    phi_grid = try
        Vector{Float64}(SHTnsKit.grid_longitudes(sht_config))
    catch
        range(0, stop=2pi, length=nlon+1)[1:end-1] |> collect |> Vector{Float64}
    end
    gauss_weights = try
        Vector{Float64}(SHTnsKit.get_gauss_weights(sht_config))
    catch
        ones(Float64, nlat)
    end
    # Construct l/m arrays matching nlm ordering
    l_vals = Vector{Int}(undef, nlm)
    m_vals = Vector{Int}(undef, nlm)
    idx = 1
    for l in 0:lmax
        for m in 0:min(l, mmax)
            if idx <= nlm
                l_vals[idx] = l
                m_vals[idx] = m
            end
            idx += 1
        end
    end

    if get_rank() == 0
        print_shtnskit_config_summary(nlat, nlon, lmax, mmax, nlm, nprocs, memory_estimate)
    end

    return SHTnsKitConfig(
        sht_config, nlat, nlon, lmax, mmax, nlm,
        pencils, fft_plans, transpose_plans, memory_estimate,
        l_vals, m_vals, theta_grid, phi_grid, gauss_weights,
        Dict{Symbol, Any}()  # Initialize empty buffer cache
    )
end

"""
    create_pencil_decomposition_shtnskit(nlat, nlon, nr, sht_config, comm, optimize)

Create PencilArrays decomposition optimized for theta-phi parallelization.
"""
function create_pencil_decomposition_shtnskit(nlat::Int, nlon::Int, nr::Int,
                                             sht_config::SHTnsKit.SHTConfig,
                                             comm, optimize::Bool=true)
    nprocs = MPI.Comm_size(comm)

    # Determine optimal process topology for theta-phi parallelization
    if optimize && nprocs > 1
        proc_dims = optimize_process_topology_shtnskit(nprocs, nlat, nlon)
    else
        proc_dims = (nprocs, 1)
    end

    # Create PencilArrays topology
    # Construct MPI-aware topology via dynamic lookup (MPITopology in recent versions)
    TopoCtor = getproperty(PencilArrays, Symbol("MPITopology"))
    topology = TopoCtor(comm, proc_dims)

    # Physical space pencils for theta-phi parallelization
    # Note: The tuple passed to Pencil specifies the two distributed axes.
    #       The remaining axis (not listed) is locally contiguous.

    dims = (nlat, nlon, nr)
    pencil_theta = Pencil(topology, dims, (2, 3))  # Theta contiguous; distributes phi and r
    pencil_phi   = Pencil(topology, dims, (1, 3))  # Phi contiguous; distributes theta and r
    pencil_r     = Pencil(topology, dims, (1, 2))  # Radial contiguous; distributes theta and phi

    # Spectral space pencil (for (l,m) modes)
    # Use SHTnsKit configuration's nlm
    nlm = sht_config.nlm
    spec_dims = (nlm, 1, nr)
    pencil_spec = Pencil(topology, spec_dims, (1, 3))  # distribute spectral and radial indices
    mixed_dims = (nlm, nlat, nr)
    pencil_mixed = Pencil(topology, mixed_dims, (1, 2))

    return (; theta=pencil_theta,
            θ=pencil_theta,
            phi=pencil_phi,
            φ=pencil_phi,
            r=pencil_r,
            spec=pencil_spec,
            mixed=pencil_mixed)
end

"""
    optimize_process_topology_shtnskit(nprocs, nlat, nlon)

Optimize MPI process topology for theta-phi parallelization.
"""
function optimize_process_topology_shtnskit(nprocs::Int, nlat::Int, nlon::Int)
    # Find factorization that balances theta and phi parallelization
    best_dims = (nprocs, 1)
    best_score = Inf

    for p_theta in 1:nprocs
        if nprocs % p_theta == 0
            p_phi = nprocs ÷ p_theta

            # Check if decomposition makes sense
            if nlat ÷ p_theta < 2 || nlon ÷ p_phi < 2
                continue
            end

            # Check load balance
            theta_imbalance = abs(nlat % p_theta) / nlat
            phi_imbalance = abs(nlon % p_phi) / nlon

            # Prefer more balanced decomposition
            score = theta_imbalance + phi_imbalance

            if score < best_score
                best_score = score
                best_dims = (p_theta, p_phi)
            end
        end
    end

    return best_dims
end

"""
    create_pencil_fft_plans(pencils, dims)

Create FFT plans for efficient phi-direction transforms.
Uses FFTW plans on the underlying arrays of PencilArrays for single-dimension transforms.
"""
function create_pencil_fft_plans(pencils, dims::Tuple{Int,Int,Int})
    nlat, nlon, nr = dims
    fft_plans = Dict{Symbol, Any}()

    try
        # Create FFT plans for phi-direction (longitude) transforms
        # For single-dimension FFTs along a specific axis, we use FFTW directly
        # on the parent array since PencilFFTPlan is designed for multi-dimensional transforms
        if haskey(pencils, :phi)
            sample_array = PencilArray{ComplexF64}(undef, pencils.phi)
            # Create FFTW plans for dimension 2 (phi direction) on the underlying array
            fft_plans[:phi_forward] = FFTW.plan_fft(parent(sample_array), 2)
            fft_plans[:phi_backward] = FFTW.plan_ifft(parent(sample_array), 2)
        end

        # Create plans for other orientations if needed
        if haskey(pencils, :theta)
            sample_theta = PencilArray{ComplexF64}(undef, pencils.theta)
            fft_plans[:theta_forward] = FFTW.plan_fft(parent(sample_theta), 2)
            fft_plans[:theta_backward] = FFTW.plan_ifft(parent(sample_theta), 2)
        end

        if get_rank() == 0
            @info "FFT plans created successfully for $(length(fft_plans) ÷ 2) orientations"
        end
    catch e
        @warn "Could not create FFT plans: $e"
        fft_plans[:fallback] = true
    end

    return fft_plans
end

"""
    create_shtnskit_transpose_plans(pencils)

Create transpose plans for efficient pencil reorientations.
"""
function create_shtnskit_transpose_plans(pencils)
    transpose_plans = Dict{Symbol, Any}()

    # Create transpose operations between adjacent pencils
    # PencilArrays requires pencils differ in at most 1 dimension
    # For multi-step transposes, store intermediate plans

    if haskey(pencils, :theta) && haskey(pencils, :phi)
        try
            transpose_plans[:theta_to_phi] = _shtns_make_transpose(pencils.theta => pencils.phi)
            transpose_plans[:phi_to_theta] = _shtns_make_transpose(pencils.phi => pencils.theta)
        catch e
            if get_rank() == 0
                @debug "Could not create theta<->phi transpose: $e"
            end
        end
    end

    if haskey(pencils, :r) && haskey(pencils, :theta)
        try
            transpose_plans[:theta_to_r] = _shtns_make_transpose(pencils.theta => pencils.r)
            transpose_plans[:r_to_theta] = _shtns_make_transpose(pencils.r => pencils.theta)
        catch e
            if get_rank() == 0
                @debug "Could not create theta<->r transpose: $e"
            end
        end
    end

    if haskey(pencils, :phi) && haskey(pencils, :r)
        try
            transpose_plans[:phi_to_r] = _shtns_make_transpose(pencils.phi => pencils.r)
            transpose_plans[:r_to_phi] = _shtns_make_transpose(pencils.r => pencils.phi)
        catch e
            # This is expected to fail if phi and r differ in >1 dimension
            # Store as multi-step transpose: phi -> theta -> r
            if haskey(transpose_plans, :phi_to_theta) && haskey(transpose_plans, :theta_to_r)
                transpose_plans[:phi_to_r] = :multi_step  # Marker for multi-step
                transpose_plans[:r_to_phi] = :multi_step
                if get_rank() == 0
                    @debug "Using multi-step transpose for phi<->r via theta"
                end
            else
                if get_rank() == 0
                    @debug "Could not create phi<->r transpose: $e"
                end
            end
        end
    end

    if get_rank() == 0
        @info "Created $(length(transpose_plans)) transpose plans for pencil reorientations"
    end

    return transpose_plans
end

"""
    estimate_memory_usage_shtnskit(nlat, nlon, lmax, field_count, T)

Estimate memory usage for SHTnsKit-based transforms with PencilArrays.
"""
function estimate_memory_usage_shtnskit(nlat::Int, nlon::Int, lmax::Int,
                                       field_count::Int, ::Type{T}) where T

    # Physical grid memory per process (distributed)
    physical_memory_per_process = (nlat * nlon * i_N * sizeof(T)) / get_nprocs()

    # Spectral memory (approximate)
    nlm = SHTnsKit.nlm_calc(lmax, lmax, 1)
    spectral_memory_per_process = (nlm * i_N * sizeof(ComplexF64) * 2) / get_nprocs()

    # PencilArrays working memory (transpose buffers)
    transpose_memory = max(physical_memory_per_process, spectral_memory_per_process)

    # PencilFFTs working memory
    fft_memory = physical_memory_per_process * 0.5

    # Total per field per process
    per_field_memory = physical_memory_per_process + spectral_memory_per_process +
                      transpose_memory + fft_memory

    # Total for all fields
    total_memory = per_field_memory * field_count

    return total_memory / (1024^2)  # Convert to MB
end

"""
    print_shtnskit_config_summary(nlat, nlon, lmax, mmax, nlm, nprocs, memory_estimate)

Print configuration summary for SHTnsKit setup.
"""
function print_shtnskit_config_summary(nlat, nlon, lmax, mmax, nlm, nprocs, memory_estimate)
    println("\n╔═══════════════════════════════════════════════════════╗")
    println("║         SHTnsKit Configuration Summary                ║")
    println("╠═══════════════════════════════════════════════════════╣")
    println("║ Grid Configuration:                                   ║")
    println("║   Physical grid:    $(lpad(nlat,4)) × $(lpad(nlon,4)) × $(lpad(i_N,4))         ║")
    println("║   Spectral modes:   lmax=$(lpad(lmax,3)), mmax=$(lpad(mmax,3))              ║")
    println("║   Total modes:      $(lpad(nlm,5))                             ║")
    println("║                                                       ║")
    println("║ Parallel Configuration:                               ║")
    println("║   MPI Processes:    $(lpad(nprocs,4))                              ║")
    println("║   Theta-Phi Parallel: PencilArrays + PencilFFTs      ║")
    println("║   SHTnsKit.jl:      Registered package               ║")
    println("║   Memory/process:   $(lpad(memory_estimate,10))                    ║")
    println("╚═══════════════════════════════════════════════════════╝")
end
