# ================================================================================
# SHTnsKit Spherical Harmonic Transforms with PencilArrays Integration
# ================================================================================
#
# This module implements spherical harmonic transforms using SHTnsKit.jl
# with MPI parallelization across theta and phi directions using PencilArrays
# and efficient FFTs using PencilFFTs.
#
# ARCHITECTURE OVERVIEW:
# ----------------------
# The geodynamo simulation uses spherical harmonic (SH) transforms to convert
# between physical space (θ, φ, r) and spectral space (l, m, r). This module
# provides the infrastructure for these transforms with MPI parallelization.
#
# Key Components:
# 1. SHTnsKitConfig - Configuration struct holding all transform parameters
# 2. Pencil Decomposition - Data distribution strategy across MPI processes
# 3. FFT Plans - Precomputed FFTW plans for efficient longitude transforms
# 4. Transpose Plans - Plans for redistributing data between pencil orientations
#
# DATA LAYOUT:
# ------------
# Physical space: (nlat, nlon, nr) - latitude × longitude × radius
# Spectral space: (nlm, 1, nr) - spectral modes × 1 × radius
#   where nlm = number of (l,m) mode pairs = Σ(min(l,mmax)+1) for l=0:lmax
#
# PENCIL DECOMPOSITION:
# ---------------------
# "Pencils" are 1D decompositions where one dimension is fully local.
# - theta pencil: θ dimension local, (φ, r) distributed
# - phi pencil: φ dimension local, (θ, r) distributed
# - r pencil: r dimension local, (θ, φ) distributed
#
# This allows efficient computation along the local dimension without
# MPI communication, with transposes used to switch between orientations.
#
# ================================================================================

using SHTnsKit
using PencilArrays
using PencilFFTs
using FFTW
using LinearAlgebra
using Base.Threads

# ================================================================================
# SHTnsKit v1.1.15+ Feature Flags
# ================================================================================
# These flags indicate which v1.1.15 features are available and should be used

const SHTNSKIT_USE_DISTRIBUTED = true      # Use dist_analysis/dist_synthesis
const SHTNSKIT_USE_QST = true              # Use SHqst_to_spat/spat_to_SHqst for 3D vectors
const SHTNSKIT_USE_SCRATCH_BUFFERS = true  # Use scratch_spatial/scratch_fft helpers

# ================================================================================
# Thread-Safe Buffer Cache Access
# ================================================================================
# The buffer cache is shared across threads and needs synchronization to avoid
# race conditions when multiple threads access or create buffers simultaneously.

"""
    _BUFFER_CACHE_LOCK

Global ReentrantLock for thread-safe access to SHTnsKitConfig buffer caches.
All access to config._buffer_cache should be protected by this lock.
"""
const _BUFFER_CACHE_LOCK = ReentrantLock()

"""
    get_cached_buffer!(create_func::Function, config, key::Symbol)
    get_cached_buffer!(config, key::Symbol) do ... end

Thread-safe accessor for buffer cache. Returns existing buffer if present,
otherwise creates a new one using `create_func()` and caches it.

Note: The function parameter comes FIRST to support Julia's `do` block syntax.
When using `do` block, Julia desugars it to pass the closure as the first argument.

# Arguments
- `create_func::Function`: Zero-argument function to create buffer if not cached
- `config`: SHTnsKitConfig object containing the buffer cache
- `key::Symbol`: Key to look up in the buffer cache

# Returns
The cached or newly created buffer.

# Example
```julia
buffer = get_cached_buffer!(config, :my_buffer) do
    zeros(Float64, nlat, nlon)
end
```
"""
function get_cached_buffer!(create_func::Function, config, key::Symbol)
    lock(_BUFFER_CACHE_LOCK) do
        if !haskey(config._buffer_cache, key)
            config._buffer_cache[key] = create_func()
        end
        return config._buffer_cache[key]
    end
end

"""
    clear_buffer_cache!(config)

Thread-safe clearing of all cached buffers. Useful when changing configurations
or to free memory.
"""
function clear_buffer_cache!(config)
    lock(_BUFFER_CACHE_LOCK) do
        empty!(config._buffer_cache)
    end
end

# ================================================================================
# Utility Functions
# ================================================================================

"""
    _shtns_make_transpose(pair)

Create a PencilArrays transpose plan between two pencil configurations.
Used internally to set up efficient data redistribution operations.

# Arguments
- `pair`: A Pair of source and destination Pencil objects (src => dest)

# Returns
- A Transposition object that can be used with `mul!` for data redistribution
"""
@inline function _shtns_make_transpose(pair)
    src = first(pair)
    dest = last(pair)

    # Create temporary arrays for planning (use Float64 as a generic type)
    src_array = PencilArray{Float64}(undef, src)
    dest_array = PencilArray{Float64}(undef, dest)

    # Create the transposition plan
    return PencilArrays.Transpositions.Transposition(dest_array, src_array)
end

# ================================================================================
# Memory Estimation Helpers
# ================================================================================

"""
    estimate_field_count() -> Int

Estimate the number of field arrays typically allocated simultaneously.
Used for memory usage estimation. Returns 6 as a reasonable default
covering velocity (3 components), temperature, composition, and magnetic field.
"""
estimate_field_count() = 6

# ================================================================================
# Default Grid Size Functions
# ================================================================================

"""
    get_default_nlat() -> Int

Get the default number of latitude points for the grid.
Tries to read from the parameter system first, falls back to 64.
Used during precompilation when parameters may not be available.
"""
function get_default_nlat()
    # Try to use parameter system if available, otherwise use reasonable default
    try
        params = get_parameters()
        return params.i_Th
    catch
        return 64  # Compatible with most SHTnsKit configurations
    end
end

"""
    get_default_nlon() -> Int

Get the default number of longitude points for the grid.
Tries to read from the parameter system first, falls back to 128.
Power of 2 is preferred for efficient FFT operations.
"""
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

# Forward declaration for fields.jl - allows type hierarchy for SHT configs
abstract type AbstractSHTnsConfig end

"""
    SHTnsKitConfig <: AbstractSHTnsConfig

Main configuration structure for spherical harmonic transforms using SHTnsKit.
This struct encapsulates all parameters needed for transforms and parallelization.

# Fields
## Core SHTnsKit Configuration
- `sht_config`: The underlying SHTnsKit.SHTConfig object

## Grid Parameters
- `nlat::Int`: Number of latitude (theta) points (Gauss-Legendre grid)
- `nlon::Int`: Number of longitude (phi) points (uniform grid)
- `lmax::Int`: Maximum spherical harmonic degree
- `mmax::Int`: Maximum spherical harmonic order (≤ lmax)
- `nlm::Int`: Total number of (l,m) spectral mode pairs

## Parallelization Infrastructure
- `pencils::NamedTuple`: Collection of PencilArrays Pencil objects for different
  data orientations (:theta, :phi, :r, :spec, :mixed)
- `fft_plans::Dict{Symbol,Any}`: Precomputed FFTW plans for FFT operations
- `transpose_plans::Dict{Symbol,Any}`: Plans for data redistribution between pencils

## Auxiliary Data
- `memory_estimate::String`: Human-readable memory usage estimate
- `l_values::Vector{Int}`: Spherical harmonic degree for each spectral index
- `m_values::Vector{Int}`: Spherical harmonic order for each spectral index
- `theta_grid::Vector{Float64}`: Latitude values (Gauss-Legendre nodes)
- `phi_grid::Vector{Float64}`: Longitude values (uniform spacing)
- `gauss_weights::Vector{Float64}`: Gauss-Legendre quadrature weights

## Internal
- `_buffer_cache::Dict{Symbol,Any}`: Reusable work arrays to reduce allocations

# Usage
```julia
config = create_shtnskit_config(lmax=32, mmax=32, nlat=64, nlon=128)
```
"""
struct SHTnsKitConfig <: AbstractSHTnsConfig
    # SHTnsKit configuration - the underlying transform engine
    sht_config::SHTnsKit.SHTConfig

    # Grid parameters defining the resolution
    nlat::Int   # Number of latitude points (Gauss-Legendre)
    nlon::Int   # Number of longitude points (equispaced)
    lmax::Int   # Maximum spherical harmonic degree
    mmax::Int   # Maximum spherical harmonic order
    nlm::Int    # Total number of spectral modes

    # PencilArrays decomposition for MPI parallelization
    # Contains :theta, :phi, :r, :spec, :mixed pencil configurations
    pencils::NamedTuple

    # FFTW plans for longitude FFTs (keyed by :phi_forward, :phi_backward, etc.)
    fft_plans::Dict{Symbol, Any}

    # Transpose plans for switching between pencil orientations
    transpose_plans::Dict{Symbol, Any}

    # Human-readable memory estimate string (e.g., "256.5 MB")
    memory_estimate::String

    # Arrays mapping spectral index to (l,m) values for convenience
    l_values::Vector{Int}
    m_values::Vector{Int}

    # Physical grid coordinates
    theta_grid::Vector{Float64}    # Latitude values [-π/2, π/2] (Gauss-Legendre nodes)
    phi_grid::Vector{Float64}      # Longitude values [0, 2π)
    gauss_weights::Vector{Float64} # Quadrature weights for integration

    # Internal buffer cache to avoid repeated allocations
    _buffer_cache::Dict{Symbol, Any}
end

"""
    create_shtnskit_config(; lmax, mmax, nlat, nlon, nr, optimize_decomp) -> SHTnsKitConfig

Create and initialize a complete SHTnsKit configuration for spherical harmonic
transforms with MPI parallelization.

# Keyword Arguments
- `lmax::Int`: Maximum spherical harmonic degree (required)
- `mmax::Int=lmax`: Maximum spherical harmonic order (≤ lmax)
- `nlat::Int`: Number of latitude points. Must be ≥ lmax+1 for numerical accuracy.
  Defaults to max(lmax+2, parameter system value)
- `nlon::Int`: Number of longitude points. Must be ≥ 2*mmax+1 for alias-free transforms.
  Powers of 2 are preferred for FFT efficiency.
- `nr::Int=i_N`: Number of radial points (from parameter system)
- `optimize_decomp::Bool=true`: Whether to optimize MPI process topology

# Returns
- `SHTnsKitConfig`: Fully initialized configuration ready for transforms

# Algorithm
1. Create base SHTnsKit configuration with Gauss-Legendre grid
2. Set up pencil decomposition for MPI parallelization
3. Create FFT plans for longitude transforms
4. Create transpose plans for data redistribution
5. Initialize grid coordinates and quadrature weights

# Example
```julia
# Create config for lmax=63 simulation
config = create_shtnskit_config(lmax=63, nlat=96, nlon=192)
```
"""
function create_shtnskit_config(; lmax::Int, mmax::Int=lmax,
                               nlat::Int=max(lmax+2, get_default_nlat()),
                               nlon::Int=max(2*lmax+1, 4, get_default_nlon()),
                               nr::Int=i_N,
                               optimize_decomp::Bool=true)

    # Step 1: Create base SHTnsKit configuration
    # Uses Gauss-Legendre quadrature for latitude (exact integration up to degree 2*nlat-1)
    # and uniform grid for longitude (FFT-based)
    sht_config = SHTnsKit.create_gauss_config(lmax, nlat;
                                            mmax=mmax,
                                            nlon=nlon,
                                            norm=:orthonormal)  # Orthonormal Y_l^m normalization

    # Disable precomputed Legendre polynomial tables to avoid version-dependent
    # dimension mismatches between SHTnsKit's table creation and transform code.
    # The on-the-fly Plm computation is reliable and the performance impact is
    # minimal for typical problem sizes.
    SHTnsKit.disable_plm_tables!(sht_config)

    # Step 2: Set up MPI parallelization infrastructure
    comm = get_comm()      # Get MPI communicator (or serial fallback)
    nprocs = get_nprocs()  # Number of MPI processes

    # Step 3: Create pencil decomposition for distributed memory parallelism
    # Pencils define how data is distributed across MPI processes
    pencils = create_pencil_decomposition_shtnskit(nlat, nlon, nr, sht_config, comm, optimize_decomp)

    # Step 4: Create FFT plans for longitude (phi) direction transforms
    # These are precomputed FFTW plans for efficiency
    fft_plans = create_pencil_fft_plans(pencils, (nlat, nlon, nr))

    # Step 5: Create transpose plans for data redistribution between pencils
    # Transposes are needed when switching which dimension is local
    transpose_plans = create_shtnskit_transpose_plans(pencils)

    # Estimate memory usage for user information
    field_count = estimate_field_count()
    memory_mb = estimate_memory_usage_shtnskit(nlat, nlon, lmax, field_count, Float64)
    memory_estimate = "$(round(memory_mb, digits=1)) MB"

    # Get total number of spectral modes from SHTnsKit
    nlm = sht_config.nlm

    # Step 6: Initialize grid coordinates and quadrature weights
    # These are used for physical space operations and integration

    # Latitude grid (Gauss-Legendre nodes, latitude in [-π/2, π/2])
    theta_grid = try
        Vector{Float64}(SHTnsKit.grid_latitudes(sht_config))
    catch
        # Fallback: uniform grid (less accurate but works)
        range(-pi/2, stop=pi/2, length=nlat) |> collect |> Vector{Float64}
    end

    # Longitude grid (uniform spacing in [0, 2π))
    phi_grid = try
        Vector{Float64}(SHTnsKit.grid_longitudes(sht_config))
    catch
        range(0, stop=2pi, length=nlon+1)[1:end-1] |> collect |> Vector{Float64}
    end

    # Gauss-Legendre quadrature weights for numerical integration over θ
    gauss_weights = try
        Vector{Float64}(SHTnsKit.get_gauss_weights(sht_config))
    catch
        ones(Float64, nlat)  # Fallback: uniform weights
    end

    # Step 7: Build spectral index to (l,m) mapping arrays
    # These allow quick lookup of the degree l and order m for each spectral index
    # Ordering follows the SHTnsKit convention: m varies fastest within each l
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

    # Step 8: Initialize buffer cache with SHTnsKit v1.1.15 scratch buffers
    buffer_cache = Dict{Symbol, Any}()
    if SHTNSKIT_USE_SCRATCH_BUFFERS
        try
            # Use SHTnsKit's native scratch buffer allocation for better memory management
            buffer_cache[:spatial_scratch] = SHTnsKit.scratch_spatial(sht_config, Float64)
            buffer_cache[:fft_scratch] = SHTnsKit.scratch_fft(sht_config, ComplexF64)
            if get_rank() == 0
                @info "SHTnsKit v1.1.15 scratch buffers allocated"
            end
        catch e
            # Fallback for older SHTnsKit versions
            if get_rank() == 0
                @debug "Could not allocate SHTnsKit scratch buffers: $e"
            end
        end
    end

    return SHTnsKitConfig(
        sht_config, nlat, nlon, lmax, mmax, nlm,
        pencils, fft_plans, transpose_plans, memory_estimate,
        l_vals, m_vals, theta_grid, phi_grid, gauss_weights,
        buffer_cache
    )
end

"""
    create_pencil_decomposition_shtnskit(nlat, nlon, nr, sht_config, comm, optimize)

Create PencilArrays decomposition optimized for spherical harmonic transforms.

# The Pencil Decomposition Strategy

In a pencil decomposition, 3D data is distributed across MPI processes such that
one dimension is always fully local (not split across processes). This enables
efficient operations along that dimension without MPI communication.

## Physical Space Pencils (nlat × nlon × nr)

```
theta pencil:  [θ local] × [φ distributed] × [r distributed]
               → Best for operations along latitude (Legendre transforms)

phi pencil:    [θ distributed] × [φ local] × [r distributed]
               → Best for FFTs along longitude

r pencil:      [θ distributed] × [φ distributed] × [r local]
               → Best for radial operations (derivatives, boundary conditions)
```

## Spectral Space Pencil (nlm × 1 × nr)
The spectral pencil stores spherical harmonic coefficients indexed by a
combined (l,m) index. The middle dimension is 1 (dummy) for compatibility.

# Arguments
- `nlat, nlon, nr`: Grid dimensions
- `sht_config`: SHTnsKit configuration (provides nlm)
- `comm`: MPI communicator
- `optimize`: Whether to optimize process topology for load balance

# Returns
NamedTuple with pencil configurations: (:theta, :θ, :phi, :φ, :r, :spec, :mixed)
"""
function create_pencil_decomposition_shtnskit(nlat::Int, nlon::Int, nr::Int,
                                             sht_config::SHTnsKit.SHTConfig,
                                             comm, optimize::Bool=true)
    nprocs = MPI.Comm_size(comm)

    # Determine optimal 2D process grid for theta-phi parallelization
    # Goal: balance load across processes while respecting grid dimensions
    if optimize && nprocs > 1
        proc_dims = optimize_process_topology_shtnskit(nprocs, nlat, nlon)
    else
        proc_dims = (nprocs, 1)  # Simple 1D decomposition
    end

    # Create PencilArrays MPI topology
    # MPITopology maps the 2D process grid to MPI ranks
    TopoCtor = getproperty(PencilArrays, Symbol("MPITopology"))
    topology = TopoCtor(comm, proc_dims)

    # Create physical space pencils
    # The tuple (i, j) in Pencil(..., (i, j)) specifies which dimensions are DISTRIBUTED
    # The remaining dimension is LOCAL (contiguous in memory on each process)
    dims = (nlat, nlon, nr)

    # Theta pencil: dimension 1 (theta) is local
    # Dimensions 2 (phi) and 3 (r) are distributed across processes
    # Use this when you need all theta values for a given (phi, r) point
    pencil_theta = Pencil(topology, dims, (2, 3))

    # Phi pencil: dimension 2 (phi) is local
    # Needed for FFTs along longitude direction
    pencil_phi = Pencil(topology, dims, (1, 3))

    # R pencil: dimension 3 (r) is local
    # Needed for radial derivatives and boundary conditions
    pencil_r = Pencil(topology, dims, (1, 2))

    # Create spectral space pencil
    # nlm = total number of (l,m) mode pairs
    nlm = sht_config.nlm
    spec_dims = (nlm, 1, nr)  # Middle dimension is dummy (size 1)
    pencil_spec = Pencil(topology, spec_dims, (1, 3))

    # Mixed pencil for intermediate computations
    mixed_dims = (nlm, nlat, nr)
    pencil_mixed = Pencil(topology, mixed_dims, (1, 2))

    # Return named tuple with both ASCII and Unicode names for convenience
    return (; theta=pencil_theta,
            θ=pencil_theta,      # Unicode alias
            phi=pencil_phi,
            φ=pencil_phi,        # Unicode alias
            r=pencil_r,
            spec=pencil_spec,
            mixed=pencil_mixed)
end

"""
    optimize_process_topology_shtnskit(nprocs, nlat, nlon) -> Tuple{Int,Int}

Find optimal 2D MPI process grid for theta-phi parallelization.

# The Optimization Problem
Given `nprocs` MPI processes, we need to factor it as `nprocs = p_theta × p_phi`
such that:
1. Each process gets at least 2 grid points in each direction
2. Load imbalance (due to non-divisibility) is minimized
3. The decomposition is valid (exact factorization)

# Algorithm
Iterates through all valid factorizations and scores them by load imbalance.
The load imbalance for a dimension is: |grid_size mod processes| / grid_size

# Example
For nprocs=12, nlat=64, nlon=128:
- Possible factorizations: (1,12), (2,6), (3,4), (4,3), (6,2), (12,1)
- (4,3) gives nlat/4=16 points/proc in θ, nlon/3≈43 points/proc in φ
- This might be optimal if it minimizes total imbalance

# Arguments
- `nprocs`: Total number of MPI processes
- `nlat`: Number of latitude points
- `nlon`: Number of longitude points

# Returns
- `(p_theta, p_phi)`: Optimal process grid dimensions
"""
function optimize_process_topology_shtnskit(nprocs::Int, nlat::Int, nlon::Int)
    # Start with default 1D decomposition
    best_dims = (nprocs, 1)
    best_score = Inf

    # Try all valid factorizations of nprocs
    for p_theta in 1:nprocs
        if nprocs % p_theta == 0
            p_phi = nprocs ÷ p_theta

            # Ensure at least 2 grid points per process in each direction
            # Fewer than 2 points causes issues with stencil operations
            if nlat ÷ p_theta < 2 || nlon ÷ p_phi < 2
                continue
            end

            # Compute load imbalance score
            # Lower is better: 0 means perfectly divisible
            theta_imbalance = abs(nlat % p_theta) / nlat
            phi_imbalance = abs(nlon % p_phi) / nlon

            # Total score is sum of imbalances
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
    create_pencil_fft_plans(pencils, dims) -> Dict{Symbol,Any}

Create precomputed FFTW plans for efficient FFT operations.

# Why Precomputed Plans?
FFTW achieves best performance when FFT plans are created once and reused.
Plan creation involves:
1. Analyzing the input size and memory layout
2. Choosing optimal algorithm (Cooley-Tukey, Bluestein, etc.)
3. Possibly benchmarking different strategies

# Plan Types Created
- `:phi_forward` / `:phi_backward`: FFT/IFFT along longitude (dimension 2)
- `:theta_forward` / `:theta_backward`: FFT/IFFT for theta pencil orientation

# Technical Notes
- Plans operate on the `parent()` array of PencilArrays (the underlying Julia array)
- We use FFTW directly rather than PencilFFTPlan because we need single-dimension
  transforms, not full multi-dimensional distributed FFTs
- The plans are tied to specific array sizes and memory layouts

# Arguments
- `pencils`: NamedTuple of Pencil configurations
- `dims`: Tuple (nlat, nlon, nr) of grid dimensions

# Returns
Dict mapping plan names to FFTW plan objects. Contains `:fallback => true` on error.
"""
function create_pencil_fft_plans(pencils, dims::Tuple{Int,Int,Int})
    nlat, nlon, nr = dims
    fft_plans = Dict{Symbol, Any}()

    try
        # Create FFT plans for phi-direction (longitude) transforms
        # These are the most commonly used for spherical harmonic transforms
        if haskey(pencils, :phi)
            # Create a sample array matching the local dimensions on this process
            sample_array = PencilArray{ComplexF64}(undef, pencils.phi)

            # Plan FFT along dimension 2 (the phi/longitude direction)
            # parent() extracts the underlying Julia array from the PencilArray
            fft_plans[:phi_forward] = FFTW.plan_fft(parent(sample_array), 2)
            fft_plans[:phi_backward] = FFTW.plan_ifft(parent(sample_array), 2)
        end

        # Create plans for theta pencil orientation (less commonly used)
        if haskey(pencils, :theta)
            sample_theta = PencilArray{ComplexF64}(undef, pencils.theta)
            fft_plans[:theta_forward] = FFTW.plan_fft(parent(sample_theta), 2)
            fft_plans[:theta_backward] = FFTW.plan_ifft(parent(sample_theta), 2)
        end

        if get_rank() == 0
            @info "FFT plans created successfully for $(length(fft_plans) ÷ 2) orientations"
        end
    catch e
        # On failure, set fallback flag so calling code can use unplanned FFTs
        @warn "Could not create FFT plans: $e"
        fft_plans[:fallback] = true
    end

    return fft_plans
end

"""
    create_shtnskit_transpose_plans(pencils) -> Dict{Symbol,Any}

Create transpose plans for redistributing data between pencil orientations.

# What is a Transpose?
In pencil decomposition, a "transpose" is an MPI all-to-all communication that
redistributes data so a different dimension becomes local. For example:
- theta → phi transpose: makes longitude local (for FFTs)
- phi → r transpose: makes radius local (for radial derivatives)

# Why Plans?
Like FFT plans, transpose plans encode the communication pattern and can
optimize buffer allocation and MPI operations.

# Transpose Constraints
PencilArrays requires that source and destination pencils differ in at most
one distributed dimension. If they differ in two dimensions, we need a
multi-step transpose (e.g., phi → theta → r).

# Plan Keys Created
- `:theta_to_phi`, `:phi_to_theta`: Switch between theta and phi local
- `:theta_to_r`, `:r_to_theta`: Switch between theta and r local
- `:phi_to_r`, `:r_to_phi`: May be direct or marked as `:multi_step`

# Usage
```julia
# Transpose data from theta-local to phi-local orientation
mul!(dest_array, transpose_plans[:theta_to_phi], src_array)
```
"""
function create_shtnskit_transpose_plans(pencils)
    transpose_plans = Dict{Symbol, Any}()

    # Create transpose operations between adjacent pencils
    # "Adjacent" means they differ in only one distributed dimension

    # Theta ↔ Phi transposes (most common for SH transforms)
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

    # Theta ↔ R transposes (for switching to radial operations)
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

    # Phi ↔ R transposes
    # These may fail if phi and r pencils differ in >1 dimension
    # In that case, use multi-step: phi → theta → r
    if haskey(pencils, :phi) && haskey(pencils, :r)
        try
            transpose_plans[:phi_to_r] = _shtns_make_transpose(pencils.phi => pencils.r)
            transpose_plans[:r_to_phi] = _shtns_make_transpose(pencils.r => pencils.phi)
        catch e
            # Expected failure - try multi-step approach
            if haskey(transpose_plans, :phi_to_theta) && haskey(transpose_plans, :theta_to_r)
                transpose_plans[:phi_to_r] = :multi_step  # Marker for calling code
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
    # Get version info for feature flags
    version = try
        string(pkgversion(SHTnsKit))
    catch
        "≥1.1.15"
    end

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
    println("║   SHTnsKit.jl:      v$(lpad(version,7))                      ║")
    println("║   Memory/process:   $(lpad(memory_estimate,10))                    ║")
    println("║                                                       ║")
    println("║ v1.1.15+ Features:                                    ║")
    println("║   Distributed transforms: $(SHTNSKIT_USE_DISTRIBUTED ? "enabled " : "disabled")                ║")
    println("║   QST vector transforms:  $(SHTNSKIT_USE_QST ? "enabled " : "disabled")                ║")
    println("║   Scratch buffers:        $(SHTNSKIT_USE_SCRATCH_BUFFERS ? "enabled " : "disabled")                ║")
    println("╚═══════════════════════════════════════════════════════╝")
end
