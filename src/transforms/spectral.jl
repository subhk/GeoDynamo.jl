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
# Spectral space: (lmax+1, mmax+1, nr) - degree × order × radius
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
# SHTnsKit v1.2+ Feature Flags
# ================================================================================
# These flags indicate which v1.2+ features are available and should be used

const SHTNSKIT_USE_DISTRIBUTED = true      # Use dist_analysis/dist_synthesis
const SHTNSKIT_USE_QST = true              # Use SHqst_to_spat/spat_to_SHqst for 3D vectors
const SHTNSKIT_USE_SCRATCH_BUFFERS = true  # Use scratch_spatial/scratch_fft helpers

abstract type AbstractTransformWorkspace end

# ================================================================================
# SHTnsBuffers — typed replacement for __buffer_cache::Dict{Symbol,Any}
# ================================================================================

"""
    SHTnsBuffers

Concrete mutable struct holding all reusable work arrays for SHTnsKit transforms.
Replaces the previous `Dict{Symbol,Any}` buffer cache with named, type-stable fields.

Fields that are `nothing` at construction time are lazily allocated on first use.
"""
mutable struct SHTnsBuffers
    # Set eagerly at config creation (CPU only)
    sht_plan                 :: Union{SHTnsKit.SHTPlan, Nothing}

    # Scalar synthesis / analysis output buffers
    synth_out                :: Union{Matrix{Float64}, Nothing}
    anal_out                 :: Union{Matrix{ComplexF64}, Nothing}

    # Vector synthesis output buffers (tangential components)
    vt_out                   :: Union{Matrix{Float64}, Nothing}
    vp_out                   :: Union{Matrix{Float64}, Nothing}

    # Vector analysis output buffers (toroidal/poloidal coefficients)
    slm_out                  :: Union{Matrix{ComplexF64}, Nothing}
    tlm_out                  :: Union{Matrix{ComplexF64}, Nothing}

    # Temporary phi-pencil arrays for transpose-based synthesis/analysis
    synthesis_phi_tmp        :: Union{PencilArray, Nothing}
    analysis_phi_tmp         :: Union{PencilArray, Nothing}

    # Coefficient extraction buffers (scalar)
    coeffs_buffer            :: Union{Matrix{ComplexF64}, Nothing}
    coeffs_buffer_gathered   :: Union{Matrix{ComplexF64}, Nothing}

    # Coefficient extraction buffers (paired, for vector transforms)
    coeffs_buffer_pair1      :: Union{Matrix{ComplexF64}, Nothing}
    coeffs_buffer_pair2      :: Union{Matrix{ComplexF64}, Nothing}
    coeffs_gathered_pair1    :: Union{Matrix{ComplexF64}, Nothing}
    coeffs_gathered_pair2    :: Union{Matrix{ComplexF64}, Nothing}

    # Radial-component poloidal synthesis buffer
    pol_rad_coeffs_buffer    :: Union{Matrix{ComplexF64}, Nothing}

    # Vector component physical-space buffers used in vector analysis
    vector_component_vt      :: Union{Matrix{Float64}, Nothing}
    vector_component_vp      :: Union{Matrix{Float64}, Nothing}

    # Generic physical-slice extraction buffers
    phi_slice_buffer         :: Union{Matrix{Float64}, Nothing}
    generic_slice_buffer     :: Union{Matrix{Float64}, Nothing}
    vector_component_buffer  :: Union{Matrix{Float64}, Nothing}

    # Local storage-coordinate -> global linear spectral mode map
    local_spectral_lm_map    :: Union{Matrix{Int}, Nothing}
    local_spectral_slot_lookup :: Union{Vector{CartesianIndex{2}}, Nothing}
    local_spectral_mode_indices :: Union{Vector{Int}, Nothing}

    # Solver-level transform workspace (set by solver/backend.jl)
    solver_transform_workspace :: Union{AbstractTransformWorkspace, Nothing}

    # Device metadata (set at config creation)
    transform_device         :: Union{AbstractArchitecture, Symbol, Nothing}

    # SHTnsKit scratch buffers (CPU only, optional)
    spatial_scratch          :: Union{AbstractArray, Nothing}
    fft_scratch              :: Union{AbstractArray, Nothing}

    # MIE vector-transform spheroidal scalar buffers
    mie_spheroidal_real      :: Union{AbstractArray, Nothing}
    mie_spheroidal_imag      :: Union{AbstractArray, Nothing}
end

"""
    SHTnsBuffers()

Construct a fully-uninitialized `SHTnsBuffers` with all fields set to `nothing`.
"""
SHTnsBuffers() = SHTnsBuffers(
    nothing, nothing, nothing, nothing, nothing, nothing, nothing,
    nothing, nothing, nothing, nothing, nothing, nothing, nothing,
    nothing, nothing, nothing, nothing, nothing, nothing, nothing,
    nothing, nothing, nothing, nothing, nothing, nothing, nothing,
    nothing, nothing,
)

# ================================================================================
# Field map: legacy Symbol key → SHTnsBuffers field name
# ================================================================================
# Used to define typed accessors for call sites that still pass a Symbol key.
# Keys not in this map (e.g. :sht_plan, :solver_transform_workspace, :transform_device,
# :spatial_scratch, :fft_scratch) are set directly and not routed through this function.

const __BUFFERS_FIELD_MAP = Dict{Symbol, Symbol}(
    :synth_out                   => :synth_out,
    :anal_out                    => :anal_out,
    :vt_out                      => :vt_out,
    :vp_out                      => :vp_out,
    :slm_out                     => :slm_out,
    :tlm_out                     => :tlm_out,
    :synthesis_phi_tmp           => :synthesis_phi_tmp,
    :analysis_phi_tmp            => :analysis_phi_tmp,
    :coeffs_buffer               => :coeffs_buffer,
    :coeffs_buffer_gathered      => :coeffs_buffer_gathered,
    :coeffs_buffer_pair1         => :coeffs_buffer_pair1,
    :coeffs_buffer_pair2         => :coeffs_buffer_pair2,
    :coeffs_gathered_pair1       => :coeffs_gathered_pair1,
    :coeffs_gathered_pair2       => :coeffs_gathered_pair2,
    :pol_rad_coeffs_buffer       => :pol_rad_coeffs_buffer,
    :vector_component_buffer_vt  => :vector_component_vt,
    :vector_component_buffer_vp  => :vector_component_vp,
    :phi_slice_buffer            => :phi_slice_buffer,
    :generic_slice_buffer        => :generic_slice_buffer,
    :vector_component_buffer     => :vector_component_buffer,
    :mie_spheroidal_real         => :mie_spheroidal_real,
    :mie_spheroidal_imag         => :mie_spheroidal_imag,
)

@inline function __shtns_buffer_field(::Val{key}) where {key}
    error("get_cached_buffer!: unknown buffer key $(repr(key)). Add it to SHTnsBuffers and __BUFFERS_FIELD_MAP.")
end

for (key, field) in __BUFFERS_FIELD_MAP
    @eval @inline __shtns_buffer_field(::Val{$(QuoteNode(key))}) = Val{$(QuoteNode(field))}()
end

# ================================================================================
# Thread-Safe Buffer Cache Access
# ================================================================================
# The buffer cache is shared across threads and needs synchronization to avoid
# race conditions when multiple threads access or create buffers simultaneously.

"""
    __BUFFER_CACHE_LOCK

Global ReentrantLock for thread-safe access to SHTnsKitConfig buffer caches.
All access to config.__buffers should be protected by this lock.
"""
const __BUFFER_CACHE_LOCK = ReentrantLock()

"""
    get_cached_buffer!(create_func, config, key::Symbol)
    get_cached_buffer!(config, key::Symbol) do ... end

Thread-safe accessor for buffer cache. Returns existing buffer if present,
otherwise creates a new one using `create_func()` and caches it.

Note: The function parameter comes FIRST to support Julia's `do` block syntax.
When using `do` block, Julia desugars it to pass the closure as the first argument.

# Arguments
- `create_func`: Zero-argument function to create buffer if not cached
- `config`: SHTnsKitConfig object containing `__buffers::SHTnsBuffers`
- `key::Symbol`: Key to look up (mapped to a field of `SHTnsBuffers`)

# Returns
The cached or newly created buffer.

# Example
```julia
buffer = get_cached_buffer!(config, :my_buffer) do
    zeros(Float64, nlat, nlon)
end
```
"""
@inline function get_cached_buffer!(create_func::F, config, key::Symbol) where {F}
    return get_cached_buffer!(create_func, config, Val(key))
end

@inline function get_cached_buffer!(create_func::F, config, ::Val{key}) where {F,key}
    return __get_cached_buffer_field!(create_func, config, __shtns_buffer_field(Val(key)))
end

@inline function __get_cached_buffer_field!(create_func::F, config, ::Val{field}) where {F,field}
    lock(__BUFFER_CACHE_LOCK) do
        b = config.__buffers
        val = getfield(b, field)
        if val === nothing
            val = create_func()
            setfield!(b, field, val)
        end
        return val
    end
end

"""
    clear_buffer_cache!(config)

Thread-safe clearing of all lazily-allocated cached buffers. Useful when changing
configurations or to free memory. Does NOT clear `sht_plan`, `solver_transform_workspace`,
`transform_device`, `spatial_scratch`, or `fft_scratch` (these are set at
construction/initialization and should persist).
"""
function clear_buffer_cache!(config)
    lock(__BUFFER_CACHE_LOCK) do
        b = config.__buffers
        b.synth_out              = nothing
        b.anal_out               = nothing
        b.vt_out                 = nothing
        b.vp_out                 = nothing
        b.slm_out                = nothing
        b.tlm_out                = nothing
        b.synthesis_phi_tmp      = nothing
        b.analysis_phi_tmp       = nothing
        b.coeffs_buffer          = nothing
        b.coeffs_buffer_gathered = nothing
        b.coeffs_buffer_pair1    = nothing
        b.coeffs_buffer_pair2    = nothing
        b.coeffs_gathered_pair1  = nothing
        b.coeffs_gathered_pair2  = nothing
        b.pol_rad_coeffs_buffer  = nothing
        b.vector_component_vt    = nothing
        b.vector_component_vp    = nothing
        b.phi_slice_buffer       = nothing
        b.generic_slice_buffer   = nothing
        b.vector_component_buffer = nothing
    end
end

# ================================================================================
# Utility Functions
# ================================================================================

"""
    __shtns_make_transpose(pair)

Create a PencilArrays transpose plan between two pencil configurations.
Used internally to set up efficient data redistribution operations.

# Arguments
- `pair`: A Pair of source and destination Pencil objects (src => dest)

# Returns
- A Transposition object that can be used with `mul!` for data redistribution
"""
@inline function __shtns_make_transpose(pair)
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

Return the fallback number of latitude points (64) used when no grid is
available at call time (e.g., during precompilation).
"""
function get_default_nlat()
    return 64
end

"""
    get_default_nlon() -> Int

Return the fallback number of longitude points (128) used when no grid is
available at call time (e.g., during precompilation). Power of 2 for FFT.
"""
function get_default_nlon()
    return 128
end

# ================================================================================
# SHTnsKit Configuration Structure
# ================================================================================

# Forward declaration for `fields/containers.jl`; allows type hierarchy for
# SHTnsKit configs.
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
- `pencils`: Collection of PencilArrays Pencil objects for different
  data orientations (:theta, :phi, :r, :spec, :mixed)
- `fft_plans`: Precomputed FFTW plans for FFT operations
- `transpose_plans`: Plans for data redistribution between pencils

## Auxiliary Data
- `memory_estimate::String`: Human-readable memory usage estimate
- `l_values::Vector{Int}`: Spherical harmonic degree for each spectral index
- `m_values::Vector{Int}`: Spherical harmonic order for each spectral index
- `theta_grid::Vector{Float64}`: Latitude values (Gauss-Legendre nodes)
- `phi_grid::Vector{Float64}`: Longitude values (uniform spacing)
- `gauss_weights::Vector{Float64}`: Gauss-Legendre quadrature weights

## Internal
- `__buffers::SHTnsBuffers`: Reusable work arrays to reduce allocations

# Usage
```julia
config = create_shtnskit_config(lmax=32, mmax=32, nlat=64, nlon=128)
```
"""
struct SHTnsKitConfig{T<:AbstractFloat,P,FP,TP,B<:SHTnsBuffers} <: AbstractSHTnsConfig
    # SHTnsKit configuration - the underlying transform engine
    sht_config::SHTnsKit.SHTConfig

    # Floating-point precision type for field data
    T::Type{T}

    # Grid parameters defining the resolution
    nlat::Int   # Number of latitude points (Gauss-Legendre)
    nlon::Int   # Number of longitude points (equispaced)
    lmax::Int   # Maximum spherical harmonic degree
    mmax::Int   # Maximum spherical harmonic order
    nlm::Int    # Total number of spectral modes

    # PencilArrays decomposition for MPI parallelization
    # Contains :theta, :phi, :r, :spec, :mixed pencil configurations
    pencils::P

    # FFTW plans for longitude FFTs (keyed by :phi_forward, :phi_backward, etc.)
    fft_plans::FP

    # Transpose plans for switching between pencil orientations
    transpose_plans::TP

    # Human-readable memory estimate string (e.g., "256.5 MB")
    memory_estimate::String

    # Arrays mapping spectral index to (l,m) values for convenience
    l_values::Vector{Int}
    m_values::Vector{Int}

    # Physical grid coordinates
    theta_grid::Vector{Float64}    # Latitude values [-π/2, π/2] (Gauss-Legendre nodes)
    phi_grid::Vector{Float64}      # Longitude values [0, 2π)
    gauss_weights::Vector{Float64} # Quadrature weights for integration

    # Internal typed buffer store to avoid repeated allocations
    __buffers::B
end

"""
    create_shtnskit_config(; lmax, mmax, nlat, nlon, nr, optimize_decomp) -> SHTnsKitConfig

Create and initialize a complete SHTnsKit configuration for spherical harmonic
transforms with MPI parallelization.

# Keyword Arguments
- `lmax::Int`: Maximum spherical harmonic degree (required)
- `mmax::Int=lmax`: Maximum spherical harmonic order (≤ lmax)
- `nlat::Int`: Number of latitude points. Must be ≥ lmax+1 for numerical accuracy.
  Defaults to max(lmax+2, 64)
- `nlon::Int`: Number of longitude points. Must be ≥ 2*mmax+1 for alias-free transforms.
  Powers of 2 are preferred for FFT efficiency.
- `nr::Int`: Number of radial points (required)
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
                               nr::Int,
                               optimize_decomp::Bool=true,
                               device::Symbol=:cpu,
                               T::Type{<:AbstractFloat}=Float64)

    # Step 1: Create base SHTnsKit configuration
    # Uses Gauss-Legendre quadrature for latitude (exact integration up to degree 2*nlat-1)
    # and uniform grid for longitude (FFT-based)
    sht_config =
        device === :cpu ?
        SHTnsKit.create_gauss_config(
            lmax,
            nlat;
            mmax=mmax,
            nlon=nlon,
            norm=:orthonormal,
        ) :
        SHTnsKit.create_gauss_config_gpu(
            lmax,
            nlat;
            mmax=mmax,
            nlon=nlon,
            device=device,
            norm=:orthonormal,
        )

    # Disable precomputed Legendre polynomial tables to avoid version-dependent
    # dimension mismatches between SHTnsKit's table creation and transform code.
    # The on-the-fly Plm computation is reliable and the performance impact is
    # minimal for typical problem sizes.
    SHTnsKit.disable_plm_tables!(sht_config)

    # Step 2: Set up MPI parallelization infrastructure
    comm = get_comm()      # Get MPI communicator (or serial fallback)
    nprocs = get_nprocs()  # Number of MPI processes

    # Step 3 front-loads the whole distributed layout contract. Downstream
    # transform kernels assume these pencils already describe which dimension is
    # local and which transposes are needed to reach an FFT-friendly layout.
    # Step 3: Create pencil decomposition for distributed memory parallelism
    # Pencils define how data is distributed across MPI processes
    pencils = create_pencil_decomposition_shtnskit(
        nlat,
        nlon,
        nr,
        sht_config,
        comm,
        optimize_decomp;
        lmax=lmax,
        mmax=mmax,
    )

    # Step 4: Create FFT plans for longitude (phi) direction transforms
    # These are precomputed FFTW plans for efficiency
    fft_plans = create_pencil_fft_plans(pencils, (nlat, nlon, nr))

    # Step 5: Create transpose plans for data redistribution between pencils
    # Transposes are needed when switching which dimension is local
    transpose_plans = create_shtnskit_transpose_plans(pencils)

    # Estimate memory usage for user information
    field_count = estimate_field_count()
    memory_mb = estimate_memory_usage_shtnskit(nlat, nlon, nr, lmax, field_count, Float64)
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
    # Ordering follows the SHTnsKit convention: m-major (m varies slowest, l varies fastest)
    # m=0: l=0,1,...,lmax; m=1: l=1,2,...,lmax; ...; m=mmax: l=mmax,...,lmax
    l_vals = Vector{Int}(undef, nlm)
    m_vals = Vector{Int}(undef, nlm)
    idx = 1
    for m in 0:mmax
        for l in m:lmax
            if idx <= nlm
                l_vals[idx] = l
                m_vals[idx] = m
            end
            idx += 1
        end
    end

    if get_rank() == 0
        print_shtnskit_config_summary(nlat, nlon, nr, lmax, mmax, nlm, nprocs, memory_estimate)
    end

    # Step 8: Initialize typed buffer store with device-aware transform metadata
    buffers = SHTnsBuffers()
    transform_device = SHTnsKit.get_config_device(sht_config)
    buffers.transform_device = transform_device

    # CPU and GPU diverge here on purpose: CPU keeps the eager transform plan,
    # while both CPU and GPU leave the large output buffers lazy so configs do
    # not pay resident-memory cost for transform paths they never touch. GPU
    # configs still skip all CPU-only plan/cache entries; the solver's GPU path
    # dispatches through the non-plan transform entry points instead.
    if transform_device == :cpu
        # Create SHTPlan for allocation-free per-radial-level transforms
        # This pre-allocates all working arrays (Legendre, FFT, scratch) once
        try
            sht_plan = SHTnsKit.SHTPlan(sht_config)
            buffers.sht_plan = sht_plan
            if get_rank() == 0
                @info "SHTnsKit SHTPlan created for allocation-free transforms"
            end
        catch e
            if get_rank() == 0
                @warn "Could not create SHTPlan (falling back to allocating transforms): $e"
            end
        end

        if SHTNSKIT_USE_SCRATCH_BUFFERS
            try
                buffers.spatial_scratch = SHTnsKit.scratch_spatial(sht_config, Float64)
                buffers.fft_scratch = SHTnsKit.scratch_fft(sht_config, ComplexF64)
            catch e
                if get_rank() == 0
                    @debug "Could not allocate SHTnsKit scratch buffers: $e"
                end
            end
        end
    end

    return SHTnsKitConfig(
        sht_config, T, nlat, nlon, lmax, mmax, nlm,
        pencils, fft_plans, transpose_plans, memory_estimate,
        l_vals, m_vals, theta_grid, phi_grid, gauss_weights,
        buffers
    )
end

"""
    create_pencil_decomposition_shtnskit(nlat, nlon, nr, sht_config, comm, optimize; lmax, mmax)

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

## Spectral Space Pencil ((lmax+1) × (mmax+1) × nr)
The spectral pencil stores spherical harmonic coefficients on a rectangular
degree/order grid. Invalid `(l,m)` slots such as `m > l` are left unmapped.

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
                                             comm, optimize::Bool=true;
                                             lmax::Int,
                                             mmax::Int)
    nprocs = MPI.Comm_size(comm)

    # Determine optimal 2D process grid for theta-phi parallelization.
    # Spectral storage uses a real (l,m,r) grid, so the same 2D topology can
    # distribute spectral modes without splitting a dummy axis.
    if optimize && nprocs > 1
        proc_dims = optimize_process_topology_shtnskit(nprocs, nlat, nlon, lmax, mmax)
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

    # Create spectral space pencil.
    # Distribute degree/order axes and keep radial (dim 3) local on each rank.
    # decomp_dims must be a 2-tuple to match MPITopology{2}
    # This matches DD_2DCODE where each rank has full radial profiles for its subset of (l,m) modes
    spec_dims = spectral_mode_grid_dims(lmax, mmax, nr)
    pencil_spec = Pencil(topology, spec_dims, (1, 2))

    # Compatibility alias for older call sites. Mixed spectral storage must obey
    # the same rectangular (l, m, r) ownership contract as the spectral pencil.

    # Return named tuple with both ASCII and Unicode names for convenience
    return (; theta=pencil_theta,
            θ=pencil_theta,      # Unicode alias
            phi=pencil_phi,
            φ=pencil_phi,        # Unicode alias
            r=pencil_r,
            spec=pencil_spec,
            mixed=pencil_spec)
end

"""
    optimize_process_topology_shtnskit(nprocs, nlat, nlon, lmax, mmax) -> Tuple{Int,Int}

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
- `lmax`: Maximum spherical harmonic degree
- `mmax`: Maximum spherical harmonic order

# Returns
- `(p_theta, p_phi)`: Optimal process grid dimensions
"""
function optimize_process_topology_shtnskit(nprocs::Int, nlat::Int, nlon::Int,
                                            lmax::Int, mmax::Int)
    physical_dims = (nlat, nlon, 1)
    spectral_dims = spectral_mode_grid_dims(lmax, mmax, 1)
    return optimize_process_topology(nprocs, physical_dims, spectral_dims)
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

        # NOTE: No FFT plans are created for the theta pencil.
        # The theta pencil has dim 1 (theta) local and dim 2 (phi) distributed.
        # Creating FFT plans along dim 2 would operate on a distributed dimension,
        # which is incorrect.  SH transforms handle the theta direction via
        # Legendre transforms (not FFTs), so theta-pencil FFT plans are not needed.

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
            transpose_plans[:theta_to_phi] = __shtns_make_transpose(pencils.theta => pencils.phi)
            transpose_plans[:phi_to_theta] = __shtns_make_transpose(pencils.phi => pencils.theta)
        catch e
            if get_rank() == 0
                @debug "Could not create theta<->phi transpose: $e"
            end
        end
    end

    # Theta ↔ R transposes (for switching to radial operations)
    if haskey(pencils, :r) && haskey(pencils, :theta)
        try
            transpose_plans[:theta_to_r] = __shtns_make_transpose(pencils.theta => pencils.r)
            transpose_plans[:r_to_theta] = __shtns_make_transpose(pencils.r => pencils.theta)
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
            transpose_plans[:phi_to_r] = __shtns_make_transpose(pencils.phi => pencils.r)
            transpose_plans[:r_to_phi] = __shtns_make_transpose(pencils.r => pencils.phi)
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
    estimate_memory_usage_shtnskit(nlat, nlon, nr, lmax, field_count, T)

Estimate memory usage for SHTnsKit-based transforms with PencilArrays.

# Arguments
- `nlat`: Number of latitude points
- `nlon`: Number of longitude points
- `nr`: Number of radial points
- `lmax`: Maximum spherical harmonic degree
- `field_count`: Number of fields to estimate for
- `T`: Element type (e.g., Float64)
"""
function estimate_memory_usage_shtnskit(nlat::Int, nlon::Int, nr::Int, lmax::Int,
                                       field_count::Int, ::Type{T};
                                       mmax::Int=lmax) where T

    # Physical grid memory per process (distributed)
    physical_memory_per_process = (nlat * nlon * nr * sizeof(T)) / get_nprocs()

    # Spectral memory (approximate)
    nlm = SHTnsKit.nlm_calc(lmax, mmax, 1)
    spectral_memory_per_process = (nlm * nr * sizeof(ComplexF64) * 2) / get_nprocs()

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
    print_shtnskit_config_summary(nlat, nlon, nr, lmax, mmax, nlm, nprocs, memory_estimate)

Print configuration summary for SHTnsKit setup.
"""
function print_shtnskit_config_summary(nlat, nlon, nr, lmax, mmax, nlm, nprocs, memory_estimate)
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
    println("║   Physical grid:    $(lpad(nlat,4)) × $(lpad(nlon,4)) × $(lpad(nr,4))         ║")
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
