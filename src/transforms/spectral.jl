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
# SHTnsKit v2.0+ Feature Flags
# ================================================================================
# These flags indicate which v2.0+ features are available and should be used

const SHTNSKIT_USE_DISTRIBUTED = true      # Use dist_analysis/dist_synthesis
const SHTNSKIT_USE_QST = true              # Use synthesis_qst/analysis_qst for 3D vectors
const SHTNSKIT_USE_SCRATCH_BUFFERS = true  # Use scratch_spatial/scratch_fft helpers

abstract type AbstractTransformWorkspace end

# ================================================================================
# SHTnsBuffers — typed replacement for _buffer_cache::Dict{Symbol,Any}
# ================================================================================

"""
    SHTnsBuffers

Concrete mutable struct holding all reusable work arrays for SHTnsKit transforms.
Replaces the previous `Dict{Symbol,Any}` buffer cache with named, type-stable fields.

Fields that are `nothing` at construction time are lazily allocated on first use.
"""
mutable struct SHTnsBuffers
    # Set eagerly at config creation (CPU only)
    sht_plan::Union{SHTnsKit.SHTPlan, Nothing}

    # Scalar synthesis / analysis output buffers
    synth_out::Union{Matrix{Float64}, Nothing}
    anal_out::Union{Matrix{ComplexF64}, Nothing}

    # Vector synthesis output buffers (tangential components)
    vt_out::Union{Matrix{Float64}, Nothing}
    vp_out::Union{Matrix{Float64}, Nothing}

    # Vector analysis output buffers (toroidal/poloidal coefficients)
    slm_out::Union{Matrix{ComplexF64}, Nothing}
    tlm_out::Union{Matrix{ComplexF64}, Nothing}

    # Temporary phi-pencil arrays for transpose-based synthesis/analysis
    synthesis_phi_tmp::Union{PencilArray, Nothing}
    analysis_phi_tmp::Union{PencilArray, Nothing}

    # Coefficient extraction buffers (scalar)
    coeffs_buffer::Union{Matrix{ComplexF64}, Nothing}
    coeffs_buffer_gathered::Union{Matrix{ComplexF64}, Nothing}

    # Coefficient extraction buffers (paired, for vector transforms)
    coeffs_buffer_pair1::Union{Matrix{ComplexF64}, Nothing}
    coeffs_buffer_pair2::Union{Matrix{ComplexF64}, Nothing}
    coeffs_gathered_pair1::Union{Matrix{ComplexF64}, Nothing}
    coeffs_gathered_pair2::Union{Matrix{ComplexF64}, Nothing}

    # Radial-component poloidal synthesis buffer
    pol_rad_coeffs_buffer::Union{Matrix{ComplexF64}, Nothing}

    # Vector component physical-space buffers used in vector analysis
    vector_component_vt::Union{Matrix{Float64}, Nothing}
    vector_component_vp::Union{Matrix{Float64}, Nothing}

    # Generic physical-slice extraction buffers
    phi_slice_buffer::Union{Matrix{Float64}, Nothing}
    generic_slice_buffer::Union{Matrix{Float64}, Nothing}
    vector_component_buffer::Union{Matrix{Float64}, Nothing}

    # Local storage-coordinate -> global linear spectral mode map
    local_spectral_lm_map::Union{Matrix{Int}, Nothing}
    local_spectral_slot_lookup::Union{Vector{CartesianIndex{2}}, Nothing}
    local_spectral_mode_indices::Union{Vector{Int}, Nothing}

    # Solver-level transform workspace (set by solver/backend.jl)
    solver_transform_workspace::Union{AbstractTransformWorkspace, Nothing}

    # Device metadata (set at config creation)
    transform_device::Union{AbstractArchitecture, Symbol, Nothing}

    # SHTnsKit scratch buffers (CPU only, optional)
    spatial_scratch::Union{AbstractArray, Nothing}
    fft_scratch::Union{AbstractArray, Nothing}

    # MIE vector-transform spheroidal scalar buffers
    mie_spheroidal_real::Union{AbstractArray, Nothing}
    mie_spheroidal_imag::Union{AbstractArray, Nothing}

    # MIE vector-transform poloidal coefficient buffers (radial-component path)
    mie_pol_coeffs_buffer::Union{AbstractArray, Nothing}
    mie_pol_coeffs_gathered::Union{AbstractArray, Nothing}

    # Phase-3 DistTransposePlan transform scratch (per-config; replaces the old
    # module-global IdDict{Any,Any} caches). Lazily built on first transform.
    disttranspose_plan::Union{Any, Nothing}        # SHTnsKit.DistTransposePlan
    disttranspose_scratch::Union{Any, Nothing}     # NamedTuple of PencilArrays (config-dependent type)
    disttranspose_mbridge::Union{Any, Nothing}     # _MBridge (concrete); built lazily on first transform
    p3_scalar_scratch::Union{Any, Nothing}         # NamedTuple (Alm/fspatial/solve)
    p3_vector_scratch::Union{Any, Nothing}         # NamedTuple (Slm/Tlm/Vr_alm/Vt/Vp/Vr/solve)
    solenoidal_prof::Union{Vector{Float64}, Nothing}  # cached nr-scratch for solenoidal coupling
    solenoidal_dpr::Union{Vector{Float64}, Nothing}
end

"""
    SHTnsBuffers()

Construct a fully-uninitialized `SHTnsBuffers` with all fields set to `nothing`.
"""
function SHTnsBuffers()
    SHTnsBuffers(
        nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing, nothing,
        nothing, nothing
    )
end

# ================================================================================
# Field map: legacy Symbol key → SHTnsBuffers field name
# ================================================================================
# Used to define typed accessors for call sites that still pass a Symbol key.
# Keys not in this map (e.g. :sht_plan, :solver_transform_workspace, :transform_device,
# :spatial_scratch, :fft_scratch) are set directly and not routed through this function.

const _BUFFERS_FIELD_MAP = Dict{Symbol, Symbol}(
    :synth_out => :synth_out,
    :anal_out => :anal_out,
    :vt_out => :vt_out,
    :vp_out => :vp_out,
    :slm_out => :slm_out,
    :tlm_out => :tlm_out,
    :synthesis_phi_tmp => :synthesis_phi_tmp,
    :analysis_phi_tmp => :analysis_phi_tmp,
    :coeffs_buffer => :coeffs_buffer,
    :coeffs_buffer_gathered => :coeffs_buffer_gathered,
    :coeffs_buffer_pair1 => :coeffs_buffer_pair1,
    :coeffs_buffer_pair2 => :coeffs_buffer_pair2,
    :coeffs_gathered_pair1 => :coeffs_gathered_pair1,
    :coeffs_gathered_pair2 => :coeffs_gathered_pair2,
    :pol_rad_coeffs_buffer => :pol_rad_coeffs_buffer,
    :vector_component_buffer_vt => :vector_component_vt,
    :vector_component_buffer_vp => :vector_component_vp,
    :phi_slice_buffer => :phi_slice_buffer,
    :generic_slice_buffer => :generic_slice_buffer,
    :vector_component_buffer => :vector_component_buffer,
    :mie_spheroidal_real => :mie_spheroidal_real,
    :mie_spheroidal_imag => :mie_spheroidal_imag,
    :mie_pol_coeffs_buffer => :mie_pol_coeffs_buffer,
    :mie_pol_coeffs_gathered => :mie_pol_coeffs_gathered
)

@inline function _shtns_buffer_field(::Val{key}) where {key}
    error("get_cached_buffer!: unknown buffer key $(repr(key)). Add it to SHTnsBuffers and _BUFFERS_FIELD_MAP.")
end

for (key, field) in _BUFFERS_FIELD_MAP
    @eval @inline _shtns_buffer_field(::Val{$(QuoteNode(key))}) = Val{$(QuoteNode(field))}()
end

# ================================================================================
# Thread-Safe Buffer Cache Access
# ================================================================================
# The buffer cache is shared across threads and needs synchronization to avoid
# race conditions when multiple threads access or create buffers simultaneously.

"""
    _BUFFER_CACHE_LOCK

Global ReentrantLock for thread-safe access to SHTnsKitConfig buffer caches.
All access to config._buffers should be protected by this lock.
"""
const _BUFFER_CACHE_LOCK = ReentrantLock()

"""
    get_cached_buffer!(create_func, config, key::Symbol)
    get_cached_buffer!(config, key::Symbol) do ... end

Thread-safe accessor for buffer cache. Returns existing buffer if present,
otherwise creates a new one using `create_func()` and caches it.

Note: The function parameter comes FIRST to support Julia's `do` block syntax.
When using `do` block, Julia desugars it to pass the closure as the first argument.

# Arguments
- `create_func`: Zero-argument function to create buffer if not cached
- `config`: SHTnsKitConfig object containing `_buffers::SHTnsBuffers`
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

@inline function get_cached_buffer!(create_func::F, config, ::Val{key}) where {F, key}
    return _get_cached_buffer_field!(create_func, config, _shtns_buffer_field(Val(key)))
end

@inline function _get_cached_buffer_field!(create_func::F, config, ::Val{field}) where {
        F, field}
    lock(_BUFFER_CACHE_LOCK) do
        b = config._buffers
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
    lock(_BUFFER_CACHE_LOCK) do
        b = config._buffers
        b.synth_out = nothing
        b.anal_out = nothing
        b.vt_out = nothing
        b.vp_out = nothing
        b.slm_out = nothing
        b.tlm_out = nothing
        b.synthesis_phi_tmp = nothing
        b.analysis_phi_tmp = nothing
        b.coeffs_buffer = nothing
        b.coeffs_buffer_gathered = nothing
        b.coeffs_buffer_pair1 = nothing
        b.coeffs_buffer_pair2 = nothing
        b.coeffs_gathered_pair1 = nothing
        b.coeffs_gathered_pair2 = nothing
        b.pol_rad_coeffs_buffer = nothing
        b.vector_component_vt = nothing
        b.vector_component_vp = nothing
        b.phi_slice_buffer = nothing
        b.generic_slice_buffer = nothing
        b.vector_component_buffer = nothing
    end
    # Prune this config's entries from the Phase-3 transform caches (separate lock).
    _clear_p3_transform_caches!(config)
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
- `_buffers::SHTnsBuffers`: Reusable work arrays to reduce allocations

# Usage
```julia
config = create_shtnskit_config(lmax=32, mmax=32, nlat=64, nlon=128)
```
"""
mutable struct SHTnsKitConfig{T <: AbstractFloat, P, FP, TP, B <: SHTnsBuffers} <:
       AbstractSHTnsConfig
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
    _buffers::B
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
- `optimize_decomp::Bool=true`: Retained for API compatibility; the current
  topology is selected by `GEODYNAMO_PROC_GRID` and this flag has no effect
- `device::Symbol=:cpu`: Execution device for transforms - `:cpu`, `:gpu`, `:cuda`,
  or `:auto` (resolves to a GPU when one is functional, otherwise `:cpu`)
- `T::Type{<:AbstractFloat}=Float64`: Floating-point element type
- `verbose::Bool=false`: Print a rank-zero configuration summary

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
function create_shtnskit_config(; lmax::Int, mmax::Int = lmax,
        nlat::Int = max(lmax+2, get_default_nlat()),
        nlon::Int = max(2*lmax+1, 4, get_default_nlon()),
        nr::Int,
        optimize_decomp::Bool = true,
        device::Symbol = :cpu,
        T::Type{<:AbstractFloat} = Float64,
        verbose::Bool = false)

    # Step 1: Create base SHTnsKit configuration
    # Uses Gauss-Legendre quadrature for latitude (exact integration up to degree 2*nlat-1)
    # and uniform grid for longitude (FFT-based)
    # `:auto` picks a GPU when one is actually usable and CPU otherwise, matching
    # what the v1 path did via SHTnsKit's own device selection.
    if device === :auto
        device = gpu_functional() ? :cuda : :cpu
    end
    device in (:cpu, :gpu, :cuda) || throw(ArgumentError(
        "device = $(repr(device)) must be :cpu, :gpu, :cuda, or :auto"))
    # SHTnsKit v2 configurations are device-neutral.  The execution device is
    # selected by each transform call and retained separately in our buffers.
    sht_config = SHTnsKit.create_gauss_config(
        lmax,
        nlat;
        mmax = mmax,
        nlon = nlon,
        norm = :orthonormal
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
        lmax = lmax,
        mmax = mmax
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

    # Physical grids read straight from the SHTnsKit Gauss configuration so the
    # stored nodes/weights MATCH the transform grid: Gauss-Legendre colatitude
    # θ ∈ [0, π] (non-uniform), uniform longitude φ ∈ [0, 2π), and the Gauss
    # quadrature weights (Σw = 2). SHTnsKit v2 advertises θ/φ/w as properties of
    # `SHTConfig` (they appear in `Base.propertynames`), so this is public API
    # rather than the internal `_grid` field the v1 code had to reach into. Warn
    # LOUDLY if they are ever unavailable (never silently substitute a uniform
    # grid, which corrupts Coriolis terms and all θ-quadrature).
    _sht_grid = try
        (θ = sht_config.θ, φ = sht_config.φ, w = sht_config.w)
    catch
        nothing
    end
    if _sht_grid !== nothing
        theta_grid    = Vector{Float64}(_sht_grid.θ)
        phi_grid      = Vector{Float64}(_sht_grid.φ)
        gauss_weights = Vector{Float64}(_sht_grid.w)
    else
        @warn "SHTnsKit Gauss grid (`θ`/`φ`/`w`) unavailable; falling back to a UNIFORM θ grid + unit weights. Physical-space integration (energy/Nusselt) and Coriolis terms will be INACCURATE — check SHTnsKit version compatibility." maxlog = 1
        theta_grid    = range(0, stop = pi, length = nlat) |> collect |> Vector{Float64}
        phi_grid      = range(0, stop = 2pi, length = nlon + 1)[1:(end - 1)] |> collect |> Vector{Float64}
        gauss_weights = ones(Float64, nlat)
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

    if verbose && get_rank() == 0
        print_shtnskit_config_summary(
            nlat, nlon, nr, lmax, mmax, nlm, nprocs, memory_estimate)
    end

    # Step 8: Initialize typed buffer store with device-aware transform metadata
    buffers = SHTnsBuffers()
    transform_device = device
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
Memo table for [`create_pencil_decomposition_shtnskit`](@ref), keyed by every input
the decomposition actually depends on.

Each decomposition allocates FOUR MPI communicators that are never freed — two
`MPITopology` Cartesian comms (the r×θ grid and the 1D θ prototype) plus the two
`make_subcomms` splits — and MPICH's default ceiling is 2048 per process. Rebuilding
one per `SolverState` walked the test suite into that ceiling, where it surfaced as
`MPI_Cart_create failed … Too many communicators` in whichever unrelated file
happened to run next.

Sharing is safe: `Pencil` and `MPITopology` are immutable descriptors, and
`PencilArray`s allocate their own data buffers, so two configs on the same grid can
hold the same decomposition without aliasing any mutable state.
"""
const _PENCIL_DECOMP_CACHE = Dict{Any, Any}()
const _PENCIL_DECOMP_LOCK = ReentrantLock()

"""
    clear_pencil_decomposition_cache!() -> Int

Drop every memoized pencil decomposition and return how many were dropped.

The communicators they hold are NOT freed — `MPI_Comm_free` is collective and
cannot be called safely from here — so this only forces subsequent configs to build
fresh topologies. Intended for tests that need a distinct decomposition object.
"""
function clear_pencil_decomposition_cache!()
    lock(_PENCIL_DECOMP_LOCK) do
        n = length(_PENCIL_DECOMP_CACHE)
        empty!(_PENCIL_DECOMP_CACHE)
        return n
    end
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
- `optimize`: accepted for API compatibility but IGNORED in Phase 1 (the 1D-θ grid
  is fixed `(nprocs, 1)`); Phase 2 (r×θ) will honor it via
  `optimize_process_topology_shtnskit`.

# Returns
NamedTuple with pencil configurations:
(:theta, :θ, :phi, :φ, :r, :spec, :mixed, :theta_phys).
`theta_phys` is a 2D `(nlat, nlon)` θ-distributed / φ-local prototype pencil whose
θ-split matches `pencils.r`, useful for structural invariant checks.
"""
function create_pencil_decomposition_shtnskit(nlat::Int, nlon::Int, nr::Int,
        sht_config::SHTnsKit.SHTConfig,
        comm, optimize::Bool = true;
        lmax::Int,
        mmax::Int)
    # The body below reads ONLY these inputs (plus read_proc_grid, which is a pure
    # function of nprocs and GEODYNAMO_PROC_GRID), so they fully determine the result.
    # `comm` is part of the key by identity: a duplicated communicator is a genuinely
    # different topology and must not share.
    # `optimize` is intentionally absent: it is accepted for API compatibility
    # but does not affect the Phase-2 topology. Including it would create an
    # identical decomposition and four additional never-freed MPI communicators.
    cache_key = (objectid(comm), MPI.Comm_size(comm), nlat, nlon, nr, lmax, mmax,
        get(ENV, "GEODYNAMO_PROC_GRID", ""))
    cached = lock(_PENCIL_DECOMP_LOCK) do
        get(_PENCIL_DECOMP_CACHE, cache_key, nothing)
    end
    cached === nothing || return cached

    nprocs = MPI.Comm_size(comm)

    # Phase 2 (r×θ 2D topology): read the process grid from GEODYNAMO_PROC_GRID.
    # At nprocs==1 this always returns (1,1) without requiring the env var.
    # The grid is (θ_ranks, r_ranks): θ is axis-1 of the MPI topology (distributed
    # across the first process-grid dimension), r is axis-2.
    θ_ranks, r_ranks = read_proc_grid(nprocs)
    proc_dims = (θ_ranks, r_ranks)

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

    # R pencil: Phase 2 — θ(dim1) distributed over θ_ranks, r(dim3) distributed over
    # r_ranks, φ(dim2) LOCAL.  decomp (1,3) maps dim1→axis1(θ) and dim3→axis2(r).
    # At (θ_ranks,r_ranks)==(1,1) both dimensions are trivially local → behaviour
    # identical to the old (1,2) single-rank case.
    pencil_r = Pencil(topology, dims, (1, 3))

    # Derive the sub-communicators from pencil_r's ACTUAL distribution (robust for any
    # grid / rank ordering). θ_comm = ranks sharing an r-slab that split θ (the SH
    # transform group — theta_phys + per-level dist_* run here); r_comm = ranks sharing
    # a θ-slab that split r (aligned with the r↔lm transpose).
    θ_comm, r_comm = make_subcomms(comm, pencil_r)

    # Create spectral space pencil.
    # Distribute degree/order axes and keep radial (dim 3) local on each rank.
    # decomp_dims must be a 2-tuple to match MPITopology{2}.
    #
    # Phase-3 layout (DistTransposePlan / spec_solve partition). The decomp tuple
    # (d1,d2) maps storage-dim d1 → topology axis-1 (θ_ranks / θ_comm) and storage-dim
    # d2 → topology axis-2 (r_ranks / r_comm).  We want m (storage dim-2) distributed
    # over θ_comm and l (storage dim-1) distributed over r_comm — i.e. exactly the
    # ownership produced by `to_spec_solve` (l-dist/r_comm, m-dist/θ_comm, r-local).
    # That is decomp (2,1): dim2(m)→axis1(θ), dim1(l)→axis2(r).
    #
    # The generic (l,m)->packed-index map (build_local_spectral_lm_map) reads
    # range_local(spec,1/2) directly, so the radial-solve mode indexing adapts to this
    # (l,m) ownership automatically.  At (θ,r)==(1,1) both axes are trivially local so
    # the layout is identical to the historical single-rank case.  NOTE: the m-axis is
    # `mmax+1` columns split evenly over θ_comm; this does NOT in general match the
    # DistTransposePlan's nbin-based m-bin split (on dealiased grids), so the scalar
    # transform performs an explicit m-axis redistribution over θ_comm (see
    # physics/nonlinear.jl) rather than a bare local copy.
    spec_dims = spectral_mode_grid_dims(lmax, mmax, nr)
    pencil_spec = Pencil(topology, spec_dims, (2, 1))

    # Compatibility alias for older call sites. Mixed spectral storage must obey
    # the same rectangular (l, m, r) ownership contract as the spectral pencil.

    # 2D (nlat,nlon) θ-distributed / φ-local prototype for SHTnsKit's per-level
    # dist_synthesis/dist_analysis (prototype_θφ). Built on θ_comm — the group of ranks
    # that share an r-slab and split θ — so a 1D Pencil of size θ_ranks over θ_comm gives
    # each rank exactly its pencil_r θ-slab (verified: theta_phys θ-split == pencils.r
    # θ-split per rank). At (θ_ranks,r_ranks)==(1,1) θ_comm has size 1 → trivially local.
    topo1d_θ = TopoCtor(θ_comm, (θ_ranks,))
    pencil_theta_phys = Pencil(topo1d_θ, (nlat, nlon), (1,))

    # Return named tuple with both ASCII and Unicode names for convenience.
    # θ_comm and r_comm are included so callers can form sub-collective operations
    # (e.g. per-r-group Allreduce on the θ-subcomm) without passing comm+grid dims
    # separately.
    decomposition = (; theta = pencil_theta,
        θ = pencil_theta,      # Unicode alias
        phi = pencil_phi,
        φ = pencil_phi,        # Unicode alias
        r = pencil_r,
        spec = pencil_spec,
        mixed = pencil_spec,
        theta_phys = pencil_theta_phys,
        θ_comm = θ_comm,
        r_comm = r_comm)

    # Memoize so an identical grid reuses these communicators instead of allocating
    # four more. Every rank computes the same key from the same inputs, so the hit/miss
    # decision stays collectively consistent — important, because a miss runs
    # MPI_Cart_create and Comm_split, which are collective.
    lock(_PENCIL_DECOMP_LOCK) do
        get!(_PENCIL_DECOMP_CACHE, cache_key, decomposition)
    end
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
function create_pencil_fft_plans(pencils, dims::Tuple{Int, Int, Int})
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
            transpose_plans[:theta_to_phi] = _shtns_make_transpose(pencils.theta =>
                pencils.phi)
            transpose_plans[:phi_to_theta] = _shtns_make_transpose(pencils.phi =>
                pencils.theta)
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
            if haskey(transpose_plans, :phi_to_theta) &&
               haskey(transpose_plans, :theta_to_r)
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
        mmax::Int = lmax) where {T}

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
function print_shtnskit_config_summary(
        nlat, nlon, nr, lmax, mmax, nlm, nprocs, memory_estimate)
    # Get version info for feature flags
    version = try
        string(pkgversion(SHTnsKit))
    catch
        "≥2.0.2"
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
    println("║ v2.0+ Features:                                       ║")
    println("║   Distributed transforms: $(SHTNSKIT_USE_DISTRIBUTED ? "enabled " : "disabled")                ║")
    println("║   QST vector transforms:  $(SHTNSKIT_USE_QST ? "enabled " : "disabled")                ║")
    println("║   Scratch buffers:        $(SHTNSKIT_USE_SCRATCH_BUFFERS ? "enabled " : "disabled")                ║")
    println("╚═══════════════════════════════════════════════════════╝")
end
