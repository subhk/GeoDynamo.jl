# ERK2 common: boundary mode-value helper, const aliases, compat constructors, phi functions, diagnostics toggles.

# `SolverERK2BoundarySide` and `SolverERK2BoundarySpec` are defined in
# `solver/state.jl` (next to the other timestep-cache types) so they can be
# stored in `TimestepCaches.erk2_boundary_specs`. The helpers that build and
# operate on them live here.

function with_boundary_mode_values(
        spec::SolverERK2BoundarySpec{T},
        inner_real::Union{Nothing, AbstractVector{T}},
        outer_real::Union{Nothing, AbstractVector{T}},
        inner_imag::Union{Nothing, AbstractVector{T}} = nothing,
        outer_imag::Union{Nothing, AbstractVector{T}} = nothing
) where {T}
    # Keep the same derivative stencils and BC types, but attach the actual
    # mode-indexed endpoint values used during stage/final enforcement.
    return SolverERK2BoundarySpec{T}(
        spec.inner,
        spec.outer,
        inner_real,
        outer_real,
        inner_imag,
        outer_imag
    )
end

const ERK2Cache = ERK2StageCache
const Phi2ConditioningMonitor = SolverPhi2ConditioningMonitor
const PHI2_MONITOR = SOLVER_PHI2_MONITOR
const ERK2BoundarySide = SolverERK2BoundarySide
const ERK2BoundarySpec = SolverERK2BoundarySpec
const ERK2InfluenceMatrix = ERK2InfluenceOp
const ERK2FieldBuffers = SolverERK2FieldBuffers

"""
    GeoDynamo.ERK2Cache{T}(...)

Compatibility constructor for the public ERK2 cache type.

The solver now stores stage caches as `ERK2StageCache`; this constructor keeps
older call sites that instantiate `GeoDynamo.ERK2Cache` directly working.
"""
function GeoDynamo.ERK2Cache{T}(
        dt::Float64,
        l_values::Vector{Int},
        E_half::Vector{Matrix{T}},
        E_full::Vector{Matrix{T}},
        phi1_half::Vector{Matrix{T}},
        phi1_full::Vector{Matrix{T}},
        phi2_full::Vector{Matrix{T}},
        use_krylov::Bool,
        krylov_m::Int,
        krylov_tol::Float64,
        mpi_consistent::Bool
) where {T}
    nr = isempty(E_half) ? 0 : size(E_half[1], 1)
    return ERK2StageCache{T}(
        dt,
        NaN,
        nr,
        l_values,
        E_half,
        E_full,
        phi1_half,
        phi1_full,
        phi2_full,
        use_krylov,
        krylov_m,
        krylov_tol,
        mpi_consistent
    )
end

"""
    GeoDynamo.ERK2Cache(args...)

Legacy constructor that builds a `Float64` ERK2 cache when no element type is
specified explicitly.
"""
GeoDynamo.ERK2Cache(args...) = GeoDynamo.ERK2Cache{Float64}(args...)

@inline compat_solver_erk2_cache(cache::ERK2StageCache{T}) where {T} = cache
@inline compat_old_erk2_cache(cache::ERK2StageCache{T}) where {T} = cache

"""
    compat_normalize_old_erk2_cache_entry(entry)

Extract an `ERK2StageCache` from legacy cache bundle entries.

Older code sometimes stores cache entries directly and sometimes under a
`:cache` key; unsupported entries return `nothing`.
"""
function compat_normalize_old_erk2_cache_entry(entry)
    if entry isa ERK2StageCache
        return compat_old_erk2_cache(entry)
    elseif entry isa Dict
        return compat_normalize_old_erk2_cache_entry(get(entry, :cache, nothing))
    elseif entry === nothing
        return nothing
    else
        return nothing
    end
end

"""
    GeoDynamo.set_erk2_diagnostics_interval!(interval)

Set how often ERK2 stage residual diagnostics are reported.
"""
function GeoDynamo.set_erk2_diagnostics_interval!(interval::Int)
    interval <= 0 && error("ERK2 diagnostics interval must be positive, got $interval")
    SOLVER_SHARED_ERK2_DIAGNOSTICS_INTERVAL[] = interval
    set_erk2_diagnostics!(SOLVER_SHARED_ERK2_DIAGNOSTICS_ENABLED[], interval)
    return interval
end

"""
    GeoDynamo.enable_erk2_diagnostics!(; interval=...)

Enable ERK2 residual diagnostics and optionally update the reporting interval.
"""
function GeoDynamo.enable_erk2_diagnostics!(;
        interval::Int = SOLVER_SHARED_ERK2_DIAGNOSTICS_INTERVAL[])
    GeoDynamo.set_erk2_diagnostics_interval!(interval)
    SOLVER_SHARED_ERK2_DIAGNOSTICS_ENABLED[] = true
    set_erk2_diagnostics!(true, interval)
    return nothing
end

"""
    GeoDynamo.disable_erk2_diagnostics!()

Disable ERK2 residual diagnostics without changing the configured interval.
"""
function GeoDynamo.disable_erk2_diagnostics!()
    SOLVER_SHARED_ERK2_DIAGNOSTICS_ENABLED[] = false
    set_erk2_diagnostics!(false, SOLVER_SHARED_ERK2_DIAGNOSTICS_INTERVAL[])
    return nothing
end

"""
    GeoDynamo.erk2_diagnostics_enabled()

Return whether ERK2 residual diagnostics are currently enabled.
"""
GeoDynamo.erk2_diagnostics_enabled() = SOLVER_SHARED_ERK2_DIAGNOSTICS_ENABLED[]

"""
    GeoDynamo.erk2_diagnostics_interval()

Return the configured ERK2 residual diagnostics interval.
"""
GeoDynamo.erk2_diagnostics_interval() = SOLVER_SHARED_ERK2_DIAGNOSTICS_INTERVAL[]

"""
    GeoDynamo.compute_phi1_function(A, expA)

Public compatibility wrapper for the solver-local phi1 matrix-function helper.
"""
function GeoDynamo.compute_phi1_function(A::Matrix{T}, expA::Matrix{T}) where {T}
    solver_compute_phi1_function(A, expA)
end

"""
    GeoDynamo.compute_phi2_function(A, expA; l=0)

Public compatibility wrapper for the solver-local phi2 matrix-function helper.
"""
function GeoDynamo.compute_phi2_function(A::Matrix{T}, expA::Matrix{T}; l::Int = 0) where {T}
    solver_compute_phi2_function(A, expA; l = l)
end

"""
    GeoDynamo.reset_phi2_monitor!()

Clear accumulated phi2 conditioning diagnostics.
"""
GeoDynamo.reset_phi2_monitor!() = reset_solver_phi2_monitor!()

"""
    GeoDynamo.report_phi2_conditioning(step; interval=100)

Report phi2 conditioning diagnostics on the requested interval.
"""
function GeoDynamo.report_phi2_conditioning(step::Int; interval::Int = 100)
    report_solver_phi2_conditioning(step; interval = interval)
end
