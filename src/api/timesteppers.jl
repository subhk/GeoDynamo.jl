abstract type AbstractTimestepper end

"""
    CNAB2(; theta=0.5)

Crank-Nicolson Adams-Bashforth second-order timestepper.
"""
struct CNAB2 <: AbstractTimestepper
    implicit_theta::Float64
end

function CNAB2(;
        theta::Union{Real, Nothing} = nothing,
        implicit_theta::Union{Real, Nothing} = nothing
)
    if !isnothing(theta) && !isnothing(implicit_theta) &&
       Float64(theta) != Float64(implicit_theta)
        throw(ArgumentError("CNAB2 received both theta and implicit_theta with different values"))
    end
    value = something(theta, something(implicit_theta, 0.5))
    return CNAB2(Float64(value))
end

"""
    ExponentialAdamsBashforth2(; krylov_dimension=20, tolerance=1e-8)

Experimental Exponential Adams-Bashforth second-order descriptor.

Not currently supported by `Simulation`: the solver's velocity-poloidal
W-split update is not implemented for this scheme, so parameter validation
rejects it before state allocation. The type remains available for internal
kernel development and source compatibility.
"""
struct ExponentialAdamsBashforth2 <: AbstractTimestepper
    krylov_dimension::Int
    tolerance::Float64
end
function ExponentialAdamsBashforth2(;
        krylov_dimension::Int = 20,
        tolerance::Real = 1e-8
)
    ExponentialAdamsBashforth2(krylov_dimension, Float64(tolerance))
end

"""
    EAB2(; krylov_dimension=20, tolerance=1e-8)

Compatibility alias for [`ExponentialAdamsBashforth2`](@ref).
"""
const EAB2 = ExponentialAdamsBashforth2

"""
    ExponentialRungeKutta2()

Explicit second-order Runge-Kutta timestepper.
"""
struct ExponentialRungeKutta2 <: AbstractTimestepper end

"""
    ERK2()

Compatibility alias for [`ExponentialRungeKutta2`](@ref).
"""
const ERK2 = ExponentialRungeKutta2

"""
    RungeKutta3()

Cavaglieri-Bewley/Williamson 2N-storage third-order IMEX Runge-Kutta
timestepper. Nonlinear terms are advanced with the three-stage low-storage RK3
recurrence, while diffusion is solved implicitly at each substage.
"""
struct RungeKutta3 <: AbstractTimestepper end

"""
    CB3()

Compatibility alias for [`RungeKutta3`](@ref).
"""
const CB3 = RungeKutta3

"""
    ETD(; krylov_dimension=20, tolerance=1e-8)

Experimental exponential time-differencing descriptor.

Not currently supported by `Simulation`: the end-to-end solver update is not
implemented for this scheme, so parameter validation rejects it before state
allocation. The type remains available for internal kernel development and
source compatibility.
"""
struct ETD <: AbstractTimestepper
    krylov_dimension::Int
    tolerance::Float64
end
function ETD(; krylov_dimension::Int = 20, tolerance::Real = 1e-8)
    ETD(krylov_dimension, Float64(tolerance))
end

"""
    ThetaMethod(; theta=0.5)

Experimental single-parameter implicit-theta descriptor.

Not currently supported by `Simulation`: the end-to-end solver update is not
implemented for this scheme, so parameter validation rejects it before state
allocation. The type remains available for internal kernel development and
source compatibility.
"""
struct ThetaMethod <: AbstractTimestepper
    theta::Float64
end
ThetaMethod(; theta::Real = 0.5) = ThetaMethod(Float64(theta))

_timestepper_scheme(::CNAB2) = :cnab2
_timestepper_scheme(::ExponentialAdamsBashforth2) = :eab2
_timestepper_scheme(::ExponentialRungeKutta2) = :erk2
_timestepper_scheme(::RungeKutta3) = :cb3
_timestepper_scheme(::ETD) = :etd
_timestepper_scheme(::ThetaMethod) = :theta
_timestepper_scheme(scheme::Symbol) = scheme

# Timestepper-derived settings come from the timestepper struct itself; the
# generic fallbacks use the standard defaults (these match the
# ExponentialAdamsBashforth2/ETD struct defaults and the CNAB2 theta default)
# for schemes that do not carry the corresponding field. SolverParameters no
# longer stores these scalars.
_timestepper_implicit_theta(timestepper, params) = 0.5
_timestepper_implicit_theta(timestepper::CNAB2, params) = timestepper.implicit_theta
_timestepper_implicit_theta(timestepper::ThetaMethod, params) = timestepper.theta

_timestepper_krylov_dimension(timestepper, params) = 20
_timestepper_krylov_dimension(
    timestepper::ExponentialAdamsBashforth2, params) = timestepper.krylov_dimension
_timestepper_krylov_dimension(timestepper::ETD, params) = timestepper.krylov_dimension

_timestepper_krylov_tolerance(timestepper, params) = 1e-8
_timestepper_krylov_tolerance(
    timestepper::ExponentialAdamsBashforth2, params) = timestepper.tolerance
_timestepper_krylov_tolerance(timestepper::ETD, params) = timestepper.tolerance

# Construct a timestepper struct from a scheme symbol and optional overrides.
# Missing overrides fall back to each struct's own defaults.
function _timestepper_from_scheme(
        scheme::Symbol,
        implicit_theta::Union{Real, Nothing},
        etd_krylov_dimension::Union{Int, Nothing},
        krylov_tolerance::Union{Real, Nothing}
)
    if scheme === :cnab2
        return CNAB2(theta = something(implicit_theta, 0.5))
    elseif scheme === :eab2
        return ExponentialAdamsBashforth2(
            krylov_dimension = something(etd_krylov_dimension, 20),
            tolerance = something(krylov_tolerance, 1e-8))
    elseif scheme === :erk2
        return ExponentialRungeKutta2()
    elseif scheme === :cb3
        return RungeKutta3()
    elseif scheme === :etd
        return ETD(krylov_dimension = something(etd_krylov_dimension, 20),
            tolerance = something(krylov_tolerance, 1e-8))
    elseif scheme === :theta
        return ThetaMethod(theta = something(implicit_theta, 0.5))
    else
        throw(ArgumentError("Unknown timestep_scheme=$scheme"))
    end
end

# Fold an explicitly requested scalar into the resolved timestepper STRUCT.
# `SolverParameters` stores only the struct, and every consumer derives theta /
# krylov settings from it (`_timestepper_implicit_theta` & friends), so an
# override that is not folded in here is silently dropped — e.g.
# `Simulation(model; Δt, implicit_theta=1.0)` would keep running Crank-Nicolson
# theta=0.5. Returning `nothing` means "this timestepper cannot carry it", which
# the caller turns into a loud error rather than a silent no-op.
_timestepper_with_theta(::Any, theta::Float64) = nothing
_timestepper_with_theta(::CNAB2, theta::Float64) = CNAB2(implicit_theta = theta)
_timestepper_with_theta(::ThetaMethod, theta::Float64) = ThetaMethod(theta = theta)

_timestepper_with_krylov(::Any, dim, tol) = nothing
function _timestepper_with_krylov(ts::ExponentialAdamsBashforth2, dim, tol)
    ExponentialAdamsBashforth2(
        krylov_dimension = something(dim, ts.krylov_dimension),
        tolerance = Float64(something(tol, ts.tolerance)))
end
function _timestepper_with_krylov(ts::ETD, dim, tol)
    ETD(krylov_dimension = something(dim, ts.krylov_dimension),
        tolerance = Float64(something(tol, ts.tolerance)))
end

function _apply_timestepper_overrides(
        effective,
        implicit_theta::Union{Real, Nothing},
        etd_krylov_dimension::Union{Int, Nothing},
        krylov_tolerance::Union{Real, Nothing}
)
    ts = effective
    if !isnothing(implicit_theta)
        theta = Float64(implicit_theta)
        updated = _timestepper_with_theta(ts, theta)
        isnothing(updated) && throw(ArgumentError(
            "implicit_theta=$theta cannot be applied to $(typeof(ts)): only CNAB2 " *
            "and ThetaMethod carry an implicit weight. Drop implicit_theta, or pass " *
            "a timestepper that uses it — silently ignoring it would run a different " *
            "scheme than requested."))
        ts = updated
    end
    if !isnothing(etd_krylov_dimension) || !isnothing(krylov_tolerance)
        updated = _timestepper_with_krylov(ts, etd_krylov_dimension, krylov_tolerance)
        isnothing(updated) && throw(ArgumentError(
            "etd_krylov_dimension/krylov_tolerance cannot be applied to $(typeof(ts)): " *
            "only ExponentialAdamsBashforth2 and ETD carry Krylov settings. Drop them, " *
            "or pass an exponential timestepper — silently ignoring them would run " *
            "with different accuracy than requested."))
        ts = updated
    end
    return ts
end

function _resolve_timestepper(
        timestepper,
        timestep_scheme::Union{Symbol, Nothing},
        implicit_theta::Union{Real, Nothing},
        etd_krylov_dimension::Union{Int, Nothing},
        krylov_tolerance::Union{Real, Nothing},
        params
)
    # Resolve the effective timestepper struct: an explicit `timestepper` wins;
    # otherwise build one from `timestep_scheme`; otherwise fall back to the
    # timestepper carried on the parameters. SolverParameters stores the
    # timestepper as a struct, so scheme/theta/krylov are derived from it.
    effective = if !isnothing(timestepper)
        # The high-level API is lenient: a bare scheme Symbol passed as
        # `timestepper` is converted to its struct so an AbstractTimestepper —
        # never a Symbol — reaches SolverParameters. (The low-level
        # SolverParameters(timestepper=…) field still requires a struct.)
        timestepper isa Symbol ?
        _timestepper_from_scheme(timestepper, implicit_theta,
            etd_krylov_dimension, krylov_tolerance) :
        timestepper
    elseif !isnothing(timestep_scheme)
        _timestepper_from_scheme(timestep_scheme, implicit_theta,
            etd_krylov_dimension, krylov_tolerance)
    else
        params.timestepper
    end

    # Apply the explicit scalar overrides to the resolved struct. Without this,
    # anything that arrives as a struct (the default `params.timestepper`, or an
    # explicit `timestepper=CNAB2()`) ignores implicit_theta / krylov kwargs
    # entirely, because SolverParameters carries no such fields.
    effective = _apply_timestepper_overrides(
        effective, implicit_theta, etd_krylov_dimension, krylov_tolerance)

    scheme = _timestepper_scheme(effective)
    if !isnothing(timestep_scheme) && timestep_scheme !== scheme
        throw(ArgumentError(
            "timestepper=$(typeof(effective)) maps to timestep_scheme=$scheme, " *
            "but timestep_scheme=$timestep_scheme was also provided",
        ))
    end

    return (
        timestepper = effective,
        timestep_scheme = scheme,
        implicit_theta = Float64(something(
            implicit_theta,
            _timestepper_implicit_theta(effective, params)
        )),
        etd_krylov_dimension = something(
            etd_krylov_dimension,
            _timestepper_krylov_dimension(effective, params)
        ),
        krylov_tolerance = Float64(something(
            krylov_tolerance,
            _timestepper_krylov_tolerance(effective, params)
        ))
    )
end
