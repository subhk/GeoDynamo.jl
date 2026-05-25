abstract type AbstractTimestepper end

"""
    CNAB2(; theta=0.5)

Crank-Nicolson Adams-Bashforth second-order timestepper.
"""
struct CNAB2 <: AbstractTimestepper
    implicit_theta::Float64
end

function CNAB2(;
        theta::Union{Real,Nothing}=nothing,
        implicit_theta::Union{Real,Nothing}=nothing,
    )
    if !isnothing(theta) && !isnothing(implicit_theta) &&
            Float64(theta) != Float64(implicit_theta)
        throw(ArgumentError("CNAB2 received both theta and implicit_theta with different values"))
    end
    value = something(theta, something(implicit_theta, 0.5))
    return CNAB2(Float64(value))
end

"""
    EAB2(; krylov_dimension=20, tolerance=1e-8)

Exponential Adams-Bashforth second-order timestepper.
"""
struct EAB2 <: AbstractTimestepper
    krylov_dimension::Int
    tolerance::Float64
end
EAB2(; krylov_dimension::Int=20, tolerance::Real=1e-8) =
    EAB2(krylov_dimension, Float64(tolerance))

"""
    ERK2()

Explicit second-order Runge-Kutta timestepper.
"""
struct ERK2 <: AbstractTimestepper end

"""
    ETD(; krylov_dimension=20, tolerance=1e-8)

Exponential time differencing timestepper.
"""
struct ETD <: AbstractTimestepper
    krylov_dimension::Int
    tolerance::Float64
end
ETD(; krylov_dimension::Int=20, tolerance::Real=1e-8) =
    ETD(krylov_dimension, Float64(tolerance))

"""
    ThetaMethod(; theta=0.5)

Single-parameter implicit theta timestepper.
"""
struct ThetaMethod <: AbstractTimestepper
    theta::Float64
end
ThetaMethod(; theta::Real=0.5) = ThetaMethod(Float64(theta))

_timestepper_scheme(::CNAB2) = :cnab2
_timestepper_scheme(::EAB2) = :eab2
_timestepper_scheme(::ERK2) = :erk2
_timestepper_scheme(::ETD) = :etd
_timestepper_scheme(::ThetaMethod) = :theta
_timestepper_scheme(scheme::Symbol) = scheme

# Timestepper-derived settings come from the timestepper struct itself; the
# generic fallbacks use the standard defaults (these match the EAB2/ETD struct
# defaults and the CNAB2 theta default) for schemes that do not carry the
# corresponding field. SolverParameters no longer stores these scalars.
_timestepper_implicit_theta(timestepper, params) = 0.5
_timestepper_implicit_theta(timestepper::CNAB2, params) = timestepper.implicit_theta
_timestepper_implicit_theta(timestepper::ThetaMethod, params) = timestepper.theta

_timestepper_krylov_dimension(timestepper, params) = 20
_timestepper_krylov_dimension(timestepper::EAB2, params) = timestepper.krylov_dimension
_timestepper_krylov_dimension(timestepper::ETD, params) = timestepper.krylov_dimension

_timestepper_krylov_tolerance(timestepper, params) = 1e-8
_timestepper_krylov_tolerance(timestepper::EAB2, params) = timestepper.tolerance
_timestepper_krylov_tolerance(timestepper::ETD, params) = timestepper.tolerance

# Construct a timestepper struct from a scheme symbol and optional overrides.
# Missing overrides fall back to each struct's own defaults.
function _timestepper_from_scheme(
        scheme::Symbol,
        implicit_theta::Union{Real,Nothing},
        etd_krylov_dimension::Union{Int,Nothing},
        krylov_tolerance::Union{Real,Nothing},
    )
    if scheme === :cnab2
        return CNAB2(theta = something(implicit_theta, 0.5))
    elseif scheme === :eab2
        return EAB2(krylov_dimension = something(etd_krylov_dimension, 20),
                    tolerance = something(krylov_tolerance, 1e-8))
    elseif scheme === :erk2
        return ERK2()
    elseif scheme === :etd
        return ETD(krylov_dimension = something(etd_krylov_dimension, 20),
                   tolerance = something(krylov_tolerance, 1e-8))
    elseif scheme === :theta
        return ThetaMethod(theta = something(implicit_theta, 0.5))
    else
        throw(ArgumentError("Unknown timestep_scheme=$scheme"))
    end
end

function _resolve_timestepper(
        timestepper,
        timestep_scheme::Union{Symbol,Nothing},
        implicit_theta::Union{Real,Nothing},
        etd_krylov_dimension::Union{Int,Nothing},
        krylov_tolerance::Union{Real,Nothing},
        params,
    )
    # Resolve the effective timestepper struct: an explicit `timestepper` wins;
    # otherwise build one from `timestep_scheme`; otherwise fall back to the
    # timestepper carried on the parameters. SolverParameters stores the
    # timestepper as a struct, so scheme/theta/krylov are derived from it.
    effective = if !isnothing(timestepper)
        timestepper
    elseif !isnothing(timestep_scheme)
        _timestepper_from_scheme(timestep_scheme, implicit_theta,
                                 etd_krylov_dimension, krylov_tolerance)
    else
        params.timestepper
    end

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
            _timestepper_implicit_theta(effective, params),
        )),
        etd_krylov_dimension = something(
            etd_krylov_dimension,
            _timestepper_krylov_dimension(effective, params),
        ),
        krylov_tolerance = Float64(something(
            krylov_tolerance,
            _timestepper_krylov_tolerance(effective, params),
        )),
    )
end
