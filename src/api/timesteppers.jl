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

_timestepper_implicit_theta(timestepper, params) = params.implicit_theta
_timestepper_implicit_theta(timestepper::CNAB2, params) = timestepper.implicit_theta
_timestepper_implicit_theta(timestepper::ThetaMethod, params) = timestepper.theta

_timestepper_krylov_dimension(timestepper, params) = params.etd_krylov_dimension
_timestepper_krylov_dimension(timestepper::EAB2, params) = timestepper.krylov_dimension
_timestepper_krylov_dimension(timestepper::ETD, params) = timestepper.krylov_dimension

_timestepper_krylov_tolerance(timestepper, params) = params.krylov_tolerance
_timestepper_krylov_tolerance(timestepper::EAB2, params) = timestepper.tolerance
_timestepper_krylov_tolerance(timestepper::ETD, params) = timestepper.tolerance

function _resolve_timestepper(
        timestepper,
        timestep_scheme::Union{Symbol,Nothing},
        implicit_theta::Union{Real,Nothing},
        etd_krylov_dimension::Union{Int,Nothing},
        krylov_tolerance::Union{Real,Nothing},
        params,
    )
    if isnothing(timestepper)
        return (
            timestep_scheme = something(timestep_scheme, params.timestep_scheme),
            implicit_theta = Float64(something(implicit_theta, params.implicit_theta)),
            etd_krylov_dimension = something(etd_krylov_dimension, params.etd_krylov_dimension),
            krylov_tolerance = Float64(something(krylov_tolerance, params.krylov_tolerance)),
        )
    end

    scheme = _timestepper_scheme(timestepper)
    if !isnothing(timestep_scheme) && timestep_scheme !== scheme
        throw(ArgumentError(
            "timestepper=$(typeof(timestepper)) maps to timestep_scheme=$scheme, " *
            "but timestep_scheme=$timestep_scheme was also provided",
        ))
    end

    return (
        timestep_scheme = scheme,
        implicit_theta = Float64(something(
            implicit_theta,
            _timestepper_implicit_theta(timestepper, params),
        )),
        etd_krylov_dimension = something(
            etd_krylov_dimension,
            _timestepper_krylov_dimension(timestepper, params),
        ),
        krylov_tolerance = Float64(something(
            krylov_tolerance,
            _timestepper_krylov_tolerance(timestepper, params),
        )),
    )
end
