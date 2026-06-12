abstract type AbstractVelocityBC end
abstract type AbstractThermalBC end
abstract type AbstractMagneticBC end

# Velocity BCs
struct NoSlip <: AbstractVelocityBC end
struct StressFree <: AbstractVelocityBC end
Base.show(io::IO, ::NoSlip) = print(io, "NoSlip()")
Base.show(io::IO, ::StressFree) = print(io, "StressFree()")

# Thermal / composition BCs — Oceananigans-canonical names; the original
# GeoDynamo names remain as const aliases so existing code keeps working.
struct ValueBoundaryCondition{T} <: AbstractThermalBC   # Dirichlet
    value::T
end
struct FluxBoundaryCondition{T} <: AbstractThermalBC    # Neumann/flux
    value::T
end
ValueBoundaryCondition() = ValueBoundaryCondition(0.0)
FluxBoundaryCondition() = FluxBoundaryCondition(0.0)
const FixedTemperature = ValueBoundaryCondition
const FixedFlux = FluxBoundaryCondition
Base.show(io::IO, bc::ValueBoundaryCondition) = print(io, "ValueBoundaryCondition($(bc.value))")
Base.show(io::IO, bc::FluxBoundaryCondition) = print(io, "FluxBoundaryCondition($(bc.value))")

# Magnetic BCs
struct InsulatingMagnetic <: AbstractMagneticBC end
struct ConductingMagnetic <: AbstractMagneticBC end
Base.show(io::IO, ::InsulatingMagnetic) = print(io, "InsulatingMagnetic()")
Base.show(io::IO, ::ConductingMagnetic) = print(io, "ConductingMagnetic()")

# Per-field wrapper holding an inner and outer BC (Oceananigans-canonical name;
# spherical inner/outer stands in for Oceananigans' bottom/top).
struct FieldBoundaryConditions{I, O}
    inner::I
    outer::O
end
FieldBoundaryConditions(; inner, outer) = FieldBoundaryConditions(inner, outer)
const BoundaryConditions = FieldBoundaryConditions
function Base.show(io::IO, bc::FieldBoundaryConditions)
    print(io, "FieldBoundaryConditions(inner = $(bc.inner), outer = $(bc.outer))")
end

Base.:(==)(::NoSlip, ::NoSlip) = true
Base.:(==)(::StressFree, ::StressFree) = true
Base.:(==)(a::FixedTemperature, b::FixedTemperature) = a.value == b.value
Base.:(==)(a::FixedFlux, b::FixedFlux) = a.value == b.value
Base.:(==)(::InsulatingMagnetic, ::InsulatingMagnetic) = true
Base.:(==)(::ConductingMagnetic, ::ConductingMagnetic) = true
function Base.:(==)(a::BoundaryConditions, b::BoundaryConditions)
    a.inner == b.inner && a.outer == b.outer
end

# ---------------------------------------------------------------------------
# Integer code dispatch — matches the convention in src/bcs/velocity_bc.jl:
#   1 = NoSlip/NoSlip
#   2 = NoSlip/StressFree
#   3 = StressFree/NoSlip
#   4 = StressFree/StressFree
# ---------------------------------------------------------------------------
_velocity_bc_code(::BoundaryConditions{NoSlip, NoSlip}) = 1
_velocity_bc_code(::BoundaryConditions{NoSlip, StressFree}) = 2
_velocity_bc_code(::BoundaryConditions{StressFree, NoSlip}) = 3
_velocity_bc_code(::BoundaryConditions{StressFree, StressFree}) = 4

# ---------------------------------------------------------------------------
# Thermal / composition codes — matches src/bcs/thermal_bc.jl:
#   1 = DD (Dirichlet/Dirichlet = FixedTemperature/FixedTemperature)
#   2 = DN (Dirichlet/Neumann   = FixedTemperature/FixedFlux)
#   3 = ND (Neumann/Dirichlet   = FixedFlux/FixedTemperature)
#   4 = NN (Neumann/Neumann     = FixedFlux/FixedFlux)
# ---------------------------------------------------------------------------
_thermal_bc_code(::BoundaryConditions{<:FixedTemperature, <:FixedTemperature}) = 1
_thermal_bc_code(::BoundaryConditions{<:FixedTemperature, <:FixedFlux}) = 2
_thermal_bc_code(::BoundaryConditions{<:FixedFlux, <:FixedTemperature}) = 3
_thermal_bc_code(::BoundaryConditions{<:FixedFlux, <:FixedFlux}) = 4

_composition_bc_code(bc) = _thermal_bc_code(bc)
