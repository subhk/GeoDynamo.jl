abstract type AbstractVelocityBC end
abstract type AbstractThermalBC    end
abstract type AbstractMagneticBC   end

# Velocity BCs
struct NoSlip     <: AbstractVelocityBC end
struct StressFree <: AbstractVelocityBC end
Base.show(io::IO, ::NoSlip) = print(io, "NoSlip()")
Base.show(io::IO, ::StressFree) = print(io, "StressFree()")

# Thermal / composition BCs
struct FixedTemperature{T} <: AbstractThermalBC; value::T end
struct FixedFlux{T}        <: AbstractThermalBC; value::T end
FixedTemperature() = FixedTemperature(0.0)
FixedFlux()        = FixedFlux(0.0)
Base.show(io::IO, bc::FixedTemperature) = print(io, "FixedTemperature($(bc.value))")
Base.show(io::IO, bc::FixedFlux) = print(io, "FixedFlux($(bc.value))")

# Magnetic BCs
struct InsulatingMagnetic <: AbstractMagneticBC end
struct ConductingMagnetic <: AbstractMagneticBC end
Base.show(io::IO, ::InsulatingMagnetic) = print(io, "InsulatingMagnetic()")
Base.show(io::IO, ::ConductingMagnetic) = print(io, "ConductingMagnetic()")

# Per-field wrapper holding an inner and outer BC
struct BoundaryConditions{I, O}
    inner :: I
    outer :: O
end
BoundaryConditions(; inner, outer) = BoundaryConditions(inner, outer)
Base.show(io::IO, bc::BoundaryConditions) =
    print(io, "BoundaryConditions(inner = $(bc.inner), outer = $(bc.outer))")

Base.:(==)(::NoSlip, ::NoSlip) = true
Base.:(==)(::StressFree, ::StressFree) = true
Base.:(==)(a::FixedTemperature, b::FixedTemperature) = a.value == b.value
Base.:(==)(a::FixedFlux, b::FixedFlux) = a.value == b.value
Base.:(==)(::InsulatingMagnetic, ::InsulatingMagnetic) = true
Base.:(==)(::ConductingMagnetic, ::ConductingMagnetic) = true
Base.:(==)(a::BoundaryConditions, b::BoundaryConditions) =
    a.inner == b.inner && a.outer == b.outer

# ---------------------------------------------------------------------------
# Integer code dispatch — matches the convention in src/bcs/velocity_bc.jl:
#   1 = NoSlip/NoSlip
#   2 = NoSlip/StressFree
#   3 = StressFree/NoSlip
#   4 = StressFree/StressFree
# ---------------------------------------------------------------------------
_velocity_bc_code(::BoundaryConditions{NoSlip,     NoSlip})     = 1
_velocity_bc_code(::BoundaryConditions{NoSlip,     StressFree}) = 2
_velocity_bc_code(::BoundaryConditions{StressFree, NoSlip})     = 3
_velocity_bc_code(::BoundaryConditions{StressFree, StressFree}) = 4

# ---------------------------------------------------------------------------
# Thermal / composition codes — matches src/bcs/thermal_bc.jl:
#   1 = DD (Dirichlet/Dirichlet = FixedTemperature/FixedTemperature)
#   2 = DN (Dirichlet/Neumann   = FixedTemperature/FixedFlux)
#   3 = ND (Neumann/Dirichlet   = FixedFlux/FixedTemperature)
#   4 = NN (Neumann/Neumann     = FixedFlux/FixedFlux)
# ---------------------------------------------------------------------------
_thermal_bc_code(::BoundaryConditions{<:FixedTemperature, <:FixedTemperature}) = 1
_thermal_bc_code(::BoundaryConditions{<:FixedTemperature, <:FixedFlux})        = 2
_thermal_bc_code(::BoundaryConditions{<:FixedFlux,        <:FixedTemperature}) = 3
_thermal_bc_code(::BoundaryConditions{<:FixedFlux,        <:FixedFlux})        = 4

_composition_bc_code(bc) = _thermal_bc_code(bc)
