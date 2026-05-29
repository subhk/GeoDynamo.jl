using KernelAbstractions

abstract type AbstractArchitecture end

struct CPU <: AbstractArchitecture end

struct GPU{B} <: AbstractArchitecture
    backend::B
end

"""
    arch_zeros(arch, FT, dims...)

Allocate a zero-filled array on the given architecture.
"""
arch_zeros(::CPU, FT::Type{T}, dims...) where {T} = zeros(FT, dims...)
function arch_zeros(::GPU, FT::Type{T}, dims...) where {T}
    error("GPU allocations require a loaded backend extension")
end

"""
    on_architecture(arch, array)

Move `array` to the device associated with `arch`.
"""
on_architecture(::CPU, a) = Array(a)
on_architecture(::GPU, a) = error("GPU data movement requires a loaded backend extension")

"""
    get_backend(arch)

Return the backend object associated with `arch`.
"""
get_backend(::CPU) = KernelAbstractions.CPU()
get_backend(g::GPU) = g.backend
