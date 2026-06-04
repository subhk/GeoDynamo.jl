# =============================================================================
# GPU Phase 0 — device-resident field containers (single GPU, no MPI/pencils).
# Shapes mirror the CPU containers: physical (nlat, nlon, nr); spectral (nlm, nr).
# Backing arrays are allocated through arch_zeros(arch, ...) → CuArray on a GPU
# backend, plain Array on CPU. No PencilArrays (single GPU has no decomposition).
# =============================================================================

"""
    GPUPhysicalField{T,A}

Device-resident physical field: `data` is an `(nlat, nlon, nr)` array on the
architecture's backend.
"""
struct GPUPhysicalField{T, A}
    config::Any   # SHTnsKit config; kept Any to avoid an upstream parametric dependency at this layer (revisit Phase 1+)
    nlat::Int
    nlon::Int
    nr::Int
    data::A
end

"""
    allocate_gpu_physical_field(T, arch, config, nr) -> GPUPhysicalField

Allocate a zero-filled `(nlat, nlon, nr)` physical field on `arch`'s backend.
"""
function allocate_gpu_physical_field(::Type{T}, arch::AbstractArchitecture, config, nr::Int) where {T}
    nlat = config.nlat
    nlon = config.nlon
    data = arch_zeros(arch, T, nlat, nlon, nr)
    return GPUPhysicalField{T, typeof(data)}(config, nlat, nlon, nr, data)
end

"""
    GPUSpectralField{T,A}

Device-resident spectral field in the DENSE `(lmax+1, mmax+1, nr)` `(l, m, r)`
layout — the same layout the CPU spectral field uses and the exact form
SHTnsKit's `gpu_synthesis`/`gpu_analysis` consume/produce per radial level (a
dense `(lmax+1, mmax+1)` coefficient matrix).  `data_real`/`data_imag` split the
complex coefficients (mirrors the CPU container).  `T = real(CT)` where `CT` is
the complex type passed to `allocate_gpu_spectral_field`; `A` is the backend
array type (e.g. `CuArray{Float64,3}` on CUDA, `Array{Float64,3}` on CPU).
"""
struct GPUSpectralField{T, A}
    config::Any   # SHTnsKit config; kept Any to avoid an upstream parametric dependency at this layer (revisit Phase 1+)
    nl::Int       # lmax + 1 (number of l-slots)
    nm::Int       # mmax + 1 (number of m-slots)
    nr::Int
    data_real::A
    data_imag::A
end

"""
    allocate_gpu_spectral_field(CT, arch, config, nr) -> GPUSpectralField

Allocate a zero-filled dense `(lmax+1, mmax+1, nr)` split-complex spectral field
on `arch`'s backend.  `CT` is the complex element type (`ComplexF64`); storage is
its real part type (`Float64`).
"""
function allocate_gpu_spectral_field(::Type{CT}, arch::AbstractArchitecture, config, nr::Int) where {CT}
    RT = real(CT)
    nl = config.lmax + 1
    nm = config.mmax + 1
    dr = arch_zeros(arch, RT, nl, nm, nr)
    di = arch_zeros(arch, RT, nl, nm, nr)
    return GPUSpectralField{RT, typeof(dr)}(config, nl, nm, nr, dr, di)
end

"""
    field_to_host(f) -> NamedTuple

Copy a GPU field's device arrays back to host `Array`s.  Returns
`(; data)` for a `GPUPhysicalField`, `(; data_real, data_imag)` for a
`GPUSpectralField`.
"""
field_to_host(f::GPUPhysicalField) = (; data = on_architecture(CPU(), f.data))
function field_to_host(f::GPUSpectralField)
    return (; data_real = on_architecture(CPU(), f.data_real),
              data_imag = on_architecture(CPU(), f.data_imag))
end

"""
    field_to_device(arch, host_phys::AbstractArray, config, nr) -> GPUPhysicalField
    field_to_device(arch, (hr, hi)::Tuple, config, nr)          -> GPUSpectralField

Copy host data onto `arch`'s backend, wrapped in the matching GPU field.
"""
function field_to_device(arch::AbstractArchitecture, host_phys::AbstractArray{T, 3}, config, nr::Int) where {T}
    @assert nr == size(host_phys, 3) "field_to_device: nr=$nr ≠ host dim-3 $(size(host_phys,3))"
    data = on_architecture(arch, host_phys)
    return GPUPhysicalField{T, typeof(data)}(config, size(host_phys, 1), size(host_phys, 2), nr, data)
end

function field_to_device(arch::AbstractArchitecture, host_spec::Tuple{<:AbstractArray, <:AbstractArray}, config, nr::Int)
    hr, hi = host_spec
    @assert nr == size(hr, 3) "field_to_device: nr=$nr ≠ host dim-3 $(size(hr,3))"
    dr = on_architecture(arch, hr)
    di = on_architecture(arch, hi)
    return GPUSpectralField{eltype(hr), typeof(dr)}(config, size(hr, 1), size(hr, 2), nr, dr, di)
end
