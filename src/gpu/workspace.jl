# =============================================================================
# GPU workspace — a lazy per-bundle scratch pool so the device step paths are
# allocation-free at steady state (compute-bound, not allocator-bound).
#
# Every scratch array is keyed by a UNIQUE Symbol at its use site, so pooled
# buffers never alias each other within a step. Arrays are created on first
# use from a reference array (same backend/eltype/shape) and reused on every
# subsequent step. The pool lives on the device bundle (`state.work`); moving
# a bundle between backends resets the pool (buffers are recreated lazily on
# the new backend).
#
# The pool Dict is `Symbol => Any`; the `::typeof(ref)` assertions at the
# accessors are function barriers that keep downstream kernel calls concretely
# typed (an abstract-typed pool read in a hot loop is an allocation storm —
# see the ERK2 boxing lesson).
# =============================================================================

struct GPUWorkspace
    pool::Dict{Symbol, Any}
end
GPUWorkspace() = GPUWorkspace(Dict{Symbol, Any}())

"""
    gpu_scratch!(ws, key, ref) -> array

Pooled `similar(ref)`: first call under `key` allocates, later calls return
the cached buffer (contents are stale — callers must overwrite). `ws ===
nothing` falls back to a plain `similar(ref)` (un-pooled paths keep working).
"""
@inline gpu_scratch!(::Nothing, ::Symbol, ref::AbstractArray) = similar(ref)
@inline function gpu_scratch!(ws::GPUWorkspace, key::Symbol, ref::AbstractArray)
    arr = get!(() -> similar(ref), ws.pool, key)::typeof(ref)
    # `typeof` pins eltype+ndims but NOT shape; guard against a same-tag/
    # different-shape collision returning a wrong-shaped buffer (silent on device).
    size(arr) == size(ref) || error(
        "gpu_scratch! tag collision on :$key — cached $(size(arr)) ≠ requested $(size(ref))")
    return arr
end

"""
    gpu_scratch_phys!(ws, key, T, arch, config, nr) -> GPUPhysicalField

Pooled physical-field scratch (the `(nlat, nlon, nr)` work fields every
transform/nonlinear pass needs). Same keying rules as [`gpu_scratch!`](@ref).
Returns are intentionally un-asserted (Array vs CuArray backing); call sites
hand the field straight to kernel wrappers, which are dispatch barriers.
"""
@inline gpu_scratch_phys!(::Nothing, ::Symbol, ::Type{T}, arch, config, nr) where {T} =
    allocate_gpu_physical_field(T, arch, config, nr)
@inline function gpu_scratch_phys!(ws::GPUWorkspace, key::Symbol, ::Type{T},
        arch, config, nr) where {T}
    f = get!(() -> allocate_gpu_physical_field(T, arch, config, nr), ws.pool, key)
    # Guard a same-tag/different-shape collision (silent on device) — mirror the
    # sibling getters. `GPUPhysicalField` backs an (nlat, nlon, nr) array in `.data`.
    size(f.data) == (config.nlat, config.nlon, nr) || error(
        "gpu_scratch_phys! tag collision on :$key — cached $(size(f.data)) ≠ requested $((config.nlat, config.nlon, nr))")
    # `T` pins the eltype, but the backing-array param `A` of `GPUPhysicalField{T,A}`
    # is erased by this UnionAll assertion, so `f.data`'s type is NOT statically
    # pinned. Call sites needing a concrete `.data` should route it through a
    # `ref`-typed getter (the 3-arg `gpu_scratch!`), which is the concrete barrier.
    return f::GPUPhysicalField{T}
end

"""
    gpu_scratch_complex!(ws, key, ref_real, dims) -> dense complex buffer

Pooled complex buffer for the per-level transform staging (the
`complex.(view, view)` materializations). `ref_real` supplies the backend.
"""
@inline gpu_scratch_complex!(::Nothing, ::Symbol, ref_real::AbstractArray{T},
    dims::Dims) where {T} = similar(ref_real, Complex{T}, dims)
@inline function gpu_scratch_complex!(ws::GPUWorkspace, key::Symbol,
        ref_real::AbstractArray{T}, dims::Dims) where {T}
    arr = get!(() -> similar(ref_real, Complex{T}, dims), ws.pool, key)
    size(arr) == dims || error(
        "gpu_scratch_complex! tag collision on :$key — cached $(size(arr)) ≠ requested $dims")
    # No `::typeof(ref_real)` barrier: the result is `Complex{T}`-eltype with
    # `length(dims)` ndims, so its type differs from the real `ref_real` — naming
    # the concrete backend-swapped type here would need an invasive `similar_type`
    # refactor. Downstream kernel wrappers are dispatch barriers, so `arr::Any`
    # here is recovered at the first wrapper call.
    return arr
end

"""
    gpu_scratch!(ws, key, ref, dims) -> array

Pooled `similar(ref, dims)` — same rules, explicit shape (e.g. one (nlat,
nlon) level buffer derived from a 3-D field).
"""
@inline gpu_scratch!(::Nothing, ::Symbol, ref::AbstractArray, dims::Dims) = similar(ref, dims)
@inline function gpu_scratch!(ws::GPUWorkspace, key::Symbol, ref::AbstractArray, dims::Dims)
    arr = get!(() -> similar(ref, dims), ws.pool, key)
    size(arr) == dims || error(
        "gpu_scratch! tag collision on :$key — cached $(size(arr)) ≠ requested $dims")
    # No `::typeof(ref)` barrier: the result has `length(dims)` ndims, which is
    # routinely fewer than `ref`'s (e.g. a 2-D (nlat, nlon) level buffer from a 3-D
    # field), so `typeof(ref)` would be the wrong type. Naming the concrete type
    # here would need an invasive `similar_type` refactor; downstream kernel
    # wrappers are dispatch barriers that recover concreteness at first use.
    return arr
end
