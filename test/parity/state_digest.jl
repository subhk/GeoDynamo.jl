"""
    ParityDigest

Bit-exact comparison of two `SolverState`s.

This is the shared comparator for the v3.0.0 refactor program. It exists because
the repo's existing end-to-end probes are INVARIANT probes: they assert a physical
property (boundary residual near zero) on the evolved state, which any correct
implementation satisfies. They cannot prove that a refactored implementation
matches the original. This module does.

Comparison is on reinterpreted bits, not `==` and not `isapprox`. `==` gives the
wrong answer twice for this purpose: `-0.0 == 0.0` is `true`, and `NaN == NaN` is
`false`. For a differential gate, matching NaNs in the same slot are a PASS —
objecting to a NaN is the invariant probes' job, not this one's.
"""
module ParityDigest

using GeoDynamo
using MPI

export FieldBits, StateDigest, digest_state, digests_equal

# Field names that must never be walked.
#
# :config and :pencil are back-references into SHTnsKitConfig / Pencil, which are
# cyclic and hold C transform plans.
# :domain is the static radial grid; it is already pinned by the recorded params.
# The four *_time fields are Ref{Float64} wall-clock counters — digesting them
# guarantees a spurious failure on every single run.
const SKIP_FIELDS = Set{Symbol}((
    :config, :pencil, :domain,
    :computation_time, :transform_time, :comm_time, :spectral_time,
))

struct FieldBits
    name::String
    dims::Vector{Int}
    values::Vector{Float64}
end

struct StateDigest
    env::Dict{String, Any}     # compared for comparability
    info::Dict{String, Any}    # recorded only, never compared
    fields::Vector{FieldBits}
    hash::UInt64
end

# Widening Float32 -> Float64 is exact and injective, so bit comparison on the
# widened values is still an exact comparison of the originals. This keeps the
# digest element-type agnostic.
_push_array!(out, name, A) =
    push!(out, FieldBits(name, collect(size(A)), Float64.(vec(A))))

function _walk!(out::Vector{FieldBits}, seen::Base.IdSet{Any}, name::String, x)
    x === nothing && return nothing

    if x isa GeoDynamo.PencilArray
        _push_array!(out, name, parent(x))
        return nothing
    elseif x isa AbstractArray
        # Integer arrays are BC type codes and mode tables: static, not evolved.
        eltype(x) <: AbstractFloat && _push_array!(out, name, x)
        return nothing
    elseif x isa GeoDynamo.SHTnsKitConfig
        return nothing
    end

    T = typeof(x)
    isstructtype(T) || return nothing
    # Numbers, Symbols, Strings are structs by isstructtype but carry no arrays.
    (x isa Number || x isa Symbol || x isa AbstractString) && return nothing

    if ismutable(x)
        x in seen && return nothing
        push!(seen, x)
    end

    for f in fieldnames(T)
        f in SKIP_FIELDS && continue
        isdefined(x, f) || continue
        _walk!(out, seen, string(name, ".", f), getfield(x, f))
    end
    return nothing
end

function _hash_fields(fields::Vector{FieldBits})
    h = UInt64(0x9e3779b97f4a7c15)
    for fb in fields
        h = hash(fb.name, h)
        h = hash(fb.dims, h)
        h = hash(reinterpret(UInt64, fb.values), h)
    end
    return h
end

"""
    digest_state(state) -> StateDigest

Walk `state.fields` and capture every floating-point array reachable from it.

`prev_nonlinear` / `prev_nl_*` are captured deliberately. They are not derived
state: they carry the CNAB2 history, and a refactor that corrupts them produces a
state that looks correct for exactly one step and diverges afterward.
"""
function digest_state(state)
    out = FieldBits[]
    _walk!(out, Base.IdSet{Any}(), "fields", state.fields)

    env = Dict{String, Any}(
        "nthreads" => Threads.nthreads(),
        "nranks" => MPI.Initialized() ? MPI.Comm_size(MPI.COMM_WORLD) : 1,
        "word_size" => Sys.WORD_SIZE,
        "julia" => string(VERSION),
    )
    # Recorded but NOT compared: a clean-break sub-project legitimately changes
    # how parameters print, and that must not read as a physics difference.
    info = Dict{String, Any}(
        "params" => string(state.parameters),
        "time" => state.time,
        "step" => state.step,
    )
    return StateDigest(env, info, out, _hash_fields(out))
end

function _first_difference(a::FieldBits, b::FieldBits)
    ba = reinterpret(UInt64, a.values)
    bb = reinterpret(UInt64, b.values)
    @inbounds for i in eachindex(ba)
        if ba[i] != bb[i]
            va, vb = a.values[i], b.values[i]
            ulps = (isfinite(va) && isfinite(vb)) ?
                   string(abs(reinterpret(Int64, va) - reinterpret(Int64, vb))) :
                   "n/a"
            return "$(a.name): differs at index $i of $(a.dims) — " *
                   "$(repr(va)) vs $(repr(vb)) ($ulps ULP)"
        end
    end
    return ""
end

"""
    digests_equal(a, b; compare_names = true) -> (ok::Bool, message::String)

Bit-compare two digests.

`compare_names = false` is for mechanism B (in-tree A/B across a clean break),
where the two implementations legitimately expose different field names for the
same physical quantity. Order, shape, and bits must still match exactly.

The stored hash is a fast-rejection convenience only. Equality is always
confirmed against the raw values before this returns `true`, so a hash collision
cannot produce a false green.
"""
function digests_equal(a::StateDigest, b::StateDigest; compare_names::Bool = true)
    if a.env != b.env
        return (false, "digests not comparable: env differs — $(a.env) vs $(b.env)")
    end
    if length(a.fields) != length(b.fields)
        return (false, "field count differs: $(length(a.fields)) vs $(length(b.fields))")
    end
    for (fa, fb) in zip(a.fields, b.fields)
        if compare_names && fa.name != fb.name
            return (false, "field name differs: $(fa.name) vs $(fb.name)")
        end
        if fa.dims != fb.dims
            return (false, "$(fa.name): dims differ — $(fa.dims) vs $(fb.dims)")
        end
        msg = _first_difference(fa, fb)
        isempty(msg) || return (false, msg)
    end
    return (true, "")
end

end # module
