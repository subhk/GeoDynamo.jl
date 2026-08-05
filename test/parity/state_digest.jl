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
# :domain and :outer_domain are the static radial grid (RadialDomain — see
# src/fields/containers.jl:194): N, r, dr_matrices, radial_laplacian,
# integration_weights. It is already pinned by the recorded params. Most field
# structs hold this under the name `domain`; SHTnsMagneticFields
# (src/physics/magnetic/field.jl:189) holds the identical RadialDomain type
# under `outer_domain` instead — same static-grid rationale, different name.
# Without this, dr_matrices::Vector{Matrix{Float64}} — a float array of float
# arrays — hits the fail-loud "unclassified array" branch below and crashes
# digest_state on any real magnetic-field state.
# The four *_time fields are Ref{Float64} wall-clock counters — digesting them
# guarantees a spurious failure on every single run.
const SKIP_FIELDS = Set{Symbol}((
    :config, :pencil, :domain, :outer_domain,
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
        if eltype(x) <: AbstractFloat
            _push_array!(out, name, x)
        elseif eltype(x) <: Integer
            # Integer arrays are BC type codes and mode tables: static, not
            # evolved. Known, deliberately-classified case — stays silent.
        elseif eltype(x) <: AbstractArray
            # Array-of-arrays — e.g. VelocityWorkspace's per-thread scratch
            # buffers, `Vector{Vector{T}}` (src/physics/velocity/field.jl:121),
            # reachable at fields.velocity.velocity_workspace once a state has
            # been through solver_step! at least once. This is a CONTAINER,
            # not a leaf: recurse into each element by deterministic linear
            # index instead of capturing or skipping the whole thing.
            # Skipping this by field name would be inconsistent with the fact
            # that this walker already captures other scratch arrays
            # (work_spectral, work_physical, advection_physical) — an
            # array-of-arrays should not be invisible merely because of its
            # nesting shape. `eachindex` on a Vector is 1:length(x), so this
            # ordering is stable across runs; it does not rely on any
            # unordered iteration. An unclassifiable element type inside the
            # container (e.g. Vector{Vector{SomeStruct}}) still recurses into
            # this same elseif chain for the inner array and falls through to
            # the `error` branch below exactly as it would at top level —
            # nesting does not launder an unknown element type into silence.
            for i in eachindex(x)
                _walk!(out, seen, string(name, "[", i, "]"), x[i])
            end
        else
            error("ParityDigest._walk!: unclassified array at $(name) — " *
                  "eltype $(eltype(x)) of $(typeof(x)) is neither a float, " *
                  "an integer, nor itself an array, so the digest cannot " *
                  "tell whether this holds evolved physics state. Classify " *
                  "it explicitly instead of letting it vanish from the " *
                  "digest silently.")
        end
        return nothing
    elseif x isa GeoDynamo.SHTnsKitConfig
        return nothing
    elseif x isa AbstractDict
        # Dicts appear as free-form metadata containers reachable from field
        # structs (e.g. BoundaryInterpolationCache.metadata::Dict{String,Any},
        # a generic escape hatch for keys the struct has no dedicated field
        # for). Do NOT let this fall through to the generic struct walk
        # below: that would descend into Dict's internal hash-table storage
        # (slots/keys/vals buffers), which is an implementation detail, not
        # logical content, and is not stable across insertion order. Instead
        # walk each *value* explicitly by key, so any float payload smuggled
        # into a Dict is still captured by the array/struct cases above and
        # below — or throws, exactly as it would anywhere else in the tree,
        # if the value itself turns out to be unclassified (e.g. a raw
        # Vector{Any} entry). Keys are treated as identifying strings, not
        # physics data, and are not separately digested.
        #
        # Dict's OWN iteration order depends on insertion/deletion history,
        # not logical content: two Dicts with an identical key set and
        # identical values can walk their entries in different orders.
        # digests_equal zips fields POSITIONALLY, so a nondeterministic
        # order would produce a spurious "field name differs" failure under
        # compare_names=true — or, worse, under compare_names=false (the
        # mechanism-B / in-tree-A/B default this harness exists to serve),
        # silently compare bits from UNRELATED entries at the same
        # position, which can manufacture a false pass. A false pass is the
        # one outcome this module must never produce. Iterate in a
        # canonical, content-determined order instead of Dict's own order.
        ks = collect(keys(x))
        sorted_keys = try
            sort(ks)
        catch
            # Heterogeneous / non-<-comparable key types: fall back to
            # sorting by string representation, which is still canonical
            # and content-determined, not insertion-order-dependent.
            sort(ks; by = string)
        end
        for k in sorted_keys
            _walk!(out, seen, string(name, "[", k, "]"), x[k])
        end
        return nothing
    end

    T = typeof(x)
    # Common scalar/sentinel leaf values that carry no arrays and are known,
    # deliberately-skipped cases:
    #   - Number, Symbol, AbstractString: thresholds, labels, format tags.
    #   - Base.Enum: e.g. bcs.BoundaryConditionSet.field_type::FieldType is
    #     reachable from state.fields.*.boundary_condition_set whenever
    #     file-based boundary conditions are loaded. @enum instances are
    #     neither isa Number/Symbol nor isstructtype, so without this they
    #     would hit the fail-loud branch below and crash digest_state on a
    #     legitimate, reachable configuration.
    #   - Missing, Function, Type: fieldless-but-legitimate leaves (a
    #     `Function` field, a `DataType`/type-parameter field, `missing`).
    #     Recursing into a `Type`'s own internal fields in particular would
    #     walk raw layout pointers, not physics data.
    #   - empty Tuple (`()`): a common no-payload sentinel/default value;
    #     non-empty tuples are unaffected and still recurse through the
    #     struct-fallthrough below exactly as before.
    (x isa Number || x isa Symbol || x isa AbstractString || x isa Base.Enum ||
        x isa Missing || x isa Function || x isa Type ||
        (x isa Tuple && isempty(x))) && return nothing

    if isstructtype(T) && fieldcount(T) > 0
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

    if isstructtype(T) &&
       (x isa GeoDynamo.AbstractVelocityBC || x isa GeoDynamo.AbstractThermalBC ||
        x isa GeoDynamo.AbstractMagneticBC)
        # BC marker singletons (NoSlip, StressFree, InsulatingMagnetic,
        # ConductingMagnetic — src/api/boundary_conditions.jl) are reachable
        # from every field's parameters (e.g.
        # fields.velocity.parameters.velocity_bcs.{inner,outer}). They are
        # fieldless by construction (fieldcount(T)==0, so this branch is only
        # reached at all because the fieldcount>0 recursion above already
        # returned for their fieldful siblings, e.g. ValueBoundaryCondition{T}
        # — those still recurse normally and any array payload in `.value` is
        # still captured, not skipped here). A marker singleton carries no
        # data beyond its type identity, so — unlike a genuinely unknown
        # fieldless struct, which still falls through to the error below —
        # it is safe to classify and skip silently.
        return nothing
    end

    # Anything else — a non-struct type not covered by the classified leaves
    # above, or a fieldless struct/singleton outside the known BC-marker
    # family — has fallen through every known case without being captured or
    # deliberately classified. Silently returning here is exactly the bug
    # this fix closes: fail loud instead.
    error("ParityDigest._walk!: unclassified value of type $(T) at $(name) " *
          "— it is not a Number/Symbol/AbstractString/Base.Enum/Missing/" *
          "Function/Type/empty-Tuple/SHTnsKitConfig/PencilArray/" *
          "AbstractArray/AbstractDict/AbstractVelocityBC/AbstractThermalBC/" *
          "AbstractMagneticBC marker, and has no fields to recurse into. " *
          "Classify it explicitly (capture, recurse, add to SKIP_FIELDS, " *
          "or handle it as a container) instead of letting it vanish from " *
          "the digest silently.")
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

# IEEE-754 bit patterns are not a monotonic integer encoding across the
# positive/negative boundary: 0.0 (0x0...0) and -0.0 (0x8...0) sit at
# opposite ends of the UInt64 range even though they are numerically equal,
# and more generally negative values' bit patterns run in the OPPOSITE
# direction to their numeric order. A naive `reinterpret(Int64, va) -
# reinterpret(Int64, vb)` can therefore subtract two values that are
# genuinely 1 ULP apart (e.g. 0.0 and -0.0, one reinterpreting as
# `typemin(Int64)`) and silently wrap under Julia's checked-off default
# integer overflow, producing a nonsense diagnostic.
#
# Standard fix: apply the order-preserving bias transform (flip the sign bit
# for non-negative patterns, complement the whole word for negative ones) so
# the transformed key is monotonic with the float's numeric value, then widen
# to Int128 before subtracting so the distance itself cannot overflow either.
_ulp_order_key(u::UInt64) =
    (u & (UInt64(1) << 63)) == 0 ? (u | (UInt64(1) << 63)) : ~u

function _first_difference(a::FieldBits, b::FieldBits)
    ba = reinterpret(UInt64, a.values)
    bb = reinterpret(UInt64, b.values)
    @inbounds for i in eachindex(ba)
        if ba[i] != bb[i]
            va, vb = a.values[i], b.values[i]
            ulps = (isfinite(va) && isfinite(vb)) ?
                   string(abs(Int128(_ulp_order_key(ba[i])) -
                              Int128(_ulp_order_key(bb[i])))) :
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
