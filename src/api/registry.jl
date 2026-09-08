# ================================================================================
# CollectiveRegistry — a named, ordered registry whose STRUCTURE decides which
# MPI collectives a simulation step enters, and which is therefore validated
# collectively once and then frozen.
# ================================================================================
#
# `Simulation.callbacks` and `Simulation.output_writers` are instances. Callback
# bodies (EnergyDiagnostics, HealthCheck, …) and writers (`write_fields!`,
# `write_restart!`) contain collectives, so a registry that differs between ranks
# — `rank == 0 && add_callback!(sim, …)` — makes one rank enter a collective the
# others never reach. That is a hang, not an error. Validating the registry on
# every step costs collectives of its own; validating it ONCE and refusing later
# mutation costs nothing per step and closes the hole for good.
# ================================================================================

"""
    CollectiveRegistry{K}

Ordered `Symbol => item` registry of kind `K` (`:callback` or `:writer`). Reads
like a dict. Writes go through [`register!`](@ref) until
[`validate_and_freeze!`](@ref) has run; afterwards the registry is immutable.
Direct `reg[name] = item` and `delete!` are rejected at any time so that the
frozen state cannot be bypassed.
"""
mutable struct CollectiveRegistry{K}
    items::OrderedDict{Symbol, Any}
    frozen::Bool
end

CollectiveRegistry{K}() where {K} =
    CollectiveRegistry{K}(OrderedDict{Symbol, Any}(), false)
CollectiveRegistry{K}(items::OrderedDict{Symbol, Any}) where {K} =
    CollectiveRegistry{K}(items, false)

isfrozen(reg::CollectiveRegistry) = reg.frozen

# ── read interface ────────────────────────────────────────────────────────────
Base.getindex(reg::CollectiveRegistry, name::Symbol) = reg.items[name]
Base.haskey(reg::CollectiveRegistry, name::Symbol) = haskey(reg.items, name)
Base.get(reg::CollectiveRegistry, name::Symbol, default) = get(reg.items, name, default)
Base.keys(reg::CollectiveRegistry) = keys(reg.items)
Base.values(reg::CollectiveRegistry) = values(reg.items)
Base.pairs(reg::CollectiveRegistry) = pairs(reg.items)
Base.length(reg::CollectiveRegistry) = length(reg.items)
Base.isempty(reg::CollectiveRegistry) = isempty(reg.items)
Base.iterate(reg::CollectiveRegistry, state...) = iterate(reg.items, state...)
Base.eltype(::Type{<:CollectiveRegistry}) = Pair{Symbol, Any}

# ── write interface ───────────────────────────────────────────────────────────
const _REGISTRY_DIRECT_MUTATION = "CollectiveRegistry does not support direct " *
    "assignment or deletion. Register callbacks with add_callback!(sim, cb; name) " *
    "before run! or the first time_step!; output writers are fixed when the " *
    "Simulation is constructed."

Base.setindex!(::CollectiveRegistry, _, ::Symbol) =
    throw(ArgumentError(_REGISTRY_DIRECT_MUTATION))
Base.delete!(::CollectiveRegistry, ::Symbol) =
    throw(ArgumentError(_REGISTRY_DIRECT_MUTATION))

"""
    register!(reg, name::Symbol, item) -> reg

Insert (or overwrite) `name => item`. Throws once the registry is frozen.
"""
function register!(reg::CollectiveRegistry{K}, name::Symbol, item) where {K}
    reg.frozen && throw(ArgumentError(
        "$(K) registry is frozen: callbacks and output writers must be registered " *
        "before run! or the first time_step!, on every rank (register on every rank " *
        "and put any rank-local test inside the body)."))
    reg.items[name] = item
    return reg
end

"""
    validate_and_freeze!(reg, signature, mismatch_message; comm = _default_comm()) -> reg

Collective; every rank in `comm` must call it together. Broadcast rank 0's
`signature(reg)` and raise `mismatch_message` on EVERY rank if any rank's
signature differs; then freeze `reg`. With no MPI or one rank, just freeze.
"""
function validate_and_freeze!(reg::CollectiveRegistry, signature::F,
        mismatch_message::AbstractString; comm = _default_comm()) where {F}
    if _collective_active(comm)
        local_signature = signature(reg)
        root_signature = root_value(() -> local_signature, comm, "registry signature")
        any_rank(local_signature != root_signature, comm) && error(mismatch_message)
    end
    reg.frozen = true
    return reg
end
