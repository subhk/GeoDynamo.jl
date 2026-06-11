"""
    mutable struct Clock{T}

Oceananigans-style tracker for simulation `time`, `iteration`, integrator
`stage`, and the last timestep `last_dt`. A `Clock` is attached to each
[`GeodynamoModel`](@ref) and synced after every advance via `sync_clock!`.

The authoritative time/step live on the solver state; this `Clock` is a mirror.
It is correct to read between steps, not mid-step.

`stage` mirrors Oceananigans' multi-stage field and is reserved for future use;
the current drivers are single-stage, so it stays `0`.
"""
mutable struct Clock{T}
    time::T
    iteration::Int
    stage::Int
    last_dt::T
end

Clock{T}() where {T} = Clock{T}(zero(T), 0, 0, zero(T))

function Clock(; time = 0.0, iteration::Int = 0, stage::Int = 0, last_dt = 0.0)
    T = promote_type(typeof(time), typeof(last_dt))
    return Clock{T}(T(time), iteration, stage, T(last_dt))
end

# state::SolverState — pull the authoritative values into the mirror.
function sync_clock!(clock::Clock{T}, state) where {T}
    clock.time = T(state.time)
    clock.iteration = state.step
    return clock
end

function Base.show(io::IO, ::MIME"text/plain", c::Clock)
    print(io, "Clock(time=$(c.time), iteration=$(c.iteration), ",
        "stage=$(c.stage), last_dt=$(c.last_dt))")
end

# ================================================================================
# Oceananigans-canonical `last_Δt` property alias for `last_dt`
# ================================================================================

function Base.getproperty(c::Clock, name::Symbol)
    name === :last_Δt && return getfield(c, :last_dt)
    return getfield(c, name)
end

function Base.setproperty!(c::Clock{T}, name::Symbol, x) where {T}
    name === :last_Δt && return setfield!(c, :last_dt, T(x))
    return setfield!(c, name, x)
end

Base.propertynames(c::Clock) = (fieldnames(Clock)..., :last_Δt)
