# Standalone clocks retain their chosen numeric type. Model clocks use the
# solver's integration state directly, exposing time and last_dt as T.
mutable struct StandaloneClockState{T}
    time::T
    step::Int
    stage::Int
    last_dt::T
end

"""
    Clock{T}

Oceananigans-style view of simulation `time`, `iteration`, integrator `stage`,
and the last completed timestep `last_dt` (also available as `last_Δt`). A model's
clock reads and writes its solver's integration state, so low-level advances and
clock resets are immediately visible. Counters advance after a completed step.

`Clock()` and the keyword/positional constructors also create standalone clocks.
`stage` is reserved for future use and is zero between steps.
"""
mutable struct Clock{T}
    state::Union{StandaloneClockState{T}, SolverTimestepState}
end

Clock{T}(time, iteration, stage, last_dt) where {T} =
    Clock{T}(StandaloneClockState{T}(time, iteration, stage, last_dt))
Clock{T}() where {T} = Clock{T}(zero(T), 0, 0, zero(T))
Clock(time::T, iteration::Int, stage::Int, last_dt::T) where {T} =
    Clock{T}(time, iteration, stage, last_dt)

function Clock(; time=0.0, iteration::Int=0, stage::Int=0, last_dt=0.0)
    T = promote_type(typeof(time), typeof(last_dt))
    return Clock{T}(T(time), iteration, stage, T(last_dt))
end

# Compatibility helper: attach a detached clock to the solver instead of copying
# counters that would become stale on the next low-level advance.
function sync_clock!(clock::Clock, state)
    setfield!(clock, :state, state.runtime.timestep_state)
    return clock
end

@inline function Base.getproperty(c::Clock{T}, name::Symbol) where {T}
    state = getfield(c, :state)
    name === :time && return T(state.time)
    name === :iteration && return state.step
    name === :stage && return state.stage
    (name === :last_dt || name === :last_Δt) && return T(state.last_dt)
    return getfield(c, name)
end

@inline function Base.setproperty!(c::Clock{T}, name::Symbol, value) where {T}
    state = getfield(c, :state)
    name === :time && return setproperty!(state, :time, T(value))
    name === :iteration && return setproperty!(state, :step, Int(value))
    name === :stage && return setproperty!(state, :stage, Int(value))
    (name === :last_dt || name === :last_Δt) &&
        return setproperty!(state, :last_dt, T(value))
    return setfield!(c, name, convert(fieldtype(typeof(c), name), value))
end

Base.propertynames(::Clock, private::Bool=false) = private ?
    (:time, :iteration, :stage, :last_dt, :last_Δt, :state) :
    (:time, :iteration, :stage, :last_dt, :last_Δt)
