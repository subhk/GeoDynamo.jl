# Clock — Oceananigans-style time/iteration tracker.
#
# The authoritative time/step live on the SolverState; this Clock is a mirror
# synced after each advance (see sync_clock!). It is correct to read between
# steps, not mid-step.
mutable struct Clock{T}
    time      :: T
    iteration :: Int
    stage     :: Int
    last_Δt   :: T
end

Clock{T}() where {T} = Clock{T}(zero(T), 0, 0, zero(T))

function Clock(; time = 0.0, iteration::Int = 0, stage::Int = 0, last_Δt = 0.0)
    T = promote_type(typeof(time), typeof(last_Δt))
    return Clock{T}(T(time), iteration, stage, T(last_Δt))
end

# state::SolverState — pull the authoritative values into the mirror.
function sync_clock!(clock::Clock{T}, state) where {T}
    clock.time      = T(state.time)
    clock.iteration = state.step
    return clock
end

function Base.show(io::IO, ::MIME"text/plain", c::Clock)
    print(io, "Clock(time=$(c.time), iteration=$(c.iteration), ",
              "stage=$(c.stage), last_Δt=$(c.last_Δt))")
end
