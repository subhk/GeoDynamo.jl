abstract type AbstractSchedule end

struct TimeInterval{T} <: AbstractSchedule
    interval :: T
end
TimeInterval() = TimeInterval(0.0)

struct IterationInterval <: AbstractSchedule
    interval :: Int
end

mutable struct WallTimeInterval{T} <: AbstractSchedule
    interval   :: T      # wall-clock seconds between firings
    _last_fire :: Float64
    WallTimeInterval(x) = new{typeof(x)}(x, 0.0)
end

# Lightweight context passed to should_fire — avoids coupling to Simulation
struct _ScheduleContext
    time  :: Float64
    step  :: Int
    wtime :: Float64   # wall-clock seconds since run! started
end

function should_fire(s::TimeInterval, ctx::_ScheduleContext)
    s.interval <= 0 && return false
    ctx.time <= 0   && return false
    return mod(ctx.time, s.interval) < 1e-10 * s.interval
end

function should_fire(s::IterationInterval, ctx::_ScheduleContext)
    s.interval <= 0 && return false
    return ctx.step % s.interval == 0
end

function should_fire(s::WallTimeInterval, ctx::_ScheduleContext)
    elapsed = ctx.wtime - s._last_fire
    if elapsed >= s.interval
        s._last_fire = ctx.wtime
        return true
    end
    return false
end
