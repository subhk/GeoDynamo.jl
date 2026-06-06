abstract type AbstractSchedule end

mutable struct TimeInterval{T} <: AbstractSchedule
    interval::T
    _fired::Int   # count of the highest interval-multiple already fired
    TimeInterval(x) = new{typeof(x)}(x, 0)
end
TimeInterval() = TimeInterval(0.0)

struct IterationInterval <: AbstractSchedule
    interval::Int
end

mutable struct WallTimeInterval{T} <: AbstractSchedule
    interval::T      # wall-clock seconds between firings
    _last_fire::Float64
    WallTimeInterval(x) = new{typeof(x)}(x, 0.0)
end

# Lightweight context passed to should_fire — avoids coupling to Simulation
struct _ScheduleContext
    time::Float64
    step::Int
    wtime::Float64   # wall-clock seconds since run! started
end

function should_fire(s::TimeInterval, ctx::_ScheduleContext)
    s.interval <= 0 && return false
    ctx.time <= 0 && return false
    # Fire once for every interval boundary the simulation time has crossed,
    # not only when `time` lands exactly on a multiple. With a dt that does not
    # divide the interval (e.g. dt=0.03, interval=0.1) the time never equals a
    # multiple, so the old `mod(time, interval) < tol` test never fired. `n` is
    # the count of multiples k*interval <= time; firing when it advances catches
    # every crossing even if a single step overshoots one.
    n = floor(Int, ctx.time / s.interval + 1e-10)
    if n > s._fired
        s._fired = n
        return true
    end
    return false
end

function should_fire(s::IterationInterval, ctx::_ScheduleContext)
    s.interval <= 0 && return false
    return ctx.step % s.interval == 0
end

function should_fire(s::WallTimeInterval, ctx::_ScheduleContext)
    s.interval <= 0 && return false
    elapsed = ctx.wtime - s._last_fire
    if elapsed >= s.interval
        s._last_fire = ctx.wtime
        return true
    end
    return false
end
