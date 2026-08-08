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

"""
    _collective_wtime(sim) -> Float64

Wall-clock seconds since `run!` started, made IDENTICAL on every rank.

`clock.time` and `clock.iteration` advance in lockstep, so schedules keyed off
them are automatically rank-consistent — a rank-local `time()` is not. Ranks
cross a `WallTimeInterval` boundary or the `wall_time_limit` microseconds apart,
which straddles an iteration boundary: one rank then enters the collective
`write_fields!` (or leaves `run!`) while the others take another step, and the
run hangs in the next collective instead of writing or stopping cleanly.
Broadcasting rank 0's elapsed time makes every rank decide the same way.

Every rank must reach this together, which they do: it is called from
`_run_callbacks!`/`_run_output_writers!` (once per step on every rank) and from
`wall_time_limit_exceeded`, whose default `IterationInterval(1)` schedule is
itself rank-consistent.
"""
function _collective_wtime(sim)
    sim._wall_start > 0.0 || return 0.0
    wtime = time() - sim._wall_start
    if MPI.Initialized()
        comm = get_comm()
        if comm !== nothing && MPI.Comm_size(comm) > 1
            buf = Float64[wtime]
            MPI.Bcast!(buf, 0, comm)
            wtime = buf[1]
        end
    end
    return wtime
end

"""
    _any_rank_flag(flag::Bool) -> Bool

`true` when `flag` is set on ANY rank (MPI.MAX reduction), else `flag` unchanged.

A stop decision taken from rank-local data — a NaN scan sees only this rank's
modes and radial slab — makes the offending ranks leave `run!` while the rest
call `time_step!` again and block forever in its next collective. Reducing the
flag first means all ranks stop together.
"""
function _any_rank_flag(flag::Bool)
    MPI.Initialized() || return flag
    comm = get_comm()
    (comm === nothing || MPI.Comm_size(comm) <= 1) && return flag
    return MPI.Allreduce(flag ? 1 : 0, MPI.MAX, comm) > 0
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

"""
    SpecifiedTimes(times...)

Schedule that fires once when simulation time reaches (or first passes) each of
the given times. Times are sorted and de-duplicated.
"""
mutable struct SpecifiedTimes <: AbstractSchedule
    times::Vector{Float64}
    _next::Int            # index of the next unfired entry
end
SpecifiedTimes(times::Real...) = SpecifiedTimes(unique(sort(Float64[times...])), 1)
SpecifiedTimes(times::AbstractVector{<:Real}) = SpecifiedTimes(unique(sort(Float64.(times))), 1)

function should_fire(s::SpecifiedTimes, ctx::_ScheduleContext)
    s._next > length(s.times) && return false
    if ctx.time >= s.times[s._next] - 1e-12
        # advance past every entry this step crossed; fire once
        while s._next <= length(s.times) && ctx.time >= s.times[s._next] - 1e-12
            s._next += 1
        end
        return true
    end
    return false
end
