# ================================================================================
# Collectives — the ONE place in src/ that issues MPI collective operations.
# ================================================================================
#
# Every helper here:
#   * takes the communicator explicitly (default: the run's communicator, or
#     `nothing` when MPI is not initialized);
#   * has a serial fast path — with no communicator, MPI uninitialized, or one
#     rank it returns its input unchanged and issues no MPI call;
#   * otherwise asserts it is not inside the threaded implicit-update region and
#     issues exactly one MPI call.
#
# A static check (test/mpi_collective_static_checks.jl) fails the suite if any
# other file under src/ calls an MPI collective directly. The point is that the
# deadlock class "a rank-local decision gates a collective" has one home to
# audit, and a reader can grep for these names to find every collective.
# ================================================================================

"""
    _default_comm() -> Union{Nothing, MPI.Comm}

The run's communicator when MPI is initialized, else `nothing`.
"""
@inline _default_comm() = MPI.Initialized() ? get_comm() : nothing

"""
    _collective_active(comm) -> Bool

Whether a collective on `comm` would actually communicate: `comm` is a
communicator, MPI is initialized, and it has more than one rank.
"""
@inline _collective_active(comm) =
    comm !== nothing && MPI.Initialized() && MPI.Comm_size(comm) > 1

# ── Threaded implicit-update guard ────────────────────────────────────────────
#
# Set for the duration of `_apply_solver_implicit_updates_threaded!`'s spawned region,
# and only when it would actually be unsafe (more than one rank). Two `@spawn`'d tasks
# each issuing a collective can interleave differently across ranks, which DEADLOCKS with
# no error — the failure mode `_solver_multirank_magnetic_collective` exists to avoid by
# keeping known-collective configs sequential.
#
# That call-site denylist can only exclude the configurations somebody remembered to
# enumerate, so this flag adds a guard at the COLLECTIVE side: a reduction issued from
# inside the threaded region raises immediately, naming the site, instead of hanging.
#
# COVERAGE: every helper in this file checks the flag once it is past its serial fast
# path, and the static check guarantees there is no other collective in src/, so the
# net is complete. The kernel-facing delegates in solver/numerics.jl additionally check
# it BEFORE the fast path, so a threaded-region collective is reported at one rank too.
#
# SCOPE: the flag lives in the TASK-local storage of the task that arms it, not in a
# process-global `Ref`. A global was wrong twice over. Two `Simulation`s stepped
# concurrently in one process shared it, so one solver's threaded region rejected the
# other's perfectly ordered reductions. And a global has to be cleared in a `finally`,
# which ran as soon as `foreach(fetch, tasks)` rethrew from the FIRST failing task —
# disarming the guard while its siblings were still running unfetched, i.e. exactly when
# it was still needed. Task storage dies with the task: nothing to clear, nothing to race.
#
# A task spawned from inside the region does not inherit the flag. Nothing in the
# implicit updates nests spawns today, and inheriting is what would re-create the
# cross-solver false positive.
const _THREADED_UPDATE_KEY = :geodynamo_in_threaded_implicit_update

"""
    _in_threaded_implicit_update() -> Bool

Whether the CURRENT task is inside the threaded implicit-update region.

Reads `current_task().storage` directly instead of calling `task_local_storage()`:
the latter allocates the storage dict on first use, and this runs on the reduction
path of every task, including the ones that never arm the guard at all.
"""
@inline function _in_threaded_implicit_update()
    storage = current_task().storage
    storage === nothing && return false
    return get(storage, _THREADED_UPDATE_KEY, false)::Bool
end

"""
    _with_threaded_update_guard(f)

Run `f` with the collective guard armed for THIS task only, restoring the previous
value on the way out — including when `f` throws.
"""
function _with_threaded_update_guard(f)
    return task_local_storage(f, _THREADED_UPDATE_KEY, true)
end

"""
    _assert_no_collective_in_threaded_update(site)

Raise if an MPI collective is issued from inside the threaded implicit-update
region, where per-rank ordering can diverge and deadlock. `site` names the caller
so the error points at the offending collective rather than at the hang.
"""
@inline function _assert_no_collective_in_threaded_update(site)
    _in_threaded_implicit_update() || return nothing
    error("MPI collective issued from inside the threaded implicit-update region " *
          "($site). Two spawned tasks each issuing a collective can interleave " *
          "differently across ranks and deadlock. Either hoist the collective out of " *
          "the spawned region, or teach `_solver_magnetic_config_has_collective` " *
          "(timestep/driver.jl) about the configuration that reaches it so the field " *
          "solves stay sequential.")
end

# ── Reductions ────────────────────────────────────────────────────────────────

"""
    global_sum(x, comm = _default_comm())

Collective; every rank in `comm` must call it together. Sum of `x` over the ranks
(`MPI.Allreduce` with `MPI.SUM`). Serial fast path returns `x`.
"""
@inline function global_sum(x, comm = _default_comm())
    _collective_active(comm) || return x
    _assert_no_collective_in_threaded_update("global_sum")
    return MPI.Allreduce(x, MPI.SUM, comm)
end

"""
    global_max(x, comm = _default_comm())

Collective; every rank in `comm` must call it together. Maximum of `x` over the
ranks (`MPI.MAX`). Serial fast path returns `x`.
"""
@inline function global_max(x, comm = _default_comm())
    _collective_active(comm) || return x
    _assert_no_collective_in_threaded_update("global_max")
    return MPI.Allreduce(x, MPI.MAX, comm)
end

"""
    global_min(x, comm = _default_comm())

Collective; every rank in `comm` must call it together. Minimum of `x` over the
ranks (`MPI.MIN`). Serial fast path returns `x`.
"""
@inline function global_min(x, comm = _default_comm())
    _collective_active(comm) || return x
    _assert_no_collective_in_threaded_update("global_min")
    return MPI.Allreduce(x, MPI.MIN, comm)
end

"""
    global_sum!(buf, comm = _default_comm())

Collective; every rank in `comm` must call it together. In-place elementwise sum
of `buf` over the ranks (`MPI.Allreduce!` with `MPI.SUM`). Returns `buf`.
"""
@inline function global_sum!(buf, comm = _default_comm())
    _collective_active(comm) || return buf
    _assert_no_collective_in_threaded_update("global_sum!")
    MPI.Allreduce!(buf, MPI.SUM, comm)
    return buf
end

# ── Flags ─────────────────────────────────────────────────────────────────────

"""
    any_rank(flag::Bool, comm = _default_comm()) -> Bool

Collective; every rank in `comm` must call it together. `true` when `flag` is set
on ANY rank (`MPI.MAX` over 0/1). Use it before acting on a rank-local decision
that must be taken by all ranks — a NaN seen on one rank, a missing file, a
registry mismatch — so every rank leaves the same way.
"""
@inline function any_rank(flag::Bool, comm = _default_comm())
    _collective_active(comm) || return flag
    _assert_no_collective_in_threaded_update("any_rank")
    return MPI.Allreduce(flag ? 1 : 0, MPI.MAX, comm) > 0
end

"""
    all_ranks(flag::Bool, comm = _default_comm()) -> Bool

Collective; every rank in `comm` must call it together. `true` only when `flag`
is set on EVERY rank (`MPI.MIN` over 0/1). Use it for capability verdicts: a
capability half the ranks believe in is not a capability.
"""
@inline function all_ranks(flag::Bool, comm = _default_comm())
    _collective_active(comm) || return flag
    _assert_no_collective_in_threaded_update("all_ranks")
    return MPI.Allreduce(flag ? 1 : 0, MPI.MIN, comm) > 0
end

# ── Root patterns ─────────────────────────────────────────────────────────────

"""
    root_value(f, comm = _default_comm(), context = "root operation")

Collective; every rank in `comm` must call it together. Rank 0 evaluates `f()`;
its result — or its failure — is broadcast, so every rank returns the same value
or every rank throws `"<context> failed on rank 0: <error>"`. The failure path is
the point: a rank-0 exception that only rank 0 saw would leave the others blocked
in the next collective. The serial fast path evaluates `f()` directly and wraps a
failure in the same `"<context> failed on rank 0: …"` text, so callers see one
error shape at any rank count.
"""
function root_value(f::F, comm = _default_comm(),
        context::AbstractString = "root operation") where {F}
    if !_collective_active(comm)
        try
            return f()
        catch err
            error("$context failed on rank 0: $(sprint(showerror, err))")
        end
    end
    _assert_no_collective_in_threaded_update("root_value")
    value = nothing
    failure = ""
    if MPI.Comm_rank(comm) == 0
        try
            value = f()
        catch err
            failure = sprint(showerror, err)
        end
    end
    value, failure = MPI.bcast((value, failure), comm; root = 0)
    isempty(failure) || error("$context failed on rank 0: $failure")
    return value
end

"""
    run_on_root!(f, comm, context)

Collective; every rank in `comm` must call it together. Run `f()` on rank 0 for
its side effects and make its failure collective (see [`root_value`](@ref)).
"""
function run_on_root!(f::F, comm, context::AbstractString) where {F}
    root_value(f, comm, context)
    return nothing
end

"""
    root_broadcast!(buf::AbstractArray, comm = _default_comm())

Collective; every rank in `comm` must call it together. Overwrite `buf` on every
rank with rank 0's contents (`MPI.Bcast!`). Returns `buf`. Use it for a handful of
flags or numbers decided on rank 0 that every rank must act on identically.
"""
@inline function root_broadcast!(buf::AbstractArray, comm = _default_comm())
    _collective_active(comm) || return buf
    _assert_no_collective_in_threaded_update("root_broadcast!")
    MPI.Bcast!(buf, 0, comm)
    return buf
end

"""
    gather_root(x, comm = _default_comm())

Collective; every rank in `comm` must call it together. `MPI.Gather` of `x` to
rank 0: rank 0 receives a vector, other ranks receive `nothing`. Serial fast path
returns `[x]`.
"""
@inline function gather_root(x, comm = _default_comm())
    _collective_active(comm) || return [x]
    _assert_no_collective_in_threaded_update("gather_root")
    return MPI.Gather(x, comm; root = 0)
end

# ── Data movement ─────────────────────────────────────────────────────────────

"""
    allgather(x, comm = _default_comm())

Collective; every rank in `comm` must call it together. `MPI.Allgather` of the
scalar `x`: every rank receives the vector of all ranks' values, rank order.
Serial fast path returns `[x]`.
"""
@inline function allgather(x, comm = _default_comm())
    _collective_active(comm) || return [x]
    _assert_no_collective_in_threaded_update("allgather")
    return MPI.Allgather(x, comm)
end

"""
    allgatherv!(send, vbuf::MPI.VBuffer, comm = _default_comm())

Collective; every rank in `comm` must call it together. `MPI.Allgatherv!` of
`send` into the variable-count receive buffer `vbuf`. Returns `vbuf.data`. Serial
fast path copies `send` into `vbuf.data`.
"""
@inline function allgatherv!(send, vbuf::MPI.VBuffer, comm = _default_comm())
    if !_collective_active(comm)
        copyto!(vbuf.data, send)
        return vbuf.data
    end
    _assert_no_collective_in_threaded_update("allgatherv!")
    MPI.Allgatherv!(send, vbuf, comm)
    return vbuf.data
end

"""
    alltoallv!(sbuf::MPI.VBuffer, rbuf::MPI.VBuffer, comm = _default_comm())

Collective; every rank in `comm` must call it together. `MPI.Alltoallv!`
personalized exchange from `sbuf` into `rbuf`. Returns `rbuf.data`. Serial fast
path copies `sbuf.data` into `rbuf.data`.
"""
@inline function alltoallv!(sbuf::MPI.VBuffer, rbuf::MPI.VBuffer,
        comm = _default_comm())
    if !_collective_active(comm)
        copyto!(rbuf.data, sbuf.data)
        return rbuf.data
    end
    _assert_no_collective_in_threaded_update("alltoallv!")
    MPI.Alltoallv!(sbuf, rbuf, comm)
    return rbuf.data
end

"""
    barrier(comm = _default_comm())

Collective; every rank in `comm` must call it together. `MPI.Barrier`. Serial
fast path is a no-op. A barrier only orders ranks; it does not complete pending
non-blocking communication.
"""
@inline function barrier(comm = _default_comm())
    _collective_active(comm) || return nothing
    _assert_no_collective_in_threaded_update("barrier")
    MPI.Barrier(comm)
    return nothing
end
