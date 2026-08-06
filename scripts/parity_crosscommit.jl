#!/usr/bin/env julia
"""
Cross-commit bit-exact parity check — mechanism C of the SP-0 harness.

    julia --project=. scripts/parity_crosscommit.jl <refA> <refB>
    GEODYNAMO_PARITY_FULL=1 julia --project=. scripts/parity_crosscommit.jl <refA> <refB>

Builds each git ref in its own throwaway worktree, runs an identical driver
under each that evolves every fixture in the selected matrix and serializes
the resulting digests, then compares the two dumps bit-for-bit using THIS
checkout's comparator (`test/parity/state_digest.jl`) — so both dumps are
judged by one implementation even if the comparator itself changed between
the two refs.

This is NOT a Pkg.test() test. It needs two builds, and cross-platform
bit-exactness is not something CI can assert — a committed reference
snapshot would fail on macOS and on Julia 1.10/1.12 for reasons unrelated to
any refactor. It is a local pre-merge gate whose result belongs in the PR
body.

KNOWN LIMITATION: the driver includes `test/parity/*.jl` from inside each
worktree, so BOTH refs being compared must already contain the parity
harness (this sub-project, SP-0). Comparing against a ref that predates SP-0
will fail with a "no such file" error while building that worktree, not a
parity result. That is expected — every later sub-project branches from a
main that already has the harness — so it is not handled specially here.

PERFORMANCE: each worktree pays a one-time `Pkg.instantiate()` before the
driver can even load GeoDynamo — a fresh worktree lives at a path Julia has
never precompiled a project for, and the precompile cache is keyed in part
on that absolute path, so the GeoDynamo package itself (and a handful of its
heavier dependencies) must be recompiled even though the calling checkout
already has a warm cache; most dependencies ARE reused from the shared
depot. Measured locally: ~90 seconds. On top of that, one case takes
roughly 3 seconds to build, evolve, and digest: ~1 minute for the 16-case
default matrix, ~10 minutes for the 192-case full matrix, per ref. Budget a
few minutes for a default two-ref comparison and 20+ minutes for a
full-matrix one — do not assume a hang before then.

Digest dumps and the throwaway worktrees live under a scratch directory
(`GEODYNAMO_PARITY_SCRATCH`, or a fresh `mktempdir()`) and are never
committed.

Single-threaded by design (`--threads=1` is hard-coded for the worktree
runs): bit-exactness is only well-defined at a fixed thread count, and this
script exists to assert bit-exactness.
"""

using Serialization

const REPO = normpath(joinpath(@__DIR__, ".."))
const JULIA = joinpath(homedir(),
    ".julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia")

# Both digest dumps hold `ParityDigest.StateDigest`/`FieldBits` values, so the
# module that defines those types must be loaded into THIS process before
# either `deserialize` call in `main` below — Julia cannot reconstruct a
# custom struct whose defining module it has not seen yet.
#
# This MUST happen at top level, not inside `main()` (where the brief
# originally placed the equivalent line, just after the two `deserialize`
# calls instead of before them). `include` run from inside an
# already-executing function only makes its new definitions visible to code
# that runs in a LATER call — Julia pins a running function's method lookups
# to the "world age" at the moment it started, so a `digests_equal` call
# later in that SAME invocation of `main` still fails, now with a world-age
# `MethodError` ("method too new to be called from this world context")
# instead of the module-not-loaded error the brief's ordering produced.
# Verified empirically: reordering the include to just be the first line of
# `main()` reproduces exactly this MethodError. Hoisting the include to top
# level sidesteps it entirely — by the time `main()` is invoked (the script's
# last line), this statement has already run as an earlier, separate
# top-level form, so the call to `main()` starts in a world that already
# contains `ParityDigest.digests_equal`.
include(joinpath(REPO, "test", "parity", "state_digest.jl"))

# Driver executed INSIDE each worktree, as a separate `julia --project=<wt>`
# process. It must only use API that exists in both refs being compared —
# keep it to Pkg, the parity modules, and Serialization.
#
# It is written to a file OUTSIDE the worktree, in the scratch directory, and
# handed the worktree's path as ARGS[1]. Writing it inside the worktree (as
# originally drafted, under `<wt>/scripts/`) would dirty a directory that
# `git worktree remove` needs to discard cleanly, for no benefit — the
# driver's own location has no need to track the ref under test, only
# test/parity's location (inside the worktree) does, and that is threaded
# through explicitly via ARGS instead of via `@__DIR__`.
#
# It runs with `--project=<wt>` (that worktree's own Manifest.toml), NOT
# `--project=REPO`. Sharing the calling checkout's already-instantiated
# environment was considered as a way to skip the per-worktree
# `Pkg.instantiate()` below, but it is not just slower-vs-faster, it is
# WRONG: under `--project=REPO`, `using GeoDynamo` resolves to whatever
# commit REPO itself currently has checked out — completely ignoring `wt`
# and the ref this call is supposed to be testing. Every comparison would
# silently become "REPO vs REPO" regardless of the two refs given on the
# command line, which defeats the one thing mechanism C exists to do.
# `Pkg.instantiate()` here is the price of actually running the OTHER ref's
# code. Measured cost is modest, not a full dependency rebuild: the shared
# depot (`~/.julia`) already has every dependency version precompiled from
# the calling checkout, so a fresh worktree path only forces Julia to
# recompile the GeoDynamo package itself (there is nothing to share for
# that specifically, since Julia's precompile cache is keyed in part on the
# project's absolute path) — measured locally at 6-9 seconds, plus Pkg's own
# resolve/registry-check overhead. A full `--project=. julia -e
# 'using Pkg; Pkg.instantiate()'` run against a brand-new path measured
# ~90 seconds end-to-end (first touch of that exact path only); an
# already-warm-depot worktree plus a full 16-case default-matrix run
# measured 78.7 seconds total (instantiate + 16 evolve-and-digest cases).
const DRIVER = raw"""
using Pkg
Pkg.instantiate()

using Serialization

wt = ARGS[1]
out = ARGS[2]

include(joinpath(wt, "test", "parity", "state_digest.jl"))
include(joinpath(wt, "test", "parity", "fixtures.jl"))
using .ParityDigest
using .ParityFixtures
using MPI
MPI.Initialized() || MPI.Init()

dumps = Dict{String, Any}()
for case in ParityFixtures.select_matrix()
    st = ParityFixtures.evolve!(ParityFixtures.build_state(case))
    dumps[string(case)] = ParityDigest.digest_state(st)
end
serialize(out, dumps)
println("wrote $(length(dumps)) digests to $out")

# Deterministic, unconditional exit rather than falling off the end of the
# script and relying on Julia's normal atexit sequence. Not required by
# anything reproduced here — a standalone run of this exact driver (build,
# evolve 16 cases, digest, serialize) exited on its own in 78.7s with no
# hang and no error — but it is free insurance against the atexit path ever
# blocking on an MPI/PencilArrays/precompile-lock resource this script does
# not otherwise control, and it means a genuine failure earlier in the
# driver (which throws before reaching here) is never masked by whatever
# exit code an implicit fall-through would produce.
MPI.Initialized() && !MPI.Finalized() && MPI.Finalize()
exit(0)
"""

"""
    _worktree_registered(wt) -> Bool

Whether git currently considers `wt` a registered worktree of REPO, as
opposed to a plain directory that merely happens to sit at that path.
`git worktree remove` errors out (exit 128, "fatal: ... is not a working
tree") on the latter, so callers must check this before removing rather than
relying on `isdir` alone.
"""
function _worktree_registered(wt::String)
    listing = read(`git -C $REPO worktree list --porcelain`, String)
    wt_real = try
        realpath(wt)
    catch
        return false
    end
    for line in split(listing, '\n')
        if startswith(line, "worktree ")
            path = line[(length("worktree ") + 1):end]
            candidate = isdir(path) ? (try
                realpath(path)
            catch
                path
            end) : path
            candidate == wt_real && return true
        end
    end
    return false
end

"""
    _remove_worktree(wt)

Best-effort removal. Prefers `git worktree remove --force` when git
considers `wt` a registered worktree; otherwise falls back to a plain
`rm -rf`, since git refuses to touch a path it does not recognize (verified:
`git worktree remove --force` on a non-worktree directory exits 128 rather
than silently removing it). Never throws — cleanup must not mask whatever
error sent us here, and a directory left behind by a crashed prior run must
not wedge every subsequent run either.
"""
function _remove_worktree(wt::String)
    isdir(wt) || return nothing
    try
        if _worktree_registered(wt)
            run(`git -C $REPO worktree remove --force $wt`)
        else
            rm(wt; recursive = true, force = true)
        end
    catch e
        @warn "failed to remove worktree/directory; continuing" wt exception = e
    end
    return nothing
end

"""
    dump_ref(ref, scratch, driver_path) -> String

Build `ref` in a fresh worktree under `scratch`, run the driver in it, and
return the path to the digest dump it wrote.

The worktree path comes from `mktempdir`, not a name derived from `ref`: two
different refs can sanitize to the same string, and even two identical refs
(the `HEAD HEAD` self-check) would otherwise collide on one shared path and
one shared output file. `git worktree add` accepts an existing EMPTY
directory as its target, so handing it one from `mktempdir` (rather than
creating that path first and then removing it, as originally drafted) is
both simpler and race-free.
"""
function dump_ref(ref::String, scratch::String, driver_path::String)
    # cleanup=false: `_remove_worktree` below is the sole, deterministic
    # owner of this path's lifecycle. Leaving `mktempdir`'s default
    # cleanup=true active as well registers a SECOND, competing removal via
    # an `atexit` hook that fires later — against a path git (or a prior
    # `rm`) may have already deleted — which surfaced locally as a
    # `Base.Filesystem` "failed to clean up" warning with a raw
    # `ErrorException("schedule: Task not runnable")` stacktrace at process
    # exit. Harmless in that it did not change the exit code, but it has no
    # reason to run at all once cleanup is handled explicitly.
    wt = mktempdir(scratch; prefix = "wt-", cleanup = false)
    out = joinpath(scratch, "digest-" * basename(wt) * ".jls")
    run(`git -C $REPO worktree add --detach $wt $ref`)
    try
        run(`$JULIA --project=$wt --threads=1 $driver_path $wt $out`)
    finally
        _remove_worktree(wt)
    end
    return out
end

function main()
    length(ARGS) == 2 || error("usage: parity_crosscommit.jl <refA> <refB>")
    refa, refb = ARGS

    scratch = get(ENV, "GEODYNAMO_PARITY_SCRATCH", "")
    isempty(scratch) && (scratch = mktempdir())
    mkpath(scratch)
    println("scratch: $scratch")

    driver_path = joinpath(scratch, "_parity_driver.jl")
    write(driver_path, DRIVER)

    da = deserialize(dump_ref(refa, scratch, driver_path))
    db = deserialize(dump_ref(refb, scratch, driver_path))

    keys_a, keys_b = sort(collect(keys(da))), sort(collect(keys(db)))
    if keys_a != keys_b
        println("FAIL: case sets differ")
        println("  only in $refa: ", setdiff(keys_a, keys_b))
        println("  only in $refb: ", setdiff(keys_b, keys_a))
        exit(1)
    end

    failures = 0
    for k in keys_a
        ok, msg = ParityDigest.digests_equal(da[k], db[k])
        if ok
            println("  ok    $k")
        else
            failures += 1
            println("  DIFF  $k")
            println("        $msg")
        end
    end

    println()
    if failures == 0
        println("PASS: $(length(keys_a)) cases bit-identical between $refa and $refb")
        exit(0)
    else
        println("FAIL: $failures of $(length(keys_a)) cases differ")
        exit(1)
    end
end

main()
