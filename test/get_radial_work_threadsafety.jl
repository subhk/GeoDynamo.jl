# Regression test: get_radial_work! must be safe under concurrent distinct-key
# inserts.
#
# `_apply_solver_implicit_updates_threaded!` (src/timestep/driver.jl) spawns one
# `Threads.@spawn` task per active field, and each task calls `get_radial_work!`
# with a DISTINCT key on the SHARED `caches.radial_work` Dict. On the first
# solver step every key is absent, so the tasks perform concurrent `setindex!`
# inserts into the same Dict. A Julia `Dict` is not thread-safe: concurrent
# inserts that trigger a rehash/resize corrupt the table and surface as an
# intermittent `UndefRefError: access to undefined reference` (observed in the
# full suite at JULIA_NUM_THREADS>=2, striking a different solver-stepping test
# each run).
#
# This test reproduces the same access pattern directly: many distinct keys
# filled concurrently into one fresh TimestepCaches, repeated to expose the
# scheduling-dependent race. The race only manifests with >=2 threads; with one
# thread the loop still checks correctness.

using Test
using GeoDynamo

@testset "get_radial_work! thread-safe concurrent distinct-key fill" begin
    nr     = 16
    nkeys  = 200
    ntrial = 60
    keys   = [Symbol("rw_", i) for i in 1:nkeys]

    if Threads.nthreads() < 2
        @info "get_radial_work! race test running with $(Threads.nthreads()) thread; " *
              "the data race only manifests with >=2 threads (correctness still checked)."
    end

    errors = 0    # concurrent inserts corrupted the Dict (UndefRefError etc.)
    badfill = 0   # finished without throwing but produced wrong/missing entries

    for _ in 1:ntrial
        caches = GeoDynamo.TimestepCaches{Float64}()
        lens = Vector{Int}(undef, nkeys)
        try
            Threads.@threads for i in 1:nkeys
                work = GeoDynamo.get_radial_work!(caches, keys[i], nr)
                lens[i] = length(work.tmp_real)
            end
        catch
            errors += 1
            continue
        end
        if !(all(==(nr), lens) && length(caches.radial_work) == nkeys)
            badfill += 1
        end
    end

    @test errors == 0
    @test badfill == 0
end
