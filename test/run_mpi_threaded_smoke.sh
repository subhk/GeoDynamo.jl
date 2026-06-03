#!/usr/bin/env bash
# Run the multi-rank + multi-thread deadlock regression test under MPI.
#
# The bug it guards (threaded field updates deadlocking at >=2 ranks) manifests
# as a HANG, not an exception, so the run is bounded by MPIEXEC_TIMEOUT: a
# regression makes the launcher kill the job after the timeout, yielding a
# non-zero exit (CI fail) instead of blocking forever. Needs >=2 ranks AND >=2
# threads/rank to bite.
#
# Usage:
#   test/run_mpi_threaded_smoke.sh
#   JULIA=/path/to/julia NRANKS=2 NTHREADS=2 MPIEXEC_TIMEOUT=180 test/run_mpi_threaded_smoke.sh
#
# Phase 2 (r×θ) requires an explicit process grid at nprocs>1; default to a
# θ-distributed / r-local grid (`NRANKSx1`, the Phase-1 layout this test was
# validated on). Override GEODYNAMO_PROC_GRID to exercise an r-distributed grid.
set -euo pipefail

: "${JULIA:=julia}"
: "${NRANKS:=2}"
: "${NTHREADS:=2}"
: "${MPIEXEC_TIMEOUT:=180}"
: "${GEODYNAMO_PROC_GRID:=${NRANKS}x1}"
export MPIEXEC_TIMEOUT NRANKS NTHREADS GEODYNAMO_PROC_GRID

cd "$(dirname "$0")/.."

# Use MPI.jl's bundled launcher (matches the linked MPI library) via a Julia
# driver; the inner ranks include the test file with GeoDynamo preloaded.
"$JULIA" --project=. -e '
    using MPI
    jl       = Base.julia_cmd()[1]
    nranks   = parse(Int, ENV["NRANKS"])
    nthreads = ENV["NTHREADS"]
    code     = "using Test, GeoDynamo; include(\"test/mpi_threaded_step_smoke.jl\")"
    launch   = MPI.mpiexec
    launch() do mpi
        run(`$mpi -n $nranks $jl -t$nthreads --project=. -e $code`)
    end
'
