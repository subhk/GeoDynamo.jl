#!/usr/bin/env bash
# Run the multi-rank CONTROL-PLANE invariants under MPI.
#
# The r×θ equivalence scripts prove the NUMERICAL core is bit-exact against the
# serial reference. They cannot reach the decisions taken around the step — stop
# conditions, health checks, output scheduling — because they run a fixed step
# count, attach no NaN, set no wall-time limit and use explicit writes. Four
# separate defects lived in exactly that gap, each one a rank-local decision
# gating a collective.
#
# Those bugs manifest as a HANG, not an exception, so the run is bounded by
# MPIEXEC_TIMEOUT: a regression makes the launcher kill the job after the timeout,
# yielding a non-zero exit (CI fail) instead of blocking forever. Needs >=2 ranks
# to bite; >=2 threads/rank additionally exercises the threaded implicit update.
#
# Usage:
#   test/run_mpi_control_plane.sh
#   JULIA=/path/to/julia NRANKS=2 NTHREADS=2 MPIEXEC_TIMEOUT=300 test/run_mpi_control_plane.sh
#
# Phase 2 (r×θ) requires an explicit process grid at nprocs>1; default to the
# θ-distributed / r-local layout. Override GEODYNAMO_PROC_GRID for an
# r-distributed grid.
set -euo pipefail

: "${JULIA:=julia}"
: "${NRANKS:=2}"
: "${NTHREADS:=2}"
: "${MPIEXEC_TIMEOUT:=300}"
: "${GEODYNAMO_PROC_GRID:=${NRANKS}x1}"
# Pin BLAS to 1 thread per rank: NRANKS ranks × NTHREADS Julia threads already
# fill the node; default multi-threaded BLAS oversubscribes cores.
: "${OPENBLAS_NUM_THREADS:=1}"
# The inner ranks must not finalize MPI while the testset is still running.
: "${GEODYNAMO_TEST_MPI_FINALIZE:=false}"
export MPIEXEC_TIMEOUT NRANKS NTHREADS GEODYNAMO_PROC_GRID OPENBLAS_NUM_THREADS
export GEODYNAMO_TEST_MPI_FINALIZE

cd "$(dirname "$0")/.."

echo "=== MPI control-plane invariants ==="
echo "Ranks:   $NRANKS   Threads/rank: $NTHREADS   Grid: $GEODYNAMO_PROC_GRID"
echo ""

# Use MPI.jl's bundled launcher (it matches the linked MPI library, and sets the
# library path the JLL mpiexec needs — a system mpiexec from a different MPI
# implementation fails to launch these ranks at all).
"$JULIA" --project=. -e '
    using MPI
    jl       = Base.julia_cmd()[1]
    nranks   = parse(Int, ENV["NRANKS"])
    nthreads = ENV["NTHREADS"]
    code     = "using Test, GeoDynamo; include(\"test/mpi_control_plane_invariants.jl\")"
    MPI.mpiexec() do mpi
        run(`$mpi -n $nranks $jl -t$nthreads --project=. -e $code`)
    end
'

echo ""
echo "=== MPI control-plane invariants PASSED ==="
