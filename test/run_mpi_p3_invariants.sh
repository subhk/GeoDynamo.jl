#!/usr/bin/env bash
# Run the Phase-3 distributed transpose and MPI layout invariants on both
# communication branches used by the CPU backend:
#   4x1 exercises the theta-distributed Alltoallv m-bridge.
#   2x2 exercises the r/theta-decomposed Allgatherv bridge.
#
# Usage:
#   test/run_mpi_p3_invariants.sh
#   JULIA=/path/to/julia MPIEXEC_TIMEOUT=300 test/run_mpi_p3_invariants.sh
set -euo pipefail

: "${JULIA:=julia}"
: "${MPIEXEC_TIMEOUT:=300}"
: "${OPENBLAS_NUM_THREADS:=1}"
export MPIEXEC_TIMEOUT OPENBLAS_NUM_THREADS

cd "$(dirname "$0")/.."
PROJECT="$(pwd)"

run_grid() {
    local grid="$1"
    echo "--- Phase-3 transpose and MPI invariants: grid=$grid (np=4) ---"
    GEODYNAMO_PROC_GRID="$grid" \
    GEODYNAMO_TEST_MPI_FINALIZE="true" \
    "$JULIA" --project="$PROJECT" -e "
        using MPI
        jl = Base.julia_cmd()[1]
        code = \"using Test, GeoDynamo; include(\\\"test/p3_transpose.jl\\\"); include(\\\"test/mpi_parallel_invariants.jl\\\")\"
        env_extra = Dict(
            \"GEODYNAMO_PROC_GRID\" => \"${grid}\",
            \"GEODYNAMO_TEST_MPI_FINALIZE\" => \"true\",
        )
        MPI.mpiexec() do mpi
            run(setenv(\`\$mpi -n 4 \$jl --project=${PROJECT} -e \$code\`,
                       merge(ENV, env_extra)))
        end
    "
}

run_grid 4x1
run_grid 2x2

echo "=== Phase-3 transpose and MPI invariants PASSED ==="
