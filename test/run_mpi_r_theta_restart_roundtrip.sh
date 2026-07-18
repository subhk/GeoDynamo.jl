#!/usr/bin/env bash
# Run the r×θ distributed restart round-trip across process grids.
#
# read_restart!'s physical-field slice must read each rank's LOCAL radial slab,
# not the full radial extent. That offset logic only exists at r_ranks>1, where a
# wrong slice silently corrupts a resumed run rather than erroring. This drives
# test/mpi_r_theta_restart_roundtrip.jl at:
#   1x1  (serial reference)
#   2x1  (θ-distributed, r-local — Phase-1 layout)
#   1x2  (r-distributed, single θ-rank)
#   2x2  (full r×θ 2D decomposition)
# Each rank writes its slab to a shared restart file, reads its slab back, and
# checks bit-exact equality with a global-index encoding. MPIEXEC_TIMEOUT bounds
# a hung collective into a non-zero exit instead of blocking forever.
#
# Usage:
#   test/run_mpi_r_theta_restart_roundtrip.sh
#   JULIA=/path/to/julia test/run_mpi_r_theta_restart_roundtrip.sh
set -euo pipefail

: "${JULIA:=julia}"
: "${MPIEXEC_TIMEOUT:=300}"
# Pin BLAS to 1 thread per rank — multi-threaded BLAS oversubscribes under MPI.
: "${OPENBLAS_NUM_THREADS:=1}"
export MPIEXEC_TIMEOUT OPENBLAS_NUM_THREADS

cd "$(dirname "$0")/.."
PROJECT="$(pwd)"
TEST_DRIVER="$PROJECT/test/mpi_r_theta_restart_roundtrip.jl"

echo "=== r×θ Distributed Restart Round-trip ==="
echo "Project: $PROJECT"
echo "Julia:   $JULIA"
echo ""

run_grid() {
    local grid="$1"
    local np="$2"
    local snap
    snap="$(mktemp -d "/tmp/rtheta_restart_${grid}_XXXX")"
    echo "--- grid=$grid (np=$np) ---"
    GEODYNAMO_PROC_GRID="$grid" \
    GEODYNAMO_TEST_MPI_FINALIZE="true" \
    RTHETA_RESTART_TMPDIR="$snap" \
    "$JULIA" --project="$PROJECT" -e "
        using MPI
        jl   = Base.julia_cmd()[1]
        code = \"using Test, GeoDynamo; include(\\\"${TEST_DRIVER}\\\")\"
        env_extra = Dict(
            \"GEODYNAMO_PROC_GRID\"        => \"${grid}\",
            \"GEODYNAMO_TEST_MPI_FINALIZE\" => \"true\",
            \"RTHETA_RESTART_TMPDIR\"      => \"${snap}\",
        )
        if ${np} == 1
            run(setenv(\`\$jl --project=${PROJECT} -e \$code\`, merge(ENV, env_extra)))
        else
            MPI.mpiexec() do mpi
                run(setenv(\`\$mpi -n ${np} \$jl --project=${PROJECT} -e \$code\`,
                           merge(ENV, env_extra)))
            end
        end
    "
    rm -rf "$snap"
    echo "--- done grid=$grid ---"
    echo ""
}

run_grid 1x1 1
run_grid 2x1 2
run_grid 1x2 2
run_grid 2x2 4

echo "=== r×θ Distributed Restart Round-trip PASSED ==="
