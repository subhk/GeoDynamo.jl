#!/usr/bin/env bash
# Run the r×θ step equivalence test across all four grid layouts and compare.
#
# This script runs test/r_theta_equivalence.jl as a driver for each of:
#   1x1  (serial reference)
#   4x1  (theta-distributed, r-local — Phase-1 equivalent)
#   1x4  (r-distributed, single theta-rank)
#   2x2  (full r×θ 2D decomposition)
#
# Each run writes a binary snapshot to /tmp/rtheta_sig_<grid>.bin.
# After all four runs, the comparison Julia script asserts physics-equivalence
# to < 1e-10.
#
# Exit code 0 means all grids are equivalent.  Non-zero means failure.
#
# Usage:
#   test/run_mpi_r_theta_equivalence.sh
#   JULIA=/path/to/julia NRANKS=4 test/run_mpi_r_theta_equivalence.sh
set -euo pipefail

: "${JULIA:=$HOME/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia}"
: "${NRANKS:=4}"
: "${MPIEXEC_TIMEOUT:=300}"
export MPIEXEC_TIMEOUT

cd "$(dirname "$0")/.."
PROJECT="$(pwd)"
TEST_DRIVER="$PROJECT/test/r_theta_equivalence.jl"
COMPARE_SCRIPT="$PROJECT/test/r_theta_compare_snapshots.jl"

echo "=== r×θ Equivalence Test ==="
echo "Project:  $PROJECT"
echo "Julia:    $JULIA"
echo "Ranks:    $NRANKS"
echo ""

# Remove stale snapshots
SNAP_DIR="$(mktemp -d /tmp/rtheta_equiv_XXXX)"
export RTHETA_TMPDIR="$SNAP_DIR"
echo "Snapshots dir: $SNAP_DIR"
echo ""

# Helper: run driver under MPI via the Julia/MPI.jl bundled launcher
run_driver() {
    local grid="$1"
    local np="$2"
    echo "--- Running grid=$grid (np=$np) ---"
    GEODYNAMO_PROC_GRID="$grid" \
    GEODYNAMO_TEST_MPI_FINALIZE="true" \
    RTHETA_TMPDIR="$SNAP_DIR" \
    "$JULIA" --project="$PROJECT" -e "
        using MPI
        jl   = Base.julia_cmd()[1]
        code = \"using Test, GeoDynamo; include(\\\"${TEST_DRIVER}\\\")\"
        env_extra = Dict(
            \"GEODYNAMO_PROC_GRID\"         => \"${grid}\",
            \"GEODYNAMO_TEST_MPI_FINALIZE\"  => \"true\",
            \"RTHETA_TMPDIR\"               => \"${SNAP_DIR}\"
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
    echo "--- Done grid=$grid ---"
    echo ""
}

# Run each grid layout
run_driver 1x1 1
run_driver 4x1 "$NRANKS"
run_driver 1x4 "$NRANKS"
run_driver 2x2 "$NRANKS"

echo "=== Cross-grid Comparison ==="
RTHETA_TMPDIR="$SNAP_DIR" "$JULIA" --project="$PROJECT" "$COMPARE_SCRIPT"

echo ""
echo "=== r×θ Equivalence Test PASSED ==="
