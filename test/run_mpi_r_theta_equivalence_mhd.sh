#!/usr/bin/env bash
# Run the r×θ MHD step equivalence test across all four grid layouts and compare.
#
# This script runs test/r_theta_equivalence_mhd.jl as a driver for each of:
#   1x1  (serial reference)
#   4x1  (theta-distributed, r-local)
#   1x4  (r-distributed, single theta-rank)
#   2x2  (full r×θ 2D decomposition)
#
# Both include_magnetic=true and include_composition=true are active, so this
# closes the gap left by run_mpi_r_theta_equivalence.sh (which only covered
# temperature + velocity).
#
# Each run writes a binary snapshot to SNAP_DIR/rtheta_mhd_sig_<grid>.bin.
# After all four runs, the comparison Julia script asserts physics-equivalence
# to < 1e-10 for all 12 tensors (6 field pairs × real+imag).
#
# Exit code 0 means all grids are equivalent.  Non-zero means failure.
#
# Usage:
#   test/run_mpi_r_theta_equivalence_mhd.sh
#   JULIA=/path/to/julia NRANKS=4 test/run_mpi_r_theta_equivalence_mhd.sh
set -euo pipefail

: "${JULIA:=$HOME/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia}"
: "${NRANKS:=4}"
: "${MPIEXEC_TIMEOUT:=300}"
export MPIEXEC_TIMEOUT

cd "$(dirname "$0")/.."
PROJECT="$(pwd)"
TEST_DRIVER="$PROJECT/test/r_theta_equivalence_mhd.jl"
COMPARE_SCRIPT="$PROJECT/test/r_theta_compare_snapshots_mhd.jl"

echo "=== r×θ MHD Equivalence Test (magnetic+composition) ==="
echo "Project:  $PROJECT"
echo "Julia:    $JULIA"
echo "Ranks:    $NRANKS"
echo ""

# Remove stale snapshots
SNAP_DIR="$(mktemp -d /tmp/rtheta_mhd_equiv_XXXX)"
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

echo "=== Cross-grid MHD Comparison ==="
RTHETA_TMPDIR="$SNAP_DIR" "$JULIA" --project="$PROJECT" "$COMPARE_SCRIPT"

echo ""
echo "=== r×θ MHD Equivalence Test PASSED ==="
