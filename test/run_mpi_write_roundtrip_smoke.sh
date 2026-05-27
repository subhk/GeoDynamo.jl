#!/usr/bin/env bash
# Run the distributed NetCDF write→read assembly test under MPI.
#
# The contract it guards (each rank writing its pencil slab at the correct
# GLOBAL offset) only exists at >=2 ranks; a 1-rank run owns the whole grid and
# never exercises the offset logic. A misplaced/overlapping/gapped write
# silently corrupts the file rather than erroring, so this asserts a bit-exact
# global readback after a real collective write. MPIEXEC_TIMEOUT bounds a hung
# collective into a non-zero exit instead of blocking forever.
#
# Usage:
#   test/run_mpi_write_roundtrip_smoke.sh
#   JULIA=/path/to/julia NRANKS=2 MPIEXEC_TIMEOUT=180 test/run_mpi_write_roundtrip_smoke.sh
set -euo pipefail

: "${JULIA:=julia}"
: "${NRANKS:=2}"
: "${MPIEXEC_TIMEOUT:=180}"
export MPIEXEC_TIMEOUT NRANKS

cd "$(dirname "$0")/.."

# Use MPI.jl's bundled launcher (matches the linked MPI library) via a Julia
# driver; the inner ranks include the test file with GeoDynamo preloaded.
"$JULIA" --project=. -e '
    using MPI
    jl     = Base.julia_cmd()[1]
    nranks = parse(Int, ENV["NRANKS"])
    code   = "using Test, GeoDynamo; include(\"test/mpi_distributed_write_roundtrip.jl\")"
    launch = MPI.mpiexec
    launch() do mpi
        run(`$mpi -n $nranks $jl --project=. -e $code`)
    end
'
