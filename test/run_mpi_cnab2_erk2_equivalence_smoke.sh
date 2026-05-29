#!/usr/bin/env bash
# Run the CNAB2 + ERK2 RHS/stage distributed-equivalence test under MPI.
#
# The contract it guards (each (l,m) mode is assembled entirely on its owning
# rank, with NO inter-rank communication) only has teeth at >=2 ranks: a 1-rank
# run owns every mode and can never expose a dropped mode or a result that
# silently depends on an Allreduce. This asserts that the distributed RHS/stage
# equals a purely-local recomputation for every owned mode, and that the owned
# modes partition the full spectral set. MPIEXEC_TIMEOUT bounds a hung
# collective into a non-zero exit instead of blocking forever.
#
# Usage:
#   test/run_mpi_cnab2_erk2_equivalence_smoke.sh
#   JULIA=/path/to/julia NRANKS=2 MPIEXEC_TIMEOUT=180 test/run_mpi_cnab2_erk2_equivalence_smoke.sh
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
    code   = "using Test, GeoDynamo; include(\"test/cnab2_rhs_distributed_equivalence.jl\")"
    launch = MPI.mpiexec
    launch() do mpi
        run(`$mpi -n $nranks $jl --project=. -e $code`)
    end
'
