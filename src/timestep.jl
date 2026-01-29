# ================================================================================
# Timestepping Module with SHTns
# ================================================================================
#
# This module implements implicit-explicit (IMEX) time-stepping schemes for
# geodynamo simulations using spectral methods with SHTnsKit.
#
# ================================================================================
# MPI PARALLELIZATION - CRITICAL DESIGN PATTERN
# ================================================================================
#
# The spectral data is distributed across MPI processes using PencilArrays.
# Each process owns a subset of (l,m) modes defined by `lm_range`.
#
# PROBLEM: Different processes may own different numbers of modes.
#   - Process 0: lm_range = 1:50   (50 modes)
#   - Process 1: lm_range = 51:100 (50 modes)
#   - Process 2: lm_range = 101:145 (45 modes)  <-- fewer modes!
#
# If we use `for lm_idx in lm_range` with MPI collectives inside, processes
# will call Allreduce different numbers of times → DEADLOCK!
#
# SOLUTION: Use GLOBAL loop bounds with ownership check:
#
#   ```julia
#   nlm_total = u.nlm  # Total number of modes (same for all processes)
#
#   for lm_idx in 1:nlm_total  # ALL processes iterate same number of times
#       owns_mode = lm_idx in lm_range  # Check ownership
#
#       # Allocate buffers (all processes)
#       ur = zeros(T, nr)
#
#       # Fill data only if this process owns the mode
#       if owns_mode
#           ll = lm_idx - first(lm_range) + 1  # Local index
#           for r in r_range
#               ur[r] = u_real[ll, 1, local_r]
#           end
#       end
#
#       # ALL processes call Allreduce together (collective operation)
#       if multi
#           Allreduce!(ur, MPI.SUM, comm)  # Gathers data from owning process
#       end
#
#       # ... perform computation on gathered data ...
#
#       # Scatter result back only if this process owns the mode
#       if owns_mode
#           for r in r_range
#               u_real[ll, 1, local_r] = ur_new[r]
#           end
#       end
#   end
#   ```
#
# KEY INVARIANT: MPI collectives (Allreduce, Allgather, Barrier) must be called
# the same number of times by ALL processes. The global loop bounds ensure this.
#
# DEBUGGING MPI ISSUES:
#   1. If simulation hangs, likely MPI deadlock from unbalanced collective calls
#   2. Check that all loops containing MPI collectives use global bounds
#   3. Use `MPI.Comm_rank(comm)` to identify which process is stuck
#   4. Enable MPI_DEBUG environment variable for detailed tracing
#
# ================================================================================
# TIME-STEPPING SCHEMES
# ================================================================================
#
# EAB2 (Exponential Adams-Bashforth 2nd order):
#   - Treats diffusion implicitly using matrix exponential
#   - Uses Krylov subspace methods for exp(A) and φ₁(A) actions
#   - Nonlinear terms treated explicitly with AB2 extrapolation
#
# CNAB2 (Crank-Nicolson Adams-Bashforth 2nd order):
#   - Diffusion: Crank-Nicolson (implicit, 2nd order)
#   - Nonlinear: Adams-Bashforth (explicit, 2nd order)
#   - Requires solving tridiagonal systems per (l,m) mode
#
# ERK2 (Explicit Runge-Kutta 2nd order):
#   - Fully explicit scheme (for testing/comparison)
#   - CFL-limited timestep required
#
# ================================================================================

using MPI
using LinearAlgebra
using Dates
using JLD2

# Include submodules in dependency order

# State and NaN detection (no dependencies)
include("timestep/state.jl")

# EAB2 scheme (no dependencies)
include("timestep/eab2.jl")

# CNAB2 scheme (no dependencies)
include("timestep/cnab2.jl")

# ERK2 submodule - boundary conditions (no dependencies)
include("timestep/erk2/boundaries.jl")

# ERK2 submodule - influence matrix (depends on boundaries)
include("timestep/erk2/influence_matrix.jl")

# ERK2 submodule - matrix functions φ₁ and φ₂ (no dependencies)
include("timestep/erk2/matrix_functions.jl")

# ERK2 submodule - cache structures (depends on matrix_functions, boundaries, influence_matrix)
include("timestep/erk2/cache.jl")

# ERK2 submodule - cache I/O (depends on cache)
include("timestep/erk2/cache_io.jl")

# ERK2 submodule - staged integration (depends on cache, boundaries, matrix_functions)
include("timestep/erk2/staging.jl")

# Exports are handled by main module
