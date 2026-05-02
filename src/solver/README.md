# Solver Rewrite Track

These files and folders are the beginner-readable solver rewrite.

The rewrite will keep the algorithmic structure close to `DD_2DCODE`, but it
will use the existing Julia backend stack directly:

- `SHTnsKit`
- `PencilArrays`
- `PencilFFTs`

The solver API should also expose architecture as a first-class choice:

- `:cpu`
- `:gpu`

even while the current executable rewrite path remains CPU-backed.

It also needs to preserve three boundary features that are already present in
the current codebase:

- ICB topography
- OCB/CMB topography
- ICB phase change through the Stefan condition

The goal is not to build a second abstraction-heavy architecture. The goal is a
plain solver path with:

- one explicit parameter object
- one explicit backend object
- one explicit solver state
- one visible main loop
- local parameter access inside timestep code

Current rewrite layout:

- `../solver.jl`
- `interop.jl`
- `parameters.jl`
- `backend.jl`
- `state.jl`
- `numerics.jl`
- `mainloop.jl`
- `../timestep/imex.jl`
- `../timestep/erk2.jl`
- `../timestep/driver.jl`
- `../physics/velocity/solver.jl`
- `../physics/temperature/solver.jl`
- `../physics/composition/solver.jl`
- `../physics/magnetic/solver.jl`
- `../physics/nonlinear.jl`
- `../physics/topography.jl`
- `../diagnostics/solver.jl`

Current responsibilities:

- `../solver.jl`: include shell for the rewritten solver path
- `parameters.jl`: readable run configuration, file-backed BC settings, pretty printing, and active `SolverParameters` synchronization
- `interop.jl`: the narrow binding layer to shared `GeoDynamo` builders, backend types, and MPI helpers that the rewrite still reuses underneath
- `backend.jl`: SHTnsKit/backend assembly for the rewrite path, including solver-local config/domain builders, field and gradient-workspace construction, implicit-matrix assembly, solver-local timestep-state storage, and file-BC loading from `SolverParameters`
- `state.jl`: top-level solver state, solver-local runtime container, field handles, solver-local implicit/ERK2/EAB2 containers, and monitor storage
- `numerics.jl`: solver-owned ERK2 matrix functions, solver-local numeric tolerances, derivative-matrix/radial-laplacian builders, dense-row extraction, banded linear solves, dense band conversion, transform synchronization, scalar and vector SHTnsKit transform entry points, vector coefficient/component pack-unpack helpers, radial-profile and radial-derivative helpers, toroidal-poloidal curl kernels, ball vector regularity helpers, solver-local MPI/range/cache backend wrappers, solver-local Krylov exponential action and BC-vector backend wrappers, solver-local execution/timing control helpers, solver-local runtime flags for timing and ERK2 diagnostics, solver-local `(ℓ,m)` index caching and local-range lookup, structural scalar-field data/config access for buoyancy terms, solver-local backend aliases for the remaining shared legacy lock/container/module types, and vector backend steps used by the velocity and magnetic nonlinear paths, including local reset/vorticity/body-force/current-density/induction/inner-core-rotation kernels
- `../diagnostics/solver.jl`: solver-owned energy and solenoidal diagnostics, including scalar physical-field updates through the solver-local transform path and solver-local default NaN-check configuration
- `../timestep/imex.jl`: solver-owned CNAB2/EAB2 assembly and local implicit solve entry points
- `../physics/topography.jl`: ICB/OCB topography loading, activation, solver-local boundary identifiers, and Stefan-state wiring
- `../physics/velocity/solver.jl`: velocity initial conditions plus velocity-specific nonlinear preparation, forcing accumulation, and finalization helpers owned by the new solver path
- `../physics/temperature/solver.jl`: conductive thermal initial conditions plus the thermal nonlinear entry point owned by the new solver path
- `../physics/composition/solver.jl`: compositional initial conditions plus the composition nonlinear entry point owned by the new solver path
- `../physics/magnetic/solver.jl`: dipole-plus-perturbation magnetic initial conditions plus magnetic-specific nonlinear preparation/application helpers owned by the new solver path
- `../physics/nonlinear.jl`: explicit nonlinear-step entry point plus solver-owned scalar gradient, transform, and backend helper kernels, including the scalar batch loop, solver-local scalar synthesis and analysis wrappers, solver-local SHT cache accessors, direct solver SHTnsKit backend binding, solver-local coeff-buffer cache access, solver-local coeff reduction helper, direct scalar synthesis body, solver-local physical-slice extraction, solver-local coefficient writeback, coefficient-extraction wrapper, solver-local `(l,m)` index lookup, coefficient-fill kernel, physical-slice storage helper, solver-local use of backend MPI/range/cache wrappers instead of direct shared-helper calls, solver-local scalar work-array reset, scalar advection, internal-source application, ball scalar transform fallback, and the top-level nonlinear driver that dispatches into the velocity, magnetic, temperature, and composition subsystems
- `../timestep/erk2.jl`: solver-owned ERK2 boundary-side/spec containers, cache containers, cache assembly, stage buffers, finalize flow, and solver-local use of primitive builders and linear-algebra entry points instead of direct `GeoDynamo` backend calls
- `../timestep/driver.jl`: field bootstrapping, implicit step, clock updates, and health checks
- `mainloop.jl`: beginner-readable solver orchestration

Still planned after this stage:

- `io.jl`
- remaining replacement of the remaining backend helpers that still come from `GeoDynamo` modules
- hybrid GPU execution through the CUDA/SHTnsKit extension path, with transforms on GPU and the rest of the solver state still CPU-backed

Recent cleanup:

- the last legacy timestep file `src/timestep/cnab2.jl` is removed
- the shared implicit-matrix/CNAB2 helper surface now lives in `src/timestep/implicit.jl`

Current GPU-cache behavior:

- CPU SHTns configs eagerly build `SHTPlan` plus reusable transform output buffers
- GPU SHTns configs record their transform device but intentionally skip those CPU-only caches
- the solver GPU path therefore goes through the non-plan transform wrappers for GPU configs
- each `SolverRuntime` now owns a `TransformWorkspace` for solver-local scratch buffers such as gathered coefficient matrices and slice/component staging arrays
- `shtns_config._buffers` (`SHTnsBuffers`) still owns backend artifacts like SHTnsKit plans and output buffers, but solver scratch keys are kept in the runtime workspace cache once that workspace is installed
- `with_gpu_backend(...)` now provides a scoped backend override for tests and alternate backend integrations, including a `scratch_zeros` hook for transform-workspace allocations, so fake or experimental GPU backends do not need to mutate package internals by hand
- scalar scratch coefficient gather/scatter now routes through backend hooks on GPU-marked runtimes
- vector scratch coefficient gather/store and vector component extract/store now also route through backend hooks on GPU-marked runtimes
- the CUDA extension now registers explicit host-backed scratch callbacks for all of those hooks, so the real backend matches the solver API surface even though scratch state is not device-resident yet

The detailed migration plan lives in:

- `docs/plans/2026-04-21-dd2d-full-rewrite.md`
