# Integration ownership and cache lifecycle implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give time/step bookkeeping one owner and rebuild all dependent solver
operators consistently when run controls change, without changing public APIs.

**Architecture:** `SolverRuntime.timestep_state` owns the integration clock,
history interval and bootstrap flag. `SolverState.time/step` and the model's
`Clock` expose that owner; standalone clocks retain their existing constructors
and numeric types. A completed step updates the clock once. An explicit operator
dependency key covers timestep, scheme, diffusion coefficients and boundary
types. Rebuilds replace derived caches transactionally while retaining scratch
buffers, and release device copies through the existing host-update hook.
Grid/domain/precision remain fixed for the lifetime of a solver state.

**Tech Stack:** Julia, MPI, SHTnsKit, KernelAbstractions, Test.

Work in the existing checkout to preserve the earlier review fixes. Implementation
is authorized; no additional approval or commits are needed.

- [x] Capture pre-refactor parity digests for CNAB2, ERK2 and RK3.
- [x] Add isolated failing regressions for shared clock views, low-level stepping,
  fixed-dt operator dependency changes, scratch reuse and rollback.
- [x] Consolidate clock ownership and successful-step bookkeeping across CPU/GPU.
- [x] Centralize operator dependency checks and derived-cache invalidation.
- [x] Isolate the earlier review regression file from suite-wide imports.
- [x] Run focused regressions, exact parity comparison, full suite and two-rank
  restart/regression checks; document results and hardware limitations.

Scheme-specific cache types and checkpoint schema versioning are subsequent
increments, outside this first implementation.

Initial red run: 39 passed and 28 failed, with no setup errors. Failures covered
stale clock views and derived operators after fixed-dt parameter changes.

Validation uses the installed Julia 1.12.4 binary directly (the juliaup launcher
configuration is unreadable), with MPI transport access outside the sandbox.
The working full-suite invocation preloads packages at top level:

```sh
/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia --project --startup-file=no -e 'using Test, GeoDynamo; include("test/runtests.jl")'
```

The direct script launch stalled in Julia method inference before producing test
output and was stopped. The invocation above immediately passed the source checks
and continued into the runtime suite.

Verification results:

- Initial focused regression run after implementation: 67/67 passed. Added
  boundary-type, device-release and diagnostic-cadence cases afterward.
- Saved pre-refactor CNAB2, ERK2 and RK3 field/history digests match bit for bit,
  including time and step. This was repeated after moving ERK2 diagnostics to the
  completed-step boundary.
- Two MPI ranks (`GEODYNAMO_PROC_GRID=2x1`): integration checks 79/79 on each rank,
  earlier review checks 33/33 and 32/32, checkpoint round trips 52/52 on each rank;
  327 checks passed in total.
- Full suite exited successfully: 201 source checks, 216 package/parity checks,
  and 11,326 extended runtime checks passed (11,743 total). The 48 skipped or
  expected-broken checks are CUDA availability checks and the existing temporal
  order limitations. The extended suite includes all 88 new integration/cache
  regressions, all 43 earlier review regressions, restart and allocation guards.
- The full suite took 4m45s for package/parity tests and 10m05s for runtime tests.
  Its two printed parity-test failures are deliberate negative controls; the
  enclosing package/parity test set passes 216/216.
- Device tests use CPU-backed kernels. CUDA hardware is unavailable.
- `git diff --check` is clean. Changes remain uncommitted in the existing checkout.
