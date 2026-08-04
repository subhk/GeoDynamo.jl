# Bit-Exact Parity Harness (SP-0)

**Status:** approved (design phase)
**Date:** 2026-08-04
**Sub-project:** SP-0 of the v3.0.0 structural refactor program

## Goal

Build the differential correctness gate that every later refactor sub-project is
verified against: given two implementations of the same physics, prove they produce
**bit-identical** evolved solver state, or report exactly which coefficient diverged.

SP-0 changes no `src/` code. It adds `test/parity/*` and `scripts/parity_crosscommit.jl`,
and rewrites the two existing end-to-end probe files onto a shared fixture.

## Program context

The v3.0.0 program has six sub-projects. Each gets its own spec, branch, and PR.

| ID | Sub-project | API preserved | Gate |
|----|-------------|---------------|------|
| SP-0 | Bit-exact parity harness | n/a | self-tested (negative controls) |
| SP-1 | Split `integrate_solver_erk2_step!` (388 lines) | yes | mechanism C |
| SP-2 | Collapse temperature/composition field containers | **no** | mechanism B |
| SP-3 | Geometry as a dispatched type | **no** | mechanism B |
| SP-4 | Deduplicate `src/gpu/` against CPU physics | yes | mechanism C |
| SP-5 | Split `numerics.jl` / `transforms.jl` / `scalar_operators.jl` | yes | mechanism C |

Sequencing is dependency-driven, not preference:

- SP-0 first, so every later claim of "behaviour preserved" is provable rather than asserted.
- SP-1 before SP-2 because it is the cheapest real exercise of the harness; a harness
  flaw surfaces during the safest refactor instead of the riskiest.
- SP-4 after SP-2, because deduplicating GPU scalar code before the CPU scalar collapse
  means deduplicating code that is about to be deleted.
- SP-5 last, unconditionally. Pure file motion conflicts with every other diff; doing it
  early would force-rebase all five other sub-projects.

Version bump to v3.0.0 happens once, at the end, not per sub-project. SP-2 and SP-3 are
the breaking changes.

### Scope corrections carried in from exploration

Two claims made earlier in scoping were wrong and are corrected here so no sub-project
re-treads them:

- **Geometry is already funnelled.** There are ~12 `params.geometry ===` decision points,
  not the ~145 an initial grep suggested (the rest were docstrings and `:shell` strings).
  `cb3.jl:166` already passes `ball = state.parameters.geometry === :ball`, so the
  cb3-ball-poloidal bug recorded in the 2026-06-24 backlog is **fixed**. SP-3's payoff is
  therefore compile-time exhaustiveness and letting the GPU path reject `:ball` by
  dispatch instead of the hand-written `error()` calls at `gpu/device_state.jl:369` and
  `gpu/erk2_state.jl:165` — not bug-fixing.
- **The temp/comp *solver driver* duplication is already fixed.** PR #44 extracted it into
  `physics/scalar_field_solver_common.jl` (288 LOC), which is why `temperature/solver.jl`
  and `composition/solver.jl` are only 130 and 97 LOC. SP-2's remaining target is strictly
  the **field containers**: `temperature/field.jl` (699 LOC) and `composition/field.jl`
  (591 LOC), whose structs are field-for-field identical apart from the payload field name
  (`temperature` / `composition`), the θ-derivative field name
  (`theta_derivative_matrix` / `∂θ`), and the declared position of `internal_sources`.
  Both already subtype `AbstractScalarField{T}` and both already define
  `get_main_physical_field`, so the abstraction seam exists and is simply unused.

## Motivation

The chosen correctness gate for the program is bit-exactness of the evolved state. Nothing
in the repo currently provides it.

What exists and why it is not sufficient:

- `test/scalar_bc_timestepper_parity.jl` and `test/velocity_magnetic_bc_timestepper_parity.jl`
  are **invariant** probes. They assert a physical property — boundary residual near zero —
  on the evolved state. They catch a physics regression. They cannot prove that a refactored
  implementation matches the original, because any implementation satisfying the invariant
  passes.
- The GPU parity tests (`gpu_bc_combo_parity.jl`, `gpu_scalar_physics_parity.jl`,
  `gpu_phase5n_solver_step.jl`) compare the device path against CPU at `atol=1e-12`,
  `rtol=1e-10`. Tolerance-based, and scoped to the GPU path only.
- `io/restart.jl` is NetCDF over a `Dict{String, Any}` of individual fields. It covers
  neither whole-`SolverState` nor a guaranteed bit-exact round-trip.

Committed golden snapshots are ruled out. The repo has already been bitten by
floating-point reference snapshots not being bit-exact across platforms; a committed
reference would fail CI on macOS and on Julia 1.10/1.12 for reasons unrelated to any
refactor. The comparison must therefore be **same-machine, same-session**.

## Design

### Unit structure

```
test/parity/fixtures.jl      ──►  test/parity/state_digest.jl  ──┬──►  scripts/parity_crosscommit.jl   (C)
(builds SolverStates)             (walks state, compares bits)   └──►  test/parity/ab_harness.jl       (B)
```

`state_digest.jl` is the only unit that knows about bits. `fixtures.jl` is the only unit
that knows about `SolverParameters`. Neither front-end depends on the other, which is what
lets SP-5 use mechanism C alone and SP-2 use mechanism B alone.

### Why two mechanisms

One mechanism does not cover six sub-projects, because SP-2 and SP-3 break the public API.

**Mechanism C — cross-commit dump-and-diff.** Build two git refs in separate worktrees,
run a fixed driver under each that dumps the digest of an evolved state, compare the dumps.
Requires that one driver script runs unchanged against both refs, so it works only where
the API survives: SP-1, SP-4, SP-5. It is the *only* meaningful gate for SP-5, where there
is no behavioural change to A/B — only file motion.

**Mechanism B — in-tree A/B.** Keep both implementations live in `src/` during the
sub-project behind dispatch, assert identical digests from the same seed, delete the legacy
path in that sub-project's final commit. Required for SP-2 and SP-3, where a single driver
cannot reference a type that exists on only one side. Runs in normal CI.

A third option — freezing a copy of the current implementation under `test/refimpl/` — was
rejected. `integrate_solver_erk2_step!` reaches into `SolverState`, the ERK2 caches, and the
boundary descriptors, all of which SP-1 and SP-2 also modify, so the frozen copy would not
stay frozen. It also risks tripping `precompile_syntax.jl`, which walks package source.

### `state_digest.jl`

```
digest_state(state::SolverState) -> StateDigest
```

Walks `state.fields` in deterministic declared-field order and emits:

- every `SHTnsSpecField`: `parent(data_real)` and `parent(data_imag)`
- every `SHTnsPhysField`
- every component of every `SHTnsVectorField`
- `prev_nonlinear` for each field that carries one
- `state.time` and `state.step`

`prev_nonlinear` is included deliberately. It is not derived state — it carries the CNAB2
history, and a refactor that corrupts it yields a state that looks correct for exactly one
step and diverges afterward.

Physical fields are included alongside spectral ones. They are derived, so in principle
redundant, but they are cheap to digest and a mismatch confined to a physical field
localises a transform bug immediately instead of after bisection.

**Comparison is on `reinterpret(UInt64, ...)`**, not `==` and not `≈`. `==` gives the wrong
answer twice for this purpose: `-0.0 == 0.0` is `true`, and `NaN == NaN` is `false`. For a
differential gate, a `NaN` occupying the same slot on both sides is a **pass** — objecting
to it is the invariant probes' job, not this one's.

`StateDigest` carries both a summary hash, for a fast assert, and the raw arrays, so a
failure reports field name, mode index, radial index, both values, and ULP distance rather
than "differs". The summary hash is `Base.hash` folded over the reinterpreted `UInt64`
words together with each array's name and size, so two arrays that differ only in shape
cannot collide. It is a convenience for fast rejection only — a digest equality claim is
always confirmed against the raw arrays before the test reports a pass, so hash collision
cannot produce a false green.

**Comparability metadata** is stamped into every digest and checked *before* any array
comparison: `Threads.nthreads()`, `MPI.Comm_size(MPI.COMM_WORLD)`, `Sys.WORD_SIZE`,
`VERSION`, and the `SolverParameters` used. A mismatch there fails as *"digests not
comparable"*, distinct from *"physics changed"*. Bit-exactness is only well-defined at
fixed thread and rank count, and without this distinction a scheduling difference presents
as a physics regression.

### `fixtures.jl`

Extracted from the near-duplicate `_sbp_state` (in `scalar_bc_timestepper_parity.jl`) and
`_vm_state` (in `velocity_magnetic_bc_timestepper_parity.jl`). Parameterised over:

- timestepper: CNAB2, ERK2, RK3
- scalar BC code: 1 (DD), 2 (DN), 3 (ND), 4 (NN)
- velocity wall code: 1 (NS/NS), 2 (NS/SF), 3 (SF/NS), 4 (SF/SF)
- `include_magnetic`: true/false
- `include_composition`: true/false

Grid is the one both existing probes already use — `lmax=4, mmax=4, nlat=12, nlon=24,
nr=16, nr_inner=4, radius_ratio=0.35` — small enough that the full matrix is affordable and
already demonstrated to exercise the ERK2 magnetic Robin, ERK2 scalar Neumann, and ERK2
l=0 NN bugs.

Perturbation is deterministic via `MersenneTwister(seed)`, identical in form to the
existing files.

**Step count.** Fixtures are evolved **4 steps** before digesting. Fewer than 2 is
unacceptable: CNAB2's `prev_nonlinear` history does not participate until the second step,
so a single-step gate is blind to exactly the class of corruption that motivated including
`prev_nonlinear` in the digest. Four matches the trajectory length the existing
`gpu_phase6_run.jl` gate already uses.

**Matrix size.** The full cross-product is 3 timesteppers x 4 scalar BC codes x 4 wall
codes x 2 magnetic x 2 composition = 192 configurations, and a differential run builds two
`SolverState`s per configuration — 384 states. That is too slow for a routine gate even
with the decomposition memoized. The harness therefore exposes two matrices:

- `PARITY_MATRIX_DEFAULT` — a covering subset in which every level of every factor appears
  at least once and every pair of factors is exercised at least once (pairwise coverage),
  approximately 16 configurations. This is what runs by default.
- `PARITY_MATRIX_FULL` — all 192, opt-in via an environment flag, run once per sub-project
  before its PR rather than on every invocation.

Both are declared as data, so a sub-project that suspects a specific interaction can name
its own subset without touching harness code.

Both existing probe files are rewritten onto this fixture as part of SP-0. This is what
makes SP-0 a net simplification rather than pure addition.

### `scripts/parity_crosscommit.jl` (mechanism C)

Takes two git refs. For each ref: `git worktree add` at that ref, run a driver in a fresh
Julia process that builds each fixture in the selected matrix, steps it 4 times, and writes
a digest file. Then compare the two digest files and report.

Digest files are written to the session scratchpad and are never committed. They are not
portable across machines, and committing one would recreate precisely the golden-file
problem this design exists to avoid.

This is not a `Pkg.test()` test. It is a local pre-merge gate, run manually and reported in
the PR body, because it needs two builds and because cross-platform bit-exactness is not
something CI can assert.

### `test/parity/ab_harness.jl` (mechanism B)

A `@testset` generator. Given two constructor functions and two step functions, it builds
both from the same seed, steps both, and asserts digest equality. SP-2 and SP-3 each add a
temporary caller file and delete it in that sub-project's final commit.

This one does run in CI, single-threaded.

## Correctness gate for SP-0 itself

The harness is what everything else is trusted against, so a green result is its output and
a comparator that cannot fail is worse than no comparator. SP-0 therefore ships negative
controls:

1. Perturb one spectral coefficient by 1 ULP; assert the comparator **fails**.
2. Change a `+0.0` to `-0.0`; assert the comparator **fails**.
3. Place identical `NaN`s in the same slot on both sides; assert the comparator **passes**.
4. Change `Threads.nthreads()` metadata; assert it fails as *not comparable*, not as a
   physics difference.
5. Two independent builds of the same ref; assert the digests match (guards against the
   harness itself being nondeterministic).

## Constraints

- **Threads and ranks pinned.** The bit-exact gate runs single-threaded, single-rank. That
  is the only regime where bit-exactness is well-defined; reduction order changes under
  `@threads` and under MPI. Multi-thread and multi-rank coverage remains with the existing
  invariant probes and MPI equivalence tests, which are tolerance-based.
- **No `src/` changes.** If SP-0 turns out to need one, that is a signal the design is
  wrong; stop and revise the spec rather than widen scope.
- **Communicator budget.** A differential run allocates two `SolverState`s per
  configuration. Each `SolverState` historically leaked four MPI communicators, and the
  suite has previously exhausted MPICH's 2048 ceiling, failing 21 unrelated downstream
  files. The fix — memoizing the pencil decomposition per grid — is commit `5d552327`,
  currently in open PR #107 and **not** in `origin/main`. **SP-0 is blocked on #107
  merging.**
- **Static checks pin source text.** `test/*_static_checks.jl` (696 LOC across 8 files)
  assert against literal source. SP-0 does not touch the files they pin, but SP-2 and SP-5
  will, and those specs must budget for rewriting them. A rewritten pin proves nothing on
  its own and must be paired with a behavioural assertion.

## Out of scope

- Adding a CUDA runner to `ci.yml`. The GPU parity tests already exist and run the device
  path on `Array`; the gap is hardware, not code, and no local CUDA device is available.
- Any change to the existing invariant probes' assertions. SP-0 moves them onto a shared
  fixture; it does not weaken or strengthen what they check.
- Multi-rank bit-exactness. Covered by existing MPI equivalence tests at tolerance.

## Deliverables

- `test/parity/state_digest.jl`
- `test/parity/fixtures.jl`
- `test/parity/ab_harness.jl`
- `test/parity/digest_negative_controls.jl` (the five controls above)
- `scripts/parity_crosscommit.jl`
- `test/scalar_bc_timestepper_parity.jl` — rewritten onto the shared fixture, assertions unchanged
- `test/velocity_magnetic_bc_timestepper_parity.jl` — rewritten onto the shared fixture, assertions unchanged
- `test/runtests.jl` — register the new test files

## Success criteria

- Full suite green via `Pkg.test()` (redirect to a file; piping through `tail` masks the
  Julia exit code).
- All five negative controls pass.
- `scripts/parity_crosscommit.jl HEAD HEAD` reports bit-identical across
  `PARITY_MATRIX_FULL` (all 192). Comparing a ref against itself is the harness's own
  determinism check, so this one run uses the full matrix rather than the default subset.
- The two rewritten probe files assert exactly what they asserted before, verified by
  reading the diff rather than by the suite going green.
