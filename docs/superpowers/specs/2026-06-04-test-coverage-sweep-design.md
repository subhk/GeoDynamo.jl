# Test Coverage Improvement Sweep — Design

- **Date:** 2026-06-04
- **Status:** Approved (objective, approach, dead-code policy)
- **Objective:** Maximize measured serial line coverage. Target **88% floor / 90% stretch** (from 82.08% baseline) via a broad, tiered test-writing sweep plus targeted dead-code removal.

## Baseline (measured on HEAD)

- Tool: `Pkg.test(coverage=true)`, single serial run (np=1), Julia 1.11.1. Per-file `.cov` aggregated by a shell one-liner (Coverage.jl is not installed).
- Coverage: **7498 / 9135** executable lines = **82.08%**; 1637 uncovered.
- Suite: **2972 pass / 3 broken / 2975 total**, 5m12s, exit 0. The 2–3 broken are the known flaky IC-normalization tests and the `poloidal_solenoidality` spec (nondeterministic — see project memory).
- Distribution: 96% of uncovered lines (1569) live in **38 files with ≥10 uncovered each**; the other 68 lines are scattered across near-complete files and are out of scope (diminishing returns).

## Coverage arithmetic

- Denominator = 9135. To reach **88%** need ≥8039 covered (**+541**). To reach **90%** need ≥8222 covered (**+724**).
- Deleting dead *executable* lines shrinks the denominator and raises % with no new tests. This is a legitimate lever under the "maximize %" objective.

## Scope

**In scope:** serial (np=1) line coverage of `src/` — the measured number.

**Out of scope:**
- MPI-only paths (`parallel/process_grid.jl::make_subcomms`, `parallel/disttranspose_adapter.jl`) — never hit serially; covered separately by the MPI shell runners. *Exception:* the cheap serial wins `read_proc_grid` and `parse_proc_grid` edge cases.
- GPU paths (no GPU in the environment).
- The 68-line long tail of near-complete files (<10 uncovered each).

## Dead-code removals (each verified before deletion)

1. **`src/bcs/timestepping.jl`** (583 lines, currently 0% / not even instrumented). Not exported, **0 external callers**, function bodies are documented placeholders ("This is a placeholder — actual implementation depends on solver structure"); real BC enforcement is matrix-embedded in the field modules. **Action:** delete the file and its `include` at `src/bcs/bcs.jl:201`. Effect: **%-neutral** (it is not in the denominator) but removes 583 dead lines — code health. Verify: grep confirms zero callers / exports / test references (done); suite green after removal.
2. **numerics `cpu_*` vector helpers** (`cpu_fill_vector_coeff_buffer!`, `cpu_store_vector_coefficients!`, `cpu_extract_vector_component!`, `cpu_store_vector_components!`, ~150 lines). The coverage audit flagged these as possibly dead, **but project memory records them as live transform copy-helpers** (had `@threads` removed from them). **Action:** resolve the contradiction by tracing the caller graph first. Delete ONLY if grep proves zero callers; otherwise treat them as Tier-2 test targets. Effect if deleted: **%-positive**.

## Quality bar (non-negotiable)

- Every new test asserts **real expected behavior** (characterization tests against known-good values derived analytically or from a trusted reference), never "call it and assert `true`" or bare `@test_nowarn` — unless the function's actual contract is no-throw.
- TDD discipline: the assertion targets the *correct* result, so a test that would pass against buggy behavior is worthless and must be rewritten.
- New tests must be **deterministic**. The suite already carries flaky IC tests; do not add more. Seed any RNG, avoid global-state ordering dependence.
- Reuse existing fixtures for state/field construction (`test/erk2.jl`, `test/programmatic_boundaries.jl`, `test/temperature_boundary_numerical.jl`, `test/banded_operators.jl`).

## Tiered plan (ROI order)

| Tier | Files (uncovered) | Tactic | Est. reachable |
|---|---|---|---|
| **1** | `timestep/erk2.jl` (434) | Full-MHD+composition+BC integration ERK2 steps (≈150 physics-branch lines) + units: cache builders (Krylov & dense), BC factories, finalize/prepare with `bc_spec`, residual stats, rotating-inner-core BC | ~300 |
| **2** | `solver/numerics.jl` (247), `timestep/imex.jl` (69) | Pure-math units (`phi1_series`, `phi2_series`, `rcond_estimate`, `solver_factorize_banded`, `solver_build_banded_A`) + implicit-step units (magnetic / scalar / velocity / composition). Resolve `cpu_*` (delete-or-test). | ~180 |
| **3** | `fields/transforms.jl` (54), `physics/nonlinear.jl` (53), `physics/magnetic/solver.jl` (45) | Targeted units on uncovered branches | ~90 |
| **4** | bcs/topography cluster (~287): `velocity_coupling` 41, `topography_data` 37, `file_bc_loader` 36, `interpolation` 33, `stefan_condition` 29, `integration` 25, `gaunt_tensors` 24, `netcdf_io` 22, `bcs.jl` 22, `thermal_coupling` 16, `common` 14, `magnetic_coupling` 12, `programmatic` 10, `topography` 10 | Fixture-driven units (load / interpolate / couple) | ~150 |
| **5** | io/core/api/physics tail (~360): `io/restart` 37, `core/initial_conditions` 33, `io/netcdf` 21, `core/parameters` 21, `api/simulation` 20, `io/field_info` 19, `solver/state` 17, `physics/{temperature,composition,velocity}/solver` 17 each, `physics/composition/field` 22, `solver/backend` 12, `physics/magnetic/field` 12, `timestep/driver` 11, `api/initial_conditions` 14, `physics/topography` 14 + `read_proc_grid`/`parse_proc_grid` edge cases | Units + small integration | ~180 |

Execute in order; **stop adding tests once the 88% floor is reached**, then continue into higher tiers toward the 90% stretch only while ROI holds (each tier's reachable lines exceed the error/GPU/MPI residue).

## Verification

- After each tier: rerun the full suite with coverage (`Pkg.test(coverage=true)`), recompute % with the aggregation one-liner, confirm the suite is still green (0 failures, ≤3 broken) and coverage rose. Record the delta.
- Wire every new test file into `test/runtests.jl` `additional_tests`.
- Final gate: suite green, coverage ≥88% (stretch 90%), dead code removed, **no new flaky/broken tests**.

## Risks / ceilings

- Error / GPU / MPI branches across erk2 / numerics / imex form a hard serial ceiling (estimated ~150–200 unreachable lines). 90% is the realistic stretch; 95%+ is not attainable serially without faking it.
- The erk2 magnetic/composition integration tests need a full mini `SolverState` — the highest-effort item. Fallback: unit-test the branch helpers directly if the fixture proves too heavy.
- A full coverage run is ~5 min; per-tier re-measurement adds up — batch tiers to limit reruns.
- Suite nondeterminism: a transient 3-failure may be a flake — rerun before attributing it to new tests (project memory).

## Success criteria

- Measured serial coverage **≥88%** (stretch 90%).
- Suite green: 0 failures, ≤3 broken, **no new** broken/flaky tests.
- `bcs/timestepping.jl` removed; `cpu_*` helpers resolved (deleted if dead, else tested).
- Every new test asserts real behavior and is wired into `runtests.jl`.
