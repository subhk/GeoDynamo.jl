# Test Coverage Improvement Sweep — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise measured serial line coverage from 82.08% to an 88% floor / 90% stretch by adding real-behavior tests across the 38 highest-gap files and removing verified-dead code.

**Architecture:** A tiered, ROI-ordered sweep. Each tier adds focused tests, wires them into `test/runtests.jl`, then re-measures coverage with a reproducible aggregation script. High-certainty units (φ-function math, process-grid parsing, dead-code deletion) have complete code below; the broad per-file sweep (Tiers 3–5) follows a rigorous per-file *characterization procedure* because the exact assertions depend on reading each file's uncovered lines — pre-guessing them would produce untested, worthless tests.

**Tech Stack:** Julia 1.11.1, `Test` stdlib, `Pkg.test(coverage=true)`, `LinearAlgebra`, SHTnsKit, PencilArrays. Direct Julia binary: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia` (the `julia` shim is broken — see project memory). Run tests with the Bash sandbox disabled.

**Baseline (measured HEAD):** 7498 / 9135 = 82.08%; suite 2972 pass / 3 broken / 2975, exit 0. Spec: `docs/superpowers/specs/2026-06-04-test-coverage-sweep-design.md`.

**Conventions used throughout:**
- `JL` = `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia`
- Run one test file standalone: `$JL --project=. -e 'using Test, GeoDynamo; include("test/<file>.jl")'`
- Full suite: `$JL --project=. -e 'using Pkg; Pkg.test()'`
- All new test files MUST be added to the `additional_tests` tuple in `test/runtests.jl`.
- Quality bar: every `@test` asserts a known-correct value (analytic or trusted-reference). No bare `@test true` / `@test_nowarn` unless the contract is literally no-throw. Deterministic only — seed any RNG.

---

## Phase 0 — Reproducible coverage tooling

### Task 0.1: Coverage aggregation script

**Files:**
- Create: `scripts/cov_aggregate.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Aggregate Julia --code-coverage .cov files into per-file + global line coverage.
# Assumes a SINGLE coverage run is present (clean stale .cov first). Picks the
# dominant PID so a stray older .cov can't pollute the numbers.
# Usage: scripts/cov_aggregate.sh [src_dir]   (default: src)
set -euo pipefail
SRC="${1:-src}"
PID=$(find "$SRC" -name '*.cov' | sed -E 's/.*\.([0-9]+)\.cov/\1/' | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')
[ -z "${PID:-}" ] && { echo "no .cov files under $SRC — run tests with coverage first"; exit 1; }
tmp=$(mktemp)
for f in $(find "$SRC" -name "*.$PID.cov"); do
  awk '{ if ($1 ~ /^[0-9]+$/) { tot++; if ($1+0>0) cov++ } }
       END { s=FILENAME; sub(/\.[0-9]+\.cov$/,"",s); if(tot>0) printf "%5.1f %6d %6d %s\n", 100*cov/tot, cov, tot, s }' "$f"
done | sort -n > "$tmp"
echo "=== per-file coverage (low -> high): pct cov tot file ==="
cat "$tmp"
echo "=== files by absolute uncovered lines ==="
awk '{printf "%5d uncov (%4.1f%%) %s\n", $3-$2, $1, $4}' "$tmp" | sort -rn | head -25
echo "=== GLOBAL ==="
awk '{c+=$2;t+=$3} END{printf "covered=%d total=%d pct=%.2f%%\n",c,t,100*c/t}' "$tmp"
rm -f "$tmp"
```

- [ ] **Step 2: Make executable and document the measure cycle**

Run: `chmod +x scripts/cov_aggregate.sh`

The full measure cycle (used after each phase):
```bash
JL=~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia
find . -name '*.cov' -delete                                   # clean stale
$JL --project=. -e 'using Pkg; Pkg.test(coverage=true)' 2>&1 | tail -5
scripts/cov_aggregate.sh src
```
Expected: global line ≈ `82.08%` on first run (the baseline).

- [ ] **Step 3: Commit**

```bash
git add scripts/cov_aggregate.sh
git commit -m "test: add coverage aggregation script"
```

---

## Phase 1 — Remove verified-dead code

### Task 1.1: Delete dead `bcs/timestepping.jl`

**Files:**
- Delete: `src/bcs/timestepping.jl` (583 lines)
- Modify: `src/bcs/bcs.jl:201` (remove its `include`)

- [ ] **Step 1: Re-verify it is dead (must print zeros / nothing)**

Run:
```bash
for fn in update_boundary_conditions_for_timestep! apply_boundary_conditions_to_rhs! \
          enforce_boundary_conditions_in_solution! log_boundary_condition_status \
          compute_boundary_condition_residual; do
  echo "$fn: $(grep -rn "$fn" src/ test/ --include='*.jl' | grep -v 'src/bcs/timestepping.jl' | wc -l | tr -d ' ') external refs"
done
```
Expected: every line prints `0 external refs`. If ANY is non-zero, STOP — it is not dead; convert to a test target instead.

- [ ] **Step 2: Remove the include line**

In `src/bcs/bcs.jl`, delete the line:
```julia
include("timestepping.jl")     # Integration with timestepping
```

- [ ] **Step 3: Delete the file**

Run: `git rm src/bcs/timestepping.jl`

- [ ] **Step 4: Verify the package still loads and suite is green**

Run: `$JL --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -6`
Expected: `Testing GeoDynamo tests passed`, pass count ≈ 2972, 0 failures, ≤3 broken. (No `UndefVarError` from the removed include.)

- [ ] **Step 5: Commit**

```bash
git add -A src/bcs/
git commit -m "refactor: remove dead bcs/timestepping.jl (0 callers, placeholder bodies)"
```

---

## Phase 2 — High-certainty unit tests (complete code)

### Task 2.1: φ-function series + rcond unit tests (`numerics.jl`)

**Files:**
- Create: `test/numerics_phi_functions.jl`
- Modify: `test/runtests.jl` (wire in)

- [ ] **Step 1: Confirm symbol access**

Run: `$JL --project=. -e 'using GeoDynamo; println(isdefined(GeoDynamo, :phi1_series), isdefined(GeoDynamo, :phi2_series), isdefined(GeoDynamo, :rcond_estimate))'`
Expected: `truetruetrue`. If any `false`, grep `src/solver/numerics.jl` for the defining module and qualify accordingly (e.g. `GeoDynamo.<mod>.phi1_series`).

- [ ] **Step 2: Write the tests**

```julia
using Test
using LinearAlgebra

# phi1(z) = (e^z - 1)/z ,  phi1(0) = 1
# phi2(z) = (e^z - 1 - z)/z^2 , phi2(0) = 1/2
# Both series operate on a matrix; a scalar multiple of I must reproduce the scalar value on the diagonal.
@testset "phi-function Taylor series (numerics.jl)" begin
    for z in (-0.30, -0.05, 0.10, 0.25)
        A = z * Matrix{Float64}(I, 3, 3)
        phi1_exact = (exp(z) - 1) / z
        phi2_exact = (exp(z) - 1 - z) / z^2
        @test GeoDynamo.phi1_series(A) ≈ phi1_exact * Matrix{Float64}(I, 3, 3) atol = 1e-12
        @test GeoDynamo.phi2_series(A) ≈ phi2_exact * Matrix{Float64}(I, 3, 3) atol = 1e-12
    end
    # Zero-matrix limits
    Z = zeros(3, 3)
    @test GeoDynamo.phi1_series(Z) ≈ Matrix{Float64}(I, 3, 3) atol = 1e-14
    @test GeoDynamo.phi2_series(Z) ≈ Matrix{Float64}(I, 3, 3) ./ 2 atol = 1e-14
end

@testset "rcond_estimate (numerics.jl)" begin
    I3 = Matrix{Float64}(I, 3, 3)
    # Well-conditioned: reciprocal condition number of the identity is 1.
    @test GeoDynamo.rcond_estimate(lu(I3), I3) ≈ 1.0 atol = 1e-10
    # anorm == 0 guard: zero-norm matrix returns one(T). (lu(I3) is an unused placeholder
    # because the guard returns before touching the factorization.)
    @test GeoDynamo.rcond_estimate(lu(I3), zeros(3, 3)) == 1.0
    # Ill-conditioned matrix -> small reciprocal condition estimate.
    B = [1.0 1.0; 1.0 1.0 + 1e-10]
    @test GeoDynamo.rcond_estimate(lu(B), B) < 1e-6
end
```

- [ ] **Step 3: Run standalone, expect PASS**

Run: `$JL --project=. -e 'using Test, GeoDynamo, LinearAlgebra; include("test/numerics_phi_functions.jl")'`
Expected: all pass. If a φ test fails by more than `atol`, the function is buggy — investigate before loosening the tolerance (this is the point of the test).

- [ ] **Step 4: Wire into `test/runtests.jl`**

Add `"numerics_phi_functions.jl",` to the `additional_tests` tuple under the "Pure unit tests" group (near `erk2_matrix_functions.jl`).

- [ ] **Step 5: Commit**

```bash
git add test/numerics_phi_functions.jl test/runtests.jl
git commit -m "test: phi1/phi2 series + rcond_estimate unit tests (numerics.jl)"
```

### Task 2.2: process-grid parsing tests (`parallel/process_grid.jl`)

**Files:**
- Create: `test/process_grid_extended.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Confirm the error type `parse_proc_grid` throws on malformed input**

Run: `$JL --project=. -e 'using GeoDynamo; try; GeoDynamo.parse_proc_grid("4", 4); catch e; println(typeof(e)); end'`
Note the printed type (likely `ErrorException` or `ArgumentError`). Use the ACTUAL type in `@test_throws` below (replace `ErrorException` if different). `parse_proc_grid` is at `src/parallel/process_grid.jl:7`; read it if the type is surprising.

- [ ] **Step 2: Write the tests**

```julia
using Test

@testset "parse_proc_grid edge cases" begin
    # Valid specs (product must equal nprocs)
    @test GeoDynamo.parse_proc_grid("4x2", 8) == (4, 2)
    @test GeoDynamo.parse_proc_grid("8x1", 8) == (8, 1)
    @test GeoDynamo.parse_proc_grid("1x4", 4) == (1, 4)
    @test GeoDynamo.parse_proc_grid(nothing, 1) == (1, 1)      # trivial single-rank
    # Errors
    @test_throws ErrorException GeoDynamo.parse_proc_grid("4x2", 6)    # product != nprocs
    @test_throws ErrorException GeoDynamo.parse_proc_grid(nothing, 4)  # unset spec, np>1
    @test_throws Exception GeoDynamo.parse_proc_grid("4", 4)           # malformed (no 'x')
    @test_throws Exception GeoDynamo.parse_proc_grid("axb", 4)         # malformed (non-numeric)
end

@testset "read_proc_grid env handling" begin
    old = get(ENV, "GEODYNAMO_PROC_GRID", nothing)
    try
        delete!(ENV, "GEODYNAMO_PROC_GRID")
        @test GeoDynamo.read_proc_grid(1) == (1, 1)            # no env needed at np==1
        ENV["GEODYNAMO_PROC_GRID"] = "2x2"
        @test GeoDynamo.read_proc_grid(4) == (2, 2)            # env-driven grid
    finally
        old === nothing ? delete!(ENV, "GEODYNAMO_PROC_GRID") : (ENV["GEODYNAMO_PROC_GRID"] = old)
    end
end
```

- [ ] **Step 3: Run standalone, expect PASS** (adjust `@test_throws` types per Step 1 if needed)

Run: `$JL --project=. -e 'using Test, GeoDynamo; include("test/process_grid_extended.jl")'`
Expected: all pass.

- [ ] **Step 4: Wire in + commit**

Add `"process_grid_extended.jl",` to `additional_tests` (Pure unit tests group).
```bash
git add test/process_grid_extended.jl test/runtests.jl
git commit -m "test: parse_proc_grid edge cases + read_proc_grid env handling"
```

### Task 2.3: Re-measure after Phase 2

- [ ] **Step 1: Run the measure cycle (Task 0.1 Step 2) and record the global %.** Expected: small rise over 82.08% from the new units. Note the number for tracking.

---

## Phase 3 — Tier 1: `timestep/erk2.jl` (434 uncovered — biggest lever)

This file holds the largest gap. Use the **erk2 test fixture template** (below, lifted from `test/erk2.jl`) plus the **per-function characterization procedure**. Targets identified by audit (verify each signature by reading the function before asserting):

| Target function (in `src/timestep/erk2.jl`) | Test angle |
|---|---|
| `create_solver_erk2_scalar_cache` (Krylov + dense) | build cache for a small domain, assert cache fields finite & correctly sized |
| `create_solver_erk2_magnetic_toroidal_cache`, `..._poloidal_cache` | same, magnetic BC variants |
| `build_solver_erk2_velocity_tor_bc` (incl. `rot_omega != 0`) | assert BC spec values for rotating inner core |
| `build_solver_erk2_velocity_pol_bc` | assert BC spec rows |
| `solver_create_stress_free_tor_bc` / `_noslip_pol_bc` / `_stress_free_pol_bc` | assert returned boundary-row descriptors |
| `solver_create_insulating_inner_bc` / `_outer_bc` | assert magnetic insulating rows |
| `prepare`/`finalize_solver_erk2_field!` with `bc_spec !== nothing` | run a step with a non-trivial bc_spec, assert BC enforced |
| `solver_erk2_stage_residual_stats` | feed buffers with known stage diffs, assert max/L2 |
| `integrate_solver_erk2_step!` magnetic & composition branches | **integration test** (Task 3.2) |

**erk2 fixture template (reuse in unit tasks):**
```julia
using Test, LinearAlgebra
nr  = 3
dom = GeoDynamo.create_radial_domain(nr)
cfg = GeoDynamo.create_shtnskit_config(lmax=0, mmax=0, nlat=2, nlon=2, nr=dom.N, optimize_decomp=false)
# spectral fields:
u  = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
nl = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
# set interior point parent(u.data_real)[1,1,2] = value ; boundaries (1,nr) are zeroed by BCs
```

### Task 3.1: erk2 cache-builder + BC-factory unit tests

**Files:**
- Create: `test/erk2_cache_builders.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: For EACH target in the table above that is a pure builder/factory (not integration), read its definition** in `src/timestep/erk2.jl` to capture the exact signature, argument types, and what it returns (struct fields / tuple shape). Record them.

- [ ] **Step 2: Write one `@testset` per function** using the fixture template. Each test calls the function with constructed inputs and asserts a *known* property of the result, e.g.:

```julia
@testset "create_solver_erk2_scalar_cache (dense + Krylov)" begin
    nr = 4
    dom = GeoDynamo.create_radial_domain(nr)
    cfg = GeoDynamo.create_shtnskit_config(lmax=0, mmax=0, nlat=2, nlon=2, nr=dom.N, optimize_decomp=false)
    diffusivity = 1.0
    dt = 0.01
    bc_code = 0  # <-- use the actual Dirichlet code constant found in Step 1
    for use_krylov in (false, true)
        cache = GeoDynamo.create_solver_erk2_scalar_cache(Float64, cfg, dom, diffusivity, dt, bc_code; use_krylov=use_krylov)
        @test cache !== nothing
        # assert the concrete invariants discovered in Step 1, e.g. matrix sizes == (nr,nr) and all-finite:
        # @test all(isfinite, cache.E_full[1]); @test size(cache.E_full[1]) == (nr, nr)
    end
end
```
Replace the commented asserts with the real field names/shapes from Step 1. Every testset must assert at least one concrete numeric/shape property, not merely `!== nothing`.

- [ ] **Step 3: Run standalone, expect PASS**

Run: `$JL --project=. -e 'using Test, GeoDynamo, LinearAlgebra; include("test/erk2_cache_builders.jl")'`

- [ ] **Step 4: Wire in + commit**

Add to `additional_tests`; `git commit -m "test: erk2 cache builders + BC factory unit tests"`.

### Task 3.2: erk2 full-physics integration step (magnetic + composition branches)

**Files:**
- Create: `test/erk2_integration_step.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Find the canonical way to build a `SolverState` with magnetic + composition enabled.** Read `test/integration_simulation.jl` and `src/solver/mainloop.jl::initialize_solver_state` (and `GeodynamoModel(grid; include_magnetic=..., include_composition=...)`). Record the minimal constructor calls.

- [ ] **Step 2: Write an integration test** that builds a tiny state (e.g. lmax=2, nr=6) with magnetic+composition ON, runs one or two `solver_step!`/`integrate_solver_erk2_step!`, and asserts the fields remain finite and changed (a real step occurred):

```julia
using Test
@testset "ERK2 full-physics integration step (magnetic + composition)" begin
    # Build state per Step 1 (fill in the discovered constructor):
    # grid  = GeoDynamo.SphericalShellGrid(...); model = GeoDynamo.GeodynamoModel(grid; include_magnetic=true, include_composition=true, time_integrator=:erk2)
    # state = <initialize from model>
    # snapshot a coefficient before:
    # before = copy(parent(state.fields.velocity.toroidal.data_real))
    # GeoDynamo.solver_step!(state)             # or integrate_solver_erk2_step!(state)
    # @test all(isfinite, parent(state.fields.magnetic.toroidal.data_real))
    # @test all(isfinite, parent(state.fields.composition.spectral.data_real))
    # @test parent(state.fields.velocity.toroidal.data_real) != before   # a step actually advanced the field
end
```
Fill in the commented lines from Step 1's discovered API. The asserts MUST include: (a) all magnetic field coefficients finite, (b) all composition coefficients finite, (c) at least one field changed after the step.

- [ ] **Step 3: Run standalone, expect PASS.** If state construction is too heavy, fall back to unit-testing `restore_solver_erk2_nonlinear_terms!` and the magnetic/composition branch helpers directly with mock buffers (see audit notes), and note the fallback in the test header.

- [ ] **Step 4: Wire in + commit** (`"erk2_integration_step.jl"`).

### Task 3.3: Re-measure after Tier 1

- [ ] Run the measure cycle. Record erk2.jl's new per-file % and the global %. Expected erk2 jump from 44.7% toward ~70%+.

---

## Phase 4 — Tier 2 remainder: `numerics.jl` + `imex.jl`

### Task 4.1: numerics banded + cpu_* vector-helper tests

**Files:**
- Create: `test/numerics_banded_and_vector.jl`
- Modify: `test/runtests.jl`

Targets (read each before asserting): `solver_factorize_banded`, `solver_build_banded_A`, `solver_phi1_action_krylov`, and the live `cpu_fill_vector_coeff_buffer!` / `cpu_store_vector_coefficients!` / `cpu_extract_vector_component!` / `cpu_store_vector_components!` (called via the public wrappers at `src/GeoDynamo.jl:109-132`).

- [ ] **Step 1: Read `solver_build_banded_A`, `solver_factorize_banded`** in `src/solver/numerics.jl` for signatures, and read the public vector-helper wrappers at `src/GeoDynamo.jl:105-135` to learn how to call the `cpu_*` path.

- [ ] **Step 2: Write tests with real assertions**, e.g. for the banded operator:
```julia
@testset "solver_build_banded_A + factorize" begin
    nr = 6
    dom = GeoDynamo.create_radial_domain(nr)
    A = GeoDynamo.solver_build_banded_A(Float64, dom, 1.0, 1)   # match real signature from Step 1
    # Assert a concrete property, e.g. it is the right size and the implicit operator is diagonally sensible:
    lu = GeoDynamo.solver_factorize_banded(A)
    @test lu !== nothing
    # round-trip: solving A x = A*e should recover e for an interior vector (fill per discovered API)
end
```
For the `cpu_*` helpers, drive them through the public wrapper with a small vector field and assert the coefficient buffer contents equal the input coefficients (a copy round-trip — a real value check, not `!== nothing`).

- [ ] **Step 3: Run standalone, PASS. Step 4: wire in + commit.**

### Task 4.2: imex implicit-step tests

**Files:**
- Create: `test/imex_implicit_steps.jl`
- Modify: `test/runtests.jl`

Targets: `solver_solve_magnetic_implicit_step!` (never tested directly), `_solver_solve_scalar_implicit_step!`, `solver_solve_velocity_implicit_step!`, `solver_solve_temperature_implicit_step!`, `solver_solve_composition_implicit_step!`.

- [ ] **Step 1: Read these functions** in `src/timestep/imex.jl` (around lines 315–490) for signatures and the matrices/argument structs they need; check `test/cnab2_rhs_distributed_equivalence.jl` for how RHS + matrices are built.

- [ ] **Step 2: Write a test** that constructs a known linear system (implicit operator `M`, known RHS `b`, known solution `x`), calls the implicit-step solver, and asserts the returned field ≈ `x` to tight tolerance. Do this for the scalar path and the magnetic path at minimum.

- [ ] **Step 3: Run standalone, PASS. Step 4: wire in + commit.**

### Task 4.3: Re-measure after Tier 2

- [ ] Measure cycle. **Checkpoint: if global ≥ 88%, the floor is met.** Continue to Tiers 3–5 toward the 90% stretch while ROI holds.

---

## Phase 5 — Tiers 3–5: broad per-file characterization sweep

Apply the **per-file procedure** below to each file in the ordered list. One commit per file (or per small group). The exact assertions are determined by reading each file's uncovered lines — this is a procedure, not a placeholder.

**Per-file procedure (repeat for each target):**

- [ ] **A. Find the file's uncovered lines:**
```bash
F=src/<path>.jl
awk 'NR==FNR{next}{}' /dev/null   # (ensure a fresh coverage run exists)
awk 'BEGIN{s=0}{ln=NR; if($1=="0"){if(s==0){s=ln;p=ln}else if(ln==p+1){p=ln}else{printf "%d-%d ",s,p;s=ln;p=ln}}}END{if(s)printf "%d-%d\n",s,p}' "$F".*.cov
```
- [ ] **B. Read those line ranges** in the source. Group them by enclosing function. Classify each: LIVE-TESTABLE / ERROR-BRANCH (triggerable?) / GPU-MPI (skip).
- [ ] **C. For each LIVE-TESTABLE function, write a `@testset`** that calls it with constructed inputs and asserts a *known-correct* result (analytic value, round-trip identity, or a property like conservation / finiteness / exact size). Reuse fixtures from neighbouring tests (see the "fixture sources" column).
- [ ] **D. Run the new test file standalone — expect PASS.** If it fails, the code (or your understanding) is wrong: investigate, don't weaken the assertion.
- [ ] **E. Wire the file into `test/runtests.jl`; commit** with `git commit -m "test: cover <file> (<old%> -> target)"`.
- [ ] **F. Every ~4 files, run the full measure cycle** to track the global % and confirm no regression/new flake.

**Ordered target list (highest uncovered first):**

*Tier 3 (numeric kernels):*
| File | uncov | fixture sources |
|---|---|---|
| `src/fields/transforms.jl` | 54 | `test/shtnskit_roundtrip.jl`, `test/theta_dist_transform.jl` |
| `src/physics/nonlinear.jl` | 53 | `test/integration_simulation.jl` |
| `src/physics/magnetic/solver.jl` | 45 | `test/magnetic_boundary_numerical.jl` |

*Tier 4 (bcs / topography cluster):*
| File | uncov | fixture sources |
|---|---|---|
| `src/bcs/topography/velocity_coupling.jl` | 41 | `test/topography_coupling.jl` |
| `src/bcs/topography/topography_data.jl` | 37 | `test/topography_data.jl` |
| `src/bcs/file_bc_loader.jl` | 36 | `test/boundary_file_io.jl` |
| `src/bcs/interpolation.jl` | 33 | `test/boundary_utilities.jl` |
| `src/bcs/topography/stefan_condition.jl` | 29 | `test/stefan_condition_sanity.jl` |
| `src/bcs/integration.jl` | 25 | `test/programmatic_boundaries.jl` |
| `src/bcs/topography/gaunt_tensors.jl` | 24 | `test/check_gaunt_tensors.jl` |
| `src/bcs/netcdf_io.jl` | 22 | `test/boundary_file_io.jl` |
| `src/bcs/bcs.jl` | 22 | `test/boundary_types.jl` |
| `src/bcs/topography/thermal_coupling.jl` | 16 | `test/topography_coupling.jl` |
| `src/bcs/common.jl` | 14 | `test/boundary_utilities.jl` |
| `src/bcs/topography/magnetic_coupling.jl` | 12 | `test/topography_coupling.jl` |
| `src/bcs/programmatic.jl` | 10 | `test/programmatic_boundaries.jl` |
| `src/bcs/topography/topography.jl` | 10 | `test/topography_coupling.jl` |

*Tier 5 (io / core / api / physics tail):*
| File | uncov | fixture sources |
|---|---|---|
| `src/io/restart.jl` | 37 | `test/io_restart_roundtrip.jl` |
| `src/core/initial_conditions.jl` | 33 | `test/initial_conditions.jl` |
| `src/io/netcdf.jl` | 21 | `test/netcdf_write_roundtrip.jl` |
| `src/core/parameters.jl` | 21 | `test/parameter_validation.jl` |
| `src/api/simulation.jl` | 20 | `test/user_api.jl`, `test/oceananigans_api.jl` |
| `src/io/field_info.jl` | 19 | `test/io_subsystem.jl` |
| `src/physics/composition/field.jl` | 22 | `test/composition_magnetic_fields.jl` |
| `src/physics/temperature/solver.jl` | 17 | `test/temperature_field.jl` |
| `src/physics/composition/solver.jl` | 17 | `test/composition_boundary_numerical.jl` |
| `src/physics/velocity/solver.jl` | 17 | `test/velocity_boundary_numerical.jl` |
| `src/solver/state.jl` | 17 | `test/field_containers.jl` |
| `src/api/initial_conditions.jl` | 14 | `test/initial_conditions.jl` |
| `src/physics/topography.jl` | 14 | `test/topography_coupling.jl` |
| `src/solver/backend.jl` | 12 | `test/integration_simulation.jl` |
| `src/physics/magnetic/field.jl` | 12 | `test/composition_magnetic_fields.jl` |
| `src/timestep/driver.jl` | 11 | `test/integration_simulation.jl` |

Stop when the global reaches the 90% stretch or when remaining uncovered lines in the list are dominated by ERROR/GPU/MPI branches (the serial ceiling).

---

## Phase 6 — Final verification

### Task 6.1: Full green run + coverage confirmation

- [ ] **Step 1: Clean + full coverage run**
```bash
JL=~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia
find . -name '*.cov' -delete
$JL --project=. -e 'using Pkg; Pkg.test(coverage=true)' 2>&1 | tee /tmp/final_cov.log | tail -8
scripts/cov_aggregate.sh src
```
- [ ] **Step 2: Confirm success criteria** (from the spec):
  - Global line coverage **≥ 88%** (target 90%).
  - Suite: **0 failures**, ≤3 broken, pass count ≥ baseline 2972 + new tests.
  - No NEW broken/flaky tests. If a transient 3-fail appears, re-run once (suite is nondeterministic on ~3 IC tests — see memory) before attributing it.
- [ ] **Step 3: Clean coverage artifacts** (they are gitignored, but tidy the tree): `find . -name '*.cov' -delete`
- [ ] **Step 4: Final commit / branch summary**
```bash
git add -A test/ scripts/ src/
git commit -m "test: complete coverage sweep to <final>% (from 82.08%)"
git log --oneline test/coverage-sweep ^main
```

### Task 6.2: Update project memory

- [ ] Append a one-line entry to `/Users/subha/.claude/projects/-Users-subha-Documents-GitHub-GeoDynamo-jl/memory/MEMORY.md` Test Coverage section: final %, suite count, dead-code removed, branch name.

---

## Self-Review notes (plan vs spec)

- **Spec coverage:** 88/90 target → Tasks across Phases 2–6; baseline+measurement → Task 0.1; dead-code (timestepping) → Task 1.1; cpu_* resolved as LIVE → Task 4.1 (test, not delete) — *spec's "delete-if-dead" resolved to "test" by the caller-graph check*; quality bar → enforced in every task's "assert known value" steps; tiered plan → Phases 3–5; verification → Task 4.3 checkpoint + Phase 6; risks (ceiling, flake) → Phase 6 Step 2.
- **Known executor reads (not placeholders):** Tasks 3.1/3.2/4.1/4.2 and all of Phase 5 require reading the target function before writing assertions — this is deliberate (characterization testing), with concrete fixture templates and asserted-property requirements supplied so the executor cannot write a hollow test.
- **Process-grid duplication:** `parse_proc_grid` has 5 existing cases in `test/r_theta_grid.jl`; Task 2.2 adds only NEW cases (malformed inputs + `read_proc_grid`).
