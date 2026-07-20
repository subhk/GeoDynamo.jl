# Velocity Boundary Condition Corrections Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make velocity boundary conditions geometry-aware, preserve complex topography coefficients, and use the documented spherical stress-free operator in every timestepper.

**Architecture:** Keep the existing boundary-condition descriptors and poloidal W-split design. Thread the already-stored real and imaginary wall values into CNAB2, CB3, and ERK2, while retaining the origin regularity constraints used by ball geometry. Add production-path regression tests before each minimal implementation change.

**Tech Stack:** Julia 1.11, GeoDynamo solver state/caches, Test stdlib, SHTnsKit spectral fields.

### Task 1: Make CB3 poloidal recovery geometry-aware

**Files:**
- Modify: `test/cb3_timestepper.jl`
- Modify: `src/timestep/cb3.jl`

**Step 1: Write the failing test**

Create a ball-geometry RK3 state, excite a nontrivial poloidal mode through temperature buoyancy, take one CB3 step, and assert that every cached `PoloidalSplitMatrices` object has `ball == true`. Also assert the recovered inner poloidal value satisfies `P'-(l+1)P/r=0` for the excited mode.

**Step 2: Run the test to verify it fails**

Run the CB3 test through the installed audit environment. Expect failure because `_get_or_build_cb3_poloidal_split!` currently builds shell matrices (`ball == false`).

**Step 3: Write the minimal implementation**

Pass `ball = state.parameters.geometry === :ball` into `build_poloidal_split_matrices`. In `_cb3_apply_poloidal_wsplit_stage!`, preserve the ball W regularity residual before imposing the wall/origin value and use it as the inner P-recovery right-hand side. Use the real or imaginary poloidal wall table for shell modes.

**Step 4: Run the test to verify it passes**

Re-run `test/cb3_timestepper.jl`; expect the new ball regression and existing shell tests to pass.

### Task 2: Apply ERK2 velocity topography values

**Files:**
- Modify: `test/topography_bc_injection.jl`
- Modify: `src/timestep/erk2/integrate.jl`
- Modify: `src/physics/velocity/solver.jl`

**Step 1: Write the failing test**

For a shell ERK2 state with zero dynamics, inject nonzero toroidal and poloidal boundary values, take one production ERK2 step, and assert that the spectral wall values equal the requested values.

**Step 2: Run the test to verify it fails**

Run `test/topography_bc_injection.jl`. Expect both velocity assertions to report zero because ERK2 currently omits velocity mode-value descriptors and resets poloidal walls to zero.

**Step 3: Write the minimal implementation**

Wrap the cached ERK2 toroidal boundary descriptor with `with_boundary_mode_values` using `get_bc_vectors(velocity.toroidal)`. During ERK2 poloidal recovery, choose the real or imaginary wall table and assign shell wall values while leaving ball-origin regularity unchanged.

**Step 4: Run the test to verify it passes**

Re-run the injection test and expect all scalar, magnetic, and velocity cases to pass.

### Task 3: Preserve imaginary velocity topography coefficients

**Files:**
- Modify: `test/topography_coupling.jl`
- Modify: `test/topography_bc_injection.jl`
- Modify: `src/bcs/topography/velocity_coupling.jl`
- Modify: `src/timestep/imex.jl`
- Modify: `src/physics/velocity/solver.jl`
- Modify: `src/timestep/cb3.jl`

**Step 1: Write the failing coupling test**

Excite an imaginary non-axisymmetric poloidal mode, evaluate the complex impermeability correction, apply velocity topography, and assert both real and imaginary boundary arrays match the correction. Expect the imaginary assertion to fail because only `real(...)` is stored.

**Step 2: Write the failing solver-consumption tests**

Inject nonzero imaginary toroidal and poloidal boundary values into CNAB2/ERK2 production paths and assert the solved wall coefficients equal them. Expect failures because the velocity implicit solver and W-split recovery currently force imaginary wall right-hand sides to zero.

**Step 3: Write the minimal implementation**

Clear and populate both `boundary_values` and `boundary_values_imag` in velocity topography coupling. Add imaginary boundary keyword arguments to `solver_solve_velocity_implicit_step!` and pass them from CNAB2 and CB3. Select the imaginary poloidal table in CNAB2, CB3, and ERK2 recovery loops.

**Step 4: Run the tests to verify they pass**

Re-run both topography tests plus the CB3 tests. Expect non-axisymmetric coefficients to survive coupling and every timestepper.

### Task 4: Correct the stress-free poloidal stencil and documentation

**Files:**
- Modify: `test/erk2_cache_builders.jl`
- Modify: `src/timestep/erk2/boundary.jl`
- Modify: `src/bcs/velocity_bc.jl`
- Modify: `test/velocity_boundary_numerical.jl`
- Modify: `docs/src/boundary-conditions.md`

**Step 1: Write the failing test**

Build the shell ERK2 poloidal descriptor for stress-free walls and compare each stencil to `D2 - 2/r * D1`. Expect failure because the builder currently stores bare `D2` rows.

**Step 2: Write the minimal implementation**

Construct the metric-aware stress-free rows in `build_solver_erk2_velocity_pol_bc` before passing them to `solver_create_stress_free_pol_bc`. Keep the helper signature compatible and clarify that its input is the complete stress-free row.

**Step 3: Correct the documentation**

Document no-slip as `P=0, P'=0` and stress-free as `P=0, P''-2P'/r=0` in user docs, source docstrings, and test comments.

**Step 4: Run the tests to verify they pass**

Re-run `test/erk2_cache_builders.jl` and `test/velocity_boundary_numerical.jl`; expect the spherical operator and numerical boundary residuals to pass.

### Task 5: Verify the integrated result

**Files:**
- Review all modified source, test, and documentation files.

**Step 1: Run focused regression suites**

Run CB3, ERK2 builder, topography coupling/injection, ball W-split, velocity numerical, and poloidal momentum tests in the audit environment.

**Step 2: Run the broad project suite**

Run the repository test entry point if the local dependency environment supports it. Record any environment-only failures separately from code failures.

**Step 3: Inspect the patch**

Run `git diff --check`, review `git diff`, and confirm `git status --short` contains only intended changes.
