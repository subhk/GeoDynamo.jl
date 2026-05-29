# Example Smoke Refresh Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refresh the simulation setup examples so `test/examples_smoke.jl` runs cleanly against the current solver API.

**Architecture:** Keep the examples aligned with the current public entrypoints instead of loosening the smoke test. The fix should be limited to stale setup calls and invalid ball-geometry defaults, then verified first with the focused smoke test and finally with the broader suite.

**Tech Stack:** Julia, GeoDynamo.jl test suite, SHTnsKit-backed solver setup

### Task 1: Reproduce the failing smoke test

**Files:**
- Test: `test/examples_smoke.jl`

**Step 1: Run the focused smoke test**

Run: `julia --project -e "using Test, GeoDynamo; include(joinpath(pwd(), \"test\", \"examples_smoke.jl\"))"`
Expected: FAIL in the simulation setup example path with a stale example/runtime mismatch.

### Task 2: Refresh the stale setup examples

**Files:**
- Modify: `examples/ball_mhd_demo.jl`
- Modify: `examples/shell_dynamo_demo.jl`
- Test: `test/examples_smoke.jl`

**Step 1: Keep the failing test as the reproducer**

Do not weaken `test/examples_smoke.jl`; use its current failure as the red step.

**Step 2: Write the minimal implementation**

- Update the simulation setup examples to call `GeoDynamo.initialize_fields!` explicitly instead of relying on an unexported binding.
- Set the ball example’s radius ratio to `0.0` so it matches `geometry = :ball`.

**Step 3: Re-run the focused smoke test**

Run: `julia --project -e "using Test, GeoDynamo; include(joinpath(pwd(), \"test\", \"examples_smoke.jl\"))"`
Expected: PASS.

### Task 3: Verify the broader suite

**Files:**
- Test: `test/runtests.jl`

**Step 1: Run the full suite**

Run: `julia --project -e "using Test, GeoDynamo; include(joinpath(pwd(), \"test\", \"runtests.jl\"))"`
Expected: All focused regressions remain green; any remaining failures must be unrelated to the example refresh.
