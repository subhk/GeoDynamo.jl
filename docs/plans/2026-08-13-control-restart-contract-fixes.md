# Control, Restart, and Timestepper Contract Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task.

**Goal:** Prevent MPI callback deadlocks and restart file clobbering, while making the documented timestepper surface match what the solver can execute.

**Architecture:** Reconcile callback-driven stop state once after every callback batch. Treat restart metadata as the state after the file being written, then restore writer-local trackers from both checkpoint metadata and the highest existing filename in each output directory. Keep planned timestepper descriptor types source-compatible, but clearly classify them as unsupported and remove runnable recommendations until their missing solver kernels exist.

**Tech Stack:** Julia 1.10+, MPI.jl, NCDatasets.jl, Test stdlib, Documenter markdown.

### Task 1: Synchronize callback stops

**Files:**
- Modify: `test/mpi_control_plane_invariants.jl`
- Modify: `src/api/callbacks.jl`

1. Add a two-rank callback that sets `sim.running = false` on one rank only.
2. Run `test/run_mpi_control_plane.sh` with a short timeout and confirm it hangs/fails.
3. Reduce `!sim.running` across ranks after `_run_callbacks!` fires the batch.
4. Re-run the MPI control-plane gate and confirm it passes.

### Task 2: Persist post-write restart counters

**Files:**
- Modify: `test/io_restart_roundtrip.jl`
- Modify: `src/io/restart.jl`
- Modify: `src/io/history.jl`

1. Change the round-trip expectation so restart file 1 restores `restart_count == 1`.
2. Add a combined history/restart assertion that the checkpoint records the newly written history count.
3. Run the focused I/O test and confirm the old counters fail.
4. Serialize `restart_count + 1` and the post-history output count without mutating the live tracker before a successful write.
5. Re-run the focused I/O test.

### Task 3: Restore scheduled-writer numbering

**Files:**
- Modify: `test/code_review_max_fixes.jl`
- Modify: `src/api/output_writers.jl`
- Modify: `src/api/simulation.jl`

1. Add unit coverage for merging restored counters with higher on-disk history/restart suffixes.
2. Add constructor coverage showing `Simulation(restart_from=...)` seeds new writer trackers.
3. Run the focused regression file and confirm both assertions fail.
4. Add tracker-copy/merge helpers and invoke them after the constructor orders its output writers.
5. Re-run the focused regression file.

### Task 4: Align the timestepper contract

**Files:**
- Modify: `test/parameter_validation.jl`
- Modify: `src/api/timesteppers.jl`
- Modify: `docs/src/configuration.md`
- Modify: `docs/src/timestepping.md`
- Modify: `docs/src/index.md`

1. Add contract checks that the supported list is exactly CNAB2, ERK2, and RK3 and unsupported errors name that list.
2. Run the focused validation test.
3. Keep validation's supported list and make API docstrings state the same contract.
4. Mark EAB2, ETD, and ThetaMethod as reserved/unsupported in API docs; remove recommendations that imply runnable support.
5. Re-run validation and documentation checks.

### Task 5: Verify the whole change

1. Run focused MPI, I/O, callback, and parameter tests.
2. Run the full `Pkg.test` suite.
3. Run `git diff --check` and inspect `git status --short`.
