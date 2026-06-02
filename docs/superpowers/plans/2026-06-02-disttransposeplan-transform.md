# DistTransposePlan Transform (Phase 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace GeoDynamo's per-level gather-to-dense SH transform with SHTnsKit's `DistTransposePlan` (batched over radial levels, transpose-based θ↔m, m-distributed) — matching DD_2DCODE — keeping the 2D r×θ grid and the r-local banded radial solve.

**Architecture:** A `DistTransposePlan` per config on the θ-subcommunicator (`nlev=nr_local`) does the transform in one batched, transpose-based call (no θ-Allreduce, no replicated dense matrix). Its `Alm` (l-local / m-dist / r-batch) is bridged to a redefined `spec_solve` (m-over-θ_ranks, l-over-r_ranks, r-local) by an `Alm↔spec_solve` transpose; the banded radial solve runs on `spec_solve` unchanged. Replace staged (scalar→vector), then delete the Phase-2 gather path.

**Tech Stack:** Julia, SHTnsKit ≥1.2.10 (`DistTransposePlan`, `dist_analysis!`/`dist_synthesis!`/`dist_*_sphtor!`), PencilArrays/PencilFFTs, MPI.

**Spec:** `docs/superpowers/specs/2026-06-02-disttransposeplan-transform-design.md`.

**Execution note:** Isolated git worktree off current `main`. Julia (shim broken): `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia`. MPI via `MPI.mpiexec()` (MPICH). Grids set via `GEODYNAMO_PROC_GRID="θxr"` (required np>1). Baseline: Phase-2 main green, r×θ step-equivalence bitwise.

## Verified DistTransposePlan facts (spiked, 2 ranks — TRUST)
- `plan = SHTnsKit.DistTransposePlan(cfg.sht_config; comm, nlev, use_rfft=true, with_vector=true)`. Has `DistTransposePlan` in GeoDynamo's pinned SHTnsKit.
- `allocate_spatial(plan)` → real PencilArray, global `(nlon, nlat, nlev)`, **φ(dim1) LOCAL / θ(dim2) DISTRIBUTED / lev(dim3) local**.
- `allocate_spectral(plan)` → complex PencilArray, global `(lmax+1, mmax+1, nlev)`, **l(dim1) LOCAL / m(dim2) DISTRIBUTED / lev(dim3) local**.
- `dist_analysis!(plan, Alm, f)` (spatial→spectral, batched all nlev), `dist_synthesis!(plan, f, Alm)` (reverse). Roundtrip err 6.75e-14 at 2 ranks.
- `dist_analysis_sphtor!`/`dist_synthesis_sphtor!(plan, ...)` exist (vector; with_vector=true).
- ⚠️ **Axis-order mismatch:** plan spatial is `(φ, θ, lev)`; GeoDynamo physical is `(θ, φ, r)`. The fill/copy between them transposes the first two axes + maps r→lev (spike pattern: `parent(fsp)[iφ, jl, k] = phys[θ_global, φ, r]`).

---

## Task 0: Spike — DistTransposePlan on the θ-subcomm + the Alm↔spec_solve transpose (GATES EVERYTHING)

**Files:** `/tmp/spike_p3.jl` (throwaway). No production code.

> Rationale: the #1 risk is bridging `Alm` (on the θ-subcommunicator, with an r-batch dim) to a `spec_solve` orientation (full 2D topology), via an l↔r redistribution over the r-subcommunicator. The comm-structure mismatch is the unknown. This spike (a) builds a `DistTransposePlan` on the θ-subcomm of a 2D `(θ_ranks,r_ranks)` grid with `nlev=nr_local`, (b) implements `Alm → spec_solve → Alm`, and (c) confirms an exact identity roundtrip at 2×2. If it can't be made to roundtrip cleanly, STOP and report BLOCKED — Phase 2 stands.

- [ ] **Step 1: Write the spike** — at 4 ranks `2x2`, using GeoDynamo's config to get `cfg.pencils.θ_comm`/`r_comm` + `nr_local`:
```julia
using MPI; MPI.Init()
using GeoDynamo, SHTnsKit, PencilArrays, PencilFFTs
comm=MPI.COMM_WORLD; rank=MPI.Comm_rank(comm)
cfg = GeoDynamo.create_shtnskit_config(lmax=8,mmax=8,nlat=12,nlon=20,nr=8)
θc = cfg.pencils.θ_comm; rc = cfg.pencils.r_comm
nr_local = length(PencilArrays.range_local(cfg.pencils.r)[3])
plan = SHTnsKit.DistTransposePlan(cfg.sht_config; comm=θc, nlev=nr_local, use_rfft=true, with_vector=true)
Alm = SHTnsKit.allocate_spectral(plan)        # (lmax+1, mmax+1, nlev): l-local, m-dist/θc, lev
# seed Alm deterministically on owned (l,m,lev); record a copy
a0 = copy(parent(Alm))
# --- the candidate Alm↔spec_solve transpose ---
# spec_solve target: m-dist over θc (axis already matches Alm dim2), l-dist over rc,
# r LOCAL (gather nlev across rc -> full nr). Implement BOTH directions and roundtrip:
solve = to_spec_solve(cfg, Alm, plan)          # Alm(l-local,m-dist/θc,r-dist/nlev) -> (l-dist/rc, m-dist/θc, r-local)
Alm2  = from_spec_solve(cfg, solve, plan)       # and back
@assert parent(Alm2) == a0   # exact identity
println(rank==0 ? "P3_TRANSPOSE_IDENTITY=OK" : "")
MPI.Finalize()
```
Implement `to_spec_solve`/`from_spec_solve` in the spike experimentally. The r-gather/scatter is over `rc` (the r-subcomm); the l-redistribution is over `rc`; m stays as-is (already θc-distributed in both `Alm` and the target). Try: (a) a manual MPI Alltoallv over `rc` that swaps the l-axis (Alm: l-full) ↔ r-axis (solve: r-full); or (b) a PencilArrays transpose if the layouts can be expressed as two pencils on `rc`. RECORD which works.

- [ ] **Step 2: Run at 4 ranks `2x2`** (and `2x1`, `1x2`):
Run: `JL=~/.julia/...; withenv("GEODYNAMO_PROC_GRID"=>"2x2") do; MPI.mpiexec() do m; run(`$m -n 4 $JL --project=. /tmp/spike_p3.jl`); end; end`
Expected: `P3_TRANSPOSE_IDENTITY=OK`. **Decision:** OK → record the working `to_spec_solve`/`from_spec_solve` mechanism + the exact `spec_solve` pencil layout into Task 1. NOT OK / no clean mechanism after real effort → STOP, report BLOCKED (the comm-mismatch defeats the design); Phase 2 stands.

- [ ] **Step 3: Also confirm the full transform roundtrip** through the plan + the transpose: seed a known physical field (r×θ, via the axis-permuted fill), `dist_analysis!` → Alm → `to_spec_solve` → `from_spec_solve` → Alm → `dist_synthesis!` → physical; assert machine-precision roundtrip. Delete the spike.

---

## Task 1: DistTransposePlan plumbing + redefined `spec_solve` + Alm↔solve transpose

**Files:**
- Modify: `src/transforms/spectral.jl` (cache the plan in config/buffers; redefine `spec`/add the solve pencil), `src/parallel/transposes.jl` (the Alm↔solve transpose, per Task-0's pinned mechanism).
- Create: `src/parallel/disttranspose_adapter.jl` (the `to_spec_solve`/`from_spec_solve` from Task 0, productionized) — included after `transforms/spectral.jl`.
- Test: `test/p3_transpose.jl`.

- [ ] **Step 1: Write the failing identity test** (uses Task-0's pinned API):
```julia
using Test, GeoDynamo, SHTnsKit, MPI, PencilArrays
MPI.Initialized() || MPI.Init()
@testset "Alm <-> spec_solve identity" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8,mmax=8,nlat=12,nlon=20,nr=8)
    plan = GeoDynamo.get_disttranspose_plan(cfg)            # cached
    Alm = SHTnsKit.allocate_spectral(plan)
    parent(Alm) .= ComplexF64(MPI.Comm_rank(MPI.COMM_WORLD)+1)
    a0 = copy(parent(Alm))
    solve = GeoDynamo.to_spec_solve(cfg, Alm, plan)
    GeoDynamo.from_spec_solve!(cfg, Alm, solve, plan)
    @test parent(Alm) == a0
end
```
- [ ] **Step 2: Run, expect FAIL** (`get_disttranspose_plan`/`to_spec_solve` undefined).
- [ ] **Step 3: Implement** — `get_disttranspose_plan(cfg)`: build+cache `DistTransposePlan(cfg.sht_config; comm=cfg.pencils.θ_comm, nlev=length(range_local(cfg.pencils.r)[3]), use_rfft=true, with_vector=true)` (guard `nlev>0`; error if `r_ranks>nr`). Add `to_spec_solve`/`from_spec_solve!` (the Task-0 mechanism) in `disttranspose_adapter.jl`. Define the `spec_solve` storage pencil (Task-0's layout: m-over-θ_ranks, l-over-r_ranks, r-local).
- [ ] **Step 4: Run, expect PASS** — single + `2x2`/`2x1`/`1x2`.
- [ ] **Step 5: Commit**
```bash
git add src/transforms/spectral.jl src/parallel/transposes.jl src/parallel/disttranspose_adapter.jl src/GeoDynamo.jl test/p3_transpose.jl
git commit -m "feat(transform): DistTransposePlan plumbing + Alm<->spec_solve transpose"
```

---

## Task 2: Scalar transforms via DistTransposePlan

**Files:** Modify `src/physics/nonlinear.jl` (`scalar_spectral_to_physical!`, `scalar_physical_to_spectral!`); Test append to `test/p3_transpose.jl`.

- [ ] **Step 1: Append the roundtrip + equivalence test** — seed deterministic spectral modes (global-index, l≥m), `scalar_spectral_to_physical!` then `scalar_physical_to_spectral!`, assert spectral preserved <1e-10; run single/2x1/1x2/2x2.
- [ ] **Step 2: Run on current code, expect FAIL** (current scalar transforms use the Phase-2 gather path / spec_transform).
- [ ] **Step 3: Rewrite** both functions: synthesis = `from_spec_solve` is not needed; instead `to_disttranspose_alm` from the field's spectral (`spec_solve`) — i.e. transpose `spec_solve → Alm`, fill the plan's spatial buffer via `dist_synthesis!(plan, fspatial, Alm)`, then copy `fspatial (φ,θ,lev)` into `phys.data (θ,φ,r)` with the axis permutation (per the verified mismatch). Analysis = copy `phys.data → fspatial` (axis-permuted), `dist_analysis!(plan, Alm, fspatial)`, `to_spec_solve` into the field's spectral storage. Reuse cached plan + cached spatial/Alm buffers.
- [ ] **Step 4: Run, expect PASS** (<1e-10), single + 2x1/1x2/2x2.
- [ ] **Step 5: Commit**
```bash
git add src/physics/nonlinear.jl test/p3_transpose.jl
git commit -m "feat(transform): scalar transforms via DistTransposePlan (batched, m-distributed)"
```

---

## Task 3: Vector transforms via dist_*_sphtor! (DistTransposePlan)

**Files:** Modify `src/solver/numerics.jl` (`vector_spectral_to_physical!`, `vector_physical_to_spectral!`) and `src/fields/transforms.jl` (`shtnskit_vector_synthesis!`, `shtnskit_vector_analysis!`); Test append.

- [ ] **Step 1: Append a vector roundtrip + equivalence test** (both solver + non-solver paths; toroidal/poloidal preserved <1e-8; v_r finite/non-zero); single/2x1/1x2/2x2.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Rewrite** all four vector functions to use `dist_synthesis_sphtor!(plan, ...)`/`dist_analysis_sphtor!(plan, ...)` (batched, on the cached plan) for the tangential (S=poloidal, T=toroidal) pair + the `Alm↔spec_solve` transpose; v_r via the scalar `dist_synthesis!` of `l(l+1)/r·P` (preserve each path's factor: numerics.jl `/r_val`, fields/transforms.jl `/r_val²`).
- [ ] **Step 4: Run, expect PASS** (<1e-8), single + 2x1/1x2/2x2; re-run scalar tests.
- [ ] **Step 5: Commit**
```bash
git add src/solver/numerics.jl src/fields/transforms.jl test/p3_transpose.jl
git commit -m "feat(transform): vector transforms via dist_*_sphtor! (DistTransposePlan)"
```

---

## Task 4: Remove the Phase-2 gather path

**Files:** Modify `src/transforms/spectral.jl` (remove `spec_transform` pencil), `src/physics/nonlinear.jl`/`src/solver/numerics.jl`/`src/fields/transforms.jl` (remove the per-level θ-Allreduce-to-dense + the old `spec_transform`/`theta_phys`-prototype path now unused), buffer cleanup.

- [ ] **Step 1:** `grep -rn 'spec_transform\|theta_phys_proto\|transpose_solve_to_transform' src` — for each, confirm zero remaining callers after Tasks 2–3. List dead.
- [ ] **Step 2: Remove** the confirmed-dead Phase-2 transform machinery (`spec_transform` pencil + its NamedTuple entry; the per-level scatter+`Allreduce!`-to-dense; `transpose_solve_to_transform!`/`transpose_transform_to_solve!` if now unused; orphaned buffers — adjust `SHTnsBuffers` struct/ctor count/map/clear consistently). KEEP `theta_phys` if still used; remove if not.
- [ ] **Step 3: Load-check + run `test/p3_transpose.jl`** (single + 2x2) — still green.
- [ ] **Step 4: Commit**
```bash
git add -A
git commit -m "refactor: remove Phase-2 gather-to-dense transform path (superseded by DistTransposePlan)"
```

---

## Task 5: Full suite + invariants + collective-count demo + gate

**Files:** Modify `test/mpi_parallel_invariants.jl` (spec_solve mode-partition changed: m-over-θ, l-over-r), `test/runtests.jl` (wire `p3_transpose.jl`); the existing `test/r_theta_equivalence.jl`/`_mhd.jl` are reused as the step-equivalence gate.

- [ ] **Step 1: Update `mpi_parallel_invariants.jl`** for the new `spec_solve` partition (m distributed over θ_ranks, l over r_ranks, r-local). Update the spec-pencil assertions accordingly; keep physical-pencil + transpose invariants.
- [ ] **Step 2: Wire `p3_transpose.jl` into `runtests.jl`.**
- [ ] **Step 3: Step-equivalence gate** — run `test/r_theta_equivalence.jl` (+ `_mhd.jl`) across grids: Phase-3 `solver_step!` must match the 1D/serial result to <1e-10 (the transform swap is physics-neutral). Run the 4-grid comparison via the existing shell runners.
- [ ] **Step 4: Single-rank full suite** — `julia --project=. -e 'using Pkg; Pkg.test()' > /tmp/p3_suite.log 2>&1; echo EXIT=$? >> /tmp/p3_suite.log; grep -E 'Extended GeoDynamo|passed|FAIL' /tmp/p3_suite.log | tail -3`. Expect 0 failed (known broken only; re-run flaky IC once).
- [ ] **Step 5: Collective-count demo** — a small script counting MPI collectives per `solver_step!` (or per transform) on Phase-3 vs Phase-2: show O(1) batched transpose replaces O(nr_local) θ-Allreduces. Record the numbers (the concrete structural win; wall-clock needs cluster).
- [ ] **Step 6: Commit**
```bash
git add test/mpi_parallel_invariants.jl test/runtests.jl
git commit -m "test(parallel): Phase-3 invariants update; suite + step-equivalence + collective-count gate"
```

---

## Self-review notes (author)

- **Spec coverage:** plan object + θ-subcomm (T1), redefined spec_solve + Alm↔solve transpose (T0/T1), scalar (T2), vector both paths (T3), remove gather path (T4), invariants + equivalence + collective-count (T5). T0 gates the comm-mismatch risk. Radial solve numerics untouched (no task — by design). ✓
- **Hard dependency:** T0 gates all; if no clean Alm↔spec_solve transpose, STOP (Phase 2 stands). The exact `spec_solve` pencil layout + the transpose mechanism are pinned by T0 (Phase-1/2 precedent for spike-pinned specifics).
- **Type/name consistency:** `get_disttranspose_plan`, `to_spec_solve`/`from_spec_solve!`, `cfg.pencils.θ_comm`/`r_comm` (T0–T3); axis-permuted physical↔spatial copy (T2–T3); v_r factors preserved per path (T3).
- **Known soft spots:** the Alm↔spec_solve transpose (T0), the (φ,θ,lev)↔(θ,φ,r) axis permutation in the physical copy (verified pattern exists), uneven-nr `nlev` per rank.
