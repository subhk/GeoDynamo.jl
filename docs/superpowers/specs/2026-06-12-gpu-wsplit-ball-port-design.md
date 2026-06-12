# GPU Port of the W-Split Velocity Dynamics + Ball Geometry — Design

**Date:** 2026-06-12
**Status:** Approved design, pre-implementation
**Base:** branch `feat/ball-geometry-mhd` @ `41af101` (stacks on PR #78; worktree `../GeoDynamo-ball`)
**Depends on:** Stage 1–4B double-curl (spec 2026-06-10), ERK2 W-split port, ball geometry port (spec 2026-06-11), GPU phases 0–6 (memory: all built + CPU-validated on the Array backend)

## 1. Problem

The single-GPU port (phases 0–6) predates the Stage-2/4B physics overhaul. Its
vector transforms are loud-gated (`_GPU_VECTOR_STAGE2_MSG` in
`src/gpu/vector_transform.jl` — both directions `error()` before their old
pre-Stage-2 bodies), which chokes every consumer: `gpu_velocity_nonlinear!`,
`gpu_magnetic_nonlinear!`, and `gpu_solver_step!` (6 call sites). Beyond the
transforms, the GPU step still implements the OLD physics throughout:

- `gpu_velocity_field_step!` runs the legacy poloidal CNAB2 (direct poloidal
  implicit solve + legacy 2×2 influence) instead of the W-split.
- `gpu_velocity_nonlinear!` uses the old curl projections (the buoyancy-dead
  formulation the CPU abandoned).
- `gpu_magnetic_nonlinear!` uses the old induction analysis (dropped-Q).
- The GPU scalar-gradient path mirrors the old sinθ-weighted tangential
  advection (the CPU fix never reached GPU).
- The spectral curls (vorticity/current) use pre-Stage-2 formulas.
- No ball-geometry awareness.

The PR #59 CI gates assert these `error()`s via `@test_throws` — un-gating
flips them back into real equivalence assertions.

## 2. Goal

`gpu_solver_step!` reproduces CPU `solver_step!` (CNAB2, full MHD) on the
Array backend for BOTH shell and ball geometry — bit-exact where loop order
allows, ≤1e-13 relative otherwise (documented per kernel). `[GPU-BOX]` CUDA
twins stay marked for hardware validation elsewhere (no local CUDA).

**Non-goals:** ERK2 on GPU (never existed; stays loud-gated with a pointer
message), EAB2 on GPU (same), conducting inner core on GPU (existing
device-state gate unchanged), distributed/MPI GPU.

## 3. Approach (chosen: mirror-the-CPU, batched)

Re-implement each changed CPU kernel in the GPU module's established style:
split-complex (real/imag) 3-D arrays `(l_slot, m_slot, r)`, batched per-l
banded operations and broadcasts, per-level serial SHTnsKit transforms with
CuArray methods supplied by the extension. All matrices, factorizations,
Green responses, and residual rows are HOST-built and device-copied — the
ball's regularity rows and the W-split operators arrive correct for free;
only the step ALGEBRA needs GPU code.

Rejected: host round-trip scaffolding (defeats the purpose; debugging crutch
only, never committed); KernelAbstractions single-source rewrite (too
invasive for this scope).

## 4. Vector transforms (`src/gpu/vector_transform.jl`)

Delete `_GPU_VECTOR_STAGE2_MSG`, both `error()` lines, and the old bodies.
New bodies match the CPU Stage-2 solenoidal convention
(`vector_spectral_to_physical_disttranspose!` / `vector_physical_to_spectral!`
in `src/solver/numerics.jl`):

**Synthesis** `gpu_vector_spectral_to_physical!`:
1. Spheroidal scratch: `S = (∂_r P)/r` — batched banded-d1 over the radial
   axis of the poloidal coefficients (existing `gpu_banded_mul!`-style per-l
   application of the SAME host d1 operator the CPU uses) times an `rinv`
   broadcast. Written to scratch arrays, NOT in place (P is still needed).
2. Tangential per level: `synthesis_sphtor(S_k, T_k)` (unchanged plumbing).
3. Radial: `v_r = scalar_synth(P · λ/r²)` — the existing `gpu_vr_scale!` with
   `rscale = rinv2` (was `rinv`) and `lfac[l+1] = l(l+1)`.
4. Optional `raw_spheroidal::Bool = false` kwarg: skip step 1 and pass the
   stored coefficients directly as the sphtor S (the tangential-basis
   primitive the CPU exposes — needed by the scalar-gradient fix and the
   force analysis).

**Analysis** `gpu_vector_physical_to_spectral!`:
1. Per level `analysis_sphtor(vθ_k, vφ_k)` → raw (S, T).
2. Default mode: toroidal ← T; poloidal ← Q-based recovery — scalar analysis
   of the radial component `v_r` → Q, then `P = r²Q/λ` (batched broadcast;
   l=0 slot zeroed), matching CPU `_poloidal_from_radial_q!`. The signature
   gains the `vr` physical argument.
3. `raw_spheroidal::Bool = false` kwarg: store raw S into the poloidal slot
   (no Q recovery; used by force/induction analysis).

## 5. Nonlinear assembly

- **Velocity** (`src/gpu/velocity_nonlinear.jl`): port Stage-4B
  `finish_velocity_nonlinear!` + `_poloidal_force_projection!`: assemble the
  physical force (advection + buoyancy + Lorentz — existing batched code),
  then raw sphtor analysis → (T_F, S_F); scalar analysis of the force radial
  component → Q_F; `nl_tor = T_F`; `nl_pol = N_W = ∂_r(r·S_F) − Q_F`
  (batched: r broadcast, banded-d1, subtract). The old projection body is
  deleted.
- **Induction** (`src/gpu/magnetic_nonlinear.jl`): port Stage-4A
  `_induction_curl_potentials!`: raw analysis of u×B → (T_E, S_E), scalar
  analysis of (u×B)_r → Q_E; `nl_pol = −r·T_E`,
  `nl_tor = −(Q_E − ∂_r(r·S_E))/r`.
- **Scalar advection** (`src/gpu/scalar_gradient.jl` / `scalar_nonlinear.jl`):
  port the sinθ fix — tangential gradient via raw sphtor SYNTHESIS of
  (gradient-spectral, field-spectral) pair + `1/r` scaling, replacing the old
  sinθ-weighted recurrence outputs, mirroring CPU
  `transform_field_and_gradients_to_physical!`. Radial gradient unchanged.
- **Spectral curls** (`src/gpu/spectral_curl.jl`): vorticity/current to the
  verified Stage-2 formulas `T_ω = (P″ − λP/r²)/r` (batched banded-d2),
  `P_ω = −r·T`.

## 6. W-split velocity step (`src/gpu/velocity_step.jl`)

Toroidal half unchanged (already the right form: CNAB2 RHS + banded solve with
BC rows). Poloidal half replaced by the W-split, mirroring
`_apply_poloidal_wsplit_cnab2!` as batched ops over all modes:

1. `W = D_pol·P` (batched per-l banded mul, host-built `dpol_op`).
2. CNAB2 RHS in W: `(Ek/dt)W + (1−θ)·Ek·D_pol·W + 1.5·nl − 0.5·nl_prev`
   (`w_linear` banded mul + broadcasts; nl_pol now carries N_W).
3. Batched `w_factor` solve (existing GPU banded-LU machinery).
4. Ball-only: ρ₁w residual dots BEFORE wall-zeroing (`d1_row_inner` dot over
   r per mode − `(l+1)·reg_r_inv·Wp[1,…]` — per-l λ factor via an l-indexed
   broadcast vector).
5. Zero rows 1, N; batched `p_factor` solve → Pp.
6. Residuals: shell ρ₁ = d1_row_inner·Pp; ρ₂ = d1_row_outer·Pp (both
   geometries). Batched dot over the radial axis.
7. Per-mode 2×2 influence solve (the influence matrices are per-l — gather
   via the existing l-indexed lookup arrays) and the correction
   `P = Pp + a₁h₁ + a₂h₂` (broadcast over modes with per-l h columns).
8. l=0 modes zeroed (as CPU).

`gpu_velocity_poloidal_influence_correction!` (legacy) loses its consumer —
deleted with its tests repointed.

## 7. Device-state additions (`src/gpu/device_state.jl`)

`build_gpu_solver_state` additionally packs from the host
`PoloidalSplitMatrices` (built lazily on the host exactly as the CPU step
does — reuse `_get_or_build_poloidal_split!` or call the builder directly):
per-l `dpol_op`/`w_linear` banded data, `w_factor`/`p_factor` LUs in the
existing device banded-LU format, `g1/g2/h1/h2` columns, 2×2 influence
entries, `d1_row_inner/outer`, `ball::Bool`, `reg_r_inv::Float64`, and the
per-mode l-index map for gathering per-l data (the existing idiom). Ball
domains contribute `rinv/rinv2` columns from the off-center grid — all
finite, no special-casing.

## 8. Validation

1. **Kernel equivalence** (Array backend, random band-limited inputs): each
   new/changed GPU kernel vs its CPU counterpart — transforms (both modes),
   N_W projection, induction potentials, scalar tangential gradient, curls,
   W-split single application. Bit-exact where the loop order matches;
   otherwise ≤1e-13 relative, documented per kernel in the test.
2. **Step equivalence (the 5n2 instrument, un-gated):**
   `gpu_solver_step!` ≈ `solver_step!`, full MHD, N-step trajectories, on
   SHELL and BALL fixtures (the PR #59 `@test_throws` gates flip back to
   equivalence assertions; the ball fixture mirrors
   `test/ball_solver_physics.jl`).
3. **Run-loop:** `gpu_run!` N-step trajectory matches CPU (existing Phase-6
   harness, re-enabled).
4. `[GPU-BOX]` twins of every gate marked for the CUDA box; Array-only here.
5. ERK2/EAB2-GPU: loud gates asserted via `@test_throws` with pointer
   messages.

## 9. File map

| File | Change |
|---|---|
| `src/gpu/vector_transform.jl` | New Stage-2 bodies, `raw_spheroidal` kwarg, `vr` analysis arg; gate deleted |
| `src/gpu/velocity_nonlinear.jl` | Stage-4B force projection (N_W) |
| `src/gpu/magnetic_nonlinear.jl` | Stage-4A induction curl potentials |
| `src/gpu/scalar_gradient.jl`, `src/gpu/scalar_nonlinear.jl` | sinθ-fix tangential gradient |
| `src/gpu/spectral_curl.jl` | Stage-2 vorticity/current formulas |
| `src/gpu/velocity_step.jl` | W-split poloidal half (shell + ball mixed influence) |
| `src/gpu/influence_correction.jl` | Legacy correction deleted (or reduced to the 2×2 solve helper if reused) |
| `src/gpu/device_state.jl` | Pack PoloidalSplitMatrices + ball flags |
| `src/gpu/solver_step.jl`, `src/gpu/run.jl` | Wiring for the new signatures |
| `test/gpu_phase3_vector_transform.jl` etc. | Gates un-gated → equivalence asserts; new kernel tests; ball step-equivalence fixture |

## 10. Risks

- **Loop-order FP drift:** batched GPU algebra reorders sums vs the CPU's
  per-mode loops — bit-exactness may not hold for the W-split resid, hence
  the documented ≤1e-13 fallback. The previous port achieved bit-exact
  scalars/magnetic and sub-ulp velocity; expect the same class.
- **Hidden old-convention consumers:** the legacy transform bodies were the
  reference for several phase tests — every `@test_throws` flip must be
  accompanied by re-derived expected values (CPU-generated), not the old
  pre-Stage-2 snapshots.
- **Scalar-gradient signature ripples:** the sinθ fix changes the gradient
  pathway shape (raw sphtor synthesis of a PAIR) — `gpu_scalar_field_step!`
  callers need the work-spectral scratch threaded.
- **Concurrent sessions:** repo gets external commits/branch switches —
  worktree isolates; scoped adds only.
