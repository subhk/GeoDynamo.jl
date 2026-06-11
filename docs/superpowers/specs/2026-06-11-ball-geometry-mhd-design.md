# Ball Geometry (Full-Sphere) MHD Port — Design

**Date:** 2026-06-11
**Status:** Approved design, pre-implementation
**Depends on:** Stage 1–4B double-curl work (`2026-06-10-poloidal-momentum-double-curl-design.md`), ERK2 W-split port (`2026-06-11-erk2-wsplit-port.md`)

## 1. Problem

Ball geometry (`geometry = :ball`, full sphere, no inner core) is currently
inconsistent and physically wrong:

- The ball nonlinear branches (`finish_velocity_nonlinear!`,
  `apply_induction_nonlinear!`) still call the legacy potential-style
  `solver_ball_vector_analysis!` — the **same dropped-Q defect** the shell
  momentum equation had (buoyancy never enters momentum; induction curl wrong).
- The poloidal implicit update is geometry-blind: ball CNAB2 runs the W-split
  expecting `nl_poloidal = N_W`, but the legacy ball branch stores potentials —
  the implicit step and the nonlinear assembly disagree.
- The ball radial grid includes a node exactly at r=0; negative powers
  (1/r, 1/r²) are hack-set to 0 there. This silently zeroes the *finite,
  nonzero* l=1 center values (u_r = λP/r² and S = P′/r have finite limits at
  r=0 for l=1) — center-crossing flow is wrong.
- Test coverage is transforms/finiteness only; no ball solver stepping tested.

## 2. Goal

Full MHD in a ball: velocity W-split + scalar transport + magnetic induction
correct in full-sphere geometry, on CNAB2 and ERK2, validated by analytic decay
rates, internal eigenvalue probes, and the Marti et al. (2014) full-sphere
benchmark.

**Non-goals:** EAB2/theta-method ball support (stays loud-gated, same message
as shell); conducting inner core for ball (meaningless — keeps its existing
error); GPU ball.

## 3. Approach (chosen: Unify)

Delete the legacy ball branches. Ball flows through the **same Stage-4
projection code as the shell** — geometry-blind. This is possible because the
new grid has no r=0 node, so every 1/r site in transforms/projections is
finite. Geometry differences are confined to exactly three places:

1. the radial domain builder (off-center nodes),
2. inner-boundary matrix rows (l-dependent regularity instead of wall
   conditions),
3. the magnetic outer BC (insulating exterior — already the non-conducting
   default path).

Rejected alternatives: a parallel ball-specific projection path (code
divergence, same physics); ball-as-ε-shell (ε-sensitivity, never exact for
l=1).

## 4. Grid + operators

`create_ball_radial_domain` (src/Ball/Ball.jl) new nodes:

```
r_n = (1 − cos(π·n/N)) / 2,   n = 1..N
```

- `r_N = 1` exactly (outer wall on grid).
- `r_1 = (1 − cos(π/N))/2 > 0` — no node at the center.
- Same cosine clustering as before (dense near center and wall); the old grid
  is this family with the n=0 node included.
- The negative-power regularization hack is removed: all 7 r-power columns
  (`r[:,1..7]`) are honest finite values.
- FD banded operators (`_populate_radial_operators!`,
  `create_derivative_matrix`) already handle arbitrary non-uniform nodes —
  unchanged.

`SOLVER_BALL_DOMAIN_BUILDER` (src/solver/backend.jl) picks this up unchanged.

## 5. Regularity rows (core of the port)

Smooth fields near the center behave as:

- poloidal P, toroidal T (velocity and magnetic): ~ r^{l+1}
- scalars Θ: ~ r^l
- W = D_pol·P: ~ r^{l+1}

(W derivation: P = a·r^{l+1} + b·r^{l+3} + …; D_pol = ∂_rr − l(l+1)/r²
annihilates r^{l+1}, and D_pol(r^{l+3}) = (4l+6)·r^{l+1}, so W's leading term
is r^{l+1}.)

Every inner-boundary row therefore becomes one Robin form, exact for the
leading behavior:

```
f′(r₁) = β · f(r₁) / r₁
   β = l+1   for velocity P, T, W, magnetic P, T
   β = l     for scalars (l=0 gives Θ′(r₁)=0 — Neumann falls out automatically)
```

Implementation: matrix builders that currently stamp inner *wall* rows gain a
`:regularity` inner mode, selected when `geometry === :ball`. All affected
matrix sets are already per-l (PoloidalSplitMatrices lookup; scalar / toroidal
/ magnetic system matrices), so l-dependent rows are natural.

Per system:

| System | Inner row (ball) | Outer row (unchanged) |
|---|---|---|
| Velocity toroidal | T′ = (l+1)T/r₁ | no-slip T=0 / stress-free T′=T/r |
| W solve (`w_factor`) | W′ = (l+1)W/r₁ | Dirichlet W |
| P recovery (`p_factor`) | P′ = (l+1)P/r₁ | P=0 wall |
| Scalars (T, C) | Θ′ = l·Θ/r₁ | per BC code |
| Magnetic toroidal | T′ = (l+1)T/r₁ | insulating T=0 |
| Magnetic poloidal | P′ = (l+1)P/r₁ | insulating ∂rP + (l+1)P/r = 0 |

**Influence correction shrinks 2×2 → 1×1 for ball.** Only the outer no-slip
(P′=0) / stress-free (P″−(2/r)P′=0) condition needs a Green column; the inner
regularity condition sits directly in the factored `w_factor`/`p_factor`
matrices and needs no correction. `create_velocity_poloidal_split_matrices`
builds one Green pair (g_outer, h_outer) and a scalar influence coefficient
for ball instead of the 2×2 M.

## 6. Nonlinear paths

- Delete the `geometry === :ball` branch in `finish_velocity_nonlinear!`
  (src/physics/velocity/solver.jl) — ball uses the projection path:
  raw sphtor analysis + Q_F + `N_W = ∂_r(r·S_F) − Q_F`.
- Delete the `geometry === :ball` branch in `apply_induction_nonlinear!`
  (src/solver/numerics.jl) — ball uses the projection-based induction curl
  (P_{∇×E} = −r·T_E, T_{∇×E} = −(Q_E − ∂r(rS_E))/r).
- `solver_ball_vector_analysis!` becomes dead → removed.
- The r=0-plane zeroing helpers (`enforce_ball_scalar_regularity!`,
  `enforce_ball_vector_regularity!`, `apply_ball_*_regularity!`,
  `ball_physical_to_spectral!`, `ball_vector_analysis!` in src/Ball/Ball.jl)
  reference a grid plane that no longer exists → removed from the solver path
  and from the Ball module exports. Regularity is now enforced by the implicit
  matrices' Robin rows, not by post-hoc plane zeroing.
- Velocity/magnetic transforms (`_solenoidal_vr_factor` = λ/r², S = P′/r,
  Q-based poloidal analysis P = r²Q/λ, vorticity formulas) run unmodified —
  finite everywhere on the new grid; l=1 center-crossing flow now exact.
- Scalar-gradient path (`transform_field_and_gradients_to_physical!`) runs
  unmodified — the 1/r tangential scaling is finite on the new grid.
- Ball conductive profile (src/physics/temperature/solver.jl:25) unchanged.

## 7. Timesteppers

- **CNAB2:** `_apply_poloidal_wsplit_cnab2!` unchanged in structure; ball
  passes through with the regularity-row split matrices and 1×1 influence.
- **ERK2:** cache builders (src/timestep/erk2/cache.jl) gain the regularity
  inner-row option threaded like `dpol_operator`; the poloidal recovery helper
  `_erk2_poloidal_recover!` uses the outer-only φ1 Green column for ball.
- **EAB2 / theta:** `_VEL_POL_STAGE4B_MSG` gate unchanged.

## 8. Validation

1. **Spherical-Bessel decay (analytic anchor).** Scalar diffusion in the ball:
   init Θ = j_l(α_{lk}·r) for a mode (l,m); exact decay rate σ = −κ·α_{lk}²
   where α_{lk} is the k-th positive zero of j_l (Θ(1)=0 Dirichlet outer).
   Same construction for the velocity toroidal equation. Sharp quantitative
   targets, seconds of compute. Tolerance: FD convergence order at the test
   resolution, asserted with margin (e.g. rel. error < 1e-3 at nr=48,
   improving under refinement).
2. **W-split decay-vs-eigenvalue probe.** Dense constrained eigenvalue of
   σ·D_pol·p = D_pol²·p with ball regularity + outer wall rows vs the
   time-stepped unforced decay — the same instrument that exonerated the
   shell W-split (agreement ~1e-3 relative).
3. **Solenoidality + transform roundtrip** on the ball grid: spectral
   per-mode divergence at machine precision; synthesis/analysis roundtrip.
4. **Hydro convective onset** in the ball (internal heating profile): step-1
   nl_pol nonzero, kinetic-energy growth above critical Ra; subcritical decay.
5. **ERK2-vs-CNAB2 consistency** on a short full-MHD ball run (relΔ < 0.05).
6. **Full-MHD stability:** N steps finite, no NaN, magnetic energy bounded.
7. **Marti et al. (2014) full-sphere benchmark**
   (`scripts/marti_ball_benchmark.jl`): hydrodynamic case first (their
   nondimensionalization mapped to GeoDynamo parameters, documented in the
   script header like the Christensen Case-0 mapping); dynamo case if the
   hydro case validates. Published target values are read from the paper at
   implementation time — not hard-coded from memory. Long-run executed like
   Case-0 (background, frozen script copy); outcome reported, not a CI test.

Items 1–6 are CI tests; item 7 is a benchmark script + report.

## 9. File map

| File | Change |
|---|---|
| `src/Ball/Ball.jl` | New off-center grid in `create_ball_radial_domain`; remove plane-zeroing helpers + their exports |
| `src/bcs/velocity_bc.jl` | `:regularity` inner rows in `create_velocity_toroidal_matrices` + `create_velocity_poloidal_split_matrices`; 1×1 ball influence |
| `src/bcs/thermal_bc.jl`, `src/bcs/compositional_bc.jl` | `:regularity` inner row (β = l) |
| `src/bcs/magnetic_bc.jl` | `:regularity` inner rows in toroidal + poloidal builders |
| `src/physics/velocity/solver.jl` | Delete ball branch in `finish_velocity_nonlinear!`; thread geometry into split builder |
| `src/solver/numerics.jl` | Delete ball branch in `apply_induction_nonlinear!`; remove `solver_ball_vector_analysis!` |
| `src/solver/backend.jl` | Thread geometry into implicit-matrix builders |
| `src/timestep/erk2/cache.jl`, `src/timestep/erk2/integrate.jl` | Regularity inner-row option; outer-only Green column for ball |
| `test/ball_roundtrip.jl`, `test/ball_finiteness.jl` | Update for off-center grid (no r=0 plane) |
| `test/ball_bessel_decay.jl` (new) | Validation item 1 |
| `test/ball_wsplit_eigen.jl` (new) | Validation item 2 |
| `test/ball_solver_physics.jl` (new) | Validation items 3–6 |
| `scripts/marti_ball_benchmark.jl` (new) | Validation item 7 |

## 10. Risks

- **Robin regularity rows are leading-order exact, not spectral-exact.** The
  next-order term in r^{l+3} introduces an O(r₁²) consistency error at the
  inner boundary; r₁ ~ (π/N)²/2 shrinks quadratically with resolution, so
  the scheme converges. The Bessel-decay test quantifies this directly.
- **Conditioning of λ/r² near center.** r₁² ~ N⁻⁴, so λ/r₁² is large at high
  l; banded LU on per-l matrices has handled stiff rows in the shell (small
  ri), but the eigenvalue probe (item 2) will expose any pathology.
- **Static-check pins.** Source-text asserts in `test/static_checks.jl` have
  broken on every refactor here; repoint as part of each task.
- **Concurrent sessions.** Repo receives commits/branch switches mid-work:
  scoped `git add` only, commit promptly, merge on diverge, never rebase.
