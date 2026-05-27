# Conducting Inner Core (Magnetic) — Design Spec

**Date:** 2026-05-26
**Status:** Design (pre-implementation)
**Author:** brainstorming session

## Problem

The magnetic solver advertises a "conducting inner core" boundary condition
(`docs/src/boundary-conditions.md`, enum `CONTINUITY_MAG` in `src/bcs/bcs.jl:161`,
param `inner_core_conductivity_ratio`), but it is **not implemented**. The
inner-core spectral fields `𝒯ⁱᶜ/𝒫ⁱᶜ` are allocated and zeroed but never evolved
or coupled at the ICB; the documented public API `enforce_magnetic_boundary_constraints!`
does not exist. A run with a "conducting inner core" is byte-for-byte identical to
an insulating inner core. Confirmed by the RED acceptance test
`test/magnetic_conducting_inner_core.jl` (inner-core field stays exactly `0.0`
after 30 MHD steps).

The insulating BCs themselves are correct and verified
(`test/magnetic_boundary_numerical.jl`).

## Scope

**This spec (MVP):**
- Equal conductivity: `σ_ic = σ_oc` ⇒ `η_ic = η_oc = η` (ratio = 1).
- Inner core co-rotating with the frame: no differential rotation, no advection ⇒
  inner core evolves by **pure magnetic diffusion**.
- `CNAB2` timestepper (the default implicit path).
- Both toroidal and poloidal.

**Explicit follow-ups (out of scope here):**
- Variable conductivity ratio `σ_ic/σ_oc ≠ 1` (derivative-jump matching; wire the
  existing `inner_core_conductivity_ratio`).
- Inner-core differential/prescribed rotation (advection term; replaces the dead
  `apply_inner_core_rotation!` scaffolding).
- `EAB2`/`ERK2` timesteppers (currently `EAB2` already throws on `CONTINUITY_MAG`).

## Physics

Toroidal/poloidal decomposition **B** = ∇×(T**r**) + ∇×∇×(P**r**). Per
spherical-harmonic degree `l`, each scalar `S ∈ {T, P}` obeys a radial diffusion
operator `∇²_l = ∂²/∂r² + (2/r)∂/∂r − l(l+1)/r²`.

- **Outer core** (shell `[ri, ro]`): `∂S/∂t = η ∇²_l S + N_S` (`N_S` = induction
  nonlinear term).
- **Inner core** (ball `[0, ri]`): `∂S/∂t = η ∇²_l S` (no flow, no rotation).
  Regularity at `r=0`: `S ~ r^l` (ball domain enforces this via
  `enforce_ball_*_regularity!`).
- **ICB** (`r = ri`), equal σ: continuity of **value and radial derivative** for
  both `T` and `P`: `S_oc(ri) = S_ic(ri)`, `∂S_oc/∂r(ri) = ∂S_ic/∂r(ri)`.
- **CMB** (`r = ro`): insulating, unchanged — toroidal `T=0`,
  poloidal `(∂/∂r + (l+1)/r)P = 0`.

## Approaches considered

**A. Combined single grid `[0, ro]`.** One diffusion domain; nonlinear term active
only for `r ≥ ri`. Matching is automatic (continuous discretization). *Cleanest math
but architecturally invasive* — the magnetic field radial dimension would change from
`nr` to `nr_inner + nr − 1`, touching field allocation, pencils, transforms, IO,
diagnostics, and the induction term. Rejected for MVP (too broad, risks the
known-good outer-core path).

**B. Two-domain bordered/block solve.** Keep fields on their domains; assemble a
combined per-`l` system `[IC dofs ; OC dofs]` with interface rows enforcing value +
derivative continuity. Architecturally aligned; directly evolves `𝒯ⁱᶜ/𝒫ⁱᶜ`. Cost:
new bordered-banded assembly + custom solve per `l`.

**C. Schur-complement / admittance (RECOMMENDED).** Eliminate the IC interior
analytically and impose its effect as a modified inner Robin BC on the outer-core
solve, reusing the **verified** matrix-embedded BC machinery and the existing
influence-matrix precedent (`erk2.jl`). Exact for the implicit operator (not an
approximation). Lowest risk: the insulating path is the special case, so it is a
clean superset.

### Chosen: Approach C

Per unique `l`, on the IC ball grid, build the implicit operator
`M_ic = (1/dt)I − θη∇²_l` (regularity at `r=0`). The IC response to an ICB value `g`
and history RHS `b_ic` is linear:

```
∂S_ic/∂r(ri) = α_l · g + φ0(b_ic)
```

- `α_l` — homogeneous admittance: derivative at ICB from `M_ic x = 0`, unit Dirichlet
  `x(ri)=1`, regularity at `r=0`. Precomputed once per `(l, dt)`.
- `φ0` — flux from the IC history `b_ic = (1/dt + (1−θ)η∇²_l)S_ic^n` with `g=0`.
  Recomputed each step from `S_ic^n`.

Outer-core inner row becomes the Robin condition (continuity of derivative):

```
(∂/∂r − α_l) S_oc(ri) = φ0          # replaces insulating inner row
```

CMB outer row unchanged (insulating). After the OC solve, set `g = S_oc(ri)` and
reconstruct `S_ic` from `M_ic S_ic = b_ic` with Dirichlet `g`. Both `𝒯/𝒫` and
`𝒯ⁱᶜ/𝒫ⁱᶜ` are updated; value continuity holds by construction, derivative continuity
by the Robin row. **Insulating is recovered when `α_l` equals the insulating
coefficient**, so the conducting path is opt-in and cannot perturb the default.

## Components / files

- **New** `src/physics/magnetic/inner_core.jl` (or section in `magnetic_bc.jl`):
  - `create_inner_core_admittance(config, ic_domain, η, dt; θ)` → per-`l` `M_ic`
    factorization + `α_l` (toroidal and poloidal variants; tor regularity differs
    only in BC bookkeeping).
  - `inner_core_history_flux!(φ0, S_ic_old, ...)` → per-step `φ0`.
  - `reconstruct_inner_core!(S_ic, g, b_ic, M_ic_lu)` → IC field after OC solve.
- **Modify** `src/bcs/magnetic_bc.jl`: conducting variants of
  `create_magnetic_{toroidal,poloidal}_matrices` that place `(∂/∂r − α_l)` on the
  inner row instead of the insulating coefficient. Gate on a flag.
- **Modify** `src/physics/magnetic/solver.jl`:
  `apply_magnetic_{toroidal,poloidal}_implicit_update!` — when conducting IC is on,
  add `φ0` to the inner RHS and call `reconstruct_inner_core!` after the OC solve.
  Supersede the existing `_magnetic_toroidal_inner_bc_increment` (`-nl_pol`) coupling
  for the conducting case.
- **Modify** `src/core/parameters.jl` + `src/api/model.jl`: add
  `magnetic_inner_bc::Symbol = :insulating` (accept `:conducting_inner_core`); when
  conducting, set `bc_type_inner .= CONTINUITY_MAG` on magnetic `𝒯/𝒫` and select the
  conducting matrices in backend setup (`src/solver/backend.jl`).
- **Docs** `docs/src/boundary-conditions.md`, `configuration.md`: flip status from
  "not implemented" to documented usage once GREEN.

## Data flow (per magnetic step, conducting IC on)

1. Compute induction nonlinear term `N_S` (OC only) — unchanged.
2. `φ0 ← inner_core_history_flux!(S_ic^n)`.
3. Build CNAB2 OC RHS; set inner boundary RHS row = `φ0`.
4. Solve OC implicit system (Robin inner row with `α_l`, insulating outer) → `S_oc^{n+1}`.
5. `g ← S_oc^{n+1}(ri)`; `reconstruct_inner_core!` → `S_ic^{n+1}`.

## Testing / acceptance

1. **`test/magnetic_conducting_inner_core.jl`** (exists, currently RED): inner-core
   field becomes nonzero after stepping → must go GREEN.
2. **ICB continuity** (add): after a step, `S_ic(ri) ≈ S_oc(ri)` and
   `∂S_ic/∂r(ri) ≈ ∂S_oc/∂r(ri)` to round-off / FD-truncation tolerance, per `l`.
3. **Admittance sanity** (add, unit): `α_l` reproduces the analytic IC diffusion
   response on a coarse grid; insulating limit recovers the existing inner row.
4. **Regression**: existing `magnetic_boundary_{numerical,static}_checks.jl` still
   pass (insulating path byte-for-byte unchanged; conducting is opt-in).
5. **Stretch (physical validation)**: with `velocity = 0` and uniform `η`, the
   coupled IC+OC dipole free-decay rate matches the analytic slowest-decay mode.

## Risks

- Sign/stencil of the one-sided ICB derivative used for `α_l` and `φ0`.
- CNAB2 `θ`-weighting of the history flux `φ0`.
- Keeping the insulating default exactly unchanged (opt-in gate must be airtight).
- Inner-core ball domain `1/r` handling near `r=0` (guarded in `Ball.jl:52`; verify
  for the diffusion operator at small `l`).
