# Rigorous Poloidal Momentum (Double-Curl) Design

**Status:** draft for review
**Date:** 2026-06-10
**Decision driver:** the timestepper-correctness audit found that the radial
component of the assembled body force — which carries **all** of thermal and
compositional buoyancy — is silently discarded by the force→spectral
conversion. Velocity at rest stays exactly at rest; convection cannot start.
The user chose the rigorous double-curl formulation over the two surgical
options (add/replace a radial term in the existing 2nd-order P-solve).

## 1. Findings this design must address (evidence)

1. **Buoyancy never enters the momentum equation.**
   `compute_velocity_body_forces!` (src/solver/numerics.jl) adds thermal and
   compositional buoyancy only into `adv_r`. `finish_velocity_nonlinear!` →
   `vector_physical_to_spectral!` → `vector_physical_to_spectral_disttranspose!`
   reads ONLY `θ_component`/`φ_component` (`dist_analysis_sphtor!`); the radial
   force component is never consumed. Verified live: 5 steps at Ra=1e4 with
   |T|=7.4 give |nl_tor|=|nl_pol|=0 and velocity exactly 0. The radial parts of
   Coriolis and Lorentz are dropped the same way.
   **Pre-dates every refactor**: the pre-rewrite backup (`../GeoDynamo_old.jl`,
   `src/solver/numerics.jl:849`) also analyzed only (v_θ, v_φ).

2. **The velocity synthesis is not a consistent solenoidal T-P representation.**
   Current synthesis (solver path): tangential spheroidal scalar S := P
   (stored poloidal coefficients fed to `dist_synthesis_sphtor!` directly), and
   u_r := l(l+1)·P/r (`vr_factor` in numerics.jl:838). For the standard
   representation u = ∇×∇×(P̂ r̂) + ∇×(T̂ r̂) the consistent pair is
   u_r = l(l+1)·P̂/r² and S = (1/r)∂_r(r·P̂) — the tangential scalar needs a
   radial derivative of P̂, which the current synthesis omits.
   `test/poloidal_solenoidality.jl` documents this: "in general the synthesized
   poloidal field is NOT divergence-free … at the transform level alone."
   Additionally the MIE path uses u_r = l(l+1)·P/r² while the solver path uses
   l(l+1)·P/r — two inconsistent radial conventions in one codebase.

3. **The poloidal evolution operator is a 2nd-order scalar diffusion solve**
   on P (same banded family as temperature), while the pressure-free rigorous
   poloidal momentum equation is 4th-order in radius.

4. Toroidal momentum is structurally sound: the toroidal projection of the
   tangential momentum equation has no pressure and no F_r dependence, and the
   existing T-scalar analysis of the tangential force is the right object
   (exact factor to be fixed in Stage 3 alongside the operator's variable
   convention).

5. **No test catches any of this.** There is no convective-onset test, no
   force-projection reference test. The full suite is green with buoyancy dead.

## 2. Target formulation

Adopt the standard toroidal–poloidal representation (Glatzmaier / Christensen
& Wicht / MagIC conventions adapted to this code's nondimensionalization):

    u = ∇×∇×(P r̂) + ∇×(T r̂)                                   (solenoidal by construction)
    u_r      = l(l+1)·P_lm/r²
    S_lm(u)  = (∂_r P_lm)/r        (tangential spheroidal scalar)
    T_lm(u)  = T_lm                      (tangential toroidal scalar)

Momentum (this code's scaling, mass coefficient Ek):

    Ek·∂t u = −Ek·(u×ω) − ẑ×u + (Pm/Pr)·Ra·r·Θ·r̂ + (1/Pm)·(∇×B)×B + Ek·∇²u − ∇p

Let F = all explicit forces (advection + Coriolis + buoyancy + Lorentz), with
QST decomposition Q_lm = F_r analysis, S_lm/T_lm = sphtor analysis of F_tan.

**Toroidal equation** — apply r̂·∇× (kills ∇p and F_r):

    Ek·(∂t − Δ_l)·[l(l+1)/r²]·T_lm = [r̂·∇×F]_lm = (l(l+1)/r²)·T_lm(F)
    ⇒ Ek·(∂t − Δ_l)·T_lm = T_lm(F)            (2nd-order solve — KEEP existing operator)

**Poloidal equation** — apply r̂·∇×∇× (kills ∇p, brings in F_r):

    Ek·(∂t − Δ_l)·[−Δ_l]·P_lm·(l(l+1)/r²) = [r̂·∇×∇×F]_lm

with the standard identity

    [r̂·∇×∇×F]_lm = (l(l+1)/r²)·[ Q_lm − ∂_r( r·S_lm(F) ) /1 ]   — exact coefficient
                                                  to be re-derived and VERIFIED
                                                  numerically in Stage 1 against a
                                                  brute-force finite-difference
                                                  curl-curl reference.

⇒ 4th-order in radius:  Ek·(∂t − Δ_l)(−Δ_l)P = Q − ∂_r(r·S_F), with 4 BCs on P
(no-penetration P=0 at both walls + no-slip ∂_r P=0 / stress-free combinations
— the BC set the code already encodes as `velocity_bc_code`).

Buoyancy is purely radial ⇒ enters through Q_lm — never droppable again by
construction.

## 3. Architecture: five verifiable stages

**Stage 1 — QST force analysis + reference-verified projection (pure addition).**
Add a distributed 3-component force analysis: Q from `dist_analysis!` of the
radial component (plumbing mirrors the existing v_r synthesis in
`vector_spectral_to_physical_disttranspose!`), S/T from the existing sphtor
path. New function `force_physical_to_qst!` beside the existing 2-component
analysis; nothing calls it from the dynamics yet. TDD: brute-force reference —
random smooth vector field on a dense grid, compute r̂·∇×F and r̂·∇×∇×F by
high-order finite differences + quadrature projection, assert the
spectral-side formulas match to discretization tolerance. This NAILS the
identity coefficients empirically before any dynamics change.

**Stage 2 — consistent solenoidal velocity transform pair.**
Synthesis: u_r = l(l+1)P/r², S = (1/r)∂_r(rP) (radial derivative via the
existing banded D_r operators on each mode profile before the sphtor
synthesis), unify the MIE/solver vr conventions. Analysis of a solenoidal u:
P = r²·Q(u)/(l(l+1)), T = T-scalar (analysis becomes Q-based instead of
S-based — the inverse of the new synthesis). Vorticity path
(`solver_compute_vorticity_spectral!`) re-derived for the same convention.
TDD: synthesis→analysis roundtrip exact on (T,P); synthesized field passes a
REAL divergence test (replace the documented non-solenoidal expectation in
`poloidal_solenoidality.jl` with a hard ∇·u ≈ 0 gate); manufactured solenoidal
field analysis recovers known (T,P).
⚠️ This changes the meaning of stored P — all existing trajectories/snapshots
shift; characterization baselines must be regenerated; expect many test
updates. GPU vector-transform ports follow in a later phase (flagged, not
silently divergent).

**Stage-3 finding (2026-06-10, derived during triage of the
`magnetic_conducting_inner_core` NaN):** under the now-consistent transform
pair, the DIFFUSION operators split by potential type. Using ΔB = −∇×∇×B for
solenoidal B and the Stage-1-verified curl-curl identity with Q = λP/r²,
S = P′/r:

    (ΔB)_r = (λ/r²)·(P″ − λP/r²)   ⇒  poloidal potentials diffuse with
                                        D_pol = ∂_rr − λ/r²   (NO 2/r term)
    Δ𝒯(T) = 𝒯(Δ_l T)               ⇒  toroidal potentials keep the full
                                        scalar Laplacian Δ_l = ∂_rr + (2/r)∂_r − λ/r²

The code currently builds ALL implicit/linear/exponential operators from the
full scalar Laplacian — correct for scalars and toroidal potentials, WRONG
(spurious (2/r)∂_r) for velocity-P and magnetic-P under the new convention.
This is the proximate cause candidate for the magnetic inner-core NaN and is
the entry point of Stage 4: the poloidal operator change must land together
with the 4th-order/split poloidal solve (matrix builders in bcs/*.jl,
CNAB2 explicit-L, EAB2 solver_build_banded_A, ERK2 caches).

**Stage 3 — toroidal equation factor audit (small).**
With Stage 2's conventions fixed, verify the toroidal RHS factor: the operator
evolves T_lm directly, so RHS must be T_lm(F) exactly (no stray l(l+1)/r²).
Adjust if needed. TDD: manufactured-solution convergence on a forced toroidal
mode.

**Stage 4 — 4th-order poloidal solve via splitting + influence matrices.**
Split (∂t − Δ_l)(−Δ_l)P = RHS into two banded 2nd-order solves:
W := −Δ_l P;  (Ek/dt − θ·Ek·Δ_l)W⁺ = RHS-with-AB2;  then −Δ_l P⁺ = W⁺.
The W-solve has no physical BCs of its own — the four P BCs couple the two
solves; enforce via the influence-matrix (Green's function) method. The code
already ships exactly this machinery for ERK2
(`create_solver_velocity_poloidal_influence_matrices`,
`apply_solver_velocity_poloidal_influence_correction!`) — generalize it to the
CNAB2/default path instead of inventing new structure. EAB2 poloidal: defer
(EAB2 has independent defects — first-order BC splitting + singular-operator
crash — tracked separately; gate it behind an explicit error for now rather
than silently wrong physics).
TDD: (a) steady Stokes balance — prescribed analytic F with known steady
solution; (b) temporal self-convergence stays order 2 for CNAB2; (c) the
audit's convective-onset test goes green: from rest with supercritical Ra,
kinetic energy grows; subcritical Ra decays.

**Stage 5 — full-system validation.**
Convective onset vs the known critical Rayleigh number for a rotating shell
(coarse bracketing, not publication-grade), energy-balance sanity, full suite,
and a new characterization snapshot for the corrected physics. Christensen et
al. (2001) benchmark Case 0 left as a follow-up acceptance target.

## 4. Explicitly out of scope

- GPU ports of the new transforms/solves (CPU first; GPU step-equivalence
  gates updated to expect the new physics in a follow-up phase).
- EAB2 poloidal path (gated with a loud error; EAB2's own defects tracked
  separately).
- Ball geometry (`:ball`) — shell first; ball regularity conditions differ at
  r→0 and follow once the shell formulation is validated.
- Pressure reconstruction/diagnostics (the double-curl eliminates p; a
  pressure Poisson diagnostic can come later).

## 5. Risks

- **Largest physics change in the codebase's history**: every dynamic
  trajectory changes (correctly). All bit-exact baselines break by design.
- The Stage-2 convention change is the riskiest step; Stage-1's
  reference-verified projections de-risk the math before dynamics move.
- Multi-session effort. Stages land independently green: 1 (pure addition),
  2 (transform pair + tests), 3 (small), 4 (poloidal solve), 5 (validation).

## 6. Stage-5 cross-validation against DD_2DCODE (2026-06-11) — CONFIRMED

The restored Fortran reference (`DD_2DCODE/`, untracked) independently confirms
every structural choice, with all apparent differences explained exactly by a
variable convention (their poloidal variable g vs ours P = r·g):

- `var_coll_TorPol2qst` (variables.f90:463): `q = l(l+1)/r·Pol`,
  `s = √(l(l+1))·(1/r + ∂_r)·Pol` — radial convention u_r = λg/r with the
  derivative-coupled spheroidal scalar. Ours: u_r = λP/r², S = ∂_rP/r.
  Identical under P = r·g (note ∂_r(rg)/r ≡ ∂_rP/r — the SAME tangential
  scalar).
- `tim_lumesh_X` + `radLap = ∂_rr + (2/r)∂_r` (meshs.f90:10): full Laplacian on
  g ≡ our D_pol = ∂_rr − λ/r² on P by exact conjugation:
  (∂_rr − λ/r²)(r·g) = r·Δ_l g.
- `non_velocity` (nonlinear.f90): force → `tra_rtp2qst` (3-component QST),
  **buoyancy added to the q scalar** (`cq += PrT·qRaT·r·T`), toroidal RHS from
  `qstllcurlr(t)`, poloidal RHS from `qstllcurlcurlr(q,s)` — our Stage-1/4B
  projections exactly (their curl-curl formula `λq/r² − √λ/r·(1/r+∂_r)s`
  matches our (λ/r²)(Q − ∂_r(rS)) under their √λ normalization).
- `vel_matrices` (velocity.f90:57): Green's-function influence method — delta
  sources at the endpoint rows, the XGre→XPol two-solve chain, a 2×2 inverted
  influence matrix per l — structurally identical to our
  `PoloidalSplitMatrices` (w_factor→p_factor chain, 2×2 correction).

**The original GeoDynamo defect, precisely isolated:** it was the Fortran
g-convention HALF-ported — u_r = λP/r and full-Laplacian operators (consistent
g-convention pieces) but with the `(1/r+∂_r)` spheroidal coupling missing from
the synthesis and the q scalar dropped from the force/induction projections.
Our Stage 2–4B is the standard P̂-convention implemented fully consistently;
the Fortran validates it end to end.
