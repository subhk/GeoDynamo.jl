# Stage 4B: Poloidal Momentum W-Split (Buoyancy Finally Drives Flow)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. ALL pieces below land TOGETHER — the nl_pol rewiring breaks the current P-equation semantics unless the W-split ships with it. Do not land partially.

**Goal:** Replace the velocity poloidal update with the pressure-free double-curl form, wiring the radial force (buoyancy) in, gated by a steady-state consistency test and the convective-onset acceptance test.

**State on entry:** suite fully green (5631/18/0). Transforms solenoidal & verified; magnetic dynamics consistent (D_pol + projection-based induction); toroidal momentum verified correct as-is. Velocity poloidal still evolves `(∂t − Δ_l)P = S_F` — wrong operator, wrong RHS, no buoyancy.

## The equations (all pieces already derived & machine-pinned)

    W        := D_pol·P,         D_pol = ∂_rr − λ/r²       (per-mode banded apply)
    advance:    Ek(∂t − D_pol)W = N_W,   N_W = ∂_r(r·S_F) − Q_F     (AB2 on N_W)
    recover:    D_pol·P⁺ = W⁺   with P(ri)=P(ro)=0 (Dirichlet rows)
    no-slip:    additionally P′(ri)=P′(ro)=0 — enforced via influence corrections on W⁺
    onset:      buoyancy F = (Pm/Pr)·Ra·r·Θ·r̂ enters through Q_F — supercritical Ra ⇒ growth

Notes: N_W has NO λ or r-power factors (they cancel: −(r²/λ)·R_pol = ∂_r(rS_F) − Q_F).
l=0 carries no poloidal content (skip; zero).

## Task 1: nl_poloidal projection = N_W (with the consumer switched in the same change)

`finish_velocity_nonlinear!` (src/physics/velocity/solver.jl:37): after the existing
raw tangential analysis (`vector_physical_to_spectral!(advection_physical, nl_toroidal,
nl_poloidal; raw_spheroidal=true)` — toroidal output stays T_F, CORRECT), do exactly what
`apply_induction_nonlinear!`/`_induction_curl_potentials!` (numerics.jl) does for the
magnetic side, adapted: Q_F via `scalar_physical_to_spectral!(advection_physical.r_component, <Q scratch>)`,
then per-mode in place: nl_pol ← ∂_r(r·S_F) − Q_F (banded D1 on r·S profiles; mirror
`_induction_curl_potentials!`'s loop; reuse `velocity.work_pol` as the Q scratch).
Identity gate (test): for a force synthesized from known (Q,S,T), nl_pol must equal the
closed form (restating Stage-1 identities — light test, do it).

## Task 2: PoloidalSplitMatrices + builder (src/bcs/velocity_bc.jl)

New struct (next to the existing ones):
    struct PoloidalSplitMatrices{T}
        dpol_op::Vector{BandedMatrix{T}}        # D_pol per l (for W := D_pol·P applies)
        w_system::Vector{BandedMatrix{T}}       # (Ek/dt)I − θ·Ek·D_pol, PDE rows at endpoints (NO BC rows)
        w_factor::Vector{BandedLU{T}}
        w_linear::Vector{BandedMatrix{T}}       # Ek·D_pol (explicit (1−θ) term)
        p_recovery::Vector{BandedMatrix{T}}     # D_pol with Dirichlet endpoint rows (P=0)
        p_factor::Vector{BandedLU{T}}
        influence::Vector{SMatrix-or-2x2-Matrix{T}}  # per-l 2×2 endpoint correction (no-slip)
        g1_W, g2_W, g1_P, g2_P::Vector{Vector{T}}    # cached Green responses per l
        l_values; lookup; theta
    end
Builder `create_velocity_poloidal_split_matrices(config, domain, Ek, dt; velocity_bc_code, theta=0.5, T)`:
base = `create_derivative_matrix(T,2,domain)`; per l subtract λ/r² on the diagonal → D_pol.
w_system: (Ek/dt)·I − θ·Ek·D_pol with NO row replacement (one-sided endpoint stencils are
part of the operator; the influence step owns the endpoint freedom). p_recovery: copy D_pol,
replace rows 1 and N with identity rows (Dirichlet P=0), factorize.
Influence (no-slip, velocity_bc_code==1): per l, for i∈{1,2}: solve w_system·g_i = e_i
(unit vector at row 1 / row N), then p_recovery-solve D_pol·h_i = g_i → h_i; build
M = [h_1′(ri) h_2′(ri); h_1′(ro) h_2′(ro)] (endpoint first-derivatives via the d1 rows);
factor/store M and (g_i, h_i). Stress-free (other codes): `error("stage 4B supports
no-slip first; stress-free poloidal split is a follow-up")` — loud, not wrong.

Register in the backend where `:velocity_pol` matrices are created (src/solver/backend.jl:364
area): build the split matrices alongside (keep the old set for the still-gated ERK2/EAB2 paths).

## Task 3: CNAB2 W-split update (src/physics/velocity/solver.jl)

Replace the CNAB2 branch of `apply_velocity_poloidal_implicit_update!`:
1. Per mode (r-local; assert like `_induction_curl_potentials!`): gather P profile,
   W = dpol_op[l]·P.
2. RHS = (Ek/dt)·W + (1−θ)·(w_linear[l]·W) + 3/2·nl_pol − 1/2·prev_nl_pol  (reuse
   `solver_build_rhs_cnab2!` if its mass/linear plumbing fits — it operates on
   SpectralFieldType triples with an ImplicitMatrixSet; if the new struct doesn't fit its
   interface, write the per-mode loop directly — it is ~15 lines and the existing kernel's
   structure is the template).
3. W⁺ = w_factor[l] \\ RHS.
4. Influence correction (no-slip): solve P̃ = p_factor[l] \\ W⁺; compute residuals
   ρ = [P̃′(ri); P̃′(ro)]; α = −M⁻¹ρ; W⁺ += α₁g_1 + α₂g_2; P⁺ = P̃ + α₁h_1 + α₂h_2
   (cached h_i avoids the second solve).
5. Write P⁺ back. Drop the old `apply_velocity_poloidal_no_penetration!` call from this
   branch (P=0 is exact by construction; P′=0 by influence) — keep it for the gated paths.
ERK2/EAB2 velocity-poloidal branches: `error("velocity poloidal under the solenoidal
convention is CNAB2-only until the exponential caches are ported (stage 4B)")`. ALSO gate
the ERK2/EAB2 magnetic-poloidal cache builders the same way (known inconsistency,
currently untested — add the loud error so it cannot be silently used).

## Task 4: Gates (TDD — written FIRST, red on the old path)

test/poloidal_momentum_split.jl:
(a) **Steady-state self-consistency:** constant manufactured force (single (l,m), smooth
    Q_F/S_F profiles), run CNAB2 to near-steady (‖ΔP‖/dt small); verify
    Ek·D_pol(D_pol·P_steady) ≈ −N_W within discretization tolerance on interior nodes
    (banded ops both sides — self-consistent, non-circular w.r.t. the stepping), and
    P(±)=0, P′(±)≈0.
(b) **Temporal order:** CNAB2 self-convergence on the new poloidal path stays ~2
    (smooth per-mode profiles per the D2≠D1∘D1 lesson).
(c) **CONVECTIVE ONSET (the acceptance):** full solver, from rest, temperature random IC:
    supercritical Ra (e.g. 1e6 at Ek=1e-2 coarse grid — calibrate empirically) ⇒ kinetic
    energy grows from 0 through >N steps and nl_pol ≠ 0 at step 1 (buoyancy projects);
    strongly subcritical Ra (e.g. 1) ⇒ any initial kinetic energy decays. The step-1
    nl_pol ≠ 0 assert alone already kills the original bug.
(d) Full suite green; characterization snapshots with nonzero velocity regenerate (named
    in the triage commit).

## Order: 4(a-c written red) → 2 → 3 → 1 → green → 4(d).
