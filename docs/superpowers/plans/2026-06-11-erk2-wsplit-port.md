# ERK2 → W-Split Port (velocity poloidal under the solenoidal convention)

> All pieces land together; the gate in `integrate_solver_erk2_step!` comes off
> only when the poloidal path is consistent. EAB2 stays gated (audit: 1st-order
> BC splitting + singular-operator crash — porting it would propagate a
> defective scheme; its overhaul is a separate effort).

**Variable choice:** evolve V := Ek·W = Ek·D_pol·P. Then
    ∂t V = D_pol·V + N_W
exactly (CNAB2-equivalent: Ek(∂t − D_pol)W = N_W). ERK2 cache built with
diffusivity 1.0 on the D_pol operator; nl_poloidal (= N_W since Stage 4B) feeds
the stage machinery UNSCALED. P-recovery: D_pol·P = V/Ek with Dirichlet walls +
influence corrections for the no-slip/stress-free residual rows.

**Influence Greens for the exponential update** (cached per l, lazily):
    g_i^full  = dt·φ1_full(l)·e_i      (i = wall rows 1, N)
    g_i^half  = (dt/2)·φ1_half(l)·e_i
    h_i = p_factor(l) \ R(g_i)         (R zeroes the Dirichlet RHS rows)
    M[j,i] = residual_row_j · h_i      (residual rows from PoloidalSplitMatrices)
Correction: ρ_j = residual_row_j·P̃; α = −M⁻¹ρ; V += Ek·Σαg (bookkeeping in V),
P = P̃ + Σαh. Same construction at stage (half) and finalize (full).

**Step flow changes in `integrate_solver_erk2_step!`:**
1. Entry: V-field := Ek·D_pol·P (per mode; `velocity.work_pol` hosts V — free
   during the update phase). vel_pol cache/buffers attach to the V-field +
   nl_poloidal.
2. prepare_solver_erk2_field!(vel_pol_buffers, V, ...) — UNCHANGED machinery.
3. apply_solver_erk2_stage!(vel_pol_buffers, V) → V holds the stage value;
   then `_erk2_poloidal_recover!(P ← V; half-Greens)` BEFORE
   compute_solver_nonlinear_terms! (the stage nonlinears must see a
   BC-consistent stage P).
4. store stage nl (nl_poloidal = N_W at the stage) — unchanged.
5. finalize_solver_erk2_field!(vel_pol_buffers, V, ...) → V⁺; then
   `_erk2_poloidal_recover!(P ← V; full-Greens)`.
6. DELETE the legacy `apply_solver_velocity_poloidal_influence_correction!`
   call and the legacy vel_pol influence-op build for this path.
7. Remove the entry gate.

**Cache wiring:** `get_solver_erk2_cache!(…, :velocity_poloidal, …)` builds
from D_pol (diffusivity 1.0): add an operator-kind argument or a dedicated
builder `create_solver_erk2_wsplit_cache` cloning the dense-method builder with
the D_pol per-l matrices (base d2 − λ/r² — same construction as
PoloidalSplitMatrices.dpol_op). The bc_spec endpoint handling inside
prepare/finalize: pass `bc_spec = nothing` for the V-field (endpoints are
handled by the recovery + influence; the `result[1]=result[nr]=0` default in
prepare/finalize is WRONG for V — V's walls carry the influence DOF. Audit
prepare/finalize: with bc_spec===nothing they zero endpoints — that suppresses
the wall response the Greens then re-inject; acceptable (the Green correction
supplies the wall DOF), but verify the residuals close (the gate test does).

**Gates (TDD):**
(a) ERK2-vs-CNAB2 trajectory consistency: same tiny state, same dt, 10 steps
    each from identical ICs — fields agree to O(dt²·steps) (both 2nd order,
    different schemes: assert rel-diff < few % and BOTH satisfy wall
    conditions P=0, residual-row=0 to 1e-6).
(b) restore `test/erk2_integration_step.jl` full-physics assertions (gate
    expectation comes OFF — finite + advanced + magnetic/composition).
(c) full suite green.
