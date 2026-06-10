# Double-Curl Stage 4: Consistent Dynamics (Magnetic First, Then Poloidal Momentum)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.

**Goal:** Make the dynamics consistent with the Stage-2 solenoidal transform pair: Part A fixes the magnetic equation (induction projections + poloidal diffusion operator) — independently landable, expected to cure the `magnetic_conducting_inner_core` NaN; Part B rebuilds the poloidal momentum solve (W-split + influence BCs) and wires the buoyancy-carrying `R_pol` force projection in.

**Derived equations (from the machine-pinned identities; see the spec):**

    Toroidal momentum:   Ek(∂t − Δ_l)T = T_F                       (current structure CORRECT; nl_tor = raw T_F ✓)
    Poloidal momentum:   Ek(∂t − D_pol)W = −(r²/λ)·R_pol(F),  W := D_pol·P,  then solve D_pol·P⁺ = W⁺ with 4 BCs (influence matrices)
    Magnetic toroidal:   (∂t − (1/Pm)Δ_l)T_B = −(r/λ)·R_pol(E)     (E = u×B)
    Magnetic poloidal:   (∂t − (1/Pm)D_pol)P_B = (r²/λ)·R_tor(E)
    D_pol = ∂_rr − λ/r²  (NO 2/r);  Δ_l = ∂_rr + (2/r)∂_r − λ/r²;  R_tor/R_pol = force_curl_projections!.

**Part A tasks (this stage's landable core):**
1. `create_poloidal_diffusion_operator` variant: base = `create_derivative_matrix(T,2,domain)` data, per-l subtract λ/r² — parallel to `create_shtns_timestepping_matrices` but WITHOUT the 2/r term, reusing the same BC-row embedding (`create_scalar_matrices`-style with `poloidal_base = true` kwarg threading, or a sibling builder). Magnetic-poloidal implicit/linear matrices switch to it; magnetic-toroidal unchanged.
2. Magnetic induction projections: in the magnetic nonlinear finish (analysis of E=u×B), replace the 2-component analysis with `force_physical_to_qst!(E, Q,S,T)` + `force_curl_projections!` + per-mode scaling `nl_tor_B = −(r/λ)R_pol`, `nl_pol_B = (r²/λ)R_tor`.
3. EAB2/ERK2 magnetic-poloidal caches: gate with a loud error if selected (their operator builders still use Δ_l; porting them follows Part B's pattern). CNAB2/default magnetic path is the Part-A target.
4. Gates: a magnetic-diffusion manufactured test (single P_B mode, pure decay vs exp(−t·eigenvalue) self-convergence order 2), the existing `magnetic_conducting_inner_core` test (NaN must be gone), full suite.

**Part B tasks (next session if budget ends):**
5. W-split poloidal velocity solve with influence-matrix BCs (machinery exists for ERK2 — generalize), R_pol(F) wiring, D_pol in the velocity-poloidal operators, EAB2/ERK2 velocity-poloidal gating/port.
6. Convective-onset test: from rest, supercritical Ra ⇒ kinetic energy grows; subcritical ⇒ decays. THE Stage-4 acceptance.
7. CNAB2 temporal self-convergence stays order 2 on the new dynamics.

**Verification discipline:** every operator/projection change is gated by a manufactured-solution or identity test BEFORE the dynamics run; the suite triage names each updated expectation.
