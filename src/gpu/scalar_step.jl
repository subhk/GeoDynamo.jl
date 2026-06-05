# =============================================================================
# GPU Phase 5f — one scalar field's full CNAB2 timestep, composing the verified
# pieces: nonlinear (5e) → RHS (5c) → implicit solve (5d) → field update +
# nl_prev rollover.  Mirrors apply_temperature_implicit_update! + the CNAB2
# history rollover (temperature/solver.jl:121-208).  Runs on Array (locally
# testable) and CuArray.  (Per-call scratch — Phase-6 may cache.)
# =============================================================================

"""
    gpu_scalar_field_step!(spec_r, spec_i, prev_nl_r, prev_nl_i, u_r, u_θ, u_φ, config,
                           d1, mvals, rinv, lin_batched, lu_batched,
                           bc_in_r, bc_in_i, bc_out_r, bc_out_i, inv_dt, linear_weight, lmax, bw) -> nothing

Advance one scalar field one CNAB2 step.  On entry `spec_*` is the field and
`prev_nl_*` the previous nonlinear term; on exit `spec_*` is the updated field and
`prev_nl_*` holds THIS step's nonlinear term (rolled over).  `lin_batched` are the
per-l linear operators `L`, `lu_batched` the per-l LU factors of the system matrix
`(I−θ·dt·L)`; `bc_*` the per-mode BC values; `inv_dt = mass_coeff/dt`,
`linear_weight = 1−θ`.  All arrays on the same backend.
"""
function gpu_scalar_field_step!(spec_r, spec_i, prev_nl_r, prev_nl_i, u_r, u_θ, u_φ, config,
        d1, mvals, rinv, lin_batched, lu_batched, bc_in_r, bc_in_i, bc_out_r, bc_out_i,
        inv_dt, linear_weight, lmax::Int, bw::Int)
    # 1. nonlinear term (5e). nl captured here from the OLD spec.
    nl_r = similar(spec_r); nl_i = similar(spec_i)        # Phase-6: move to a workspace struct
    gpu_scalar_nonlinear!(nl_r, nl_i, spec_r, spec_i, u_r, u_θ, u_φ, config, d1, mvals, rinv, lmax, bw)
    # 2. CNAB2 RHS from the OLD field, nl, prev_nl (5c). rhs is a separate scratch buffer —
    #    rhs ≠ spec is REQUIRED because build_rhs reads spec as input.
    rhs_r = similar(spec_r); rhs_i = similar(spec_i)      # Phase-6: move to a workspace struct
    gpu_build_rhs_cnab2!(rhs_r, rhs_i, spec_r, spec_i, nl_r, nl_i, prev_nl_r, prev_nl_i,
                         lin_batched, inv_dt, linear_weight, bw)
    # 3. implicit solve (BC rows + batched solve, in-place → solution in rhs) (5d)
    gpu_implicit_solve_field!(rhs_r, rhs_i, lu_batched, bc_in_r, bc_in_i, bc_out_r, bc_out_i, bw)
    # ⚠️ ORDERING INVARIANT: build_rhs (step 2) reads the OLD spec; the solve (step 3) writes
    #    the NEW field into rhs; spec is overwritten ONLY HERE, after both reads. Do not move
    #    this copy earlier. (The future workspace refactor could solve directly into spec and
    #    drop this copy — but only if build_rhs no longer needs spec intact.)
    # 4. update the field with the solution
    spec_r .= rhs_r
    spec_i .= rhs_i
    # 5. roll the history: prev_nl ← this step's nl (nl was captured at step 1 from the old
    #    spec, so this commutes with step 4).
    prev_nl_r .= nl_r
    prev_nl_i .= nl_i
    return nothing
end
