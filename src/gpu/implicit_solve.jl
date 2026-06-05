# =============================================================================
# GPU Phase 5d — one field's CNAB2 implicit solve: set the RHS boundary rows to
# the prescribed BC values (the system matrix has the BC equations embedded at
# rows 1 and nr), then batched-solve the per-mode banded systems (Phase 4).
# Mirrors _solver_solve_scalar_implicit_step! (imex.jl:343-366). Composes Phase 5c
# (RHS) + Phase 4 (solve). Broadcast + reused solve → runs on Array (locally
# testable) and CuArray.
# =============================================================================

"""
    gpu_apply_bc_rows!(x_r, x_i, bc_in_r, bc_in_i, bc_out_r, bc_out_i) -> nothing

Overwrite the boundary rows of the per-mode radial RHS with the prescribed BC
values: row 1 (inner) ← `bc_in_*`, row `nr` (outer) ← `bc_out_*`.  `x_*` are
`(nl,nm,nr)`; `bc_*` are `(nl,nm)` (per-`(l,m)` boundary value).
"""
function gpu_apply_bc_rows!(x_r, x_i, bc_in_r, bc_in_i, bc_out_r, bc_out_i)
    nr = size(x_r, 3)
    @views x_r[:, :, 1]  .= bc_in_r
    @views x_i[:, :, 1]  .= bc_in_i
    @views x_r[:, :, nr] .= bc_out_r
    @views x_i[:, :, nr] .= bc_out_i
    return nothing
end

"""
    gpu_implicit_solve_field!(x_r, x_i, lu_batched, bc_in_r, bc_in_i, bc_out_r, bc_out_i, bw) -> nothing

One field's CNAB2 implicit solve: `x` holds the RHS on entry; this sets the BC
boundary rows (`gpu_apply_bc_rows!`) then batched-solves the per-mode banded
systems in place (Phase 4 `gpu_batched_banded_solve!`, `X===B` supported), leaving
the solution in `x`.  `lu_batched` `(2bw+1,nr,nl)` are the per-l LU factors.
"""
function gpu_implicit_solve_field!(x_r, x_i, lu_batched, bc_in_r, bc_in_i, bc_out_r, bc_out_i, bw::Int)
    gpu_apply_bc_rows!(x_r, x_i, bc_in_r, bc_in_i, bc_out_r, bc_out_i)
    gpu_batched_banded_solve!(x_r, x_r, lu_batched, bw)   # in-place: solution overwrites RHS
    gpu_batched_banded_solve!(x_i, x_i, lu_batched, bw)
    return nothing
end
