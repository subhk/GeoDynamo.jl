# =============================================================================
# GPU Phase 2 — element-wise physical-space nonlinear product kernels.
# Each operates on plain (nlat, nlon, nr) component arrays via broadcast, so the
# same code runs on Array (CPU) and CuArray (GPU, auto-compiled by CUDA.jl).
# Inputs (gradients, vorticity ω, current J) are produced by spectral transforms/
# curls in other phases; these kernels just assemble the products.
# Formulas mirror the CPU implementation exactly (see the Phase-2 plan for refs).
# =============================================================================

"""
    gpu_scalar_advection!(out, u_r, u_θ, u_φ, ∇r, ∇θ, ∇φ) -> out

`out = -(u_r·∇r + u_θ·∇θ + u_φ·∇φ)` — the scalar advection `-(u·∇)s`.
"""
function gpu_scalar_advection!(out, u_r, u_θ, u_φ, ∇r, ∇θ, ∇φ)
    @. out = -(u_r * ∇r + u_θ * ∇θ + u_φ * ∇φ)
    return out
end

"""
    gpu_cross!(out_r, out_θ, out_φ, a_r, a_θ, a_φ, b_r, b_θ, b_φ, coeff) -> nothing

Overwrite `out = coeff·(a×b)` component-wise.  Cross-product order matches the CPU
Lorentz (`coeff=1/Pm`), velocity advection (`coeff=E`), and induction (`coeff=1`).
"""
function gpu_cross!(out_r, out_θ, out_φ, a_r, a_θ, a_φ, b_r, b_θ, b_φ, coeff)
    @. out_r = coeff * (a_θ * b_φ - a_φ * b_θ)
    @. out_θ = coeff * (a_φ * b_r - a_r * b_φ)
    @. out_φ = coeff * (a_r * b_θ - a_θ * b_r)
    return nothing
end

"""
    gpu_cross_add!(out_r, out_θ, out_φ, a_r, a_θ, a_φ, b_r, b_θ, b_φ, coeff) -> nothing

Accumulate `out += coeff·(a×b)` component-wise (e.g. add the Lorentz force onto an
advection accumulator).
"""
function gpu_cross_add!(out_r, out_θ, out_φ, a_r, a_θ, a_φ, b_r, b_θ, b_φ, coeff)
    @. out_r += coeff * (a_θ * b_φ - a_φ * b_θ)
    @. out_θ += coeff * (a_φ * b_r - a_r * b_φ)
    @. out_φ += coeff * (a_r * b_θ - a_θ * b_r)
    return nothing
end

"""
    gpu_coriolis_sub!(out_r, out_θ, out_φ, u_r, u_θ, u_φ, sinθ, cosθ) -> nothing

Subtract the Coriolis term `ẑ×u` from the accumulator (CPU: `adv_i -= (ẑ×u)_i`):
`(ẑ×u)_r=-sinθ·u_φ`, `(ẑ×u)_θ=-cosθ·u_φ`, `(ẑ×u)_φ=cosθ·u_θ+sinθ·u_r`.
`sinθ`,`cosθ` are length-`nlat` (latitude = dim 1).  The `2Ω` factor is absorbed
into the nondimensional coefficients upstream (Ekman number), matching the CPU.
"""
function gpu_coriolis_sub!(out_r, out_θ, out_φ, u_r, u_θ, u_φ, sinθ, cosθ)
    s = reshape(sinθ, :, 1, 1)
    c = reshape(cosθ, :, 1, 1)
    @. out_r -= (-s * u_φ)
    @. out_θ -= (-c * u_φ)
    @. out_φ -= (c * u_θ + s * u_r)
    return nothing
end

"""
    gpu_buoyancy_add!(force_r, s, r_vec, factor) -> nothing

Add the radial buoyancy/codensity force `force_r += factor·r·s`, with `r` per
radial level (`r_vec` length-`nr`, radial = dim 3).  Use `factor=(Pm/Pr)·Ra` for
thermal buoyancy or `factor=(Pm/Sc)·Ra_C` for compositional (matching the CPU).
"""
function gpu_buoyancy_add!(force_r, s, r_vec, factor)
    rr = reshape(r_vec, 1, 1, :)
    @. force_r += factor * rr * s
    return nothing
end
