# GPU Phase 2 — Nonlinear Product Kernels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute the element-wise physical-space nonlinear products of the geodynamo equations (scalar advection, Coriolis, thermal/compositional buoyancy, and the generic vector cross-product used by Lorentz `J×B`, velocity advection `u×ω`, and induction `u×B`) on a single GPU, matching the CPU formulas exactly.

**Architecture:** Each term, given its physical-space input component arrays, is a pure element-wise operation (each output point depends only on same-index inputs). They are implemented as **broadcast** kernels over plain component arrays (`(nlat, nlon, nr)`), which CUDA.jl auto-compiles to GPU kernels on `CuArray`s and which run on `Array`s on the CPU — so the kernel *logic* is verified locally against the literal CPU expression, and only `CuArray` execution is GPU-gated. The curl/derivative operations that *produce* vorticity (`ω`) and current (`J`) are spectral stencils, NOT part of this phase; these kernels take `ω`/`J` (and gradients) as given inputs. Bespoke hand-written CUDA kernels are reserved for Phase 4 (batched banded solve); for these element-wise products, broadcast is both optimal and portable.

**Tech Stack:** Julia broadcast (`@.`), CUDA.jl (weakdep; broadcast on `CuArray` runs on GPU automatically — no extension methods needed for the kernels themselves).

---

## Scope

**In scope (pure element-wise — verified by the Phase-2 source map):**
1. **Scalar advection** `(u·∇)s`: `out = -(u_r·∇r + u_θ·∇θ + u_φ·∇φ)` — CPU ref `src/physics/nonlinear.jl:848-851`.
2. **Generic cross-product** `out_i = coeff·(a×b)_i` — reused for Lorentz `(1/Pm)(J×B)` (`src/solver/numerics.jl:1336-1339`), velocity advection `E·(u×ω)` (`numerics.jl:1218-1220`), induction `(u×B)` with `coeff=1` (`numerics.jl:1565-1584`).
3. **Coriolis** `2Ω×u`: `c_r=-sinθ·u_φ`, `c_θ=-cosθ·u_φ`, `c_φ=cosθ·u_θ+sinθ·u_r` (`numerics.jl:1222-1228`; `sinθ`/`cosθ` are per-latitude vectors).
4. **Buoyancy** (thermal + compositional): `force_r += factor·r·s` where `r` is per-radial-level and `factor=(Pm/Pr)·Ra` (thermal, `numerics.jl:1274-1286`) or `(Pm/Sc)·Ra_C` (compositional, `numerics.jl:1292-1318`).

**Out of scope (need spectral curl/derivatives — later phases):** computing `ω`=∇×u, `J`=∇×B, gradients `∇s`, and curl(u×B). These produce the *inputs* to the kernels above; this phase consumes them.

## Testing without a local GPU

- **[LOCAL]** — broadcast kernels run on `Array`; tests assert the kernel output **equals the literal CPU expression** on random inputs (exact `==`: same per-element arithmetic). Real verification, not skipped.
- **[GPU-BOX]** — same on `CuArray`; asserts GPU result ≈ CPU result (`atol`/`rtol`, reduction-free element-wise so very tight). Guarded by `if !GeoDynamo.gpu_functional() … @test_skip`.

Julia: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=.` from worktree `/Users/subha/Documents/GitHub/GeoDynamo-gpu` (branch `feat/gpu-phase0`, building on Phases 0+1). **Never pipe test runs through `tail`.**

## File Structure

- **Create** `src/gpu/nonlinear.jl` — the four kernel families (operate on plain arrays; backend-agnostic broadcast).
- **Modify** `src/GeoDynamo.jl` — `include("gpu/nonlinear.jl")` (after `include("gpu/scalar_transform.jl")`); export the kernels.
- **Create** `test/gpu_phase2_nonlinear.jl` — `[LOCAL]` (Array) + `[GPU-BOX]` (CuArray) tests.
- **Modify** `test/runtests.jl` — register the new test file.

Locked interfaces (operate on `AbstractArray{T,3}` `(nlat,nlon,nr)` component arrays; backend-agnostic):

```julia
gpu_scalar_advection!(out, u_r, u_θ, u_φ, ∇r, ∇θ, ∇φ)                        # out = -(u_r·∇r + u_θ·∇θ + u_φ·∇φ)
gpu_cross!(out_r, out_θ, out_φ, a_r, a_θ, a_φ, b_r, b_θ, b_φ, coeff)         # out_i = coeff·(a×b)_i  (overwrites out)
gpu_cross_add!(out_r, out_θ, out_φ, a_r, a_θ, a_φ, b_r, b_θ, b_φ, coeff)     # out_i += coeff·(a×b)_i (accumulates)
gpu_coriolis_sub!(out_r, out_θ, out_φ, u_r, u_θ, u_φ, sinθ, cosθ)           # out_i -= (ẑ×u)_i  (sinθ,cosθ: length-nlat vectors)
gpu_buoyancy_add!(force_r, s, r_vec, factor)                                 # force_r[:,:,k] += factor·r_vec[k]·s[:,:,k]
```

`sinθ`/`cosθ` are length-`nlat` vectors (latitude index = dim 1); `r_vec` is length-`nr` (radial index = dim 3).

---

## Task 1: scalar advection kernel

**Files:** Create `src/gpu/nonlinear.jl`; Modify `src/GeoDynamo.jl`; Test `test/gpu_phase2_nonlinear.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Create `test/gpu_phase2_nonlinear.jl`:

```julia
using Test
using GeoDynamo

@testset "GPU Phase 2 — Nonlinear Kernels" begin
    nlat, nlon, nr = 6, 8, 3
    rnd() = rand(Float64, nlat, nlon, nr)

    @testset "scalar advection [LOCAL]" begin
        u_r, u_θ, u_φ = rnd(), rnd(), rnd()
        gr, gθ, gφ = rnd(), rnd(), rnd()
        out = zeros(Float64, nlat, nlon, nr)
        GeoDynamo.gpu_scalar_advection!(out, u_r, u_θ, u_φ, gr, gθ, gφ)
        ref = similar(out)
        @inbounds for i in eachindex(ref)
            ref[i] = -(u_r[i] * gr[i] + u_θ[i] * gθ[i] + u_φ[i] * gφ[i])
        end
        @test out == ref               # exact: same per-element arithmetic
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase2_nonlinear.jl")'`
Expected: FAIL — `gpu_scalar_advection!` undefined.

- [ ] **Step 3: Implement**

Create `src/gpu/nonlinear.jl`:

```julia
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
```

- [ ] **Step 4: Include + export**

In `src/GeoDynamo.jl`, after `include("gpu/scalar_transform.jl")` add `include("gpu/nonlinear.jl")`. Add a GPU export line:
```julia
export gpu_scalar_advection!, gpu_cross!, gpu_cross_add!, gpu_coriolis_sub!, gpu_buoyancy_add!
```
(Exporting names defined in later tasks now is legal in Julia; all resolve once Tasks 2–4 land.)

- [ ] **Step 5: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase2_nonlinear.jl")'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/nonlinear.jl src/GeoDynamo.jl test/gpu_phase2_nonlinear.jl
git commit -m "feat(gpu): scalar advection nonlinear kernel (Phase 2)"
```

---

## Task 2: cross-product kernels (Lorentz / u×ω / u×B)

**Files:** Modify `src/gpu/nonlinear.jl`; Test `test/gpu_phase2_nonlinear.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Add to `test/gpu_phase2_nonlinear.jl`:

```julia
@testset "cross product (overwrite + accumulate) [LOCAL]" begin
    a_r, a_θ, a_φ = rnd(), rnd(), rnd()
    b_r, b_θ, b_φ = rnd(), rnd(), rnd()
    coeff = 0.37
    or, oθ, oφ = zeros(nlat,nlon,nr), zeros(nlat,nlon,nr), zeros(nlat,nlon,nr)
    GeoDynamo.gpu_cross!(or, oθ, oφ, a_r, a_θ, a_φ, b_r, b_θ, b_φ, coeff)
    # reference: coeff·(a×b)  (cross order matches CPU Lorentz/u×ω/u×B)
    rr = similar(or); rθ = similar(oθ); rφ = similar(oφ)
    @inbounds for i in eachindex(rr)
        rr[i] = coeff * (a_θ[i]*b_φ[i] - a_φ[i]*b_θ[i])
        rθ[i] = coeff * (a_φ[i]*b_r[i] - a_r[i]*b_φ[i])
        rφ[i] = coeff * (a_r[i]*b_θ[i] - a_θ[i]*b_r[i])
    end
    @test or == rr && oθ == rθ && oφ == rφ

    # accumulate variant adds onto existing contents
    base_r, base_θ, base_φ = rnd(), rnd(), rnd()
    ar, aθ, aφ = copy(base_r), copy(base_θ), copy(base_φ)
    GeoDynamo.gpu_cross_add!(ar, aθ, aφ, a_r, a_θ, a_φ, b_r, b_θ, b_φ, coeff)
    @test ar == base_r .+ rr && aθ == base_θ .+ rθ && aφ == base_φ .+ rφ
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase2_nonlinear.jl")'`
Expected: FAIL — `gpu_cross!` undefined.

- [ ] **Step 3: Implement**

Append to `src/gpu/nonlinear.jl`:

```julia
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
```

- [ ] **Step 4: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase2_nonlinear.jl")'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gpu/nonlinear.jl test/gpu_phase2_nonlinear.jl
git commit -m "feat(gpu): cross-product nonlinear kernels (Lorentz/u×ω/u×B) (Phase 2)"
```

---

## Task 3: Coriolis kernel

**Files:** Modify `src/gpu/nonlinear.jl`; Test `test/gpu_phase2_nonlinear.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Add to `test/gpu_phase2_nonlinear.jl`:

```julia
@testset "Coriolis subtract [LOCAL]" begin
    u_r, u_θ, u_φ = rnd(), rnd(), rnd()
    sinθ = rand(Float64, nlat); cosθ = rand(Float64, nlat)
    or, oθ, oφ = rnd(), rnd(), rnd()
    base_r, base_θ, base_φ = copy(or), copy(oθ), copy(oφ)
    GeoDynamo.gpu_coriolis_sub!(or, oθ, oφ, u_r, u_θ, u_φ, sinθ, cosθ)
    rr = similar(or); rθ = similar(oθ); rφ = similar(oφ)
    @inbounds for k in 1:nr, j in 1:nlon, i in 1:nlat
        cr = -sinθ[i] * u_φ[i,j,k]
        cθ = -cosθ[i] * u_φ[i,j,k]
        cφ =  cosθ[i] * u_θ[i,j,k] + sinθ[i] * u_r[i,j,k]
        rr[i,j,k] = base_r[i,j,k] - cr
        rθ[i,j,k] = base_θ[i,j,k] - cθ
        rφ[i,j,k] = base_φ[i,j,k] - cφ
    end
    @test or == rr && oθ == rθ && oφ == rφ
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase2_nonlinear.jl")'`
Expected: FAIL — `gpu_coriolis_sub!` undefined.

- [ ] **Step 3: Implement**

Append to `src/gpu/nonlinear.jl`. `sinθ`/`cosθ` are length-`nlat`; reshape to `(nlat,1,1)` so broadcast applies them along the latitude axis:

```julia
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
```

- [ ] **Step 4: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase2_nonlinear.jl")'`
Expected: PASS. (On GPU, `reshape` of a `CuVector` is a `CuArray` → broadcast stays on device.)

- [ ] **Step 5: Commit**

```bash
git add src/gpu/nonlinear.jl test/gpu_phase2_nonlinear.jl
git commit -m "feat(gpu): Coriolis nonlinear kernel (Phase 2)"
```

---

## Task 4: buoyancy kernel

**Files:** Modify `src/gpu/nonlinear.jl`; Test `test/gpu_phase2_nonlinear.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Add to `test/gpu_phase2_nonlinear.jl`:

```julia
@testset "buoyancy add [LOCAL]" begin
    s = rnd()
    r_vec = collect(range(0.5, 1.0; length = nr))
    factor = 1.7
    force_r = rnd(); base = copy(force_r)
    GeoDynamo.gpu_buoyancy_add!(force_r, s, r_vec, factor)
    ref = similar(force_r)
    @inbounds for k in 1:nr, j in 1:nlon, i in 1:nlat
        ref[i,j,k] = base[i,j,k] + factor * r_vec[k] * s[i,j,k]
    end
    @test force_r == ref
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase2_nonlinear.jl")'`
Expected: FAIL — `gpu_buoyancy_add!` undefined.

- [ ] **Step 3: Implement**

Append to `src/gpu/nonlinear.jl`. `r_vec` is length-`nr`; reshape to `(1,1,nr)` for the radial axis:

```julia
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
```

- [ ] **Step 4: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase2_nonlinear.jl")'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gpu/nonlinear.jl test/gpu_phase2_nonlinear.jl
git commit -m "feat(gpu): buoyancy nonlinear kernel (Phase 2)"
```

---

## Task 5: GPU-box gate + register + regression

**Files:** Modify `test/gpu_phase2_nonlinear.jl`, `test/runtests.jl`

- [ ] **Step 1: Add the GPU-box gate** `[GPU-BOX]`

Add to `test/gpu_phase2_nonlinear.jl`:

```julia
@testset "GPU execution + GPU≈CPU parity (Phase-2 gate) [GPU-BOX]" begin
    if !GeoDynamo.gpu_functional()
        @test_skip "requires a functional CUDA GPU"
    else
        # scalar advection: CPU(Array) reference vs GPU(CuArray)
        u_r, u_θ, u_φ = rnd(), rnd(), rnd(); gr, gθ, gφ = rnd(), rnd(), rnd()
        cout = zeros(Float64, nlat, nlon, nr)
        GeoDynamo.gpu_scalar_advection!(cout, u_r, u_θ, u_φ, gr, gθ, gφ)        # CPU
        d(x) = GeoDynamo.on_architecture(GPU(), x)
        gout = d(zeros(Float64, nlat, nlon, nr))
        GeoDynamo.gpu_scalar_advection!(gout, d(u_r), d(u_θ), d(u_φ), d(gr), d(gθ), d(gφ))
        @test gout isa CUDA.CuArray
        @test isapprox(Array(gout), cout; atol = 1e-13, rtol = 1e-12)

        # cross product
        a = (rnd(), rnd(), rnd()); b = (rnd(), rnd(), rnd()); coeff = 0.42
        cr, cθ, cφ = zeros(nlat,nlon,nr), zeros(nlat,nlon,nr), zeros(nlat,nlon,nr)
        GeoDynamo.gpu_cross!(cr, cθ, cφ, a..., b..., coeff)
        gr3 = (d(zeros(nlat,nlon,nr)), d(zeros(nlat,nlon,nr)), d(zeros(nlat,nlon,nr)))
        GeoDynamo.gpu_cross!(gr3..., d.(a)..., d.(b)..., coeff)
        @test isapprox(Array(gr3[1]), cr; atol = 1e-13)
        @test isapprox(Array(gr3[3]), cφ; atol = 1e-13)
    end
end
```

- [ ] **Step 2: Run locally** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo; include("test/gpu_phase2_nonlinear.jl")'`
Expected: `[LOCAL]` testsets pass; the gate testset **skips** (no GPU). Mark **"implemented; awaiting GPU-box parity."**

- [ ] **Step 3: Register**

In `test/runtests.jl`, add `"gpu_phase2_nonlinear.jl"` to the test-file list (next to the Phase 1 entry).

- [ ] **Step 4: CPU regression** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI; include("test/gpu_phase1_scalar_transform.jl"); include("test/allocation_runtime_checks.jl")'`
Expected: Phase 1 green; allocation guards 39/39 (Phase 2 adds only new files + include/export).

- [ ] **Step 5: Commit**

```bash
git add test/gpu_phase2_nonlinear.jl test/runtests.jl
git commit -m "test(gpu): Phase-2 GPU-box gate + register nonlinear kernels"
```

---

## GPU-box validation handoff

On the GPU box (CUDA loaded):
```julia
using CUDA, Test, GeoDynamo
@assert GeoDynamo.gpu_functional()
include("test/gpu_phase2_nonlinear.jl")    # the [GPU-BOX] gate must PASS
```
**Phase 2 passes when:** each kernel runs on `CuArray` and matches the CPU(Array) result to ~1e-13 (element-wise, reduction-free → very tight). Report any failure (a broadcast-on-CuArray surprise, a `reshape`-of-CuVector issue, or a coefficient/sign mismatch) before Phase 3.

---

## Self-Review

**Spec coverage (design-doc Phase 2: "nonlinear kernels: advection, Coriolis, Lorentz, buoyancy"):** scalar advection (Task 1), Lorentz/u×ω/u×B via the generic cross-product (Task 2), Coriolis (Task 3), thermal+compositional buoyancy (Task 4), GPU gate + regression (Task 5). The momentum-advection `u×ω` and Lorentz `J×B` *inputs* (`ω`, `J`) come from spectral curls in a later phase — explicitly out of scope here (this phase assembles the products). Covered.

**Placeholder scan:** none — every code step has complete code; every run step has the exact command + expected result.

**Type consistency:** `gpu_scalar_advection!(out, u_r,u_θ,u_φ, ∇r,∇θ,∇φ)`, `gpu_cross!`/`gpu_cross_add!(out_r,out_θ,out_φ, a_r,a_θ,a_φ, b_r,b_θ,b_φ, coeff)`, `gpu_coriolis_sub!(out_r,out_θ,out_φ, u_r,u_θ,u_φ, sinθ,cosθ)`, `gpu_buoyancy_add!(force_r, s, r_vec, factor)` — names + arg orders consistent across tasks and the interface block. All kernels are backend-agnostic broadcast over `(nlat,nlon,nr)` arrays; `sinθ`/`cosθ` length-`nlat`, `r_vec` length-`nr`. Formulas copied from the CPU references cited in Scope.
