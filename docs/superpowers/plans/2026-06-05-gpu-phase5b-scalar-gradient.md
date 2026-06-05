# GPU Phase 5b — Scalar Gradient (∇θ) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute the spectral gradient `∇s` of a scalar field (temperature/composition) on a single GPU — radial `∂s/∂r`, latitudinal `∂s/∂θ`, longitudinal `∂s/∂φ` — matching the CPU exactly. These feed the GPU scalar-advection kernel (Phase 2 `gpu_scalar_advection!`).

**Architecture:** From the CPU map (`src/physics/nonlinear.jl:18-280`), the gradient of a dense `(lmax+1, mmax+1, nr)` scalar field has three components, all algebraic except the radial:
- **Radial** `∇r = d1·s` — banded mat-vec (REUSE Phase 5a `gpu_batched_banded_matvec!`); NOT scaled by `1/r`.
- **Longitudinal** `∇φ`: `∇φ_r = −m·s_i`, `∇φ_i = m·s_r` (`i·m·s`) — element-wise per m-slot.
- **Latitudinal** `∇θ`: a Legendre `(l±1, m)` recurrence — `∇θ = A₊·s[l+1,m] + A₋·s[l−1,m]` (neighbor-coupled along the l-axis), a KernelAbstractions `@kernel`.
- **Geometric**: `∇θ` and `∇φ` multiplied by `1/r` (tangential only); `∇r` unscaled.

The recurrence and the `i·m` are exact algebra → **locally testable** (CPU backend, exact `==`). The recurrence reads `l±1` neighbors at dense slots `[li±1, mi, r]`.

**Tech Stack:** Julia, KernelAbstractions (the θ recurrence kernel), broadcast (∇φ, the `1/r` factor). Reuses Phase 5a `gpu_batched_banded_matvec!`. No CUDA extension methods.

---

## Background (CPU reference — `src/physics/nonlinear.jl`)

- **θ recurrence** (`compute_theta_gradient_spectral!:84-114`), for mode `(l,m)` (`abs_m=m`, m≥0 in storage):
  - if `l < lmax`: `A₊ = l·sqrt((l+m+1)(l−m+1) / ((2l+1)(2l+3)))`, add `A₊·s[l+1,m]`.
  - if `l > m`: `A₋ = −(l+1)·sqrt((l+m)(l−m) / ((2l−1)(2l+1)))`, add `A₋·s[l−1,m]`.
  - ⚠️ empty slots (`l < m`, lower triangle) must be set to 0 — the `A₊` `sqrt((l−m+1)…)` goes negative there → NaN.
- **φ** (`compute_phi_gradient_spectral!:142-153`): `∇φ_r = −m·s_i`, `∇φ_i = m·s_r`.
- **radial** (`compute_radial_gradient_spectral!:195-205`): `∇r = d1·s` (banded mat-vec, `∂r` matrix).
- **geometric** (`apply_geometric_factors_spectral!:230-275`): `∇θ`,`∇φ` × `r⁻¹` (= `domain.r[r,3]`); at `r=0` set to 0 (ball center); `∇r` unscaled.

In the dense `(lmax+1, mmax+1, nr)` layout, mode `(l,m)` is at slot `[l+1, m+1, r]` → `li=l+1`, `mi=m+1`; neighbor `(l±1,m)` at `[li±1, mi, r]`.

## Testing without a local GPU

- **[LOCAL]** — the recurrence/φ run on Array; tests assert each `(l,m,r)` **equals** an independent reference of the exact CPU formula (exact `==`). The radial reuses the 5a mat-vec (already verified). Real verification.
- **[GPU-BOX]** — same on `CuArray`; guarded by `if !GeoDynamo.gpu_functional() … @test_skip`.

Julia: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=.` from worktree `/Users/subha/Documents/GitHub/GeoDynamo-gpu` (branch `feat/gpu-phase0`, on Phases 0–5a). **Never pipe test runs through `tail`.**

## File Structure

- **Create** `src/gpu/scalar_gradient.jl` — `gpu_phi_gradient!`, the θ recurrence kernel + `gpu_theta_gradient!`, and `gpu_scalar_gradient!` (the assembling driver).
- **Modify** `src/GeoDynamo.jl` — `include("gpu/scalar_gradient.jl")` (after `gpu/spectral_curl.jl`); export the three public functions.
- **Create** `test/gpu_phase5b_scalar_gradient.jl` — `[LOCAL]` + `[GPU-BOX]` tests.
- **Modify** `test/runtests.jl` — register.

Locked interfaces:

```julia
gpu_phi_gradient!(gφ_r, gφ_i, s_r, s_i, mvals)          # gφ_r=−m·s_i, gφ_i=m·s_r ; mvals length nm (m per slot)
gpu_theta_gradient!(gθ_r, gθ_i, s_r, s_i, lmax)         # Legendre (l±1,m) recurrence; empty slots (l<m) → 0
gpu_scalar_gradient!(gr_r,gr_i, gθ_r,gθ_i, gφ_r,gφ_i, s_r,s_i, d1, mvals, rinv, lmax, bw)
    # ∇r = d1·s (no 1/r) ; ∇θ = recurrence · rinv ; ∇φ = i·m·s · rinv
```

All field arrays `(nl,nm,nr)`; `d1` banded `(2bw+1,nr)`; `mvals` length-`nm` (`mvals[m+1]=m`); `rinv` length-`nr` (`1/r`, 0 at r=0); same backend. Outputs distinct from inputs.

---

## Task 1: `gpu_phi_gradient!`

**Files:** Create `src/gpu/scalar_gradient.jl`; Modify `src/GeoDynamo.jl`; Test `test/gpu_phase5b_scalar_gradient.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Create `test/gpu_phase5b_scalar_gradient.jl`:

```julia
using Test
using GeoDynamo
using Random

@testset "GPU Phase 5b — Scalar Gradient" begin
    @testset "phi gradient i·m·s [LOCAL]" begin
        nl, nm, nr = 5, 4, 3
        sr = rand(MersenneTwister(1), nl, nm, nr); si = rand(MersenneTwister(2), nl, nm, nr)
        mvals = Float64.(0:(nm - 1))                 # m per m-slot
        gφr = zeros(nl,nm,nr); gφi = zeros(nl,nm,nr)
        GeoDynamo.gpu_phi_gradient!(gφr, gφi, sr, si, mvals)
        for l in 1:nl, m in 1:nm, r in 1:nr
            @test gφr[l,m,r] == -mvals[m] * si[l,m,r]
            @test gφi[l,m,r] ==  mvals[m] * sr[l,m,r]
        end
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5b_scalar_gradient.jl")'`
Expected: FAIL — `gpu_phi_gradient!` undefined.

- [ ] **Step 3: Implement**

Create `src/gpu/scalar_gradient.jl`:

```julia
# =============================================================================
# GPU Phase 5b — scalar gradient ∇s (temperature/composition), matching the CPU
# (src/physics/nonlinear.jl:18-280).  Radial ∇r = d1·s (reuse Phase 5a mat-vec,
# no 1/r).  Longitudinal ∇φ = i·m·s (element-wise).  Latitudinal ∇θ = Legendre
# (l±1,m) recurrence (KA kernel).  Geometric 1/r on the tangential (∇θ,∇φ) only.
# Feeds the Phase-2 scalar advection.  Curls/derivatives reused from Phase 5a.
# =============================================================================

"""
    gpu_phi_gradient!(gφ_r, gφ_i, s_r, s_i, mvals) -> nothing

Longitudinal gradient `∂s/∂φ = i·m·s`: `gφ_r = −m·s_i`, `gφ_i = m·s_r`.
`mvals[m+1] = m` (length `nm`, m-slot index).
"""
function gpu_phi_gradient!(gφ_r, gφ_i, s_r, s_i, mvals)
    mm = reshape(mvals, 1, :, 1)            # (1, nm, 1) — m over the m-slot axis
    @. gφ_r = -mm * s_i
    @. gφ_i = mm * s_r
    return nothing
end
```

- [ ] **Step 4: Include + export**

In `src/GeoDynamo.jl`, after `include("gpu/spectral_curl.jl")` add `include("gpu/scalar_gradient.jl")`. Add export:
```julia
export gpu_phi_gradient!, gpu_theta_gradient!, gpu_scalar_gradient!
```

- [ ] **Step 5: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5b_scalar_gradient.jl")'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/scalar_gradient.jl src/GeoDynamo.jl test/gpu_phase5b_scalar_gradient.jl
git commit -m "feat(gpu): gpu_phi_gradient! (i·m·s longitudinal gradient) (Phase 5b)"
```

---

## Task 2: `gpu_theta_gradient!` (Legendre recurrence)

**Files:** Modify `src/gpu/scalar_gradient.jl`; Test `test/gpu_phase5b_scalar_gradient.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Add to `test/gpu_phase5b_scalar_gradient.jl`:

```julia
@testset "theta gradient (l±1,m) recurrence [LOCAL]" begin
    lmax, mmax, nr = 6, 6, 3
    nl, nm = lmax + 1, mmax + 1
    # band-limited source: only valid (l>=m) slots populated
    sr = zeros(nl,nm,nr); si = zeros(nl,nm,nr)
    rng = MersenneTwister(5)
    for mi in 1:nm, li in mi:nl, r in 1:nr     # l>=m
        sr[li,mi,r] = rand(rng); si[li,mi,r] = rand(rng)
    end
    gθr = fill(NaN, nl,nm,nr); gθi = fill(NaN, nl,nm,nr)
    GeoDynamo.gpu_theta_gradient!(gθr, gθi, sr, si, lmax)
    # independent reference: exact CPU recurrence
    for li in 1:nl, mi in 1:nm, r in 1:nr
        l = li - 1; m = mi - 1
        if l < m
            @test gθr[li,mi,r] == 0.0 && gθi[li,mi,r] == 0.0
            continue
        end
        dtr = 0.0; dti = 0.0
        if l < lmax
            ap = Float64(l) * sqrt(Float64((l+m+1)*(l-m+1)) / Float64((2l+1)*(2l+3)))
            dtr += ap * sr[li+1, mi, r]; dti += ap * si[li+1, mi, r]
        end
        if l > m
            am = -Float64(l+1) * sqrt(Float64((l+m)*(l-m)) / Float64((2l-1)*(2l+1)))
            dtr += am * sr[li-1, mi, r]; dti += am * si[li-1, mi, r]
        end
        @test gθr[li,mi,r] == dtr
        @test gθi[li,mi,r] == dti
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5b_scalar_gradient.jl")'`
Expected: FAIL — `gpu_theta_gradient!` undefined.

- [ ] **Step 3: Implement**

Append to `src/gpu/scalar_gradient.jl`:

```julia
# One workitem per (l-slot li, m-slot mi). l=li-1, m=mi-1 (m≥0). Reads l±1 neighbors
# at [li±1, mi, r]. Mirrors compute_theta_gradient_spectral!:84-114 exactly.
# Empty slots (l<m) → 0 (the A₊ sqrt would be NaN there; guard skips them).
@kernel function _theta_grad_kernel!(gθr, gθi, @Const(sr), @Const(si), lmax::Int, nr::Int)
    li, mi = @index(Global, NTuple)
    T = eltype(gθr)
    l = li - 1
    m = mi - 1
    @inbounds if l < m
        for r in 1:nr
            gθr[li, mi, r] = zero(T)
            gθi[li, mi, r] = zero(T)
        end
    else
        for r in 1:nr
            dtr = zero(T)
            dti = zero(T)
            if l < lmax
                ap = T(l) * sqrt(T((l + m + 1) * (l - m + 1)) / T((2l + 1) * (2l + 3)))
                dtr += ap * sr[li + 1, mi, r]
                dti += ap * si[li + 1, mi, r]
            end
            if l > m
                am = -T(l + 1) * sqrt(T((l + m) * (l - m)) / T((2l - 1) * (2l + 1)))
                dtr += am * sr[li - 1, mi, r]
                dti += am * si[li - 1, mi, r]
            end
            gθr[li, mi, r] = dtr
            gθi[li, mi, r] = dti
        end
    end
end

"""
    gpu_theta_gradient!(gθ_r, gθ_i, s_r, s_i, lmax) -> nothing

Latitudinal gradient `∂s/∂θ` via the Legendre `(l±1, m)` recurrence (matching the
CPU). Empty slots (`l < m`) are zeroed. Outputs must be distinct from inputs (the
recurrence reads neighbor l-slots). Backend inferred from `gθ_r`.
"""
function gpu_theta_gradient!(gθ_r, gθ_i, s_r, s_i, lmax::Int)
    nl, nm, nr = size(gθ_r)
    backend = KernelAbstractions.get_backend(gθ_r)
    _theta_grad_kernel!(backend)(gθ_r, gθ_i, s_r, s_i, lmax, nr; ndrange = (nl, nm))
    KernelAbstractions.synchronize(backend)
    return nothing
end
```

- [ ] **Step 4: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5b_scalar_gradient.jl")'`
Expected: PASS — recurrence matches the reference, empty slots zeroed (no NaN).

- [ ] **Step 5: Commit**

```bash
git add src/gpu/scalar_gradient.jl test/gpu_phase5b_scalar_gradient.jl
git commit -m "feat(gpu): gpu_theta_gradient! Legendre (l±1,m) recurrence kernel (Phase 5b)"
```

---

## Task 3: `gpu_scalar_gradient!` (assemble ∇r/∇θ/∇φ + 1/r)

**Files:** Modify `src/gpu/scalar_gradient.jl`; Test `test/gpu_phase5b_scalar_gradient.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Add to `test/gpu_phase5b_scalar_gradient.jl`:

```julia
@testset "scalar gradient assembly (∇r/∇θ/∇φ + 1/r) [LOCAL]" begin
    lmax, mmax, nr, bw = 5, 5, 4, 2
    nl, nm = lmax + 1, mmax + 1
    # reuse a banded matrix builder from Phase 5a's test idea
    function band(::Type{TT}, N, bw; seed) where {TT}
        rng = MersenneTwister(seed); d = zeros(TT, 2bw+1, N)
        for j in 1:N, i in max(1,j-bw):min(N,j+bw); d[bw+1+i-j,j] = rand(rng,TT)-TT(0.5); end
        GeoDynamo.BandedMatrix{TT}(d, bw, N)
    end
    d1 = band(Float64, nr, bw; seed = 31)
    sr = zeros(nl,nm,nr); si = zeros(nl,nm,nr); rng = MersenneTwister(33)
    for mi in 1:nm, li in mi:nl, r in 1:nr; sr[li,mi,r]=rand(rng); si[li,mi,r]=rand(rng); end
    mvals = Float64.(0:(nm-1)); rinv = [1.0/(0.5+0.1k) for k in 1:nr]
    grr=zeros(nl,nm,nr); gri=zeros(nl,nm,nr); gtr=zeros(nl,nm,nr); gti=zeros(nl,nm,nr); gpr=zeros(nl,nm,nr); gpi=zeros(nl,nm,nr)
    GeoDynamo.gpu_scalar_gradient!(grr,gri, gtr,gti, gpr,gpi, sr,si, d1.data, mvals, rinv, lmax, bw)
    # ∇r reference: d1·s per (l,m), NO 1/r
    for li in 1:nl, mi in 1:nm
        ref = zeros(nr); GeoDynamo.apply_radial_derivative!(ref, d1, collect(sr[li,mi,:]))
        @test grr[li,mi,:] == ref
    end
    # ∇φ reference: i·m·s, then ×1/r
    for li in 1:nl, mi in 1:nm, r in 1:nr
        @test gpr[li,mi,r] == (-(mvals[mi]) * si[li,mi,r]) * rinv[r]
    end
    # ∇θ reference: recurrence × 1/r (spot-check a valid mode)
    li, mi, r = 4, 2, 2; l = li-1; m = mi-1
    dtr = 0.0
    if l < lmax
        ap = Float64(l)*sqrt(Float64((l+m+1)*(l-m+1))/Float64((2l+1)*(2l+3))); dtr += ap*sr[li+1,mi,r]
    end
    if l > m
        am = -Float64(l+1)*sqrt(Float64((l+m)*(l-m))/Float64((2l-1)*(2l+1))); dtr += am*sr[li-1,mi,r]
    end
    @test gtr[li,mi,r] == dtr * rinv[r]
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5b_scalar_gradient.jl")'`
Expected: FAIL — `gpu_scalar_gradient!` undefined.

- [ ] **Step 3: Implement**

Append to `src/gpu/scalar_gradient.jl`:

```julia
"""
    gpu_scalar_gradient!(gr_r,gr_i, gθ_r,gθ_i, gφ_r,gφ_i, s_r,s_i,
                         d1, mvals, rinv, lmax, bw) -> nothing

Assemble the scalar gradient: `∇r = d1·s` (banded mat-vec, NOT scaled by 1/r),
`∇θ` via the Legendre recurrence, `∇φ = i·m·s`, then multiply the tangential
components (`∇θ`,`∇φ`) by `rinv = 1/r` (0 at r=0).  `d1` banded `(2bw+1,nr)`;
`mvals` length-`nm`; `rinv` length-`nr`; all on the same backend; outputs distinct
from `s_r`/`s_i`.
"""
function gpu_scalar_gradient!(gr_r, gr_i, gθ_r, gθ_i, gφ_r, gφ_i, s_r, s_i,
        d1, mvals, rinv, lmax::Int, bw::Int)
    gpu_batched_banded_matvec!(gr_r, s_r, d1, bw)     # ∇r real (no 1/r)
    gpu_batched_banded_matvec!(gr_i, s_i, d1, bw)     # ∇r imag
    gpu_theta_gradient!(gθ_r, gθ_i, s_r, s_i, lmax)   # ∇θ (pre-1/r)
    gpu_phi_gradient!(gφ_r, gφ_i, s_r, s_i, mvals)    # ∇φ (pre-1/r)
    ri = reshape(rinv, 1, 1, :)                       # geometric 1/r on tangential only
    @. gθ_r *= ri
    @. gθ_i *= ri
    @. gφ_r *= ri
    @. gφ_i *= ri
    return nothing
end
```

- [ ] **Step 4: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5b_scalar_gradient.jl")'`
Expected: PASS — ∇r matches mat-vec, ∇φ/∇θ match formula × 1/r.

- [ ] **Step 5: Commit**

```bash
git add src/gpu/scalar_gradient.jl test/gpu_phase5b_scalar_gradient.jl
git commit -m "feat(gpu): gpu_scalar_gradient! (assemble ∇r/∇θ/∇φ + 1/r) (Phase 5b)"
```

---

## Task 4: GPU-box gate + register + regression

**Files:** Test `test/gpu_phase5b_scalar_gradient.jl`, `test/runtests.jl`

- [ ] **Step 1: Add the GPU-box gate** `[GPU-BOX]`

Add to `test/gpu_phase5b_scalar_gradient.jl`:

```julia
@testset "GPU execution + GPU≈CPU parity (Phase-5b gate) [GPU-BOX]" begin
    if !GeoDynamo.gpu_functional()
        @test_skip "requires a functional CUDA GPU"
    else
        lmax, mmax, nr, bw = 6, 6, 4, 2
        nl, nm = lmax + 1, mmax + 1
        function band(::Type{TT}, N, bw; seed) where {TT}
            rng = MersenneTwister(seed); dd = zeros(TT, 2bw+1, N)
            for j in 1:N, i in max(1,j-bw):min(N,j+bw); dd[bw+1+i-j,j]=rand(rng,TT)-TT(0.5); end
            GeoDynamo.BandedMatrix{TT}(dd, bw, N)
        end
        d1 = band(Float64, nr, bw; seed = 41)
        sr = zeros(nl,nm,nr); si = zeros(nl,nm,nr); rng = MersenneTwister(43)
        for mi in 1:nm, li in mi:nl, r in 1:nr; sr[li,mi,r]=rand(rng); si[li,mi,r]=rand(rng); end
        mvals = Float64.(0:(nm-1)); rinv = [1.0/(0.5+0.1k) for k in 1:nr]
        z() = zeros(Float64, nl,nm,nr)
        c = (z(),z(),z(),z(),z(),z())
        GeoDynamo.gpu_scalar_gradient!(c..., sr,si, d1.data, mvals, rinv, lmax, bw)
        d(x) = GeoDynamo.on_architecture(GPU(), x)
        g = (d(z()),d(z()),d(z()),d(z()),d(z()),d(z()))
        GeoDynamo.gpu_scalar_gradient!(g..., d(sr),d(si), d(d1.data), d(mvals), d(rinv), lmax, bw)
        @test g[1] isa CUDA.CuArray
        for k in 1:6
            @test isapprox(Array(g[k]), c[k]; atol = 1e-12, rtol = 1e-10)
        end
    end
end
```

- [ ] **Step 2: Run locally** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5b_scalar_gradient.jl")'`
Expected: `[LOCAL]` testsets pass; the gate skips.

- [ ] **Step 3: Register**

In `test/runtests.jl`, add `"gpu_phase5b_scalar_gradient.jl"` (next to the Phase 5a entry).

- [ ] **Step 4: CPU regression** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5a_spectral_curl.jl")'` then separately `… -e 'using Test, GeoDynamo, MPI; include("test/allocation_runtime_checks.jl")'`
Expected: Phase 5a green; allocation guards 39/39.

- [ ] **Step 5: Commit**

```bash
git add test/gpu_phase5b_scalar_gradient.jl test/runtests.jl
git commit -m "test(gpu): Phase-5b GPU-box gate + register scalar gradient"
```

---

## GPU-box validation handoff

On the GPU box:
```julia
using CUDA, Test, GeoDynamo, Random
@assert GeoDynamo.gpu_functional()
include("test/gpu_phase5b_scalar_gradient.jl")    # the [GPU-BOX] gate must PASS
```
**Phase 5b passes when:** all three gradient components on `CuArray` match the CPU result to ~1e-12. Report any failure (a recurrence neighbor-index surprise, a NaN at empty slots, or a `1/r` placement issue) before Phase 5c.

---

## What this unblocks / what's next

This completes the **scalar gradient** → with Phase 5a's curls + Phase 2's nonlinear kernels, all the nonlinear-term inputs (∇s, ω, J) now exist on GPU. Remaining:
- **Phase 5c — RHS assembly (CNAB2 `build_rhs_cnab2!`) + the full GPU `solver_step!`** (transform → curl/gradient → nonlinear products → analyze → RHS → batched solve → update → BCs) + the GPU≈CPU full-step gate.
- **Phase 6 — `run!`/`Simulation` device-resident loop + IO host-gather.**

---

## Self-Review

**Spec coverage:** the design-doc nonlinear path needs `∇s` (deferred from Phase 2). This delivers it: `gpu_phi_gradient!` (i·m·s, Task 1), `gpu_theta_gradient!` (Legendre recurrence, Task 2), `gpu_scalar_gradient!` (assembly + 1/r reusing 5a's mat-vec for ∇r, Task 3), GPU gate + regression (Task 4). The `r=0` (ball center) zeroing is handled by the caller passing `rinv=0` at r=0 (a data concern, like the curl's `rinv`); Phase 5c wires the real domain. Covered.

**Placeholder scan:** none — every code step has complete code; every run step has the exact command + expected result. The `band` test helper is fully defined inline.

**Type consistency:** `gpu_phi_gradient!(gφ_r,gφ_i, s_r,s_i, mvals)`, `gpu_theta_gradient!(gθ_r,gθ_i, s_r,s_i, lmax)`, `gpu_scalar_gradient!(gr_r,gr_i, gθ_r,gθ_i, gφ_r,gφ_i, s_r,s_i, d1, mvals, rinv, lmax, bw)` — consistent across tasks and the interface block. The recurrence A₊/A₋ match `compute_theta_gradient_spectral!:96-107` exactly (incl. the `l<lmax`/`l>m` bounds and the `l<m`→0 guard). φ matches `:142-153`. Radial reuses `gpu_batched_banded_matvec!` (Phase 5a, == `apply_radial_derivative!`). 1/r on tangential only matches `apply_geometric_factors_spectral!`. `mvals`→`(1,nm,1)`, `rinv`→`(1,1,nr)` reshape axes; dense slot `(l,m)→[l+1,m+1]`, neighbors `[li±1,mi]`.
