# GPU Phase 5e — Scalar Nonlinear (explicit half) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute a scalar field's (temperature/composition) **nonlinear term** on a single GPU — `nl = analyze( −(u·∇s) )` — by composing the already-verified GPU kernels: gradient (5b) → transform ∇s to physical (1) → advection (2) → analyze (1). This is the first composition/orchestration increment (the explicit half of a scalar field's timestep); it wires the isolated kernels into the CPU's `solver_compute_temperature_nonlinear!` pipeline.

**Architecture:** From the CPU (`src/physics/temperature/solver.jl:71-119`): compute `∇s` in spectral space → transform `∇s` to physical → form `−(u·∇s)` in physical (velocity supplied, already physical) → transform the advection back to spectral → `nl`. The GPU `gpu_scalar_nonlinear!` chains: `gpu_scalar_gradient!` (5b) → `gpu_scalar_spectral_to_physical!` ×3 on the gradient components (1) → `gpu_scalar_advection!` (2) → `gpu_scalar_physical_to_spectral!` (1). Each sub-kernel is already verified `==`/≈ its CPU counterpart per phase; this increment verifies the **wiring** (buffers, layouts, order) by composition-equality against a manual chain. (Internal sources / geometric pre-factors of the analyze are out of scope — pure advection, matching the advection sub-path.)

**Tech Stack:** Julia, reuses Phase 5b `gpu_scalar_gradient!`, Phase 1 `gpu_scalar_spectral_to_physical!`/`gpu_scalar_physical_to_spectral!`, Phase 2 `gpu_scalar_advection!`, Phase 0 field containers. No new kernel.

---

## Prerequisite / layout

- Spectral arrays (`spec`, `∇*`, `nl`) are dense `(lmax+1, mmax+1, nr)` split real/imag (Phase 0 `GPUSpectralField` layout).
- Physical arrays (`u_r/u_θ/u_φ`, `∇*_phys`, `adv`) are `(nlat, nlon, nr)` (Phase 0 `GPUPhysicalField` layout).
- The Phase-1 transforms take `GPUSpectralField`/`GPUPhysicalField` (config-bound); the gradient/advection take raw arrays — the orchestration wraps/unwraps via the Phase-0 containers.
- `config` is a real `SHTnsKitConfig` (the Phase-1 transform uses its `get_disttranspose_plan`); single-rank works (Phase 1 tested with a real config).

## Testing without a local GPU

- **[LOCAL]** — every sub-kernel runs on Array (the transforms via SHTnsKit CPU fallback). The test asserts `gpu_scalar_nonlinear!`'s output **equals a manual chain of the same calls** (exact `==` — same arithmetic), confirming correct wiring (no buffer/layout/order bug). The sub-kernels' physics is already verified vs CPU per phase.
- **[GPU-BOX]** — same on `CuArray`; guarded by `if !GeoDynamo.gpu_functional() … @test_skip`.

Julia: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=.` from worktree `/Users/subha/Documents/GitHub/GeoDynamo-gpu` (branch `feat/gpu-phase0`, on Phases 0–5d). **Never pipe test runs through `tail`.**

## File Structure

- **Create** `src/gpu/scalar_nonlinear.jl` — `gpu_scalar_nonlinear!`.
- **Modify** `src/GeoDynamo.jl` — `include("gpu/scalar_nonlinear.jl")` (after `gpu/implicit_solve.jl`); export it.
- **Create** `test/gpu_phase5e_scalar_nonlinear.jl` — `[LOCAL]` + `[GPU-BOX]` tests.
- **Modify** `test/runtests.jl` — register.

Locked interface:

```julia
gpu_scalar_nonlinear!(nl_r, nl_i, s_r, s_i, u_r, u_θ, u_φ, config, d1, mvals, rinv, lmax, bw)
    # nl = analyze( −(u_r·∇r s + u_θ·∇θ s + u_φ·∇φ s) ), composing 5b→1→2→1
```

`nl_*`/`s_*` dense `(nl,nm,nr)`; `u_*` physical `(nlat,nlon,nr)`; `d1` banded `(2bw+1,nr)`; `mvals` len-`nm`; `rinv` len-`nr`; same backend; `nl_*` distinct from `s_*`.

---

## Task 1: `gpu_scalar_nonlinear!` (compose gradient→transform→advection→analyze)

**Files:** Create `src/gpu/scalar_nonlinear.jl`; Modify `src/GeoDynamo.jl`; Test `test/gpu_phase5e_scalar_nonlinear.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Create `test/gpu_phase5e_scalar_nonlinear.jl`:

```julia
using Test
using GeoDynamo
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5e — Scalar Nonlinear (explicit half)" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 24, nlon = 48, nr = 3)
    nl, nm, nr = cfg.lmax + 1, cfg.mmax + 1, 3
    nlat, nlon = cfg.nlat, cfg.nlon
    bw = 2
    # banded d1, mvals, rinv
    function band(::Type{T}, N, bw; seed) where {T}
        rng = MersenneTwister(seed); d = zeros(T, 2bw+1, N)
        for j in 1:N, i in max(1,j-bw):min(N,j+bw); d[bw+1+i-j,j]=rand(rng,T)-T(0.5); end
        GeoDynamo.BandedMatrix{T}(d, bw, N)
    end
    d1 = band(Float64, nr, bw; seed = 1).data
    mvals = Float64.(0:(nm-1)); rinv = [1.0/(0.5+0.1k) for k in 1:nr]
    rng = MersenneTwister(2)
    s_r = zeros(nl,nm,nr); s_i = zeros(nl,nm,nr)
    for mi in 1:nm, li in mi:nl, r in 1:nr; s_r[li,mi,r]=rand(rng); s_i[li,mi,r]=rand(rng); end
    u_r = rand(rng, nlat,nlon,nr); u_θ = rand(rng, nlat,nlon,nr); u_φ = rand(rng, nlat,nlon,nr)

    @testset "compose == manual chain [LOCAL]" begin
        nl_r = zeros(nl,nm,nr); nl_i = zeros(nl,nm,nr)
        GeoDynamo.gpu_scalar_nonlinear!(nl_r, nl_i, s_r, s_i, u_r, u_θ, u_φ, cfg, d1, mvals, rinv, cfg.lmax, bw)

        # manual reference: the same kernels, same order
        gr_r=zeros(nl,nm,nr); gr_i=zeros(nl,nm,nr); gt_r=zeros(nl,nm,nr); gt_i=zeros(nl,nm,nr); gp_r=zeros(nl,nm,nr); gp_i=zeros(nl,nm,nr)
        GeoDynamo.gpu_scalar_gradient!(gr_r,gr_i, gt_r,gt_i, gp_r,gp_i, s_r,s_i, d1, mvals, rinv, cfg.lmax, bw)
        mkspec(a,b) = GeoDynamo.GPUSpectralField{Float64,typeof(a)}(cfg, nl, nm, nr, a, b)
        mkphys() = GeoDynamo.allocate_gpu_physical_field(Float64, CPU(), cfg, nr)
        grP=mkphys(); gtP=mkphys(); gpP=mkphys()
        GeoDynamo.gpu_scalar_spectral_to_physical!(mkspec(gr_r,gr_i), grP)
        GeoDynamo.gpu_scalar_spectral_to_physical!(mkspec(gt_r,gt_i), gtP)
        GeoDynamo.gpu_scalar_spectral_to_physical!(mkspec(gp_r,gp_i), gpP)
        adv = mkphys()
        GeoDynamo.gpu_scalar_advection!(adv.data, u_r, u_θ, u_φ, grP.data, gtP.data, gpP.data)
        advspec_r = zeros(nl,nm,nr); advspec_i = zeros(nl,nm,nr)
        GeoDynamo.gpu_scalar_physical_to_spectral!(mkspec(advspec_r, advspec_i), adv)

        @test nl_r == advspec_r
        @test nl_i == advspec_i
        @test all(isfinite, nl_r) && all(isfinite, nl_i)
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5e_scalar_nonlinear.jl")'`
Expected: FAIL — `gpu_scalar_nonlinear!` undefined.

- [ ] **Step 3: Implement**

Create `src/gpu/scalar_nonlinear.jl`:

```julia
# =============================================================================
# GPU Phase 5e — scalar field nonlinear term (explicit half), composing the
# verified kernels: gradient (5b) → transform ∇s to physical (1) → advection (2)
# → analyze (1).  Mirrors solver_compute_temperature_nonlinear! (temperature/
# solver.jl:71-119, pure-advection part).  Each sub-kernel is verified vs CPU per
# phase; this wires them.  Runs on Array (locally testable) and CuArray.
# =============================================================================

"""
    gpu_scalar_nonlinear!(nl_r, nl_i, s_r, s_i, u_r, u_θ, u_φ, config, d1, mvals, rinv, lmax, bw) -> nothing

Compute a scalar field's nonlinear term `nl = analyze( −(u·∇s) )`: gradient of
`s` (spectral) → transform the gradient components to physical → advection
`−(u_r·∇r + u_θ·∇θ + u_φ·∇φ)` against the supplied physical velocity → analyze the
product back to spectral.  `nl_*`/`s_*` are dense `(nl,nm,nr)`; `u_*` physical
`(nlat,nlon,nr)`; `d1`/`mvals`/`rinv` as in `gpu_scalar_gradient!`.  All on the
same backend; `nl_*` distinct from `s_*`.  (Per-call scratch — Phase-6 may cache.)
"""
function gpu_scalar_nonlinear!(nl_r, nl_i, s_r, s_i, u_r, u_θ, u_φ, config, d1, mvals, rinv, lmax::Int, bw::Int)
    nl_size = size(s_r)
    nr = nl_size[3]
    arch = u_r isa Array ? CPU() : GPU()
    # 1. gradient (spectral)
    gr_r = similar(s_r); gr_i = similar(s_i)
    gt_r = similar(s_r); gt_i = similar(s_i)
    gp_r = similar(s_r); gp_i = similar(s_i)
    gpu_scalar_gradient!(gr_r, gr_i, gt_r, gt_i, gp_r, gp_i, s_r, s_i, d1, mvals, rinv, lmax, bw)
    # 2. transform each ∇ component to physical (wrap in Phase-0 containers)
    spec(a, b) = GPUSpectralField{eltype(a), typeof(a)}(config, nl_size[1], nl_size[2], nr, a, b)
    grP = allocate_gpu_physical_field(eltype(u_r), arch, config, nr)
    gtP = allocate_gpu_physical_field(eltype(u_r), arch, config, nr)
    gpP = allocate_gpu_physical_field(eltype(u_r), arch, config, nr)
    gpu_scalar_spectral_to_physical!(spec(gr_r, gr_i), grP)
    gpu_scalar_spectral_to_physical!(spec(gt_r, gt_i), gtP)
    gpu_scalar_spectral_to_physical!(spec(gp_r, gp_i), gpP)
    # 3. advection in physical space
    adv = allocate_gpu_physical_field(eltype(u_r), arch, config, nr)
    gpu_scalar_advection!(adv.data, u_r, u_θ, u_φ, grP.data, gtP.data, gpP.data)
    # 4. analyze the product back to spectral → nl
    gpu_scalar_physical_to_spectral!(spec(nl_r, nl_i), adv)
    return nothing
end
```

- [ ] **Step 4: Include + export**

In `src/GeoDynamo.jl`, after `include("gpu/implicit_solve.jl")` add `include("gpu/scalar_nonlinear.jl")`. Add export `gpu_scalar_nonlinear!`.

- [ ] **Step 5: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5e_scalar_nonlinear.jl")'`
Expected: PASS — orchestration output equals the manual chain (correct wiring), finite.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/scalar_nonlinear.jl src/GeoDynamo.jl test/gpu_phase5e_scalar_nonlinear.jl
git commit -m "feat(gpu): gpu_scalar_nonlinear! (explicit half: gradient→transform→advection→analyze) (Phase 5e)"
```

---

## Task 2: GPU-box gate + register + regression

**Files:** Test `test/gpu_phase5e_scalar_nonlinear.jl`, `test/runtests.jl`

- [ ] **Step 1: Add the GPU-box gate** `[GPU-BOX]`

Add to `test/gpu_phase5e_scalar_nonlinear.jl` (inside the outer testset, reusing `cfg`/`d1`/`mvals`/`rinv`/`s_*`/`u_*`):

```julia
@testset "GPU execution + GPU≈CPU parity (Phase-5e gate) [GPU-BOX]" begin
    if !GeoDynamo.gpu_functional()
        @test_skip "requires a functional CUDA GPU"
    else
        # CPU reference
        cnl_r = zeros(nl,nm,nr); cnl_i = zeros(nl,nm,nr)
        GeoDynamo.gpu_scalar_nonlinear!(cnl_r, cnl_i, s_r, s_i, u_r, u_θ, u_φ, cfg, d1, mvals, rinv, cfg.lmax, bw)
        # GPU
        d(x) = GeoDynamo.on_architecture(GPU(), x)
        gnl_r = d(zeros(nl,nm,nr)); gnl_i = d(zeros(nl,nm,nr))
        GeoDynamo.gpu_scalar_nonlinear!(gnl_r, gnl_i, d(s_r), d(s_i), d(u_r), d(u_θ), d(u_φ),
                                        cfg, d(d1), d(mvals), d(rinv), cfg.lmax, bw)
        @test gnl_r isa CUDA.CuArray
        @test isapprox(Array(gnl_r), cnl_r; atol = 1e-10, rtol = 1e-9)
        @test isapprox(Array(gnl_i), cnl_i; atol = 1e-10, rtol = 1e-9)
    end
end
```

(Tolerance is looser — 1e-10 — because the transform's GPU path goes through `gpu_synthesis`/`gpu_analysis` whose Legendre/FFT may differ from the CPU `SHTnsKit.synthesis` by more than the reduction-free kernels.)

- [ ] **Step 2: Run locally** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5e_scalar_nonlinear.jl")'`
Expected: `[LOCAL]` testsets pass; the gate skips.

- [ ] **Step 3: Register**

In `test/runtests.jl`, add `"gpu_phase5e_scalar_nonlinear.jl"` (next to the Phase 5d entry).

- [ ] **Step 4: CPU regression** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, Random; include("test/gpu_phase5d_implicit_solve.jl")'` then separately `… -e 'using Test, GeoDynamo, MPI; include("test/allocation_runtime_checks.jl")'`
Expected: Phase 5d green; allocation guards 39/39.

- [ ] **Step 5: Commit**

```bash
git add test/gpu_phase5e_scalar_nonlinear.jl test/runtests.jl
git commit -m "test(gpu): Phase-5e GPU-box gate + register scalar nonlinear"
```

---

## GPU-box validation handoff

On the GPU box:
```julia
using CUDA, Test, GeoDynamo, MPI, Random
@assert GeoDynamo.gpu_functional()
include("test/gpu_phase5e_scalar_nonlinear.jl")    # the [GPU-BOX] gate must PASS
```
**Phase 5e passes when:** the composed scalar nonlinear term on `CuArray` matches the CPU(Array) result to ~1e-10. Report any failure (a buffer/layout wiring bug, or a transform-on-CuArray surprise).

---

## What this unblocks / what's next

The scalar explicit half now runs on GPU. A full scalar field step is then `gpu_scalar_nonlinear!` (this) → `gpu_build_rhs_cnab2!` (5c) → `gpu_implicit_solve_field!` (5d), plus `nl_prev` rollover. Remaining toward the full solver:
- **Phase 5f — full scalar field step** (compose 5e + 5c + 5d + nl_prev) — temperature/composition end-to-end vs the CPU per-field step.
- **Phase 5g — vector field nonlinear/step** (velocity, magnetic): vector transform (3) + curls (5a) + cross-product nonlinears (2: u×ω, J×B, u×B) + the velocity/magnetic BC variants (rotation, poloidal influence, magnetic conducting IC).
- **Phase 5h — the full multi-field `gpu_solver_step!`** orchestration (velocity→magnetic→temperature→composition dependency order) + device `SolverState` plumbing + the GPU≈CPU full-step gate.
- **Phase 6 — `run!`/`Simulation` loop + IO host-gather.**

---

## Self-Review

**Spec coverage:** the scalar field's explicit half (nonlinear-term computation) — `gpu_scalar_nonlinear!` composing gradient (5b) → transform (1) → advection (2) → analyze (1), Task 1; GPU gate + regression, Task 2. Matches `solver_compute_temperature_nonlinear!`'s pure-advection pipeline (internal sources / non-advection terms deferred). Covered for the scalar explicit half.

**Placeholder scan:** none — complete code in every step; exact commands + expected results. `band` helper fully defined.

**Type consistency:** `gpu_scalar_nonlinear!(nl_r,nl_i, s_r,s_i, u_r,u_θ,u_φ, config, d1, mvals, rinv, lmax, bw)` — consistent across the task and the interface block. Reuses `gpu_scalar_gradient!` (5b: `(gr,gt,gp, s, d1,mvals,rinv,lmax,bw)`), `gpu_scalar_spectral_to_physical!`/`gpu_scalar_physical_to_spectral!` (1, on `GPUSpectralField`/`GPUPhysicalField`), `gpu_scalar_advection!` (2: `(out, u_r,u_θ,u_φ, ∇r,∇θ,∇φ)`), `allocate_gpu_physical_field`/`GPUSpectralField` (0). The `GPUSpectralField` wrap uses fields `(config, nl, nm, nr, data_real, data_imag)` (Phase-0 dense). The test's manual chain is the reference (the parts are verified per-phase). Tolerance 1e-10 on the GPU gate reflects the transform path.
