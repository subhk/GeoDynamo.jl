# GPU Phase 5h — Magnetic Nonlinear (induction ∇×(u×B)) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute the magnetic field's nonlinear term (induction `nl = ∇×(u×B)`) on a single GPU — composing the verified pieces: vector transform (3, B→physical) → cross-product (2, u×B) → vector analyze (3, u×B→spectral) → spectral curl (5a) — matching the CPU `apply_induction_nonlinear!`.

**Architecture:** From the CPU (`src/solver/numerics.jl:1491-1603`): (1) `solver_compute_velocity_cross_magnetic!` forms `uB = u×B` in physical (`uB_r = u_θ·B_φ − u_φ·B_θ`, etc); (2) `vector_physical_to_spectral!(uB, work_tor, work_pol)` analyzes the tangential `(uB_θ, uB_φ)` to spectral `(work_tor = T, work_pol = S)`; (3) `solver_compute_curl_of_induction!` applies `spectral_curl_torpol!(work_tor, work_pol → nl_tor, nl_pol)`. The EXTRA curl (vs the velocity force projection) is because the induction equation is `∂B/∂t = ∇×(u×B)`. The GPU `gpu_magnetic_nonlinear!` chains: `gpu_vector_spectral_to_physical!` (3, B) → `gpu_cross!` (2, `u×B`, coeff 1) → `gpu_vector_physical_to_spectral!` (3, uB→work) → `gpu_spectral_curl!` (5a, curl(work)→nl). All sub-pieces verified per phase; this verifies wiring.

**Tech Stack:** Julia, reuses Phase 3 vector transforms, Phase 2 `gpu_cross!`, Phase 5a `gpu_spectral_curl!`, Phase 0 containers + `arch_of`. No new kernel. `u` physical is supplied (computed in the velocity nonlinear, Phase 5g).

---

## Layout / operator inputs

- Spectral (`B_tor`,`B_pol`,`nl_tor`,`nl_pol`,`work_tor`,`work_pol`) dense `(nl,nm,nr)`; physical (`u_*`,`B_*`,`uB_*`) `(nlat,nlon,nr)`.
- `gpu_vector_spectral_to_physical!(vr,vθ,vφ, tor,pol, config, lfac, rscale)` (3); `gpu_cross!(or,oθ,oφ, a…,b…, coeff)` (2); `gpu_vector_physical_to_spectral!(tor,pol, vθ,vφ, config)` (3); `gpu_spectral_curl!(dtr,dti,dpr,dpi, str,sti,spr,spi, d1,d2,lfac,rinv,rinv2, bw)` (5a).

## Testing without a local GPU

- **[LOCAL]** — every sub-piece runs on Array. The test asserts `gpu_magnetic_nonlinear!`'s output `==` a manual chain (exact `==`). Sub-pieces verified per phase.
- **[GPU-BOX]** — same on `CuArray`; guarded by `if !GeoDynamo.gpu_functional() … @test_skip`.

Julia: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=.` from worktree `/Users/subha/Documents/GitHub/GeoDynamo-gpu` (branch `feat/gpu-phase0`, on Phases 0–5g). **Never pipe test runs through `tail`.**

## File Structure

- **Create** `src/gpu/magnetic_nonlinear.jl` — `gpu_magnetic_nonlinear!`.
- **Modify** `src/GeoDynamo.jl` — `include("gpu/magnetic_nonlinear.jl")` (after `gpu/velocity_nonlinear.jl`); export it.
- **Create** `test/gpu_phase5h_magnetic_nonlinear.jl` — `[LOCAL]` + `[GPU-BOX]` tests.
- **Modify** `test/runtests.jl` — register.

Locked interface:

```julia
gpu_magnetic_nonlinear!(nl_tor_r, nl_tor_i, nl_pol_r, nl_pol_i, B_tor_r, B_tor_i, B_pol_r, B_pol_i,
                        u_r, u_θ, u_φ, config, d1, d2, lfac, rinv, rinv2, rscale, lmax, bw)
    # nl = curl( analyze( u×B ) ), composing 3→2→3→5a
```

`nl_*`/`B_*` dense `(nl,nm,nr)`; `u_*` physical `(nlat,nlon,nr)`; `d1`/`d2` `(2bw+1,nr)`; `lfac`/`rinv`/`rinv2`/`rscale` len-`nl`/`nr`; same backend; outputs distinct from inputs.

---

## Task 1: `gpu_magnetic_nonlinear!`

**Files:** Create `src/gpu/magnetic_nonlinear.jl`; Modify `src/GeoDynamo.jl`; Test `test/gpu_phase5h_magnetic_nonlinear.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Create `test/gpu_phase5h_magnetic_nonlinear.jl`:

```julia
using Test
using GeoDynamo
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5h — Magnetic Nonlinear (induction ∇×(u×B))" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 24, nlon = 48, nr = 4)
    nl, nm, nr = cfg.lmax + 1, cfg.mmax + 1, 4
    nlat, nlon = cfg.nlat, cfg.nlon
    bw = 2
    function band(::Type{T}, N, bw; seed) where {T}
        rng = MersenneTwister(seed); d = zeros(T, 2bw+1, N)
        for j in 1:N, i in max(1,j-bw):min(N,j+bw); d[bw+1+i-j,j]=rand(rng,T)-T(0.5); end
        d
    end
    d1 = band(Float64, nr, bw; seed = 1); d2 = band(Float64, nr, bw; seed = 2)
    lfac = Float64[l*(l+1) for l in 0:cfg.lmax]
    rinv = [1.0/(0.5+0.1k) for k in 1:nr]; rinv2 = rinv .^ 2; rscale = copy(rinv)
    rng = MersenneTwister(3)
    btr=zeros(nl,nm,nr); bti=zeros(nl,nm,nr); bpr=zeros(nl,nm,nr); bpi=zeros(nl,nm,nr)
    for mi in 1:nm, li in mi:nl, r in 1:nr
        btr[li,mi,r]=rand(rng); bti[li,mi,r]=rand(rng); bpr[li,mi,r]=rand(rng); bpi[li,mi,r]=rand(rng)
    end
    u_r=rand(rng,nlat,nlon,nr); u_θ=rand(rng,nlat,nlon,nr); u_φ=rand(rng,nlat,nlon,nr)

    @testset "magnetic nonlinear == manual chain [LOCAL]" begin
        ntr=zeros(nl,nm,nr); nti=zeros(nl,nm,nr); npr=zeros(nl,nm,nr); npi=zeros(nl,nm,nr)
        GeoDynamo.gpu_magnetic_nonlinear!(ntr,nti, npr,npi, btr,bti, bpr,bpi, u_r,u_θ,u_φ,
            cfg, d1, d2, lfac, rinv, rinv2, rscale, cfg.lmax, bw)

        # manual chain
        spec(a,b) = GeoDynamo.GPUSpectralField{Float64,typeof(a)}(cfg, nl, nm, nr, a, b)
        ph() = GeoDynamo.allocate_gpu_physical_field(Float64, CPU(), cfg, nr)
        # 1. B → physical
        Br=ph(); Bθ=ph(); Bφ=ph()
        GeoDynamo.gpu_vector_spectral_to_physical!(Br,Bθ,Bφ, spec(btr,bti), spec(bpr,bpi), cfg, lfac, rscale)
        # 2. uB = u×B
        ubr=ph(); ubθ=ph(); ubφ=ph()
        GeoDynamo.gpu_cross!(ubr.data,ubθ.data,ubφ.data, u_r,u_θ,u_φ, Br.data,Bθ.data,Bφ.data, 1.0)
        # 3. uB → spectral (work_tor, work_pol)
        wtr=zeros(nl,nm,nr); wti=zeros(nl,nm,nr); wpr=zeros(nl,nm,nr); wpi=zeros(nl,nm,nr)
        GeoDynamo.gpu_vector_physical_to_spectral!(spec(wtr,wti), spec(wpr,wpi), ubθ, ubφ, cfg)
        # 4. curl(work) → nl
        mntr=zeros(nl,nm,nr); mnti=zeros(nl,nm,nr); mnpr=zeros(nl,nm,nr); mnpi=zeros(nl,nm,nr)
        GeoDynamo.gpu_spectral_curl!(mntr,mnti, mnpr,mnpi, wtr,wti, wpr,wpi, d1,d2, lfac, rinv, rinv2, bw)

        @test ntr == mntr
        @test nti == mnti
        @test npr == mnpr
        @test npi == mnpi
        @test all(isfinite, ntr) && all(isfinite, npr)
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5h_magnetic_nonlinear.jl")'`
Expected: FAIL — `gpu_magnetic_nonlinear!` undefined.

- [ ] **Step 3: Implement**

Create `src/gpu/magnetic_nonlinear.jl`:

```julia
# =============================================================================
# GPU Phase 5h — magnetic field nonlinear term (induction nl = ∇×(u×B)),
# composing: vector transform (3, B→physical) → cross u×B (2) → vector analyze
# (3, u×B → work_tor/work_pol) → spectral curl (5a, curl(work) → nl).  Mirrors
# apply_induction_nonlinear! (numerics.jl:1491-1603).  The extra curl (vs the
# velocity force projection) is the ∇× of the induction equation.  u physical is
# supplied (from the velocity nonlinear).  Runs on Array + CuArray.
# (Per-call scratch — Phase-6 may cache. Inner-core rotation coupling deferred.)
# =============================================================================

"""
    gpu_magnetic_nonlinear!(nl_tor_r, nl_tor_i, nl_pol_r, nl_pol_i, B_tor_r, B_tor_i, B_pol_r, B_pol_i,
                            u_r, u_θ, u_φ, config, d1, d2, lfac, rinv, rinv2, rscale, lmax, bw) -> nothing

Magnetic induction nonlinear `nl = ∇×(u×B)`.  `B_tor`/`B_pol` the magnetic
toroidal/poloidal spectral; `u_*` the physical velocity (supplied); `nl_tor`/`nl_pol`
the toroidal/poloidal induction nonlinear.  `d1`/`d2`/`lfac`/`rinv`/`rinv2`/`rscale`
as in the curl/transform.  All on the same backend; outputs distinct from inputs.
"""
function gpu_magnetic_nonlinear!(nl_tor_r, nl_tor_i, nl_pol_r, nl_pol_i, B_tor_r, B_tor_i, B_pol_r, B_pol_i,
        u_r, u_θ, u_φ, config, d1, d2, lfac, rinv, rinv2, rscale, lmax::Int, bw::Int)
    arch = arch_of(B_tor_r)
    sz = size(B_tor_r); nr = sz[3]
    spec(a, b) = GPUSpectralField{eltype(a), typeof(a)}(config, sz[1], sz[2], nr, a, b)
    ph() = allocate_gpu_physical_field(eltype(B_tor_r), arch, config, nr)
    # 1. B (tor,pol) → physical (B_r,B_θ,B_φ)
    Br = ph(); Bθ = ph(); Bφ = ph()
    gpu_vector_spectral_to_physical!(Br, Bθ, Bφ, spec(B_tor_r, B_tor_i), spec(B_pol_r, B_pol_i), config, lfac, rscale)
    # 2. uB = u×B (physical), coeff 1
    ubr = ph(); ubθ = ph(); ubφ = ph()
    gpu_cross!(ubr.data, ubθ.data, ubφ.data, u_r, u_θ, u_φ, Br.data, Bθ.data, Bφ.data, one(eltype(B_tor_r)))
    # 3. uB → spectral (work_tor = T, work_pol = S), tangential analyze
    wtr = similar(B_tor_r); wti = similar(B_tor_i); wpr = similar(B_pol_r); wpi = similar(B_pol_i)
    gpu_vector_physical_to_spectral!(spec(wtr, wti), spec(wpr, wpi), ubθ, ubφ, config)
    # 4. curl(work) → nl  (∇× of the induction)
    gpu_spectral_curl!(nl_tor_r, nl_tor_i, nl_pol_r, nl_pol_i, wtr, wti, wpr, wpi, d1, d2, lfac, rinv, rinv2, bw)
    return nothing
end
```

- [ ] **Step 4: Include + export**

In `src/GeoDynamo.jl`, after `include("gpu/velocity_nonlinear.jl")` add `include("gpu/magnetic_nonlinear.jl")`. Add export `gpu_magnetic_nonlinear!`.

- [ ] **Step 5: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5h_magnetic_nonlinear.jl")'`
Expected: PASS — the magnetic nonlinear equals the manual chain, finite.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/magnetic_nonlinear.jl src/GeoDynamo.jl test/gpu_phase5h_magnetic_nonlinear.jl
git commit -m "feat(gpu): gpu_magnetic_nonlinear! (induction ∇×(u×B)) (Phase 5h)"
```

---

## Task 2: GPU-box gate + register + regression

**Files:** Test `test/gpu_phase5h_magnetic_nonlinear.jl`, `test/runtests.jl`

- [ ] **Step 1: Add the GPU-box gate** `[GPU-BOX]`

Add to `test/gpu_phase5h_magnetic_nonlinear.jl` (inside the outer testset, reusing setup):

```julia
@testset "GPU execution + GPU≈CPU parity (Phase-5h gate) [GPU-BOX]" begin
    if !GeoDynamo.gpu_functional()
        @test_skip "requires a functional CUDA GPU"
    else
        cntr=zeros(nl,nm,nr); cnti=zeros(nl,nm,nr); cnpr=zeros(nl,nm,nr); cnpi=zeros(nl,nm,nr)
        GeoDynamo.gpu_magnetic_nonlinear!(cntr,cnti, cnpr,cnpi, btr,bti, bpr,bpi, u_r,u_θ,u_φ,
            cfg, d1, d2, lfac, rinv, rinv2, rscale, cfg.lmax, bw)
        d(x) = GeoDynamo.on_architecture(GPU(), x)
        gntr=d(zeros(nl,nm,nr)); gnti=d(zeros(nl,nm,nr)); gnpr=d(zeros(nl,nm,nr)); gnpi=d(zeros(nl,nm,nr))
        GeoDynamo.gpu_magnetic_nonlinear!(gntr,gnti, gnpr,gnpi, d(btr),d(bti), d(bpr),d(bpi),
            d(u_r),d(u_θ),d(u_φ), cfg, d(d1), d(d2), d(lfac), d(rinv), d(rinv2), d(rscale), cfg.lmax, bw)
        @test gntr isa CUDA.CuArray
        @test isapprox(Array(gntr), cntr; atol = 1e-9, rtol = 1e-8)
        @test isapprox(Array(gnpr), cnpr; atol = 1e-9, rtol = 1e-8)
    end
end
```

- [ ] **Step 2: Run locally** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5h_magnetic_nonlinear.jl")'`
Expected: `[LOCAL]` testsets pass; the gate skips.

- [ ] **Step 3: Register**

In `test/runtests.jl`, add `"gpu_phase5h_magnetic_nonlinear.jl"` (next to the Phase 5g entry).

- [ ] **Step 4: CPU regression** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5g_velocity_nonlinear.jl")'` then separately `… -e 'using Test, GeoDynamo, MPI; include("test/allocation_runtime_checks.jl")'`
Expected: Phase 5g green; allocation guards 39/39.

- [ ] **Step 5: Commit**

```bash
git add test/gpu_phase5h_magnetic_nonlinear.jl test/runtests.jl
git commit -m "test(gpu): Phase-5h GPU-box gate + register magnetic nonlinear"
```

---

## GPU-box validation handoff

On the GPU box:
```julia
using CUDA, Test, GeoDynamo, MPI, Random
@assert GeoDynamo.gpu_functional()
include("test/gpu_phase5h_magnetic_nonlinear.jl")    # the [GPU-BOX] gate must PASS
```
**Phase 5h passes when:** the magnetic nonlinear on `CuArray` matches the CPU(Array) result (nl_tor, nl_pol) to ~1e-9.

---

## What this unblocks / what's next

Both the velocity (5g) and magnetic (5h) nonlinears now run on GPU. Remaining toward the full solver:
- **Coupled velocity terms**: buoyancy (`gpu_buoyancy_add!` with T physical) + Lorentz (`gpu_cross_add!` with current J + B physical) accumulated into the velocity advection before its analyze.
- **Vector field STEP**: RHS + implicit solve + the field-specific BCs (velocity `l=1,m=0` rotation, poloidal influence-matrix correction, magnetic conducting-inner-core reconstruction + inner-core rotation coupling `apply_inner_core_rotation!`).
- **Phase 6 — full `gpu_solver_step!`** (velocity→magnetic→temperature→composition order) + device `SolverState` plumbing + GPU≈CPU full-step gate + `run!`/IO.

---

## Self-Review

**Spec coverage:** the magnetic field's induction nonlinear `∇×(u×B)` — `gpu_magnetic_nonlinear!` composing vector transform (3, B) → cross u×B (2) → vector analyze (3) → spectral curl (5a), Task 1; GPU gate + regression, Task 2. Matches `apply_induction_nonlinear!` (`solver_compute_velocity_cross_magnetic!` + `vector_physical_to_spectral!` + `solver_compute_curl_of_induction!`). The inner-core rotation coupling (`apply_inner_core_rotation!`) is deferred. Covered for the induction nonlinear.

**Placeholder scan:** none — complete code; exact commands + expected results. `band` helper defined.

**Type consistency:** `gpu_magnetic_nonlinear!(nl_tor_r,nl_tor_i, nl_pol_r,nl_pol_i, B_tor_r,B_tor_i, B_pol_r,B_pol_i, u_r,u_θ,u_φ, config, d1,d2, lfac,rinv,rinv2,rscale, lmax, bw)` — consistent across the task and the interface block. Reuses `gpu_vector_spectral_to_physical!`(3), `gpu_cross!`(2, coeff `one(...)`=1 for u×B), `gpu_vector_physical_to_spectral!`(3, tangential `ubθ`/`ubφ`), `gpu_spectral_curl!`(5a, `src_tor=work_tor`, `src_pol=work_pol`). The cross is `u×B` (u=a, B=b) matching `solver_compute_velocity_cross_magnetic!` (`uB_r=u_θ·B_φ−u_φ·B_θ`). The curl input is the analyzed `(work_tor, work_pol)` of `u×B` (`solver_compute_curl_of_induction!`). `spec`/`ph` wrap helpers match Phase 5g.
