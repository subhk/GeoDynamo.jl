# GPU Phase 5g — Velocity Nonlinear (u×ω + Coriolis) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute the velocity field's core nonlinear term on a single GPU — `nl = analyze( E·(u×ω) − ẑ×u )` — by composing the verified GPU pieces: vector transform (3) → vorticity curl (5a) → vector transform (3) → cross-product + Coriolis (2) → vector analyze (3). This is the first VECTOR nonlinear orchestration (velocity-only; buoyancy/Lorentz couplings are added later when temperature/magnetic physical fields are available).

**Architecture:** From the CPU (`src/physics/velocity/solver.jl:10-50` + `numerics.jl:1160-1247`): (1) `prepare_velocity_fields!` transforms velocity `(T,P)`→physical `u`, computes vorticity `ω` (curl) in spectral, transforms `ω`→physical; (2) `accumulate_velocity_nonlinear_terms!` forms `adv = E·(u×ω) − (ẑ×u)` [+ buoyancy + Lorentz, deferred]; (3) `finish_velocity_nonlinear!` projects the tangential force `(adv_θ, adv_φ)` to `(nl_pol, nl_tor)` via `analysis_sphtor` — **the same `gpu_vector_physical_to_spectral!` as the plain transform (no post-scaling, `adv_r` discarded)**. The GPU `gpu_velocity_nonlinear!` chains: `gpu_vector_spectral_to_physical!` (3, velocity) → `gpu_spectral_curl!` (5a, vorticity) → `gpu_vector_spectral_to_physical!` (3, vorticity) → `gpu_cross!` (2, `E·u×ω`) + `gpu_coriolis_sub!` (2) → `gpu_vector_physical_to_spectral!` (3, analyze). All sub-pieces verified per phase; this verifies wiring.

**Tech Stack:** Julia, reuses Phase 3 vector transforms, Phase 5a `gpu_spectral_curl!`, Phase 2 `gpu_cross!`/`gpu_coriolis_sub!`, Phase 0 containers + `arch_of`. No new kernel.

---

## Layout / operator inputs

- Spectral fields (`tor`,`pol`,`nl_tor`,`nl_pol`,`ω_tor`,`ω_pol`) dense `(nl,nm,nr)` split real/imag.
- Physical (`u_*`,`ω_*`,`adv_*`) `(nlat,nlon,nr)`.
- `gpu_vector_spectral_to_physical!(vr,vθ,vφ, tor,pol, config, lfac, rscale)` (Phase 3): `lfac[l+1]=l(l+1)`, `rscale=1/r` (solver v_r factor).
- `gpu_spectral_curl!(dtr,dti,dpr,dpi, str,sti,spr,spi, d1,d2, lfac, rinv, rinv2, bw)` (5a): `d1`/`d2` `(2bw+1,nr)`, `rinv=1/r`, `rinv2=1/r²`.
- `gpu_cross!(or,oθ,oφ, a_r,a_θ,a_φ, b_r,b_θ,b_φ, coeff)` (2), `gpu_coriolis_sub!(or,oθ,oφ, u_r,u_θ,u_φ, sinθ,cosθ)` (2): `sinθ`/`cosθ` length-`nlat`.
- `gpu_vector_physical_to_spectral!(tor,pol, vθ,vφ, config)` (Phase 3).

## Testing without a local GPU

- **[LOCAL]** — every sub-piece runs on Array (transforms via SHTnsKit fallback). The test asserts `gpu_velocity_nonlinear!`'s output (`nl_tor`,`nl_pol`) **equals a manual chain** of the same calls (exact `==`), verifying wiring. Sub-pieces verified per phase.
- **[GPU-BOX]** — same on `CuArray`; guarded by `if !GeoDynamo.gpu_functional() … @test_skip`.

Julia: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=.` from worktree `/Users/subha/Documents/GitHub/GeoDynamo-gpu` (branch `feat/gpu-phase0`, on Phases 0–5f). **Never pipe test runs through `tail`.**

## File Structure

- **Create** `src/gpu/velocity_nonlinear.jl` — `gpu_velocity_nonlinear!`.
- **Modify** `src/GeoDynamo.jl` — `include("gpu/velocity_nonlinear.jl")` (after `gpu/scalar_step.jl`); export it.
- **Create** `test/gpu_phase5g_velocity_nonlinear.jl` — `[LOCAL]` + `[GPU-BOX]` tests.
- **Modify** `test/runtests.jl` — register.

Locked interface:

```julia
gpu_velocity_nonlinear!(nl_tor_r, nl_tor_i, nl_pol_r, nl_pol_i, tor_r, tor_i, pol_r, pol_i,
                        config, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, lmax, bw)
    # nl = analyze( E·(u×ω) − ẑ×u ), composing 3→5a→3→2→2→3
```

`nl_*`/`tor_*`/`pol_*` dense `(nl,nm,nr)`; `d1`/`d2` `(2bw+1,nr)`; `lfac`/`rinv`/`rinv2`/`rscale` len-`nl`/`nr`; `sinθ`/`cosθ` len-`nlat`; `E` scalar; same backend; outputs distinct from inputs.

---

## Task 1: `gpu_velocity_nonlinear!`

**Files:** Create `src/gpu/velocity_nonlinear.jl`; Modify `src/GeoDynamo.jl`; Test `test/gpu_phase5g_velocity_nonlinear.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Create `test/gpu_phase5g_velocity_nonlinear.jl`:

```julia
using Test
using GeoDynamo
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5g — Velocity Nonlinear (u×ω + Coriolis)" begin
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
    sinθ = [sin(π*(i-0.5)/nlat) for i in 1:nlat]; cosθ = [cos(π*(i-0.5)/nlat) for i in 1:nlat]
    E = 1e-3
    rng = MersenneTwister(3)
    tor_r=zeros(nl,nm,nr); tor_i=zeros(nl,nm,nr); pol_r=zeros(nl,nm,nr); pol_i=zeros(nl,nm,nr)
    for mi in 1:nm, li in mi:nl, r in 1:nr
        tor_r[li,mi,r]=rand(rng); tor_i[li,mi,r]=rand(rng); pol_r[li,mi,r]=rand(rng); pol_i[li,mi,r]=rand(rng)
    end

    @testset "velocity nonlinear == manual chain [LOCAL]" begin
        ntr=zeros(nl,nm,nr); nti=zeros(nl,nm,nr); npr=zeros(nl,nm,nr); npi=zeros(nl,nm,nr)
        GeoDynamo.gpu_velocity_nonlinear!(ntr,nti, npr,npi, tor_r,tor_i, pol_r,pol_i,
            cfg, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, cfg.lmax, bw)

        # manual chain
        spec(a,b) = GeoDynamo.GPUSpectralField{Float64,typeof(a)}(cfg, nl, nm, nr, a, b)
        ph() = GeoDynamo.allocate_gpu_physical_field(Float64, CPU(), cfg, nr)
        # 1. velocity → physical
        ur=ph(); uθ=ph(); uφ=ph()
        GeoDynamo.gpu_vector_spectral_to_physical!(ur,uθ,uφ, spec(tor_r,tor_i), spec(pol_r,pol_i), cfg, lfac, rscale)
        # 2. vorticity (curl) spectral
        wtr=zeros(nl,nm,nr); wti=zeros(nl,nm,nr); wpr=zeros(nl,nm,nr); wpi=zeros(nl,nm,nr)
        GeoDynamo.gpu_spectral_curl!(wtr,wti, wpr,wpi, tor_r,tor_i, pol_r,pol_i, d1,d2, lfac, rinv, rinv2, bw)
        # 3. vorticity → physical
        wr=ph(); wθ=ph(); wφ=ph()
        GeoDynamo.gpu_vector_spectral_to_physical!(wr,wθ,wφ, spec(wtr,wti), spec(wpr,wpi), cfg, lfac, rscale)
        # 4. adv = E·(u×ω) − ẑ×u
        ar=ph(); aθ=ph(); aφ=ph()
        GeoDynamo.gpu_cross!(ar.data,aθ.data,aφ.data, ur.data,uθ.data,uφ.data, wr.data,wθ.data,wφ.data, E)
        GeoDynamo.gpu_coriolis_sub!(ar.data,aθ.data,aφ.data, ur.data,uθ.data,uφ.data, sinθ, cosθ)
        # 5. analyze tangential → nl_tor/nl_pol
        mntr=zeros(nl,nm,nr); mnti=zeros(nl,nm,nr); mnpr=zeros(nl,nm,nr); mnpi=zeros(nl,nm,nr)
        GeoDynamo.gpu_vector_physical_to_spectral!(spec(mntr,mnti), spec(mnpr,mnpi), aθ, aφ, cfg)

        @test ntr == mntr
        @test nti == mnti
        @test npr == mnpr
        @test npi == mnpi
        @test all(isfinite, ntr) && all(isfinite, npr)
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5g_velocity_nonlinear.jl")'`
Expected: FAIL — `gpu_velocity_nonlinear!` undefined.

- [ ] **Step 3: Implement**

Create `src/gpu/velocity_nonlinear.jl`:

```julia
# =============================================================================
# GPU Phase 5g — velocity field nonlinear term (core: advection E·(u×ω) +
# Coriolis −ẑ×u), composing verified kernels: vector transform (3) → vorticity
# curl (5a) → vector transform (3) → cross + Coriolis (2) → vector analyze (3).
# Mirrors prepare_velocity_fields! + compute_velocity_body_forces! +
# finish_velocity_nonlinear! (velocity/solver.jl:10-50, numerics.jl:1160-1247),
# velocity-only part. Buoyancy (needs T) + Lorentz (needs J,B) accumulate before
# the analyze — added when those couplings are wired (5h). The force→(tor,pol)
# projection is the plain vector analysis (tangential only, no scaling, adv_r
# discarded — confirmed from finish_velocity_nonlinear!). Runs on Array + CuArray.
# =============================================================================

"""
    gpu_velocity_nonlinear!(nl_tor_r, nl_tor_i, nl_pol_r, nl_pol_i, tor_r, tor_i, pol_r, pol_i,
                            config, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, lmax, bw) -> nothing

Velocity nonlinear term `nl = analyze( E·(u×ω) − ẑ×u )` (velocity-only).  `tor`/`pol`
are the velocity toroidal/poloidal spectral; `nl_tor`/`nl_pol` the toroidal/poloidal
nonlinear spectral.  `d1`/`d2` radial derivative ops, `lfac=l(l+1)`, `rinv=1/r`,
`rinv2=1/r²`, `rscale` the v_r scaling, `sinθ`/`cosθ` the Coriolis grid factors,
`E` the Ekman number.  All on the same backend; outputs distinct from inputs.
(Per-call scratch — Phase-6 may cache.)
"""
function gpu_velocity_nonlinear!(nl_tor_r, nl_tor_i, nl_pol_r, nl_pol_i, tor_r, tor_i, pol_r, pol_i,
        config, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, lmax::Int, bw::Int)
    arch = arch_of(tor_r)
    sz = size(tor_r); nr = sz[3]
    spec(a, b) = GPUSpectralField{eltype(a), typeof(a)}(config, sz[1], sz[2], nr, a, b)
    ph() = allocate_gpu_physical_field(eltype(tor_r), arch, config, nr)
    # 1. velocity (tor,pol) → physical (u_r,u_θ,u_φ)
    ur = ph(); uθ = ph(); uφ = ph()
    gpu_vector_spectral_to_physical!(ur, uθ, uφ, spec(tor_r, tor_i), spec(pol_r, pol_i), config, lfac, rscale)
    # 2. vorticity ω = ∇×u (spectral)
    wtr = similar(tor_r); wti = similar(tor_i); wpr = similar(pol_r); wpi = similar(pol_i)
    gpu_spectral_curl!(wtr, wti, wpr, wpi, tor_r, tor_i, pol_r, pol_i, d1, d2, lfac, rinv, rinv2, bw)
    # 3. vorticity → physical (ω_r,ω_θ,ω_φ)
    wr = ph(); wθ = ph(); wφ = ph()
    gpu_vector_spectral_to_physical!(wr, wθ, wφ, spec(wtr, wti), spec(wpr, wpi), config, lfac, rscale)
    # 4. adv = E·(u×ω) − ẑ×u  (physical)
    ar = ph(); aθ = ph(); aφ = ph()
    gpu_cross!(ar.data, aθ.data, aφ.data, ur.data, uθ.data, uφ.data, wr.data, wθ.data, wφ.data, E)
    gpu_coriolis_sub!(ar.data, aθ.data, aφ.data, ur.data, uθ.data, uφ.data, sinθ, cosθ)
    # 5. analyze the tangential force → (nl_pol = S, nl_tor = T); adv_r discarded (CPU does the same)
    gpu_vector_physical_to_spectral!(spec(nl_tor_r, nl_tor_i), spec(nl_pol_r, nl_pol_i), aθ, aφ, config)
    return nothing
end
```

- [ ] **Step 4: Include + export**

In `src/GeoDynamo.jl`, after `include("gpu/scalar_step.jl")` add `include("gpu/velocity_nonlinear.jl")`. Add export `gpu_velocity_nonlinear!`.

- [ ] **Step 5: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5g_velocity_nonlinear.jl")'`
Expected: PASS — the velocity nonlinear equals the manual chain, finite.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/velocity_nonlinear.jl src/GeoDynamo.jl test/gpu_phase5g_velocity_nonlinear.jl
git commit -m "feat(gpu): gpu_velocity_nonlinear! (u×ω advection + Coriolis, vector pipeline) (Phase 5g)"
```

---

## Task 2: GPU-box gate + register + regression

**Files:** Test `test/gpu_phase5g_velocity_nonlinear.jl`, `test/runtests.jl`

- [ ] **Step 1: Add the GPU-box gate** `[GPU-BOX]`

Add to `test/gpu_phase5g_velocity_nonlinear.jl` (inside the outer testset, reusing setup):

```julia
@testset "GPU execution + GPU≈CPU parity (Phase-5g gate) [GPU-BOX]" begin
    if !GeoDynamo.gpu_functional()
        @test_skip "requires a functional CUDA GPU"
    else
        cntr=zeros(nl,nm,nr); cnti=zeros(nl,nm,nr); cnpr=zeros(nl,nm,nr); cnpi=zeros(nl,nm,nr)
        GeoDynamo.gpu_velocity_nonlinear!(cntr,cnti, cnpr,cnpi, tor_r,tor_i, pol_r,pol_i,
            cfg, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, cfg.lmax, bw)
        d(x) = GeoDynamo.on_architecture(GPU(), x)
        gntr=d(zeros(nl,nm,nr)); gnti=d(zeros(nl,nm,nr)); gnpr=d(zeros(nl,nm,nr)); gnpi=d(zeros(nl,nm,nr))
        GeoDynamo.gpu_velocity_nonlinear!(gntr,gnti, gnpr,gnpi, d(tor_r),d(tor_i), d(pol_r),d(pol_i),
            cfg, d(d1), d(d2), d(lfac), d(rinv), d(rinv2), d(rscale), d(sinθ), d(cosθ), E, cfg.lmax, bw)
        @test gntr isa CUDA.CuArray
        @test isapprox(Array(gntr), cntr; atol = 1e-9, rtol = 1e-8)
        @test isapprox(Array(gnpr), cnpr; atol = 1e-9, rtol = 1e-8)
    end
end
```

- [ ] **Step 2: Run locally** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5g_velocity_nonlinear.jl")'`
Expected: `[LOCAL]` testsets pass; the gate skips.

- [ ] **Step 3: Register**

In `test/runtests.jl`, add `"gpu_phase5g_velocity_nonlinear.jl"` (next to the Phase 5f entry).

- [ ] **Step 4: CPU regression** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5f_scalar_step.jl")'` then separately `… -e 'using Test, GeoDynamo, MPI; include("test/allocation_runtime_checks.jl")'`
Expected: Phase 5f green; allocation guards 39/39.

- [ ] **Step 5: Commit**

```bash
git add test/gpu_phase5g_velocity_nonlinear.jl test/runtests.jl
git commit -m "test(gpu): Phase-5g GPU-box gate + register velocity nonlinear"
```

---

## GPU-box validation handoff

On the GPU box:
```julia
using CUDA, Test, GeoDynamo, MPI, Random
@assert GeoDynamo.gpu_functional()
include("test/gpu_phase5g_velocity_nonlinear.jl")    # the [GPU-BOX] gate must PASS
```
**Phase 5g passes when:** the velocity nonlinear on `CuArray` matches the CPU(Array) result (nl_tor, nl_pol) to ~1e-9.

---

## What this unblocks / what's next

The velocity-only nonlinear (u×ω + Coriolis) now runs on GPU. Remaining toward the full solver:
- **Phase 5g+ — coupled velocity terms + magnetic nonlinear + vector step**: buoyancy (`+= (Pm/Pr)·Ra·r·T`, accumulate before the analyze using `gpu_buoyancy_add!` once T physical is supplied), Lorentz (`+= (1/Pm)·J×B` via `gpu_cross_add!` once current+B physical), magnetic induction (`u×B`, curl), and the vector field STEP (RHS + solve + the velocity `l=1,m=0` rotation BC, poloidal influence-matrix correction, magnetic conducting-IC reconstruction).
- **Phase 5h — full multi-field `gpu_solver_step!`** + device `SolverState` plumbing + GPU≈CPU full-step gate.
- **Phase 6 — `run!`/`Simulation` loop + IO host-gather.**

---

## Self-Review

**Spec coverage:** the velocity field's core nonlinear (advection u×ω + Coriolis) — `gpu_velocity_nonlinear!` composing vector transform (3) → vorticity (5a) → vector transform (3) → cross + Coriolis (2) → analyze (3), Task 1; GPU gate + regression, Task 2. Matches `prepare_velocity_fields!` + `compute_velocity_body_forces!` (velocity-only) + `finish_velocity_nonlinear!`. Buoyancy/Lorentz couplings are explicitly deferred (they accumulate before the analyze when T/magnetic physical are wired). The projection is the plain `gpu_vector_physical_to_spectral!` (confirmed: tangential only, no scaling, `adv_r` discarded). Covered for the velocity-only nonlinear.

**Placeholder scan:** none — complete code; exact commands + expected results. `band` helper defined.

**Type consistency:** `gpu_velocity_nonlinear!(nl_tor_r,nl_tor_i, nl_pol_r,nl_pol_i, tor_r,tor_i, pol_r,pol_i, config, d1,d2, lfac,rinv,rinv2,rscale, sinθ,cosθ, E, lmax, bw)` — consistent across the task and the interface block. Reuses `gpu_vector_spectral_to_physical!`(3: `(vr,vθ,vφ, tor,pol, config, lfac, rscale)`), `gpu_spectral_curl!`(5a), `gpu_cross!`(2: `(or,oθ,oφ, a…, b…, coeff)`), `gpu_coriolis_sub!`(2: `(or,oθ,oφ, u…, sinθ,cosθ)`), `gpu_vector_physical_to_spectral!`(3: `(tor,pol, vθ,vφ, config)`), Phase-0 `GPUSpectralField`/`allocate_gpu_physical_field`/`arch_of`. The `spec`/`ph` wrap helpers match Phase 5e's. `cross`/`coriolis` take `.data` of the physical containers. The analyze consumes the tangential `aθ`/`aφ` physical fields.
