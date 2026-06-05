# GPU Phase 5i — Coupled Velocity Terms (buoyancy + Lorentz) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the velocity field's nonlinear term on a single GPU by adding the coupled forcing — thermal buoyancy, compositional buoyancy, and the Lorentz force — accumulated into the advection before the analyze, matching the CPU `compute_velocity_body_forces!`. Extends `gpu_velocity_nonlinear!` (Phase 5g) with optional coupling inputs (backward-compatible: no couplings = the velocity-only path).

**Architecture:** From the CPU (`src/solver/numerics.jl:1226-1244`): after the `E·(u×ω) − ẑ×u` core, the force accumulates, in order: thermal buoyancy `adv_r += (Pm/Pr)·Ra·r·T`, compositional buoyancy `adv_r += (Pm/Sc)·Ra_C·r·C`, Lorentz `adv += (1/Pm)·(J×B)`. The GPU extends `gpu_velocity_nonlinear!` with optional keyword inputs and conditionally calls `gpu_buoyancy_add!` (Phase 2: `force_r += factor·r·s`) and `gpu_cross_add!` (Phase 2: `out += coeff·(a×b)`) before the final `gpu_vector_physical_to_spectral!`. All sub-pieces verified per phase; this verifies the wiring of the coupled terms.

**Tech Stack:** Julia, reuses Phase 2 `gpu_buoyancy_add!`/`gpu_cross_add!`, the Phase-5g `gpu_velocity_nonlinear!` pipeline. No new kernel. Temperature/composition/current/B physical are supplied (computed in the full step).

---

## Background (CPU reference — `src/solver/numerics.jl:1234-1244`, `1283`, `1337-1339`)

```
# after adv = E·(u×ω) − ẑ×u :
if temperature:   solver_add_thermal_buoyancy_force!(adv_r, T, (Pm/Pr)·Ra, domain)   # adv_r += (Pm/Pr)·Ra·r·T
if composition:   add_compositional_buoyancy_force!(adv_r, C, (Pm/Sc)·Ra_C, domain)   # adv_r += (Pm/Sc)·Ra_C·r·C
if magnetic:      solver_add_lorentz_force!(velocity, magnetic, Pm)                   # adv += (1/Pm)·(J×B)
```
`gpu_buoyancy_add!(force_r, s, r_vec, factor)` = `force_r += factor·r·s` (Phase 2; `r_vec` = the `r` values). `gpu_cross_add!(or,oθ,oφ, a…, b…, coeff)` = `out += coeff·(a×b)` (Phase 2; `a=J`, `b=B`, `coeff=1/Pm`).

## Testing without a local GPU

- **[LOCAL]** — the test (a) re-runs the existing velocity-only path (no kwargs) and confirms it's unchanged; (b) runs with all couplings and asserts `==` a manual chain (core + `gpu_buoyancy_add!` ×2 + `gpu_cross_add!` + analyze), exact `==`.
- **[GPU-BOX]** — same on `CuArray`; guarded by `if !GeoDynamo.gpu_functional() … @test_skip`.

Julia: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=.` from worktree `/Users/subha/Documents/GitHub/GeoDynamo-gpu` (branch `feat/gpu-phase0`, on Phases 0–5h). **Never pipe test runs through `tail`.**

## File Structure

- **Modify** `src/gpu/velocity_nonlinear.jl` — add the coupling keyword inputs + conditional accumulation to `gpu_velocity_nonlinear!`.
- **Create** `test/gpu_phase5i_coupled_velocity.jl` — `[LOCAL]` + `[GPU-BOX]` tests.
- **Modify** `test/runtests.jl` — register.

Extended interface (keywords, all default to the velocity-only path):

```julia
gpu_velocity_nonlinear!(...positional as Phase 5g...;
    T_phys = nothing, thermal_factor = 0, r_vec = nothing,        # thermal buoyancy: adv_r += thermal_factor·r·T_phys
    C_phys = nothing, comp_factor = 0,                            # compositional: adv_r += comp_factor·r·C_phys
    J_r = nothing, J_θ = nothing, J_φ = nothing,                  # Lorentz current
    B_r = nothing, B_θ = nothing, B_φ = nothing, lorentz_coeff = 0)  # adv += lorentz_coeff·(J×B)
```

`T_phys`/`C_phys`/`J_*`/`B_*` physical `(nlat,nlon,nr)`; `r_vec` len-`nr`. Accumulation order: thermal → compositional → Lorentz, before the analyze.

---

## Task 1: extend `gpu_velocity_nonlinear!` with the coupled terms

**Files:** Modify `src/gpu/velocity_nonlinear.jl`; Test `test/gpu_phase5i_coupled_velocity.jl`

- [ ] **Step 1: Write the failing test** `[LOCAL]`

Create `test/gpu_phase5i_coupled_velocity.jl`:

```julia
using Test
using GeoDynamo
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5i — Coupled Velocity (buoyancy + Lorentz)" begin
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
    r_vec = [0.5+0.1k for k in 1:nr]
    sinθ = [sin(π*(i-0.5)/nlat) for i in 1:nlat]; cosθ = [cos(π*(i-0.5)/nlat) for i in 1:nlat]
    E = 1e-3
    rng = MersenneTwister(3)
    tor_r=zeros(nl,nm,nr); tor_i=zeros(nl,nm,nr); pol_r=zeros(nl,nm,nr); pol_i=zeros(nl,nm,nr)
    for mi in 1:nm, li in mi:nl, r in 1:nr
        tor_r[li,mi,r]=rand(rng); tor_i[li,mi,r]=rand(rng); pol_r[li,mi,r]=rand(rng); pol_i[li,mi,r]=rand(rng)
    end
    Tp = rand(rng,nlat,nlon,nr); Cp = rand(rng,nlat,nlon,nr)
    Jr=rand(rng,nlat,nlon,nr); Jθ=rand(rng,nlat,nlon,nr); Jφ=rand(rng,nlat,nlon,nr)
    Br=rand(rng,nlat,nlon,nr); Bθ=rand(rng,nlat,nlon,nr); Bφ=rand(rng,nlat,nlon,nr)
    tf = 0.7; cf = 0.4; lc = 1.0/0.3

    @testset "coupled == core + buoyancy + Lorentz manual chain [LOCAL]" begin
        ntr=zeros(nl,nm,nr); nti=zeros(nl,nm,nr); npr=zeros(nl,nm,nr); npi=zeros(nl,nm,nr)
        GeoDynamo.gpu_velocity_nonlinear!(ntr,nti, npr,npi, tor_r,tor_i, pol_r,pol_i,
            cfg, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, cfg.lmax, bw;
            T_phys = Tp, thermal_factor = tf, r_vec = r_vec, C_phys = Cp, comp_factor = cf,
            J_r = Jr, J_θ = Jθ, J_φ = Jφ, B_r = Br, B_θ = Bθ, B_φ = Bφ, lorentz_coeff = lc)

        # manual chain: core pipeline + buoyancy ×2 + Lorentz, then analyze
        spec(a,b) = GeoDynamo.GPUSpectralField{Float64,typeof(a)}(cfg, nl, nm, nr, a, b)
        ph() = GeoDynamo.allocate_gpu_physical_field(Float64, CPU(), cfg, nr)
        ur=ph(); uθ=ph(); uφ=ph()
        GeoDynamo.gpu_vector_spectral_to_physical!(ur,uθ,uφ, spec(tor_r,tor_i), spec(pol_r,pol_i), cfg, lfac, rscale)
        wtr=zeros(nl,nm,nr); wti=zeros(nl,nm,nr); wpr=zeros(nl,nm,nr); wpi=zeros(nl,nm,nr)
        GeoDynamo.gpu_spectral_curl!(wtr,wti, wpr,wpi, tor_r,tor_i, pol_r,pol_i, d1,d2, lfac, rinv, rinv2, bw)
        wr=ph(); wθ=ph(); wφ=ph()
        GeoDynamo.gpu_vector_spectral_to_physical!(wr,wθ,wφ, spec(wtr,wti), spec(wpr,wpi), cfg, lfac, rscale)
        ar=ph(); aθ=ph(); aφ=ph()
        GeoDynamo.gpu_cross!(ar.data,aθ.data,aφ.data, ur.data,uθ.data,uφ.data, wr.data,wθ.data,wφ.data, E)
        GeoDynamo.gpu_coriolis_sub!(ar.data,aθ.data,aφ.data, ur.data,uθ.data,uφ.data, sinθ, cosθ)
        GeoDynamo.gpu_buoyancy_add!(ar.data, Tp, r_vec, tf)
        GeoDynamo.gpu_buoyancy_add!(ar.data, Cp, r_vec, cf)
        GeoDynamo.gpu_cross_add!(ar.data,aθ.data,aφ.data, Jr,Jθ,Jφ, Br,Bθ,Bφ, lc)
        mntr=zeros(nl,nm,nr); mnti=zeros(nl,nm,nr); mnpr=zeros(nl,nm,nr); mnpi=zeros(nl,nm,nr)
        GeoDynamo.gpu_vector_physical_to_spectral!(spec(mntr,mnti), spec(mnpr,mnpi), aθ, aφ, cfg)

        @test ntr == mntr
        @test nti == mnti
        @test npr == mnpr
        @test npi == mnpi
    end

    @testset "no couplings == velocity-only (5g unchanged) [LOCAL]" begin
        # with kwargs omitted, must equal the velocity-only path
        a1=zeros(nl,nm,nr); a2=zeros(nl,nm,nr); a3=zeros(nl,nm,nr); a4=zeros(nl,nm,nr)
        GeoDynamo.gpu_velocity_nonlinear!(a1,a2, a3,a4, tor_r,tor_i, pol_r,pol_i,
            cfg, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, cfg.lmax, bw)
        @test all(isfinite, a1) && all(isfinite, a3)   # velocity-only path still runs
    end
end
```

- [ ] **Step 2: Run it, verify it fails**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5i_coupled_velocity.jl")'`
Expected: FAIL — `gpu_velocity_nonlinear!` has no `T_phys`/etc. keyword (MethodError).

- [ ] **Step 3: Extend the function**

In `src/gpu/velocity_nonlinear.jl`, change the signature to accept the keywords and add the conditional accumulation between the Coriolis call and the analyze. Replace the function header + the step-4/5 region:

```julia
function gpu_velocity_nonlinear!(nl_tor_r, nl_tor_i, nl_pol_r, nl_pol_i, tor_r, tor_i, pol_r, pol_i,
        config, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, lmax::Int, bw::Int;
        T_phys = nothing, thermal_factor = zero(eltype(tor_r)), r_vec = nothing,
        C_phys = nothing, comp_factor = zero(eltype(tor_r)),
        J_r = nothing, J_θ = nothing, J_φ = nothing,
        B_r = nothing, B_θ = nothing, B_φ = nothing, lorentz_coeff = zero(eltype(tor_r)))
```

Keep steps 1–3 (transform, curl, transform) and the step-4 `gpu_cross!` + `gpu_coriolis_sub!` exactly as they are. Then, BEFORE the step-5 analyze, insert:

```julia
    # 4b. coupled forcing accumulated into adv (CPU order: thermal → compositional → Lorentz)
    if T_phys !== nothing
        gpu_buoyancy_add!(ar.data, T_phys, r_vec, thermal_factor)     # adv_r += thermal_factor·r·T
    end
    if C_phys !== nothing
        gpu_buoyancy_add!(ar.data, C_phys, r_vec, comp_factor)        # adv_r += comp_factor·r·C
    end
    if J_r !== nothing
        gpu_cross_add!(ar.data, aθ.data, aφ.data, J_r, J_θ, J_φ, B_r, B_θ, B_φ, lorentz_coeff)  # adv += lorentz_coeff·(J×B)
    end
```

Update the docstring to document the keywords (thermal/compositional buoyancy, Lorentz; defaults give the velocity-only path).

- [ ] **Step 4: Run the test, verify it passes** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5i_coupled_velocity.jl")'`
Expected: PASS — coupled equals the manual chain; velocity-only path still runs.

- [ ] **Step 5: Confirm Phase 5g unchanged** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5g_velocity_nonlinear.jl")'`
Expected: Phase 5g (no-kwargs) still PASSES — the keyword extension is backward-compatible.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/velocity_nonlinear.jl test/gpu_phase5i_coupled_velocity.jl
git commit -m "feat(gpu): velocity nonlinear coupled terms (buoyancy + Lorentz) (Phase 5i)"
```

---

## Task 2: GPU-box gate + register + regression

**Files:** Test `test/gpu_phase5i_coupled_velocity.jl`, `test/runtests.jl`

- [ ] **Step 1: Add the GPU-box gate** `[GPU-BOX]`

Add to `test/gpu_phase5i_coupled_velocity.jl` (inside the outer testset, reusing setup):

```julia
@testset "GPU execution + GPU≈CPU parity (Phase-5i gate) [GPU-BOX]" begin
    if !GeoDynamo.gpu_functional()
        @test_skip "requires a functional CUDA GPU"
    else
        cntr=zeros(nl,nm,nr); cnti=zeros(nl,nm,nr); cnpr=zeros(nl,nm,nr); cnpi=zeros(nl,nm,nr)
        GeoDynamo.gpu_velocity_nonlinear!(cntr,cnti, cnpr,cnpi, tor_r,tor_i, pol_r,pol_i,
            cfg, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, cfg.lmax, bw;
            T_phys=Tp, thermal_factor=tf, r_vec=r_vec, C_phys=Cp, comp_factor=cf,
            J_r=Jr, J_θ=Jθ, J_φ=Jφ, B_r=Br, B_θ=Bθ, B_φ=Bφ, lorentz_coeff=lc)
        d(x) = GeoDynamo.on_architecture(GPU(), x)
        gntr=d(zeros(nl,nm,nr)); gnti=d(zeros(nl,nm,nr)); gnpr=d(zeros(nl,nm,nr)); gnpi=d(zeros(nl,nm,nr))
        GeoDynamo.gpu_velocity_nonlinear!(gntr,gnti, gnpr,gnpi, d(tor_r),d(tor_i), d(pol_r),d(pol_i),
            cfg, d(d1), d(d2), d(lfac), d(rinv), d(rinv2), d(rscale), d(sinθ), d(cosθ), E, cfg.lmax, bw;
            T_phys=d(Tp), thermal_factor=tf, r_vec=d(r_vec), C_phys=d(Cp), comp_factor=cf,
            J_r=d(Jr), J_θ=d(Jθ), J_φ=d(Jφ), B_r=d(Br), B_θ=d(Bθ), B_φ=d(Bφ), lorentz_coeff=lc)
        @test gntr isa CUDA.CuArray
        @test isapprox(Array(gntr), cntr; atol = 1e-9, rtol = 1e-8)
        @test isapprox(Array(gnti), cnti; atol = 1e-9, rtol = 1e-8)
        @test isapprox(Array(gnpr), cnpr; atol = 1e-9, rtol = 1e-8)
        @test isapprox(Array(gnpi), cnpi; atol = 1e-9, rtol = 1e-8)
    end
end
```

- [ ] **Step 2: Run locally** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5i_coupled_velocity.jl")'`
Expected: `[LOCAL]` testsets pass; the gate skips.

- [ ] **Step 3: Register**

In `test/runtests.jl`, add `"gpu_phase5i_coupled_velocity.jl"` (next to the Phase 5h entry).

- [ ] **Step 4: CPU regression** `[LOCAL]`

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5h_magnetic_nonlinear.jl")'` then separately `… -e 'using Test, GeoDynamo, MPI; include("test/allocation_runtime_checks.jl")'`
Expected: Phase 5h green; allocation guards 39/39.

- [ ] **Step 5: Commit**

```bash
git add test/gpu_phase5i_coupled_velocity.jl test/runtests.jl
git commit -m "test(gpu): Phase-5i GPU-box gate + register coupled velocity"
```

---

## GPU-box validation handoff

On the GPU box:
```julia
using CUDA, Test, GeoDynamo, MPI, Random
@assert GeoDynamo.gpu_functional()
include("test/gpu_phase5i_coupled_velocity.jl")    # the [GPU-BOX] gate must PASS
```
**Phase 5i passes when:** the coupled velocity nonlinear on `CuArray` matches the CPU(Array) result to ~1e-9.

---

## What this unblocks / what's next

The full velocity nonlinear (advection + Coriolis + buoyancy + Lorentz) now runs on GPU. Remaining toward the full solver:
- **Vector field STEP**: RHS + implicit solve + the field-specific BCs (velocity `l=1,m=0` rotation, poloidal influence-matrix correction, magnetic conducting-inner-core reconstruction + inner-core rotation coupling).
- **Full `gpu_solver_step!`** (velocity→magnetic→temperature→composition order) + device `SolverState` plumbing (all fields + caches + BC + the curl/gradient operator arrays on device) + GPU≈CPU full-step gate.
- **`run!`/`Simulation` loop + IO host-gather.**

---

## Self-Review

**Spec coverage:** the coupled velocity forcing — `gpu_velocity_nonlinear!` extended with thermal buoyancy + compositional buoyancy + Lorentz (Task 1), GPU gate + regression (Task 2). Matches `compute_velocity_body_forces!:1234-1244` order (thermal → compositional → Lorentz, accumulated before the analyze). Backward-compatible: no kwargs = the Phase-5g velocity-only path (verified by re-running the 5g test). Covered for the coupled velocity nonlinear.

**Placeholder scan:** none — complete code; exact commands + expected results. `band` helper defined.

**Type consistency:** the extended `gpu_velocity_nonlinear!` keeps all Phase-5g positional args + adds keyword `T_phys`/`thermal_factor`/`r_vec`/`C_phys`/`comp_factor`/`J_*`/`B_*`/`lorentz_coeff` (all default to the velocity-only path). Uses `gpu_buoyancy_add!(force_r, s, r_vec, factor)` (Phase 2: `force_r += factor·r·s`) and `gpu_cross_add!(or,oθ,oφ, a…, b…, coeff)` (Phase 2: `out += coeff·(a×b)`; `a=J`, `b=B`). Accumulation order (thermal → compositional → Lorentz, before analyze) matches the CPU. The buoyancy adds only to `ar.data` (radial); Lorentz adds to all three. The test's manual chain is the reference; the no-kwargs test confirms backward compatibility.
