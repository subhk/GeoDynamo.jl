# GPU Phase 5m — Magnetic Field CNAB2 Step (insulating) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compose one full CNAB2 timestep of the magnetic field (toroidal + poloidal, **insulating** inner core) on the GPU from the verified kernels — magnetic nonlinear (5h) → CNAB2 RHS (5c) → implicit solve (5d) → history rollover — bit-exact against a manual chain. The toroidal optionally carries the `CONTINUITY_MAG` inner-boundary coupling. The conducting-inner-core path is a separate phase (5m2).

**Architecture:** A new file `src/gpu/magnetic_step.jl` with one orchestrator `gpu_magnetic_field_step!`. Pure composition — **no new kernels**. Like the velocity step (5k) but: the magnetic nonlinear (5h, induction ∇×(u×B)) replaces the velocity nonlinear; the mass coefficient is `1` (not `E`); there is no poloidal influence correction; the toroidal inner boundary optionally takes the `CONTINUITY_MAG` increment `-nl_pol[ICB] + prev_nl_pol[ICB]` (computed internally from the just-computed poloidal nonlinear and the old poloidal history); the poloidal is fully homogeneous. The magnetic nonlinear needs the physical velocity `u` (supplied — from the velocity step in the full solver). Runs on Array (locally testable) and CuArray.

**Tech Stack:** Julia, KernelAbstractions (via the composed kernels), `src/gpu/*` kernels from Phases 5h/5c/5d.

---

## Background — the pieces + the CPU reference

```julia
# 5h — magnetic nonlinear (induction ∇×(u×B); returns both nl_tor and nl_pol)
gpu_magnetic_nonlinear!(nl_tor_r, nl_tor_i, nl_pol_r, nl_pol_i, B_tor_r, B_tor_i, B_pol_r, B_pol_i,
    u_r, u_θ, u_φ, config, d1, d2, lfac, rinv, rinv2, rscale, lmax, bw)

# 5c — CNAB2 RHS: rr = inv_dt·u + 1.5·nl − 0.5·nl_prev + linear_weight·(lin·u)
gpu_build_rhs_cnab2!(rr, ri, ur, ui, nr_, ni_, pr, pi_, lin_batched, inv_dt, linear_weight, bw)

# 5d — implicit solve: set BC rows then in-place batched banded solve
gpu_implicit_solve_field!(x_r, x_i, lu_batched, bc_in_r, bc_in_i, bc_out_r, bc_out_i, bw)
```

CPU reference: `apply_magnetic_toroidal_implicit_update!` / `apply_magnetic_poloidal_implicit_update!` (`src/physics/magnetic/solver.jl`), the `timestepper isa CNAB2`, `adm_set === nothing` (insulating) branches (toroidal lines 296-324, poloidal lines 418-440), plus `_magnetic_toroidal_inner_bc_increment` (lines 101-139):

- **Mass coefficient is 1** (magnetic diffusivity η, baked into the caller's `inv_dt = 1/dt` and `lin = L`; the velocity's `E` factor is absent).
- **Toroidal insulating BC:** RHS row 1 = `mag_bc_inner − prev_bc_inner`. For `CONTINUITY_MAG` modes that is `(−nl_pol[ICB]) − (−prev_nl_pol[ICB]) = −nl_pol[ICB] + prev_nl_pol[ICB]`; otherwise 0. RHS row `nr` (outer/CMB) = 0. (`ICB` = outer-core radial index **1**.)
- **Poloidal insulating BC:** fully homogeneous (RHS rows 1 and `nr` = 0); the l-dependent insulating stencils live in the matrix.
- **Rollover:** `prev_nl ← this step's nl` (done by the solver's history roll).

**Template:** `src/gpu/velocity_step.jl` (5k). Same ORDERING INVARIANT: the nonlinear, both `build_rhs` calls, AND the `CONTINUITY_MAG` BC (which reads the poloidal nl + the OLD poloidal prev_nl) all read state that must not be overwritten until step 4.

**GPU layout:** dense `(nl, nm, nr)`; `nl_pol[:, :, 1]` is the ICB slice (a `(nl,nm)` array).

---

## Task 1: `gpu_magnetic_field_step!`

**Files:**
- Create: `src/gpu/magnetic_step.jl`
- Modify: `src/GeoDynamo.jl` (add `include("gpu/magnetic_step.jl")` after the `gpu/inner_core.jl` include; add `export gpu_magnetic_field_step!`)
- Test: `test/gpu_phase5m_magnetic_step.jl`

- [ ] **Step 1: Write the failing test**

Create `test/gpu_phase5m_magnetic_step.jl`:

```julia
using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5m — Magnetic Field CNAB2 Step (insulating, tor + pol)" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 24, nlon = 48, nr = 4)
    nl, nm, nr = cfg.lmax + 1, cfg.mmax + 1, 4
    nlat, nlon = cfg.nlat, cfg.nlon
    bw = 2
    rng = MersenneTwister(13)

    function band(N, b; seed)
        r = MersenneTwister(seed); d = zeros(2b+1, N)
        for j in 1:N, i in max(1,j-b):min(N,j+b); d[b+1+i-j, j] = rand(r) - 0.5; end
        d
    end
    d1 = band(nr, bw; seed = 1); d2 = band(nr, bw; seed = 2)
    lfac = Float64[l*(l+1) for l in 0:cfg.lmax]
    rinv = [1.0/(0.5 + 0.1k) for k in 1:nr]; rinv2 = rinv .^ 2; rscale = copy(rinv)

    function batched(seed)
        a = zeros(2bw+1, nr, nl); r = MersenneTwister(seed)
        for li in 1:nl, j in 1:nr, i in max(1,j-bw):min(nr,j+bw); a[bw+1+i-j, j, li] = rand(r) - 0.5; end
        for li in 1:nl, j in 1:nr; a[bw+1, j, li] += 5.0; end
        a
    end
    lin_tor = batched(10); lu_tor = batched(11); lin_pol = batched(20); lu_pol = batched(21)

    u_r = rand(rng, nlat, nlon, nr) .- 0.5
    u_θ = rand(rng, nlat, nlon, nr) .- 0.5
    u_φ = rand(rng, nlat, nlon, nr) .- 0.5

    inv_dt = 1.0 / 5.0e-4          # mass_coeff(1) / dt
    linear_weight = 0.5

    mk() = (a = zeros(nl, nm, nr); for mi in 1:nm, li in mi:nl, r in 1:nr; a[li,mi,r] = rand(rng) - 0.5; end; a)
    bt_r0 = mk(); bt_i0 = mk(); bp_r0 = mk(); bp_i0 = mk()
    pnt_r0 = mk(); pnt_i0 = mk(); pnp_r0 = mk(); pnp_i0 = mk()

    nlops = (; d1, d2, lfac, rinv, rinv2, rscale)

    function run_gpu(arch_dev, continuity)
        d = arch_dev === :cpu ? identity : (x -> GeoDynamo.on_architecture(GPU(), x))
        tor = (; spec_r = d(copy(bt_r0)), spec_i = d(copy(bt_i0)),
                 prev_nl_r = d(copy(pnt_r0)), prev_nl_i = d(copy(pnt_i0)), lin = d(lin_tor), lu = d(lu_tor))
        pol = (; spec_r = d(copy(bp_r0)), spec_i = d(copy(bp_i0)),
                 prev_nl_r = d(copy(pnp_r0)), prev_nl_i = d(copy(pnp_i0)), lin = d(lin_pol), lu = d(lu_pol))
        nlo = arch_dev === :cpu ? nlops :
            (; d1 = d(d1), d2 = d(d2), lfac = d(lfac), rinv = d(rinv), rinv2 = d(rinv2), rscale = d(rscale))
        GeoDynamo.gpu_magnetic_field_step!(tor, pol, d(copy(u_r)), d(copy(u_θ)), d(copy(u_φ)),
            cfg, nlo, inv_dt, linear_weight, cfg.lmax, bw; continuity_mag = continuity)
        return tor, pol
    end

    # manual chain on CPU for the given continuity flag
    function manual(continuity)
        bt_r = copy(bt_r0); bt_i = copy(bt_i0); bp_r = copy(bp_r0); bp_i = copy(bp_i0)
        pnt_r = copy(pnt_r0); pnt_i = copy(pnt_i0); pnp_r = copy(pnp_r0); pnp_i = copy(pnp_i0)
        nlt_r = similar(bt_r); nlt_i = similar(bt_i); nlp_r = similar(bp_r); nlp_i = similar(bp_i)
        GeoDynamo.gpu_magnetic_nonlinear!(nlt_r, nlt_i, nlp_r, nlp_i, bt_r, bt_i, bp_r, bp_i,
            copy(u_r), copy(u_θ), copy(u_φ), cfg, d1, d2, lfac, rinv, rinv2, rscale, cfg.lmax, bw)
        bcin_r = zeros(nl, nm); bcin_i = zeros(nl, nm); z = zeros(nl, nm)
        if continuity
            bcin_r .= .-nlp_r[:, :, 1] .+ pnp_r[:, :, 1]
            bcin_i .= .-nlp_i[:, :, 1] .+ pnp_i[:, :, 1]
        end
        rt_r = similar(bt_r); rt_i = similar(bt_i); rp_r = similar(bp_r); rp_i = similar(bp_i)
        GeoDynamo.gpu_build_rhs_cnab2!(rt_r, rt_i, bt_r, bt_i, nlt_r, nlt_i, pnt_r, pnt_i, lin_tor, inv_dt, linear_weight, bw)
        GeoDynamo.gpu_implicit_solve_field!(rt_r, rt_i, lu_tor, bcin_r, bcin_i, z, z, bw)
        GeoDynamo.gpu_build_rhs_cnab2!(rp_r, rp_i, bp_r, bp_i, nlp_r, nlp_i, pnp_r, pnp_i, lin_pol, inv_dt, linear_weight, bw)
        GeoDynamo.gpu_implicit_solve_field!(rp_r, rp_i, lu_pol, z, z, z, z, bw)
        return (rt_r, rt_i, rp_r, rp_i, nlt_r, nlt_i, nlp_r, nlp_i)
    end

    @testset "step == manual chain, no continuity (exact) [LOCAL]" begin
        tor, pol = run_gpu(:cpu, false)
        rt_r, rt_i, rp_r, rp_i, nlt_r, nlt_i, nlp_r, nlp_i = manual(false)
        @test tor.spec_r == rt_r && tor.spec_i == rt_i
        @test pol.spec_r == rp_r && pol.spec_i == rp_i
        @test tor.prev_nl_r == nlt_r && tor.prev_nl_i == nlt_i
        @test pol.prev_nl_r == nlp_r && pol.prev_nl_i == nlp_i
        @test all(isfinite, tor.spec_r) && all(isfinite, pol.spec_r)
    end

    @testset "step == manual chain, CONTINUITY_MAG (exact) [LOCAL]" begin
        tor, pol = run_gpu(:cpu, true)
        rt_r, rt_i, rp_r, rp_i, nlt_r, nlt_i, nlp_r, nlp_i = manual(true)
        @test tor.spec_r == rt_r && tor.spec_i == rt_i
        @test pol.spec_r == rp_r && pol.spec_i == rp_i
        @test tor.prev_nl_r == nlt_r && pol.prev_nl_r == nlp_r
    end

    @testset "continuity changes the toroidal result [LOCAL]" begin
        # sanity: the CONTINUITY_MAG BC actually does something (toroidal differs)
        t0, _ = run_gpu(:cpu, false)
        t1, _ = run_gpu(:cpu, true)
        @test t0.spec_r != t1.spec_r
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5m gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            ctor, cpol = run_gpu(:cpu, true)
            gtor, gpol = run_gpu(:gpu, true)
            @test gtor.spec_r isa CUDA.CuArray
            @test gpol.spec_r isa CUDA.CuArray
            @test isapprox(Array(gtor.spec_r), ctor.spec_r; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gtor.spec_i), ctor.spec_i; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gpol.spec_r), cpol.spec_r; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gpol.spec_i), cpol.spec_i; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gtor.prev_nl_r), ctor.prev_nl_r; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gpol.prev_nl_i), cpol.prev_nl_i; atol = 1e-9, rtol = 1e-8)
        end
    end
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5m_magnetic_step.jl")'
```
Expected: FAIL — `UndefVarError: gpu_magnetic_field_step!` not defined.

- [ ] **Step 3: Write the implementation**

Create `src/gpu/magnetic_step.jl`:

```julia
# =============================================================================
# GPU Phase 5m — one magnetic field CNAB2 timestep (toroidal + poloidal,
# INSULATING inner core), composing the verified pieces: magnetic nonlinear
# (5h, induction ∇×(u×B), returns BOTH nl_tor and nl_pol) → CNAB2 RHS (5c,
# mass_coeff=1) → implicit solve (5d) for tor then pol → field update + nl_prev
# rollover.  Mirrors apply_magnetic_toroidal/poloidal_implicit_update! (insulating
# CNAB2 branch).  No new kernels.  The toroidal inner boundary optionally takes
# the CONTINUITY_MAG increment −nl_pol[ICB] + prev_nl_pol[ICB]; the poloidal is
# homogeneous.  The conducting-inner-core path is Phase 5m2.  Runs on Array +
# CuArray.  (Per-call scratch — Phase-6 may cache.)
#
# Bundles:  tor/pol :: (; spec_r, spec_i, prev_nl_r, prev_nl_i, lin, lu)
#           nlops   :: (; d1, d2, lfac, rinv, rinv2, rscale)
# =============================================================================

"""
    gpu_magnetic_field_step!(tor, pol, u_r, u_θ, u_φ, config, nlops, inv_dt, linear_weight,
                             lmax, bw; continuity_mag=false) -> nothing

Advance the magnetic field one CNAB2 step (insulating inner core).  `tor`/`pol`
are NamedTuple bundles `(; spec_r, spec_i, prev_nl_r, prev_nl_i, lin, lu)`; on exit
`*.spec_*` is the updated field and `*.prev_nl_*` holds THIS step's nonlinear term.
`u_*` is the physical velocity (supplied — from the velocity step).  `nlops` carries
the magnetic nonlinear/curl operators.  `inv_dt = 1/dt` and `lin` carry the magnetic
mass coefficient (η is in `lin`); `linear_weight = 1−θ`.

`continuity_mag=true` applies the `CONTINUITY_MAG` toroidal inner-boundary coupling:
the toroidal inner RHS row is set to `−nl_pol[ICB] + prev_nl_pol[ICB]` (ICB = radial
index 1), computed from the just-formed poloidal nonlinear and the OLD poloidal
history.  Otherwise the toroidal inner row is 0.  The poloidal is fully homogeneous.

ORDERING INVARIANT (as in `gpu_velocity_field_step!`): the nonlinear, both
`build_rhs` calls, and the `CONTINUITY_MAG` BC all read OLD state (`*.spec_*`,
`pol.prev_nl_*`); the field/history are overwritten ONLY after every such read.
All arrays on the same backend.
"""
function gpu_magnetic_field_step!(tor, pol, u_r, u_θ, u_φ, config, nlops,
        inv_dt, linear_weight, lmax::Int, bw::Int; continuity_mag::Bool = false)
    nl, nm, _ = size(tor.spec_r)
    # 1. magnetic nonlinear (5h): nl_tor/nl_pol from the OLD B (tor/pol spec).
    nlt_r = similar(tor.spec_r); nlt_i = similar(tor.spec_i)   # Phase-6: workspace
    nlp_r = similar(pol.spec_r); nlp_i = similar(pol.spec_i)
    gpu_magnetic_nonlinear!(nlt_r, nlt_i, nlp_r, nlp_i,
        tor.spec_r, tor.spec_i, pol.spec_r, pol.spec_i, u_r, u_θ, u_φ, config,
        nlops.d1, nlops.d2, nlops.lfac, nlops.rinv, nlops.rinv2, nlops.rscale, lmax, bw)

    # 2. toroidal BC rows. Inner = CONTINUITY_MAG increment −nl_pol[ICB]+prev_nl_pol[ICB]
    #    (computed from nl_pol + OLD pol.prev_nl, both read before any overwrite) or 0;
    #    outer = 0. zin/zout/z are (nl,nm) on the same backend.
    z = similar(tor.spec_r, nl, nm); fill!(z, zero(eltype(tor.spec_r)))
    bcin_r = similar(z); bcin_i = similar(z)
    if continuity_mag
        @views bcin_r .= .-nlp_r[:, :, 1] .+ pol.prev_nl_r[:, :, 1]
        @views bcin_i .= .-nlp_i[:, :, 1] .+ pol.prev_nl_i[:, :, 1]
    else
        fill!(bcin_r, zero(eltype(bcin_r))); fill!(bcin_i, zero(eltype(bcin_i)))
    end

    # 3. toroidal CNAB2 RHS (5c) from OLD tor spec, then implicit solve (5d).
    rt_r = similar(tor.spec_r); rt_i = similar(tor.spec_i)     # rt ≠ tor.spec — build_rhs reads tor.spec
    gpu_build_rhs_cnab2!(rt_r, rt_i, tor.spec_r, tor.spec_i, nlt_r, nlt_i,
        tor.prev_nl_r, tor.prev_nl_i, tor.lin, inv_dt, linear_weight, bw)
    gpu_implicit_solve_field!(rt_r, rt_i, tor.lu, bcin_r, bcin_i, z, z, bw)

    # 4. poloidal CNAB2 RHS (5c) from OLD pol spec, homogeneous solve (5d).
    rp_r = similar(pol.spec_r); rp_i = similar(pol.spec_i)     # rp ≠ pol.spec — build_rhs reads pol.spec
    gpu_build_rhs_cnab2!(rp_r, rp_i, pol.spec_r, pol.spec_i, nlp_r, nlp_i,
        pol.prev_nl_r, pol.prev_nl_i, pol.lin, inv_dt, linear_weight, bw)
    gpu_implicit_solve_field!(rp_r, rp_i, pol.lu, z, z, z, z, bw)

    # 5. update the fields (AFTER every read of old spec / old pol.prev_nl).
    tor.spec_r .= rt_r; tor.spec_i .= rt_i
    pol.spec_r .= rp_r; pol.spec_i .= rp_i
    # 6. roll histories.
    tor.prev_nl_r .= nlt_r; tor.prev_nl_i .= nlt_i
    pol.prev_nl_r .= nlp_r; pol.prev_nl_i .= nlp_i
    return nothing
end
```

Modify `src/GeoDynamo.jl` — add the include immediately after `include("gpu/inner_core.jl")`:

```julia
include("gpu/magnetic_step.jl")
```

And add the export after `export gpu_pack_inner_core, gpu_inner_core_history_flux!, gpu_reconstruct_inner_core!`:

```julia
export gpu_magnetic_field_step!
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5m_magnetic_step.jl")'
```
Expected: the three `[LOCAL]` testsets PASS (no-continuity exact `==`, CONTINUITY_MAG exact `==`, continuity-changes-result); the `[GPU-BOX]` testset shows 1 Broken (`@test_skip`).

- [ ] **Step 5: Verify the module still loads**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using GeoDynamo; println("LOAD OK")'
```
Expected: `LOAD OK`.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/magnetic_step.jl src/GeoDynamo.jl test/gpu_phase5m_magnetic_step.jl
git commit -m "feat(gpu): Phase 5m magnetic field CNAB2 step (insulating, tor + pol)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Register the test + regression check

**Files:**
- Modify: `test/runtests.jl` (add the Phase 5m entry after the Phase 5l entry)

- [ ] **Step 1: Add the test to the suite**

In `test/runtests.jl`, after `"gpu_phase5l_inner_core.jl"`, add (same indentation):

```julia
    "gpu_phase5m_magnetic_step.jl",
```

- [ ] **Step 2: Confirm the new test still passes in isolation**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5m_magnetic_step.jl")' > /tmp/phase5m.log 2>&1; echo "exit=$?"; tail -20 /tmp/phase5m.log
```
Expected: `exit=0`, the three `[LOCAL]` testsets pass, 1 Broken for the GPU-box gate.

- [ ] **Step 3: Confirm the allocation guards still pass**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/allocation_runtime_checks.jl")' > /tmp/allocguards.log 2>&1; echo "exit=$?"; tail -8 /tmp/allocguards.log
```
Expected: `exit=0`, 39/39 unchanged.

- [ ] **Step 4: Commit**

```bash
git add test/runtests.jl
git commit -m "test(gpu): register Phase 5m magnetic step in suite

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** magnetic nonlinear ONCE (5h) → tor RHS+solve → pol RHS+solve → field update → rollover ✓; mass coefficient 1 (caller's `inv_dt=1/dt`, `lin=L`) ✓; toroidal optional `CONTINUITY_MAG` inner BC `−nl_pol[ICB]+prev_nl_pol[ICB]` (combined incremental, ICB=index 1) ✓; toroidal outer + poloidal both homogeneous ✓; ORDERING INVARIANT (nl + both build_rhs + the BC read old state; overwrite only at step 5) ✓; runs on Array + CuArray ✓.

**Placeholder scan:** none.

**Type consistency:** `gpu_magnetic_field_step!` signature identical across impl + test (`run_gpu`); bundle fields `spec_r/spec_i/prev_nl_r/prev_nl_i/lin/lu` and `nlops.{d1,d2,lfac,rinv,rinv2,rscale}` match; the composed-kernel arg orders match the Background signatures; `nlp_r[:,:,1]`/`pol.prev_nl_r[:,:,1]` are the ICB slices.

**Ordering:** the `CONTINUITY_MAG` BC reads `nlp_r` (this step) and `pol.prev_nl_r` (OLD) — both before step 6 overwrites `pol.prev_nl_r`; the `.= rt`/`.= rp` copies and the rollover happen only at steps 5–6.

**Continuity sign:** `−nl_pol[ICB] + prev_nl_pol[ICB]` = `mag_bc_inner − prev_bc_inner` with `mag_bc_inner=−nl_pol[ICB]`, `prev_bc_inner=−prev_nl_pol[ICB]` (matches `_magnetic_toroidal_inner_bc_increment` + the solver's incremental row).
