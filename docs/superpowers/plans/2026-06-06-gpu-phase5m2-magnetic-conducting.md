# GPU Phase 5m2 — Magnetic Field Step: Conducting Inner Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `gpu_magnetic_field_step!` (Phase 5m, insulating) with the conducting-inner-core CNAB2 path — the inner boundary RHS becomes the inner-core history flux `φ0` (5l) and the inner-core radial profile is reconstructed (5l) after the outer-core solve — bit-exact against a manual chain. The insulating path stays byte-identical.

**Architecture:** Modify `src/gpu/magnetic_step.jl`. Add an optional `ic` keyword bundle to `gpu_magnetic_field_step!`. When `ic === nothing` (default) the function behaves exactly as Phase 5m (insulating, optional `CONTINUITY_MAG`). When `ic` is supplied (conducting): the toroidal/poloidal inner BC is the `φ0` history flux from `gpu_inner_core_history_flux!` (5l) instead of the continuity increment; after the outer-core solve and field update, the inner-core spectral state is reconstructed via `gpu_reconstruct_inner_core!` (5l) using the new outer-core ICB value. No new kernels. Runs on Array (locally testable) and CuArray.

**Tech Stack:** Julia, the Phase 5h/5c/5d/5l GPU kernels.

---

## Background — the CPU reference (read, do not modify)

`apply_magnetic_toroidal_implicit_update!` / `apply_magnetic_poloidal_implicit_update!` conducting CNAB2 branch (`src/physics/magnetic/solver.jl:262-294` tor, `385-416` pol):

```
φ0 = _magnetic_conducting_history_flux(magnetic, toroidal_ic/poloidal_ic, adm)   # 5l history flux per mode
solver_build_rhs_cnab2!(...)                                                       # same RHS (mass_coeff 1)
solver_solve_magnetic_implicit_step!(..., mag_bc_inner = φ0_real, mag_bc_inner_imag = φ0_imag)
    # inner BC row = φ0 (NOT incremental — no prev subtraction); outer BC = 0
_magnetic_conducting_reconstruct!(oc_spec, ic_spec, adm)                           # 5l reconstruct
    # g = oc_spec[ICB] = the NEW outer-core value at radial index 1; writes ic_spec[1..Nic]
```

Key facts:
- The conducting path **supersedes** the `CONTINUITY_MAG` increment (CPU comment: "We supersede `_magnetic_toroidal_inner_bc_increment` here"). So when `ic` is supplied, `continuity_mag` is ignored.
- Inner BC = `φ0` directly (no `bc − prev_bc` incremental subtraction, unlike insulating CONTINUITY_MAG). Outer BC = 0.
- The reconstruct's `g` is the **NEW** outer-core spec at ICB (radial index 1), read **after** the outer-core solve + field update.
- `toroidal_ic`/`poloidal_ic` are the inner-core spectral scalars, shape `(nl, nm, Nic)` (Nic = inner-core radial points, generally ≠ nr).

The 5l GPU pieces (already on `main`):
```julia
gpu_inner_core_history_flux!(φ0_r, φ0_i, S_old_r, S_old_i, ic_adm)              # φ0 (nl,nm) from S_old (nl,nm,Nic)
gpu_reconstruct_inner_core!(S_new_r, S_new_i, S_old_r, S_old_i, g_r, g_i, ic_adm) # S_new (nl,nm,Nic); g (nl,nm)
# ic_adm = gpu_pack_inner_core(adm, nl, arch) = (; lin_ic, lu_ic, d1_top, inv_dt, weight, Nic, bw)
```
`gpu_reconstruct_inner_core!` requires `S_new` not alias `S_old` (use scratch).

**ORDERING:** the history flux reads OLD `ic.*_ic_*`; the reconstruct reads OLD `ic.*_ic_*` AND the NEW outer-core spec ICB slice. So: compute φ0 (from old ic state) BEFORE the solve; reconstruct (into scratch, then copy to ic state) AFTER the field update. The `g` slice `tor.spec_r[:, :, 1]` (no `@view`) materializes a `(nl,nm)` copy of the NEW spec.

---

## Task 1: Add the conducting path to `gpu_magnetic_field_step!`

**Files:**
- Modify: `src/gpu/magnetic_step.jl`
- Test: `test/gpu_phase5m2_magnetic_conducting.jl`

- [ ] **Step 1: Write the failing test**

Create `test/gpu_phase5m2_magnetic_conducting.jl`:

```julia
using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5m2 — Magnetic Conducting Inner Core Step" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 24, nlon = 48, nr = 4)
    nl, nm, nr = cfg.lmax + 1, cfg.mmax + 1, 4
    Nic = 5
    nlat, nlon = cfg.nlat, cfg.nlon
    bw = 2
    rng = MersenneTwister(17)
    dt = 5.0e-4; theta = 0.5

    function band(N, b; seed)
        r = MersenneTwister(seed); d = zeros(2b+1, N)
        for j in 1:N, i in max(1,j-b):min(N,j+b); d[b+1+i-j, j] = rand(r) - 0.5; end
        d
    end
    d1 = band(nr, bw; seed = 1); d2 = band(nr, bw; seed = 2)
    lfac = Float64[l*(l+1) for l in 0:cfg.lmax]
    rinv = [1.0/(0.5 + 0.1k) for k in 1:nr]; rinv2 = rinv .^ 2; rscale = copy(rinv)

    function batched(N, seed)
        a = zeros(2bw+1, N, nl); r = MersenneTwister(seed)
        for li in 1:nl, j in 1:N, i in max(1,j-bw):min(N,j+bw); a[bw+1+i-j, j, li] = rand(r) - 0.5; end
        for li in 1:nl, j in 1:N; a[bw+1, j, li] += 5.0; end
        a
    end
    lin_tor = batched(nr, 10); lu_tor = batched(nr, 11); lin_pol = batched(nr, 20); lu_pol = batched(nr, 21)

    # synthetic InnerCoreAdmittance for tor + pol (degrees 1..lmax)
    function make_adm(seed0)
        facs = GeoDynamo.BandedLU{Float64}[]; lins = GeoDynamo.BandedMatrix{Float64}[]
        lookup = Dict{Int,Int}()
        for (idx, l) in enumerate(1:cfg.lmax)
            M = GeoDynamo.BandedMatrix{Float64}(
                (d = band(Nic, bw; seed = seed0 + l); for j in 1:Nic; d[bw+1, j] += 5.0; end; d), bw, Nic)
            push!(facs, GeoDynamo.factorize_banded(M))
            push!(lins, GeoDynamo.BandedMatrix{Float64}(band(Nic, bw; seed = seed0 + 100 + l), bw, Nic))
            lookup[l] = idx
        end
        GeoDynamo.InnerCoreAdmittance{Float64}(facs, rand(rng, cfg.lmax), rand(rng, Nic) .- 0.5,
                                               lookup, Nic, lins, dt, theta)
    end
    adm_tor = make_adm(300); adm_pol = make_adm(400)

    u_r = rand(rng, nlat, nlon, nr) .- 0.5; u_θ = rand(rng, nlat, nlon, nr) .- 0.5; u_φ = rand(rng, nlat, nlon, nr) .- 0.5
    inv_dt = 1.0 / dt; linear_weight = 1.0 - theta

    mk(N) = (a = zeros(nl, nm, N); for mi in 1:nm, li in mi:nl, r in 1:N; a[li,mi,r] = rand(rng) - 0.5; end; a)
    bt_r0 = mk(nr); bt_i0 = mk(nr); bp_r0 = mk(nr); bp_i0 = mk(nr)
    pnt_r0 = mk(nr); pnt_i0 = mk(nr); pnp_r0 = mk(nr); pnp_i0 = mk(nr)
    itr0 = mk(Nic); iti0 = mk(Nic); ipr0 = mk(Nic); ipi0 = mk(Nic)   # inner-core tor/pol state

    nlops = (; d1, d2, lfac, rinv, rinv2, rscale)

    @testset "conducting step == manual chain (exact) [LOCAL]" begin
        tor_adm = GeoDynamo.gpu_pack_inner_core(adm_tor, nl, CPU())
        pol_adm = GeoDynamo.gpu_pack_inner_core(adm_pol, nl, CPU())
        tor = (; spec_r = copy(bt_r0), spec_i = copy(bt_i0), prev_nl_r = copy(pnt_r0), prev_nl_i = copy(pnt_i0),
                 lin = lin_tor, lu = lu_tor)
        pol = (; spec_r = copy(bp_r0), spec_i = copy(bp_i0), prev_nl_r = copy(pnp_r0), prev_nl_i = copy(pnp_i0),
                 lin = lin_pol, lu = lu_pol)
        ic = (; tor_adm = tor_adm, pol_adm = pol_adm,
                tor_ic_r = copy(itr0), tor_ic_i = copy(iti0), pol_ic_r = copy(ipr0), pol_ic_i = copy(ipi0))
        GeoDynamo.gpu_magnetic_field_step!(tor, pol, copy(u_r), copy(u_θ), copy(u_φ), cfg, nlops,
                                           inv_dt, linear_weight, cfg.lmax, bw; ic = ic)

        # ---- manual chain ----
        bt_r = copy(bt_r0); bt_i = copy(bt_i0); bp_r = copy(bp_r0); bp_i = copy(bp_i0)
        pnt_r = copy(pnt_r0); pnt_i = copy(pnt_i0); pnp_r = copy(pnp_r0); pnp_i = copy(pnp_i0)
        nlt_r = similar(bt_r); nlt_i = similar(bt_i); nlp_r = similar(bp_r); nlp_i = similar(bp_i)
        GeoDynamo.gpu_magnetic_nonlinear!(nlt_r, nlt_i, nlp_r, nlp_i, bt_r, bt_i, bp_r, bp_i,
            copy(u_r), copy(u_θ), copy(u_φ), cfg, d1, d2, lfac, rinv, rinv2, rscale, cfg.lmax, bw)
        φt_r = zeros(nl, nm); φt_i = zeros(nl, nm); φp_r = zeros(nl, nm); φp_i = zeros(nl, nm)
        GeoDynamo.gpu_inner_core_history_flux!(φt_r, φt_i, copy(itr0), copy(iti0), tor_adm)
        GeoDynamo.gpu_inner_core_history_flux!(φp_r, φp_i, copy(ipr0), copy(ipi0), pol_adm)
        z = zeros(nl, nm)
        rt_r = similar(bt_r); rt_i = similar(bt_i); rp_r = similar(bp_r); rp_i = similar(bp_i)
        GeoDynamo.gpu_build_rhs_cnab2!(rt_r, rt_i, bt_r, bt_i, nlt_r, nlt_i, pnt_r, pnt_i, lin_tor, inv_dt, linear_weight, bw)
        GeoDynamo.gpu_implicit_solve_field!(rt_r, rt_i, lu_tor, φt_r, φt_i, z, z, bw)
        GeoDynamo.gpu_build_rhs_cnab2!(rp_r, rp_i, bp_r, bp_i, nlp_r, nlp_i, pnp_r, pnp_i, lin_pol, inv_dt, linear_weight, bw)
        GeoDynamo.gpu_implicit_solve_field!(rp_r, rp_i, lu_pol, φp_r, φp_i, z, z, bw)
        # reconstruct from the NEW outer-core ICB value (g = rt[:,:,1] / rp[:,:,1])
        ntr_r = similar(itr0); ntr_i = similar(iti0); npr_r = similar(ipr0); npr_i = similar(ipi0)
        GeoDynamo.gpu_reconstruct_inner_core!(ntr_r, ntr_i, copy(itr0), copy(iti0), rt_r[:, :, 1], rt_i[:, :, 1], tor_adm)
        GeoDynamo.gpu_reconstruct_inner_core!(npr_r, npr_i, copy(ipr0), copy(ipi0), rp_r[:, :, 1], rp_i[:, :, 1], pol_adm)

        @test tor.spec_r == rt_r && tor.spec_i == rt_i
        @test pol.spec_r == rp_r && pol.spec_i == rp_i
        @test tor.prev_nl_r == nlt_r && tor.prev_nl_i == nlt_i
        @test pol.prev_nl_r == nlp_r && pol.prev_nl_i == nlp_i
        @test ic.tor_ic_r == ntr_r && ic.tor_ic_i == ntr_i
        @test ic.pol_ic_r == npr_r && ic.pol_ic_i == npr_i
        @test all(isfinite, ic.tor_ic_r) && all(isfinite, ic.pol_ic_r)
    end

    @testset "insulating path unchanged when ic=nothing [LOCAL]" begin
        # ic=nothing must reproduce the Phase-5m insulating result exactly
        tor = (; spec_r = copy(bt_r0), spec_i = copy(bt_i0), prev_nl_r = copy(pnt_r0), prev_nl_i = copy(pnt_i0),
                 lin = lin_tor, lu = lu_tor)
        pol = (; spec_r = copy(bp_r0), spec_i = copy(bp_i0), prev_nl_r = copy(pnp_r0), prev_nl_i = copy(pnp_i0),
                 lin = lin_pol, lu = lu_pol)
        GeoDynamo.gpu_magnetic_field_step!(tor, pol, copy(u_r), copy(u_θ), copy(u_φ), cfg, nlops,
                                           inv_dt, linear_weight, cfg.lmax, bw)   # no ic, no continuity
        # manual insulating (no continuity, all-zero BC)
        bt_r = copy(bt_r0); bt_i = copy(bt_i0); bp_r = copy(bp_r0); bp_i = copy(bp_i0)
        nlt_r = similar(bt_r); nlt_i = similar(bt_i); nlp_r = similar(bp_r); nlp_i = similar(bp_i)
        GeoDynamo.gpu_magnetic_nonlinear!(nlt_r, nlt_i, nlp_r, nlp_i, bt_r, bt_i, bp_r, bp_i,
            copy(u_r), copy(u_θ), copy(u_φ), cfg, d1, d2, lfac, rinv, rinv2, rscale, cfg.lmax, bw)
        z = zeros(nl, nm)
        rt_r = similar(bt_r); rt_i = similar(bt_i); rp_r = similar(bp_r); rp_i = similar(bp_i)
        GeoDynamo.gpu_build_rhs_cnab2!(rt_r, rt_i, bt_r, bt_i, nlt_r, nlt_i, copy(pnt_r0), copy(pnt_i0), lin_tor, inv_dt, linear_weight, bw)
        GeoDynamo.gpu_implicit_solve_field!(rt_r, rt_i, lu_tor, z, z, z, z, bw)
        GeoDynamo.gpu_build_rhs_cnab2!(rp_r, rp_i, bp_r, bp_i, nlp_r, nlp_i, copy(pnp_r0), copy(pnp_i0), lin_pol, inv_dt, linear_weight, bw)
        GeoDynamo.gpu_implicit_solve_field!(rp_r, rp_i, lu_pol, z, z, z, z, bw)
        @test tor.spec_r == rt_r && pol.spec_r == rp_r
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5m2 gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            # CPU reference (conducting)
            ctor_adm = GeoDynamo.gpu_pack_inner_core(adm_tor, nl, CPU())
            cpol_adm = GeoDynamo.gpu_pack_inner_core(adm_pol, nl, CPU())
            ctor = (; spec_r = copy(bt_r0), spec_i = copy(bt_i0), prev_nl_r = copy(pnt_r0), prev_nl_i = copy(pnt_i0),
                      lin = lin_tor, lu = lu_tor)
            cpol = (; spec_r = copy(bp_r0), spec_i = copy(bp_i0), prev_nl_r = copy(pnp_r0), prev_nl_i = copy(pnp_i0),
                      lin = lin_pol, lu = lu_pol)
            cic = (; tor_adm = ctor_adm, pol_adm = cpol_adm,
                     tor_ic_r = copy(itr0), tor_ic_i = copy(iti0), pol_ic_r = copy(ipr0), pol_ic_i = copy(ipi0))
            GeoDynamo.gpu_magnetic_field_step!(ctor, cpol, copy(u_r), copy(u_θ), copy(u_φ), cfg, nlops,
                                               inv_dt, linear_weight, cfg.lmax, bw; ic = cic)
            d(x) = GeoDynamo.on_architecture(GPU(), x)
            gtor_adm = GeoDynamo.gpu_pack_inner_core(adm_tor, nl, GPU())
            gpol_adm = GeoDynamo.gpu_pack_inner_core(adm_pol, nl, GPU())
            gnlops = (; d1 = d(d1), d2 = d(d2), lfac = d(lfac), rinv = d(rinv), rinv2 = d(rinv2), rscale = d(rscale))
            gtor = (; spec_r = d(copy(bt_r0)), spec_i = d(copy(bt_i0)), prev_nl_r = d(copy(pnt_r0)), prev_nl_i = d(copy(pnt_i0)),
                      lin = d(lin_tor), lu = d(lu_tor))
            gpol = (; spec_r = d(copy(bp_r0)), spec_i = d(copy(bp_i0)), prev_nl_r = d(copy(pnp_r0)), prev_nl_i = d(copy(pnp_i0)),
                      lin = d(lin_pol), lu = d(lu_pol))
            gic = (; tor_adm = gtor_adm, pol_adm = gpol_adm,
                     tor_ic_r = d(copy(itr0)), tor_ic_i = d(copy(iti0)), pol_ic_r = d(copy(ipr0)), pol_ic_i = d(copy(ipi0)))
            GeoDynamo.gpu_magnetic_field_step!(gtor, gpol, d(copy(u_r)), d(copy(u_θ)), d(copy(u_φ)), cfg, gnlops,
                                               inv_dt, linear_weight, cfg.lmax, bw; ic = gic)
            @test gtor.spec_r isa CUDA.CuArray
            @test gic.tor_ic_r isa CUDA.CuArray
            @test isapprox(Array(gtor.spec_r), ctor.spec_r; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gpol.spec_i), cpol.spec_i; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gic.tor_ic_r), cic.tor_ic_r; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gic.pol_ic_i), cic.pol_ic_i; atol = 1e-9, rtol = 1e-8)
        end
    end
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5m2_magnetic_conducting.jl")'
```
Expected: FAIL — `gpu_magnetic_field_step!` has no `ic` keyword (`MethodError`/unsupported kwarg).

- [ ] **Step 3: Extend the implementation**

In `src/gpu/magnetic_step.jl`, replace the `gpu_magnetic_field_step!` function (and update the file header comment + docstring) to add the `ic` keyword and the conducting branch. Keep the insulating path (steps 1, 3–6) byte-identical when `ic === nothing`.

Replace the function signature line and BC-computation block. The new function:

```julia
function gpu_magnetic_field_step!(tor, pol, u_r, u_θ, u_φ, config, nlops,
        inv_dt, linear_weight, lmax::Int, bw::Int; continuity_mag::Bool = false, ic = nothing)
    nl, nm, _ = size(tor.spec_r)
    # 1. magnetic nonlinear (5h): nl_tor/nl_pol from the OLD B (tor/pol spec).
    nlt_r = similar(tor.spec_r); nlt_i = similar(tor.spec_i)   # Phase-6: workspace
    nlp_r = similar(pol.spec_r); nlp_i = similar(pol.spec_i)
    gpu_magnetic_nonlinear!(nlt_r, nlt_i, nlp_r, nlp_i,
        tor.spec_r, tor.spec_i, pol.spec_r, pol.spec_i, u_r, u_θ, u_φ, config,
        nlops.d1, nlops.d2, nlops.lfac, nlops.rinv, nlops.rinv2, nlops.rscale, lmax, bw)

    # 2. inner-boundary RHS rows for tor/pol. Three cases:
    #    - conducting (ic given): inner = φ0 history flux (5l) from the OLD inner-core state.
    #    - insulating + continuity_mag: tor inner = −nl_pol[ICB]+prev_nl_pol[ICB]; pol inner = 0.
    #    - insulating plain: both 0.  Outer = 0 always.
    z = similar(tor.spec_r, nl, nm); fill!(z, zero(eltype(tor.spec_r)))
    bcin_tor_r = similar(z); bcin_tor_i = similar(z)
    bcin_pol_r = similar(z); bcin_pol_i = similar(z)
    if ic !== nothing
        # conducting: φ0 supersedes the CONTINUITY_MAG increment (continuity_mag ignored).
        gpu_inner_core_history_flux!(bcin_tor_r, bcin_tor_i, ic.tor_ic_r, ic.tor_ic_i, ic.tor_adm)
        gpu_inner_core_history_flux!(bcin_pol_r, bcin_pol_i, ic.pol_ic_r, ic.pol_ic_i, ic.pol_adm)
    elseif continuity_mag
        @views bcin_tor_r .= .-nlp_r[:, :, 1] .+ pol.prev_nl_r[:, :, 1]
        @views bcin_tor_i .= .-nlp_i[:, :, 1] .+ pol.prev_nl_i[:, :, 1]
        fill!(bcin_pol_r, zero(eltype(bcin_pol_r))); fill!(bcin_pol_i, zero(eltype(bcin_pol_i)))
    else
        fill!(bcin_tor_r, zero(eltype(bcin_tor_r))); fill!(bcin_tor_i, zero(eltype(bcin_tor_i)))
        fill!(bcin_pol_r, zero(eltype(bcin_pol_r))); fill!(bcin_pol_i, zero(eltype(bcin_pol_i)))
    end

    # 3. toroidal CNAB2 RHS (5c) from OLD tor spec, then implicit solve (5d).
    rt_r = similar(tor.spec_r); rt_i = similar(tor.spec_i)     # rt ≠ tor.spec — build_rhs reads tor.spec
    gpu_build_rhs_cnab2!(rt_r, rt_i, tor.spec_r, tor.spec_i, nlt_r, nlt_i,
        tor.prev_nl_r, tor.prev_nl_i, tor.lin, inv_dt, linear_weight, bw)
    gpu_implicit_solve_field!(rt_r, rt_i, tor.lu, bcin_tor_r, bcin_tor_i, z, z, bw)

    # 4. poloidal CNAB2 RHS (5c) from OLD pol spec, implicit solve (5d).
    rp_r = similar(pol.spec_r); rp_i = similar(pol.spec_i)     # rp ≠ pol.spec — build_rhs reads pol.spec
    gpu_build_rhs_cnab2!(rp_r, rp_i, pol.spec_r, pol.spec_i, nlp_r, nlp_i,
        pol.prev_nl_r, pol.prev_nl_i, pol.lin, inv_dt, linear_weight, bw)
    gpu_implicit_solve_field!(rp_r, rp_i, pol.lu, bcin_pol_r, bcin_pol_i, z, z, bw)

    # 5. update the fields (AFTER every read of old spec / old pol.prev_nl — ORDERING INVARIANT).
    tor.spec_r .= rt_r; tor.spec_i .= rt_i
    pol.spec_r .= rp_r; pol.spec_i .= rp_i

    # 5b. conducting: reconstruct the inner-core profile from the NEW outer-core ICB value
    #     (g = spec[:,:,1], materialized). Reconstruct reads the OLD ic state (5l requires
    #     S_new ≠ S_old → scratch), then writes the new profile back into the ic state.
    if ic !== nothing
        gt_r = tor.spec_r[:, :, 1]; gt_i = tor.spec_i[:, :, 1]
        gp_r = pol.spec_r[:, :, 1]; gp_i = pol.spec_i[:, :, 1]
        nic_tr = similar(ic.tor_ic_r); nic_ti = similar(ic.tor_ic_i)
        nic_pr = similar(ic.pol_ic_r); nic_pi = similar(ic.pol_ic_i)
        gpu_reconstruct_inner_core!(nic_tr, nic_ti, ic.tor_ic_r, ic.tor_ic_i, gt_r, gt_i, ic.tor_adm)
        gpu_reconstruct_inner_core!(nic_pr, nic_pi, ic.pol_ic_r, ic.pol_ic_i, gp_r, gp_i, ic.pol_adm)
        ic.tor_ic_r .= nic_tr; ic.tor_ic_i .= nic_ti
        ic.pol_ic_r .= nic_pr; ic.pol_ic_i .= nic_pi
    end

    # 6. roll histories: prev_nl ← this step's nl (captured at step 1).
    tor.prev_nl_r .= nlt_r; tor.prev_nl_i .= nlt_i
    pol.prev_nl_r .= nlp_r; pol.prev_nl_i .= nlp_i
    return nothing
end
```

Update the file header comment (line ~9) and the docstring to describe the conducting path:
- Header: change "The conducting-inner-core path is Phase 5m2." to a sentence noting the conducting path is now supported via the `ic` keyword (φ0 history flux inner BC + post-solve reconstruct).
- Docstring: add a paragraph documenting the `ic` keyword bundle `(; tor_adm, pol_adm, tor_ic_r, tor_ic_i, pol_ic_r, pol_ic_i)`, that supplying it selects the conducting path (inner BC = φ0 history flux, NOT incremental; `continuity_mag` ignored), and that the inner-core spectral state `*_ic_*` is reconstructed in place after the outer-core solve.

- [ ] **Step 4: Run the test to verify it passes**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5m2_magnetic_conducting.jl")'
```
Expected: the two `[LOCAL]` testsets PASS (conducting exact `==` incl reconstructed IC state; insulating path unchanged); the `[GPU-BOX]` testset shows 1 Broken (`@test_skip`).

- [ ] **Step 5: Confirm the Phase 5m insulating test STILL passes (backward compatibility)**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5m_magnetic_step.jl")'
```
Expected: PASS (the `ic=nothing` default keeps the insulating path byte-identical).

- [ ] **Step 6: Verify the module still loads**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using GeoDynamo; println("LOAD OK")'
```
Expected: `LOAD OK`.

- [ ] **Step 7: Commit**

```bash
git add src/gpu/magnetic_step.jl test/gpu_phase5m2_magnetic_conducting.jl
git commit -m "feat(gpu): Phase 5m2 magnetic conducting inner-core step (φ0 BC + reconstruct)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Register the test + regression check

**Files:**
- Modify: `test/runtests.jl` (add the Phase 5m2 entry after the Phase 5m entry)

- [ ] **Step 1: Add the test to the suite**

In `test/runtests.jl`, after `"gpu_phase5m_magnetic_step.jl"`, add (same indentation):

```julia
    "gpu_phase5m2_magnetic_conducting.jl",
```

- [ ] **Step 2: Confirm the new test still passes in isolation**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5m2_magnetic_conducting.jl")' > /tmp/phase5m2.log 2>&1; echo "exit=$?"; tail -20 /tmp/phase5m2.log
```
Expected: `exit=0`, the two `[LOCAL]` testsets pass, 1 Broken for the GPU-box gate.

- [ ] **Step 3: Confirm the allocation guards still pass**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/allocation_runtime_checks.jl")' > /tmp/allocguards.log 2>&1; echo "exit=$?"; tail -8 /tmp/allocguards.log
```
Expected: `exit=0`, 39/39 unchanged.

- [ ] **Step 4: Commit**

```bash
git add test/runtests.jl
git commit -m "test(gpu): register Phase 5m2 magnetic conducting step in suite

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** `ic` keyword selects conducting path ✓; inner BC = φ0 history flux (5l, from OLD ic state, NOT incremental) ✓; outer BC 0 ✓; reconstruct (5l) after field update using NEW spec ICB value `spec[:,:,1]` (materialized), into scratch then written back to the ic state ✓; `continuity_mag` ignored when `ic` given ✓; insulating path byte-identical when `ic === nothing` (Phase-5m test still passes) ✓.

**Placeholder scan:** none.

**Type consistency:** the `ic` bundle fields `(tor_adm, pol_adm, tor_ic_r, tor_ic_i, pol_ic_r, pol_ic_i)` match between the test and the function body; `gpu_inner_core_history_flux!`/`gpu_reconstruct_inner_core!` arg orders match Phase 5l; `*_adm` are the `gpu_pack_inner_core` bundles.

**Ordering:** φ0 reads OLD ic state (before reconstruct overwrites it); reconstruct reads OLD ic state + NEW spec (after step 5); spec/prev_nl overwritten only at steps 5/6; reconstruct uses scratch (`S_new ≠ S_old` per 5l's contract).

**Aliasing:** the `spec[:,:,1]` g-slices are materialized copies (no `@view`), so reconstruct's `g` does not alias the spec; the `nic_*` reconstruct outputs are fresh scratch, copied into the ic state after.
