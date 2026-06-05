# GPU Phase 5k — Velocity Field CNAB2 Step (toroidal + poloidal) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compose one full CNAB2 timestep of the velocity field (both toroidal and poloidal components) on the GPU from the already-verified kernels — nonlinear (5i) → CNAB2 RHS (5c) → implicit solve (5d) → poloidal influence correction (5j) → field update + history rollover — bit-exact against a manual chain of the same kernels.

**Architecture:** A new file `src/gpu/velocity_step.jl` with one orchestrator `gpu_velocity_field_step!`. It is pure composition — **no new kernels**. To tame the argument count (two components × {spec, prev_nl, linear op, LU, 4 BC vectors}), per-component state is grouped into `NamedTuple` bundles `tor` and `pol`, the nonlinear operators into `nlops`, and the poloidal correction into `influence`. The toroidal and poloidal each get their own RHS+solve; the poloidal additionally gets the 2×2 influence correction (5j). The velocity nonlinear is computed ONCE (it returns both `nl_tor` and `nl_pol`). Mirrors `apply_velocity_toroidal_implicit_update!` + `apply_velocity_poloidal_implicit_update!` + the CNAB2 history rollover. Runs on Array (locally testable) and CuArray.

**Tech Stack:** Julia, KernelAbstractions (via the composed kernels), the existing `src/gpu/*` GPU kernels.

---

## Background — the pieces being composed (signatures, current `main`)

```julia
# 5i — velocity nonlinear (returns BOTH nl_tor and nl_pol; coupling kwargs optional)
gpu_velocity_nonlinear!(nl_tor_r, nl_tor_i, nl_pol_r, nl_pol_i, tor_r, tor_i, pol_r, pol_i,
    config, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, lmax, bw;
    T_phys=nothing, thermal_factor=0, r_vec=nothing, C_phys=nothing, comp_factor=0,
    J_r=nothing, J_θ=nothing, J_φ=nothing, B_r=nothing, B_θ=nothing, B_φ=nothing, lorentz_coeff=0)

# 5c — CNAB2 RHS: rr = inv_dt·u + 1.5·nl − 0.5·nl_prev + linear_weight·(lin·u)
gpu_build_rhs_cnab2!(rr, ri, ur, ui, nr_, ni_, pr, pi_, lin_batched, inv_dt, linear_weight, bw)
#   ur/ui = field, nr_/ni_ = this step's nl, pr/pi_ = prev nl

# 5d — implicit solve: set BC rows then in-place batched banded solve (solution overwrites x)
gpu_implicit_solve_field!(x_r, x_i, lu_batched, bc_in_r, bc_in_i, bc_out_r, bc_out_i, bw)

# 5j — poloidal influence correction (in-place on real + imag)
gpu_velocity_poloidal_influence_correction!(x_r, x_i, Gre_b, invG_b)
```

**Template:** `src/gpu/scalar_step.jl` (Phase 5f) is the scalar analogue — read it. The same ORDERING INVARIANT applies: `build_rhs` reads the OLD spec, so the spec is overwritten with the solution ONLY after every read of the old spec (the nonlinear at step 1 and BOTH `build_rhs` calls). Solutions live in separate `rhs_*` scratch buffers until the final copy.

**Velocity-specific facts (from the CPU `apply_velocity_*_implicit_update!`):**
- Mass coefficient is `E` (Ekman) for both toroidal and poloidal: the caller bakes `E` into `inv_dt = E/dt` AND into the per-l linear operators (`lin = E·L`), so the generic 5c kernel reproduces `RHS = (E/dt)·u + (1−θ)·E·L·u + 1.5·nl − 0.5·nl_prev`. `linear_weight = 1−θ`.
- Toroidal and poloidal use **different** linear operators / LU factors / BC rows (different boundary stencils) → `tor.lin`/`tor.lu` ≠ `pol.lin`/`pol.lu`.
- Toroidal BC rows are caller-supplied: homogeneous (0) except the `l=1, m=0` mode under a rotating inner core, whose inner row carries `rot_omega·r_inner` (incremental form subtracts the current field value). This Phase keeps the step generic — it applies whatever `tor.bc_in_*`/`tor.bc_out_*` vectors it is given; the rotation VALUE assembly is the caller's job (Phase 5n / the helper below).
- Poloidal BC rows are homogeneous (0); the poloidal gets the influence correction after its solve.

---

## Task 1: `gpu_velocity_field_step!`

**Files:**
- Create: `src/gpu/velocity_step.jl`
- Modify: `src/GeoDynamo.jl` (add `include("gpu/velocity_step.jl")` immediately after the `include("gpu/influence_correction.jl")` line; add `export gpu_velocity_field_step!` near the other `gpu_*` exports)
- Test: `test/gpu_phase5k_velocity_step.jl`

- [ ] **Step 1: Write the failing test**

Create `test/gpu_phase5k_velocity_step.jl`:

```julia
using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5k — Velocity Field CNAB2 Step (tor + pol)" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 8, mmax = 8, nlat = 24, nlon = 48, nr = 4)
    nl, nm, nr = cfg.lmax + 1, cfg.mmax + 1, 4
    nlat, nlon = cfg.nlat, cfg.nlon
    bw = 2
    rng = MersenneTwister(11)

    # banded operators (2bw+1, nr) shared by the nonlinear curls
    function band(N, b; seed)
        rng2 = MersenneTwister(seed); d = zeros(2b+1, N)
        for j in 1:N, i in max(1,j-b):min(N,j+b); d[b+1+i-j,j] = rand(rng2) - 0.5; end
        d
    end
    d1 = band(nr, bw; seed = 1); d2 = band(nr, bw; seed = 2)
    lfac = Float64[l*(l+1) for l in 0:cfg.lmax]
    rinv = [1.0/(0.5 + 0.1k) for k in 1:nr]; rinv2 = rinv .^ 2; rscale = copy(rinv)
    sinθ = sin.(range(0.1, π - 0.1; length = nlat)); cosθ = cos.(range(0.1, π - 0.1; length = nlat))
    E = 1.3e-3

    # per-l linear ops + LU for tor and pol — batched (2bw+1, nr, nl). (Wiring test:
    # same matrices feed GPU step and the manual chain, so exact == holds regardless
    # of conditioning. Make the diagonal dominant so the solve is well-posed.)
    function batched(seed)
        a = zeros(2bw+1, nr, nl); r = MersenneTwister(seed)
        for li in 1:nl, j in 1:nr, i in max(1,j-bw):min(nr,j+bw)
            a[bw+1+i-j, j, li] = rand(r) - 0.5
        end
        for li in 1:nl, j in 1:nr; a[bw+1, j, li] += 5.0; end   # diagonal dominance
        a
    end
    lin_tor = batched(10); lu_tor = batched(11)
    lin_pol = batched(20); lu_pol = batched(21)

    # BC vectors (nl, nm) — random toroidal (exercises BC propagation incl an l=1,m=0
    # rotation value), zero poloidal (homogeneous).
    bc_in_tor_r  = rand(rng, nl, nm) .- 0.5; bc_in_tor_i  = rand(rng, nl, nm) .- 0.5
    bc_out_tor_r = rand(rng, nl, nm) .- 0.5; bc_out_tor_i = rand(rng, nl, nm) .- 0.5
    bc_in_pol_r  = zeros(nl, nm); bc_in_pol_i  = zeros(nl, nm)
    bc_out_pol_r = zeros(nl, nm); bc_out_pol_i = zeros(nl, nm)

    # poloidal influence operators
    influence_dict = Dict{Int, GeoDynamo.ERK2InfluenceOp{Float64}}()
    for l in 1:cfg.lmax
        influence_dict[l] = GeoDynamo.ERK2InfluenceOp{Float64}(rand(rng, nr, 2) .- 0.5, rand(rng, 2, 2) .- 0.5, l)
    end
    Gre_b, invG_b = GeoDynamo.gpu_pack_influence(influence_dict, nl, nr, CPU())

    inv_dt = E / 5.0e-4          # mass_coeff(E) / dt
    linear_weight = 0.5          # 1 − θ

    # initial spectral state + history (random, upper-triangle modes only nonzero is
    # fine; the kernels handle empty modes as zeros)
    mk() = (a = zeros(nl, nm, nr); for mi in 1:nm, li in mi:nl, r in 1:nr; a[li,mi,r] = rand(rng) - 0.5; end; a)
    tor_r0 = mk(); tor_i0 = mk(); pol_r0 = mk(); pol_i0 = mk()
    pnt_r0 = mk(); pnt_i0 = mk(); pnp_r0 = mk(); pnp_i0 = mk()

    nlops = (; d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E)
    influence = (; Gre_b, invG_b)

    @testset "step == manual chain (exact) [LOCAL]" begin
        # ---- GPU step (on copies) ----
        tor = (; spec_r = copy(tor_r0), spec_i = copy(tor_i0),
                 prev_nl_r = copy(pnt_r0), prev_nl_i = copy(pnt_i0),
                 lin = lin_tor, lu = lu_tor,
                 bc_in_r = bc_in_tor_r, bc_in_i = bc_in_tor_i,
                 bc_out_r = bc_out_tor_r, bc_out_i = bc_out_tor_i)
        pol = (; spec_r = copy(pol_r0), spec_i = copy(pol_i0),
                 prev_nl_r = copy(pnp_r0), prev_nl_i = copy(pnp_i0),
                 lin = lin_pol, lu = lu_pol,
                 bc_in_r = bc_in_pol_r, bc_in_i = bc_in_pol_i,
                 bc_out_r = bc_out_pol_r, bc_out_i = bc_out_pol_i)
        GeoDynamo.gpu_velocity_field_step!(tor, pol, cfg, nlops, influence,
                                           inv_dt, linear_weight, cfg.lmax, bw)

        # ---- manual chain (same kernels, same order, on independent copies) ----
        mtr = copy(tor_r0); mti = copy(tor_i0); mpr = copy(pol_r0); mpi = copy(pol_i0)
        mpnt_r = copy(pnt_r0); mpnt_i = copy(pnt_i0); mpnp_r = copy(pnp_r0); mpnp_i = copy(pnp_i0)
        nlt_r = similar(mtr); nlt_i = similar(mti); nlp_r = similar(mpr); nlp_i = similar(mpi)
        GeoDynamo.gpu_velocity_nonlinear!(nlt_r, nlt_i, nlp_r, nlp_i, mtr, mti, mpr, mpi,
            cfg, d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E, cfg.lmax, bw)
        rt_r = similar(mtr); rt_i = similar(mti); rp_r = similar(mpr); rp_i = similar(mpi)
        GeoDynamo.gpu_build_rhs_cnab2!(rt_r, rt_i, mtr, mti, nlt_r, nlt_i, mpnt_r, mpnt_i,
            lin_tor, inv_dt, linear_weight, bw)
        GeoDynamo.gpu_implicit_solve_field!(rt_r, rt_i, lu_tor,
            bc_in_tor_r, bc_in_tor_i, bc_out_tor_r, bc_out_tor_i, bw)
        GeoDynamo.gpu_build_rhs_cnab2!(rp_r, rp_i, mpr, mpi, nlp_r, nlp_i, mpnp_r, mpnp_i,
            lin_pol, inv_dt, linear_weight, bw)
        GeoDynamo.gpu_implicit_solve_field!(rp_r, rp_i, lu_pol,
            bc_in_pol_r, bc_in_pol_i, bc_out_pol_r, bc_out_pol_i, bw)
        GeoDynamo.gpu_velocity_poloidal_influence_correction!(rp_r, rp_i, Gre_b, invG_b)

        @test tor.spec_r == rt_r
        @test tor.spec_i == rt_i
        @test pol.spec_r == rp_r
        @test pol.spec_i == rp_i
        @test tor.prev_nl_r == nlt_r
        @test tor.prev_nl_i == nlt_i
        @test pol.prev_nl_r == nlp_r
        @test pol.prev_nl_i == nlp_i
        @test all(isfinite, tor.spec_r) && all(isfinite, pol.spec_r)
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5k gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            # CPU reference
            ctor = (; spec_r = copy(tor_r0), spec_i = copy(tor_i0),
                      prev_nl_r = copy(pnt_r0), prev_nl_i = copy(pnt_i0),
                      lin = lin_tor, lu = lu_tor, bc_in_r = bc_in_tor_r, bc_in_i = bc_in_tor_i,
                      bc_out_r = bc_out_tor_r, bc_out_i = bc_out_tor_i)
            cpol = (; spec_r = copy(pol_r0), spec_i = copy(pol_i0),
                      prev_nl_r = copy(pnp_r0), prev_nl_i = copy(pnp_i0),
                      lin = lin_pol, lu = lu_pol, bc_in_r = bc_in_pol_r, bc_in_i = bc_in_pol_i,
                      bc_out_r = bc_out_pol_r, bc_out_i = bc_out_pol_i)
            GeoDynamo.gpu_velocity_field_step!(ctor, cpol, cfg, nlops, influence,
                                               inv_dt, linear_weight, cfg.lmax, bw)
            d(x) = GeoDynamo.on_architecture(GPU(), x)
            gGre, ginvG = GeoDynamo.gpu_pack_influence(influence_dict, nl, nr, GPU())
            gnlops = (; d1 = d(d1), d2 = d(d2), lfac = d(lfac), rinv = d(rinv), rinv2 = d(rinv2),
                        rscale = d(rscale), sinθ = d(sinθ), cosθ = d(cosθ), E = E)
            gtor = (; spec_r = d(copy(tor_r0)), spec_i = d(copy(tor_i0)),
                      prev_nl_r = d(copy(pnt_r0)), prev_nl_i = d(copy(pnt_i0)),
                      lin = d(lin_tor), lu = d(lu_tor),
                      bc_in_r = d(bc_in_tor_r), bc_in_i = d(bc_in_tor_i),
                      bc_out_r = d(bc_out_tor_r), bc_out_i = d(bc_out_tor_i))
            gpol = (; spec_r = d(copy(pol_r0)), spec_i = d(copy(pol_i0)),
                      prev_nl_r = d(copy(pnp_r0)), prev_nl_i = d(copy(pnp_i0)),
                      lin = d(lin_pol), lu = d(lu_pol),
                      bc_in_r = d(bc_in_pol_r), bc_in_i = d(bc_in_pol_i),
                      bc_out_r = d(bc_out_pol_r), bc_out_i = d(bc_out_pol_i))
            GeoDynamo.gpu_velocity_field_step!(gtor, gpol, cfg,
                gnlops, (; Gre_b = gGre, invG_b = ginvG), inv_dt, linear_weight, cfg.lmax, bw)
            @test gtor.spec_r isa CUDA.CuArray
            @test isapprox(Array(gtor.spec_r), ctor.spec_r; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gpol.spec_r), cpol.spec_r; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gtor.prev_nl_r), ctor.prev_nl_r; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gpol.prev_nl_i), cpol.prev_nl_i; atol = 1e-9, rtol = 1e-8)
        end
    end
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5k_velocity_step.jl")'
```
Expected: FAIL — `UndefVarError: gpu_velocity_field_step!` not defined.

- [ ] **Step 3: Write the implementation**

Create `src/gpu/velocity_step.jl`:

```julia
# =============================================================================
# GPU Phase 5k — one velocity field CNAB2 timestep (toroidal + poloidal),
# composing the verified pieces: velocity nonlinear (5i, returns BOTH nl_tor and
# nl_pol) → CNAB2 RHS (5c, per component) → implicit solve (5d, per component) →
# poloidal influence-matrix correction (5j) → field update + nl_prev rollover.
# Mirrors apply_velocity_toroidal_implicit_update! + apply_velocity_poloidal_
# implicit_update! + the CNAB2 history rollover.  No new kernels — pure
# composition.  Runs on Array + CuArray.  (Per-call scratch — Phase-6 may cache.)
#
# Per-component state is grouped into NamedTuple bundles `tor`/`pol` (mutated in
# place through their array fields) to keep the argument list legible:
#   tor/pol :: (; spec_r, spec_i, prev_nl_r, prev_nl_i, lin, lu,
#                 bc_in_r, bc_in_i, bc_out_r, bc_out_i)
#   nlops   :: (; d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E)
#   influence :: (; Gre_b, invG_b)
# =============================================================================

"""
    gpu_velocity_field_step!(tor, pol, config, nlops, influence, inv_dt, linear_weight,
                             lmax, bw; <coupling kwargs>) -> nothing

Advance the velocity field one CNAB2 step.  `tor`/`pol` are NamedTuple bundles of
the toroidal/poloidal arrays (see the file header); on entry `*.spec_*` is the
field and `*.prev_nl_*` the previous nonlinear term, on exit `*.spec_*` is the
updated field and `*.prev_nl_*` holds THIS step's nonlinear term (rolled over).
`nlops` carries the nonlinear/curl operators, `influence` the poloidal 2×2
correction operators.  `inv_dt = E/dt` and the per-l `lin` operators must already
carry the mass coefficient `E` (so 5c reproduces the velocity RHS); `linear_weight
= 1−θ`.  Coupling kwargs (thermal/compositional buoyancy, Lorentz) are forwarded
to [`gpu_velocity_nonlinear!`](@ref); omit them for the velocity-only step.  All
arrays on the same backend.

ORDERING INVARIANT (as in `gpu_scalar_field_step!`): the nonlinear and BOTH
`build_rhs` calls read the OLD `*.spec_*`; the spec is overwritten with the
solution ONLY after every such read.  Do not move the field-update copies earlier.
"""
function gpu_velocity_field_step!(tor, pol, config, nlops, influence,
        inv_dt, linear_weight, lmax::Int, bw::Int;
        T_phys = nothing, thermal_factor = zero(eltype(tor.spec_r)), r_vec = nothing,
        C_phys = nothing, comp_factor = zero(eltype(tor.spec_r)),
        J_r = nothing, J_θ = nothing, J_φ = nothing,
        B_r = nothing, B_θ = nothing, B_φ = nothing, lorentz_coeff = zero(eltype(tor.spec_r)))
    # 1. velocity nonlinear (5i): nl_tor / nl_pol captured from the OLD tor/pol spec.
    nlt_r = similar(tor.spec_r); nlt_i = similar(tor.spec_i)   # Phase-6: workspace
    nlp_r = similar(pol.spec_r); nlp_i = similar(pol.spec_i)
    gpu_velocity_nonlinear!(nlt_r, nlt_i, nlp_r, nlp_i,
        tor.spec_r, tor.spec_i, pol.spec_r, pol.spec_i, config,
        nlops.d1, nlops.d2, nlops.lfac, nlops.rinv, nlops.rinv2, nlops.rscale,
        nlops.sinθ, nlops.cosθ, nlops.E, lmax, bw;
        T_phys = T_phys, thermal_factor = thermal_factor, r_vec = r_vec,
        C_phys = C_phys, comp_factor = comp_factor,
        J_r = J_r, J_θ = J_θ, J_φ = J_φ, B_r = B_r, B_θ = B_θ, B_φ = B_φ,
        lorentz_coeff = lorentz_coeff)

    # 2. toroidal CNAB2 RHS (5c) from OLD tor spec, then implicit solve (5d).
    rt_r = similar(tor.spec_r); rt_i = similar(tor.spec_i)     # Phase-6: workspace
    gpu_build_rhs_cnab2!(rt_r, rt_i, tor.spec_r, tor.spec_i, nlt_r, nlt_i,
        tor.prev_nl_r, tor.prev_nl_i, tor.lin, inv_dt, linear_weight, bw)
    gpu_implicit_solve_field!(rt_r, rt_i, tor.lu,
        tor.bc_in_r, tor.bc_in_i, tor.bc_out_r, tor.bc_out_i, bw)

    # 3. poloidal CNAB2 RHS (5c) from OLD pol spec, implicit solve (5d), then the
    #    2×2 influence correction (5j) on the poloidal solution.
    rp_r = similar(pol.spec_r); rp_i = similar(pol.spec_i)     # Phase-6: workspace
    gpu_build_rhs_cnab2!(rp_r, rp_i, pol.spec_r, pol.spec_i, nlp_r, nlp_i,
        pol.prev_nl_r, pol.prev_nl_i, pol.lin, inv_dt, linear_weight, bw)
    gpu_implicit_solve_field!(rp_r, rp_i, pol.lu,
        pol.bc_in_r, pol.bc_in_i, pol.bc_out_r, pol.bc_out_i, bw)
    gpu_velocity_poloidal_influence_correction!(rp_r, rp_i, influence.Gre_b, influence.invG_b)

    # 4. update the fields (AFTER every read of the old spec — ORDERING INVARIANT).
    tor.spec_r .= rt_r; tor.spec_i .= rt_i
    pol.spec_r .= rp_r; pol.spec_i .= rp_i
    # 5. roll the histories: prev_nl ← this step's nl (captured at step 1).
    tor.prev_nl_r .= nlt_r; tor.prev_nl_i .= nlt_i
    pol.prev_nl_r .= nlp_r; pol.prev_nl_i .= nlp_i
    return nothing
end
```

Modify `src/GeoDynamo.jl` — add the include immediately after `include("gpu/influence_correction.jl")`:

```julia
include("gpu/velocity_step.jl")
```

And add the export near the other `gpu_*` exports (after `export gpu_pack_influence, gpu_velocity_poloidal_influence_correction!`):

```julia
export gpu_velocity_field_step!
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5k_velocity_step.jl")'
```
Expected: the `[LOCAL]` testset PASSES (8 exact `==` outputs + finiteness); the `[GPU-BOX]` testset shows 1 Broken (`@test_skip`).

- [ ] **Step 5: Verify the module still loads**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using GeoDynamo; println("LOAD OK")'
```
Expected: `LOAD OK`.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/velocity_step.jl src/GeoDynamo.jl test/gpu_phase5k_velocity_step.jl
git commit -m "feat(gpu): Phase 5k velocity field CNAB2 step (tor + pol)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Register the test + regression check

**Files:**
- Modify: `test/runtests.jl` (add the Phase 5k entry after the Phase 5j entry)

- [ ] **Step 1: Add the test to the suite**

In `test/runtests.jl`, find the line that includes `"gpu_phase5j_influence_correction.jl"` and add immediately after it (same indentation):

```julia
    "gpu_phase5k_velocity_step.jl",
```

- [ ] **Step 2: Confirm the new test still passes in isolation**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5k_velocity_step.jl")' > /tmp/phase5k.log 2>&1; echo "exit=$?"; tail -20 /tmp/phase5k.log
```
Expected: `exit=0`, the `[LOCAL]` testset passes, 1 Broken for the GPU-box gate.

- [ ] **Step 3: Confirm the allocation guards still pass**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/allocation_guards.jl")' > /tmp/allocguards.log 2>&1; echo "exit=$?"; tail -8 /tmp/allocguards.log
```
Expected: `exit=0`, 39/39 unchanged. (If the file name differs, locate with `grep -rl "alloc" test/ | head`.)

- [ ] **Step 4: Commit**

```bash
git add test/runtests.jl
git commit -m "test(gpu): register Phase 5k velocity step in suite

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** one velocity nonlinear call → tor RHS+solve → pol RHS+solve+influence → field update → rollover, all composed from existing kernels ✓; mass coefficient `E` carried by `inv_dt`/`lin` (documented as caller responsibility) ✓; tor/pol use separate `lin`/`lu`/BC ✓; toroidal BC vectors applied generically (rotation value is the caller's job — documented) ✓; coupling kwargs forwarded to the nonlinear ✓; ORDERING INVARIANT preserved (spec overwritten only after all old-spec reads) ✓; runs on Array + CuArray ✓.

**Placeholder scan:** none — every step has full code/commands.

**Type consistency:** `gpu_velocity_field_step!` signature `(tor, pol, config, nlops, influence, inv_dt, linear_weight, lmax, bw; kwargs)` identical across the impl and both test call sites; the NamedTuple field names (`spec_r/spec_i/prev_nl_r/prev_nl_i/lin/lu/bc_in_r/bc_in_i/bc_out_r/bc_out_i`, `nlops.{d1,d2,lfac,rinv,rinv2,rscale,sinθ,cosθ,E}`, `influence.{Gre_b,invG_b}`) match between the test bundles and the function body; composed-kernel arg orders match the signatures in the Background section.

**Ordering:** nonlinear + both `build_rhs` read old spec; `.= rt_*`/`.= rp_*` copies happen only at step 4 — documented invariant, mirrors `gpu_scalar_field_step!`.
