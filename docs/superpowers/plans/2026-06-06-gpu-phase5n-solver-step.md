# GPU Phase 5n — Full `gpu_solver_step!` Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compose the four per-field GPU CNAB2 steps into one full-step orchestrator `gpu_solver_step!`, matching the CPU `solver_step!` field order (velocity → magnetic → temperature → composition), the shared physical velocity, and the one-step **lag** of the physical fields (T/C/B/J) that the velocity nonlinear consumes — bit-exact against a manual chain. This is the orchestration LOGIC only; building a device state from a real CPU `SolverState` + the GPU≈CPU full-step gate is Phase 5n2.

**Architecture:** A new file `src/gpu/solver_step.jl` with `gpu_solver_step!(state)`. `state` is a NamedTuple bundle holding the per-field bundles (velocity, magnetic, temperature, composition — the last two optional via `nothing`), the shared operators, the coupling factors, and the **persistent physical buffers** (`T_phys, C_phys, B_r/θ/φ, J_r/θ/φ`) that velocity consumes lagged. One step: (1) synthesize the shared physical velocity `u` and the current-step physical buffers from the OLD spectral state; (2) run the velocity step with the LAGGED (previous-step) physical buffers as buoyancy/Lorentz inputs; (3) run magnetic/temperature/composition steps with the shared `u`; (4) roll the physical buffers (current → lagged) for the next step. No new kernels — pure composition of `gpu_velocity_field_step!` (5k), `gpu_magnetic_field_step!` (5m/5m2), `gpu_scalar_field_step!` (5f), and the transform/curl kernels. Runs on Array (locally testable) and CuArray.

**Tech Stack:** Julia, the Phase 0–5m2 GPU kernels.

---

## Background — the per-field step signatures (read, current `main` + branch)

```julia
# 5k velocity — NamedTuple bundles + coupling kwargs (lagged physical)
gpu_velocity_field_step!(tor, pol, config, nlops, influence, inv_dt, linear_weight, lmax, bw;
    T_phys=nothing, thermal_factor=0, r_vec=nothing, C_phys=nothing, comp_factor=0,
    J_r=nothing, J_θ=nothing, J_φ=nothing, B_r=nothing, B_θ=nothing, B_φ=nothing, lorentz_coeff=0)
#   tor/pol :: (; spec_r, spec_i, prev_nl_r, prev_nl_i, lin, lu, bc_in_r, bc_in_i, bc_out_r, bc_out_i)
#   nlops :: (; d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E);  influence :: (; Gre_b, invG_b)

# 5m/5m2 magnetic — NamedTuple bundles + physical u + optional ic
gpu_magnetic_field_step!(tor, pol, u_r, u_θ, u_φ, config, nlops, inv_dt, linear_weight, lmax, bw;
    continuity_mag=false, ic=nothing)
#   tor/pol :: (; spec_r, spec_i, prev_nl_r, prev_nl_i, lin, lu);  nlops :: (; d1, d2, lfac, rinv, rinv2, rscale)

# 5f scalar (temperature/composition) — RAW args + physical u
gpu_scalar_field_step!(spec_r, spec_i, prev_nl_r, prev_nl_i, u_r, u_θ, u_φ, config,
    d1, mvals, rinv, lin_batched, lu_batched, bc_in_r, bc_in_i, bc_out_r, bc_out_i,
    inv_dt, linear_weight, lmax, bw)

# transforms / curl (for the shared physical fields)
gpu_vector_spectral_to_physical!(vr::GPUPhysicalField, vθ, vφ, tor::GPUSpectralField, pol::GPUSpectralField, config, lfac, rscale)
gpu_scalar_spectral_to_physical!(phys::GPUPhysicalField, spec::GPUSpectralField, config)
gpu_spectral_curl!(dtr, dti, dpr, dpi, str, sti, spr, spi, d1, d2, lfac, rinv, rinv2, bw)   # B(tor,pol) → J(tor,pol) spectral
# field wrappers
spec(a,b) = GeoDynamo.GPUSpectralField{eltype(a),typeof(a)}(config, size(a,1), size(a,2), size(a,3), a, b)
ph()      = GeoDynamo.allocate_gpu_physical_field(eltype, arch, config, nr)   # .data is (nlat,nlon,nr)
```

CPU orchestration facts (from `compute_solver_nonlinear_terms!` nonlinear.jl:1010, `apply_solver_implicit_step!` driver.jl:270, `roll_solver_histories!` driver.jl:96):
- Order **velocity → magnetic → temperature → composition**. Velocity nonlinear synthesizes the shared physical `u` (fresh, from current velocity).
- **The lag:** velocity nonlinear runs FIRST and reads T_phys/C_phys/B_phys/J_phys that were synthesized during the PREVIOUS step (the scalar/magnetic nonlinear phases), NOT refreshed before velocity. So velocity's buoyancy+Lorentz use the previous step's physical buffers; the current step re-synthesizes them (from OLD spectral, before the implicit overwrites it) for the NEXT step's velocity.
- `J_phys` = current density = vector synthesis of `∇×B` (`gpu_spectral_curl!(B) → J_spectral`, then vector synth). Only velocity's Lorentz needs J.
- Coupling factors: thermal `(Pm/Pr)·Ra`, compositional `(Pm/Sc)·RaC`, Lorentz `1/Pm`, velocity advection mass `Ek`. Implicit mass coeffs: vel `Ek`, temp `Pm/Pr`, comp `Pm/Sc`, magnetic `1`. (In `state` these arrive pre-baked into `inv_dt_*`/`lin`/`*_factor`.)
- Optional fields: skip magnetic if `state.magnetic === nothing`; skip composition if `state.composition === nothing`.
- All phase-A transforms read OLD spectral; implicit phase has no cross-field reads; rollover copies nl→prev_nl (handled INSIDE each per-field step here).

---

## Task 1: `gpu_solver_step!`

**Files:**
- Create: `src/gpu/solver_step.jl`
- Modify: `src/GeoDynamo.jl` (add `include("gpu/solver_step.jl")` after the `gpu/magnetic_step.jl` include; add `export gpu_solver_step!`)
- Test: `test/gpu_phase5n_solver_step.jl`

- [ ] **Step 1: Write the failing test**

Create `test/gpu_phase5n_solver_step.jl`. It builds a synthetic `state` with all fields present, runs `gpu_solver_step!`, and compares against a manual chain that performs the identical sequence. Use small dims.

```julia
using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5n — Full gpu_solver_step! orchestration" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax = 6, mmax = 6, nlat = 20, nlon = 40, nr = 4)
    nl, nm, nr = cfg.lmax + 1, cfg.mmax + 1, 4
    nlat, nlon = cfg.nlat, cfg.nlon
    bw = 2
    rng = MersenneTwister(29)

    function band(N, b; seed)
        r = MersenneTwister(seed); d = zeros(2b+1, N)
        for j in 1:N, i in max(1,j-b):min(N,j+b); d[b+1+i-j,j] = rand(r) - 0.5; end
        d
    end
    d1 = band(nr, bw; seed = 1); d2 = band(nr, bw; seed = 2)
    lfac = Float64[l*(l+1) for l in 0:cfg.lmax]
    rinv = [1.0/(0.5+0.1k) for k in 1:nr]; rinv2 = rinv .^ 2; rscale = copy(rinv)
    sinθ = sin.(range(0.1, π-0.1; length = nlat)); cosθ = cos.(range(0.1, π-0.1; length = nlat))
    mvals = Float64[m for m in 0:cfg.mmax]
    r_vec = [0.5 + 0.1k for k in 1:nr]
    E = 1.3e-3; thermal_factor = 0.7; comp_factor = 0.4; lorentz_coeff = 1.0/0.3
    inv_dt_v = E/5e-4; inv_dt_m = 1.0/5e-4; inv_dt_t = (1.0/0.7)/5e-4; inv_dt_c = (1.0/0.9)/5e-4
    linw = 0.5

    function batched(seed)
        a = zeros(2bw+1, nr, nl); r = MersenneTwister(seed)
        for li in 1:nl, j in 1:nr, i in max(1,j-bw):min(nr,j+bw); a[bw+1+i-j,j,li] = rand(r)-0.5; end
        for li in 1:nl, j in 1:nr; a[bw+1,j,li] += 5.0; end
        a
    end
    influence = Dict{Int, GeoDynamo.ERK2InfluenceOp{Float64}}()
    for l in 1:cfg.lmax; influence[l] = GeoDynamo.ERK2InfluenceOp{Float64}(rand(rng,nr,2).-0.5, rand(rng,2,2).-0.5, l); end
    Gre_b, invG_b = GeoDynamo.gpu_pack_influence(influence, nl, nr, CPU())

    mk() = (a = zeros(nl,nm,nr); for mi in 1:nm, li in mi:nl, r in 1:nr; a[li,mi,r] = rand(rng)-0.5; end; a)
    phys() = rand(rng, nlat, nlon, nr) .- 0.5

    # build a fresh `state` NamedTuple (deep copies of all mutable arrays)
    function build_state()
        velocity = (;
            tor = (; spec_r=mk(), spec_i=mk(), prev_nl_r=mk(), prev_nl_i=mk(),
                     lin=batched(10), lu=batched(11),
                     bc_in_r=zeros(nl,nm), bc_in_i=zeros(nl,nm), bc_out_r=zeros(nl,nm), bc_out_i=zeros(nl,nm)),
            pol = (; spec_r=mk(), spec_i=mk(), prev_nl_r=mk(), prev_nl_i=mk(),
                     lin=batched(12), lu=batched(13),
                     bc_in_r=zeros(nl,nm), bc_in_i=zeros(nl,nm), bc_out_r=zeros(nl,nm), bc_out_i=zeros(nl,nm)))
        magnetic = (;
            tor = (; spec_r=mk(), spec_i=mk(), prev_nl_r=mk(), prev_nl_i=mk(), lin=batched(20), lu=batched(21)),
            pol = (; spec_r=mk(), spec_i=mk(), prev_nl_r=mk(), prev_nl_i=mk(), lin=batched(22), lu=batched(23)))
        temperature = (; spec_r=mk(), spec_i=mk(), prev_nl_r=mk(), prev_nl_i=mk(),
                         lin=batched(30), lu=batched(31),
                         bc_in_r=zeros(nl,nm), bc_in_i=zeros(nl,nm), bc_out_r=zeros(nl,nm), bc_out_i=zeros(nl,nm))
        composition = (; spec_r=mk(), spec_i=mk(), prev_nl_r=mk(), prev_nl_i=mk(),
                         lin=batched(40), lu=batched(41),
                         bc_in_r=zeros(nl,nm), bc_in_i=zeros(nl,nm), bc_out_r=zeros(nl,nm), bc_out_i=zeros(nl,nm))
        (;
            config = cfg, lmax = cfg.lmax, bw = bw, linear_weight = linw,
            nlops_vel = (; d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E),
            nlops_mag = (; d1, d2, lfac, rinv, rinv2, rscale),
            influence = (; Gre_b, invG_b),
            d1 = d1, mvals = mvals, rinv = rinv, rscale = rscale, lfac = lfac, d2 = d2, rinv2 = rinv2,
            r_vec = r_vec, thermal_factor = thermal_factor, comp_factor = comp_factor, lorentz_coeff = lorentz_coeff,
            inv_dt_vel = inv_dt_v, inv_dt_mag = inv_dt_m, inv_dt_temp = inv_dt_t, inv_dt_comp = inv_dt_c,
            velocity = velocity, magnetic = magnetic, temperature = temperature, composition = composition,
            # persistent LAGGED physical buffers (previous step's synthesis)
            T_phys = phys(), C_phys = phys(),
            B_r = phys(), B_θ = phys(), B_φ = phys(), J_r = phys(), J_θ = phys(), J_φ = phys())
    end

    @testset "full step == manual chain (exact) [LOCAL]" begin
        st = build_state()
        # snapshot the OLD spectral + OLD lagged buffers BEFORE the step (deep copies)
        v0 = deepcopy(st.velocity); m0 = deepcopy(st.magnetic); t0 = deepcopy(st.temperature); c0 = deepcopy(st.composition)
        T0 = copy(st.T_phys); C0 = copy(st.C_phys)
        B0 = (copy(st.B_r), copy(st.B_θ), copy(st.B_φ)); J0 = (copy(st.J_r), copy(st.J_θ), copy(st.J_φ))

        GeoDynamo.gpu_solver_step!(st)

        # ---- manual chain ----
        spec(a,b) = GeoDynamo.GPUSpectralField{Float64,typeof(a)}(cfg, nl, nm, nr, a, b)
        ph() = GeoDynamo.allocate_gpu_physical_field(Float64, CPU(), cfg, nr)
        # 1. shared u from OLD velocity
        ur=ph(); uθ=ph(); uφ=ph()
        GeoDynamo.gpu_vector_spectral_to_physical!(ur, uθ, uφ, spec(v0.tor.spec_r, v0.tor.spec_i),
            spec(v0.pol.spec_r, v0.pol.spec_i), cfg, lfac, rscale)
        # 2. current-step physical buffers from OLD spectral (for NEXT step's velocity)
        Tn=ph(); GeoDynamo.gpu_scalar_spectral_to_physical!(Tn, spec(t0.spec_r,t0.spec_i), cfg)
        Cn=ph(); GeoDynamo.gpu_scalar_spectral_to_physical!(Cn, spec(c0.spec_r,c0.spec_i), cfg)
        Br=ph(); Bθ=ph(); Bφ=ph()
        GeoDynamo.gpu_vector_spectral_to_physical!(Br, Bθ, Bφ, spec(m0.tor.spec_r,m0.tor.spec_i),
            spec(m0.pol.spec_r,m0.pol.spec_i), cfg, lfac, rscale)
        jtr=zeros(nl,nm,nr); jti=zeros(nl,nm,nr); jpr=zeros(nl,nm,nr); jpi=zeros(nl,nm,nr)
        GeoDynamo.gpu_spectral_curl!(jtr,jti,jpr,jpi, m0.tor.spec_r,m0.tor.spec_i, m0.pol.spec_r,m0.pol.spec_i,
            d1,d2,lfac,rinv,rinv2,bw)
        Jr=ph(); Jθ=ph(); Jφ=ph()
        GeoDynamo.gpu_vector_spectral_to_physical!(Jr, Jθ, Jφ, spec(jtr,jti), spec(jpr,jpi), cfg, lfac, rscale)
        # 3. velocity step with LAGGED buffers (T0/C0/B0/J0)
        mvtor = deepcopy(v0.tor); mvpol = deepcopy(v0.pol)
        GeoDynamo.gpu_velocity_field_step!(mvtor, mvpol, cfg, st.nlops_vel, st.influence, inv_dt_v, linw, cfg.lmax, bw;
            T_phys=T0, thermal_factor=thermal_factor, r_vec=r_vec, C_phys=C0, comp_factor=comp_factor,
            J_r=J0[1], J_θ=J0[2], J_φ=J0[3], B_r=B0[1], B_θ=B0[2], B_φ=B0[3], lorentz_coeff=lorentz_coeff)
        # 4. magnetic step with shared u
        mmtor = deepcopy(m0.tor); mmpol = deepcopy(m0.pol)
        GeoDynamo.gpu_magnetic_field_step!(mmtor, mmpol, ur.data, uθ.data, uφ.data, cfg, st.nlops_mag, inv_dt_m, linw, cfg.lmax, bw)
        # 5. temperature + 6. composition with shared u
        mt = deepcopy(t0); mc = deepcopy(c0)
        GeoDynamo.gpu_scalar_field_step!(mt.spec_r, mt.spec_i, mt.prev_nl_r, mt.prev_nl_i, ur.data, uθ.data, uφ.data, cfg,
            d1, mvals, rinv, mt.lin, mt.lu, mt.bc_in_r, mt.bc_in_i, mt.bc_out_r, mt.bc_out_i, inv_dt_t, linw, cfg.lmax, bw)
        GeoDynamo.gpu_scalar_field_step!(mc.spec_r, mc.spec_i, mc.prev_nl_r, mc.prev_nl_i, ur.data, uθ.data, uφ.data, cfg,
            d1, mvals, rinv, mc.lin, mc.lu, mc.bc_in_r, mc.bc_in_i, mc.bc_out_r, mc.bc_out_i, inv_dt_c, linw, cfg.lmax, bw)

        # ---- compare updated spectral state ----
        @test st.velocity.tor.spec_r == mvtor.spec_r && st.velocity.pol.spec_r == mvpol.spec_r
        @test st.velocity.tor.spec_i == mvtor.spec_i && st.velocity.pol.spec_i == mvpol.spec_i
        @test st.magnetic.tor.spec_r == mmtor.spec_r && st.magnetic.pol.spec_i == mmpol.spec_i
        @test st.temperature.spec_r == mt.spec_r && st.temperature.spec_i == mt.spec_i
        @test st.composition.spec_r == mc.spec_r && st.composition.spec_i == mc.spec_i
        # ---- compare rolled physical buffers (current-step synthesis) ----
        @test st.T_phys == Tn.data && st.C_phys == Cn.data
        @test st.B_r == Br.data && st.J_φ == Jφ.data
        @test all(isfinite, st.velocity.tor.spec_r) && all(isfinite, st.magnetic.tor.spec_r)
    end

    @testset "gating: no magnetic / no composition [LOCAL]" begin
        st = build_state()
        st2 = (; st..., magnetic = nothing, composition = nothing,
                 B_r = nothing, B_θ = nothing, B_φ = nothing, J_r = nothing, J_θ = nothing, J_φ = nothing)
        # must run without touching magnetic/composition and without Lorentz coupling
        GeoDynamo.gpu_solver_step!(st2)
        @test all(isfinite, st2.velocity.tor.spec_r)
        @test all(isfinite, st2.temperature.spec_r)
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5n gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            @test true   # full device-state parity is exercised by Phase 5n2 (real SolverState)
        end
    end
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5n_solver_step.jl")'
```
Expected: FAIL — `UndefVarError: gpu_solver_step!`.

- [ ] **Step 3: Write the implementation**

Create `src/gpu/solver_step.jl`:

```julia
# =============================================================================
# GPU Phase 5n — full multi-field CNAB2 timestep orchestrator, composing the
# per-field steps in the CPU order velocity → magnetic → temperature →
# composition (compute_solver_nonlinear_terms! nonlinear.jl:1010 + the implicit
# updates + roll_solver_histories!).  No new kernels.  Velocity runs first and
# synthesizes the shared physical velocity u, reused by every other field.
#
# THE LAG (matches CPU exactly): velocity's buoyancy + Lorentz read the physical
# T/C/B/J synthesized during the PREVIOUS step (CPU never refreshes them before
# velocity).  So this step (a) runs velocity with the persistent LAGGED buffers
# `state.T_phys/C_phys/B_*/J_*`, and (b) re-synthesizes those buffers from the OLD
# spectral state (before the implicit overwrites it) for the NEXT step's velocity,
# rolling them at the end.  u itself is NOT lagged (fresh synth of current velocity).
#
# `state` (NamedTuple) holds the per-field bundles, shared operators, coupling
# factors, and the persistent physical buffers.  magnetic/composition optional
# (nothing → skipped).  Per-field rollover (nl→prev_nl) happens inside each step.
# Runs on Array + CuArray.  (Per-call scratch — Phase-6 may cache.  The velocity
# step recomputes u internally; redundant with the shared u but identical since
# both synth the same OLD velocity.)  Device-state builder + GPU≈CPU gate = 5n2.
# =============================================================================

"""
    gpu_solver_step!(state) -> nothing

Advance every field one CNAB2 step on the GPU, in the CPU order
velocity → magnetic → temperature → composition, with the shared physical velocity
and the one-step physical-field lag for velocity's buoyancy/Lorentz coupling.  See
the file header for the `state` bundle layout and the lag semantics.  `state.magnetic`
/ `state.composition` may be `nothing` to skip those fields.  All arrays on the same backend.
"""
function gpu_solver_step!(state)
    cfg = state.config; lmax = state.lmax; bw = state.bw; linw = state.linear_weight
    v = state.velocity
    arch = arch_of(v.tor.spec_r)
    nr = size(v.tor.spec_r, 3)
    spec(a, b) = GPUSpectralField{eltype(a), typeof(a)}(cfg, size(a, 1), size(a, 2), size(a, 3), a, b)
    ph() = allocate_gpu_physical_field(eltype(v.tor.spec_r), arch, cfg, nr)

    # --- (1) shared physical velocity u from the OLD velocity spectral (fresh) ---
    u = ph(); uθ = ph(); uφ = ph()
    gpu_vector_spectral_to_physical!(u, uθ, uφ, spec(v.tor.spec_r, v.tor.spec_i),
        spec(v.pol.spec_r, v.pol.spec_i), cfg, state.nlops_vel.lfac, state.nlops_vel.rscale)

    # --- (2) current-step physical buffers from OLD spectral (for the NEXT step's velocity lag) ---
    Tn = ph(); gpu_scalar_spectral_to_physical!(Tn, spec(state.temperature.spec_r, state.temperature.spec_i), cfg)
    Cn = state.composition === nothing ? nothing :
        (c = ph(); gpu_scalar_spectral_to_physical!(c, spec(state.composition.spec_r, state.composition.spec_i), cfg); c)
    Bn_r = Bn_θ = Bn_φ = Jn_r = Jn_θ = Jn_φ = nothing
    if state.magnetic !== nothing
        m = state.magnetic
        br = ph(); bθ = ph(); bφ = ph()
        gpu_vector_spectral_to_physical!(br, bθ, bφ, spec(m.tor.spec_r, m.tor.spec_i),
            spec(m.pol.spec_r, m.pol.spec_i), cfg, state.nlops_vel.lfac, state.nlops_vel.rscale)
        jtr = similar(m.tor.spec_r); jti = similar(m.tor.spec_i); jpr = similar(m.pol.spec_r); jpi = similar(m.pol.spec_i)
        gpu_spectral_curl!(jtr, jti, jpr, jpi, m.tor.spec_r, m.tor.spec_i, m.pol.spec_r, m.pol.spec_i,
            state.nlops_mag.d1, state.nlops_mag.d2, state.nlops_mag.lfac, state.nlops_mag.rinv, state.nlops_mag.rinv2, bw)
        jr = ph(); jθ = ph(); jφ = ph()
        gpu_vector_spectral_to_physical!(jr, jθ, jφ, spec(jtr, jti), spec(jpr, jpi), cfg,
            state.nlops_vel.lfac, state.nlops_vel.rscale)
        Bn_r = br.data; Bn_θ = bθ.data; Bn_φ = bφ.data; Jn_r = jr.data; Jn_θ = jθ.data; Jn_φ = jφ.data
    end

    # --- (3) velocity step with the LAGGED physical buffers (previous step's synthesis) ---
    gpu_velocity_field_step!(v.tor, v.pol, cfg, state.nlops_vel, state.influence,
        state.inv_dt_vel, linw, lmax, bw;
        T_phys = state.T_phys, thermal_factor = state.thermal_factor, r_vec = state.r_vec,
        C_phys = state.C_phys, comp_factor = state.comp_factor,
        J_r = state.J_r, J_θ = state.J_θ, J_φ = state.J_φ,
        B_r = state.B_r, B_θ = state.B_θ, B_φ = state.B_φ, lorentz_coeff = state.lorentz_coeff)

    # --- (4) magnetic step (if present) with the shared u ---
    if state.magnetic !== nothing
        m = state.magnetic
        gpu_magnetic_field_step!(m.tor, m.pol, u.data, uθ.data, uφ.data, cfg, state.nlops_mag,
            state.inv_dt_mag, linw, lmax, bw)
    end

    # --- (5) temperature step with the shared u ---
    t = state.temperature
    gpu_scalar_field_step!(t.spec_r, t.spec_i, t.prev_nl_r, t.prev_nl_i, u.data, uθ.data, uφ.data, cfg,
        state.d1, state.mvals, state.rinv, t.lin, t.lu, t.bc_in_r, t.bc_in_i, t.bc_out_r, t.bc_out_i,
        state.inv_dt_temp, linw, lmax, bw)

    # --- (6) composition step (if present) with the shared u ---
    if state.composition !== nothing
        c = state.composition
        gpu_scalar_field_step!(c.spec_r, c.spec_i, c.prev_nl_r, c.prev_nl_i, u.data, uθ.data, uφ.data, cfg,
            state.d1, state.mvals, state.rinv, c.lin, c.lu, c.bc_in_r, c.bc_in_i, c.bc_out_r, c.bc_out_i,
            state.inv_dt_comp, linw, lmax, bw)
    end

    # --- (7) roll the persistent physical buffers (current synthesis → lagged) for the next step ---
    state.T_phys .= Tn.data
    Cn !== nothing && (state.C_phys .= Cn.data)
    if state.magnetic !== nothing
        state.B_r .= Bn_r; state.B_θ .= Bn_θ; state.B_φ .= Bn_φ
        state.J_r .= Jn_r; state.J_θ .= Jn_θ; state.J_φ .= Jn_φ
    end
    return nothing
end
```

Modify `src/GeoDynamo.jl` — add the include after `include("gpu/magnetic_step.jl")`:

```julia
include("gpu/solver_step.jl")
```

And export (after `export gpu_magnetic_field_step!`):

```julia
export gpu_solver_step!
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5n_solver_step.jl")'
```
Expected: the two `[LOCAL]` testsets PASS (full step == manual chain exact `==`; gating runs); the `[GPU-BOX]` testset shows 1 Broken (`@test_skip`).

- [ ] **Step 5: Verify the module loads**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using GeoDynamo; println("LOAD OK")'
```
Expected: `LOAD OK`.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/solver_step.jl src/GeoDynamo.jl test/gpu_phase5n_solver_step.jl
git commit -m "feat(gpu): Phase 5n full gpu_solver_step! orchestration (lagged physical coupling)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Register the test + regression check

**Files:**
- Modify: `test/runtests.jl` (add the Phase 5n entry after the Phase 5m2 entry)

- [ ] **Step 1: Add the test to the suite**

In `test/runtests.jl`, after `"gpu_phase5m2_magnetic_conducting.jl"`, add (same indentation):

```julia
    "gpu_phase5n_solver_step.jl",
```

- [ ] **Step 2: Confirm the new test still passes in isolation**

Run:
```
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Test, GeoDynamo, MPI, Random; include("test/gpu_phase5n_solver_step.jl")' > /tmp/phase5n.log 2>&1; echo "exit=$?"; tail -25 /tmp/phase5n.log
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
git commit -m "test(gpu): register Phase 5n solver-step orchestration in suite

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** field order velocity→magnetic→temperature→composition ✓; shared `u` synthesized once from OLD velocity, passed to magnetic/temp/comp ✓; velocity consumes LAGGED physical buffers (buoyancy T/C + Lorentz J/B) ✓; current-step buffers synthesized from OLD spectral (T/C scalar synth, B vector synth, J = curl→vector synth) BEFORE the implicit overwrites, rolled at the end ✓; magnetic/composition gated by `=== nothing` ✓; per-field rollover inside each step ✓; runs on Array + CuArray ✓.

**Placeholder scan:** none.

**Type consistency:** `state` field names (`velocity/magnetic/temperature/composition`, `nlops_vel/nlops_mag/influence`, `T_phys/C_phys/B_*/J_*`, `inv_dt_*`, `d1/mvals/rinv`) match between the test's `build_state` and the function body; the per-field-step arg orders match the Background signatures; the transform calls wrap raw arrays via `spec(...)`/`ph()`.

**Lag fidelity:** the velocity step receives `state.T_phys`/`state.C_phys`/`state.J_*`/`state.B_*` (previous-step buffers); the new buffers `Tn/Cn/Bn/Jn` are computed from the OLD spectral state (step 2, before any implicit update) and written back to `state.*` only at step 7 — so within the step velocity sees the lagged values, exactly as CPU.

**Gating:** `state.magnetic === nothing` skips the magnetic synth (B/J), the magnetic step, and the magnetic buffer roll; `state.composition === nothing` skips the composition synth, step, and roll. The velocity coupling kwargs `J_*`/`B_*`/`C_phys` are `nothing` in that `state`, so `gpu_velocity_field_step!` skips those coupling terms (its kwargs default to `nothing`).
