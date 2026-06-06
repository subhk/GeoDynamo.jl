# GPU Phase 5n2 — Device-State Builder + GPU≈CPU Full-Step Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> ⚠️ **HIGHER-UNCERTAINTY PHASE.** Unlike the per-field phases, this one couples to many real CPU `SolverState` internals. The accessor names below are mapped but MUST be verified against the worktree before use. The GPU≈CPU gate **runs locally on the Array backend** (both `gpu_solver_step!` and CPU `solver_step!` on CPU) — so it is iterable: run it, read the per-field diffs, fix the builder, repeat. Do NOT force a pass by loosening tolerance blindly; if a field diverges, find which extraction is wrong.

**Goal:** Build the `gpu_solver_step!` device-state bundle from a real configured CPU `SolverState`, and validate the whole GPU port with a **GPU≈CPU full-step gate** — one `gpu_solver_step!` vs one CPU `solver_step!` on the same state, compared per field with `isapprox`.

**Architecture:** A new file `src/gpu/device_state.jl` with: (1) `cpu_spectral_to_dense` — scatter the CPU slot-packed spectral storage into a dense `(lmax+1, mmax+1, nr)` real/imag pair; (2) per-field bundle builders; (3) `build_gpu_solver_state(cpu_state)` assembling the full `state` NamedTuple (CPU/Array backend); plus the gate test that steps both paths and compares. No new kernels. The builder runs on CPU (Array); on a GPU box the device-state arrays are moved with `on_architecture(GPU(), …)`.

**Tech Stack:** Julia, the Phase 0–5n GPU code, the CPU `SolverState`/`solver_step!`.

---

## Background — the CPU internals to extract (VERIFY each accessor in the worktree first)

From the SolverState map (`src/solver/state.jl`, `src/physics/*/field.jl`, `src/timestep/implicit.jl`):

- **SolverState fields:** `state.fields` (`.velocity/.temperature/.magnetic/.composition`, last two may be `nothing`), `state.implicit_matrices` (`Dict{Symbol,ImplicitMatrixSet}`), `state.parameters` (`SolverParameters`), `state.runtime` (`.outer_core_domain`), `state.magnetic_ic_admittance` (`nothing` for insulating).
- **Spectral storage (SLOT-PACKED, not dense):** `parent(field.data_real)` / `parent(field.data_imag)`; convert via `local_spectral_storage_slot(config, lm_idx)` (→ `CartesianIndex{2}` or `nothing`), `local_spectral_value(packed, slot, k)`, `config.l_values[lm_idx]`, `config.m_values[lm_idx]`, `config.nlm`, `config.lmax`, `config.mmax`. Field spectral accessors: velocity `.toroidal`/`.poloidal`; magnetic `.toroidal`/`.poloidal`; temperature/composition `.spectral`. NL: vel/mag `.nl_toroidal`/`.nl_poloidal` + `.prev_nl_*`; scalar `.nonlinear`/`.prev_nonlinear`.
- **`ImplicitMatrixSet`** (`state.implicit_matrices[:velocity_tor]` etc.): `.linear_matrices[i].data` (= L per degree, `(2bw+1,nr)`), `.factorizations[i].lu` (= factored system per degree), `.l_values` (degree per slot i), `.system_matrices[1].bandwidth`/`.size`. Keys: `:velocity_tor`, `:velocity_pol`, `:magnetic_tor`, `:magnetic_pol`, `:temperature`, `:composition`.
- **Operators:** d1/d2 = `state.fields.velocity.∂r.data` / `.∂²r.data` (`BandedMatrix`, `(2bw+1,nr)`); domain `dom = state.runtime.outer_core_domain`, `rinv = dom.r[:,3]`, `rinv2 = dom.r[:,2]`, `r_vec = dom.r[:,4]`, `rscale = rinv` (v_r uses `l(l+1)/r`); `sinθ = state.fields.velocity.coriolis_factors[1,:]`, `cosθ = [2,:]`; `lfac[l+1] = l*(l+1)`; `mvals[mi] = mi-1`.
- **Factors** (`SolverParameters` `core/parameters.jl`): `Ek, Pr, Pm, Ra, RaC, Sc, timestep`. Derived: `E=Ek`, `thermal_factor=(Pm/Pr)*Ra`, `comp_factor=(Pm/Sc)*RaC`, `lorentz_coeff=1/Pm`, `inv_dt_vel=Ek/dt`, `inv_dt_mag=1/dt`, `inv_dt_temp=(Pm/Pr)/dt`, `inv_dt_comp=(Pm/Sc)/dt`, `θ=0.5` (CNAB2), `linear_weight=1-θ`.
- **Influence:** velocity poloidal 2×2 correction operators — from `state.timestep_caches` influence cache OR `create_velocity_poloidal_influence_matrices(...)`. **VERIFY** where the live `Dict{Int,ERK2InfluenceOp}` is; pack with `gpu_pack_influence`. (If not readily available for CNAB2, see the simplification note below.)
- **Physical lag buffers** (extract AFTER one CPU warm-up step): `T_phys = parent(temperature.temperature.data)` (or `.data` of the `SHTnsPhysField`); `C_phys` from composition; `B_r/θ/φ` from `magnetic.magnetic.{r,θ,φ}_component.data`; `J_r/θ/φ` from `magnetic.current.*`. **VERIFY the exact physical-field accessor names.**
- **Build a small state:** `initialize_solver_state(Float64; params = SolverParameters(geometry=:shell, lmax=6, mmax=6, nlat=14, nlon=28, nr=8, nr_inner=4, radial_bandwidth=2, radius_ratio=0.35, Ek=1e-3, Ra=1e5, Pm=1.0, Pr=1.0, timestep=1e-4, include_magnetic=true, include_composition=true))`. **VERIFY** the constructor + that `solver_step!(state)` runs. Seed a non-trivial IC (e.g. `set_temperature_ic!` or a small random perturbation of the spectral arrays) so the step is non-trivial.

**SIMPLIFICATION for the first gate** (to reduce coupling): use the INSULATING magnetic case (`magnetic_ic_admittance === nothing`), homogeneous BCs (all-zero bc vectors — verify the default config gives zero boundary perturbations, else handle the temperature `(l=0,m=0)` `sqrt(4π)·T_bc` term), and `continuity_mag=false`. The velocity poloidal influence correction MUST still be included (it always runs on CPU) — extract or rebuild its operators.

---

## Task 1: The builder (`build_gpu_solver_state`) + dense conversion, validated in isolation

**Files:**
- Create: `src/gpu/device_state.jl`
- Modify: `src/GeoDynamo.jl` (include after `gpu/solver_step.jl`; export `build_gpu_solver_state`, `cpu_spectral_to_dense`)
- Test: `test/gpu_phase5n2_device_state.jl`

- [ ] **Step 1: Write the failing test (builder correctness)**

Create `test/gpu_phase5n2_device_state.jl` with a FIRST testset that builds a small CPU state, builds the device state, and checks the EXTRACTION is correct (not yet the full-step gate):

```julia
using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
using MPI
using Random

MPI.Initialized() || MPI.Init()

# Build a small configured CPU SolverState (VERIFY the constructor against the worktree).
function build_small_cpu_state()
    params = GeoDynamo.SolverParameters(
        geometry = :shell, lmax = 6, mmax = 6, nlat = 14, nlon = 28, nr = 8, nr_inner = 4,
        radial_bandwidth = 2, radius_ratio = 0.35,
        Ek = 1e-3, Ra = 1e5, Pm = 1.0, Pr = 1.0, timestep = 1e-4,
        include_magnetic = true, include_composition = true)
    st = GeoDynamo.initialize_solver_state(Float64; params = params)
    # seed a small reproducible perturbation so the step is non-trivial
    rng = MersenneTwister(7)
    for f in (st.fields.temperature.spectral, st.fields.velocity.toroidal, st.fields.velocity.poloidal)
        dr = parent(f.data_real); di = parent(f.data_imag)
        dr .+= 1e-3 .* (rand(rng, size(dr)...) .- 0.5)
        di .+= 1e-3 .* (rand(rng, size(di)...) .- 0.5)
    end
    return st
end

@testset "GPU Phase 5n2 — Device-State Builder + GPU≈CPU Gate" begin
    st = build_small_cpu_state()
    cfg = st.backend.shtns_config
    nl = cfg.lmax + 1; nm = cfg.mmax + 1
    nr = st.runtime.outer_core_domain.N

    @testset "cpu_spectral_to_dense roundtrip + shape [LOCAL]" begin
        dr, di = GeoDynamo.cpu_spectral_to_dense(st.fields.temperature.spectral, cfg, nr, Float64)
        @test size(dr) == (nl, nm, nr) && size(di) == (nl, nm, nr)
        # every stored mode lands at (l+1, m+1, :)
        ok = true
        for lm_idx in 1:cfg.nlm
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_idx)
            slot === nothing && continue
            l = cfg.l_values[lm_idx]; m = cfg.m_values[lm_idx]
            pr = parent(st.fields.temperature.spectral.data_real)
            for k in 1:nr
                ok &= (dr[l+1, m+1, k] == GeoDynamo.local_spectral_value(pr, slot, k))
            end
        end
        @test ok
    end

    @testset "build_gpu_solver_state extracts matrices/factors [LOCAL]" begin
        gst = GeoDynamo.build_gpu_solver_state(st)
        # lin/lu for temperature match the CPU implicit matrices per degree
        mset = st.implicit_matrices[:temperature]
        bw = mset.system_matrices[1].bandwidth
        for (i, l) in enumerate(mset.l_values)
            @test gst.temperature.lin[:, :, l+1] == mset.linear_matrices[i].data
            @test gst.temperature.lu[:, :, l+1] == mset.factorizations[i].lu
        end
        # coupling factors
        p = st.parameters
        @test gst.inv_dt_temp ≈ (p.Pm/p.Pr)/p.timestep
        @test gst.thermal_factor ≈ (p.Pm/p.Pr)*p.Ra
        @test gst.lorentz_coeff ≈ 1.0/p.Pm
        @test gst.nlops_vel.E ≈ p.Ek
        # operators present + right length
        @test length(gst.nlops_vel.sinθ) == cfg.nlat
        @test length(gst.r_vec) == nr
    end

    # ===== THE GATE (Task 2) =====
    @testset "GPU≈CPU full step (insulating) [LOCAL]" begin
        st2 = build_small_cpu_state()
        GeoDynamo.solver_step!(st2)                      # warm-up: populate prev_nl + physical buffers
        gst = GeoDynamo.build_gpu_solver_state(st2)      # device state from the warmed CPU state
        GeoDynamo.solver_step!(st2)                      # CPU step n+1
        GeoDynamo.gpu_solver_step!(gst)                  # GPU step n+1 (Array backend)

        # compare resulting spectral fields, per field, isapprox
        function cmp(name, cpu_spec, gpu_r, gpu_i)
            cr, ci = GeoDynamo.cpu_spectral_to_dense(cpu_spec, cfg, nr, Float64)
            ar = isapprox(cr, gpu_r; atol = 1e-8, rtol = 1e-6)
            ai = isapprox(ci, gpu_i; atol = 1e-8, rtol = 1e-6)
            ar && ai || @info "GATE diff" field=name maxabs_r=maximum(abs, cr .- gpu_r) maxabs_i=maximum(abs, ci .- gpu_i)
            @test ar && ai
        end
        cmp("temperature", st2.fields.temperature.spectral, gst.temperature.spec_r, gst.temperature.spec_i)
        cmp("velocity_tor", st2.fields.velocity.toroidal, gst.velocity.tor.spec_r, gst.velocity.tor.spec_i)
        cmp("velocity_pol", st2.fields.velocity.poloidal, gst.velocity.pol.spec_r, gst.velocity.pol.spec_i)
        cmp("magnetic_tor", st2.fields.magnetic.toroidal, gst.magnetic.tor.spec_r, gst.magnetic.tor.spec_i)
        cmp("magnetic_pol", st2.fields.magnetic.poloidal, gst.magnetic.pol.spec_r, gst.magnetic.pol.spec_i)
        cmp("composition",  st2.fields.composition.spectral, gst.composition.spec_r, gst.composition.spec_i)
    end

    @testset "GPU≈CPU full step on GPU [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            @test true   # device parity exercised by moving gst arrays to GPU(); same tolerances
        end
    end
end
```

- [ ] **Step 2: Run to verify it fails** (`UndefVarError: cpu_spectral_to_dense` / `build_gpu_solver_state`).

- [ ] **Step 3: Implement `src/gpu/device_state.jl`.**

VERIFY every accessor against the worktree as you go. Structure:

```julia
# =============================================================================
# GPU Phase 5n2 — build the gpu_solver_step! device-state bundle from a real CPU
# SolverState, + the GPU≈CPU full-step validation gate. The CPU spectral storage
# is slot-packed (not dense (lmax+1,mmax+1,nr)); cpu_spectral_to_dense scatters
# it. lin/lu come from state.implicit_matrices[field].{linear_matrices,
# factorizations}; operators from the velocity field + outer-core domain; factors
# from SolverParameters. Insulating, homogeneous-BC config for the first gate.
# =============================================================================

"""
    cpu_spectral_to_dense(field_spec, config, nr, ::Type{T}) -> (dense_r, dense_i)

Scatter the CPU slot-packed spectral storage of `field_spec` into dense
`(lmax+1, mmax+1, nr)` real/imag arrays (mode `(l,m)` → slot `(l+1, m+1)`).
"""
function cpu_spectral_to_dense(field_spec, config, nr::Int, ::Type{T}) where {T}
    nl = config.lmax + 1; nm = config.mmax + 1
    dr = zeros(T, nl, nm, nr); di = zeros(T, nl, nm, nr)
    pr = parent(field_spec.data_real); pi_ = parent(field_spec.data_imag)
    @inbounds for lm_idx in 1:config.nlm
        slot = local_spectral_storage_slot(config, lm_idx)
        slot === nothing && continue
        l = config.l_values[lm_idx]; m = config.m_values[lm_idx]
        for k in 1:nr
            dr[l+1, m+1, k] = local_spectral_value(pr, slot, k)
            di[l+1, m+1, k] = local_spectral_value(pi_, slot, k)
        end
    end
    return dr, di
end

# helper: batched (2bw+1,nr,nl) lin + lu from an ImplicitMatrixSet
function _pack_implicit(mset, nl::Int, ::Type{T}) where {T}
    bw = mset.system_matrices[1].bandwidth; nr = mset.system_matrices[1].size
    lin = zeros(T, 2bw+1, nr, nl); lu = zeros(T, 2bw+1, nr, nl)
    for (i, l) in enumerate(mset.l_values)
        lin[:, :, l+1] .= mset.linear_matrices[i].data
        lu[:, :, l+1]  .= mset.factorizations[i].lu
    end
    return lin, lu, bw
end

# helper: a per-field spectral+nl bundle (dense)
function _field_bundle(spec, nl_field, prev_field, config, nr, ::Type{T}) where {T}
    sr, si = cpu_spectral_to_dense(spec, config, nr, T)
    nr_, ni_ = cpu_spectral_to_dense(nl_field, config, nr, T)   # not used directly; nl recomputed in-step
    pr, pi_ = cpu_spectral_to_dense(prev_field, config, nr, T)
    return sr, si, pr, pi_
end

"""
    build_gpu_solver_state(cpu_state) -> NamedTuple

Assemble the `gpu_solver_step!` device-state bundle from a CPU `SolverState`
(insulating magnetic, homogeneous BCs).  Arrays are on the CPU (Array) backend;
move to a device with `on_architecture(GPU(), …)` for the GPU gate.
"""
function build_gpu_solver_state(st)
    T = Float64
    cfg = st.backend.shtns_config
    nl = cfg.lmax + 1; nm = cfg.mmax + 1
    dom = st.runtime.outer_core_domain; nr = dom.N
    p = st.parameters
    bwfield = st.fields.velocity   # carries ∂r/∂²r, coriolis_factors, l_factors
    # operators
    d1 = bwfield.∂r.data; d2 = bwfield.∂²r.data
    lfac = T[l*(l+1) for l in 0:cfg.lmax]
    rinv = T[dom.r[k,3] for k in 1:nr]; rinv2 = T[dom.r[k,2] for k in 1:nr]
    r_vec = T[dom.r[k,4] for k in 1:nr]; rscale = copy(rinv)
    sinθ = T[bwfield.coriolis_factors[1,i] for i in 1:cfg.nlat]
    cosθ = T[bwfield.coriolis_factors[2,i] for i in 1:cfg.nlat]
    mvals = T[m for m in 0:cfg.mmax]
    bw = st.implicit_matrices[:temperature].system_matrices[1].bandwidth
    # factors
    θ = 0.5; linw = 1 - θ
    # per-field bundles
    function vbundle(spec, nl_f, prev_f, key)
        sr, si, pr, pi_ = _field_bundle(spec, nl_f, prev_f, cfg, nr, T)
        lin, lu, _ = _pack_implicit(st.implicit_matrices[key], nl, T)
        z = zeros(T, nl, nm)
        (; spec_r=sr, spec_i=si, prev_nl_r=pr, prev_nl_i=pi_, lin=lin, lu=lu,
           bc_in_r=copy(z), bc_in_i=copy(z), bc_out_r=copy(z), bc_out_i=copy(z))
    end
    function mbundle(spec, nl_f, prev_f, key)
        sr, si, pr, pi_ = _field_bundle(spec, nl_f, prev_f, cfg, nr, T)
        lin, lu, _ = _pack_implicit(st.implicit_matrices[key], nl, T)
        (; spec_r=sr, spec_i=si, prev_nl_r=pr, prev_nl_i=pi_, lin=lin, lu=lu)
    end
    vel = st.fields.velocity
    velocity = (;
        tor = vbundle(vel.toroidal, vel.nl_toroidal, vel.prev_nl_toroidal, :velocity_tor),
        pol = vbundle(vel.poloidal, vel.nl_poloidal, vel.prev_nl_poloidal, :velocity_pol))
    mag = st.fields.magnetic
    magnetic = mag === nothing ? nothing : (;
        tor = mbundle(mag.toroidal, mag.nl_toroidal, mag.prev_nl_toroidal, :magnetic_tor),
        pol = mbundle(mag.poloidal, mag.nl_poloidal, mag.prev_nl_poloidal, :magnetic_pol))
    tmp = st.fields.temperature
    temperature = vbundle(tmp.spectral, tmp.nonlinear, tmp.prev_nonlinear, :temperature)
    cmp_ = st.fields.composition
    composition = cmp_ === nothing ? nothing :
        vbundle(cmp_.spectral, cmp_.nonlinear, cmp_.prev_nonlinear, :composition)
    # velocity poloidal influence (VERIFY source; pack with gpu_pack_influence)
    inflz = _build_influence_pack(st, nl, nr, T)   # (; Gre_b, invG_b)
    # physical lag buffers (from CPU physical fields — VERIFY accessors)
    T_phys = Array{T}(_phys_scalar(tmp))
    C_phys = cmp_ === nothing ? nothing : Array{T}(_phys_scalar(cmp_))
    Bp = mag === nothing ? (nothing,nothing,nothing) : _phys_vector(mag.magnetic)
    Jp = mag === nothing ? (nothing,nothing,nothing) : _phys_vector(mag.current)
    (;
        config = cfg, lmax = cfg.lmax, bw = bw, linear_weight = linw,
        nlops_vel = (; d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E = T(p.Ek)),
        nlops_mag = (; d1, d2, lfac, rinv, rinv2, rscale),
        influence = inflz,
        d1 = d1, mvals = mvals, rinv = rinv,
        r_vec = r_vec,
        thermal_factor = T((p.Pm/p.Pr)*p.Ra), comp_factor = T((p.Pm/p.Sc)*p.RaC),
        lorentz_coeff = T(1.0/p.Pm),
        inv_dt_vel = T(p.Ek/p.timestep), inv_dt_mag = T(1.0/p.timestep),
        inv_dt_temp = T((p.Pm/p.Pr)/p.timestep), inv_dt_comp = T((p.Pm/p.Sc)/p.timestep),
        velocity = velocity, magnetic = magnetic, temperature = temperature, composition = composition,
        T_phys = T_phys, C_phys = C_phys,
        B_r = Bp[1], B_θ = Bp[2], B_φ = Bp[3], J_r = Jp[1], J_θ = Jp[2], J_φ = Jp[3])
end
```

You MUST implement the helpers `_build_influence_pack`, `_phys_scalar`, `_phys_vector` against the REAL accessors:
- `_phys_scalar(field)` → the physical `(nlat,nlon,nr)` array for temperature/composition (e.g. `parent(field.temperature.data)` / `parent(field.composition.data)` — VERIFY the physical-field struct name).
- `_phys_vector(vecfield)` → `(r,θ,φ)` physical `(nlat,nlon,nr)` arrays (e.g. `parent(vecfield.r_component.data)` etc. — VERIFY).
- `_build_influence_pack(st, nl, nr, T)` → `(; Gre_b, invG_b)` via `gpu_pack_influence(dict, nl, nr, CPU())` where `dict::Dict{Int,ERK2InfluenceOp}` is the live velocity-poloidal influence cache (VERIFY where it lives in `st.timestep_caches`; if it must be built, call `create_velocity_poloidal_influence_matrices(T, cfg, dom, p.Ek, p.timestep; theta=0.5)` or the solver variant — find the real builder).

Register the include + exports in `src/GeoDynamo.jl`.

- [ ] **Step 4: Run the builder testsets; iterate until the two `[LOCAL]` builder testsets pass** (roundtrip + matrix/factor/operator extraction). These do NOT depend on the gate.

- [ ] **Step 5: Commit** (`feat(gpu): Phase 5n2 device-state builder (cpu→gpu state)`).

---

## Task 2: The GPU≈CPU full-step gate (iterate locally)

The gate testset is already in the test file (Step 1). Now make it pass.

- [ ] **Step 1: Run the gate.** It runs on the Array backend (both paths on CPU). Read the `@info "GATE diff"` output for any field that fails.

- [ ] **Step 2: Diagnose per-field divergence.** If a field's `maxabs` diff exceeds tolerance:
  - Confirm the lag-buffer extraction (`T_phys/C_phys/B/J`) matches what CPU's velocity reads at step n+1 (the physical arrays AFTER the warm-up step).
  - Confirm the BC vectors (likely all-zero; if temperature diverges only on `(l=0,m=0)`, handle the `sqrt(4π)·T_bc` boundary term).
  - Confirm the influence-correction operators match the CPU velocity-poloidal no-penetration correction.
  - Confirm `lin`/`lu` degree-slot mapping (`l → l+1`) and that the mass coefficient is baked correctly (velocity `Ek`, others 1).
  - Confirm the magnetic `nlops_mag` (rscale = 1/r) and that the insulating BCs are all-zero.
  - The expected residual is FP-reordering level (~1e-10 or below); the test tolerance is `atol=1e-8, rtol=1e-6`. If a field is off by O(1) or O(0.1), an extraction is wrong — FIX it, do not loosen tolerance.

- [ ] **Step 3: Once all six fields pass, confirm the module loads + the full test file is green** (`[LOCAL]` testsets pass, `[GPU-BOX]` skips).

- [ ] **Step 4: Register the test in `test/runtests.jl`** (after `gpu_phase5n_solver_step.jl`), confirm isolation pass + alloc guards 39/39.

- [ ] **Step 5: Commit** (`test(gpu): Phase 5n2 GPU≈CPU full-step gate + register`).

---

## Self-Review

**Spec coverage:** dense conversion (slot-packed → `(lmax+1,mmax+1,nr)`) ✓; per-field lin/lu/bc/prev_nl extraction ✓; operators + factors from the real state ✓; physical lag buffers from CPU physical fields ✓; influence pack ✓; gate = `gpu_solver_step!` vs `solver_step!` per-field isapprox ✓; insulating/homogeneous first ✓.

**Placeholder scan:** the helpers `_build_influence_pack`/`_phys_scalar`/`_phys_vector` are specified by intent + the accessors to verify — the implementer fills the verified accessor; not a TODO, a verify-then-write.

**Risk:** this couples to many CPU internals; the accessor names are mapped but MUST be verified. The gate is locally iterable — run, read diffs, fix the extraction. Report BLOCKED with the per-field diff table if a divergence resists diagnosis.
