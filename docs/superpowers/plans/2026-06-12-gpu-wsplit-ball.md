# GPU W-Split + Ball Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `gpu_solver_step!` reproduces CPU `solver_step!` (CNAB2, full MHD) on the Array backend for shell AND ball geometry — un-gating the Stage-2/4B GPU paths.

**Architecture:** Mirror-the-CPU batched: every changed CPU kernel re-implemented in the GPU module's established style (split-complex `(lmax+1, mmax+1, nr)` arrays, batched per-l banded ops, per-level SHTnsKit transforms). Matrices/LUs/Green columns/residual rows are host-built and device-copied — only step algebra needs GPU code. Spec: `docs/superpowers/specs/2026-06-12-gpu-wsplit-ball-port-design.md`.

**Tech Stack:** Julia 1.11, KernelAbstractions (CPU backend locally; CUDA via extension on the GPU box), SHTnsKit serial transforms, existing GPU phase-0–6 module (`src/gpu/`).

---

## Critical context for the implementer

- **Worktree:** `/Users/subha/Documents/GitHub/GeoDynamo-ball`, branch `feat/gpu-wsplit-ball` (based on the ball port @ `41af101`). NEVER touch `/Users/subha/Documents/GitHub/GeoDynamo.jl`.
- **Julia:** `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=.` from the worktree root. Long output → /tmp files; never pipe `Pkg.test` through `tail`. Prefix test runs with `GEODYNAMO_TEST_MPI_FINALIZE=false` when chaining includes.
- **GPU module idioms (read these files first, they are short):**
  - `src/gpu/banded_solve.jl`: `gpu_pack_banded_lu(lus, arch)` → `(2bw+1, nr, nl)` batched LU (dim-3 = l-slot = l+1); `gpu_batched_banded_solve!(X, B, lu_b, bw)` solves every `(l,m)` mode's radial profile, in-place OK.
  - `src/gpu/spectral_curl.jl`: `gpu_batched_banded_matvec!(Y, X, mat, bw)` — l-INDEPENDENT banded op applied to all modes, ascending-j accumulation (bit-exact vs the CPU loop); `Y` must not alias `X`.
  - Split-complex convention: every spectral quantity is a pair `(x_r, x_i)` of real `(nl, nm, nr)` arrays; transforms loop radial levels and call SHTnsKit on materialized `(nl, nm)` complex matrices (`complex.(view(...), view(...))` — NEVER a bare `@view`, the CuArray method dispatch needs a materialized array).
  - Per-l data is packed dense by l-slot (`l+1`), missing degrees zero (see `_pack_implicit` in `src/gpu/device_state.jl`).
  - `arch_of`, `on_architecture`, `allocate_gpu_physical_field`, `GPUSpectralField` — see `src/gpu/fields.jl`/`device.jl`.
- **CPU reference functions (the truth — mirror them exactly):**
  - Transforms: `vector_spectral_to_physical_disttranspose!` + `_spheroidal_from_poloidal!` + `_solenoidal_vr_factor` and `vector_physical_to_spectral!` + `_poloidal_from_radial_q!` in `src/solver/numerics.jl`.
  - Velocity projection: `finish_velocity_nonlinear!` + `_poloidal_force_projection!` in `src/physics/velocity/solver.jl` (N_W = ∂r(r·S_F) − Q_F).
  - Induction: `apply_induction_nonlinear!` + `_induction_curl_potentials!` in `src/solver/numerics.jl` (P = −r·T_E, T = −(Q_E − ∂r(r·S_E))/r).
  - Scalar gradient fix: `transform_field_and_gradients_to_physical!` + `_scale_tangential_gradient_by_inv_r!` in `src/physics/nonlinear.jl`.
  - Vorticity/current: the Stage-2 curl in `src/solver/numerics.jl` (grep `r_inv` near the vorticity assembly: `T_ω = rinv·(d2·P − λ·rinv²·P)`, `P_ω = −r·T`).
  - W-split step: `_apply_poloidal_wsplit_cnab2!` in `src/physics/velocity/solver.jl`; matrices: `create_velocity_poloidal_split_matrices` in `src/bcs/velocity_bc.jl`; struct `PoloidalSplitMatrices` in `src/solver/state.jl` (fields incl. `ball`, `reg_r_inv`).
- **Equivalence tolerances:** kernels built from `gpu_batched_banded_matvec!` + broadcasts in CPU loop order are expected BIT-EXACT vs the CPU on the Array backend (assert `==` or `atol=0`); anything involving `dot`/BLAS on the CPU side (W-split residuals, influence) is sub-ulp — assert `≤ 1e-13` relative and DOCUMENT the reason in the test. Step/trajectory gates: max-abs diff ≤ 1e-12·scale.
- **Tests:** `ls test/gpu_*` for the full phase-test list. PR #59 wrapped the Stage-2-dependent assertions in `@test_throws ErrorException` gates — each task flips ITS gates back to real assertions with freshly CPU-generated expectations (never reuse pre-Stage-2 snapshots). `[GPU-BOX]` testsets skip without CUDA — leave their markers intact, update their bodies to match the new signatures.
- **Test traps:** static checks pin source text — repoint, don't weaken. `similar()` scratch must be fully written before read (zero-init if accumulating).

---

### Task 0: Baseline

- [ ] **Step 1: Confirm branch + run the current GPU test set to record the gated baseline**

```bash
cd /Users/subha/Documents/GitHub/GeoDynamo-ball
git branch --show-current   # must print feat/gpu-wsplit-ball
ls test/gpu_*.jl > /tmp/gpu_tests.txt; cat /tmp/gpu_tests.txt
for f in $(ls test/gpu_*.jl | xargs -n1 basename); do GEODYNAMO_TEST_MPI_FINALIZE=false ~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e "using GeoDynamo; include(\"test/$f\")" > /tmp/g0_$f.log 2>&1; echo "$f exit=$?"; done
```
All must exit 0 at baseline (gates are `@test_throws`, so they pass). Record which files contain `@test_throws` Stage-2 gates:
```bash
grep -ln "test_throws.*ErrorException\|_GPU_VECTOR_STAGE2_MSG\|Stage-2" test/gpu_*.jl
```

---

### Task 1: Vector transforms — Stage-2 solenoidal convention

**Files:**
- Modify: `src/gpu/vector_transform.jl` (replace both gated bodies; delete `_GPU_VECTOR_STAGE2_MSG`)
- Modify: every call site of the two functions (`grep -rn "gpu_vector_spectral_to_physical!\|gpu_vector_physical_to_spectral!" src/gpu/ test/`) — signatures change
- Test: `test/gpu_phase3_vector_transform.jl`

- [ ] **Step 1: Write the failing kernel-equivalence test.** In `test/gpu_phase3_vector_transform.jl`, replace the Stage-2 `@test_throws` gates with real equivalence testsets. Build a small CPU fixture exactly the way the existing non-gated testsets in that file do (read the file: it has a config + random band-limited spectral fixture pattern and a `cpu_spectral_to_dense` usage). The new assertions:

```julia
@testset "GPU vector synthesis == CPU (Stage-2 solenoidal)" begin
    # fixture: cfg, domain (SHELL), random smooth (tor, pol) on the CPU side,
    # CPU reference via GeoDynamo.vector_spectral_to_physical! on real
    # SHTnsSpecField/SHTnsVectorField objects (the public CPU path), then
    # compare against the GPU dense-array result level by level.
    # GPU inputs: dense (nl,nm,nr) split-complex scattered from the same CPU
    # spectral fields via GeoDynamo.cpu_spectral_to_dense.
    # d1 = velocity_fields.∂r.data (the SAME banded operator the CPU uses),
    # rinv = domain.r[1:nr,3], rscale = domain.r[1:nr,2]  (λP/r² convention!)
    # ... allocate GPU fields, call:
    GeoDynamo.gpu_vector_spectral_to_physical!(vr, vθ, vφ, tor_g, pol_g, gcfg,
        lfac, rscale, d1_dev, rinv_dev, bw)
    @test maximum(abs, vr.data .- vr_cpu) <= 1e-13 * max(1.0, maximum(abs, vr_cpu))
    @test maximum(abs, vθ.data .- vθ_cpu) <= 1e-13 * max(1.0, maximum(abs, vθ_cpu))
    @test maximum(abs, vφ.data .- vφ_cpu) <= 1e-13 * max(1.0, maximum(abs, vφ_cpu))
end

@testset "GPU vector analysis == CPU (Q-based + raw mode)" begin
    # default mode: random solenoidal physical field generated by the CPU
    # synthesis from random (tor,pol); analyze on both paths; poloidal must
    # come back through Q (vr consumed); compare (tor,pol) spectral.
    GeoDynamo.gpu_vector_physical_to_spectral!(tor_g, pol_g, vθ_g, vφ_g, gcfg;
        vr = vr_g, lfac = lfac, r2 = r2_dev)
    # raw mode: raw_spheroidal=true returns the raw sphtor S in the poloidal slot
    GeoDynamo.gpu_vector_physical_to_spectral!(torR_g, polR_g, vθ_g, vφ_g, gcfg;
        raw_spheroidal = true)
    # CPU raw reference: GeoDynamo.vector_physical_to_spectral!(..., raw_spheroidal=true)
end
```

(The exact fixture plumbing follows the file's existing pattern — read it; the assertions and the new signatures above are the contract. Synthesis tolerance is 1e-13 because the CPU `_spheroidal_from_poloidal!` may use `mul!` whose accumulation order differs from the GPU kernel; if you measure exact equality, tighten to `== 0` with a comment.)

- [ ] **Step 2: Run — must FAIL at the `error(_GPU_VECTOR_STAGE2_MSG)`.**

- [ ] **Step 3: Implement.** New signatures and bodies in `src/gpu/vector_transform.jl`:

```julia
"""
    gpu_vector_spectral_to_physical!(vr, vθ, vφ, tor, pol, config, lfac, rscale,
                                     d1, rinv, bw; raw_spheroidal=false) -> nothing

Stage-2 solenoidal synthesis: tangential (vθ,vφ) per level via
synthesis_sphtor(S, T) with S = (d1·P)·(1/r) (raw_spheroidal=true passes the
stored poloidal coefficients directly as S — the tangential-basis primitive);
radial v_r per level via scalar synthesis of P·lfac·rscale with
rscale = 1/r² (u_r = l(l+1)·P/r²). Mirrors the CPU
vector_spectral_to_physical_disttranspose! convention.
"""
function gpu_vector_spectral_to_physical!(vr::GPUPhysicalField, vθ::GPUPhysicalField,
        vφ::GPUPhysicalField, tor::GPUSpectralField, pol::GPUSpectralField, config,
        lfac, rscale, d1, rinv, bw::Int; raw_spheroidal::Bool = false)
    sht = config.sht_config
    nr = pol.nr
    # Spheroidal scalar S: raw mode = the stored coefficients; solenoidal mode
    # = (∂_r P)/r via the batched banded d1 + 1/r broadcast (NOT in place — P
    # is still needed for v_r).
    S_r = similar(pol.data_real); S_i = similar(pol.data_imag)
    if raw_spheroidal
        S_r .= pol.data_real; S_i .= pol.data_imag
    else
        gpu_batched_banded_matvec!(S_r, pol.data_real, d1, bw)
        gpu_batched_banded_matvec!(S_i, pol.data_imag, d1, bw)
        ri = reshape(rinv, 1, 1, :)
        @. S_r *= ri
        @. S_i *= ri
    end
    # v_r source coefficients: P·λ·rscale (rscale = 1/r² in the Stage-2 convention).
    vr_alm_r = similar(pol.data_real); vr_alm_i = similar(pol.data_imag)
    gpu_vr_scale!(vr_alm_r, vr_alm_i, pol.data_real, pol.data_imag, lfac, rscale)
    for k in 1:nr
        S_k = complex.(view(S_r, :, :, k), view(S_i, :, :, k))
        T_k = complex.(view(tor.data_real, :, :, k), view(tor.data_imag, :, :, k))
        vt, vp = _vector_synth_sphtor(sht, S_k, T_k)
        vθ.data[:, :, k] .= vt
        vφ.data[:, :, k] .= vp
        vra_k = complex.(view(vr_alm_r, :, :, k), view(vr_alm_i, :, :, k))
        vr.data[:, :, k] .= _scalar_synth(sht, vra_k)
    end
    return nothing
end

"""
    gpu_vector_physical_to_spectral!(tor, pol, vθ, vφ, config;
                                     vr=nothing, lfac=nothing, r2=nothing,
                                     raw_spheroidal=false) -> nothing

Stage-2 analysis. Per level analysis_sphtor(vθ, vφ) → raw (S, T); toroidal ← T.
raw_spheroidal=true stores raw S into the poloidal slot. Default mode performs
the Q-based poloidal recovery (requires vr, lfac, r2): Q = scalar analysis of
vr; P = r²·Q/λ, with the l=0 slot zeroed. Mirrors CPU
vector_physical_to_spectral! + _poloidal_from_radial_q!.
"""
function gpu_vector_physical_to_spectral!(tor::GPUSpectralField, pol::GPUSpectralField,
        vθ::GPUPhysicalField, vφ::GPUPhysicalField, config;
        vr = nothing, lfac = nothing, r2 = nothing, raw_spheroidal::Bool = false)
    sht = config.sht_config
    nr = pol.nr
    for k in 1:nr
        vt_k = vθ.data[:, :, k]
        vp_k = vφ.data[:, :, k]
        S_k, T_k = _vector_anal_sphtor(sht, vt_k, vp_k)
        pol.data_real[:, :, k] .= real.(S_k)
        pol.data_imag[:, :, k] .= imag.(S_k)
        tor.data_real[:, :, k] .= real.(T_k)
        tor.data_imag[:, :, k] .= imag.(T_k)
    end
    raw_spheroidal && return nothing
    (vr === nothing || lfac === nothing || r2 === nothing) && throw(ArgumentError(
        "gpu_vector_physical_to_spectral!: default (solenoidal) mode requires vr, lfac, r2"))
    # Q-based poloidal recovery: P = r²·Q/λ (λ=l(l+1); l=0 slot → 0).
    for k in 1:nr
        q_k = _scalar_anal(sht, vr.data[:, :, k])
        pol.data_real[:, :, k] .= real.(q_k)
        pol.data_imag[:, :, k] .= imag.(q_k)
    end
    inv_lfac = similar(lfac)
    @. inv_lfac = ifelse(lfac > 0, 1 / lfac, zero(eltype(lfac)))
    lf = reshape(inv_lfac, :, 1, 1)
    rr2 = reshape(r2, 1, 1, :)
    @. pol.data_real *= lf * rr2
    @. pol.data_imag *= lf * rr2
    return nothing
end
```

(`_scalar_anal` — the per-level scalar analysis helper; check its actual name in `src/gpu/scalar_transform.jl` and use that. If no analysis helper exists there, it does for `gpu_scalar_physical_to_spectral!` — reuse its inner call.) Delete `_GPU_VECTOR_STAGE2_MSG` and the dead old bodies. Update ALL call sites (velocity_nonlinear ×3, magnetic_nonlinear ×2, solver_step ×3) to pass `d1, rinv, bw` — every caller's `nlops` bundle already carries `d1`/`rinv`/`bw` or has them in scope (solver_step has `state.nlops_*` and `bw`); the `rscale` packed in device_state must CHANGE to `rinv2` — that is Task 7's wiring, but to keep this task green, update `device_state.jl`'s `rscale` packing NOW (grep `rscale` there; it currently packs `1/r` for the solver path — switch to the `r[:,2]` column with a Stage-2 comment).

- [ ] **Step 4: Run the phase-3 test (PASS) + the other gpu tests that compile against the changed signatures:**

```bash
for f in $(ls test/gpu_*.jl | xargs -n1 basename); do GEODYNAMO_TEST_MPI_FINALIZE=false ~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e "using GeoDynamo; include(\"test/$f\")" > /tmp/g1_$f.log 2>&1; echo "$f exit=$?"; grep -E "Fail|Error" /tmp/g1_$f.log | head -2; done
```
Files whose gates still expect `error()` from downstream paths (velocity/magnetic nonlinear, step) keep passing via their own `@test_throws` only if those paths still throw — they no longer will from the transforms. EXPECTED fallout: their gated testsets may now FAIL because the old expectations were pre-Stage-2. For each: if the testset is within THIS task's scope (pure transform), fix here; if it belongs to a later task (nonlinear/step equivalence), convert the gate to `@test_skip` with a `# un-gated in Task N` comment — NEVER leave a silently-wrong assertion.

- [ ] **Step 5: Commit**

```bash
git add src/gpu/vector_transform.jl src/gpu/velocity_nonlinear.jl src/gpu/magnetic_nonlinear.jl src/gpu/solver_step.jl src/gpu/device_state.jl test/gpu_phase3_vector_transform.jl
git commit -m "feat(gpu): Stage-2 solenoidal vector transforms — S=P'/r, vr=lP/r^2, Q-based analysis"
```
(plus any `@test_skip` conversions in other gpu test files)

---

### Task 2: Spectral curl — Stage-2 vorticity/current formulas

**Files:**
- Modify: `src/gpu/spectral_curl.jl` (`gpu_spectral_curl!` formula block)
- Test: `test/gpu_phase5a_spectral_curl.jl`

- [ ] **Step 1: Update the test.** The CPU reference is the Stage-2 vorticity: `dst_tor = rinv·(d2·P − λ·rinv²·P)`, `dst_pol = −r·T`. Generate expectations by calling the CPU curl on a real velocity-fields fixture (grep the CPU vorticity assembly in `src/solver/numerics.jl` — the function that fills `vorticity` spectral from `(toroidal, poloidal)`; use IT, not a hand-rolled formula). Assert GPU == CPU bit-exact if the accumulation order matches (the kernel uses the same ascending-j loop), else ≤1e-13. NOTE the signature gains `r_vec` (the curl now needs r, not just 1/r):

```julia
GeoDynamo.gpu_spectral_curl!(dst_tor_r, dst_tor_i, dst_pol_r, dst_pol_i,
    src_tor_r, src_tor_i, src_pol_r, src_pol_i, d1, d2, lfac, rinv, rinv2, r_vec, bw)
```

- [ ] **Step 2: Run — FAIL (old formula / no r_vec arg).**

- [ ] **Step 3: Implement.** In `gpu_spectral_curl!`: add `r_vec` argument (after `rinv2`); replace the formula block:

```julia
    # Stage-2 verified curl projections (see the double-curl design spec):
    #   T_curl = (d2·P − λ/r²·P)/r        P_curl = −r·T
    # d1 is retained in the signature for callers that batch-pack operators,
    # but the Stage-2 toroidal no longer uses the 2/r·d1 term.
    lf  = reshape(lfac, :, 1, 1)
    ri  = reshape(rinv, 1, 1, :)
    ri2 = reshape(rinv2, 1, 1, :)
    rr  = reshape(r_vec, 1, 1, :)
    @. dst_tor_r = ri * (d2Pr - lf * ri2 * src_pol_r)
    @. dst_tor_i = ri * (d2Pi - lf * ri2 * src_pol_i)
    @. dst_pol_r = -rr * src_tor_r
    @. dst_pol_i = -rr * src_tor_i
```
The two d1 matvecs become dead — delete them (and their scratch). Update the docstring. Update ALL call sites (`grep -rn "gpu_spectral_curl!" src/ test/`) to pass `r_vec` — callers' bundles need `r` (add `r = domain.r[1:nr, 4]` to the `nlops` packing in `device_state.jl` and thread it; solver_step's current-density call included).

- [ ] **Step 4: Run phase-5a + the full gpu set (same loop as Task 1 Step 4; convert any newly-broken later-task gates to `@test_skip # un-gated in Task N`).**

- [ ] **Step 5: Commit**

```bash
git add src/gpu/spectral_curl.jl src/gpu/velocity_nonlinear.jl src/gpu/magnetic_nonlinear.jl src/gpu/solver_step.jl src/gpu/device_state.jl test/gpu_phase5a_spectral_curl.jl
git commit -m "feat(gpu): Stage-2 spectral curl — T=(P''-lP/r^2)/r, P=-rT"
```

---

### Task 3: Scalar tangential gradient — the sinθ fix

**Files:**
- Modify: `src/gpu/scalar_nonlinear.jl` (the gradient-to-physical pathway; read it first — it composes `gpu_scalar_gradient!` + scalar synthesis into the advection)
- Modify: `src/gpu/scalar_gradient.jl` (the θ-recurrence path loses its physical-advection consumer; keep the kernels — they still serve the radial/spectral uses — but the tangential PHYSICAL gradient route changes)
- Test: `test/gpu_phase5b_scalar_gradient.jl` and/or the scalar-advection equivalence test (find it: `grep -ln "advection\|scalar_nonlinear" test/gpu_*.jl`)

- [ ] **Step 1: Write the failing test.** CPU reference: `transform_field_and_gradients_to_physical!(𝔽, ws, domain)` (src/physics/nonlinear.jl) — the FIXED path: tangential gradient = raw sphtor synthesis with S = the scalar's spectral coefficients (T-input zero) scaled by 1/r afterwards; radial = scalar synthesis of d1·s. Build a temperature-field fixture, run the CPU function, capture `parent(𝔽.gradient.θ_component.data)` etc., and assert the GPU pathway matches ≤1e-13 (sphtor internals differ in order from the recurrence they replace — do NOT expect bit-exactness vs the OLD GPU path; expect it vs CPU because both now call the same per-level sphtor).

- [ ] **Step 2: Run — FAIL (GPU still produces the sinθ-weighted tangential gradient).**

- [ ] **Step 3: Implement.** In the GPU scalar nonlinear pathway, replace the physical tangential-gradient production: instead of synthesizing the θ-recurrence spectral gradients, call the Task-1 transform in raw mode with a ZERO toroidal input and the scalar coefficients as the spheroidal input, then scale tangential by 1/r:

```julia
    # Tangential gradient via the raw sphtor basis (S∇₁Y): gθ = ∂θs, gφ = (1/sinθ)∂φs,
    # then ×1/r — the exact-gradient fix (CPU: transform_field_and_gradients_to_physical!).
    zt_r = similar(s_r); zt_i = similar(s_i)            # zero toroidal input
    fill!(zt_r, zero(eltype(zt_r))); fill!(zt_i, zero(eltype(zt_i)))
    gr_phys = ...  # radial: scalar synthesis of (d1·s) — unchanged route
    gpu_vector_spectral_to_physical!(gdump, gθ_phys, gφ_phys,
        spec(zt_r, zt_i), spec(s_r, s_i), config, lfac, rscale, d1, rinv, bw;
        raw_spheroidal = true)
    ri3 = reshape(rinv, 1, 1, :)
    @. gθ_phys.data *= ri3
    @. gφ_phys.data *= ri3
```
(`gdump` = scratch physical field for the unused radial output. Read the actual function in `scalar_nonlinear.jl` to place this — the advection then dots (u_r,u_θ,u_φ)·(g_r,g_θ,g_φ) as before.)

- [ ] **Step 4: Run the gradient/advection gpu tests + the scalar step test (`gpu_phase5e*`/`5f*` if present — `ls test/gpu_*`); convert later-task gate fallout to `@test_skip` as before.**

- [ ] **Step 5: Commit**

```bash
git add src/gpu/scalar_nonlinear.jl src/gpu/scalar_gradient.jl test/
git commit -m "fix(gpu): exact tangential scalar gradient via raw sphtor — port the sinth advection fix"
```

---

### Task 4: Velocity nonlinear — Stage-4B force projection (N_W)

**Files:**
- Modify: `src/gpu/velocity_nonlinear.jl` (step 5: the analyze)
- Test: the velocity-nonlinear equivalence test (`grep -ln "velocity_nonlinear" test/gpu_*.jl`)

- [ ] **Step 1: Write the failing test.** CPU reference: run `compute_solver_nonlinear_terms!` (or the narrower `finish_velocity_nonlinear!` on a prepared force) on a real shell SolverState fixture with a thermal seed, capture `velocity.nl_toroidal/nl_poloidal` dense; GPU: `gpu_velocity_nonlinear!` on the scattered inputs with the same coupling kwargs. Assert ≤1e-13 (multiple transforms chain). IMPORTANT: the CPU fixture must call `initialize_solver_fields!` first and refresh scalar physical fields the way `compute_solver_nonlinear_terms!` does — copy the fixture idiom from the existing 5g/5i test file.

- [ ] **Step 2: Run — FAIL (GPU still discards the radial force and uses the legacy tangential-only projection).**

- [ ] **Step 3: Implement.** Replace step 5 of `gpu_velocity_nonlinear!` (and extend the signature with `r_vec` if not already threaded by Task 2):

```julia
    # 5. Stage-4B momentum projections (CPU: finish_velocity_nonlinear! +
    #    _poloidal_force_projection!):
    #    raw sphtor analysis → (T_F, S_F); Q_F = scalar analysis of adv_r;
    #    nl_tor = T_F;  nl_pol = N_W = ∂r(r·S_F) − Q_F.
    sf_r = similar(nl_pol_r); sf_i = similar(nl_pol_i)
    gpu_vector_physical_to_spectral!(spec(nl_tor_r, nl_tor_i), spec(sf_r, sf_i),
        aθ, aφ, config; raw_spheroidal = true)
    qf_r = similar(nl_pol_r); qf_i = similar(nl_pol_i)
    gpu_scalar_physical_to_spectral!(spec(qf_r, qf_i), ar, config)   # Q_F
    rr = reshape(r_vec, 1, 1, :)
    @. sf_r *= rr
    @. sf_i *= rr                                   # r·S_F (in place; S_F dead after)
    gpu_batched_banded_matvec!(nl_pol_r, sf_r, d1, bw)   # ∂r(r·S_F)
    gpu_batched_banded_matvec!(nl_pol_i, sf_i, d1, bw)
    @. nl_pol_r -= qf_r
    @. nl_pol_i -= qf_i
```
(Check the actual name/signature of the GPU scalar analysis in `src/gpu/scalar_transform.jl` and adapt. `adv_r` (`ar`) now carries buoyancy — the whole point: Q_F transmits it to N_W.)

- [ ] **Step 4: Run the velocity-nonlinear tests; later-task fallout → `@test_skip`.**

- [ ] **Step 5: Commit**

```bash
git add src/gpu/velocity_nonlinear.jl test/
git commit -m "feat(gpu): Stage-4B velocity force projection — N_W = dr(r S_F) − Q_F (buoyancy lives)"
```

---

### Task 5: Magnetic induction — Stage-4A curl potentials

**Files:**
- Modify: `src/gpu/magnetic_nonlinear.jl` (steps 3–4)
- Test: the magnetic-nonlinear equivalence test (`grep -ln "magnetic_nonlinear" test/gpu_*.jl`)

- [ ] **Step 1: Failing test** — CPU reference `apply_induction_nonlinear!` on a full-MHD shell fixture; capture `magnetic.nl_toroidal/nl_poloidal` dense; assert GPU ≤1e-13.

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement.** Replace steps 3–4 of `gpu_magnetic_nonlinear!`:

```julia
    # 3. raw sphtor analysis of u×B tangential → (T_E, S_E); Q_E from (u×B)_r.
    te_r = similar(nl_tor_r); te_i = similar(nl_tor_i)
    se_r = similar(nl_pol_r); se_i = similar(nl_pol_i)
    gpu_vector_physical_to_spectral!(spec(te_r, te_i), spec(se_r, se_i),
        ubθ, ubφ, config; raw_spheroidal = true)
    qe_r = similar(nl_pol_r); qe_i = similar(nl_pol_i)
    gpu_scalar_physical_to_spectral!(spec(qe_r, qe_i), ubr, config)
    # 4. Stage-4A curl potentials (CPU: _induction_curl_potentials!):
    #    nl_pol = −r·T_E;   nl_tor = −(Q_E − ∂r(r·S_E))/r.
    rr = reshape(r_vec, 1, 1, :)
    ri = reshape(rinv, 1, 1, :)
    @. nl_pol_r = -rr * te_r
    @. nl_pol_i = -rr * te_i
    @. se_r *= rr
    @. se_i *= rr                                   # r·S_E in place
    gpu_batched_banded_matvec!(nl_tor_r, se_r, d1, bw)   # ∂r(r·S_E)
    gpu_batched_banded_matvec!(nl_tor_i, se_i, d1, bw)
    @. nl_tor_r = -ri * (qe_r - nl_tor_r)
    @. nl_tor_i = -ri * (qe_i - nl_tor_i)
```
Signature gains `r_vec`. The old `gpu_spectral_curl!` call in this function is GONE (the curl is now algebraic in the potentials). Update callers.

- [ ] **Step 4: Run; commit:**

```bash
git add src/gpu/magnetic_nonlinear.jl test/
git commit -m "feat(gpu): Stage-4A induction curl potentials — P=-rT_E, T=-(Q_E-dr(rS_E))/r"
```

---

### Task 6: W-split GPU velocity step (shell + ball)

**Files:**
- Modify: `src/gpu/velocity_step.jl` (poloidal half), `src/gpu/banded_solve.jl` or `spectral_curl.jl` (one new per-l matvec kernel), new pack helpers in `src/gpu/device_state.jl`
- Delete: `src/gpu/influence_correction.jl` legacy correction (`gpu_velocity_poloidal_influence_correction!`, `gpu_pack_influence`, `_influence_correction_kernel!`) once consumers are gone; repoint its test file
- Test: the velocity-step test (`grep -ln "velocity_field_step\|phase5k" test/gpu_*.jl`) + a NEW W-split kernel test

- [ ] **Step 1: New per-l banded matvec.** The existing `gpu_batched_banded_matvec!` applies ONE operator to all modes; D_pol is per-l. Add next to it:

```julia
# One workitem per (l,m); applies the degree's own banded operator
# mat_b[:,:,li] (dim-3 = l-slot). Same ascending-j accumulation. Y ≠ X.
@kernel function _banded_matvec_perl_kernel!(Y, @Const(X), @Const(mat_b), bw::Int, nr::Int)
    li, mi = @index(Global, NTuple)
    T = eltype(Y)
    @inbounds for i in 1:nr
        s = zero(T)
        for j in max(1, i - bw):min(nr, i + bw)
            s += mat_b[bw + 1 + i - j, j, li] * X[li, mi, j]
        end
        Y[li, mi, i] = s
    end
end

"""
    gpu_batched_banded_matvec_perl!(Y, X, mat_b, bw) -> Y

Per-degree banded matvec: `Y[l,m,:] = mat_b[:,:,l] · X[l,m,:]` (dim-3 of
`mat_b` = l-slot). `Y` must not alias `X`. Backend inferred from `Y`.
"""
function gpu_batched_banded_matvec_perl!(Y, X, mat_b, bw::Int)
    nl, nm, nr = size(Y)
    backend = KernelAbstractions.get_backend(Y)
    _banded_matvec_perl_kernel!(backend)(Y, X, mat_b, bw, nr; ndrange = (nl, nm))
    KernelAbstractions.synchronize(backend)
    return Y
end
```

Unit test (same file as the step test or the phase-4 test): pack the host `split.dpol_op` per l-slot, compare per-mode against CPU `mul!` — ≤1e-13 (CPU `mul!` order may differ) or `==` if measured exact.

- [ ] **Step 2: Pack helper** in `device_state.jl`:

```julia
# Pack the per-degree banded operators of a PoloidalSplitMatrices (indexed via
# split.lookup) into an (2bw+1, nr, nl) l-slot batched array. Degrees absent
# from the split (none in practice — unique_l covers 0..lmax) get zero columns.
function gpu_pack_split_banded(ops::AbstractVector, lookup::Dict{Int, Int},
        nl::Int, bw::Int, nr::Int, ::Type{T}) where {T}
    out = zeros(T, 2bw + 1, nr, nl)
    for (l, idx) in lookup
        (0 <= l <= nl - 1) || continue
        out[:, :, l + 1] .= ops[idx].data
    end
    return out
end

# Per-degree vectors (h1/h2 Green responses, nr each) → (nr, nl); 2×2 influence → (2,2,nl).
function gpu_pack_split_vectors(vs::AbstractVector, lookup::Dict{Int, Int},
        nl::Int, nr::Int, ::Type{T}) where {T}
    out = zeros(T, nr, nl)
    for (l, idx) in lookup
        (0 <= l <= nl - 1) || continue
        out[:, l + 1] .= vs[idx]
    end
    return out
end

function gpu_pack_split_influence(ms::AbstractVector, lookup::Dict{Int, Int},
        nl::Int, ::Type{T}) where {T}
    out = zeros(T, 2, 2, nl)
    for (l, idx) in lookup
        (0 <= l <= nl - 1) || continue
        out[:, :, l + 1] .= ms[idx]
    end
    return out
end
```

LU packs: build l-slot-ordered vectors `[split.w_factor[split.lookup[l]] for l in 0:lmax]` and feed `gpu_pack_banded_lu` (every l exists — unique_l covers 0..lmax; assert that).

The wsplit bundle (built in Task 7's device-state, but define the NamedTuple shape now in the velocity_step docstring):

```
wsplit :: (; dpol_b, wlin_b, wlu_b, plu_b,        # (2bw+1,nr,nl) ×4 (wlu/plu are LU packs)
             h1_b, h2_b,                          # (nr,nl)
             M_b,                                 # (2,2,nl)
             d1in, d1out,                         # (nr,) residual rows
             lp1_reg,                             # (nl,) ball: (l+1)*reg_r_inv per l-slot; zeros for shell
             ball::Bool, inv_dt, linear_weight)
```

- [ ] **Step 3: The W-split residual/correction kernel** (in `velocity_step.jl`):

```julia
# One workitem per (l,m). Stage-4B W-split influence: computes
#   rho1 = ball ? rho1w_buf[li,mi]                      (precomputed on Wp pre-zeroing)
#               : Σ_r d1in[r]·Pp[li,mi,r]
#   rho2 = Σ_r d1out[r]·Pp[li,mi,r]
# solves [M11 M12; M21 M22]·a = −rho (per-l M_b[:,:,li], Cramer — matching the
# CPU _apply_poloidal_wsplit_cnab2! algebra), then
#   P[li,mi,r] = Pp[li,mi,r] + a1·h1_b[r,li] + a2·h2_b[r,li]
# l=0 (li==1) modes are zeroed (CPU parity).
@kernel function _wsplit_correction_kernel!(P, @Const(Pp), @Const(rho1w_buf),
        @Const(d1in), @Const(d1out), @Const(h1_b), @Const(h2_b), @Const(M_b),
        ball::Bool, nr::Int)
    li, mi = @index(Global, NTuple)
    T = eltype(P)
    @inbounds if li == 1
        for r in 1:nr
            P[li, mi, r] = zero(T)
        end
    else
        r1 = zero(T); r2 = zero(T)
        for r in 1:nr
            r2 += d1out[r] * Pp[li, mi, r]
        end
        if ball
            r1 = rho1w_buf[li, mi]
        else
            for r in 1:nr
                r1 += d1in[r] * Pp[li, mi, r]
            end
        end
        m11 = M_b[1, 1, li]; m12 = M_b[1, 2, li]
        m21 = M_b[2, 1, li]; m22 = M_b[2, 2, li]
        det = m11 * m22 - m12 * m21
        a1 = (-r1 * m22 + r2 * m12) / det
        a2 = (-r2 * m11 + r1 * m21) / det
        for r in 1:nr
            P[li, mi, r] = Pp[li, mi, r] + a1 * h1_b[r, li] + a2 * h2_b[r, li]
        end
    end
end

# Ball-only pre-zeroing residual: rho1w[li,mi] = Σ_r d1in[r]·Wp[li,mi,r] − lp1_reg[li]·Wp[li,mi,1]
@kernel function _wsplit_rho1w_kernel!(rho1w, @Const(Wp), @Const(d1in), @Const(lp1_reg), nr::Int)
    li, mi = @index(Global, NTuple)
    T = eltype(rho1w)
    s = zero(T)
    @inbounds for r in 1:nr
        s += d1in[r] * Wp[li, mi, r]
    end
    @inbounds rho1w[li, mi] = s - lp1_reg[li] * Wp[li, mi, 1]
end
```

- [ ] **Step 4: Rewrite the poloidal half of `gpu_velocity_field_step!`.** Signature change: `influence` argument becomes `wsplit` (the bundle above). The poloidal block (3) becomes, for EACH of (real, imag) — write a small helper `_wsplit_apply_half!(spec_x, nl_x, prev_nl_x, wsplit, bw)` called twice:

```julia
function _wsplit_apply_half!(P_x, nl_x, pnl_x, ws, bw::Int)
    nl_, nm_, nr_ = size(P_x)
    backend = KernelAbstractions.get_backend(P_x)
    W   = similar(P_x); LW = similar(P_x); rhs = similar(P_x)
    Wp  = similar(P_x); Pp = similar(P_x)
    gpu_batched_banded_matvec_perl!(W, P_x, ws.dpol_b, bw)      # W = D_pol·P
    gpu_batched_banded_matvec_perl!(LW, W, ws.wlin_b, bw)       # Ek·D_pol·W
    @. rhs = ws.inv_dt * W + ws.linear_weight * LW + 1.5 * nl_x - 0.5 * pnl_x
    gpu_batched_banded_solve!(Wp, rhs, ws.wlu_b, bw)
    rho1w = similar(P_x, nl_, nm_)
    if ws.ball
        _wsplit_rho1w_kernel!(backend)(rho1w, Wp, ws.d1in, ws.lp1_reg, nr_; ndrange = (nl_, nm_))
        KernelAbstractions.synchronize(backend)
    else
        fill!(rho1w, zero(eltype(rho1w)))
    end
    Wp[:, :, 1] .= 0
    Wp[:, :, nr_] .= 0
    gpu_batched_banded_solve!(Pp, Wp, ws.plu_b, bw)
    _wsplit_correction_kernel!(backend)(P_x, Pp, rho1w, ws.d1in, ws.d1out,
        ws.h1_b, ws.h2_b, ws.M_b, ws.ball, nr_; ndrange = (nl_, nm_))
    KernelAbstractions.synchronize(backend)
    return nothing
end
```

In `gpu_velocity_field_step!`: poloidal block = `_wsplit_apply_half!(rp_r, nlp_r, pol.prev_nl_r, wsplit, bw)`-style — CAREFUL with the ordering invariant: the W-split reads the OLD `pol.spec_*` (the `P_x` input) — so copy `pol.spec_*` into the output scratch FIRST or pass input/output separately; mirror the existing invariant comments. The CPU 1.5/−0.5 AB2 weights and `inv_dt = Ek/dt`, `linear_weight = 1−θ` match the existing toroidal block's conventions (wlin_b is packed from `split.w_linear` which already carries Ek). Delete the legacy `gpu_build_rhs_cnab2!`+`gpu_implicit_solve_field!`+`gpu_velocity_poloidal_influence_correction!` poloidal block. The toroidal block is UNCHANGED.

- [ ] **Step 5: Failing-then-passing test.** New testset in the velocity-step test file: CPU `_apply_poloidal_wsplit_cnab2!` on a real shell SolverState (one application on captured spectral+nl inputs) vs `_wsplit_apply_half!` on the scattered dense arrays — ≤1e-13 (CPU uses `dot`/`mul!` ⇒ not bit-exact). Then the SAME for a ball fixture (`geometry = :ball`; split built with `ball=true`). Run, then run the whole gpu set; convert later-task fallout to `@test_skip # un-gated in Task 7`.

- [ ] **Step 6: Delete the legacy influence machinery** (file `src/gpu/influence_correction.jl` and its `_build_influence_pack` in device_state.jl) once `grep -rn "gpu_velocity_poloidal_influence_correction!\|gpu_pack_influence" src/ test/` shows only the phase-5j test — rewrite that test file as the W-split kernel test (or delete it if Step 5's tests live elsewhere; keep SOME file exercising the correction kernel directly).

- [ ] **Step 7: Commit**

```bash
git add src/gpu/ test/
git commit -m "feat(gpu): W-split poloidal velocity step — batched D_pol, mixed influence, shell+ball"
```

---

### Task 7: Device-state wiring + step/run equivalence gates (shell + ball)

**Files:**
- Modify: `src/gpu/device_state.jl` (build the `wsplit` bundle; `rscale`→rinv2 + `r` already from Tasks 1–2; drop `_build_influence_pack`)
- Modify: `src/gpu/solver_step.jl`, `src/gpu/run.jl` (pass `wsplit`; signature ripples)
- Test: `test/gpu_phase5n2*.jl` (or wherever `build_gpu_solver_state` + the GPU≈CPU gate live — `grep -ln "build_gpu_solver_state" test/`), plus the run-loop test

- [ ] **Step 1: Implement the wsplit pack in `build_gpu_solver_state`:**

```julia
    # Stage-4B W-split operators (host-built, device-copied). The CPU lazy
    # builder is the single source of truth — ball geometry, BC codes, theta
    # all resolved there.
    split = _get_or_build_poloidal_split!(st, _velocity_bc_code(st.parameters.velocity_bcs))
    nl = cfg.lmax + 1
    bw_s = radial_bandwidth(st.runtime.outer_core_domain)
    nr_s = st.runtime.outer_core_domain.N
    @assert all(l -> haskey(split.lookup, l), 0:cfg.lmax) "W-split must cover every degree"
    wlu_slot = [split.w_factor[split.lookup[l]] for l in 0:cfg.lmax]
    plu_slot = [split.p_factor[split.lookup[l]] for l in 0:cfg.lmax]
    lp1_reg = zeros(T, nl)
    if split.ball
        for l in 0:cfg.lmax
            lp1_reg[l + 1] = T((l + 1) * split.reg_r_inv)
        end
    end
    wsplit = (;
        dpol_b = on_architecture(arch, gpu_pack_split_banded(split.dpol_op, split.lookup, nl, bw_s, nr_s, T)),
        wlin_b = on_architecture(arch, gpu_pack_split_banded(split.w_linear, split.lookup, nl, bw_s, nr_s, T)),
        wlu_b = gpu_pack_banded_lu(wlu_slot, arch),
        plu_b = gpu_pack_banded_lu(plu_slot, arch),
        h1_b = on_architecture(arch, gpu_pack_split_vectors(split.h1, split.lookup, nl, nr_s, T)),
        h2_b = on_architecture(arch, gpu_pack_split_vectors(split.h2, split.lookup, nl, nr_s, T)),
        M_b = on_architecture(arch, gpu_pack_split_influence(split.influence, split.lookup, nl, T)),
        d1in = on_architecture(arch, T.(split.d1_row_inner)),
        d1out = on_architecture(arch, T.(split.d1_row_outer)),
        lp1_reg = on_architecture(arch, lp1_reg),
        ball = split.ball,
        inv_dt = T(split.mass_coeff / st.parameters.timestep),
        linear_weight = T(1 - split.theta),
    )
```
Thread `wsplit` through the state NamedTuple → `gpu_solver_step!` → `gpu_velocity_field_step!`. Remove the legacy influence from the bundle.

- [ ] **Step 2: Un-gate the step-equivalence tests.** In the 5n2 test file: flip the `@test_throws`/`@test_skip` gates to the real assertions (the file's pre-PR-59 shape): build a CPU SolverState (full MHD shell), `initialize_solver_fields!`, run `solver_step!` N times on a CPU copy and `gpu_solver_step!` N times on the device state built from the SAME initial state, compare dense spectral fields per step: `maximum(abs, diff) ≤ 1e-12 * max(1, maximum(abs, cpu))` per field (document: W-split dots ⇒ sub-ulp, compounded over N steps). ADD the ball twin: same gate with `geometry = :ball, radius_ratio = 0.0` (fixture mirroring `test/ball_solver_physics.jl` `_ball_test_params`; hydro-only is acceptable for ball if the full-MHD ball dt-stiffness makes N steps impractical — use dt=1e-7, N=5, include_magnetic=true; if runtime is excessive, hydro ball + a 2-step MHD ball).

- [ ] **Step 3: Run-loop gate** (`gpu_run!` test file): un-gate similarly, N-step trajectory shell full-MHD.

- [ ] **Step 4: Run the ENTIRE gpu test set + the CPU suite quick gates:**

```bash
for f in $(ls test/gpu_*.jl | xargs -n1 basename); do GEODYNAMO_TEST_MPI_FINALIZE=false ~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e "using GeoDynamo; include(\"test/$f\")" > /tmp/g7_$f.log 2>&1; echo "$f exit=$?"; grep -E "Fail|Error" /tmp/g7_$f.log | head -2; done
GEODYNAMO_TEST_MPI_FINALIZE=false ~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using GeoDynamo; include("test/poloidal_momentum_split.jl"); include("test/ball_solver_physics.jl")' > /tmp/g7cpu.log 2>&1; echo exit=$?; tail -3 /tmp/g7cpu.log
```
NO remaining `@test_skip # un-gated in Task N` may survive this task — grep for them and resolve each.

- [ ] **Step 5: Commit**

```bash
git add src/gpu/ test/
git commit -m "feat(gpu): device-state W-split pack + GPU≈CPU step gates un-gated (shell+ball)"
```

---

### Task 8: Full suite + spec status + finish

- [ ] **Step 1: Full suite** (background-safe):

```bash
cd /Users/subha/Documents/GitHub/GeoDynamo-ball
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()' > /tmp/gpu_suite.log 2>&1; echo "exit=$?"; grep -E "Test Summary|Pass|Fail|Broken" /tmp/gpu_suite.log | tail -5
```
Known flakes: scalar-IC normalization tests can flake once — re-run before investigating. Baseline green = the ball-port suite numbers (5575/0/28) plus the newly un-gated GPU assertions.

- [ ] **Step 2: Spec status section** appended to `docs/superpowers/specs/2026-06-12-gpu-wsplit-ball-port-design.md`: what was un-gated, measured equivalence numbers (max diffs per field, shell + ball), surviving gates (ERK2/EAB2-GPU, conducting-IC), `[GPU-BOX]` items awaiting CUDA hardware. Commit.

- [ ] **Step 3: Finish** — superpowers:finishing-a-development-branch. Note for the menu: base branch is `feat/ball-geometry-mhd` (PR #78) — a PR targets THAT (stacked), or wait for #78 to merge and retarget.

---

## Self-review notes (already applied)

- Spec coverage: §4→Task 1, §5→Tasks 3–5, §5 curls→Task 2, §6→Task 6, §7→Task 7 pack, §8→Tasks 1–7 tests + Task 7 gates, §9 file map→all, ERK2/EAB2 gates→Task 7 Step 2 (assert via @test_throws).
- The `@test_skip # un-gated in Task N` discipline keeps every intermediate commit green without silently-wrong assertions; Task 7 Step 4 sweeps them to zero.
- Signature ripples are sequenced: Task 1 changes transform signatures and patches ALL callers in the same commit; Tasks 2/4/5 add `r_vec` where needed; Task 6/7 replace the influence bundle.
- `gpu_scalar_physical_to_spectral!` name must be verified against `src/gpu/scalar_transform.jl` in Tasks 4–5 (the helper exists for the scalar steps; adapt the call to its real signature).
