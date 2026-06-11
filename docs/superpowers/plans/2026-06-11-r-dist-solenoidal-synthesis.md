# r-Distributed Solenoidal Vector Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Full MHD `solver_step!` runs on r-distributed process grids (1x4, 2x2) by moving the solenoidal synthesis' radial derivative from the Alm layout (r-slab) to the spectral storage layout (r always fully local).

**Architecture:** `vector_spectral_to_physical_disttranspose!` (src/solver/numerics.jl:887) currently bridges P,T to Alm layout, then computes `S=(∂_r P)/r` and the v_r coefficients there — impossible on r-slabs (the `error()` gate at :914-918). New flow: compute S and Vr coefficient fields in STORAGE layout (`pencils.spec`, full r per rank) with banded D1 per (l,m) column, then push S, T, Vr through the existing `spec_storage_to_solve!`→`from_spec_solve!` bridge unchanged. +1 bridge collective per synthesis; bit-exact at 1x1 (bridge is an exact copy, D1 matvec per mode is order-identical). Spec: `docs/superpowers/specs/2026-06-11-r-dist-solenoidal-synthesis-design.md`.

**Tech Stack:** Julia 1.11, SHTnsKit ≥1.2.12, PencilArrays, MPI.jl. Julia binary: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=.` (the `julia` shim is broken). Repo root. Branch `feat/r-dist-solenoidal-synthesis`.

**⚠️ Environment trap:** if multi-rank runs fail with `m-distribution mismatch`, the LOCAL Manifest has SHTnsKit <1.2.12 — run `Pkg.status("SHTnsKit")`, fix with `Pkg.update("SHTnsKit")`. Not a code bug.

**⚠️ Test-suite traps:** never pipe `Pkg.test()` through `tail` (masks exit code) — redirect to a file. ~3 IC-normalization test failures in a full run can be a known flake — re-run before attributing. `static_checks.jl` pins source text — if it fails after edits, read its assertion.

---

## Layout contracts (read before coding)

- **Storage** (`config.pencils.spec`, decomp `(2,1)`): `parent(field.data_real/imag)` = `(l_local, m_local, nr)` — l over r_comm, m over θ_comm, **r LOCAL (full nr on every rank)**. Mode access: `slot = local_spectral_storage_slot(config, lm)` (returns `CartesianIndex(l_slot,m_slot)` or `nothing`), then `local_spectral_value(arr, slot, r_idx)` / `set_local_spectral_value!(arr, slot, r_idx, v)` (src/fields/transforms.jl:613-652). `config.l_values[lm]` gives l.
- **Alm** (SHTnsKit plan layout): `(lmax+1, m_local, nr_local)` — l full, m-bin local subset, **r-slab**. This is where the derivative is impossible under r-distribution.
- **Bridge**: `spec_storage_to_solve!(config, solve, sr, si, plan)` then `from_spec_solve!(config, Alm, solve, plan)` (src/parallel/disttranspose_adapter.jl). Takes separate real/imag Float64 storage arrays; zero-fills trailing m-bins. Exact copy — no arithmetic.
- Radii: `domain.r[r_idx, 4]`; `domain.N` = nr. `D1 = create_derivative_matrix(Float64, 1, domain)` (banded; apply with `mul!(out, D1, in)` on full-r vectors).
- Reference idiom for storage-layout per-mode radial work: `_poloidal_force_projection!` (src/physics/velocity/solver.jl:68-100).

### Task 1: Storage-layout helpers + scratch

**Files:**
- Modify: `src/physics/nonlinear.jl:359-373` (`_build_vector_scratch`)
- Modify: `src/solver/numerics.jl` (add two helpers near `_fill_vr_alm!`, :847)
- Create: `test/r_dist_solenoidal_synthesis.jl`
- Modify: `test/runtests.jl` (register)

- [ ] **Step 1: Write the failing test**

Create `test/r_dist_solenoidal_synthesis.jl`:

```julia
# Storage-layout solenoidal coefficient helpers must reproduce the per-mode
# reference computation bit-exactly. The storage pencil keeps r fully local,
# so these helpers work on ANY process grid — this is what un-blocks
# r-distributed (1x4 / 2x2) solenoidal synthesis.
using Test
using MPI
using LinearAlgebra
using GeoDynamo

MPI.Initialized() || MPI.Init()

const RD_NR   = 16
const RD_LMAX = 8

function _rd_setup()
    cfg = GeoDynamo.create_shtnskit_config(
        lmax = RD_LMAX, mmax = RD_LMAX,
        nlat = 2 * RD_LMAX + 4, nlon = 4 * RD_LMAX + 8,
        nr   = RD_NR)
    dom = GeoDynamo.create_radial_domain(RD_NR)
    return cfg, dom
end

_rd_spec(cfg, dom) =
    GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)

# Deterministic, rank-independent P: value depends only on (lm, r_idx).
function _rd_fill_poloidal!(P, cfg)
    pr = parent(P.data_real); pi_ = parent(P.data_imag)
    fill!(pr, 0.0); fill!(pi_, 0.0)
    for lm in 1:cfg.nlm
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue
        m = cfg.m_values[lm]
        for r_idx in 1:RD_NR
            GeoDynamo.set_local_spectral_value!(pr, slot, r_idx,
                sinpi(0.3 * (lm + 7 * r_idx)))
            if m > 0
                GeoDynamo.set_local_spectral_value!(pi_, slot, r_idx,
                    cospi(0.3 * (lm - 5 * r_idx)))
            end
        end
    end
    return P
end

@testset "storage-layout solenoidal coefficients" begin
    cfg, dom = _rd_setup()
    P = _rd_fill_poloidal!(_rd_spec(cfg, dom), cfg)
    pr = parent(P.data_real); pi_ = parent(P.data_imag)

    S  = _rd_spec(cfg, dom)
    Vr = _rd_spec(cfg, dom)
    sr = parent(S.data_real);  si = parent(S.data_imag)
    vr = parent(Vr.data_real); vi = parent(Vr.data_imag)

    GeoDynamo._storage_spheroidal_from_poloidal!(sr, si, pr, pi_, cfg, dom)
    GeoDynamo._storage_vr_coeffs!(vr, vi, pr, pi_, cfg, dom,
        GeoDynamo._solenoidal_vr_factor)

    # Reference: plain per-mode loops, same D1, same op order.
    D1   = GeoDynamo.create_derivative_matrix(Float64, 1, dom)
    prof = Vector{Float64}(undef, RD_NR)
    dpr  = Vector{Float64}(undef, RD_NR)
    rN   = dom.r[RD_NR, 4]
    @testset "S = (∂_r P)/r and Vr = l(l+1)·P/r², bit-exact" begin
        for lm in 1:cfg.nlm
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            l = cfg.l_values[lm]
            for (src, s_out, v_out) in ((pr, sr, vr), (pi_, si, vi))
                for r_idx in 1:RD_NR
                    prof[r_idx] = GeoDynamo.local_spectral_value(src, slot, r_idx)
                end
                mul!(dpr, D1, prof)
                for r_idx in 1:RD_NR
                    r = dom.r[r_idx, 4]
                    @test GeoDynamo.local_spectral_value(s_out, slot, r_idx) ==
                          dpr[r_idx] / r
                    vref = r > eps(Float64) * rN ?
                        prof[r_idx] * GeoDynamo._solenoidal_vr_factor(l, r) : 0.0
                    @test GeoDynamo.local_spectral_value(v_out, slot, r_idx) == vref
                end
            end
        end
    end

    @testset "vector scratch has storage-layout slabs" begin
        plan = GeoDynamo.get_disttranspose_plan(cfg)
        sc   = GeoDynamo._vector_scratch(cfg, plan)
        for f in (sc.Ssto_re, sc.Ssto_im, sc.Vrsto_re, sc.Vrsto_im)
            @test size(f) == size(pr)
            @test eltype(f) == Float64
        end
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. \
  -e 'using Test, GeoDynamo; include("test/r_dist_solenoidal_synthesis.jl")'
```

Expected: ERROR `UndefVarError: _storage_spheroidal_from_poloidal! not defined`.

- [ ] **Step 3: Implement the helpers**

In `src/solver/numerics.jl`, after `_fill_vr_alm!` (ends :870), add:

```julia
# Storage-layout solenoidal coupling: S = (∂_r P)/r per (l,m) mode on the
# spectral STORAGE arrays (pencils.spec keeps r fully local on every rank,
# unlike the Alm layout's r-slab) — this is what makes the solenoidal
# synthesis work on r-distributed grids. Same banded D1, same per-mode op
# order as the old Alm-layout `_spheroidal_from_poloidal!`, so 1x1 results
# are bit-exact.
function _storage_spheroidal_from_poloidal!(s_re, s_im, p_re, p_im, config, domain)
    nr = domain.N
    r_range = local_range(config.pencils.spec, 3)
    length(r_range) == nr || error(
        "spectral storage must keep the radial axis fully local " *
        "(got $(length(r_range)) of $nr levels)")
    D1   = create_derivative_matrix(Float64, 1, domain)
    prof = Vector{Float64}(undef, nr)
    dpr  = Vector{Float64}(undef, nr)
    for (src, dst) in ((p_re, s_re), (p_im, s_im))
        fill!(dst, 0.0)
        @inbounds for lm in 1:config.nlm
            slot = local_spectral_storage_slot(config, lm)
            slot === nothing && continue
            for r_idx in 1:nr
                prof[r_idx] = local_spectral_value(src, slot, r_idx)
            end
            mul!(dpr, D1, prof)
            for r_idx in 1:nr
                r = domain.r[r_idx, 4]
                set_local_spectral_value!(dst, slot, r_idx, dpr[r_idx] / r)
            end
        end
    end
    return nothing
end

# Storage-layout v_r coefficients: vr = vr_factor(l, r)·P per (l,m) mode.
# Same eps-guard near r=0 as the old Alm-layout `_fill_vr_alm!`.
function _storage_vr_coeffs!(vr_re, vr_im, p_re, p_im, config, domain,
        vr_factor::F) where {F}
    nr = domain.N
    r_range = local_range(config.pencils.spec, 3)
    length(r_range) == nr || error(
        "spectral storage must keep the radial axis fully local " *
        "(got $(length(r_range)) of $nr levels)")
    rN = domain.r[nr, 4]
    for (src, dst) in ((p_re, vr_re), (p_im, vr_im))
        fill!(dst, 0.0)
        @inbounds for lm in 1:config.nlm
            slot = local_spectral_storage_slot(config, lm)
            slot === nothing && continue
            l = config.l_values[lm]
            for r_idx in 1:nr
                r_val = domain.r[r_idx, 4]
                r_val > eps(Float64) * rN || continue
                set_local_spectral_value!(dst, slot, r_idx,
                    local_spectral_value(src, slot, r_idx) * vr_factor(l, r_val))
            end
        end
    end
    return nothing
end
```

In `src/physics/nonlinear.jl`, replace `_build_vector_scratch` (:359-373) body's
return and add the slabs:

```julia
function _build_vector_scratch(config, plan)
    Slm    = SHTnsKit.allocate_spectral(plan)   # spheroidal/poloidal
    Tlm    = SHTnsKit.allocate_spectral(plan)   # toroidal
    Vr_alm = SHTnsKit.allocate_spectral(plan)   # radial-scaled poloidal (l(l+1)/r²·P)
    Vt     = SHTnsKit.allocate_spatial(plan)    # θ-tangential (nlon, nlat_local, nlev)
    Vp     = SHTnsKit.allocate_spatial(plan)    # φ-tangential
    Vr     = SHTnsKit.allocate_spatial(plan)    # radial (scalar synthesis of vr coeffs)
    scratch = _get_disttranspose_scratch(config, plan)
    # Share scratch.solve (the adapter's persistent buffer) so that
    # from_spec_solve! can use the cached Transposition plan t_bwd.
    # The vector path calls from_spec_solve! up to three times (S, T, Vr) but
    # always sequentially, so sharing scratch.solve is safe.
    solve = scratch.solve
    # Storage-layout slabs (l_local, m_local, nr) for the solenoidal S and v_r
    # coefficient fields, computed where r is fully local (r-dist support).
    spec_dims = length.(PencilArrays.range_local(config.pencils.spec))
    Ssto_re  = zeros(Float64, spec_dims)
    Ssto_im  = zeros(Float64, spec_dims)
    Vrsto_re = zeros(Float64, spec_dims)
    Vrsto_im = zeros(Float64, spec_dims)
    return (; Slm, Tlm, Vr_alm, Vt, Vp, Vr, solve,
              Ssto_re, Ssto_im, Vrsto_re, Vrsto_im)
end
```

- [ ] **Step 4: Run test to verify it passes**

Same command as Step 2. Expected: all PASS.

- [ ] **Step 5: Register the test**

In `test/runtests.jl`, add next to the other solenoidal/transform includes
(find with `grep -n solenoidal_transform_pair test/runtests.jl`):

```julia
include("r_dist_solenoidal_synthesis.jl")
```

- [ ] **Step 6: Commit**

```bash
git add src/solver/numerics.jl src/physics/nonlinear.jl test/r_dist_solenoidal_synthesis.jl test/runtests.jl
git commit -m "feat(parallel): storage-layout solenoidal coefficient helpers

S=(∂_r P)/r and vr=vr_factor·P computed on the spec storage pencil,
where r is fully local on every rank — groundwork for r-distributed
solenoidal synthesis.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 2: Restructure the synthesis core

**Files:**
- Modify: `src/solver/numerics.jl:887-958` (`vector_spectral_to_physical_disttranspose!`)

- [ ] **Step 1: Replace the function body**

Replace `vector_spectral_to_physical_disttranspose!` (:887-958) with:

```julia
function vector_spectral_to_physical_disttranspose!(
        config, plan,
        toroidal, poloidal, vector_field, domain, vr_factor;
        raw_spheroidal::Bool = false)
    sc  = _vector_scratch(config, plan)
    Slm = sc.Slm   # spheroidal/poloidal
    Tlm = sc.Tlm   # toroidal
    Vt  = sc.Vt
    Vp  = sc.Vp
    Vr  = sc.Vr
    Vr_alm = sc.Vr_alm
    solve  = sc.solve

    solenoidal = !raw_spheroidal && domain !== nothing

    # 1. Spheroidal input to the sphtor synthesis. Solenoidal convention:
    #    S = (∂_r P)/r, computed in STORAGE layout where r is fully local
    #    (works on r-distributed grids; the Alm layout only has an r-slab).
    #    Raw mode feeds the stored coefficients verbatim (tangential-basis
    #    primitive — see the doc comment above).
    if solenoidal
        _storage_spheroidal_from_poloidal!(sc.Ssto_re, sc.Ssto_im,
            parent(poloidal.data_real), parent(poloidal.data_imag),
            config, domain)
        spec_storage_to_solve!(config, solve, sc.Ssto_re, sc.Ssto_im, plan)
    else
        spec_storage_to_solve!(config, solve, parent(poloidal.data_real),
                               parent(poloidal.data_imag), plan)
    end
    from_spec_solve!(config, Slm, solve, plan)

    # 2. Toroidal, unchanged.
    spec_storage_to_solve!(config, solve, parent(toroidal.data_real),
                           parent(toroidal.data_imag), plan)
    from_spec_solve!(config, Tlm, solve, plan)

    # 3. Distributed sphtor synthesis: (Slm, Tlm) → (Vt, Vp), batched over nr_local.
    SHTnsKit.dist_synthesis_sphtor!(plan, Vt, Vp, Slm, Tlm)

    _VECTOR_DISTTRANSPOSE_COUNT[] += 1

    # 4. Copy Vt/Vp (nlon, nlat_local, lev) → v_θ/v_φ (nlat_local, nlon, nr_local)
    #    with the first-two-axis transpose; lev ↔ r_local align per rank.
    #    Function barrier: sc is ::Any from IdDict cache; parent(Vt/Vp) would be
    #    ::Any and box every element.  Use the concrete-typed helper.
    v_theta = parent(vector_field.θ_component.data)
    v_phi   = parent(vector_field.φ_component.data)
    v_r     = parent(vector_field.r_component.data)
    _copy_spatial2_to_physical2!(v_theta, v_phi, parent(Vt), parent(Vp))

    # 5. v_r = vr_factor(l,r)·P (both the solenoidal and the legacy
    #    raw-with-domain paths), computed in storage layout and bridged.
    #    Without a domain there are no radii: zero-fill.
    if domain !== nothing
        _storage_vr_coeffs!(sc.Vrsto_re, sc.Vrsto_im,
            parent(poloidal.data_real), parent(poloidal.data_imag),
            config, domain, vr_factor)
        spec_storage_to_solve!(config, solve, sc.Vrsto_re, sc.Vrsto_im, plan)
        from_spec_solve!(config, Vr_alm, solve, plan)
        SHTnsKit.dist_synthesis!(plan, Vr, Vr_alm)
        # Function barrier: Vr is ::Any from sc (IdDict); parent(Vr) would be ::Any.
        _copy_spatial_to_physical!(v_r, parent(Vr))
    else
        fill!(v_r, zero(eltype(v_r)))
    end

    return vector_field
end
```

Notes: the r-local `error()` gate, `r_range_ph`, `_fill_vr_alm!` calls and the
Alm-layout `_spheroidal_from_poloidal!` call are all gone. Keep the doc
comment block above the function (:872-886) — it still describes the
convention; update its last paragraph mentioning Alm-layout derivative if
present.

- [ ] **Step 2: Run the targeted regression tests**

```bash
J=~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia
$J --startup-file=no --project=. -e 'using Test, GeoDynamo;
  include("test/r_dist_solenoidal_synthesis.jl");
  include("test/solenoidal_transform_pair.jl");
  include("test/poloidal_solenoidality.jl")' 2>&1 | tee /tmp/rd_task2.log
```

Expected: all PASS (the ∇·u hard gate and the transform-pair fixtures verify
the restructure changed nothing at 1x1).

- [ ] **Step 3: Run the allocation guards**

```bash
$J --startup-file=no --project=. -e 'using Test, GeoDynamo;
  include("test/allocation_runtime_checks.jl")' 2>&1 | tee /tmp/rd_alloc.log
```

Expected: PASS. If a vector-synthesis allocation budget trips: the only new
steady-state allocation allowed is `create_derivative_matrix` inside
`_storage_spheroidal_from_poloidal!` (the old Alm path allocated the same D1
per call) plus the two `Vector{Float64}(undef, nr)` profile buffers; if the
guard counts these, hoist `D1`/`prof`/`dpr` into the storage scratch
(NamedTuple fields `D1sto`, `prof`, `dpr` built in `_build_vector_scratch`)
rather than loosening the budget.

- [ ] **Step 4: Commit**

```bash
git add src/solver/numerics.jl
git commit -m "feat(parallel)!: r-distributed solenoidal vector synthesis

Compute S=(∂_r P)/r and v_r coefficients in spec storage layout (r fully
local) and bridge S/T/Vr to the Alm layout, instead of differentiating
in the Alm layout's r-slab. Removes the r-local gate: full MHD steps now
run on 1x4/2x2 grids. +1 bridge collective per vector synthesis; 1x1
results bit-exact (bridge is an exact copy, per-mode D1 order unchanged).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 3: Delete the dead Alm-layout helpers

**Files:**
- Modify: `src/solver/numerics.jl` (`_fill_vr_alm!` :847-870, `_spheroidal_from_poloidal!` :960-990)

- [ ] **Step 1: Verify no remaining callers**

```bash
grep -rn "_fill_vr_alm\|_spheroidal_from_poloidal" src/ test/
```

Expected: only the two definitions in `src/solver/numerics.jl` (verified
2026-06-11; re-check in case of concurrent sessions). If a caller appeared,
STOP and reconcile before deleting.

- [ ] **Step 2: Delete both functions** (including their doc comments).

- [ ] **Step 3: Quick compile + targeted test**

```bash
$J --startup-file=no --project=. -e 'using GeoDynamo;
  using Test; include("test/r_dist_solenoidal_synthesis.jl")'
```

Expected: PASS, no `UndefVarError`.

- [ ] **Step 4: Commit**

```bash
git add src/solver/numerics.jl
git commit -m "refactor: drop dead Alm-layout solenoidal helpers

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 4: MPI acceptance gates

**Files:** none modified (validation only)

- [ ] **Step 1: Hydro equivalence, all four grids**

```bash
JULIA=$HOME/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia \
  test/run_mpi_r_theta_equivalence.sh > /tmp/rd_eq_hydro.log 2>&1; echo "EXIT=$?"
grep -E "max diff|equivalent|FAIL" /tmp/rd_eq_hydro.log
```

Expected: `EXIT=0`; 1x1, 4x1, 1x4, 2x2 all present and equivalent < 1e-10.
(Before this work, 1x4/2x2 died at the solenoidal gate.)

- [ ] **Step 2: MHD equivalence (magnetic + composition), all four grids**

```bash
JULIA=$HOME/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia \
  test/run_mpi_r_theta_equivalence_mhd.sh > /tmp/rd_eq_mhd.log 2>&1; echo "EXIT=$?"
grep -E "max diff|equivalent|FAIL" /tmp/rd_eq_mhd.log
```

Expected: `EXIT=0`, all 12 tensors < 1e-10 on every grid. This is the
acceptance gate from the spec.

- [ ] **Step 3: r-dist scaling smoke (optional but record the numbers)**

```bash
GEODYNAMO_PROC_GRID=1x4 OPENBLAS_NUM_THREADS=1 NP=4 $J --project=. -e '
  using MPI
  jl = Base.julia_cmd()[1]
  MPI.mpiexec() do mpi
      run(`$mpi -n $(parse(Int, ENV["NP"])) $jl -t1 --project=. /tmp/scaling_p3.jl`)
  end'
```

Expected: `RESULT grid=1x4 np=4 step_ms=<number>` (no gate error). Repeat with
`GEODYNAMO_PROC_GRID=2x2`. If `/tmp/scaling_p3.jl` is gone, skip — the
equivalence suites above are the correctness gate.

- [ ] **Step 4: Full suite**

```bash
$J --startup-file=no --project=. -e 'using Pkg; Pkg.test()' > /tmp/rd_full_suite.log 2>&1
echo "EXIT=$?"; tail -5 /tmp/rd_full_suite.log
```

Expected: EXIT=0. If ~3 IC-normalization failures appear, re-run once before
investigating (known flake).

- [ ] **Step 5: Commit any stragglers, then finish**

```bash
git status --short   # should be clean apart from untracked DD_2DCODE/
```

Use superpowers:finishing-a-development-branch (merge vs PR decision with the
user).
