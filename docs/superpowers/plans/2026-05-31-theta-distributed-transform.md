# θ-Distributed Transform (Phase 1 of r×θ) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace GeoDynamo's gather-replicate SH transform with SHTnsKit 1.2.10's θ-distributed transform, making the (currently 97%-serial) transform phase scalable on the θ-axis while keeping the radial dimension local.

**Architecture:** Physical fields become θ-distributed / φ-local (1D-θ process grid); each radial level is transformed via `dist_synthesis`/`dist_analysis` (scalar) and `dist_synthesis_sphtor!`/`dist_analysis_sphtor!` (vector). A small adapter maps GeoDynamo's `(l,m,r)` spectral storage to SHTnsKit's per-level distributed spectral input. The implicit radial solve is untouched (r stays local). Phase 2 (distribute r + r↔lm transpose) is a later plan.

**Tech Stack:** Julia, SHTnsKit ≥1.2.10 (`ext/ParallelTransforms.jl`), PencilArrays, PencilFFTs, MPI.

**Execution note:** Run in an isolated git worktree (`superpowers:using-git-worktrees`) off the current branch — `main` is being edited by another session. Validate correctness here; in-solver *scaling* is validated on cluster hardware later (laptop single-node MPI cannot show it).

---

## File Structure

- `src/parallel/pencils.jl` (modify) — add a θ-distributed/φ-local physical pencil + 1D-θ topology constructor. Keep the spectral pencil contract (l,m distributed, r local).
- `src/parallel/spectral_pencil_adapter.jl` (create) — the `(l,m,r)` ⇄ SHTnsKit per-level distributed-spectral adapter. One responsibility; isolated + unit-tested. Included from the root module after `transforms/spectral.jl`.
- `src/physics/nonlinear.jl` (modify) — `scalar_spectral_to_physical!`, `scalar_physical_to_spectral!`.
- `src/fields/transforms.jl` (modify) — `shtnskit_vector_synthesis!`, `shtnskit_vector_analysis!`.
- `test/theta_dist_adapter.jl` (create) — adapter round-trip unit tests.
- `test/theta_dist_transform.jl` (create) — transform roundtrip (serial + MPI) tests.

---

## Task 0: Spike — pin the SHTnsKit dist API shapes (no production code)

**Files:** `test/spike_dist_api.jl` (create, throwaway — delete after).

- [ ] **Step 1: Write a spike that exercises the exact calls the adapter/transforms will use**

```julia
using SHTnsKit, MPI, PencilArrays, PencilFFTs
MPI.Init()
comm = MPI.COMM_WORLD
lmax = 8; nlat = lmax + 2; nlon = 2*lmax + 1
cfg = SHTnsKit.create_gauss_config(lmax, nlat; mmax = lmax, nlon = nlon, norm = :orthonormal)
pen = Pencil((nlat, nlon), comm)                 # θ-distributed, φ local (PencilFFTs layout)
flocal = randn(Float64, PencilArrays.size_local(pen)...)
f = PencilArray(pen, flocal)
alm = SHTnsKit.dist_analysis(cfg, f)             # -> distributed spectral
@show typeof(alm)                                 # RECORD: PencilArray? DistributedSpectralArray? dims?
@show size(alm) PencilArrays.range_local(alm)     # RECORD: (l,m) layout, local owned modes
frec = SHTnsKit.dist_synthesis(cfg, alm; prototype_θφ = f, real_output = true)
@show typeof(frec) size(frec)
# vector:
Slm = alm; Tlm = alm
@show methods(SHTnsKit.dist_synthesis_sphtor!)    # RECORD exact in-place signature
MPI.Finalize()
```

- [ ] **Step 2: Run it (1 rank) and RECORD the printed types/shapes**

Run: `julia --project=. -e 'using MPI; MPI.mpiexec() do m; run(`$m -n 1 $(Base.julia_cmd()[1]) --project=. test/spike_dist_api.jl`); end'`
Expected: prints `typeof(alm)`, its local range, and the sphtor signature. **Write these into Task 2's adapter code** (replace the `# SPIKE:` annotations there with the real types).

- [ ] **Step 3: Delete the spike**

```bash
rm test/spike_dist_api.jl
```

> Rationale: SHTnsKit's distributed spectral object type/layout (`alm`) is the one shape this plan can't assert from outside. The spike pins it so Tasks 2–5 use real signatures, not guesses. Reference: `~/.julia/packages/SHTnsKit/*/ext/ParallelTransforms.jl` (`dist_analysis` ~263, `dist_synthesis` ~652/769, `dist_synthesis_sphtor!`).

---

## Task 1: θ-distributed / φ-local physical pencil

**Files:**
- Modify: `src/parallel/pencils.jl` (`create_computation_pencils`)
- Test: `test/theta_dist_transform.jl`

- [ ] **Step 1: Write the failing test**

```julia
using Test, GeoDynamo, MPI
MPI.Initialized() || MPI.Init()
@testset "theta-dist physical pencil" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=16, nr=4)
    p = cfg.pencils.theta_phys                       # NEW pencil
    θr = GeoDynamo.range_local(p, 1); φr = GeoDynamo.range_local(p, 2)
    @test length(φr) == 16                           # φ FULLY LOCAL
    # on 1 rank θ is full; the multi-rank split is checked in the MPI test
    @test length(θr) == 12
end
```

- [ ] **Step 2: Run it, expect FAIL** (`theta_phys` not a field)

Run: `julia --project=. -e 'using Test,GeoDynamo,MPI; include("test/theta_dist_transform.jl")'`
Expected: FAIL — `type NamedTuple has no field theta_phys`.

- [ ] **Step 3: Add the pencil in `create_computation_pencils`**

In `src/parallel/pencils.jl`, inside `create_computation_pencils`, add a θ-distributed/φ-local physical pencil (PencilArrays decomposes only dim 1 = θ; φ and r local):

```julia
    # θ-distributed, φ + r local — the layout SHTnsKit's θ-dist transform consumes.
    pencil_theta_phys = Pencil(topology, dims, (1,))   # only θ (dim 1) distributed
```

and add it to the returned NamedTuple:

```julia
    return (θ = pencil_θ,
        φ = pencil_φ,
        r = pencil_r,
        spec = pencil_spec,
        mixed = pencil_spec,
        theta_phys = pencil_theta_phys)
```

(The 1D-θ topology: `create_pencil_topology` must allow `proc_dims = (nprocs, 1)` for this path — it already falls back to that; no change needed for Phase 1.)

- [ ] **Step 4: Run test, expect PASS**

Run: same as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/parallel/pencils.jl test/theta_dist_transform.jl
git commit -m "feat(parallel): add theta-distributed/phi-local physical pencil"
```

---

## Task 2: Spectral adapter `(l,m,r)` ⇄ distributed spectral

**Files:**
- Create: `src/parallel/spectral_pencil_adapter.jl`
- Modify: `src/GeoDynamo.jl` (include after `transforms/spectral.jl`)
- Test: `test/theta_dist_adapter.jl`

- [ ] **Step 1: Write the failing round-trip test**

```julia
using Test, GeoDynamo, SHTnsKit, MPI
MPI.Initialized() || MPI.Init()
@testset "spectral adapter round-trip" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=16, nr=4)
    # a known (l,m) coefficient matrix for one radial level
    alm_mat = zeros(ComplexF64, 9, 9); alm_mat[3,1] = 1.0; alm_mat[5,2] = 0.5 + 0.2im
    sht_alm = GeoDynamo.to_sht_spectral(cfg, alm_mat)        # (l,m) matrix -> SHTnsKit dist spectral
    back    = GeoDynamo.from_sht_spectral(cfg, sht_alm)      # and back
    @test back ≈ alm_mat
end
```

- [ ] **Step 2: Run it, expect FAIL** (functions undefined)

Run: `julia --project=. -e 'using Test,GeoDynamo,SHTnsKit,MPI; include("test/theta_dist_adapter.jl")'`
Expected: FAIL — `to_sht_spectral not defined`.

- [ ] **Step 3: Implement the adapter using the types pinned in Task 0**

Create `src/parallel/spectral_pencil_adapter.jl`. Fill the `# SPIKE:` types from Task 0's output:

```julia
# Adapter between GeoDynamo's dense (l,m) coefficient matrix (one radial level)
# and SHTnsKit's distributed spectral object. SPIKE (Task 0) pins `SHTAlm`.
# const SHTAlm = <type printed by spike>   # e.g. PencilArray over (l,m)

"""
    to_sht_spectral(cfg, alm_mat::Matrix{ComplexF64})

Pack a full (lmax+1, mmax+1) coefficient matrix into SHTnsKit's distributed
spectral object for `dist_synthesis`. On the rank that owns each (l,m), copy the
coefficient; elsewhere leave zero. Uses `SHTnsKit.create_spectral_pencil(cfg)` +
`PencilArrays.range_local` to find owned modes (confirm the exact constructor in
ext/ParallelDispatch.jl:12).
"""
function to_sht_spectral(cfg, alm_mat::AbstractMatrix{ComplexF64})
    pen = SHTnsKit.create_spectral_pencil(cfg; comm = mpi_comm())
    out = PencilArray(pen, zeros(ComplexF64, PencilArrays.size_local(pen)...))
    olr = PencilArrays.range_local(out)            # owned (l,m) index ranges
    od  = parent(out)
    @inbounds for (jl, jg) in enumerate(olr[2]), (il, ig) in enumerate(olr[1])
        od[il, jl] = alm_mat[ig, jg]               # global (l,m) -> local block
    end
    return out
end

"""
    from_sht_spectral(cfg, alm) -> Matrix{ComplexF64}

Inverse: gather the distributed spectral object back to a full (l,m) matrix
(Allreduce over the owned-disjoint blocks). Used to write `dist_analysis` output
back into GeoDynamo's (l,m,r) storage.
"""
function from_sht_spectral(cfg, alm)
    full = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    olr = PencilArrays.range_local(alm); ad = parent(alm)
    @inbounds for (jl, jg) in enumerate(olr[2]), (il, ig) in enumerate(olr[1])
        full[ig, jg] = ad[il, jl]
    end
    allreduce_sum_in_place!(full, mpi_comm())      # disjoint blocks -> complete matrix
    return full
end
```

Include it from `src/GeoDynamo.jl` right after `include("transforms/spectral.jl")`:

```julia
    include("parallel/spectral_pencil_adapter.jl")
```

- [ ] **Step 4: Run test, expect PASS**

Run: same as Step 2. Expected: PASS (`back ≈ alm_mat`).

- [ ] **Step 5: Commit**

```bash
git add src/parallel/spectral_pencil_adapter.jl src/GeoDynamo.jl test/theta_dist_adapter.jl
git commit -m "feat(parallel): (l,m,r) <-> distributed-spectral adapter"
```

> Note: `from_sht_spectral`'s Allreduce is over the *spectral* communicator (disjoint owned blocks summed), NOT the old per-(θ,φ) gather. This is bounded O(nlm), not the replicate-everything gather.

---

## Task 3: Scalar synthesis via `dist_synthesis`

**Files:**
- Modify: `src/physics/nonlinear.jl` (`scalar_spectral_to_physical!`)
- Test: `test/theta_dist_transform.jl`

- [ ] **Step 1: Write the failing roundtrip test** (append to `test/theta_dist_transform.jl`)

```julia
@testset "scalar synthesis (theta-dist) matches reference" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=16, nr=4)
    dom = GeoDynamo.create_radial_domain(4)
    tf  = GeoDynamo.create_shtns_temperature_field(Float64, cfg, dom)
    GeoDynamo.set_analytical_initial_conditions!(tf, :temperature, :conductive, amplitude=1.0)
    sr0 = copy(parent(tf.spectral.data_real)); si0 = copy(parent(tf.spectral.data_imag))
    GeoDynamo.scalar_spectral_to_physical!(tf.spectral, tf.temperature)
    GeoDynamo.scalar_physical_to_spectral!(tf.temperature, tf.spectral)
    @test maximum(abs.(parent(tf.spectral.data_real) .- sr0)) < 1e-10
    @test maximum(abs.(parent(tf.spectral.data_imag) .- si0)) < 1e-10
end
```

- [ ] **Step 2: Run it on the CURRENT (gather) code, expect PASS** (establishes the reference contract before refactor)

Run: `julia --project=. -e 'using Test,GeoDynamo,MPI; include("test/theta_dist_transform.jl")'`
Expected: PASS (the existing gather path already roundtrips; this test now guards the behavior we must preserve).

- [ ] **Step 3: Rewrite `scalar_spectral_to_physical!` to use `dist_synthesis` per level**

Replace the gather+serial body (physics/nonlinear.jl ~305–364) so that, per radial level, it builds the full (l,m) matrix from the owned modes, adapts it, and calls `dist_synthesis` into the θ-local physical slab:

```julia
function scalar_spectral_to_physical!(spec::SpectralFieldType{T}, phys::PhysicalFieldType{T}) where T
    config = spec.config
    sht    = config.sht_config
    nr     = size(parent(phys.data), 3)
    pen    = config.pencils.theta_phys            # θ-dist / φ-local prototype
    @inbounds for r in 1:nr
        alm_mat = scalar_level_coeff_matrix(spec, r, config)   # existing gather of one level -> full (l,m)
        sht_alm = to_sht_spectral(config, alm_mat)
        fθφ     = SHTnsKit.dist_synthesis(sht, sht_alm; prototype_θφ = theta_phys_prototype(config, r),
                                          real_output = true)
        store_theta_phys_level!(phys, fθφ, r)
    end
    return phys
end
```

`scalar_level_coeff_matrix` reuses the existing `fill_scalar_coeff_buffer!` + the spectral Allreduce for ONE level (already O(nlm)); `theta_phys_prototype`/`store_theta_phys_level!` are thin helpers writing the θ-local slab of `phys.data[:, :, r]`. Keep them in nonlinear.jl next to this function.

- [ ] **Step 4: Run the roundtrip test, expect PASS**

Run: same as Step 2. Expected: PASS (err < 1e-10) — synthesis now via `dist_synthesis`.

- [ ] **Step 5: Commit**

```bash
git add src/physics/nonlinear.jl test/theta_dist_transform.jl
git commit -m "feat(transform): scalar synthesis via dist_synthesis (theta-dist)"
```

---

## Task 4: Scalar analysis via `dist_analysis`

**Files:** Modify `src/physics/nonlinear.jl` (`scalar_physical_to_spectral!`); Test reuses the Task 3 roundtrip.

- [ ] **Step 1:** The roundtrip test from Task 3 already covers analysis; it will FAIL once Task 3 lands if analysis still gathers a θ-dist field. Run it to confirm the failure:

Run: `julia --project=. -e 'using Test,GeoDynamo,MPI; include("test/theta_dist_transform.jl")'`
Expected: FAIL (layout mismatch — physical is now θ-dist/φ-local, old analysis expects gathered slabs).

- [ ] **Step 2: Rewrite `scalar_physical_to_spectral!` to use `dist_analysis` per level**

```julia
function scalar_physical_to_spectral!(phys::PhysicalFieldType{T}, spec::SpectralFieldType{T}) where T
    config = spec.config; sht = config.sht_config
    nr = size(parent(phys.data), 3)
    @inbounds for r in 1:nr
        fθφ     = theta_phys_level_pencil(phys, r, config)     # wrap slab as PencilArray
        sht_alm = SHTnsKit.dist_analysis(sht, fθφ)
        alm_mat = from_sht_spectral(config, sht_alm)
        store_scalar_level_coeffs!(spec, alm_mat, r, config)   # full (l,m) -> owned (l,m,r) storage
    end
    return spec
end
```

`store_scalar_level_coeffs!` reuses the existing `cpu_store_scalar_coefficients!` logic (write owned modes from the full matrix).

- [ ] **Step 3: Run roundtrip test, expect PASS** (err < 1e-10). Run as Step 1.

- [ ] **Step 4: Commit**

```bash
git add src/physics/nonlinear.jl
git commit -m "feat(transform): scalar analysis via dist_analysis (theta-dist)"
```

---

## Task 5: Vector synthesis/analysis via `dist_*_sphtor!`

**Files:** Modify `src/fields/transforms.jl` (`shtnskit_vector_synthesis!`, `shtnskit_vector_analysis!`); Test: `test/theta_dist_transform.jl`.

- [ ] **Step 1: Write failing vector roundtrip test** (velocity tor/pol → physical → back)

```julia
@testset "vector synth/analysis (theta-dist) roundtrip" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=16, nr=4)
    dom = GeoDynamo.create_radial_domain(4)
    vf  = GeoDynamo.create_shtns_velocity_fields(Float64, cfg, dom)
    GeoDynamo.set_analytical_initial_conditions!(vf, :velocity, :solid_body, amplitude=1.0)
    t0 = copy(parent(vf.toroidal.data_real)); p0 = copy(parent(vf.poloidal.data_real))
    GeoDynamo.shtnskit_vector_synthesis!(vf.toroidal, vf.poloidal, vf.velocity; domain=dom)
    GeoDynamo.shtnskit_vector_analysis!(vf.velocity, vf.toroidal, vf.poloidal; domain=dom)
    @test maximum(abs.(parent(vf.toroidal.data_real) .- t0)) < 1e-8
    @test maximum(abs.(parent(vf.poloidal.data_real) .- p0)) < 1e-8
end
```

- [ ] **Step 2: Run, expect FAIL** (layout mismatch after Tasks 1–4). Run as in Task 3 Step 2.

- [ ] **Step 3: Rewrite the tangential vector transform via `dist_*_sphtor!`** using the in-place signature pinned in Task 0. Per radial level: adapt (S,T) spectral matrices → `dist_synthesis_sphtor!(plan, vt, vp, Slm, Tlm)` into θ-local (vt,vp) slabs; analysis the reverse. The radial (vr from poloidal) `mie_pol_coeffs` path stays as-is (already buffered + `domain!==nothing`). Show the per-level call:

```julia
    SHTnsKit.dist_synthesis_sphtor!(vplan, vt_slab, vp_slab, to_sht_spectral(cfg, S_mat), to_sht_spectral(cfg, T_mat))
```

(`vplan` from `SHTnsKit`'s sphtor plan constructor — confirm name in ext/ParallelTransforms.jl from the Task 0 `methods(...)` output.)

- [ ] **Step 4: Run vector roundtrip, expect PASS** (err < 1e-8). Run as Step 2.

- [ ] **Step 5: Commit**

```bash
git add src/fields/transforms.jl test/theta_dist_transform.jl
git commit -m "feat(transform): vector synth/analysis via dist_*_sphtor (theta-dist)"
```

---

## Task 6: Solver integration + full-suite gate

**Files:** Modify any call site that assumed φ-distribution (audit `pencil.axes_local` users in BC apply / diagnostics / IO); Test: full suite + MPI invariants.

- [ ] **Step 1: Audit φ-distribution assumptions**

Run: `grep -rn 'axes_local\|range_local(.*, *2)' src --include='*.jl'`
For each consumer of the physical pencil's dim-2 (φ) range, confirm φ-local is handled (φ range is now the full `1:nlon`). Fix any that assumed a partial φ range.

- [ ] **Step 2: Run the single-rank full suite, expect green**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' > /tmp/p1_suite.log 2>&1; grep -E 'passed|failed' /tmp/p1_suite.log | tail -2`
Expected: `2793 passed` (or current baseline), 0 failed, 2 broken.

- [ ] **Step 3: Run the 2-rank MPI invariant + roundtrip, expect green**

Run: `julia --project=. -e 'using MPI; jl=Base.julia_cmd()[1]; MPI.mpiexec() do m; run(`$m -n 2 $jl --project=. -e "using Test,GeoDynamo; include(\"test/theta_dist_transform.jl\"); include(\"test/mpi_parallel_invariants.jl\")"`); end'`
Expected: all testsets pass (roundtrip err < 1e-10/1e-8; invariants 15/15).

- [ ] **Step 4: Scaling evidence (not a gate)**

Run: `SHT_LMAX_LIST=128,256 julia --project=. -e 'using MPI; jl=Base.julia_cmd()[1]; MPI.mpiexec() do m; run(`$m -n 2 $jl --project=. scripts/sht_scaling_benchmark.jl`); end'`
Expected: θ-decomposition speedup ≥ 1 at lmax 256 (the standalone path; in-solver cluster scaling validated separately).

- [ ] **Step 5: Commit + open PR**

```bash
git add -A
git commit -m "feat(transform): wire theta-dist transform into solver; phi-local audit"
```

---

## Self-review notes (author)

- **Spec coverage:** layout change (Task 1), adapter (Task 2), scalar synth/analysis (Tasks 3–4), vector (Task 5), φ-local audit + suite gate (Task 6). Radial solve untouched (no task — by design). ✓
- **Type consistency:** `to_sht_spectral`/`from_sht_spectral` used identically in Tasks 2–5; `theta_phys` pencil field name consistent (Tasks 1,3,4).
- **Known soft spots (require Task 0 spike output before coding Tasks 2/5):** exact `SHTAlm` type and `dist_synthesis_sphtor!` plan/signature. The spike pins them; do Task 0 first.
- **Scaling validation:** correctness is gated here; in-solver strong scaling needs a cluster (out of scope to validate in this env).
