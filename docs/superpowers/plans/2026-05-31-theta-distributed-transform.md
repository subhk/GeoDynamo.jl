# θ-Distributed Transform (Phase 1 of r×θ) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace GeoDynamo's gather-replicate SH transform with SHTnsKit 1.2.10's θ-distributed transform, making the (currently 97%-serial) transform phase scalable on the θ-axis while keeping the radial dimension local.

**Architecture:** The process grid becomes **1D-θ** (`proc_dims=(nprocs,1)`). KEY EMPIRICAL
FINDING (verified 2 ranks): under `(P,1)`, the EXISTING physical-field pencil `pencils.r`
(`Pencil(topology,dims,(1,2))`) is already **θ-distributed / φ-local(1:nlon) / r-local** —
identical to `pencils.φ`. So **physical fields do NOT move pencils**, and every
`axes_local[2]` φ-index map degrades to the identity `1:nlon` (works transparently — NO
field-constructor / I/O / index-map / transpose-routing rewrite). The only structural
change is forcing `(P,1)`; the transform functions then feed each radial level to
`dist_synthesis`/`dist_analysis` (scalar) and `dist_synthesis_sphtor!`/`dist_analysis_sphtor!`
(vector). A 2D `(nlat,nlon)` θ-distributed prototype pencil (θ-split == `pencils.r`,
verified) supplies the `prototype_θφ` for the dist calls. The spectral side of `dist_*` is a
**dense `(lmax+1,mmax+1)` matrix replicated on all ranks** (Task 0 spike), so the adapter is
a thin seam over GeoDynamo's existing per-level dense-matrix builder. The implicit radial
solve is untouched (r stays local). Phase 2 (distribute r + r↔lm transpose, restoring a 2D
`r×θ` grid) is a later plan.

> Earlier blast-radius fear (repoint fields r→φ, rewrite I/O + index maps, ~15–20 sites)
> was based on assuming the 2D topology persists. Forcing `(P,1)` makes `pencils.r` φ-local
> directly, collapsing the change to: topology + transforms + one invariant-test update.

**Tech Stack:** Julia, SHTnsKit ≥1.2.10 (`ext/ParallelTransforms.jl`), PencilArrays, PencilFFTs, MPI.

**Execution note:** Run in an isolated git worktree (`superpowers:using-git-worktrees`) off the current branch — `main` is being edited by another session. Validate correctness here; in-solver *scaling* is validated on cluster hardware later (laptop single-node MPI cannot show it).

---

## File Structure

- `src/parallel/pencils.jl` (modify) — force `proc_dims=(nprocs,1)` (1D-θ); add a 2D `(nlat,nlon)` θ-distributed prototype pencil (`theta_phys`) for the dist_* calls. Spectral pencil contract preserved (now l-distributed only under `(P,1)`; r local — radial solve unaffected).
- `src/parallel/spectral_pencil_adapter.jl` (create) — thin `(l,m,r)` ⇄ dense `(lmax+1,mmax+1)` matrix seams (`to_sht_spectral`/`from_sht_spectral`). One responsibility; isolated + unit-tested. Included from the root module after `transforms/spectral.jl`.
- `src/physics/nonlinear.jl` (modify) — `scalar_spectral_to_physical!`, `scalar_physical_to_spectral!`.
- `src/fields/transforms.jl` (modify) — `shtnskit_vector_synthesis!`, `shtnskit_vector_analysis!`.
- `test/mpi_parallel_invariants.jl` (modify, Task 6) — update the layout invariant from θ×φ-distributed to 1D-θ (φ-local).
- `test/theta_dist_adapter.jl` (create) — adapter round-trip unit tests.
- `test/theta_dist_transform.jl` (create) — transform roundtrip (serial + MPI) tests.

---

## Task 0: Spike — pin the SHTnsKit dist API shapes (no production code) — ✅ DONE 2026-05-31

**RESULTS (single + 2 ranks, SHTnsKit 1.2.10):**

- `dist_analysis(cfg, f_θφ::PencilArray) → Matrix{ComplexF64}` of size `(lmax+1, mmax+1)`,
  **DENSE and REPLICATED on every rank** (NOT a PencilArray; size identical np=1 vs np=2).
  The θ-Allreduce inside produces the complete spectral matrix everywhere.
- `dist_synthesis(cfg, alm::AbstractMatrix; prototype_θφ=f_θφ, real_output=true)` →
  **θ-distributed physical** (np=2: `(9,33)` = nlat 18 split 9/rank, φ-local nlon 33 full).
  Dispatches on `AbstractMatrix` (dense) — the spectral input is a plain dense matrix.
- `dist_synthesis_sphtor!(plan::DistSphtorPlan, Vt::PencilArray, Vp::PencilArray, S::AbstractMatrix, T::AbstractMatrix)`
  — spectral S,T are **dense matrices**; Vt,Vp are θ-distributed PencilArrays.
- `dist_analysis_sphtor!(plan::DistSphtorPlan, S::AbstractMatrix, T::AbstractMatrix, Vt::PencilArray, Vp::PencilArray)`
- `create_spectral_pencil(cfg)` / `create_spatial_pencil(cfg)` exist but are **NOT needed** —
  the spectral side is a dense matrix, no PencilArray packing required.
- Extension load requires `using MPI, PencilArrays, PencilFFTs` all three (SHTnsKitParallelExt trigger).

**Consequence:** the "highest-risk adapter" shrinks. The SHTnsKit-side spectral is a dense
`(lmax+1,mmax+1)` matrix == exactly what GeoDynamo's `extract_coefficients_for_shtnskit`
already builds. `to_sht_spectral` = build dense from owned modes (existing gather);
`from_sht_spectral` = slice dense → owned modes (NO Allreduce — `dist_analysis` already
returned the complete replicated matrix). See revised Task 2.

<details><summary>original spike steps (kept for reference)</summary>

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

</details>

---

## Task 1: 1D-θ topology + 2D θ-distributed prototype pencil

Two coupled facts (both verified empirically on 2 ranks, `/tmp/check_1d_topo.jl`):
1. Under `proc_dims=(nprocs,1)`, the existing `pencils.r = Pencil(topology,dims,(1,2))` is
   **θ-distributed / φ-local(1:nlon) / r-local** — the layout the dist transform needs. No
   field repoint required; physical fields stay in `pencils.r`.
2. A 2D `Pencil((nlat,nlon),(1,),comm)` has the **same θ-split** as `pencils.r` (rank0
   `1:6`, rank1 `7:12` for nlat=12). It is the `prototype_θφ` the dist calls consume.

So Task 1 = (a) force `(nprocs,1)` in `create_pencil_topology`, and (b) add the 2D θ
prototype pencil to the pencils NamedTuple as `theta_phys`.

**Files:**
- Modify: `src/parallel/pencils.jl` (`create_pencil_topology`, `create_computation_pencils`)
- Test: `test/theta_dist_transform.jl`

- [ ] **Step 1: Write the failing test**

```julia
using Test, GeoDynamo, MPI, PencilArrays
MPI.Initialized() || MPI.Init()
@testset "1D-theta layout + prototype pencil" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=16, nr=4)
    # physical-field pencil is phi-local, r-local under the forced (P,1) grid
    rloc = PencilArrays.range_local(cfg.pencils.r)
    @test length(rloc[2]) == 16          # φ FULLY LOCAL (1:nlon)
    @test length(rloc[3]) == 4           # r LOCAL (1:nr)
    # NEW 2D (nlat,nlon) theta-distributed prototype, same theta-split as pencils.r
    p2 = cfg.pencils.theta_phys
    p2loc = PencilArrays.range_local(p2)
    @test length(p2loc[2]) == 16         # φ full on the 2D prototype too
    @test p2loc[1] == rloc[1]            # θ-split MATCHES pencils.r (critical for slab mapping)
end
```

- [ ] **Step 2: Run it, expect FAIL** (`theta_phys` not a field)

Run: `julia --project=. -e 'using Test,GeoDynamo,MPI,PencilArrays; include("test/theta_dist_transform.jl")'`
Expected: FAIL — `type NamedTuple has no field theta_phys`.

- [ ] **Step 3a: Force the 1D-θ grid in `create_pencil_topology`**

In `src/parallel/pencils.jl`, replace the `optimize`-driven `proc_dims` selection with the
Phase-1 1D-θ grid (keep `optimize_process_topology` defined for Phase 2):

```julia
    # Phase 1 (θ-distributed transform) uses a 1D-θ process grid: θ distributed,
    # φ + r local on every rank. Under (nprocs,1), pencils.r is already
    # θ-dist / φ-local / r-local — the layout SHTnsKit's dist_* transform consumes.
    # Phase 2 (r×θ) will reintroduce a 2D grid via optimize_process_topology.
    proc_dims = (nprocs, 1)
```

- [ ] **Step 3b: Add the 2D θ-prototype pencil in `create_computation_pencils`**

```julia
    # 2D (nlat,nlon) θ-distributed / φ-local prototype for SHTnsKit's per-level
    # dist_synthesis/dist_analysis (prototype_θφ). Its θ-split matches pencils.r,
    # so phys.data[:,:,r] maps directly onto this prototype's local block.
    pencil_theta_phys = Pencil(topology, (nlat, nlon), (1,))
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

> `Pencil(topology, (nlat,nlon), (1,))` builds a 2D pencil over the 2D `(P,1)` topology
> decomposing only dim 1 (θ); the size-1 second topology axis leaves φ local. Confirmed
> valid + θ-split-matching on 2 ranks.

- [ ] **Step 4: Run test, expect PASS** (single rank: θ full 1:12, φ 1:16, r 1:4; 2-rank split checked in Task 6 MPI gate). Run as Step 2.

- [ ] **Step 5: Commit**

```bash
git add src/parallel/pencils.jl test/theta_dist_transform.jl
git commit -m "feat(parallel): force 1D-theta grid + add 2D theta prototype pencil"
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

- [ ] **Step 3: Implement the adapter (DENSE matrix — per Task 0 spike)**

The Task 0 spike showed SHTnsKit's `dist_*` spectral side is a **dense `(lmax+1,mmax+1)`
`Matrix{ComplexF64}` replicated on every rank** — NOT a PencilArray. So the adapter does
not pack a spectral pencil. It maps between GeoDynamo's distributed `(l,m,r)` packed storage
(one radial level) and that dense matrix.

Create `src/parallel/spectral_pencil_adapter.jl`:

```julia
# Adapter: GeoDynamo (l,m,r) packed spectral storage (one radial level) <-> the
# DENSE (lmax+1, mmax+1) coefficient matrix that SHTnsKit's dist_* consume/produce.
# Task 0 spike pinned: dist_analysis returns Matrix{ComplexF64} replicated on all
# ranks; dist_synthesis consumes an AbstractMatrix. No spectral PencilArray.

"""
    to_sht_spectral(cfg, alm_mat::AbstractMatrix{ComplexF64}) -> Matrix{ComplexF64}

Hand a full (lmax+1, mmax+1) coefficient matrix to the dist transform. The matrix
is already the dense form `dist_synthesis` wants, so this is (currently) identity /
defensive copy. Kept as a named seam so the call sites read symmetrically with
`from_sht_spectral` and so a future distributed-spectral layout can be slotted here.
"""
to_sht_spectral(cfg, alm_mat::AbstractMatrix{ComplexF64}) = alm_mat

"""
    from_sht_spectral(cfg, alm::AbstractMatrix{ComplexF64}) -> Matrix{ComplexF64}

`dist_analysis` already returns the complete dense matrix replicated on every rank,
so no gather is needed — return it (defensive copy). Named seam, symmetric with
`to_sht_spectral`.
"""
from_sht_spectral(cfg, alm::AbstractMatrix{ComplexF64}) = alm
```

> The real `(l,m,r)` ⇄ dense-matrix work lives in the existing per-level helpers
> (`extract_coefficients_for_shtnskit` builds the dense matrix from owned modes via the
> O(nlm) spectral gather; `cpu_store_scalar_coefficients!` writes owned modes back). Tasks
> 3–4 call those plus these seams; the round-trip test below pins the seam contract.

Include it from `src/GeoDynamo.jl` right after `include("transforms/spectral.jl")`:

```julia
    include("parallel/spectral_pencil_adapter.jl")
```

The Step-1 round-trip test still holds: `from_sht_spectral(cfg, to_sht_spectral(cfg, M)) ≈ M`.

- [ ] **Step 4: Run test, expect PASS**

Run: same as Step 2. Expected: PASS (`back ≈ alm_mat`).

- [ ] **Step 5: Commit**

```bash
git add src/parallel/spectral_pencil_adapter.jl src/GeoDynamo.jl test/theta_dist_adapter.jl
git commit -m "feat(parallel): (l,m,r) <-> dense-spectral adapter seams"
```

> Note: the dense spectral matrix is replicated by `dist_analysis`'s own θ-Allreduce
> (O(nlm)); GeoDynamo's per-level gather to build the matrix for synthesis is the same
> bounded O(nlm) it already does — NOT the old replicate-everything physical-grid gather.
> The expensive Legendre work is what `dist_*` now distributes over θ.

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

Run: `grep -rn 'axes_local\|range_local' src --include='*.jl'`
For each consumer of the physical pencil's dim-2 (φ) range, confirm φ-local is handled (φ
range is now the full `1:nlon`, so local-to-global φ maps are the identity). The
investigation (2026-05-31) found these degrade transparently — `j_global = φ_range[j_local]`
with `φ_range=1:nlon` is `j_global==j_local`. VERIFY each still holds; fix any that
hard-assume a partial φ range. Known transparent sites: `physics/nonlinear.jl` (extract/store
physical slice), `fields/transforms.jl` (`store_*_component_generic!`), `io/restart.jl`,
`io/netcdf.jl`.

- [ ] **Step 1b: Update the layout invariant test**

`test/mpi_parallel_invariants.jl` currently asserts the physical grid is θ×φ-distributed
(θ×φ exact-cover). Under the 1D-θ grid the invariant CHANGES BY DESIGN: φ is now local
(`1:nlon` on every rank), θ is distributed and θ-exact-covers `1:nlat`, r is local. Update
the relevant assertions to the 1D-θ contract (φ full on every rank; θ partition exact-covers;
r local). Do NOT weaken unrelated invariants. Re-run to confirm the updated set is green at 2
ranks.

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
