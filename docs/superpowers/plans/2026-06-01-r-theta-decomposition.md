# r×θ Decomposition (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 2D `r×θ` MPI process grid so GeoDynamo's SH transform scales on both the radial and latitudinal axes, with an r↔lm transpose bracketing the (unchanged) banded radial solve.

**Architecture:** Explicit grid `GEODYNAMO_PROC_GRID="θxr"` → split `COMM_WORLD` into a θ-subcommunicator (SH transform) and an r-subcommunicator (transpose). Physical fields become r-distributed; the per-radial-level dist transform runs on the θ-subcomm; two spectral orientations (`spec_solve` = modes-dist/r-local = the existing Phase-1 layout, unchanged for the radial solve; `spec_transform` = r-dist + mode-axis-dist over θ) are connected by a PencilArrays r↔lm transpose. Correctness-only gate (transpose identity, transform roundtrip, radial-solve equivalence vs Phase-1, full suite); scaling validated on a cluster later.

**Tech Stack:** Julia, SHTnsKit ≥1.2.10 (`dist_*`/`dist_*_sphtor` on a sub-communicator), PencilArrays (transpose machinery), MPI (`Comm_split`).

**Spec:** `docs/superpowers/specs/2026-06-01-r-theta-decomposition-design.md`.

**Execution note:** Implement in an isolated git worktree off main `71d94b9` (or current main) — main is under concurrent edits. Julia binary (shim broken): `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia`. MPI via `MPI.mpiexec()` (MPICH), NOT a system mpiexec. Baseline suite green = 2814 pass / 2 broken.

---

## File Structure

- `src/parallel/process_grid.jl` (create) — parse `GEODYNAMO_PROC_GRID`, validate against `nprocs/nlat/nr`, build θ/r subcommunicators. One responsibility; unit-tested.
- `src/transforms/spectral.jl` (modify) — `create_pencil_decomposition_shtnskit`: 2D grid from the parser; `theta_phys` on the θ-subcomm; add `spec_transform` pencil + θ/r subcomms to the returned config.
- `src/parallel/pencils.jl` (modify) — `create_pencil_topology`/`create_computation_pencils`: same 2D grid; reconcile `theta_phys`.
- `src/parallel/transposes.jl` (modify) — add the r↔lm spectral transpose plan(s).
- `src/physics/nonlinear.jl` (modify) — scalar transforms: θ-subcomm dist calls + transpose to/from `spec_solve`.
- `src/solver/numerics.jl` + `src/fields/transforms.jl` (modify) — vector transforms likewise.
- `src/solver/mainloop.jl` (modify, Task 6) — bracket the radial solve with the r↔lm transpose.
- `test/r_theta_grid.jl` (create) — process-grid parsing + subcomm + 2D pencil tests.
- `test/r_theta_transpose.jl` (create) — transpose-roundtrip-identity + transform-roundtrip tests.
- `test/r_theta_equivalence.jl` (create) — `solver_step!` 2D-vs-1D equivalence.

---

## Task 0: Spike — θ-subcommunicator `dist_*` feasibility (GATES EVERYTHING)

**Files:** `/tmp/spike_subcomm.jl` (throwaway). No production code.

> Rationale: in Phase 1 every rank was the θ-group (`COMM_WORLD`). Phase 2 needs the SH transform to run on a θ-SUBcommunicator (the ranks sharing one r-slab). If SHTnsKit's `dist_synthesis`/`dist_analysis`/`dist_*_sphtor` cannot run on a sub-comm (e.g. hardcode `COMM_WORLD`), the whole transpose architecture is blocked → STOP and escalate (fall back to the replicate-radial-solve variant). This spike answers that before any production code.

- [ ] **Step 1: Write the spike**

```julia
using MPI; MPI.Init()
using SHTnsKit, PencilArrays, PencilFFTs
comm = MPI.COMM_WORLD; rank = MPI.Comm_rank(comm); np = MPI.Comm_size(comm)
# split into 2 θ-groups (color = r-group id). e.g. np=4 -> 2 r-groups × 2 θ-ranks.
r_ranks = 2; θ_ranks = np ÷ r_ranks
rgrp = rank ÷ θ_ranks                      # which r-slab this rank is in
θcomm = MPI.Comm_split(comm, rgrp, rank)   # θ-subcomm: ranks sharing this r-slab
lmax = 12; nlat = lmax+2; nlon = 2*lmax+1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)
# band-limited dense field
alm0 = zeros(ComplexF64, lmax+1, lmax+1)
for m in 0:lmax, l in m:lmax; s=1.0/(1+l); alm0[l+1,m+1]= m==0 ? complex(s) : complex(s,0.5s); end
f_full = SHTnsKit.synthesis(cfg, alm0; real_output=true)
# θ-distributed PencilArray OVER THE θ-SUBCOMM (not COMM_WORLD)
pen = Pencil((nlat,nlon),(1,), θcomm)
fθ  = PencilArray(pen, zeros(Float64, PencilArrays.size_local(pen)...))
lr = PencilArrays.range_local(pen)
for (jl,jg) in enumerate(lr[2]), (il,ig) in enumerate(lr[1]); parent(fθ)[il,jl]=f_full[ig,jg]; end
alm = SHTnsKit.dist_analysis(cfg, fθ)        # MUST distribute over θcomm only
frec = SHTnsKit.dist_synthesis(cfg, alm; prototype_θφ=fθ, real_output=true)
# correctness within this θ-group: alm should match serial; roundtrip machine precision
err_a = maximum(abs.(alm .- SHTnsKit.analysis(cfg, f_full)))
recp = frec isa PencilArray ? parent(frec) : frec
err_r = 0.0
for (jl,jg) in enumerate(lr[2]), (il,ig) in enumerate(lr[1]); err_r=max(err_r, abs(recp[il,jl]-f_full[ig,jg])); end
g_a = MPI.Allreduce(err_a, MPI.MAX, comm); g_r = MPI.Allreduce(err_r, MPI.MAX, comm)
rank==0 && println("SUBCOMM_ANALYSIS_ERR=", g_a, "  ROUNDTRIP_ERR=", g_r,
                   "  => ", (g_a<1e-10 && g_r<1e-10) ? "SUBCOMM_OK" : "SUBCOMM_FAIL")
# also confirm dist_analysis did NOT reduce across the OTHER r-group:
# seed different fields per r-group; alm must differ between groups (no cross-group comm)
MPI.Finalize()
```

- [ ] **Step 2: Run on 4 ranks (2 r-groups × 2 θ-ranks)**

Run: `JL=~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia; $JL --project=. -e 'using MPI; jl=ENV["HOME"]*"/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia"; MPI.mpiexec() do m; run(`$m -n 4 $jl --project=. /tmp/spike_subcomm.jl`); end'`
Expected: `SUBCOMM_OK` (analysis + roundtrip < 1e-10). Decision point:
- **`SUBCOMM_OK`** → record that `dist_*` honor the prototype's sub-comm; proceed to Task 1.
- **`SUBCOMM_FAIL`** or error mentioning `COMM_WORLD` → STOP. Report BLOCKED with the exact error; the architecture must change (replicate-solve fallback) — escalate to the human.

- [ ] **Step 3: Record the result** in this plan (replace this line) and delete the spike (`rm /tmp/spike_subcomm.jl`).

---

## Task 1: `GEODYNAMO_PROC_GRID` parser + θ/r subcommunicators

**Files:** Create `src/parallel/process_grid.jl`; Modify `src/GeoDynamo.jl` (include it); Test `test/r_theta_grid.jl`.

- [ ] **Step 1: Write the failing test**

```julia
using Test, GeoDynamo
@testset "GEODYNAMO_PROC_GRID parsing" begin
    # explicit grid, valid
    @test GeoDynamo.parse_proc_grid("4x2", 8) == (4, 2)        # (θ_ranks, r_ranks)
    @test GeoDynamo.parse_proc_grid("8x1", 8) == (8, 1)
    # product must equal nprocs
    @test_throws ErrorException GeoDynamo.parse_proc_grid("4x2", 6)
    # np==1 → trivial (1,1) without requiring the env var
    @test GeoDynamo.parse_proc_grid(nothing, 1) == (1, 1)
    # np>1 without the env var → error
    @test_throws ErrorException GeoDynamo.parse_proc_grid(nothing, 4)
end
```

- [ ] **Step 2: Run, expect FAIL** (`parse_proc_grid` undefined).

Run: `$JL --project=. -e 'using Test,GeoDynamo; include("test/r_theta_grid.jl")'`

- [ ] **Step 3: Implement `src/parallel/process_grid.jl`**

```julia
"""
    parse_proc_grid(spec::Union{AbstractString,Nothing}, nprocs::Int) -> (θ_ranks, r_ranks)

Parse an explicit process grid "θxr" (e.g. "4x2"). At nprocs==1 returns (1,1) without
requiring `spec`. At nprocs>1 `spec` is REQUIRED and must satisfy θ_ranks·r_ranks==nprocs.
"""
function parse_proc_grid(spec::Union{AbstractString,Nothing}, nprocs::Int)
    if nprocs == 1
        return (1, 1)
    end
    spec === nothing && error("GEODYNAMO_PROC_GRID must be set at nprocs>1 (e.g. \"4x2\" = θ_ranks×r_ranks)")
    parts = split(spec, 'x')
    length(parts) == 2 || error("GEODYNAMO_PROC_GRID must be \"θxr\" (e.g. \"4x2\"), got \"$spec\"")
    θr = parse(Int, parts[1]); rr = parse(Int, parts[2])
    θr * rr == nprocs || error("GEODYNAMO_PROC_GRID $spec = $(θr*rr) ranks != nprocs=$nprocs")
    return (θr, rr)
end

read_proc_grid(nprocs::Int) = parse_proc_grid(get(ENV, "GEODYNAMO_PROC_GRID", nothing), nprocs)

"""
    make_subcomms(comm, θ_ranks, r_ranks) -> (θ_comm, r_comm)

Split `comm` (row-major rank = r_group·θ_ranks + θ_index) into the θ-subcomm (ranks
sharing an r-slab) and the r-subcomm (ranks sharing a θ-column).
"""
function make_subcomms(comm, θ_ranks::Int, r_ranks::Int)
    rank = MPI.Comm_rank(comm)
    r_group = rank ÷ θ_ranks          # which r-slab
    θ_index = rank % θ_ranks          # position within the θ-group
    θ_comm = MPI.Comm_split(comm, r_group, rank)   # ranks sharing this r-slab
    r_comm = MPI.Comm_split(comm, θ_index, rank)   # ranks sharing this θ-column
    return θ_comm, r_comm
end
```

Include from `src/GeoDynamo.jl` near the other `parallel/` includes (before `transforms/spectral.jl`, since spectral.jl will call `read_proc_grid`/`make_subcomms`).

- [ ] **Step 4: Run test, expect PASS.** Run as Step 2.

- [ ] **Step 5: Commit**

```bash
git add src/parallel/process_grid.jl src/GeoDynamo.jl test/r_theta_grid.jl
git commit -m "feat(parallel): GEODYNAMO_PROC_GRID parser + theta/r subcommunicators"
```

---

## Task 2: 2D grid + r-distributed physical pencil + θ-subcomm `theta_phys`

**Files:** Modify `src/transforms/spectral.jl` (`create_pencil_decomposition_shtnskit`) and `src/parallel/pencils.jl`; Test append to `test/r_theta_grid.jl`.

- [ ] **Step 1: Append the failing test**

```julia
using MPI, PencilArrays
MPI.Initialized() || MPI.Init()
@testset "2D r×θ pencils (single rank)" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=20, nr=6)
    # physical pencil: φ local; r now a DISTRIBUTED dim (local==full only at np=1)
    rloc = PencilArrays.range_local(cfg.pencils.r)
    @test length(rloc[2]) == 20                 # φ local
    @test haskey(keys(cfg.pencils), :theta_phys) # prototype present
    # config exposes the subcomms
    @test cfg.θ_comm !== nothing
    @test cfg.r_comm !== nothing
end
```

- [ ] **Step 2: Run, expect FAIL** (`cfg.θ_comm` not a field / wrong pencil). Run as Task 1 Step 2 form.

- [ ] **Step 3: Rewire `create_pencil_decomposition_shtnskit`**

Replace the Phase-1 `proc_dims = (nprocs, 1)` with:
```julia
    θ_ranks, r_ranks = read_proc_grid(nprocs)
    proc_dims = (θ_ranks, r_ranks)
    θ_comm, r_comm = make_subcomms(comm, θ_ranks, r_ranks)
```
The 2D topology `TopoCtor(comm, (θ_ranks, r_ranks))`. Physical pencils:
- `pencil_r = Pencil(topology, dims, (1, 3))` → θ over θ_ranks (dim1), r over r_ranks (dim3), **φ (dim2) local**. (This makes physical θ-dist / φ-local / r-dist.)

> NOTE: re-derive the exact decomp tuples for `pencil_θ/φ/spec` under the 2D grid so the
> invariants hold (φ-local physical; spec modes-dist/r-local). Verify with the test +
> the Task-7 invariants. The Phase-1 `(1,2)` for `pencil_r` gave r-local — change to
> `(1,3)` for r-dist/φ-local.

`theta_phys` on the **θ-subcomm**: `Pencil(MPITopology(θ_comm, (θ_ranks,)), (nlat, nlon), (1,))`. Its θ-split must match `pencil_r`'s θ-split *within the θ-group* — assert in the multi-rank gate (Task 7).

Add `θ_comm`, `r_comm` (and the `spec_transform` pencil — Task 3) to the returned NamedTuple / config struct. Mirror the grid change in `pencils.jl::create_pencil_topology` (+ add `theta_phys` there).

- [ ] **Step 4: Run test, expect PASS** (single rank: r local==full, theta_phys present, subcomms set). Run as Step 2.

- [ ] **Step 5: Commit**

```bash
git add src/transforms/spectral.jl src/parallel/pencils.jl test/r_theta_grid.jl
git commit -m "feat(parallel): 2D r×θ grid, r-distributed physical pencil, theta_phys on theta-subcomm"
```

---

## Task 3: `spec_transform` pencil + r↔lm transpose (+ identity test)

**Files:** Modify `src/transforms/spectral.jl` (add `spec_transform`) and `src/parallel/transposes.jl` (add the plan); Test `test/r_theta_transpose.jl`.

- [ ] **Step 1: Write the failing transpose-roundtrip-identity test**

```julia
using Test, GeoDynamo, MPI, PencilArrays
MPI.Initialized() || MPI.Init()
@testset "r↔lm transpose roundtrip is identity" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=20, nr=6)
    a_solve = GeoDynamo.create_pencil_array(ComplexF64, cfg.pencils.spec; init=:zero)  # solve orientation
    # fill local block with a rank-unique signature
    p = parent(a_solve); p .= ComplexF64(MPI.Comm_rank(MPI.COMM_WORLD) + 1)
    a0 = copy(p)
    b_tr  = GeoDynamo.create_pencil_array(ComplexF64, cfg.pencils.spec_transform; init=:zero)
    GeoDynamo.transpose_solve_to_transform!(b_tr, a_solve, cfg)   # spec_solve -> spec_transform
    GeoDynamo.transpose_transform_to_solve!(a_solve, b_tr, cfg)   # and back
    @test parent(a_solve) == a0          # identity roundtrip
    # global sum preserved through the intermediate orientation
    s0 = MPI.Allreduce(sum(a0), MPI.SUM, MPI.COMM_WORLD)
    sb = MPI.Allreduce(sum(parent(b_tr)), MPI.SUM, MPI.COMM_WORLD)
    @test s0 ≈ sb
end
```

- [ ] **Step 2: Run, expect FAIL** (`spec_transform`/`transpose_*` undefined).

Run: `$JL --project=. -e 'using Test,GeoDynamo,MPI,PencilArrays; include("test/r_theta_transpose.jl")'`

- [ ] **Step 3: Add `spec_transform` pencil + the transpose**

In `create_pencil_decomposition_shtnskit`, add the transform-orientation spectral pencil over the 2D topology and `spec_dims=(nl,nm,nr)`: r distributed over r_ranks + one mode-axis distributed over θ_ranks, the other local. Pin the exact decomp so a PencilArrays transpose to/from `spec` (the solve orientation) is valid:
```julia
    # solve orientation `spec` = Pencil(topology, spec_dims, (1,2))  # (l,m) dist, r local (Phase-1)
    # transform orientation: distribute r (dim3) over r_ranks + m (dim2) over θ_ranks, l (dim1) local
    pencil_spec_transform = Pencil(topology, spec_dims, (2, 3))
```
In `src/parallel/transposes.jl`, build the transpose plan(s) between `spec` and `spec_transform` (reuse `create_transpose_plans`/`Transpositions.Transposition` exactly as the θ↔φ↔r plans are built) and expose:
```julia
transpose_solve_to_transform!(dst, src, cfg) = transpose_with_timer!(dst, src, :spec_solve_to_transform)
transpose_transform_to_solve!(dst, src, cfg) = transpose_with_timer!(dst, src, :spec_transform_to_solve)
```
If `(1,2)`↔`(2,3)` is a two-axis change PencilArrays won't transpose directly, add the intermediate orientation PencilArrays needs and chain it inside these two helpers (keep the public interface identical). The identity test is the gate.

- [ ] **Step 4: Run test, expect PASS** (identity roundtrip + sum preserved). Run as Step 2; also run 2 + 4 ranks (`-n 2`, `-n 4` with `GEODYNAMO_PROC_GRID` set).

- [ ] **Step 5: Commit**

```bash
git add src/transforms/spectral.jl src/parallel/transposes.jl test/r_theta_transpose.jl
git commit -m "feat(parallel): spec_transform pencil + r<->lm transpose (identity-tested)"
```

---

## Task 4: Scalar transforms on θ-subcomm + transpose to solve orientation

**Files:** Modify `src/physics/nonlinear.jl` (`scalar_spectral_to_physical!`, `scalar_physical_to_spectral!`); Test append to `test/r_theta_transpose.jl`.

- [ ] **Step 1: Append the failing roundtrip test** (physical r-dist → spectral solve-orientation → physical)

```julia
@testset "scalar r×θ transform roundtrip" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=20, nr=6)
    dom = GeoDynamo.create_radial_domain(6)
    tf  = GeoDynamo.create_shtns_temperature_field(Float64, cfg, dom)
    sr = parent(tf.spectral.data_real); si = parent(tf.spectral.data_imag); sr.=0; si.=0
    for k in 1:size(sr,3); sr[min(2,size(sr,1)),1,k]=0.7; end
    sr0=copy(sr); si0=copy(si)
    GeoDynamo.scalar_spectral_to_physical!(tf.spectral, tf.temperature)
    GeoDynamo.scalar_physical_to_spectral!(tf.temperature, tf.spectral)
    @test maximum(abs.(parent(tf.spectral.data_real).-sr0)) < 1e-10
    @test maximum(abs.(parent(tf.spectral.data_imag).-si0)) < 1e-10
end
```

- [ ] **Step 2: Run on current code, expect FAIL** (Phase-1 scalar transform assumes r-local + COMM_WORLD; now physical is r-dist + spectral home is solve-orientation). Run as Task 3 Step 2 form.

- [ ] **Step 3: Rewrite the scalar transforms**

`scalar_spectral_to_physical!` (spectral solve-orientation → physical r-dist):
1. `transpose_solve_to_transform!` the spectral into the transform orientation buffer.
2. For each **local** radial level: gather the θ-distributed mode-axis over `cfg.θ_comm` to a full dense `(lmax+1,mmax+1)` (O(nlm), the Phase-1-style coeff gather but on the θ-subcomm); `dist_synthesis(cfg.sht_config, dense; prototype_θφ=cfg.pencils.theta_phys-backed proto, real_output=true)` → θ-local block; `copyto!(view(phys_data,:,:,r_local), out)`.

`scalar_physical_to_spectral!` (physical r-dist → spectral solve-orientation):
1. For each local radial level: wrap `view(phys_data,:,:,r_local)` as a θ-subcomm PencilArray slab; `dist_analysis(cfg.sht_config, slab)` → full dense `(l,m)` (replicated on the θ-subcomm); slice to this rank's θ-distributed mode-axis into the transform-orientation buffer.
2. `transpose_transform_to_solve!` into the spectral solve-orientation storage.

Reuse the cached buffers (`theta_phys_proto`, `theta_phys_slab`) but on the θ-subcomm; add a cached `spec_transform` array (per config). The dense gather/slice over `cfg.θ_comm` (NOT `COMM_WORLD`).

- [ ] **Step 4: Run roundtrip, expect PASS** (<1e-10), single + 2 + 4 ranks. Run as Step 2.

- [ ] **Step 5: Commit**

```bash
git add src/physics/nonlinear.jl test/r_theta_transpose.jl
git commit -m "feat(transform): scalar transform on theta-subcomm + r<->lm transpose (r×θ)"
```

---

## Task 5: Vector transforms on θ-subcomm + transpose (numerics.jl + fields/transforms.jl)

**Files:** Modify `src/solver/numerics.jl` (`vector_spectral_to_physical!`, `vector_physical_to_spectral!`) and `src/fields/transforms.jl` (`shtnskit_vector_synthesis!`, `shtnskit_vector_analysis!`); Test append to `test/r_theta_transpose.jl`.

- [ ] **Step 1: Append the failing vector roundtrip test** (both the solver path AND the non-solver path)

```julia
@testset "vector r×θ transform roundtrip (solver + non-solver)" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=20, nr=6)
    dom = GeoDynamo.create_radial_domain(6)
    for (synth, anal) in ((GeoDynamo.vector_spectral_to_physical!, GeoDynamo.vector_physical_to_spectral!),
                          (GeoDynamo.shtnskit_vector_synthesis!,    GeoDynamo.shtnskit_vector_analysis!))
        vf = GeoDynamo.create_shtns_velocity_fields(Float64, cfg, dom)
        tr=parent(vf.toroidal.data_real); pr=parent(vf.poloidal.data_real); tr.=0; pr.=0
        for k in 1:size(tr,3); tr[min(2,size(tr,1)),1,k]=0.5; end
        tr0=copy(tr); pr0=copy(pr)
        synth === GeoDynamo.vector_spectral_to_physical! ?
            synth(vf.toroidal, vf.poloidal, vf.velocity; domain=dom) :
            synth(vf.toroidal, vf.poloidal, vf.velocity; domain=dom)
        anal === GeoDynamo.vector_physical_to_spectral! ?
            anal(vf.velocity, vf.toroidal, vf.poloidal; domain=dom) :
            anal(vf.velocity, vf.toroidal, vf.poloidal; domain=dom)
        @test maximum(abs.(parent(vf.toroidal.data_real).-tr0)) < 1e-8
        @test maximum(abs.(parent(vf.poloidal.data_real).-pr0)) < 1e-8
    end
end
```

- [ ] **Step 2: Run, expect FAIL** (r-local / COMM_WORLD assumptions). Run as Task 3 Step 2.

- [ ] **Step 3: Rewrite the vector transforms** — apply the SAME pattern as Task 4 to both implementations, using `dist_synthesis_sphtor`/`dist_analysis_sphtor` on `cfg.θ_comm` and the r↔lm transpose for the (S,T) spectral pair. Per local radial level: gather the θ-mode-axis to dense (S,T) over `cfg.θ_comm`, `dist_synthesis_sphtor(cfg.sht_config, S, T; prototype_θφ=proto)` → copy to v_θ/v_φ slabs; v_r via `dist_synthesis` (scalar) as in Phase 1; analysis the reverse + `transpose_transform_to_solve!`. Preserve each path's existing v_r physics factor (numerics.jl `l*(l+1)/r_val`; fields/transforms.jl `l*(l+1)/r_val²`).

- [ ] **Step 4: Run vector roundtrip, expect PASS** (<1e-8), single + 2 + 4 ranks. Run as Step 2.

- [ ] **Step 5: Commit**

```bash
git add src/solver/numerics.jl src/fields/transforms.jl test/r_theta_transpose.jl
git commit -m "feat(transform): vector transforms on theta-subcomm + r<->lm transpose (r×θ)"
```

---

## Task 6: Wire the transpose into the step; radial-solve equivalence

**Files:** Modify `src/solver/mainloop.jl` (and `timestep/imex.jl`/`erk2.jl` only if the transpose must sit there); Test `test/r_theta_equivalence.jl`.

> By Tasks 4–5, each transform already does its own transpose internally (spectral home
> stays the solve orientation; the transform transposes to/from `spec_transform` around
> the dist calls). So the radial solve already sees the solve orientation with NO change.
> Task 6 VERIFIES this end-to-end and fixes any remaining site that assumed r-local
> physical or `COMM_WORLD` reductions in the step (diagnostics, energy norms, CFL).

- [ ] **Step 1: Write the equivalence test** — one `solver_step!` on a 1D-θ grid vs a 2D r×θ grid must agree.

```julia
using Test, GeoDynamo, MPI
MPI.Initialized() || MPI.Init()
@testset "solver_step! 2D r×θ matches 1D-θ" begin
    # Build identical model; step once. Compare spectral state.
    # (Run this file under -n 1 to get the 1D reference, and -n 4 GEODYNAMO_PROC_GRID=2x2
    #  for the 2D run; compare the gathered global spectral arrays.)
    model = GeoDynamo.build_reference_model(lmax=8, nlat=12, nlon=20, nr=8)  # helper, deterministic IC
    GeoDynamo.solver_step!(model)
    sig = GeoDynamo.global_spectral_signature(model)   # gather to rank 0, hash/checksum
    ref = GeoDynamo.load_reference_signature()         # precomputed 1D-θ result (committed fixture)
    MPI.Comm_rank(MPI.COMM_WORLD)==0 && @test sig ≈ ref atol=1e-10
end
```
(If a committed fixture is awkward, instead run BOTH grids in one driver via `MPI.mpiexec` and diff. The invariant: identical post-step spectral state to 1e-10.)

- [ ] **Step 2: Run, expect FAIL** if any step-level site assumed r-local/`COMM_WORLD`. Audit with `grep -rn 'COMM_WORLD\|get_comm()\|axes_local\|range_local' src/solver src/diagnostics src/timestep` and confirm each reduction uses the correct communicator (global energy norms over `COMM_WORLD`; θ-only ops over `θ_comm`).

- [ ] **Step 3: Fix the audited sites** so a 2D-grid step is physics-identical to a 1D step (correct comm per reduction; no partial-r assumption).

- [ ] **Step 4: Run equivalence, expect PASS** (2D == 1D to 1e-10).

- [ ] **Step 5: Commit**

```bash
git add src/solver/mainloop.jl test/r_theta_equivalence.jl
git commit -m "feat(solver): r×θ step equivalence to 1D-theta (comm audit)"
```

---

## Task 7: Full suite + 2D invariants + multi-rank gate

**Files:** Modify `test/mpi_parallel_invariants.jl` (2D contract) + `test/runtests.jl` (wire new tests).

- [ ] **Step 1: Wire `r_theta_grid.jl`, `r_theta_transpose.jl`, `r_theta_equivalence.jl` into `runtests.jl`.**

- [ ] **Step 2: Update `mpi_parallel_invariants.jl`** to the 2D r×θ contract: physical θ-dist/φ-local/**r-dist** (r exact-covers `1:nr` across r_ranks); spectral solve-orientation modes-dist/r-local; the r↔lm transpose preserves global ordering. Set `GEODYNAMO_PROC_GRID` in the multi-rank runner.

- [ ] **Step 3: Single-rank full suite, expect green** (≈ 2814 + new-test count pass / 2 broken / 0 fail).

Run: `$JL --project=. -e 'using Pkg; Pkg.test()' > /tmp/p2_suite.log 2>&1; grep -E 'Extended GeoDynamo tests|passed' /tmp/p2_suite.log | tail -2`

- [ ] **Step 4: Multi-rank gate** (4 ranks, `GEODYNAMO_PROC_GRID=2x2`): invariants + transpose + transform roundtrips + equivalence all green.

Run: `$JL --project=. -e 'using MPI; jl=ENV["HOME"]*"/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia"; withenv("GEODYNAMO_PROC_GRID"=>"2x2") do; MPI.mpiexec() do m; run(`$m -n 4 $jl --project=. -e "using Test,GeoDynamo; include(\"test/r_theta_transpose.jl\"); include(\"test/mpi_parallel_invariants.jl\")"`); end; end'`

- [ ] **Step 5: Commit + summary**

```bash
git add test/runtests.jl test/mpi_parallel_invariants.jl
git commit -m "test(parallel): wire r×θ tests; 2D mpi invariants; multi-rank gate"
```

---

## Self-review notes (author)

- **Spec coverage:** process grid + subcomms (T1), r-dist physical + θ-subcomm theta_phys (T2), spec_transform + transpose (T3), scalar transform (T4), vector transforms both paths (T5), step equivalence + comm audit (T6), suite + 2D invariants (T7). θ-subcomm feasibility gated first (T0). Radial solve untouched (verified, not rewritten). ✓
- **Hard dependency:** T0 gates all; if `SUBCOMM_FAIL`, STOP and escalate (replicate-solve fallback). T3's transpose decomp is pinned in-task with the identity test as gate (Phase-1 precedent for spike-pinned specifics).
- **Type/name consistency:** `parse_proc_grid`/`read_proc_grid`/`make_subcomms` (T1); `cfg.θ_comm`/`cfg.r_comm`/`cfg.pencils.spec_transform`/`cfg.pencils.theta_phys` (T2–T3); `transpose_solve_to_transform!`/`transpose_transform_to_solve!` (T3, used T4–T5). v_r factors preserved per path (T5).
- **Known soft spots:** the exact 2D decomp tuples (T2 `pencil_r`=(1,3); T3 `spec_transform`=(2,3)) and whether the (1,2)↔(2,3) transpose needs an intermediate — pinned in T3 against the identity test. The equivalence fixture mechanism (T6) — committed signature vs single-driver diff; pick whichever the harness supports.
