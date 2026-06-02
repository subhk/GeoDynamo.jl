# DistTransposePlan Transform (Phase 3) — Design

Date: 2026-06-02
Status: approved (design)
Builds on: Phase 1 (θ-dist, main `71d94b9`) + Phase 2 (r×θ, main `db8c349`/`4811efd`).

## Motivation

Phase 2 gave a working, correct 2D r×θ decomposition, but its SH transform uses a
**gather-to-dense** scheme: per radial level, scatter owned modes + `MPI.Allreduce!`
over the θ-subcommunicator to build a full dense `(l,m)` matrix, then call
`dist_synthesis`. That means `nr_local` separate transform calls per field, each with a
θ-Allreduce and a replicated dense matrix.

The Fortran reference **DD_2DCODE** (`GeoDynamo.jl/DD_2DCODE`) — which is also θ×r,
φ-local, r-distributed (NOT θ×φ; an earlier note was wrong) — does the transform the
**transpose-based** way: local φ-FFT → rotating all-to-all θ↔m exchange (no Allreduce) →
m-distributed Legendre (`tra_p2s`/`tra_s2p` in `modules/transform.fftw3.f90`).

SHTnsKit ≥1.2.10 ships exactly this as **`DistTransposePlan`** (verified present in
GeoDynamo's pinned version; spiked: batched roundtrip err 6.75e-14 at 2 ranks). It is:
- **batched over radial levels** (`extra_dims=(nlev,)`) — ONE call for all local levels,
- **transpose-based** (PencilFFTs Alltoall θ↔m) — no gather/replicate,
- **m-distributed** spectral (`Alm`: l-local / m-dist / nlev-batch) — no dense matrix.

Phase 3 replaces the gather-to-dense transform with `DistTransposePlan`, bringing
GeoDynamo's transform in line with DD_2DCODE's proven scalable method. The 2D r×θ grid
and the r-local banded radial solve are kept. Gate is **correctness-only** (same as
Phases 1–2; wall-clock scaling validated on a cluster later — but the **collective-count
reduction**, O(nr_local) → O(1) per transform, is demonstrable here).

## Decisions (from brainstorming)

1. **Replace, staged** (scalar → vector, then remove the Phase-2 gather path). One
   transform path at the end.
2. **Keep the banded radial solve numerics unchanged** (r-local); bridge via a transpose.
3. **Redefine `spec_solve`** so the Alm↔solve step is a clean single axis-pair swap.

## Plan object

One `DistTransposePlan` per config, built on the **θ-subcommunicator** (`cfg.pencils.θ_comm`):
```julia
DistTransposePlan(cfg.sht_config; comm = cfg.pencils.θ_comm, nlev = nr_local,
                  use_rfft = true, with_vector = true)
```
Cached (built once). Layouts:
- spatial `(nlon, nlat, nlev)`: φ-local / θ-dist / r-batch — matches the physical r×θ field.
- `Alm` `(lmax+1, mmax+1, nlev)`: **l-local / m-dist(θ-subcomm) / r-batch**.

`nr_local` = this rank's local radial levels; uneven `nr` ⇒ each rank builds its own plan
with its own `nlev`. Guard `r_ranks ≤ nr` (else `nlev = 0`).

## Spectral orientations

- **Phase 2 `spec_solve`** (to be replaced): l over θ_ranks, m over r_ranks, r-local.
- **Phase 3 `spec_solve`**: **m over θ_ranks, l over r_ranks, r-local.** Non-redundant
  (modes spread over both process axes), r-local (radial solve home, numerics unchanged).
  The mode→rank partition changes (l and m swap which process axis carries them); the
  radial solve still iterates owned modes with full r → bit-equivalent.
- **`Alm`** (DistTransposePlan): l-local, m-dist(θ-subcomm), r-dist(nlev).

The `Alm ↔ spec_solve` transpose is then an **l↔r swap over the r-subcommunicator**
(m stays θ-distributed): gather r to full + distribute l across r_ranks. Conceptually
like Phase 2's clean single-axis transpose, but on the r-axis.

## Per-step data flow (scalar; vector mirrors with the S,T pair + v_r)

```
physical(θ-dist, φ-local, r-dist)
  --DistTransposePlan dist_analysis! (θ-subcomm, batched nlev)-->  Alm(l-local, m-dist/θ, r-dist)
  --Alm↔solve transpose (r-subcomm: gather r, distribute l)-->     spec_solve(l-dist/r, m-dist/θ, r-local)
        radial implicit solve   (r-local, UNCHANGED numerics)
  --reverse transpose-->  Alm  --dist_synthesis! (batched)-->      physical
```
Vector: `DistTransposePlan` provides `dist_synthesis_sphtor!`/`dist_analysis_sphtor!`
(toroidal/poloidal pair); the v_r poloidal radial-component path is kept (scalar
`dist_synthesis!` of `l(l+1)/r·P`).

## Sequencing (~6 tasks)

- **T0** spike (GATES): build `DistTransposePlan` + `Alm → spec_solve → Alm` (the
  l↔r-over-r_subcomm transpose); exact identity roundtrip at 2×2. If unworkable → STOP.
- **T1** plumbing: plan built/cached; redefined `spec_solve`; `Alm↔spec_solve` transpose
  helpers. Gate: transpose identity (single + multi-rank).
- **T2** scalar transforms → `dist_synthesis!`/`dist_analysis!` (batched) + Alm↔solve.
  Gate: roundtrip + step-equivalence, single/2×1/1×2/2×2.
- **T3** vector transforms (numerics.jl solver + fields/transforms.jl) → `dist_*_sphtor!`
  + v_r. Gate: vector roundtrip + equivalence.
- **T4** remove the Phase-2 gather path (`spec_transform` pencil, per-level θ-Allreduce,
  the per-level loop). One transform path remains.
- **T5** full suite + `mpi_parallel_invariants` update (spec_solve mode-partition changed)
  + demonstrate the collective-count drop. Multi-rank gate.

## Verification (correctness-only — the gate)

1. `Alm↔spec_solve` transpose roundtrip = **identity** (multi-rank).
2. Transform roundtrip phys→spec→phys at machine precision (single + 2×1/1×2/2×2).
3. **Step equivalence:** Phase-3 `solver_step!` == Phase-2 result to **<1e-10** (single
   + 2D grids) — proves the transform swap is physics-neutral. *The key gate.*
4. Full suite green; `mpi_parallel_invariants` updated to the new mode-partition.
5. Collective-count reduction demonstrated (O(nr_local) θ-Allreduces → O(1) batched
   transpose per transform). Wall-clock validated on cluster later.

## Risks / open items

- **#1 (T0): the Alm↔spec_solve transpose.** `Alm` lives on the θ-subcommunicator (with
  an r-batch dim); `spec_solve` lives on the full 2D topology. Bridging them (the l↔r
  redistribution over the r-subcomm + the comm-structure mismatch) is the fiddly, risky
  part — more involved than Phase 2's transpose. T0 spike gates it.
- **spec_solve mode-partition change** ripples to `mpi_parallel_invariants` + any code
  assuming Phase-2's (l-over-θ, m-over-r) partition. Audit (T5).
- **Uneven nr** ⇒ `nlev` varies per r_rank; the Alm↔solve transpose must handle uneven
  r-slabs. Verify.
- **`r_ranks ≤ nr`** must hold (else `nlev=0`); guard + clear error.
- **GPU** unchanged — `DistTransposePlan` is CPU/MPI (PencilFFTs); GPU configs out of scope.

## Done criteria

- Single transpose-based transform path via `DistTransposePlan` (gather-to-dense removed).
- Transpose identity + transform roundtrip + step-equivalence + full suite all green
  (single + multi-rank); collective-count reduction shown.
- 2D r×θ grid + r-local banded radial solve unchanged.
