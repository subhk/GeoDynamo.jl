# θ-Distributed Transform (Phase 1 of r×θ) — Design

Date: 2026-05-31
Status: approved (design)

## Motivation

GeoDynamo's MPI parallelization gives **memory capacity but no speed scaling**.
Measured (8-core node, fixed L=32/nr=32): step time 268→298→410 ms at np=1→2→4
(anti-scaling). Amdahl breakdown of `solver_step!`: the SH-transform-dominated
nonlinear phase is **97%** of a step and is **replicated** — each rank
Allreduce-gathers the full spectral matrix, runs a serial SHTnsKit transform, and
slices its tile. Only the 3% implicit-solve fraction distributes. Adding ranks
adds gather/contention with no transform-compute reduction → slower.

SHTnsKit ≥ 1.2.10 provides a **transpose-based θ-distributed** transform that does
*not* gather — each rank transforms its own θ-slab, one Alltoall per slab.
Standalone benchmark (`scripts/sht_scaling_benchmark.jl`): θ-decomposition beats
serial-replicate from lmax ≈ 128, ~1.7× at lmax 256 on 2 ranks, growing with lmax
and ranks. (φ-distribution is the documented footgun — Allgathers/replicates;
not used.)

This is **Phase 1** of the eventual **r×θ** decomposition. Phase 2 (distribute r +
add an r↔lm transpose for the radial solve) is designed separately; r is the free
second axis for the SHT (each radial level is an independent 2D transform).

## Scope

**In:** swap the scalar and vector transforms from gather-replicate to SHTnsKit's
θ-distributed path; change the physical field layout to θ-distributed / φ-local;
keep r local.

**Out (Phase 2):** distributing the radial dimension; the r↔lm transpose; any
change to the implicit radial solve.

## Architecture change — data layout

| | current | Phase 1 |
|---|---|---|
| process grid | 2D `(θ,φ)` | 1D `θ` (φ **local** per rank) |
| physical field | `(θ,φ)` distributed, r local | **θ distributed, φ local**, r local |
| spectral field | `(l,m)` distributed, r local | `(l,m)` distributed, r local (unchanged contract) |
| transform | per-level gather→serial→slice | per-level **`dist_synthesis`/`dist_analysis`** |
| radial solve | r-local per owned mode | **unchanged** |

Dropping φ-distribution caps usable ranks at ≈ nlat (Phase 2's r-axis restores 2D
scaling). PencilFFTs requires φ local for the FFT, so this matches the only
scalable SHTnsKit path.

## Transform rewire (the hot per-radial-level loop)

- `scalar_spectral_to_physical!` (physics/nonlinear.jl): replace
  `fill coeff stack → Allreduce-gather → serial synthesis! → slice` with, per
  radial level, `dist_synthesis(cfg, alm_pencil; prototype_θφ, real_output=true)`.
- `scalar_physical_to_spectral!`: replace gather+serial analysis with
  `dist_analysis(cfg, fθφ_pencil)`.
- Vector (fields/transforms.jl `shtnskit_vector_synthesis!`/`_analysis!`):
  `dist_synthesis_sphtor!` / `dist_analysis_sphtor!`.
- The manual `_SCALAR_GATHER_REDUCE_COUNT` Allreduce path is removed (the
  distributed transform owns the communication).

## Key components touched

- `parallel/pencils.jl` — add a θ-distributed/φ-local physical pencil; the
  process topology becomes 1D over θ. Spectral pencil contract (l,m distributed,
  r local) is preserved.
- `physics/nonlinear.jl` — scalar transforms.
- `fields/transforms.jl` — vector transforms.
- **Spectral adapter** — a mapping between GeoDynamo's `(l,m,r)` spectral storage
  and SHTnsKit's distributed spectral PencilArray (per radial level). This is the
  fiddly bit and the main source of risk; it gets its own well-tested unit.

## Verification

- **Correctness:** spec→phys→spec roundtrip at machine precision (serial and
  multi-rank); full suite green (baseline 2793 pass / 2 broken). The existing
  `mpi_parallel_invariants.jl` and `shtnskit_roundtrip.jl` gate the layout +
  transform.
- **Scaling:** the standalone `sht_scaling_benchmark.jl` already shows the θ-path
  scales; an in-solver scaling curve needs a real multi-node cluster and is
  **out of scope to validate in this environment** (laptop MPI is the wrong
  instrument — single node has no extra cores/bandwidth per rank). Correctness is
  the gate here; cluster scaling is validated on cluster hardware later.

## Risks / open items

- **(l,m,r) ↔ distributed-spectral adapter** — the highest-risk unit; needs
  isolated tests (round-trip a known field through the adapter both ways).
- **φ-local requirement** — any code assuming φ-distribution must be audited
  (BC application, diagnostics, IO that read `pencil.axes_local`).
- **Rank-count regression** — θ-only caps ranks at ≈ nlat vs nlat·nlon today;
  acceptable for Phase 1, restored by Phase 2 (r×θ).
- **Concurrent edits** — implement in an isolated git worktree to avoid clobber
  by the other active session.

## Done criteria

- Scalar + vector transforms run through `dist_*`; no manual gather remains on the
  transform path.
- Roundtrip + full suite green (single and ≥2 ranks).
- Physical pencil is θ-distributed / φ-local; r local; radial solve untouched.
