# r×θ Decomposition (Phase 2) — Design

Date: 2026-06-01
Status: approved (design)
Builds on: `2026-05-31-theta-distributed-transform-design.md` (Phase 1, merged to main `71d94b9`)

## Motivation

Phase 1 made the SH transform scale on the **θ** axis (1D-θ process grid,
`proc_dims=(nprocs,1)`); the radial dimension stays local on every rank. That caps
usable ranks at ≈ nlat and leaves the radial axis unexploited. Phase 2 adds **r** as
a second process axis (a 2D `r×θ` grid), so the transform scales on both axes and the
banded radial solve distributes across modes. This is the decomposition the Phase-1
design named as its sequel.

Validation here is **correctness-only** (same as Phase 1): laptop single-node MPI
cannot measure strong scaling. Scaling is validated later on cluster hardware. The
gate is correctness (roundtrips, transpose identity, radial-solve equivalence vs
Phase 1, full suite), single + multi-rank.

## Decisions (from brainstorming)

1. **Validation:** correctness-only here; scaling deferred to cluster.
2. **Architecture:** transpose-based full r×θ — two spectral orientations connected by
   an r↔lm transpose around the radial solve. Nothing globally replicated; both the
   transform and the radial solve scale. (The alternative "replicate the radial solve
   via Allgather" was rejected — kept only as the fallback if Task 0 fails.)
3. **Process grid:** explicit only — `GEODYNAMO_PROC_GRID="θxr"` (e.g. `"4x2"`),
   REQUIRED at np>1; no auto-optimizer. Predictable for controlled scaling experiments.

## Process grid

2D grid `(θ_ranks × r_ranks)` parsed from `GEODYNAMO_PROC_GRID`. Constraints:
`θ_ranks·r_ranks == nprocs`, `θ_ranks ≤ nlat`, `r_ranks ≤ nr`. Error clearly at np>1
if unset/invalid. Split `COMM_WORLD` into:
- **θ-subcommunicator** — the ranks sharing one r-slab (the SH transform runs here),
- **r-subcommunicator** — the ranks sharing one θ-slab (the r↔lm transpose runs here).

## Data layouts

| | Phase 1 | Phase 2 |
|---|---|---|
| physical | θ-dist / φ-local / **r-local** | θ-dist / φ-local / **r-dist** (2D) |
| spectral *solve* | modes-dist / r-local | modes-dist / r-local — **unchanged** (radial-solve home) |
| spectral *transform* | — (r-local) | **r-dist (r_ranks) + one mode-axis-dist (θ_ranks)** (new pencil) |
| transform comm | `COMM_WORLD` | **θ-subcommunicator** |

KEY FOUND FACT: the radial implicit solve already iterates locally-owned modes with r
local (`local_spectral_mode_indices` + `@solver_local_spectral_modes`, numerics.jl).
So the **solve orientation = the existing Phase-1 spectral layout**; the radial solve
code itself does not change. Phase 2 brackets it with transposes.

## Per-step flow

```
physical (r-dist, θ-dist)
  --dist_analysis on θ-subcomm, per LOCAL r-level-->  spec_transform (r-dist, lm-dist/θ)
  --r↔lm TRANSPOSE (PencilArrays Alltoall, r-subcomm)-->  spec_solve (lm-dist, r-local)
      radial implicit solve            (r-local, UNCHANGED)
  --r↔lm TRANSPOSE-->  spec_transform (r-dist, lm-dist/θ)
  --dist_synthesis on θ-subcomm, per LOCAL r-level-->  physical (r-dist, θ-dist)
```

Around each per-level `dist_synthesis`/`dist_analysis`: a small O(nlm) gather/slice of
the θ-distributed mode-axis over the **θ-subcommunicator** to assemble/scatter the full
dense (l,m) the dist call needs (consistent with Phase 1's existing coeff gather). Two
Alltoalls per step (the r↔lm transpose, there and back); nothing globally replicated.

## Transpose mechanism

Both spectral orientations are PencilArrays pencils over the 2D `(l,m,r)` grid; the
r↔lm transpose reuses the existing transpose machinery (`create_transpose_plans` /
`transpose!`, already used for θ↔φ↔r):
- `spec_solve`: (l,m) distributed, **r local** — Phase-1 layout on the 2D topo.
- `spec_transform`: **r distributed** (r_ranks) + one mode-axis distributed (θ_ranks),
  the other mode-axis local.
- Exact decomp tuples and any intermediate orientation are pinned in implementation
  (Task 3); the transpose-roundtrip-identity test gates correctness.

## Unwinding Phase-1 hardcodes

The Phase-1 forward-risks, addressed here:
- `create_pencil_decomposition_shtnskit` (live path) + `create_pencil_topology`:
  replace `proc_dims=(nprocs,1)` with the parsed `(θ_ranks, r_ranks)`; reconcile both
  paths (add `theta_phys` to the `create_pencil_topology` NamedTuple too).
- `theta_phys`: rebuild on the **θ-subcommunicator** (replaces the Phase-1
  separate-1D-full-`COMM_WORLD` topology). Its per-r-level θ-split must match the
  physical field's θ-slab on the θ-subcomm.
- `copyto!(view(phys,:,:,r_local))` and the v_r `r_idx_global = r_local + first(r_range) - 1`
  already iterate the local radial range and offset globally — they survive
  r-distribution unchanged.

## Scope

All four fields (temperature, composition, velocity, magnetic) and both timestep paths
(CNAB2, ERK2), since the r↔lm transpose brackets the shared radial solve. The radial
solve numerics are untouched (only their data orientation is reached via transpose).

## Verification (correctness-only — the gate)

1. **Transpose roundtrip = identity:** r↔lm there-and-back preserves data exactly
   (multi-rank).
2. **r-distributed transform roundtrip:** physical→spectral→physical at machine
   precision (single + multi-rank).
3. **Radial-solve equivalence (key gate):** a full `solver_step!` on the 2D r×θ grid
   produces an identical result to Phase-1 (1D-θ) for the same problem (<1e-10) —
   proves the transpose + redistribution is physics-neutral.
4. **Full suite green** (single + multi-rank); `mpi_parallel_invariants` updated to the
   2D r×θ contract.

## Risks / open items

- **#1 (Task-0 spike, gates everything): θ-subcommunicator `dist_*`.** SHTnsKit's
  `dist_synthesis`/`dist_analysis`/`dist_*_sphtor` must run on the θ-subcommunicator,
  not `COMM_WORLD` (in Phase 1 all ranks were the θ-group). The `theta_phys` prototype
  PencilArray would carry the θ-subcomm. If SHTnsKit hardcodes `COMM_WORLD` or won't
  honor a sub-comm prototype, the transpose architecture is blocked → fall back to the
  "replicate radial solve" variant (surface to the user before proceeding).
- **r↔lm transpose decomp:** the two-axis change (r local↔dist, one mode-axis
  dist↔local) may need an intermediate pencil; pin in Task 3 with the identity test.
- **Concurrent edits:** implement in an isolated git worktree off main `71d94b9`.
- **GPU:** unchanged from Phase 1 — `dist_*` are CPU/MPI-only; GPU configs silently use
  the CPU path. Out of scope (tracked follow-up).

## Done criteria

- 2D `r×θ` grid from `GEODYNAMO_PROC_GRID`; θ/r subcommunicators.
- Physical r-distributed; transform on the θ-subcomm; r↔lm transpose brackets the
  (unchanged) radial solve; both spectral orientations + transpose tested.
- Transpose identity + transform roundtrip + radial-solve equivalence + full suite all
  green (single + multi-rank).

## Sequencing (~8 tasks; may split into plans 2a/2b)

- **T0** spike: θ-subcomm `dist_*` feasibility (#1 risk; gates everything).
- **T1** `GEODYNAMO_PROC_GRID` → 2D topology + θ/r subcomms; unwind `(nprocs,1)`.
- **T2** r-distributed physical pencil + θ-subcomm `theta_phys`.
- **T3** `spec_transform` pencil + r↔lm transpose + transpose-roundtrip-identity test.
- **T4** scalar transforms → r-dist + θ-subcomm + transpose to solve orientation.
- **T5** vector transforms (numerics.jl + fields/transforms.jl) likewise.
- **T6** wire transposes into the step around the radial solve; radial-solve-equivalence gate.
- **T7** full suite + 2D `mpi_parallel_invariants` update + multi-rank gate.
