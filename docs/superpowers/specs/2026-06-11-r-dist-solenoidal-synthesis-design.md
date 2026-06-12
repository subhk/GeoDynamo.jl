# r-Distributed Solenoidal Vector Synthesis — Design

**Date:** 2026-06-11
**Branch:** `feat/r-dist-solenoidal-synthesis` (off `main` post-PR-#59)
**Status:** approved design, pre-implementation

## Goal

Remove the r-local restriction from the solenoidal vector synthesis so a full
MHD `solver_step!` runs on r-distributed process grids (`1x4`, `2x2`).
Today every such run dies at `src/solver/numerics.jl:916`:

```
solenoidal vector synthesis requires the radial axis fully local
(got 8 of 32 levels); r-distributed support is a Stage-2 follow-up
```

**Acceptance gate:** `test/run_mpi_r_theta_equivalence_mhd.sh` green at all four
grids (1x1, 4x1, 1x4, 2x2), physics-equivalent to the serial reference at
< 1e-10, plus the hydro `run_mpi_r_theta_equivalence.sh` and the full
single-rank suite.

## Why only synthesis is blocked

The r×θ topology distributes the **physical** pencil in r, but the spectral
**storage** pencil (`pencils.spec`, decomp `(2,1)`) keeps the full radial axis
on every rank (m over θ_comm, l over r_comm, r local). Consequences, verified
in code 2026-06-11:

- Analysis side is already r-dist safe: Q-based poloidal recovery
  `P = r²·Q/(l(l+1))` (`_poloidal_from_radial_q!`) is per-level and runs on
  storage.
- The three other r-local guards — `_poloidal_force_projection!`
  (`velocity/solver.jl:78`), poloidal W-split (`velocity/solver.jl:271`),
  `_induction_curl_potentials!` (`numerics.jl:1656`) — all iterate the spec
  storage pencil, whose r-range is always full in this topology. They are
  unreachable defensive guards; they stay.
- Only `vector_spectral_to_physical_disttranspose!` (`numerics.jl:887`)
  computes a radial derivative **after** bridging to the SHTnsKit Alm layout
  (l-full, m-local, **r-slab**), where ∂_r needs levels the rank doesn't own.

## Architecture change (one function)

Current synthesis flow:

```
storage P ─bridge→ Alm(P) ─ _fill_vr_alm!(from P) ─ _spheroidal_from_poloidal!
                            (mutate Alm: S=(∂_r P)/r)        ⟵ needs full r
storage T ─bridge→ Alm(T) ────────────────────────→ dist_synthesis_sphtor!
```

New flow — radial derivative moves to storage layout, where r is always full:

```
storage P ─ compute into storage scratch (all-local, banded D1 per (l,m) column):
              S  = (∂_r P)/r
              Vr = vr_factor(l, r)·P
storage S  ─bridge→ Alm(S)  ─┐
storage T  ─bridge→ Alm(T)  ─┴→ dist_synthesis_sphtor!  → v_θ, v_φ
storage Vr ─bridge→ Alm(Vr) ──→ dist_synthesis!          → v_r
```

"bridge" = the existing `spec_storage_to_solve!` → `from_spec_solve!` pair
(m-axis Allgatherv + r_comm l↔r transpose), unchanged.

The Alm-layout helpers `_spheroidal_from_poloidal!` and the P-sourced
`_fill_vr_alm!` call are replaced by storage-layout equivalents written in the
established `local_spectral_storage_slot` / `local_spectral_value` idiom (same
shape as `_poloidal_force_projection!`). Each (l,m) column's banded D1 matvec
is independent, so the 1x1 result is bit-exact vs the old path.

Unchanged paths:

- `raw_spheroidal = true` (tangential-basis primitive for tests): no derivative,
  no v_r from P — keep as is.
- `domain === nothing`: zero-filled v_r, raw coefficients — keep as is.
- The legacy raw-coefficient v_r branch (`!solenoidal && domain !== nothing`)
  switches to the same storage-computed Vr scratch (same values, same
  collectives) so the function has one bridging structure.

## Components

- **Modify** `src/solver/numerics.jl`:
  - Restructure `vector_spectral_to_physical_disttranspose!`; delete the
    r-local `error()` gate at :914-918.
  - New `_storage_spheroidal_from_poloidal!(s_re, s_im, p_re, p_im, cfg, domain)`
    and `_storage_vr_coeffs!(vr_re, vr_im, p_re, p_im, cfg, domain, vr_factor)`.
  - Remove `_spheroidal_from_poloidal!` / the P-sourced `_fill_vr_alm!` use if
    no caller remains (grep before deleting; `_fill_vr_alm!` may also serve the
    raw path).
- **Scratch:** two storage-layout (real, imag) array pairs sized
  `(l_slots, m_slots, nr)` cached on `config._buffers` (build-once under
  `_DISTTRANSPOSE_LOCK`, like the existing disttranspose scratch). No per-step
  allocation — `allocation_runtime_checks.jl` guards must stay green.
- **Callers untouched** (all go through the one core): solver path, MIE
  wrappers (`fields/transforms.jl`), diagnostics writer, Ball, magnetic
  synthesis.

## Communication cost

3 bridge passes per vector synthesis (S, T, Vr) vs current 2 (P, T); the
separate `dist_synthesis!` for v_r already existed. Net +1 batched collective
per synthesis. Bounded and acceptable; halo-exchange optimization (point-to-point
boundary levels in Alm layout) deliberately rejected as premature — revisit only
if profiles at large r_ranks show the extra collective matters.

## Testing (TDD)

1. **RED:** `test/r_dist_solenoidal_synthesis.jl` — at 1x1, the storage-layout
   S and Vr computations must equal the old Alm-layout results bit-exact on a
   deterministic spectral field; register in `runtests.jl`. (Initially the new
   helpers don't exist → test errors.)
2. **GREEN:** implement; single-rank suite green, specifically
   `solenoidal_transform_pair.jl`, `poloidal_solenoidality.jl` (hard ∇·u gate),
   MIE/vector roundtrips, allocation guards.
3. **MPI acceptance:** `run_mpi_r_theta_equivalence_mhd.sh` — 1x4 and 2x2 now
   pass (currently die at the gate); hydro `run_mpi_r_theta_equivalence.sh`
   all four grids.
4. Full `Pkg.test()` with output redirected to a file (never piped through
   `tail`); re-run before attributing the known IC-normalization flakes.

## Edge cases

- Rank owning a single radial level: derivative runs in storage (full r) — no
  slab-width constraint exists anywhere after this change; the plan builder
  already errors loudly on 0-level ranks.
- Uneven r splits (e.g. nr=32 over 3 ranks): bridge and transpose are
  PencilArrays-range-driven; no uniformity assumption introduced.
- Dealiased vs canonical grids: m-bridge behavior unchanged.
- Environment trap (found 2026-06-11): a stale local Manifest with
  SHTnsKit < 1.2.12 reproduces an unrelated-looking
  `m-distribution mismatch` failure on ALL multi-rank runs — check
  `Pkg.status("SHTnsKit")` before debugging multi-rank failures.

## Out of scope

- GPU vector-transform port (own gate, `gpu/vector_transform.jl:52`).
- EAB2 poloidal W-split port.
- Performance work (θ-scaling at high lmax, FFTW threading).
- Removing the three defensive spec-layout guards.
