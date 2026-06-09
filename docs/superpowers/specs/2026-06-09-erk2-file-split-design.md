# erk2.jl File-Split Design

**Status:** approved (design phase)
**Date:** 2026-06-09

## Goal

Decompose `src/timestep/erk2.jl` (2952 LOC, 80 functions — the largest file in the
codebase) into a thin include spine plus five focused files, each with one coherent
responsibility. This is a **pure code move**: every function body is relocated
verbatim, no logic changes, behavior-preserving by construction.

## Motivation

`erk2.jl` is the timestepping monolith: BC constructors, cache builders, cache
accessors, influence matrices, phi functions, field buffers, stage execution, bundle
persistence, and the main `integrate_solver_erk2_step!` all live in one file. At 2952
lines it is the single biggest cognitive-load source in the repo and the hardest file
to edit reliably. Splitting it along its existing thematic clusters makes each unit
understandable and editable in isolation, with no change to behavior or public API.

This was selected after verifying that the two higher-ranked simplification candidates
were already resolved: the temperature/composition scalar-driver duplication (fixed in
PR #44) and the two vector-transform implementations (already consolidated by the
Phase-3 `DistTransposePlan` migration into the shared
`vector_spectral_to_physical_disttranspose!` core).

## Constraints / what makes this safe

- **No struct definitions move.** `erk2.jl` defines zero structs (`ERK2StageCache`,
  `SolverERK2FieldBuffers`, etc. all live in `timestep/state.jl`, included earlier via
  `solver.jl`). `erk2.jl` holds only function defs plus 7 `const` aliases.
- **Flat include semantics.** Julia includes splice file contents into the enclosing
  module; a function's file location is invisible to callers. No call site outside
  `erk2.jl` changes.
- **Single ordering constraint:** the 7 const aliases (current lines 79–84 and 1967)
  must be parsed before any function whose signature annotates with an alias name. They
  all go into `common.jl`, included first. They reference `Solver*` symbols already
  defined in `state.jl`, so they resolve.
- **Public/internal pairs stay together.** Many functions come as a `solver_foo`
  implementation plus a `GeoDynamo.foo` public twin that delegates to it. Each pair is
  kept in the same file.

## Architecture

`src/timestep/erk2.jl` becomes a ~30-line spine:

```julia
include("erk2/common.jl")     # const aliases first (parse-order constraint)
include("erk2/boundary.jl")
include("erk2/cache.jl")
include("erk2/influence.jl")
include("erk2/integrate.jl")
```

The single include site (`src/solver.jl:20`, `include("timestep/erk2.jl")`) is
unchanged — zero blast radius outside the file.

### File mapping

Every function is relocated verbatim. Current line ranges shown for traceability.

| File | ~LOC | Functions (current lines) |
|------|------|---------------------------|
| `erk2/common.jl` | ~150 | 7 const aliases (79–84, 1967); `ERK2Cache{T}` compat ctor (94); `compat_normalize_old_erk2_cache_entry` (144); `with_boundary_mode_values` (60); phi functions `compute_phi1_function` (1063), `compute_phi2_function` (1072), `report_phi2_conditioning` (1088); diagnostics toggles `set_erk2_diagnostics_interval!` (435), `enable_erk2_diagnostics!` (447), `disable_erk2_diagnostics!` (460) — they gate the stage-residual logging, cross-cutting config |
| `erk2/boundary.jl` | ~490 | `solver_enforce_erk2_bc!`/`enforce_erk2_bc!` (165, 212); all `create_*_bc` constructors + public twins — dirichlet/neumann/stress-free-tor/noslip-pol/stress-free-pol/insulating-inner/outer (235–426); `build_solver_erk2_scalar_bc` (488); `build_solver_erk2_velocity_tor_bc`/`_pol_bc` (1257, 1300); `build_solver_erk2_magnetic_tor_bc`/`_pol_bc` (1325, 1337) |
| `erk2/cache.jl` | ~900 | cache builders: `create_solver_erk2_scalar_cache` (514), `create_solver_erk2_cache` (603), `create_solver_erk2_magnetic_toroidal_cache` (699), `create_solver_erk2_magnetic_poloidal_cache` (787); public `create_erk2_cache*` twins (892–1036); `_get_or_build_erk2_cache` (1100), `_get_or_build_erk2_scalar_cache` (1147); cache accessors `get_solver_erk2_temperature_cache!` (1497), `_composition_cache!` (1531), `get_solver_erk2_cache!` ×3 (1567, 1601, 1636), `_magnetic_toroidal_cache!` (1669), `_magnetic_poloidal_cache!` (1714); bundle `save_erk2_cache_bundle` (2433), `load_erk2_cache_bundle` (2459), `install_erk2_cache_bundle!` ×2 (2475, 2492), `load_erk2_cache_bundle!` (2509) |
| `erk2/influence.jl` | ~310 | `_get_or_build_erk2_influence_entry` (1198); `create_solver_velocity_poloidal_influence_matrices` (1356); `get_solver_erk2_influence_matrices!` (1762); `apply_solver_influence_matrix_correction!` (1795); `apply_solver_velocity_poloidal_influence_correction!` (1821); public twins `create_velocity_poloidal_influence_matrices` (1868), `apply_influence_matrix_correction!` (1893), `apply_velocity_poloidal_influence_correction!` (1912) |
| `erk2/integrate.jl` | ~640 | field buffers: `SolverERK2FieldBuffers` ctor (1930), `erk2_field_buffers_match` (1969), `get_solver_erk2_field_buffers!` (1988); stage exec: `prepare_solver_erk2_field!` (2011) + `erk2_prepare_field!` (2143), `apply_solver_erk2_stage!` (2168) + twin (2182), `store_solver_erk2_stage_nonlinear!` (2194) + twin (2208), `finalize_solver_erk2_field!` (2223) + twin (2327); residual `solver_erk2_stage_residual_stats` (2350) + twin (2390), `maybe_log_solver_erk2_stage_residual!` (2400) + twin (2420); `restore_solver_erk2_nonlinear_terms!` (2529); `_get_or_build_erk2_boundary_spec!` (2566); `integrate_solver_erk2_step!` (2590) |

All 80 functions are accounted for exactly once.

## Verification

Pure relocation → no new behavior test needed; the existing suite already exercises
ERK2 heavily (`imex_implicit_steps.jl`, `integration_simulation.jl`, the erk2 cache
tests, the allocation guards).

1. **Load check:** `using GeoDynamo; println("LOAD_OK")` — catches any alias/ordering
   miss or dropped symbol immediately.
2. **No function lost or duplicated:** the set of 80 `^function ` signatures across
   `erk2/*.jl` matches the pre-split set from `erk2.jl` exactly (one occurrence each).
3. **Pure move:** `git` confirms lines deleted from `erk2.jl` equal lines added across
   `erk2/*.jl` (allowing only the spine's new `include` lines + per-file section
   banners). No function body content changes.
4. **Full suite green:** `Pkg.test()` passes (`Testing GeoDynamo tests passed`); the
   18-broken baseline (GPU-skip gates) is unchanged.

## Risk

Low. The only realistic failure is a const-alias placed after a use, or a function
dropped/double-pasted during the move — all three are caught by the load check and the
function-count check before the suite even runs.

## Out of scope

- No logic changes, no signature changes, no dead-code removal (even if spotted —
  separate change).
- No touching the GPU ERK2 path, `timestep/state.jl` struct defs, or `timestep/driver.jl`.
- Not splitting `cache.jl` further (it stays ~900 LOC — coherent single responsibility:
  ERK2 cache lifecycle). Revisit only if it keeps growing.
