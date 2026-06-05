# IC NetCDF I/O + Time-Aware Restart Selection — Design

Date: 2026-06-06

## Problem

Four review findings, three already fixed (docs `include_magnetic_field`,
`Simulation(; timestepper=:cnab2)` leniency, `FieldWriter` field filtering,
writer `geometry` mislabeling). This spec covers the remaining two, which the
user chose to fully implement rather than patch:

1. **IC NetCDF load/save are unimplemented stubs that report success.**
   `load_initial_conditions!` (`src/core/initial_conditions.jl`) warns "NetCDF
   loading not implemented", applies an analytic fallback, then prints "Initial
   conditions loaded successfully". A corrupt or wrong file silently becomes an
   analytic IC. `save_initial_conditions` warns and writes nothing but prints
   "Initial conditions saved". Both are exported and documented.

2. **`find_restart_files` ignores `target_time`.** It sorts candidates by file
   modification time (`src/io/restart.jl`), so `read_restart!(…, restart_time,
   …)` can load the newest file instead of the one nearest the requested time.

## Decisions (locked with user)

- **Implement** real NetCDF IC I/O (not error-out, not just honest messaging).
- **Hybrid format**: a new, self-contained per-field NetCDF format, decoupled
  from the restart/output format. Save gathers distributed coefficients to rank
  0; load reads on rank 0 and scatters to local pencil ranges. Works both
  serial (no MPI) and distributed.
- **One field per file** (matches the `(field, field_type, path)` API).
- **Strict validation on load**: mismatched `field_type` or spectral dimensions
  (`nlm`/`nr`) throw a clear error. No silent wrong IC.
- **Restart**: honor `target_time` by reading each candidate's stored `time`;
  fall back to mtime when `target_time` is non-finite or time metadata is
  unreadable.

## Component 1 — IC file format

A single field per NetCDF file, written and read serially by rank 0.

- **Dimensions**: `nlm` (global spectral mode count), `nr` (global radial
  points).
- **Variables**:
  - scalar (`:temperature`, `:composition`): `spectral_real[nlm, nr]`,
    `spectral_imag[nlm, nr]`.
  - vector / magnetic (`:velocity`, `:magnetic`): `toroidal_real[nlm, nr]`,
    `toroidal_imag[nlm, nr]`, `poloidal_real[nlm, nr]`, `poloidal_imag[nlm, nr]`.
  - `l_values[nlm]`, `m_values[nlm]` — for validation / human inspection.
- **Global attributes**: `field_type` (string), `lmax`, `mmax`, `nlm`, `nr`,
  `geodynamo_ic_version` (format version int).

This format is intentionally distinct from the restart/output format. `FileIC`
loads only files produced by `save_initial_conditions`. It round-trips with
itself.

## Component 2 — gather / scatter

The global coefficient matrix is `[nlm, nr]`. A field stores a local pencil
slice `[lm_local, 1, r_local]`. Pencil blocks are a disjoint exact cover of the
global index space (established invariant; see project memory).

- **Save (gather)**: each rank allocates a zero global `[nlm, nr]`, writes its
  local block at `(lm_range, r_range)`, then `MPI.Reduce(+, root=0)`. Disjoint
  blocks ⇒ the sum reconstructs the global matrix exactly. Rank 0 writes the
  file. With MPI uninitialized or a single rank, local == global and no
  collective runs.
- **Load (scatter)**: rank 0 reads the full `[nlm, nr]`, `MPI.Bcast` to all
  ranks, then each rank copies its `(lm_range, r_range)` block into the field
  arrays. Serial path reads and copies directly.

Local ranges come from `get_local_range(pencil, 1)` (lm) and
`get_local_range(pencil, 3)` (r), the same indexing `randomize_*` uses. After a
load, apply `_maybe_enforce_ball_scalar!` / `_maybe_enforce_ball_vector!` so a
ball domain stays regular at r=0, matching the randomize path.

## Component 3 — load / save dispatch + validation

- `save_initial_conditions(field, field_type, path)`:
  - scalar ⇒ gather `field.spectral`; vector/magnetic ⇒ gather `field.toroidal`
    and `field.poloidal`.
  - rank 0 writes the NetCDF via a serial `NCDataset`; `MPI.Barrier` after.
  - remove the false "Initial conditions saved" print — the write is now real.
  - return `path`.
- `load_initial_conditions!(field, field_type, path)`:
  - keep `isfile(path)` ⇒ `ArgumentError`.
  - rank 0 reads the header (`field_type`, `nlm`, `nr`) and `MPI.Bcast`es it.
  - **all ranks validate** against their own field config and throw the *same*
    error on mismatch, so a mismatch never causes a rank-0-only throw followed
    by a collective deadlock. Strict rules: file `field_type` must equal the
    requested `field_type`; file `nlm`/`nr` must equal the target field's global
    dimensions.
  - on success, Bcast the data arrays and scatter into the local field slice;
    apply ball regularity; remove the false "loaded successfully" print.
  - the per-field stubs (`load_temperature_initial_conditions!`,
    `load_magnetic_initial_conditions!`, `load_velocity_initial_conditions!`,
    `load_composition_initial_conditions!`) are replaced by real readers, folded
    into one generic helper parameterized by field_type / component set.
- **Module imports**: add `NCDatasets` and `MPI` (and `get_comm` from the parent
  module) to `InitialConditions`.

## Component 4 — restart `target_time`

`find_restart_files(restart_dir, target_time)`:

- collect candidate files (`endswith(".nc") && contains("restart")`).
- if `isfinite(target_time)`: attempt a serial open of each candidate and read
  its `time` variable. If every candidate yields a time, sort by
  `abs(time - target_time)` ascending and return. If any candidate is
  unreadable as NetCDF or lacks a `time` variable, fall back to mtime
  newest-first.
- if `!isfinite(target_time)` (e.g. `Inf`): mtime newest-first.

The existing unit test writes dummy files whose contents are `"x"` and calls
with `target_time = 0.0`; opening them as NetCDF throws, so the function falls
back to mtime newest-first and the test stays green. `read_restart!` continues
to take `restart_files[1]`, now the nearest-time file.

## Error handling

- Missing file ⇒ `ArgumentError` (unchanged).
- Dimension / field_type mismatch on load ⇒ clear `error`/`ArgumentError`
  naming expected vs file values; raised on all ranks.
- Non-finite values after load ⇒ reuse the existing finiteness check pattern
  from `randomize_*` (error).
- Corrupt / non-NetCDF restart candidate ⇒ skipped for time selection, falls
  back to mtime ordering (no crash).

## Testing

New `test/ic_netcdf_io.jl`:
- save → zero → load round-trip for scalar, vector, and magnetic fields
  (serial); compare arrays bitwise.
- strict mismatch: `@test_throws` when loading a temperature file into a
  composition field, and when `nlm`/`nr` differ.

`test/io_restart_roundtrip.jl`:
- add a case with real small `.nc` files carrying differing `time` values and a
  `target_time`, asserting the nearest file is returned. Keep the existing
  dummy-file newest-first test.

Update existing placeholder tests to the new contract:
- `test/initial_conditions.jl:89,94` and `test/tail_coverage_extended.jl:431,
  437` — drop the `@test_logs (:warn, "NetCDF … not implemented")` expectations;
  replace with round-trip and strict-error assertions.

## Files touched

- `src/core/initial_conditions.jl` — implement load/save, gather/scatter,
  validation, imports.
- `src/io/restart.jl` — time-aware `find_restart_files`.
- `test/ic_netcdf_io.jl` (new), `test/io_restart_roundtrip.jl`,
  `test/initial_conditions.jl`, `test/tail_coverage_extended.jl`.

## Out of scope

- Restart/output-format compatibility for `FileIC` (decoupled per the chosen
  hybrid format).
- Parallel MPI-IO for IC files (gather/scatter to a serial rank-0 file instead).
- Resolution remapping on load (strict dimension match only).
