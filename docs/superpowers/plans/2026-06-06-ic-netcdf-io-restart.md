# IC NetCDF I/O + Time-Aware Restart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement real NetCDF initial-condition load/save (self-contained per-field format, gather/scatter, strict validation) and make `find_restart_files` honor `target_time`.

**Architecture:** A new per-field NetCDF format written/read serially by rank 0. Distributed spectral coefficients are gathered to a global `[nlm, nr]` matrix via a sum-reduction of disjoint pencil blocks (save) and broadcast + sliced back to local pencil ranges (load). Load validates `field_type`/`nlm`/`nr` strictly on every rank to stay deadlock-free. `find_restart_files` reads each candidate's stored `time` and selects the nearest, falling back to mtime.

**Tech Stack:** Julia, NCDatasets, MPI.jl, PencilArrays, the existing `SHTnsKitConfig` / `RadialDomain` field stack.

**Spec:** `docs/superpowers/specs/2026-06-06-ic-netcdf-io-restart-design.md`

---

## Reference facts (verified in codebase)

- Scalar field (temperature/composition): `field.spectral` with `.data_real`, `.data_imag` (3-D PencilArrays `[lm_local, 1, r_local]`), `.pencil`, `.config`.
- Vector/magnetic field: `field.toroidal`, `field.poloidal` (each like a scalar `.spectral`).
- Radial domain accessor: `field.domain` for temperature/composition/velocity; `field.outer_domain` for magnetic.
- `config` (`SHTnsKitConfig`) fields: `.lmax`, `.mmax`, `.nlm`, `.l_values::Vector{Int}`, `.m_values::Vector{Int}`.
- `get_local_range(pencil, dim) = range_local(pencil, dim)` (already imported into `InitialConditions`). `dim 1` = lm, `dim 3` = r.
- Global dims: `PencilArrays.size_global(pencil)` returns `(nlm, _, nr)`.
- `MPI`, `NCDatasets`, `PencilArrays` are all `Project.toml` deps. `get_comm()` lives in the parent module.
- Field constructors used in tests: `create_shtns_temperature_field(T, cfg, domain)`, `create_shtns_composition_field(T, cfg, domain)`, `create_shtns_velocity_fields(T, cfg, domain)`, `create_shtns_magnetic_fields(T, cfg, outer_domain, inner_domain)`. Build cfg with `create_shtnskit_config(; lmax, mmax, nlat, nlon, nr)` and a domain with `create_radial_domain(nr)`.
- `_maybe_enforce_ball_scalar!(field, domain)` / `_maybe_enforce_ball_vector!(field, domain)` already exist in the module and no-op when `domain === nothing` or the domain is not a ball.

## File structure

- Modify `src/core/initial_conditions.jl`: add imports; add helpers (`_ic_components`, `_ic_type_code`, `_ic_field_domain`, `_ic_gather_global`, `_ic_scatter_local!`); rewrite `save_initial_conditions` and `load_initial_conditions!`; delete the four per-field load stubs.
- Modify `src/io/restart.jl`: rewrite `find_restart_files` to be time-aware.
- Create `test/ic_netcdf_io.jl`: gather/scatter unit test, save→load round-trip, strict-mismatch errors.
- Modify `test/initial_conditions.jl`: replace the "Placeholder load/save behavior" testset with the new round-trip/strict contract.
- Modify `test/tail_coverage_extended.jl`: replace the placeholder load/save assertions.
- Modify `test/io_restart_roundtrip.jl`: add a time-aware selection test.
- Modify `test/runtests.jl` (or the test include list): include `test/ic_netcdf_io.jl` if not auto-included.

---

## Task 1: Module imports and small helpers

**Files:**
- Modify: `src/core/initial_conditions.jl` (import block near line 13-22; helpers near line 36)

- [ ] **Step 1: Add imports**

In `src/core/initial_conditions.jl`, the current import block is:

```julia
using LinearAlgebra
using Random
using SHTnsKit

# Import functions from parent module (GeoDynamo)
# These will be available when the module is included in GeoDynamo.jl
import ..get_local_range
import ..local_spectral_storage_slot
import ..set_local_spectral_value!
import ..local_spectral_value
```

Replace it with:

```julia
using LinearAlgebra
using Random
using SHTnsKit
using NCDatasets
using MPI
import PencilArrays

# Import functions from parent module (GeoDynamo)
# These will be available when the module is included in GeoDynamo.jl
import ..get_local_range
import ..local_spectral_storage_slot
import ..set_local_spectral_value!
import ..local_spectral_value
import ..get_comm
```

- [ ] **Step 2: Add dispatch/type/domain helpers**

Immediately after the `_maybe_enforce_ball_vector!` function (around line 48, before `randomize_scalar_field!`), add:

```julia
# IC NetCDF format version (bumped if the on-disk layout changes).
const GEODYNAMO_IC_VERSION = 1

# Spectral components carried by each field type.
function _ic_components(field_type::Symbol)
    if field_type in (:temperature, :composition)
        return (:scalar,)
    elseif field_type in (:velocity, :magnetic)
        return (:toroidal, :poloidal)
    else
        throw(ArgumentError("Unknown field type: $field_type"))
    end
end

# Stable integer code for a field type, used for deadlock-safe header broadcast.
function _ic_type_code(field_type::Symbol)
    field_type === :temperature && return 1
    field_type === :composition && return 2
    field_type === :velocity && return 3
    field_type === :magnetic && return 4
    return 0
end

# The spectral field objects to read/write for a given field.
function _ic_spectrals(field, field_type::Symbol)
    _ic_components(field_type) === (:scalar,) ? (field.spectral,) :
        (field.toroidal, field.poloidal)
end

# Radial domain accessor (magnetic stores it as outer_domain).
_ic_field_domain(field, field_type::Symbol) =
    field_type === :magnetic ? field.outer_domain : field.domain
```

- [ ] **Step 3: Verify the package still loads**

Run:

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using GeoDynamo; println("ok")'
```

Expected: prints `ok` (precompiles cleanly; new imports resolve).

- [ ] **Step 4: Commit**

```bash
git add src/core/initial_conditions.jl
git commit -m "feat(ic): imports and helpers for NetCDF initial-condition IO"
```

---

## Task 2: gather/scatter helpers

**Files:**
- Modify: `src/core/initial_conditions.jl` (add helpers after the helpers from Task 1)
- Test: `test/ic_netcdf_io.jl` (new)

- [ ] **Step 1: Write the failing test**

Create `test/ic_netcdf_io.jl`:

```julia
using Test
using MPI

@testset "IC NetCDF IO" begin
    if !MPI.Initialized() && !MPI.Finalized()
        MPI.Init()
    end

    lmax = 4
    mmax = 4
    nlat = max(lmax + 2, 10)
    nlon = max(2lmax + 1, 16)
    nr = 6

    cfg = GeoDynamo.create_shtnskit_config(
        lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr)
    shell = GeoDynamo.create_radial_domain(nr)

    @testset "gather/scatter round-trips a scalar field (serial)" begin
        temp = GeoDynamo.create_shtns_temperature_field(Float64, cfg, shell)
        GeoDynamo.generate_random_initial_conditions!(
            temp, :temperature, amplitude = 0.3, modes_range = 1:3, seed = 11)

        ref_real = copy(parent(temp.spectral.data_real))
        ref_imag = copy(parent(temp.spectral.data_imag))

        gr, gi = GeoDynamo.InitialConditions._ic_gather_global(temp.spectral)
        fill!(parent(temp.spectral.data_real), 0.0)
        fill!(parent(temp.spectral.data_imag), 0.0)
        GeoDynamo.InitialConditions._ic_scatter_local!(temp.spectral, gr, gi)

        @test parent(temp.spectral.data_real) == ref_real
        @test parent(temp.spectral.data_imag) == ref_imag
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using GeoDynamo; include("test/ic_netcdf_io.jl")'
```

Expected: FAIL — `UndefVarError: _ic_gather_global not defined` (or `_ic_scatter_local!`).

- [ ] **Step 3: Implement gather/scatter**

In `src/core/initial_conditions.jl`, after the helpers from Task 1, add:

```julia
# Assemble the global [nlm, nr] coefficient matrix for one spectral component
# from its distributed pencil slices onto every rank. Pencil blocks are a
# disjoint exact cover, so a sum-reduction of zero-padded local blocks is exact.
function _ic_gather_global(spectral)
    pencil = spectral.pencil
    gdims = PencilArrays.size_global(pencil)
    nlm = gdims[1]
    nr = gdims[3]
    lm_range = get_local_range(pencil, 1)
    r_range = get_local_range(pencil, 3)
    real_local = parent(spectral.data_real)
    imag_local = parent(spectral.data_imag)
    gr = zeros(Float64, nlm, nr)
    gi = zeros(Float64, nlm, nr)
    for (li, g_lm) in enumerate(lm_range), (ri, g_r) in enumerate(r_range)
        (li <= size(real_local, 1) && ri <= size(real_local, 3)) || continue
        gr[g_lm, g_r] = Float64(real_local[li, 1, ri])
        gi[g_lm, g_r] = Float64(imag_local[li, 1, ri])
    end
    if MPI.Initialized()
        comm = get_comm()
        if MPI.Comm_size(comm) > 1
            MPI.Allreduce!(gr, +, comm)
            MPI.Allreduce!(gi, +, comm)
        end
    end
    return gr, gi
end

# Copy the local pencil block of a global [nlm, nr] matrix into a spectral
# component, zeroing the rest first.
function _ic_scatter_local!(spectral, gr::AbstractMatrix, gi::AbstractMatrix)
    pencil = spectral.pencil
    lm_range = get_local_range(pencil, 1)
    r_range = get_local_range(pencil, 3)
    real_local = parent(spectral.data_real)
    imag_local = parent(spectral.data_imag)
    fill!(real_local, zero(eltype(real_local)))
    fill!(imag_local, zero(eltype(imag_local)))
    for (li, g_lm) in enumerate(lm_range), (ri, g_r) in enumerate(r_range)
        (li <= size(real_local, 1) && ri <= size(real_local, 3)) || continue
        real_local[li, 1, ri] = convert(eltype(real_local), gr[g_lm, g_r])
        imag_local[li, 1, ri] = convert(eltype(imag_local), gi[g_lm, g_r])
    end
    return spectral
end
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using GeoDynamo; include("test/ic_netcdf_io.jl")'
```

Expected: PASS — gather/scatter round-trips bitwise.

- [ ] **Step 5: Commit**

```bash
git add src/core/initial_conditions.jl test/ic_netcdf_io.jl
git commit -m "feat(ic): gather/scatter helpers for distributed spectral coefficients"
```

---

## Task 3: Implement save_initial_conditions and load_initial_conditions!

**Files:**
- Modify: `src/core/initial_conditions.jl` (rewrite `save_initial_conditions` ~line 883; rewrite `load_initial_conditions!` ~line 189; delete the four `load_*_initial_conditions!` stubs ~line 230-277)
- Test: `test/ic_netcdf_io.jl`

- [ ] **Step 1: Write the failing tests (round-trip + strict mismatch)**

Append to the `@testset "IC NetCDF IO"` block in `test/ic_netcdf_io.jl`, before its closing `end`:

```julia
    @testset "save -> load round-trips a scalar field (serial)" begin
        temp = GeoDynamo.create_shtns_temperature_field(Float64, cfg, shell)
        GeoDynamo.generate_random_initial_conditions!(
            temp, :temperature, amplitude = 0.25, modes_range = 1:3, seed = 5)
        ref_real = copy(parent(temp.spectral.data_real))
        ref_imag = copy(parent(temp.spectral.data_imag))

        path = joinpath(mktempdir(), "temp_ic.nc")
        @test GeoDynamo.save_initial_conditions(temp, :temperature, path) == path
        @test isfile(path)

        fill!(parent(temp.spectral.data_real), 0.0)
        fill!(parent(temp.spectral.data_imag), 0.0)
        GeoDynamo.load_initial_conditions!(temp, :temperature, path)
        @test parent(temp.spectral.data_real) == ref_real
        @test parent(temp.spectral.data_imag) == ref_imag
    end

    @testset "save -> load round-trips a velocity field (serial)" begin
        vel = GeoDynamo.create_shtns_velocity_fields(Float64, cfg, shell)
        GeoDynamo.randomize_vector_field!(vel, amplitude = 0.2, lmax = 3)
        ref_tr = copy(parent(vel.toroidal.data_real))
        ref_pr = copy(parent(vel.poloidal.data_real))

        path = joinpath(mktempdir(), "vel_ic.nc")
        GeoDynamo.save_initial_conditions(vel, :velocity, path)
        fill!(parent(vel.toroidal.data_real), 0.0)
        fill!(parent(vel.poloidal.data_real), 0.0)
        fill!(parent(vel.toroidal.data_imag), 0.0)
        fill!(parent(vel.poloidal.data_imag), 0.0)
        GeoDynamo.load_initial_conditions!(vel, :velocity, path)
        @test parent(vel.toroidal.data_real) == ref_tr
        @test parent(vel.poloidal.data_real) == ref_pr
    end

    @testset "load is strict about field_type and dimensions" begin
        temp = GeoDynamo.create_shtns_temperature_field(Float64, cfg, shell)
        GeoDynamo.generate_random_initial_conditions!(
            temp, :temperature, amplitude = 0.1, modes_range = 1:2, seed = 3)
        path = joinpath(mktempdir(), "temp_ic2.nc")
        GeoDynamo.save_initial_conditions(temp, :temperature, path)

        # Wrong field_type: load a temperature file into a composition field.
        comp = GeoDynamo.create_shtns_composition_field(Float64, cfg, shell)
        @test_throws Exception GeoDynamo.load_initial_conditions!(comp, :composition, path)

        # Wrong dimensions: a field built at a different resolution.
        cfg2 = GeoDynamo.create_shtnskit_config(
            lmax = 6, mmax = 6, nlat = max(8, 10), nlon = 16, nr = 8)
        shell2 = GeoDynamo.create_radial_domain(8)
        temp2 = GeoDynamo.create_shtns_temperature_field(Float64, cfg2, shell2)
        @test_throws Exception GeoDynamo.load_initial_conditions!(temp2, :temperature, path)
    end

    @testset "load errors on missing file" begin
        temp = GeoDynamo.create_shtns_temperature_field(Float64, cfg, shell)
        missing_path = joinpath(mktempdir(), "nope.nc")
        @test_throws ArgumentError GeoDynamo.load_initial_conditions!(temp, :temperature, missing_path)
    end
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using GeoDynamo; include("test/ic_netcdf_io.jl")'
```

Expected: FAIL — round-trip mismatches (the current `save_initial_conditions` writes nothing and `load_initial_conditions!` applies an analytic fallback), and the strict-mismatch tests do not throw.

- [ ] **Step 3: Rewrite save_initial_conditions**

In `src/core/initial_conditions.jl`, replace the entire current `save_initial_conditions` function (the body that prints "Saving…", warns "NetCDF saving not implemented", prints "Initial conditions saved") with:

```julia
"""
    save_initial_conditions(field, field_type::Symbol, file_path::String)

Save a field's spectral coefficients to a self-contained NetCDF file usable as a
`FileIC` initial condition. Distributed coefficients are gathered to rank 0,
which writes the file. Returns `file_path`.
"""
function save_initial_conditions(field, field_type::Symbol, file_path::String)
    comps = _ic_components(field_type)
    spectrals = _ic_spectrals(field, field_type)
    cfg = spectrals[1].config
    gdims = PencilArrays.size_global(spectrals[1].pencil)
    nlm = gdims[1]
    nr = gdims[3]
    gathered = map(_ic_gather_global, spectrals)  # each is (gr, gi)

    write_here = !MPI.Initialized() || MPI.Comm_rank(get_comm()) == 0
    if write_here
        dir = dirname(file_path)
        isempty(dir) || isdir(dir) || mkpath(dir)
        NCDataset(file_path, "c") do ds
            defDim(ds, "nlm", nlm)
            defDim(ds, "nr", nr)
            ds.attrib["field_type"] = String(field_type)
            ds.attrib["lmax"] = cfg.lmax
            ds.attrib["mmax"] = cfg.mmax
            ds.attrib["nlm"] = nlm
            ds.attrib["nr"] = nr
            ds.attrib["geodynamo_ic_version"] = GEODYNAMO_IC_VERSION
            defVar(ds, "l_values", collect(Int, cfg.l_values), ("nlm",))
            defVar(ds, "m_values", collect(Int, cfg.m_values), ("nlm",))
            if comps === (:scalar,)
                defVar(ds, "spectral_real", gathered[1][1], ("nlm", "nr"))
                defVar(ds, "spectral_imag", gathered[1][2], ("nlm", "nr"))
            else
                defVar(ds, "toroidal_real", gathered[1][1], ("nlm", "nr"))
                defVar(ds, "toroidal_imag", gathered[1][2], ("nlm", "nr"))
                defVar(ds, "poloidal_real", gathered[2][1], ("nlm", "nr"))
                defVar(ds, "poloidal_imag", gathered[2][2], ("nlm", "nr"))
            end
        end
    end
    if MPI.Initialized() && MPI.Comm_size(get_comm()) > 1
        MPI.Barrier(get_comm())
    end
    return file_path
end
```

- [ ] **Step 4: Rewrite load_initial_conditions!**

In `src/core/initial_conditions.jl`, replace the entire current `load_initial_conditions!` function (the `isfile` check, `println("Loading…")`, the `try` dispatching to per-field stubs, `println("Initial conditions loaded successfully")`) with:

```julia
"""
    load_initial_conditions!(field, field_type::Symbol, file_path::String)

Load a field's spectral coefficients from a NetCDF file written by
`save_initial_conditions`. The file's `field_type` and spectral dimensions
(`nlm`, `nr`) must match `field`; a mismatch raises an error rather than
silently substituting a fallback. Coefficients are read on rank 0, broadcast,
and scattered into each rank's local pencil block.
"""
function load_initial_conditions!(field, field_type::Symbol, file_path::String)
    if !isfile(file_path)
        throw(ArgumentError("Initial conditions file not found: $file_path"))
    end

    comps = _ic_components(field_type)
    spectrals = _ic_spectrals(field, field_type)
    exp_dims = PencilArrays.size_global(spectrals[1].pencil)
    exp_nlm = exp_dims[1]
    exp_nr = exp_dims[3]

    comm_active = MPI.Initialized() && MPI.Comm_size(get_comm()) > 1
    read_here = !comm_active || MPI.Comm_rank(get_comm()) == 0

    # --- Header: read on rank 0, broadcast, validate on ALL ranks ---
    header = zeros(Int, 3)  # [type_code, nlm, nr]
    if read_here
        NCDataset(file_path, "r") do ds
            ft = Symbol(get(ds.attrib, "field_type", ""))
            header[1] = _ic_type_code(ft)
            header[2] = Int(get(ds.attrib, "nlm", 0))
            header[3] = Int(get(ds.attrib, "nr", 0))
        end
    end
    if comm_active
        MPI.Bcast!(header, 0, get_comm())
    end
    file_code, file_nlm, file_nr = header[1], header[2], header[3]
    if file_code != _ic_type_code(field_type)
        error("Initial conditions file $file_path has field_type code $file_code, " *
              "expected $(field_type) (code $(_ic_type_code(field_type)))")
    end
    if file_nlm != exp_nlm || file_nr != exp_nr
        error("Initial conditions file $file_path has nlm=$file_nlm nr=$file_nr, " *
              "expected nlm=$exp_nlm nr=$exp_nr")
    end

    # --- Data: read on rank 0, broadcast, scatter ---
    for (ci, spectral) in enumerate(spectrals)
        gr = zeros(Float64, exp_nlm, exp_nr)
        gi = zeros(Float64, exp_nlm, exp_nr)
        if read_here
            rname, iname = if comps === (:scalar,)
                ("spectral_real", "spectral_imag")
            elseif ci == 1
                ("toroidal_real", "toroidal_imag")
            else
                ("poloidal_real", "poloidal_imag")
            end
            NCDataset(file_path, "r") do ds
                gr .= Array(ds[rname][:, :])
                gi .= Array(ds[iname][:, :])
            end
        end
        if comm_active
            MPI.Bcast!(gr, 0, get_comm())
            MPI.Bcast!(gi, 0, get_comm())
        end
        _ic_scatter_local!(spectral, gr, gi)
    end

    # --- Ball regularity + finiteness (mirror randomize_*) ---
    domain = _ic_field_domain(field, field_type)
    if comps === (:scalar,)
        _maybe_enforce_ball_scalar!(field, domain)
    else
        _maybe_enforce_ball_vector!(field, domain)
    end
    for spectral in spectrals
        if any(isnan, parent(spectral.data_real)) || any(isinf, parent(spectral.data_real))
            error("Non-finite values after loading initial conditions from $file_path")
        end
    end
    return field
end
```

- [ ] **Step 5: Delete the per-field load stubs**

In `src/core/initial_conditions.jl`, delete these four now-unused functions (each is the `@warn "NetCDF loading not implemented…"` + `set_analytical_*` fallback): `load_temperature_initial_conditions!`, `load_magnetic_initial_conditions!`, `load_velocity_initial_conditions!`, `load_composition_initial_conditions!`. Leave the `set_analytical_*` functions intact (they remain used by `AnalyticIC`).

- [ ] **Step 6: Run tests to verify they pass**

Run:

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using GeoDynamo; include("test/ic_netcdf_io.jl")'
```

Expected: PASS — all round-trip and strict-mismatch testsets green.

- [ ] **Step 7: Commit**

```bash
git add src/core/initial_conditions.jl test/ic_netcdf_io.jl
git commit -m "feat(ic): real NetCDF load/save with strict validation"
```

---

## Task 4: Update existing placeholder tests to the new contract

**Files:**
- Modify: `test/initial_conditions.jl` (the `@testset "Placeholder load/save behavior stays explicit"` ~line 80-97)
- Modify: `test/tail_coverage_extended.jl` (the load/save placeholder asserts ~line 423-438)

- [ ] **Step 1: Replace the testset in test/initial_conditions.jl**

Replace the whole `@testset "Placeholder load/save behavior stays explicit" begin … end` block with:

```julia
    @testset "NetCDF load/save round-trips and validates" begin
        temp = GeoDynamo.create_shtns_temperature_field(Float64, cfg, shell)
        GeoDynamo.generate_random_initial_conditions!(
            temp, :temperature, amplitude = 0.2, modes_range = 1:3, seed = 9)
        ref_real = copy(parent(temp.spectral.data_real))

        missing_path = joinpath(mktempdir(), "missing_initial_conditions.nc")
        @test_throws ArgumentError GeoDynamo.load_initial_conditions!(temp, :temperature, missing_path)

        save_path = joinpath(mktempdir(), "saved_initial_conditions.nc")
        @test GeoDynamo.save_initial_conditions(temp, :temperature, save_path) == save_path
        @test isfile(save_path)

        fill!(parent(temp.spectral.data_real), 0.0)
        loaded = GeoDynamo.load_initial_conditions!(temp, :temperature, save_path)
        @test loaded === temp
        @test parent(temp.spectral.data_real) == ref_real
    end
```

- [ ] **Step 2: Inspect and replace the tail_coverage_extended.jl asserts**

Open `test/tail_coverage_extended.jl` around lines 423-438. It currently reads (approximately):

```julia
        # ---- load_initial_conditions! placeholder behavior ----
        @test_throws ArgumentError GeoDynamo.load_initial_conditions!(
            <temp>, :temperature, <missing_path>)
        loaded = @test_logs (:warn, r"NetCDF loading not implemented") GeoDynamo.load_initial_conditions!(
            <temp>, :temperature, <existing_path>)
        # ---- save_initial_conditions placeholder behavior ----
        saved = @test_logs (:warn, r"NetCDF saving not implemented") GeoDynamo.save_initial_conditions(
            <temp>, :temperature, <save_path>)
```

Replace the two `@test_logs (:warn, …)` assertions with a real round-trip, keeping the existing field/`temp` variable and the `ArgumentError`-on-missing assertion. Use the surrounding test's existing `temp`/`cfg`/domain variables (read the file to get their exact names). The replacement body:

```julia
        # ---- load/save: real NetCDF round-trip ----
        missing_path = joinpath(mktempdir(), "missing.nc")
        @test_throws ArgumentError GeoDynamo.load_initial_conditions!(
            temp, :temperature, missing_path)

        save_path = joinpath(mktempdir(), "tail_ic.nc")
        ref_real = copy(parent(temp.spectral.data_real))
        @test GeoDynamo.save_initial_conditions(temp, :temperature, save_path) == save_path
        fill!(parent(temp.spectral.data_real), 0.0)
        @test GeoDynamo.load_initial_conditions!(temp, :temperature, save_path) === temp
        @test parent(temp.spectral.data_real) == ref_real
```

If the surrounding `temp` was never randomized (so `ref_real` is all zeros), that is fine — the round-trip still must reproduce it. If the local field variable has a different name, substitute it.

- [ ] **Step 3: Run both test files**

Run:

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using GeoDynamo; include("test/initial_conditions.jl")' 2>&1 | grep -E "Test Summary|Pass|Fail|Error"
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using GeoDynamo; include("test/tail_coverage_extended.jl")' 2>&1 | grep -E "Test Summary|Pass|Fail|Error"
```

Expected: both PASS, no failures, no remaining "not implemented" warning expectations.

- [ ] **Step 4: Commit**

```bash
git add test/initial_conditions.jl test/tail_coverage_extended.jl
git commit -m "test(ic): replace placeholder IC IO assertions with round-trip contract"
```

---

## Task 5: Time-aware find_restart_files

**Files:**
- Modify: `src/io/restart.jl` (`find_restart_files` ~line 341-354)
- Test: `test/io_restart_roundtrip.jl` (add a testset after the existing `find_restart_files` testset, ~line 51)

- [ ] **Step 1: Write the failing test**

In `test/io_restart_roundtrip.jl`, immediately after the existing `@testset "find_restart_files filters and orders newest-first" … end` block (around line 51), add:

```julia
    @testset "find_restart_files picks the file nearest target_time" begin
        dir = mktempdir()
        # Three real NetCDF restart files with distinct stored times, written
        # oldest-first so mtime order (newest-first) differs from time order.
        for (name, t) in [("geodynamo_shell_restart_1.nc", 1.0),
                          ("geodynamo_shell_restart_2.nc", 5.0),
                          ("geodynamo_shell_restart_3.nc", 9.0)]
            NCDataset(joinpath(dir, name), "c") do ds
                defDim(ds, "scalar", 1)
                defVar(ds, "time", Float64, ("scalar",))[1] = t
            end
            sleep(0.02)
        end

        # Closest to 4.5 is the t=5.0 file (restart_2), NOT the newest (restart_3).
        found = GeoDynamo.find_restart_files(dir, 4.5)
        @test endswith(found[1], "geodynamo_shell_restart_2.nc")

        # Closest to 8.0 is the t=9.0 file (restart_3).
        found2 = GeoDynamo.find_restart_files(dir, 8.0)
        @test endswith(found2[1], "geodynamo_shell_restart_3.nc")

        # Non-finite target falls back to mtime newest-first (restart_3).
        found3 = GeoDynamo.find_restart_files(dir, Inf)
        @test endswith(found3[1], "geodynamo_shell_restart_3.nc")
    end
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using GeoDynamo; using NCDatasets, MPI; MPI.Initialized() || MPI.Init(); include("test/io_restart_roundtrip.jl")' 2>&1 | grep -E "nearest target_time|Fail|Error|Test Summary"
```

Expected: FAIL — current `find_restart_files` returns mtime newest-first (restart_3) for `target_time = 4.5`, so the first assertion fails.

- [ ] **Step 3: Rewrite find_restart_files**

In `src/io/restart.jl`, replace the whole `find_restart_files` function with:

```julia
"""
    find_restart_files(restart_dir, target_time)

Return restart NetCDF files ordered for selection.

When `target_time` is finite and every candidate carries a readable `time`
variable, files are ordered by closeness to `target_time` (nearest first).
Otherwise (non-finite target, or any candidate not a readable NetCDF with a
`time` variable) the function falls back to modification-time order
(newest first).
"""
function find_restart_files(restart_dir::String, target_time::Float64)
    files = readdir(restart_dir)
    restart_files = filter(f -> endswith(f, ".nc") && contains(f, "restart"), files)
    if isempty(restart_files)
        return String[]
    end
    full_paths = [joinpath(restart_dir, f) for f in restart_files]

    if isfinite(target_time)
        times = Vector{Union{Float64, Nothing}}(undef, length(full_paths))
        all_have_time = true
        for (i, p) in enumerate(full_paths)
            t = try
                NCDataset(p, "r") do ds
                    haskey(ds, "time") ? Float64(ds["time"][1]) : nothing
                end
            catch
                nothing
            end
            times[i] = t
            all_have_time &= (t !== nothing)
        end
        if all_have_time
            order = sortperm([abs(times[i] - target_time) for i in eachindex(times)])
            return full_paths[order]
        end
    end

    # Fallback: modification time, newest first.
    sort!(full_paths, by = mtime, rev = true)
    return full_paths
end
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using GeoDynamo; using NCDatasets, MPI; MPI.Initialized() || MPI.Init(); include("test/io_restart_roundtrip.jl")' 2>&1 | grep -E "Fail|Error|Test Summary"
```

Expected: PASS — both the new nearest-time testset and the existing newest-first testset (dummy `"x"` files with `target_time = 0.0` fall back to mtime) are green.

- [ ] **Step 5: Commit**

```bash
git add src/io/restart.jl test/io_restart_roundtrip.jl
git commit -m "feat(restart): honor target_time in find_restart_files"
```

---

## Task 6: Ensure new test file is in the suite + full verification

**Files:**
- Modify: `test/runtests.jl` (add `include("ic_netcdf_io.jl")` if the suite uses an explicit include list)

- [ ] **Step 1: Check how the suite includes test files**

Run:

```bash
grep -nE "ic_netcdf_io|include\(|for .* in .*readdir|@testset" test/runtests.jl | head -40
```

If `runtests.jl` auto-includes every `test/*.jl`, no change is needed. If it has an explicit list, add `include("ic_netcdf_io.jl")` next to the other initial-conditions includes.

- [ ] **Step 2: Make the edit if needed**

If an explicit list exists, add the include line in the appropriate spot (near `include("initial_conditions.jl")`).

- [ ] **Step 3: Run the full test suite to a log file**

Per project memory, do NOT pipe `Pkg.test` through `tail` (it masks the exit code). Redirect to a file:

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Pkg; Pkg.test()' > /tmp/ic_suite.log 2>&1; echo "EXIT=$?"
grep -E "Test Summary|FAIL|Error|Pass|Broken" /tmp/ic_suite.log | tail -40
```

Expected: `EXIT=0`. Note from project memory: the suite is nondeterministic and a handful of scalar-IC normalization tests can flake; a single run showing ~3 IC failures may be a flake — re-run before attributing to this change. Green baseline is ~2758 pass / 2 broken.

- [ ] **Step 4: Commit any include change**

```bash
git add test/runtests.jl
git commit -m "test(ic): include ic_netcdf_io.jl in the suite"
```

(Skip if no change was needed.)

---

## Self-review notes

- **Spec coverage:** format (Task 3 save), gather/scatter (Task 2), strict validation + deadlock-safe header (Task 3 load), restart target_time (Task 5), test updates (Task 4), new tests (Tasks 2-3, 5). All spec sections mapped.
- **Type consistency:** helper names used identically across tasks — `_ic_components`, `_ic_spectrals`, `_ic_type_code`, `_ic_field_domain`, `_ic_gather_global`, `_ic_scatter_local!`, `GEODYNAMO_IC_VERSION`. Variable names `gr`/`gi` are the gathered real/imag matrices throughout.
- **Out of scope (unchanged):** restart/output-format compatibility for `FileIC`; parallel MPI-IO for IC files; resolution remapping on load.
- **MPI note:** all collectives are guarded by `MPI.Initialized()` + `Comm_size > 1`, so the serial test path runs no collectives; the strict header check runs on all ranks before any data broadcast to avoid a rank-0-only throw deadlocking the others.
