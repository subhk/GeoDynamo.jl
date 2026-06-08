# Phase-3 Transform Scratch State-Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the process-global mutable scratch state behind the Phase-3 `DistTransposePlan` transform — five `IdDict{Any,Any}` caches plus the `VELOCITY_WS` `Ref` — by making each piece of scratch owned by the object whose lifetime it shares (the per-config `SHTnsBuffers`, or the velocity field), so scratch is created, found, and garbage-collected with its owner instead of living in module globals.

**Architecture:** The codebase already has the right pattern: `SHTnsKitConfig{T,P,FP,TP,B<:SHTnsBuffers}` holds a lazily-populated, typed `mutable struct SHTnsBuffers` (`config._buffers`) for transform scratch, accessed via `get_cached_buffer!`. We extend that holder to also own the Phase-3 plan / transpose-scratch / m-bridge / scalar-scratch / vector-scratch, and repoint the five `_get_*` accessors at `config._buffers` instead of the global `IdDict`s. The velocity workspace moves onto `SHTnsVelocityFields` the same way. No transform math changes; the build functions are reused verbatim — only *where their result is stored* changes.

**Scope note:** This is one subsystem (Phase-3 transform scratch + velocity workspace). It does **not** touch `GEODYNAMO_PARAMS` (the global-parameter-state sweep — separate plan) or the `solver`/`interop.jl` layer collapse (the larger follow-on this de-risks). Diagnostic counters (`_*_REDUCE_COUNT`) and config flags (`ENABLE_TIMING`, `ERK2_DIAGNOSTICS_*`) are intentionally left as globals — they are benign and not scratch.

**Honest expectation:** The primary win is architectural — 6 fewer module-global mutable objects, scratch lifetime tied to its owner, no unbounded cross-config accumulation in the dicts, no manual `delete!` teardown, and reduced global coupling (which makes the future `interop.jl` collapse safer). Type-stability/allocations are **unchanged** for the `PencilArray`-bearing scratch (its concrete type is config-dependent and already handled by the existing function barriers); the one measurable perf bonus is concretely typing the config-independent m-bridge struct, removing its `::Any` field-access boxing.

**Tech Stack:** Julia 1.10/1.11, PencilArrays, SHTnsKit 1.2.12, MPI. Run tests with the direct binary `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia` (the `julia` shim is broken on this box). Correctness oracle: the existing bit-exact transform/equivalence suites.

---

## File Structure

- `src/transforms/spectral.jl` — `SHTnsBuffers` gains five typed scratch fields (init `nothing`); `SHTnsBuffers()` ctor + `_BUFFERS_FIELD_MAP` updated. **Owns** all Phase-3 transform scratch.
- `src/parallel/disttranspose_adapter.jl` — `get_disttranspose_plan`, `_get_disttranspose_scratch`, `_get_mbridge` repointed to `cfg._buffers`; the three `IdDict{Any,Any}` consts deleted; new concrete `struct _MBridge` (config-independent fields) replaces the m-bridge `NamedTuple`.
- `src/physics/nonlinear.jl` — `_scalar_scratch`, `_vector_scratch` repointed to `config._buffers`; the two `_P3_*_SCRATCH_CACHE` consts + the `delete!`-based teardown deleted.
- `src/physics/velocity/field.jl` — `VELOCITY_WS` global removed; workspace becomes a lazily-built field on `SHTnsVelocityFields` (this repeats the reverted commit `b315691`'s velocity half — see git history for the exact diff).
- `src/GeoDynamo.jl` — drop `set_velocity_workspace!`/`get_velocity_workspace` from exports.
- Tests reused as the oracle (no new transform tests needed — equivalence already exists): `test/p3_transpose.jl`, `test/theta_dist_transform.jl`, `test/r_theta_transpose.jl`, `test/shtnskit_roundtrip.jl`, `test/r_theta_equivalence.jl`, `test/cnab2_rhs_distributed_equivalence.jl`, `test/allocation_runtime_checks.jl`, `test/field_containers.jl`, `test/ball_finiteness.jl`.

Order matters: do the **plan cache** first (simplest, single value), then **m-bridge** (gets the concrete-type bonus), then the **transpose scratch**, then the **two P3 scratches**, then **VELOCITY_WS**. Each phase is independently green-able.

---

## Task 1: Add Phase-3 scratch fields to `SHTnsBuffers`

**Files:**
- Modify: `src/transforms/spectral.jl` (the `mutable struct SHTnsBuffers` block ~69-134 and `SHTnsBuffers()` ctor ~140-149)

- [ ] **Step 1: Add fields to the struct.** Append, before the closing `end` of `mutable struct SHTnsBuffers`:

```julia
    # Phase-3 DistTransposePlan transform scratch (per-config; replaces the old
    # module-global IdDict{Any,Any} caches). Lazily built on first transform.
    disttranspose_plan::Union{Any, Nothing}        # SHTnsKit.DistTransposePlan
    disttranspose_scratch::Union{Any, Nothing}     # NamedTuple of PencilArrays (config-dependent type)
    disttranspose_mbridge::Union{Any, Nothing}     # _MBridge (concrete; see Task 3)
    p3_scalar_scratch::Union{Any, Nothing}         # NamedTuple (Alm/fspatial/solve)
    p3_vector_scratch::Union{Any, Nothing}         # NamedTuple (Slm/Tlm/Vt/Vp/Vr/Vr_alm/solve)
```

Note: `::Union{Any,Nothing}` matches the existing holder's treatment of config-dependent PencilArray scratch (cf. `synthesis_phi_tmp::Union{PencilArray,Nothing}`). The boxing on use is already handled by the function barriers in `numerics.jl`/`disttranspose_adapter.jl`; we are changing *ownership*, not the access pattern.

- [ ] **Step 2: Update the `SHTnsBuffers()` constructor** to pass five trailing `nothing`s. The ctor sets every field to `nothing`; add five more `nothing` arguments matching the new fields' order.

- [ ] **Step 3: Run the load + an existing transform test to confirm the struct still constructs.**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using GeoDynamo; include("test/p3_transpose.jl")'`
Expected: PASS (fields unused yet — pure additive change).

- [ ] **Step 4: Commit**

```bash
git add src/transforms/spectral.jl
git commit -m "refactor(transform): add Phase-3 scratch slots to SHTnsBuffers (unused)"
```

---

## Task 2: Move the `DistTransposePlan` cache onto the config

**Files:**
- Modify: `src/parallel/disttranspose_adapter.jl` (`get_disttranspose_plan` ~128-144; `const _DISTTRANSPOSE_PLAN_CACHE` ~42)

- [ ] **Step 1: Rewrite `get_disttranspose_plan` to use `cfg._buffers`.** Replace the body:

```julia
function get_disttranspose_plan(cfg)
    b = cfg._buffers
    p = b.disttranspose_plan
    p !== nothing && return p
    lock(_DISTTRANSPOSE_LOCK) do
        if b.disttranspose_plan === nothing
            b.disttranspose_plan = _build_disttranspose_plan(cfg)
        end
        return b.disttranspose_plan
    end
end
```

- [ ] **Step 2: Delete the global.** Remove `const _DISTTRANSPOSE_PLAN_CACHE = IdDict{Any, Any}()` (line ~42). Keep `_DISTTRANSPOSE_LOCK` (still guards the lazy build).

- [ ] **Step 3: Verify the transpose transform round-trips.**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using GeoDynamo; include("test/p3_transpose.jl")'`
Expected: PASS (`vector spec→phys→spec roundtrip (DistTransposePlan, dealiased)` green).

- [ ] **Step 4: Confirm no other reference to the deleted const.**

Run: `grep -rn "_DISTTRANSPOSE_PLAN_CACHE" src/`
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add src/parallel/disttranspose_adapter.jl
git commit -m "refactor(transform): config-own the DistTransposePlan (drop global IdDict)"
```

---

## Task 3: Concrete `_MBridge` struct + config-owned m-bridge

The m-bridge fields are all config-**independent** in type (`MPI.Comm`, `Int`, `UnitRange`, `Vector{ComplexF64}`, `Array{ComplexF64,3}`, `MPI.VBuffer`, `Vector{Int}`), so a concrete struct removes the `mb.x::Any` boxing (and the type-asserts in `spec_storage_to_solve!`).

**Files:**
- Modify: `src/parallel/disttranspose_adapter.jl` (`_build_mbridge` ~260-291; `_get_mbridge` ~293-300; `const _DISTTRANSPOSE_MBRIDGE_CACHE` ~258)

- [ ] **Step 1: Define the concrete struct** (place above `_build_mbridge`). Field names/types must mirror exactly the `NamedTuple` currently returned by `_build_mbridge` (read it: `θ_comm, θ_size, spec_m_range, spec_l_range, nr, l_local, mmax, m_counts, m_firsts, recvcounts, send, recv, vbuf, full3, local_full`). Type each field concretely from the build code (e.g. `send::Vector{ComplexF64}`, `vbuf::MPI.VBuffer{Vector{ComplexF64},Vector{Cint}}`, `spec_m_range::UnitRange{Int}`); where a type is uncertain at write-time, read the corresponding `_build_mbridge` line and use its exact constructed type.

- [ ] **Step 2: Change `_build_mbridge`'s `return (; …)`** to `return _MBridge(θ_comm, θ_size, …, local_full)` (positional, same order as the struct).

- [ ] **Step 3: Rewrite `_get_mbridge` to use `cfg._buffers`:**

```julia
function _get_mbridge(cfg, plan)
    b = cfg._buffers
    mb = b.disttranspose_mbridge
    mb !== nothing && return mb::_MBridge
    lock(_DISTTRANSPOSE_LOCK) do
        if b.disttranspose_mbridge === nothing
            b.disttranspose_mbridge = _build_mbridge(cfg, plan)
        end
        return b.disttranspose_mbridge::_MBridge
    end
end
```

- [ ] **Step 4: Delete the global** `const _DISTTRANSPOSE_MBRIDGE_CACHE` (~258).

- [ ] **Step 5: Simplify `spec_storage_to_solve!`** — its `mb.send::Vector{ComplexF64}` etc. asserts are now redundant (the struct is concrete); they can stay (harmless) or be dropped. If dropping, remove only the `::Type` suffixes on `mb.*` field reads. (Optional within this task.)

- [ ] **Step 6: Verify + check allocations dropped.**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using GeoDynamo; include("test/p3_transpose.jl"); include("test/theta_dist_transform.jl")'`
Expected: PASS. Then run `/tmp/perf_audit.jl` (rebuild from the audit in this session's notes) — `_get_mbridge` should no longer appear in the top allocation sites.

- [ ] **Step 7: Commit**

```bash
git add src/parallel/disttranspose_adapter.jl
git commit -m "refactor(transform): concrete _MBridge struct, config-owned (drop global IdDict + ::Any boxing)"
```

---

## Task 4: Move the transpose scratch onto the config

**Files:**
- Modify: `src/parallel/disttranspose_adapter.jl` (`_get_disttranspose_scratch` ~146-161; `const _DISTTRANSPOSE_SCRATCH_CACHE` ~47)

- [ ] **Step 1: Rewrite `_get_disttranspose_scratch`:**

```julia
@inline function _get_disttranspose_scratch(cfg, plan)
    b = cfg._buffers
    s = b.disttranspose_scratch
    s !== nothing && return s
    lock(_DISTTRANSPOSE_LOCK) do
        if b.disttranspose_scratch === nothing
            b.disttranspose_scratch = _build_disttranspose_scratch(cfg, plan)
        end
        return b.disttranspose_scratch
    end
end
```

- [ ] **Step 2: Delete `const _DISTTRANSPOSE_SCRATCH_CACHE`** (~47).

- [ ] **Step 3: Verify (the scratch drives `to_spec_solve`/`from_spec_solve!`, which are barriered).**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using GeoDynamo; include("test/p3_transpose.jl"); include("test/r_theta_transpose.jl"); include("test/cnab2_rhs_distributed_equivalence.jl")'`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/parallel/disttranspose_adapter.jl
git commit -m "refactor(transform): config-own the DistTranspose scratch (drop global IdDict)"
```

---

## Task 5: Move the scalar + vector P3 scratch onto the config

**Files:**
- Modify: `src/physics/nonlinear.jl` (`_scalar_scratch` ~341-?, `_vector_scratch` ~367-?, the two `const _P3_*_SCRATCH_CACHE` ~339/365, and the `delete!`-based teardown ~398-407)

- [ ] **Step 1: Rewrite `_scalar_scratch`:**

```julia
@inline function _scalar_scratch(config, plan)
    b = config._buffers
    sc = b.p3_scalar_scratch
    sc !== nothing && return sc
    lock(_DISTTRANSPOSE_LOCK) do
        if b.p3_scalar_scratch === nothing
            b.p3_scalar_scratch = _build_scalar_scratch(config, plan)   # use the existing builder body
        end
        return b.p3_scalar_scratch
    end
end
```

(If the build is currently an inline `do`-block inside the `get!`, lift it into a named `_build_scalar_scratch(config, plan)` first — same code, just extracted.)

- [ ] **Step 2: Rewrite `_vector_scratch`** identically against `b.p3_vector_scratch` / `_build_vector_scratch`.

- [ ] **Step 3: Delete the two globals** (`_P3_SCALAR_SCRATCH_CACHE`, `_P3_VECTOR_SCRATCH_CACHE`) and the teardown function that does `delete!(_P3_*_SCRATCH_CACHE, config)` (~398-407) — scratch now GC's with the config, so the explicit teardown is dead. Check for callers of that teardown function and remove them.

- [ ] **Step 4: Verify the scalar + vector solver transforms.**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using GeoDynamo; include("test/theta_dist_transform.jl"); include("test/shtnskit_roundtrip.jl")'`
Expected: PASS (`solver vector dist transform roundtrip`, scalar+vector roundtrip green).

- [ ] **Step 5: Confirm no dangling references.**

Run: `grep -rn "_P3_SCALAR_SCRATCH_CACHE\|_P3_VECTOR_SCRATCH_CACHE" src/`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add src/physics/nonlinear.jl
git commit -m "refactor(transform): config-own the P3 scalar/vector scratch (drop global IdDicts + teardown)"
```

---

## Task 6: State-own the velocity workspace (re-land the reverted velocity half)

This repeats the velocity-workspace half of reverted commit `b315691` (its ERK2 half stayed). Use `git show b315691 -- src/physics/velocity/field.jl src/solver/numerics.jl src/GeoDynamo.jl test/field_containers.jl test/ball_finiteness.jl` as the exact diff to re-apply.

**Files:**
- Modify: `src/physics/velocity/field.jl` (struct `SHTnsVelocityFields` + ctor + `VELOCITY_WS`/`get`/`set` + the `compute_vorticity_spectral_full!` dispatcher)
- Modify: `src/solver/numerics.jl` (the `get_velocity_workspace`/`set_velocity_workspace!` reuse site ~1067)
- Modify: `src/GeoDynamo.jl` (exports)
- Modify: `test/field_containers.jl`, `test/ball_finiteness.jl`

- [ ] **Step 1: Re-apply the velocity-workspace diff** from `b315691` (adds `velocity_workspace::Union{VelocityWorkspace{T},Nothing}` field defaulting to `nothing`, a `_get_or_build_velocity_workspace!(𝒰, nr, nthreads)` helper, removes `VELOCITY_WS`/`set_`/`get_velocity_workspace`, repoints the two reuse sites and the two tests).

- [ ] **Step 2: Verify.**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using GeoDynamo; include("test/field_containers.jl")'`
and `... include("test/ball_finiteness.jl")`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/physics/velocity/field.jl src/solver/numerics.jl src/GeoDynamo.jl test/field_containers.jl test/ball_finiteness.jl
git commit -m "refactor(velocity): state-own the velocity workspace (drop VELOCITY_WS global)"
```

---

## Task 7: Full-suite verification + global-state inventory check

- [ ] **Step 1: Run the full suite.**

Run: `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --project=. -e 'using Pkg; Pkg.test()' > /tmp/ft.log 2>&1; grep -E "Testing GeoDynamo tests passed|Some tests did not pass" /tmp/ft.log`
Expected: `Testing GeoDynamo tests passed`. (If `erk2_integration_step.jl:84` flakes, re-run — it is the documented env/FP-adjacent test; magnetic-poloidal finiteness is otherwise deterministic post the `b315691` ERK2-buffer fix.)

- [ ] **Step 2: Confirm the global count dropped.**

Run: `grep -rcE "^const [A-Za-z_]+ = (Ref|IdDict|Dict)" src/ | grep -v ':0'`
Expected: 6 fewer entries than baseline (5 IdDicts + VELOCITY_WS gone); remaining globals are counters/flags/the dims-keyed `_BC_SHTNS_CONFIG_CACHE`/`SOLVER_MODE_INDEX_CACHE` (out of scope).

- [ ] **Step 3: Re-profile to confirm no regression** (allocations equal-or-better; m-bridge boxing gone).

Run the session's `/tmp/perf_audit.jl`. Expected: CNAB2 ≤ 545 KB/step, ERK2 ≤ 1733 KB/step.

---

## Risks & Notes

- **Thread-safety:** keep `_DISTTRANSPOSE_LOCK` guarding every lazy build; the double-checked-lock pattern above is required (check field, lock, re-check, build).
- **Bit-exactness:** no math changes — the build functions are reused verbatim, only their storage moves. The equivalence suites (`r_theta_equivalence`, `p3_transpose`, `cnab2_rhs_distributed_equivalence`) are the hard oracle; if any diff appears, a build function was altered by mistake.
- **Concurrent session:** this repo gets concurrent auto-commits on `feat/ic-netcdf-io-restart`. Commit each task promptly; before editing a file, re-read it (it may have moved).
- **`_MBridge` field types (Task 3):** the only place exact types must be transcribed from the build code — read `_build_mbridge` line-by-line; do not guess `VBuffer`'s type parameters.
- **Out of scope / follow-ons:** `GEODYNAMO_PARAMS` global-parameter elimination (separate plan); the `solver/interop.jl` 56-alias + `SOLVER_*_BUILDER` collapse (the larger simplification this de-risks — plan separately once global scratch coupling is gone).
