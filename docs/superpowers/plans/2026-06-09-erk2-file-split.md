# erk2.jl File-Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the 2952-LOC `src/timestep/erk2.jl` into a thin include spine plus five focused files, relocating every function verbatim (no logic changes, behavior-preserving).

**Architecture:** `erk2.jl` becomes a ~30-line spine that `include`s `erk2/{common,boundary,cache,influence,integrate}.jl` in that order. Julia splices includes flatly into the `GeoDynamo` module, so no call site outside the file changes. The only parse-order constraint is that `common.jl` (which holds all 7 const aliases) is included first; everything else resolves at call time. No struct definitions move (they live in `timestep/state.jl`).

**Tech Stack:** Julia 1.11, run with the direct binary `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=.`. Design spec: `docs/superpowers/specs/2026-06-09-erk2-file-split-design.md`.

**This is a pure-move refactor — NOT TDD.** There is no new behavior to red-green. Each task is verified by a module-load check; the final task by a function-set-equality check plus the full suite. Do not change any function body, signature, comment, or docstring while moving — relocate blocks byte-for-byte (including each function's preceding docstring/comment).

---

## File Structure

- **Create** `src/timestep/erk2/common.jl` — const aliases, compat constructors, `with_boundary_mode_values`, phi functions, diagnostics toggles. Included first (alias-ordering).
- **Create** `src/timestep/erk2/boundary.jl` — BC constructors + `build_*_bc` builders.
- **Create** `src/timestep/erk2/cache.jl` — ERK2 cache lifecycle: builders, accessors, bundle persistence.
- **Create** `src/timestep/erk2/influence.jl` — poloidal-velocity influence matrices + corrections.
- **Create** `src/timestep/erk2/integrate.jl` — field buffers, stage execution, residual logging, the main `integrate_solver_erk2_step!`.
- **Modify** `src/timestep/erk2.jl` — reduced to the five `include` lines (the spine).
- **Untouched:** `src/solver.jl:20` (`include("timestep/erk2.jl")`) stays as-is; `timestep/state.jl`, `timestep/driver.jl`, GPU paths.

**Extraction method (applies to every task):** keep a pristine reference of the original file (Task 1 saves it to `/tmp/erk2_orig.jl`). For each new file, move the listed functions **by signature** out of `erk2.jl` — find each `function <name>(...) ... end` block (including the docstring/comment immediately above it) and relocate it verbatim, preserving source order. Add a one-line section banner comment at the top of each new file. After moving a cluster, delete the now-empty lines from `erk2.jl` and add the file's `include` line to the spine.

---

### Task 1: Extract `common.jl` (aliases first)

**Files:**
- Create: `src/timestep/erk2/common.jl`
- Modify: `src/timestep/erk2.jl`

- [ ] **Step 1: Save the pristine original as a reference**

```bash
cd /Users/subha/Documents/GitHub/GeoDynamo.jl
git show HEAD:src/timestep/erk2.jl > /tmp/erk2_orig.jl
grep -cE "^function " /tmp/erk2_orig.jl   # expect: 80
grep -oE "^function [A-Za-z0-9_.!{}]+" /tmp/erk2_orig.jl | sort > /tmp/erk2_funcs_orig.txt
wc -l /tmp/erk2_funcs_orig.txt            # expect: 80
```

- [ ] **Step 2: Create `src/timestep/erk2/common.jl`**

Move these blocks out of `erk2.jl` into the new file, in this source order. The original-file line numbers are given only to locate them in `/tmp/erk2_orig.jl`; copy each block (with its docstring/comment) verbatim.

Header line for the new file:
```julia
# ERK2 common: const aliases, compat constructors, phi functions, diagnostics toggles.
```

Blocks to move (in order):
1. `with_boundary_mode_values` (orig line 60)
2. The 7 const aliases as a contiguous block (orig 79–84): `const ERK2Cache = ERK2StageCache`, `const Phi2ConditioningMonitor = SolverPhi2ConditioningMonitor`, `const PHI2_MONITOR = SOLVER_PHI2_MONITOR`, `const ERK2BoundarySide = SolverERK2BoundarySide`, `const ERK2BoundarySpec = SolverERK2BoundarySpec`, `const ERK2InfluenceMatrix = ERK2InfluenceOp` — **plus** `const ERK2FieldBuffers = SolverERK2FieldBuffers` (orig 1967; move it up here with the others). The aliases MUST appear before block 3.
3. `function GeoDynamo.ERK2Cache{T}(...)` compat ctor (orig 94) **and** the one-line `GeoDynamo.ERK2Cache(args...) = GeoDynamo.ERK2Cache{Float64}(args...)` (orig 131) that follows it.
4. `compat_normalize_old_erk2_cache_entry` (orig 144)
5. `set_erk2_diagnostics_interval!` (orig 435), `enable_erk2_diagnostics!` (orig 447), `disable_erk2_diagnostics!` (orig 460)
6. `compute_phi1_function` (orig 1063), `compute_phi2_function` (orig 1072), `report_phi2_conditioning` (orig 1088)
7. **Assignment-form one-liner methods (NOT `^function` blocks — easy to miss).** They belong to the compat/diagnostics/phi clusters that all live in common.jl, so move them here beside their cluster. Leaving them in `erk2.jl` would make Task 5 silently DROP them when the spine is finalized.
   - `@inline compat_solver_erk2_cache(cache::ERK2StageCache{T}) where {T} = cache` (orig 133) and `@inline compat_old_erk2_cache(cache::ERK2StageCache{T}) where {T} = cache` (orig 134) — place with the compat cluster, right after the `GeoDynamo.ERK2Cache(args...)` one-liner.
   - `GeoDynamo.erk2_diagnostics_enabled() = SOLVER_SHARED_ERK2_DIAGNOSTICS_ENABLED[]` (orig 471) and `GeoDynamo.erk2_diagnostics_interval() = SOLVER_SHARED_ERK2_DIAGNOSTICS_INTERVAL[]` (orig 478) — place with the diagnostics toggles.
   - `GeoDynamo.reset_phi2_monitor!() = reset_solver_phi2_monitor!()` (orig 1081) — place with the phi functions.

That is 9 `^function` blocks + 6 assignment-form one-liners (the `ERK2Cache(args...)` ctor + the 5 above) + 7 consts. (The `@eval GeoDynamo begin … end` stub block at orig lines 16–53 does NOT move — it stays atop the `erk2.jl` spine before all includes; see Task 5.)

- [ ] **Step 3: Reduce `erk2.jl` to a spine-with-remainder**

At the very TOP of `erk2.jl` (before any remaining code), add:
```julia
include("erk2/common.jl")
```
Delete the moved blocks from their original positions in `erk2.jl`. The file now = that one `include` line + the 71 not-yet-moved `^function` blocks.

- [ ] **Step 4: Verify the module loads**

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using GeoDynamo; println("LOAD_OK")' 2>&1 | tail -3
```
Expected: `LOAD_OK`. A `UndefVarError`/`MethodError` here means an alias landed after a use, or a block was dropped — fix before committing.

- [ ] **Step 5: Commit**

```bash
git add src/timestep/erk2.jl src/timestep/erk2/common.jl
git commit -m "refactor(erk2): extract common.jl (aliases, compat, phi, diagnostics)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Extract `boundary.jl`

**Files:**
- Create: `src/timestep/erk2/boundary.jl`
- Modify: `src/timestep/erk2.jl`

- [ ] **Step 1: Create `src/timestep/erk2/boundary.jl`**

Header:
```julia
# ERK2 boundary conditions: endpoint BC constructors and per-field BC-spec builders.
```

Move these 21 `^function` blocks (with docstrings), in source order, out of `erk2.jl`:
`solver_enforce_erk2_bc!`, `GeoDynamo.enforce_erk2_bc!`, `solver_create_dirichlet_bc`, `GeoDynamo.create_dirichlet_bc`, `solver_create_neumann_bc`, `GeoDynamo.create_neumann_bc`, `solver_create_stress_free_tor_bc`, `GeoDynamo.create_stress_free_tor_bc`, `solver_create_noslip_pol_bc`, `GeoDynamo.create_noslip_pol_bc`, `solver_create_stress_free_pol_bc`, `GeoDynamo.create_stress_free_pol_bc`, `solver_create_insulating_inner_bc`, `GeoDynamo.create_insulating_inner_bc`, `solver_create_insulating_outer_bc`, `GeoDynamo.create_insulating_outer_bc`, `build_solver_erk2_scalar_bc`, `build_solver_erk2_velocity_tor_bc`, `build_solver_erk2_velocity_pol_bc`, `build_solver_erk2_magnetic_tor_bc`, `build_solver_erk2_magnetic_pol_bc`.

- [ ] **Step 2: Add the include to the spine**

In `erk2.jl`, add after the `common.jl` include line:
```julia
include("erk2/boundary.jl")
```
Delete the moved blocks from `erk2.jl`.

- [ ] **Step 3: Verify the module loads**

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using GeoDynamo; println("LOAD_OK")' 2>&1 | tail -3
```
Expected: `LOAD_OK`.

- [ ] **Step 4: Commit**

```bash
git add src/timestep/erk2.jl src/timestep/erk2/boundary.jl
git commit -m "refactor(erk2): extract boundary.jl (BC constructors + builders)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Extract `cache.jl`

**Files:**
- Create: `src/timestep/erk2/cache.jl`
- Modify: `src/timestep/erk2.jl`

- [ ] **Step 1: Create `src/timestep/erk2/cache.jl`**

Header:
```julia
# ERK2 cache lifecycle: cache builders, memoized accessors, and bundle persistence.
```

Move these 24 `^function` blocks (with docstrings), in source order:
`create_solver_erk2_scalar_cache`, `create_solver_erk2_cache`, `create_solver_erk2_magnetic_toroidal_cache`, `create_solver_erk2_magnetic_poloidal_cache`, `GeoDynamo.create_erk2_cache`, `GeoDynamo.create_erk2_cache_scalar`, `GeoDynamo.create_erk2_cache_temperature`, `GeoDynamo.create_erk2_cache_composition`, `GeoDynamo.create_erk2_cache_magnetic_toroidal`, `GeoDynamo.create_erk2_cache_magnetic_poloidal`, `_get_or_build_erk2_cache`, `_get_or_build_erk2_scalar_cache`, `get_solver_erk2_temperature_cache!`, `get_solver_erk2_composition_cache!`, **all three methods of** `get_solver_erk2_cache!` (orig lines 1567, 1601, 1636), `get_solver_erk2_magnetic_toroidal_cache!`, `get_solver_erk2_magnetic_poloidal_cache!`, `save_erk2_cache_bundle`, `load_erk2_cache_bundle`, **both methods of** `install_erk2_cache_bundle!` (orig 2475, 2492), `load_erk2_cache_bundle!`.

> Note: do NOT move `_get_or_build_erk2_influence_entry` (orig 1198) — it belongs to `influence.jl` in Task 4. It sits between `_get_or_build_erk2_scalar_cache` and the cache accessors; skip over it.

- [ ] **Step 2: Add the include to the spine**

In `erk2.jl`, add after the `boundary.jl` include line:
```julia
include("erk2/cache.jl")
```
Delete the moved blocks from `erk2.jl`.

- [ ] **Step 3: Verify the module loads**

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using GeoDynamo; println("LOAD_OK")' 2>&1 | tail -3
```
Expected: `LOAD_OK`.

- [ ] **Step 4: Commit**

```bash
git add src/timestep/erk2.jl src/timestep/erk2/cache.jl
git commit -m "refactor(erk2): extract cache.jl (builders, accessors, bundle persistence)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Extract `influence.jl`

**Files:**
- Create: `src/timestep/erk2/influence.jl`
- Modify: `src/timestep/erk2.jl`

- [ ] **Step 1: Create `src/timestep/erk2/influence.jl`**

Header:
```julia
# ERK2 influence matrices: poloidal-velocity boundary-condition correction.
```

Move these 8 `^function` blocks (with docstrings), in source order:
`_get_or_build_erk2_influence_entry`, `create_solver_velocity_poloidal_influence_matrices`, `get_solver_erk2_influence_matrices!`, `apply_solver_influence_matrix_correction!`, `apply_solver_velocity_poloidal_influence_correction!`, `GeoDynamo.create_velocity_poloidal_influence_matrices`, `GeoDynamo.apply_influence_matrix_correction!`, `GeoDynamo.apply_velocity_poloidal_influence_correction!`.

- [ ] **Step 2: Add the include to the spine**

In `erk2.jl`, add after the `cache.jl` include line:
```julia
include("erk2/influence.jl")
```
Delete the moved blocks from `erk2.jl`.

- [ ] **Step 3: Verify the module loads**

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using GeoDynamo; println("LOAD_OK")' 2>&1 | tail -3
```
Expected: `LOAD_OK`.

- [ ] **Step 4: Commit**

```bash
git add src/timestep/erk2.jl src/timestep/erk2/influence.jl
git commit -m "refactor(erk2): extract influence.jl (poloidal-velocity BC correction)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Extract `integrate.jl` (spine becomes pure)

**Files:**
- Create: `src/timestep/erk2/integrate.jl`
- Modify: `src/timestep/erk2.jl`

- [ ] **Step 1: Create `src/timestep/erk2/integrate.jl`**

Header:
```julia
# ERK2 integration: field buffers, stage execution, residual logging, and the step entry point.
```

Move the remaining 18 `^function` blocks (with docstrings), in source order:
`SolverERK2FieldBuffers`, `erk2_field_buffers_match`, `get_solver_erk2_field_buffers!`, `prepare_solver_erk2_field!`, `GeoDynamo.erk2_prepare_field!`, `apply_solver_erk2_stage!`, `GeoDynamo.erk2_apply_stage!`, `store_solver_erk2_stage_nonlinear!`, `GeoDynamo.erk2_store_stage_nonlinear!`, `finalize_solver_erk2_field!`, `GeoDynamo.erk2_finalize_field!`, `solver_erk2_stage_residual_stats`, `GeoDynamo.erk2_stage_residual_stats`, `maybe_log_solver_erk2_stage_residual!`, `GeoDynamo.maybe_log_erk2_stage_residual!`, `restore_solver_erk2_nonlinear_terms!`, `_get_or_build_erk2_boundary_spec!`, `integrate_solver_erk2_step!`.

- [ ] **Step 2: Finish the spine**

In `erk2.jl`, add after the `influence.jl` include line:
```julia
include("erk2/integrate.jl")
```
Delete the moved blocks.

**Do NOT blindly overwrite the whole file** — the `@eval GeoDynamo begin … end` stub block at the top of `erk2.jl` (orig lines 16–53) declares the public `GeoDynamo.*` function names that the part files extend; it MUST be preserved at the top of the spine, before any include. After all extractions, `erk2.jl` should consist of exactly: that `@eval` block, then the five include lines (plus the existing module preamble/comments above the `@eval` block, if any). Verify the only remaining top-level definitions are the `@eval` block and the includes:
```bash
grep -cE "^function " src/timestep/erk2.jl                          # expect: 0
grep -nE "^(@inline |GeoDynamo\.[A-Za-z0-9_.!]*\(.*\) =|const )" src/timestep/erk2.jl   # expect: no output (all one-liners/consts moved to parts)
grep -c "^@eval GeoDynamo begin" src/timestep/erk2.jl              # expect: 1 (the stub block is preserved)
grep -cE "^include\(\"erk2/" src/timestep/erk2.jl                  # expect: 5
```
The resulting spine should look like:
```julia
# ERK2 timestepper include spine. Implementation lives in erk2/*.jl.
# <existing preamble comments, if any>
@eval GeoDynamo begin
    # ... existing stub declarations, unchanged ...
end

# common.jl MUST come first: it defines the const aliases the other files use.
include("erk2/common.jl")
include("erk2/boundary.jl")
include("erk2/cache.jl")
include("erk2/influence.jl")
include("erk2/integrate.jl")
```

- [ ] **Step 3: Verify the module loads**

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using GeoDynamo; println("LOAD_OK")' 2>&1 | tail -3
```
Expected: `LOAD_OK`.

- [ ] **Step 4: Commit**

```bash
git add src/timestep/erk2.jl src/timestep/erk2/integrate.jl
git commit -m "refactor(erk2): extract integrate.jl; erk2.jl is now a pure spine

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Repoint static checks, then verify the split

**Files:**
- Modify: `test/temperature_boundary_static_checks.jl:31`
- Modify: `test/composition_boundary_static_checks.jl:31`
- Modify: `test/magnetic_boundary_static_checks.jl:26`
- Modify: `test/velocity_boundary_static_checks.jl:28`

- [ ] **Step 1: Function-set equality (nothing lost or duplicated)**

```bash
cd /Users/subha/Documents/GitHub/GeoDynamo.jl
# Every function now lives in erk2/*.jl; the spine has none.
grep -cE "^function " src/timestep/erk2.jl            # expect: 0
cat src/timestep/erk2/*.jl | grep -cE "^function "    # expect: 80
cat src/timestep/erk2/*.jl | grep -oE "^function [A-Za-z0-9_.!{}]+" | sort > /tmp/erk2_funcs_new.txt
diff /tmp/erk2_funcs_orig.txt /tmp/erk2_funcs_new.txt && echo "FUNC_SET_IDENTICAL"
# The 6 non-^function assignment-form one-liners must ALL have moved into the parts
# (they are NOT in the 80-count, so verify them explicitly — Task 1 moved them to common.jl):
for sig in "GeoDynamo.ERK2Cache(args...) =" "compat_solver_erk2_cache(cache::ERK2StageCache" "compat_old_erk2_cache(cache::ERK2StageCache" "GeoDynamo.erk2_diagnostics_enabled() =" "GeoDynamo.erk2_diagnostics_interval() =" "GeoDynamo.reset_phi2_monitor!() ="; do
  s=$(grep -rFl "$sig" src/timestep/erk2/ | wc -l | tr -d ' ')   # in a part file?
  e=$(grep -rFl "$sig" src/timestep/erk2.jl | wc -l | tr -d ' ') # NOT in spine?
  echo "[$sig] inParts=$s inSpine=$e"
done
```
Expected: spine has 0 functions, the parts have 80, `diff` prints nothing then `FUNC_SET_IDENTICAL`, and every one-liner shows `inParts=1 inSpine=0` (moved, not dropped, not duplicated).

- [ ] **Step 2: Confirm the aliases are present exactly once across the parts**

```bash
cat src/timestep/erk2/*.jl | grep -cE "^const ERK2Cache = |^const ERK2FieldBuffers = |^const ERK2BoundarySpec = "   # expect: 3
```
Expected: `3` (one each — not duplicated, not dropped).

- [ ] **Step 3: Proactively repoint the four static-source checks**

Four `test/*_static_checks.jl` files read `src/timestep/erk2.jl` as text and slice the bodies of `prepare_solver_erk2_field!`, `finalize_solver_erk2_field!`, and `integrate_solver_erk2_step!` — all of which moved to `erk2/integrate.jl`. (Velocity also asserts the whole-file text `theta = _timestepper_implicit_theta(params.timestepper, params)`, which is inside `integrate_solver_erk2_step!`, so it moves too.) Every slice these four read now lives in `integrate.jl`, so each file needs ONE edit: repoint its `erk2` source from `erk2.jl` to `erk2/integrate.jl`. This is a deterministic consequence of the split — do it now, do not wait for a failure.

In `test/temperature_boundary_static_checks.jl` (line 31), change:
```julia
    erk2 = _temperature_bc_static_source("src", "timestep", "erk2.jl")
```
to:
```julia
    erk2 = _temperature_bc_static_source("src", "timestep", "erk2", "integrate.jl")
```

In `test/composition_boundary_static_checks.jl` (line 31), change:
```julia
    erk2 = _composition_bc_static_source("src", "timestep", "erk2.jl")
```
to:
```julia
    erk2 = _composition_bc_static_source("src", "timestep", "erk2", "integrate.jl")
```

In `test/magnetic_boundary_static_checks.jl` (line 26), change:
```julia
    erk2 = _magnetic_bc_static_source("src", "timestep", "erk2.jl")
```
to:
```julia
    erk2 = _magnetic_bc_static_source("src", "timestep", "erk2", "integrate.jl")
```

In `test/velocity_boundary_static_checks.jl` (line 28), change:
```julia
    erk2 = _velocity_bc_static_source("src", "timestep", "erk2.jl")
```
to:
```julia
    erk2 = _velocity_bc_static_source("src", "timestep", "erk2", "integrate.jl")
```

Then confirm no other static check still reads the old monolith path:
```bash
grep -rn "timestep.*\"erk2.jl\"" test/*_static_checks.jl
```
Expected: no output (all four now read `erk2/integrate.jl`; the `_*_bc_static_source` helpers `joinpath` their varargs, so the multi-part path resolves).

- [ ] **Step 4: Run the full test suite**

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()' > /tmp/erk2_split_suite.log 2>&1; echo "EXIT=$?"
grep -E "Testing GeoDynamo tests passed|Pass  Error|Pass  Fail" /tmp/erk2_split_suite.log | tail -5
```
Expected: `EXIT=0` and `Testing GeoDynamo tests passed`. Broken count stays at the 18-baseline (GPU-skip gates). If a static check still fails, run `grep -rn "timestep.*erk2" test/*_static_checks.jl` and confirm every `erk2` source var reads `erk2/integrate.jl` (no slice reads a function that landed in a different part — per the spec mapping, all of prepare/finalize/integrate are in `integrate.jl`).

- [ ] **Step 5: Commit the static-check repoint**

```bash
git add test/temperature_boundary_static_checks.jl test/composition_boundary_static_checks.jl test/magnetic_boundary_static_checks.jl test/velocity_boundary_static_checks.jl
git commit -m "test: repoint erk2 static-check source paths after file split

prepare/finalize/integrate ERK2 bodies moved to timestep/erk2/integrate.jl;
the four boundary static checks now read that file instead of the old monolith.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Confirm net effect and finish**

```bash
git diff --stat HEAD~6..HEAD -- src/timestep/
wc -l src/timestep/erk2.jl src/timestep/erk2/*.jl
```
Expected: `erk2.jl` is ~6 lines; the five parts sum to ~2900 lines; total ≈ unchanged (pure move). The largest part (`cache.jl`) is ~900 LOC.

---

## Self-Review

**Spec coverage:** All five target files (common/boundary/cache/influence/integrate) from the spec each have an extraction task (Tasks 1–5). The spine reduction and the single-include-site invariant are in Task 5. The verification gates from the spec (load check, function-set equality, pure-move git check, full suite) are Tasks 1–5 Step "verify" + Task 6. The diagnostics-toggle placement (common.jl) matches the corrected spec. All 80 `^function` blocks are assigned exactly once: common 9, boundary 21, cache 24, influence 8, integrate 18 = 80.

**Blast radius (verified against the actual test suite):** Four `test/*_static_checks.jl` (temperature, composition, magnetic, velocity) read `src/timestep/erk2.jl` as text and slice the `prepare_solver_erk2_field!` / `finalize_solver_erk2_field!` / `integrate_solver_erk2_step!` bodies — all of which move to `integrate.jl`. Velocity additionally asserts a whole-file `_timestepper_implicit_theta` line that is inside `integrate_solver_erk2_step!`. This is the spec's documented "single include site unchanged" caveat in practice: the source call sites are untouched, but these text-reading tests are not call sites. Task 6 Step 3 repoints all four (one line each → read `erk2/integrate.jl`) proactively, since the break is deterministic. No other `src/` or `test/` file references erk2 functions by file path (confirmed via grep).

**Placeholder scan:** No TBD/TODO. Every move is specified by exact function signature; every command is concrete with expected output.

**Type consistency:** No new types or signatures are introduced — this is a relocation. Function names are quoted exactly as they appear in `grep -E "^function "` output (including the `GeoDynamo.` qualifier and `!`/`{T}` suffixes). The `get_solver_erk2_cache!` (3 methods) and `install_erk2_cache_bundle!` (2 methods) multiplicities are called out explicitly so no method is left behind. The `_get_or_build_erk2_influence_entry` exception (skip in Task 3, take in Task 4) is flagged in both tasks.
