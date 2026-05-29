# SciML Style Conformance — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring GeoDynamo.jl toward SciML Style — convert script-glyph identifiers to ASCII (keeping conventional math Unicode), CamelCase the two lowercase modules, double-underscore internal helpers, and reflow to ≤92 chars — with zero behavior change.

**Architecture:** Pure rename + reformat. Every batch is verified by the existing test suite staying at **2247 pass / 0 fail** plus the 2-rank equivalence smoke; a green suite is the proof of behavior preservation. Renames are applied atomically per glyph/name (a half-renamed struct field will not compile), composites before bare glyphs, and the auto-formatter runs **last** (ASCII names are longer than glyphs and create new >92-char lines).

**Tech Stack:** Julia 1.11, `perl`/`rg` for Unicode-safe substitution, `JuliaFormatter` (`SciMLStyle`), MPI.jl for the 2-rank check.

Spec: `docs/superpowers/specs/2026-05-29-sciml-conformance-design.md`

---

## Prerequisites (read first — zero-context notes)

- **Julia launcher is broken** (`juliaup.json` is root-owned). Use the direct binary and disable the agent sandbox for every Julia/MPI command:
  ```
  JULIA=~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia
  ```
- **Full suite (the green gate):** `"$JULIA" --project=. test/runtests.jl`
  Expected tail: `Extended GeoDynamo tests | 2247  2  2249` (pass / broken / total), `✓ GeoDynamo test suite completed`. Pass count must **not drop** (renames add no tests).
- **2-rank equivalence:** `JULIA="$JULIA" test/run_mpi_cnab2_erk2_equivalence_smoke.sh` — expect exit 0, no `Test Failed`.
- **Repo is actively committed to.** `git pull`/refresh before each task; exact-match edits fail loudly if a target moved.
- **Commits:** the repo owner's standing rule is *never commit without explicit permission*. Each task ends with a commit step — get the owner's go-ahead (or have them run it) before committing.
- **Policy reminder (per-glyph):** CONVERT only the script families `𝒯 𝒫 𝒟` (+ `ⁱᶜ/ᵒᶜ/ᵀ/ᴾ` decorations), `ℓ→l`, `Δt→dt`. **KEEP** `α β η κ ν θ φ ∇θ ∇φ ∂r ∂²r r⁻¹ r⁻² nₙ nₙ₋₁ uₙ` everywhere. `apply_∂r!` keeps `∂r` (per policy).

---

## Task 0: Audit — freeze the CONVERT mapping

**Files:** none (discovery only). Produces the exact substitution list Task 1 runs.

- [ ] **Step 1: Enumerate every non-ASCII identifier and its frequency**

Run:
```bash
cd /Users/subha/Documents/GitHub/GeoDynamo.jl
rg -oN --no-filename '[A-Za-z0-9_]*[^\x00-\x7F][A-Za-z0-9_∂∇⁻¹²ᵀᴾⁱᶜᵒₙ]*' src test examples docs \
  | sort | uniq -c | sort -rn
```
Expected: a frequency list including `𝒯 𝒫 𝒯ⁱᶜ 𝒫ⁱᶜ 𝒟ᵒᶜ 𝒟ⁱᶜ nlᵀ nlᴾ ℓ Δt` (CONVERT) and `α β η κ ν θ φ ∇θ ∂r r⁻¹ nₙ …` (KEEP).

- [ ] **Step 2: Confirm the CONVERT set matches the spec; flag surprises**

Cross-check the list against the spec's CONVERT table. For any **script glyph** (`𝒯/𝒫/𝒟` family) or `ⁱᶜ/ᵒᶜ/ᵀ/ᴾ`-decorated identifier **not** already mapped (e.g. a bare `𝒟`, or `𝓑`), choose an ASCII name and append it to the Task 1 perl script. Everything in the KEEP list stays untouched.
Expected: the only additions, if any, are a handful of script-glyph composites.

- [ ] **Step 3: Record the `Δt`/`dt` and `ℓ`/`l` co-occurrence sites**

Run:
```bash
rg -n 'Δt' src | rg '\bdt\b'      # scopes holding BOTH Δt and dt
rg -n '\bℓ\b' src | rg '\bl\b'    # scopes holding BOTH ℓ and l
```
Expected: list (likely empty or few). These are the only sites needing a human glance after Task 1's global replace (same concept → merge is safe; confirm no scope uses `dt`/`l` for something else).

No commit (no file change).

---

## Task 1: Rename script-glyph identifiers → ASCII (atomic)

**Files:** Modify every file containing the glyphs (across `src/ test/ examples/ docs/`). A struct-field rename must hit all call sites in one commit or the package won't load.

- [ ] **Step 1: Apply the ordered substitution (composites before bare glyphs)**

Run (the `-CSD` flag makes perl UTF-8-clean; order matters — `𝒯ⁱᶜ` before `𝒯`):
```bash
cd /Users/subha/Documents/GitHub/GeoDynamo.jl
FILES=$(rg -l -e '𝒯' -e '𝒫' -e '𝒟' -e 'nlᵀ' -e 'nlᴾ' src test examples docs)
perl -CSD -i -pe '
  s/𝒯ⁱᶜ/toroidal_ic/g; s/𝒫ⁱᶜ/poloidal_ic/g;
  s/𝒟ᵒᶜ/outer_core_domain/g; s/𝒟ⁱᶜ/inner_core_domain/g;
  s/nlᵀ/nl_toroidal/g; s/nlᴾ/nl_poloidal/g;
  s/𝒯/toroidal/g; s/𝒫/poloidal/g;
' $FILES
```
(Append any Task 0 Step 2 additions to this script before running.)

- [ ] **Step 2: Verify no script glyphs remain**

Run: `rg -n '𝒯|𝒫|𝒟|nlᵀ|nlᴾ' src test examples docs`
Expected: no output.

- [ ] **Step 3: Run the suite**

Run: `"$JULIA" --project=. test/runtests.jl 2>&1 | tail -3`
Expected: `Extended GeoDynamo tests | 2247  2  2249`, `✓ GeoDynamo test suite completed`.

- [ ] **Step 4: Run the 2-rank equivalence (timestep/solver fields touched)**

Run: `JULIA="$JULIA" test/run_mpi_cnab2_erk2_equivalence_smoke.sh 2>&1 | tail -3; echo "exit=$?"`
Expected: `exit=0`, no `Test Failed`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(style): ASCII-ize script-glyph identifiers (𝒯/𝒫/𝒟→toroidal/poloidal/…)

BREAKING: exported fields SHTns{Magnetic,Velocity}Fields.𝒯/𝒫/nlᵀ/nlᴾ renamed.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `ℓ → l`

**Files:** every file containing `ℓ` (`src/ test/ examples/ docs/`).

- [ ] **Step 1: Replace**

```bash
FILES=$(rg -l 'ℓ' src test examples docs)
perl -CSD -i -pe 's/ℓ/l/g' $FILES
rg -n 'ℓ' src test examples docs   # expect: no output
```

- [ ] **Step 2: Eyeball the co-occurrence sites from Task 0 Step 3**

For each site that previously held both `ℓ` and `l`, confirm they were the same harmonic-degree concept (merge is correct). Expected: no functional change.

- [ ] **Step 3: Suite** — `"$JULIA" --project=. test/runtests.jl 2>&1 | tail -3` → `2247  2  2249`.

- [ ] **Step 4: Commit**
```bash
git add -A && git commit -m "refactor(style): rename harmonic-degree ℓ → l

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `Δt → dt` (merges with existing `dt`)

**Files:** every file containing `Δt` (covers the `last_Δt → last_dt` kwarg automatically).

- [ ] **Step 1: Replace**

```bash
FILES=$(rg -l 'Δt' src test examples docs)
perl -CSD -i -pe 's/Δt/dt/g' $FILES
rg -n 'Δt' src test examples docs   # expect: no output
```

- [ ] **Step 2: Eyeball the co-occurrence sites from Task 0 Step 3**

Confirm each scope that held both `Δt` and `dt` used them for the same timestep value (merge correct). Watch for a now-duplicated keyword like `dt = dt` in a constructor — fix to a single binding if it appears.

- [ ] **Step 3: Suite** → `2247  2  2249`.
- [ ] **Step 4: 2-rank** — `JULIA="$JULIA" test/run_mpi_cnab2_erk2_equivalence_smoke.sh; echo exit=$?` → `exit=0`.
- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "refactor(style): rename timestep Δt → dt (BREAKING: last_Δt kwarg → last_dt)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Module `topography → Topography`

**Files:** Modify `src/bcs/topography/topography.jl` (decl), `src/solver/interop.jl` (`TopographyAPI` alias), every `*.topography` reference, exports, tests.

- [ ] **Step 1: Find all reference forms**

Run: `rg -n 'module topography|\.topography\b|:topography\b|\btopography\.' src test`
Expected: the `module topography` line, `getproperty(..., :topography)`, `GeoDynamo.bcs.topography`/`.topography.` call sites, any `isdefined(..., :topography)` test.

- [ ] **Step 2: Replace (qualified forms only — avoid matching the word in prose/paths)**

```bash
FILES=$(rg -l 'module topography|\.topography\b|:topography\b|\btopography\.' src test)
perl -i -pe '
  s/\bmodule topography\b/module Topography/g;
  s/\.topography\b/.Topography/g;
  s/:topography\b/:Topography/g;
  s/\btopography\./Topography./g;
' $FILES
```
Then verify the module is reachable: `rg -n 'module Topography' src/bcs/topography/topography.jl`.

- [ ] **Step 3: Suite** → `2247  2  2249`. (If a test asserted `isdefined(GeoDynamo.bcs, :topography)`, update it to `:Topography` and re-run.)

- [ ] **Step 4: Commit**
```bash
git add -A && git commit -m "refactor(style): CamelCase module topography → Topography

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Module `bcs → Bcs`

**Files:** `src/bcs/bcs.jl` (decl), `src/solver/interop.jl` (`getproperty(GeoDynamo, :bcs)`), all `GeoDynamo.bcs` / `..bcs` / `:bcs` refs, exports in `src/GeoDynamo.jl`, the `isdefined(GeoDynamo, :bcs)` + `isdefined(GeoDynamo.bcs, …)` tests. **Not** `BoundaryConditions` — that struct exists.

- [ ] **Step 1: Inventory reference forms**

Run: `rg -n 'module bcs\b|\bbcs\b' src test | rg -v '#'`
Note the forms: `module bcs`, `GeoDynamo.bcs`, `..bcs`/`import ..bcs`, `:bcs`, `bcs.`, `const BC_Ball = GeoDynamo.bcs`. Confirm `bcs` never appears as a substring of another identifier (it does not — it is always the module).

- [ ] **Step 2: Replace whole-word `bcs` → `Bcs`**

```bash
FILES=$(rg -l '\bbcs\b' src test)
perl -i -pe 's/\bbcs\b/Bcs/g' $FILES
rg -n '\bbcs\b' src test   # expect: no output (only Bcs remains)
```

- [ ] **Step 3: Suite** → `2247  2  2249`. (Update any `:bcs` symbol test to `:Bcs`.)

- [ ] **Step 4: Commit**
```bash
git add -A && git commit -m "refactor(style): CamelCase module bcs → Bcs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Double-underscore internal helpers `_x → __x`

**Files:** every file defining/calling a single-underscore internal (`src/`, plus `test/` call sites).

- [ ] **Step 1: Enumerate single-underscore identifiers**

Run:
```bash
rg -oN --no-filename '\b_[a-zA-Z][A-Za-z0-9_]*!?' src | sort -u > /tmp/underscore_names.txt
rg -oN --no-filename '\b_[A-Z][A-Za-z0-9_]*' src | sort -u   # modules like _Statistics
cat /tmp/underscore_names.txt
```
Expected: `_populate_radial_operators!`, `_build_implicit_matrices_dict`, `_magnetic_toroidal_inner_bc_increment`, `_ic_build_bic`, `_scalar_*`, `_solver_*`, `_get_*`, `_Statistics`, … Exclude anything that is a field of a struct accessed as `x._foo` only if it is genuinely external (none expected here — all are package-internal).

- [ ] **Step 2: Replace each enumerated name (def + all call sites) with a leading second underscore**

```bash
# For each NAME in the enumerated list (longest first to avoid prefix clashes):
FILES=$(rg -l "\b${NAME}\b" src test)
perl -i -pe "s/\b\Q${NAME}\E\b/_${NAME}/g" $FILES   # _foo -> __foo
```
Run once per name (script it over `/tmp/underscore_names.txt`, sorted longest-first). Then:
`rg -n '\b_[a-zA-Z][A-Za-z0-9_]*' src | rg -v '__' | rg -v '#'` → expect: no single-underscore identifiers remain (besides `_` placeholders in `for _ in`).

- [ ] **Step 3: Suite** → `2247  2  2249`.
- [ ] **Step 4: 2-rank** — `JULIA="$JULIA" test/run_mpi_cnab2_erk2_equivalence_smoke.sh; echo exit=$?` → `exit=0`.
- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "refactor(style): double-underscore internal helpers (_x → __x)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Auto-format to SciML style (line length + spacing)

**Files:** Create `.JuliaFormatter.toml`; reformat all `src/`, `test/`.

- [ ] **Step 1: Add formatter config**

Create `/Users/subha/Documents/GitHub/GeoDynamo.jl/.JuliaFormatter.toml`:
```toml
style = "sciml"
```

- [ ] **Step 2: Run the formatter (in a throwaway env — JuliaFormatter is a dev tool, not a package dep)**

```bash
"$JULIA" -e 'using Pkg; Pkg.activate(temp=true); Pkg.add("JuliaFormatter"); using JuliaFormatter; format("/Users/subha/Documents/GitHub/GeoDynamo.jl"; verbose=true)'
```
Expected: prints formatted files; exit 0.

- [ ] **Step 3: Check residual long lines**

Run: `for f in $(rg -l '' -g '*.jl' src); do awk 'length>92{n++}END{if(n)print n,FILENAME}' "$f"; done`
Expected: near-zero. Any survivors are usually long string literals/URLs — leave them.

- [ ] **Step 4: Suite** → `2247  2  2249` (formatting must not change behavior).
- [ ] **Step 5: 2-rank** — `JULIA="$JULIA" test/run_mpi_cnab2_erk2_equivalence_smoke.sh; echo exit=$?` → `exit=0`.
- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "style: apply JuliaFormatter SciMLStyle (≤92 cols, spacing)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Version bump + migration note (BREAKING)

**Files:** Modify `Project.toml`; create `CHANGELOG.md` (or append).

- [ ] **Step 1: Bump version**

In `Project.toml`, change `version = "1.0.10"` → `version = "2.0.0"`.

- [ ] **Step 2: Write the migration note**

Create/append `CHANGELOG.md`:
```markdown
## 2.0.0

BREAKING — public identifiers ASCII-ized:
- `SHTnsMagneticFields` / `SHTnsVelocityFields` fields: `𝒯→toroidal`, `𝒫→poloidal`,
  `nlᵀ→nl_toroidal`, `nlᴾ→nl_poloidal`; inner-core `𝒯ⁱᶜ→toroidal_ic`, `𝒫ⁱᶜ→poloidal_ic`.
- Domain accessors `𝒟ᵒᶜ→outer_core_domain`, `𝒟ⁱᶜ→inner_core_domain`.
- Keyword `last_Δt → last_dt`.
- Modules `bcs → Bcs`, `topography → Topography`.

Conventional math Unicode (`α β η κ ν θ φ ∇ ∂r r⁻¹ nₙ`) is unchanged.
```

- [ ] **Step 3: Suite** → `2247  2  2249` (sanity; Project.toml change shouldn't affect it).
- [ ] **Step 4: Commit**
```bash
git add Project.toml CHANGELOG.md
git commit -m "release: 2.0.0 — ASCII public identifiers + CamelCase modules (BREAKING)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Deferred / open (decide before or during execution)

- **`apply_∂r!`** stays as-is per the per-glyph policy (Unicode in one exported name). If strict public-API ASCII is wanted later: rename the *exported* name to `apply_dr!` (keep internal `∂r`) via `s/\bapply_∂r!/apply_dr!/g` + export update — its own task.
- **CI format-check** (hold conformance): add a GitHub Action running `format(...; verbose=true)` in check mode. Not required for this plan.

## Notes on verification philosophy

Every task is a rename or reformat — no new behavior. The 2247-green suite + the 2-rank equivalence smoke are the characterization; if they stay green, behavior is preserved. There is therefore no per-task "write a failing test" step; the failing-state to avoid is a red suite, checked after every batch.
