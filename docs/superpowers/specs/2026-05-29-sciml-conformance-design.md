# SciML Style Conformance Refactor — Design

Date: 2026-05-29
Status: approved (design); implementation plan pending

## Goal

Bring GeoDynamo.jl into conformance with the [SciML Style Guide](https://docs.sciml.ai/SciMLStyle/stable/).
The codebase already follows base-Julia naming faithfully (CamelCase types,
`Abstract`-prefixed abstract types, snake_case functions, trailing `!` mutators,
SCREAMING_SNAKE value constants). Four deviations remain. This refactor closes
modules, line-length, and internal-marker fully, and addresses Unicode under a
house policy (convert script glyphs, keep conventional math notation) rather than
SciML's strict public-API ban. Every change is behavior-preserving — the test
suite (2247 pass / 0 fail) and the 2-rank equivalence tests are the contract.

## Scope (the four deviations)

1. **Unicode identifiers → selective ASCII (per-glyph house policy).** Convert
   decorative/script glyphs to ASCII; keep conventional math Unicode. This is
   *not* strict SciML on Unicode — SciML would ban Unicode from public APIs
   entirely — but it is an explicit house policy: drop the unusual script/
   blackboard glyphs, retain universally-legible math notation.
   - **CONVERT:** script field names `𝒯 𝒫 𝒟` and their `ⁱᶜ/ᵒᶜ/ᵀ/ᴾ` super/
     subscripts; plus the trivially-ASCII `ℓ → l` and `Δt → dt`.
   - **KEEP (internal and in names):** `α β η κ ν`, `θ φ`, `∇θ ∇φ`, `∂r ∂²r`,
     `r⁻¹ r⁻²`, `nₙ nₙ₋₁ uₙ`.
   - Far fewer than the full 2622 glyph sites — only the script-glyph families
     plus `ℓ`/`Δt`.
2. **CamelCase the two lowercase modules** — `bcs`, `topography`.
3. **Wrap lines to ≤ 92 chars** — ~954 lines currently exceed it.
4. **Double-underscore internal markers** — `_foo` → `__foo`.

## Non-goals

- No behavior, numerics, or algorithm changes. Pure rename + reformat.
- No restructuring of modules/files beyond the renames above.
- Math/physics meaning is preserved in names (`𝒯` → `toroidal`, not `T`).

## Execution strategy

Split by change type — they carry very different risk:

- **Renames (identifiers, modules, underscores)** are semantic. They cannot be
  applied by blind global substitution because the ASCII targets collide with
  existing names (see Phase 0). They are applied in vetted, subsystem-sized
  batches, each followed by the full suite and a commit (bisectable history).
- **Formatting (line-wrap, spacing)** is mechanical. Use `JuliaFormatter` with
  `SciMLStyle()`. Run it **last**: ASCII names are longer than glyphs
  (`toroidal` vs `𝒯`), so the renames will create new >92-char lines that the
  formatter then resolves in one pass.

Ordering: Phase 0 (audit) → Phase 1 (identifier renames) → Phase 2 (modules) →
Phase 3 (underscores) → Phase 4 (formatter). Renames before formatter.

## Phase 0 — Mapping table + collision audit (no code change)

The critical artifact. Produce an **exhaustive** glyph → ASCII mapping before any
edit. Enumerate every non-ASCII identifier:

```
rg -o '[^\x00-\x7F][A-Za-z0-9_]*|[A-Za-z0-9_]*[^\x00-\x7F]+[A-Za-z0-9_]*' src/ | sort -u
```

**CONVERT** (collision-aware — must be completed exhaustively in Phase 0):

| Glyph | ASCII | Collision / note |
|---|---|---|
| `𝒯` `𝒫` | `toroidal` `poloidal` | NOT `T`/`P` — `T` is the type parameter everywhere |
| `𝒯ⁱᶜ` `𝒫ⁱᶜ` | `toroidal_ic` `poloidal_ic` | inner-core scalars |
| `𝒟ᵒᶜ` `𝒟ⁱᶜ` | `outer_core_domain` `inner_core_domain` | |
| `nlᵀ` `nlᴾ` | `nl_toroidal` `nl_poloidal` | the `ᵀ/ᴾ` superscripts |
| `ℓ` | `l` | check loop-variable shadowing per site |
| `Δt` | `dt` | **merge** — `dt` already exists with the same meaning; reconcile per site |

Plus any other script-glyph (`𝒯/𝒫/𝒟` family) or `ⁱᶜ/ᵒᶜ/ᵀ/ᴾ`-decorated identifiers
the audit turns up.

**KEEP as Unicode** (conventional math notation — house policy, internal and in names):
`α β η κ ν`, `θ φ`, `∇θ ∇φ`, `∂r ∂²r`, `r⁻¹ r⁻²`, `nₙ nₙ₋₁ uₙ`.

Sharp edges to resolve in the audit:
- `Δt`/`dt` merge — confirm no scope holds both with different values.
- `ℓ → l` — check no shadowing of an existing `l` in the same scope.
- `θ` is KEPT, so the `θ`-vs-`matrices.theta` collision does not arise.
- Composites: only `ⁱᶜ ᵒᶜ ᵀ ᴾ` convert (on script glyphs); `ₙ ⁻¹ ⁻²` stay.
- `apply_∂r!` (exported) keeps `∂r` under the per-glyph policy → leftover public-API
  Unicode (see Open items). The clearly-public converts are the `𝒯/𝒫/nlᵀ/nlᴾ`
  fields and the `last_Δt → last_dt` kwarg.

Output: a complete frozen mapping table, reviewed before Phase 1 starts.

## Phase 1 — Identifier renames, batched by subsystem

Apply the frozen mapping. Batch order, leaf → root, suite-green + commit after each:

`numerics` → `fields` → `physics` → `timestep` → `solver` → `bcs` → `io` → `core`/`api`.

Renames cross file boundaries (e.g. `magnetic.𝒯` is read in solver + timestep),
so a "batch" is a coherent rename set (e.g. "magnetic field scalars") applied
across all its call sites at once, not a single file. Each batch must leave the
suite green before the next.

Exported names that change (breaking — see Risks): struct fields `𝒯 𝒫 nlᵀ nlᴾ`
on `SHTnsMagneticFields`/`SHTnsVelocityFields`, and kwarg `last_Δt → last_dt`.
Exported `apply_∂r!` keeps `∂r` under the per-glyph policy (Open items). Update
docs, examples, and READMEs in the same batch as the field renames.

## Phase 2 — Module renames

- `module topography` → `module Topography` (verify no `Topography` type exists).
- `module bcs` → `module Bcs` — **not** `BoundaryConditions`; that struct already
  exists and would collide.
- Update every `GeoDynamo.bcs.*` / `GeoDynamo.topography.*` reference, the
  `const` interop aliases, exports, and the `isdefined(GeoDynamo, :bcs)` tests.

## Phase 3 — Double-underscore internals

`_name` → `__name` for internal helpers and the `_Statistics` module
(`_populate_radial_operators!`, `_build_implicit_matrices_dict`,
`_magnetic_toroidal_inner_bc_increment`, `_ic_build_bic`, …). Mechanical but
must hit every definition + call site together. Suite-green + commit.

## Phase 4 — Formatter + guard

- Add `.JuliaFormatter.toml` with `style = "sciml"`.
- Run `using JuliaFormatter; format(".", SciMLStyle())`; suite-green; commit.
- Optional: a CI job running `format(...; verbose=true)` in check mode to hold
  conformance going forward.

## Verification

- **Per batch:** full suite (`julia --project=. test/runtests.jl`) must stay at
  2247 pass / 0 fail.
- **Distribution:** `test/run_mpi_cnab2_erk2_equivalence_smoke.sh` (2-rank) after
  any batch touching `timestep`/`solver`.
- **Per-batch commits** keep the (large) diff reviewable and bisectable.
- Because every change is a rename or reformat, a green suite is sufficient proof
  of behavior preservation; no new numeric assertions are required.

## Risks

1. **Breaking public API → semver-major.** Renaming exported struct fields
   (`𝒯 𝒫 nlᵀ nlᴾ`) and the `last_Δt → last_dt` kwarg breaks downstream users.
   Bump `1.0.10` → `2.0.0`; document in a changelog/migration note. Update all
   docs and examples.
2. **Partial Unicode conformance (by design).** The per-glyph policy keeps math
   notation (`α β η κ ν θ φ ∇θ ∂r r⁻¹ nₙ …`), so internal Unicode remains (SciML
   permits this) and `apply_∂r!` retains Unicode in a public name (SciML does
   not). Accepted: readability over strict conformance for these glyphs.
3. **Collisions.** The Phase 0 audit must be exhaustive; `Δt`/`dt` and `θ`
   dual-use are the primary hazards.
4. **Diff size.** Only the script-glyph families (`𝒯/𝒫/𝒟` + `ⁱᶜ/ᵒᶜ/ᵀ/ᴾ`) plus
   `ℓ`/`Δt` convert — far fewer than the 2622 total glyph sites — plus ~954 wrap
   lines handled by the formatter. Per-batch commits keep it reviewable.
5. **Concurrent edits.** The repo is actively committed to; rebase/refresh before
   each batch and rely on exact-match edits (a moved target fails loudly).

## Open items to confirm during Phase 0

- `apply_∂r!`: keep `∂r` (per-glyph policy, leftover public-API Unicode) or
  rename the exported name to `apply_dr!` while keeping internal `∂r`?
- Final `bcs` module name (`Bcs` proposed).
- Whether to ship the `2.0.0` bump + migration note as part of this work or
  separately.
- CI format-check: include now or defer.
