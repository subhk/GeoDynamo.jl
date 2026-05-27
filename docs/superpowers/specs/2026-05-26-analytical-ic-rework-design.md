# Analytical Initial-Condition Preset Rework

Date: 2026-05-26

## Goal

Make the analytical IC presets in `src/core/initial_conditions.jl` physically
clean and correct, removing crude per-mode spectral seeds.

## Scope

- **Scalar presets** (temperature, composition): keep the direct spectral
  construction but make the "blob" patterns physically meaningful radial
  features (l=0 only). Vector presets are out of scope as a transform target
  (they are not scalar fields).
- **Vector presets** (magnetic, velocity): audit and fix in the natural
  poloidal/toroidal basis.

Out of scope: `generate_random_initial_conditions!` (random per-mode seed; its
l=0 base already carries the √(4π) fix), and any physical→analysis transform
machinery (unnecessary once blobs are radial l=0).

## Conventions (verified in code)

- Scalar SH is `:orthonormal` (Y_0^0 = 1/√(4π)); a uniform physical mean `v` is
  stored as the (0,0) coefficient `v·√(4π)` (already applied to scalar presets).
- Poloidal → radial field: `B_r = l(l+1)/r · P_lm` (one power of r, see
  `src/fields/transforms.jl:590`). So a **uniform axial field needs poloidal
  P(r) ∝ r** (not r²): `B_r = 2/r · (c·r) = const`.
- Tangential components come from `SHTnsKit.synthesis_sphtor` (orthonormal). The
  exact coefficient relating the l=1 scalar to the physical amplitude is
  convention-dependent and is **calibrated by a synthesis test** rather than
  hand-derived.

## Changes

### Scalars (`set_analytical_temperature!`, `set_analytical_composition!`)
- `:hot_blob`, `:blob` → radial Gaussian shells, l=0 only:
  `value(r) = background(r) + A·exp(−½((r−r_center)/blob_width)²)`.
  Drop the l=1/l=2 (temp) and l=1 (comp) seed lines.
- Keep the `√(4π)` on the (0,0) values (set directly; no transform).
- `:conductive`, `:stratified` unchanged (already radial l=0 + √(4π)).

Consequence: blob presets no longer seed non-axisymmetric modes (use
`generate_random_initial_conditions!` for symmetry breaking).

### Vectors (`set_analytical_magnetic!`, `set_analytical_velocity!`)
- `:uniform_field` (fix the l=0-poloidal bug):
  - `:z` → poloidal **l=1, m=0**, `P(r) = c·r`, `c` calibrated so `B_z = amplitude`.
  - `:x` → poloidal **l=1, m=1**, `P(r) = c·r`.
- `:solid_rotation` (implement the stub):
  - toroidal **l=1, m=0**, `T(r) = k·r`, `k` calibrated so `v_φ = amplitude·r·sinθ`.
- `:dipole`, `:convective` unchanged.

## Tests (synthesis oracles)

- Uniform field `:z`: synthesize → `B_φ ≈ 0`, `B_r ≈ B₀cosθ`, `B_θ ≈ −B₀sinθ`,
  spatially uniform (independent of r, φ); `B₀ ≈ amplitude`.
- Solid rotation: synthesize → `v_r ≈ v_θ ≈ 0`, `v_φ ≈ amplitude·r·sinθ`.
- Scalar blob: l=0 radial profile peaks at `r_center`, reconstructs physically
  via the (0,0) → physical helper; no l>0 content.
- Ball regularity (l>0 → 0 at r=0) still holds; existing `initial_conditions.jl`
  (119 tests) stays green.

## Process

Implement in a worktree with TDD (synthesis tests as oracle for vector
coefficients); merge to main. The spec doubles as the plan given the bounded
scope (~4 functions + tests).

## Outcome (2026-05-26)

**Shipped:** the scalar half — `:hot_blob`/`:blob` are now radial Gaussian shells
(l=0), with the `√(4π)` mean preserved (`test/analytical_blob_radial.jl`, 450/450).

**Vectors deferred** — investigating them surfaced a core finding: the vector
synthesis reconstructs the radial component of a poloidal field as
`v_r = l(l+1)/r·pol`, while `pol` is recovered (in analysis) purely from the
tangential field as the spheroidal coefficient `S`. That `v_r` does not satisfy
the solenoidal constraint `(1/r²)d(r²Q)/dr = l(l+1)/r·S`, so synthesized poloidal
fields are **not divergence-free** (per-mode `∇·V = l(l+1)[S/r²+S'/r−S/r]`;
empirically `B_r/cosθ`=const but `B_θ/sinθ`∝r for `P₁₀=r`). This affects the
physical `v_r`/`B_r` feeding velocity advection and magnetic induction, and blocks
a correct `:uniform_field` at the IC layer. `:solid_rotation` (purely toroidal,
divergence-free by construction) is NOT blocked — it only needs amplitude
calibration — but is deferred with `:uniform_field` pending the reconstruction
decision. See memory `proj_geodynamo_poloidal_synthesis_nonsolenoidal`.

**Recommendation:** do not change the poloidal reconstruction without first adding
a dynamo benchmark (e.g. Christensen et al.) — it is core physics with no current
regression coverage.
