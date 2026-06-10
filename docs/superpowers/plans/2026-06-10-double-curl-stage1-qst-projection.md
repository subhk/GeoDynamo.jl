# Double-Curl Stage 1: QST Force Analysis + Reference-Verified Projections

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 3-component (QST) spectral analysis for force fields plus the two momentum-equation projections `[r̂·∇×F]_lm` and `[r̂·∇×∇×F]_lm`, each verified against an independent finite-difference reference — WITHOUT wiring anything into the dynamics yet.

**Architecture:** Stage 1 of `docs/superpowers/specs/2026-06-10-poloidal-momentum-double-curl-design.md`. One new src file (`physics/force_projection.jl`) holding `force_physical_to_qst!` (Q from the existing scalar analysis of the radial component; S,T from the existing tangential sphtor analysis) and `force_curl_projections!` (per-mode radial-profile formulas using the banded D_r operator). One new test file with an independent physical-space curl reference (banded D_r for radial derivatives — pure radial calculus, independent of the sphtor identities under test — and the code's scalar gradient machinery for angular derivatives). The dynamics (`finish_velocity_nonlinear!`, implicit updates) are NOT touched.

**Tech Stack:** Julia 1.11, SHTnsKit, PencilArrays. Run all commands with `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=.` from the repo root. Single-rank (MPI initialized but np=1).

**⚠️ Candidate identities (the tests are the arbiter, NOT the formulas):**
With Q_lm = scalar analysis of F_r, and (S_lm, T_lm) = the sphtor analysis of (F_θ, F_φ) exactly as `vector_physical_to_spectral!` produces them today (S lands in its "poloidal" output, T in "toroidal"), λ = l(l+1):

    R_tor[lm](r) = −(λ/r) · T_lm(r)                       (candidate A)
    R_pol[lm](r) = (λ/r²) · ( Q_lm(r) − ∂_r(r·S_lm(r)) )  (candidate B)

Sign, and the power of r, are the plausible failure points (SHTns unit-sphere
∇₁ conventions). If a test fails ONLY by a constant factor/sign across all
modes and radii, fix the formula constant to the (small-integer / sign)
correction the reference demands and re-derive on paper to confirm it is
convention bookkeeping — never fit non-rational factors; a non-constant
mismatch means the structure (which derivative, which scalar) is wrong and the
derivation must be redone.

---

## File Structure

- **Create** `src/physics/force_projection.jl` — QST analysis + curl projections for force fields. Nothing else.
- **Modify** `src/solver.jl` — add `include("physics/force_projection.jl")` after `include("physics/nonlinear.jl")` (it reuses `scalar_physical_to_spectral!`/`vector_physical_to_spectral!` defined there; all calls resolve at call time, but keeping the include adjacent documents the dependency).
- **Create** `test/force_projection_reference.jl` — independent-reference verification tests.
- **Modify** `test/runtests.jl` — register the new test file in the `additional_tests` list.

## Shared test fixture (used by every test in this plan)

Small serial config, exactly the pattern of `test/ball_roundtrip.jl` but shell:

```julia
using Test
using MPI
using LinearAlgebra
using Random

MPI.Initialized() || MPI.Init()

const FP_NR = 48     # fine enough for FD reference accuracy
const FP_LMAX = 8

function _fp_setup()
    cfg = GeoDynamo.create_shtnskit_config(
        lmax = FP_LMAX, mmax = FP_LMAX, nlat = 2 * FP_LMAX + 4, nlon = 4 * FP_LMAX + 8,
        nr = FP_NR)
    dom = GeoDynamo.create_radial_domain(FP_NR)   # shell domain (default radius ratio)
    return cfg, dom
end

_fp_spec(cfg, dom) = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
_fp_vec(cfg, dom) = GeoDynamo.create_shtns_vector_field(Float64, cfg, dom,
    (cfg.pencils.θ, cfg.pencils.φ, cfg.pencils.r))
_fp_phys(cfg, dom) = GeoDynamo.create_shtns_physical_field(Float64, cfg, dom, cfg.pencils.r)
```

(If `create_radial_domain(FP_NR)` does not exist under that name, the
constructor used by the solver is reachable via
`GeoDynamo.create_shtns_radial_domain` — check `grep -rn "function create_.*radial_domain" src/` and use the shell constructor the solver itself uses; `test/analytical_blob_radial.jl` line ~9 shows the working pattern.)

---

### Task 1: Independent curl reference harness (test-only code)

**Files:**
- Create: `test/force_projection_reference.jl` (harness part)

The reference computes `r̂·∇×F` and `r̂·∇×∇×F` ON THE GRID using only:
(a) the banded radial derivative `create_derivative_matrix(Float64, 1, dom)` —
pure radial calculus, shares nothing with the sphtor identities under test;
(b) angular derivatives obtained by scalar spectral differentiation of grid
functions (`scalar_physical_to_spectral!` → the per-mode recurrence in
`compute_theta_gradient_spectral!`/`compute_phi_gradient_spectral!` →
`scalar_spectral_to_physical!`) — convention-independent facts about grid
functions, also not the identities under test.

- [ ] **Step 1: Write the harness**

Append to the fixture in `test/force_projection_reference.jl`:

```julia
# ---- Independent reference machinery ----------------------------------------
# Angular derivative of a physical grid function g(θ,φ;r):
# returns (∂θ g, ∂φ g) as new physical fields, via the scalar gradient
# machinery (spectral recurrences — independent of sphtor vector identities).
function _fp_angular_derivs(cfg, dom, g_phys)
    g_spec = _fp_spec(cfg, dom)
    GeoDynamo.scalar_physical_to_spectral!(g_phys, g_spec)
    # Scalar gradient workspace exactly as the solver builds it:
    ws = GeoDynamo.create_solver_gradient_workspace(Float64, cfg, dom)
    # Wrap the spectral coefficients in a scalar-field-shaped container by
    # writing them into a temperature-field clone is heavyweight; instead call
    # the low-level per-mode θ/φ gradient kernels through a thin scalar field:
    # the solver gradient API takes an AbstractScalarField; build one:
    tf = GeoDynamo.create_shtns_temperature_field(Float64, cfg, dom)
    copyto!(parent(tf.spectral.data_real), parent(g_spec.data_real))
    copyto!(parent(tf.spectral.data_imag), parent(g_spec.data_imag))
    GeoDynamo.compute_all_gradients_spectral!(tf, dom, ws)
    dθ = _fp_phys(cfg, dom); dφ = _fp_phys(cfg, dom)
    GeoDynamo.scalar_spectral_to_physical!(ws.∇θ_spec, dθ)
    GeoDynamo.scalar_spectral_to_physical!(ws.∇φ_spec, dφ)
    return dθ, dφ   # NOTE: these are the PHYSICAL gradient components,
                    # i.e. (1/r)∂θ g and (1/(r sinθ))∂φ g — verify which
                    # scaling the solver's gradient uses by the calibration
                    # test in Step 3 and adapt the curl formulas below.
end

# Radial derivative of per-(θ,φ) profiles: apply banded D_r to each radial
# pencil of a physical field. dom-local: serial run, r fully local.
function _fp_radial_deriv(cfg, dom, g_phys)
    D1 = GeoDynamo.create_derivative_matrix(Float64, 1, dom)
    g = parent(g_phys.data)
    out_phys = _fp_phys(cfg, dom)
    out = parent(out_phys.data)
    nlat, nlon, nr = size(g)
    prof = zeros(nr); dprof = zeros(nr)
    for j in 1:nlon, i in 1:nlat
        @views prof .= g[i, j, :]
        GeoDynamo.LinearAlgebra.mul!(dprof, D1, prof)
        @views out[i, j, :] .= dprof
    end
    return out_phys
end

# r̂·∇×F on the grid:  (1/(r sinθ))[∂θ(sinθ F_φ) − ∂φ F_θ]
# Computed from PHYSICAL components with the helpers above. The sinθ values
# come from cfg.theta_grid (colatitude). Returns a physical field.
function _fp_radial_curl(cfg, dom, F)
    nlat, nlon = cfg.nlat, cfg.nlon
    sinθ = sin.(cfg.theta_grid)
    # sinθ·F_φ as a grid function
    sFφ = _fp_phys(cfg, dom)
    a = parent(sFφ.data); fφ = parent(F.φ_component.data)
    for k in axes(a, 3), j in 1:nlon, i in 1:nlat
        a[i, j, k] = sinθ[i] * fφ[i, j, k]
    end
    dθ_sFφ, _ = _fp_angular_derivs(cfg, dom, sFφ)
    fθ_phys = _fp_phys(cfg, dom)
    copyto!(parent(fθ_phys.data), parent(F.θ_component.data))
    _, dφ_Fθ = _fp_angular_derivs(cfg, dom, fθ_phys)
    out = _fp_phys(cfg, dom); o = parent(out.data)
    dA = parent(dθ_sFφ.data); dB = parent(dφ_Fθ.data)
    r_range = GeoDynamo.range_local(cfg.pencils.r, 3)
    for k in axes(o, 3)
        r_idx = k + first(r_range) - 1
        r = dom.r[min(r_idx, dom.N), 4]
        for j in 1:nlon, i in 1:nlat
            # If _fp_angular_derivs already includes the 1/r and 1/sinθ
            # physical scalings (calibration test, Step 3), the prefactors
            # here reduce accordingly — the calibration test pins this.
            o[i, j, k] = (dA[i, j, k] / sinθ[i] - dB[i, j, k]) / r
        end
    end
    return out
end

# Full ∇×F (all three components), so the reference can apply it twice for
# the double curl. Components in spherical coordinates:
#   (∇×F)_r = (1/(r sinθ))[∂θ(sinθFφ) − ∂φFθ]
#   (∇×F)_θ = (1/(r sinθ))∂φFr − (1/r)∂r(rFφ)
#   (∇×F)_φ = (1/r)∂r(rFθ) − (1/r)∂θFr
function _fp_curl(cfg, dom, F)
    G = _fp_vec(cfg, dom)
    # ... assemble the three components with _fp_angular_derivs +
    # _fp_radial_deriv following the formulas above; ~40 lines of the same
    # index loops as _fp_radial_curl. The implementer writes these loops in
    # full; each component mirrors _fp_radial_curl's structure (multiply by
    # r/sinθ grids, differentiate, combine, divide by r).
    return G
end
```

The `_fp_curl` body must be written out fully (the plan shows the exact
component formulas; the loops are mechanical copies of `_fp_radial_curl`'s
pattern). `r·F_θ`-type products are formed the same way as `sinθ·F_φ`.

- [ ] **Step 2: Calibration test — pin the gradient helper's scaling**

The harness's one convention unknown is whether the solver's scalar gradient
returns ∂θg or (1/r)∂θg. Pin it with a function whose gradient is known:
g = r·cos(θ) (= z): physical ∂θ-component of ∇g is −sin(θ)·1 (the
θ-component of ẑ), and (1/r)∂θ(g) = −sinθ as well — degenerate; use
g = r²cosθ instead: (1/r)∂θ g = −r·sinθ while ∂θ g = −r²sinθ.

```julia
@testset "calibration: scalar gradient scaling" begin
    cfg, dom = _fp_setup()
    g = _fp_phys(cfg, dom)
    arr = parent(g.data)
    r_range = GeoDynamo.range_local(cfg.pencils.r, 3)
    for k in axes(arr, 3), j in 1:cfg.nlon, i in 1:cfg.nlat
        r = dom.r[min(k + first(r_range) - 1, dom.N), 4]
        arr[i, j, k] = r^2 * cos(cfg.theta_grid[i])
    end
    dθ, _ = _fp_angular_derivs(cfg, dom, g)
    a = parent(dθ.data)
    i, j, k = 3, 4, div(FP_NR, 2)
    r = dom.r[k + first(r_range) - 1, 4]
    ratio = a[i, j, k] / (-sin(cfg.theta_grid[i]))
    @info "gradient scaling ratio (r²⇒physical-(1/r): r; bare ∂θ: r²)" ratio r r^2
    @test isapprox(ratio, r; rtol = 1e-6) || isapprox(ratio, r^2; rtol = 1e-6)
end
```

Record which scaling holds and adjust `_fp_radial_curl`/`_fp_curl` prefactors
accordingly (delete the redundant 1/r if the helper already applies it).

- [ ] **Step 3: Run the calibration standalone**

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using GeoDynamo; using Test; include("test/force_projection_reference.jl")' 2>&1 | tail -5
```
Expected: calibration testset passes and the @info line reports the ratio.
(The later testsets don't exist yet — only the calibration block is in the
file at this point.)

- [ ] **Step 4: Commit**

```bash
git add test/force_projection_reference.jl
git commit -m "test: independent curl reference harness for force projections (stage 1)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `force_physical_to_qst!` (RED → GREEN)

**Files:**
- Create: `src/physics/force_projection.jl`
- Modify: `src/solver.jl` (add include)
- Test: `test/force_projection_reference.jl` (append testset)

- [ ] **Step 1: Write the failing test**

A field with KNOWN (Q,S,T) content: synthesize F_r from a chosen Q via the
scalar synthesis, and (F_θ,F_φ) from chosen (S,T) via the existing tangential
sphtor synthesis — then `force_physical_to_qst!` must recover all three (the
tangential pair round-trips by the existing analysis; Q round-trips by the
scalar pair). Append:

```julia
@testset "force_physical_to_qst! recovers Q, S, T" begin
    cfg, dom = _fp_setup()
    Random.seed!(7)
    # build known spectral content
    Qin = _fp_spec(cfg, dom); Sin = _fp_spec(cfg, dom); Tin = _fp_spec(cfg, dom)
    for spec in (Qin, Sin, Tin)
        sr = parent(spec.data_real); si = parent(spec.data_imag)
        for lm in 1:cfg.nlm
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            l = cfg.l_values[lm]; m = cfg.m_values[lm]
            (1 <= l <= FP_LMAX - 2) || continue   # keep band-limited margin
            for r_idx in 1:dom.N
                x = (dom.r[r_idx, 4] - dom.r[1, 4]) / (dom.r[dom.N, 4] - dom.r[1, 4])
                v = sinpi(x) * randn() * 1e-2
                GeoDynamo.set_local_spectral_value!(sr, slot, r_idx, v)
                m > 0 && GeoDynamo.set_local_spectral_value!(si, slot, r_idx, 0.7v)
            end
        end
    end
    # synthesize physical F: tangential from (S,T) via existing sphtor synthesis,
    # radial from Q via scalar synthesis
    F = _fp_vec(cfg, dom)
    GeoDynamo.vector_spectral_to_physical!(Tin, Sin, F)   # writes θ,φ (and an r we overwrite)
    Fr_phys = _fp_phys(cfg, dom)
    GeoDynamo.scalar_spectral_to_physical!(Qin, Fr_phys)
    copyto!(parent(F.r_component.data), parent(Fr_phys.data))

    Q = _fp_spec(cfg, dom); S = _fp_spec(cfg, dom); T_ = _fp_spec(cfg, dom)
    GeoDynamo.force_physical_to_qst!(F, Q, S, T_)

    for (out, ref, name) in ((Q, Qin, "Q"), (S, Sin, "S"), (T_, Tin, "T"))
        a = vcat(vec(parent(out.data_real)), vec(parent(out.data_imag)))
        b = vcat(vec(parent(ref.data_real)), vec(parent(ref.data_imag)))
        @test isapprox(a, b; rtol = 1e-8, atol = 1e-12)
    end
end
```

NOTE: `vector_spectral_to_physical!(Tin, Sin, F)` argument order is
(toroidal, poloidal, vec) per `src/solver/numerics.jl:827`, and "poloidal"
slot IS the spheroidal scalar in the current convention. It also synthesizes
an r-component (l(l+1)S/r with `domain=nothing` it zero-fills) — the test
overwrites F_r from Q regardless.

- [ ] **Step 2: Run — expect FAIL with UndefVarError force_physical_to_qst!**

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using GeoDynamo; using Test; include("test/force_projection_reference.jl")' 2>&1 | tail -6
```

- [ ] **Step 3: Implement**

Create `src/physics/force_projection.jl`:

```julia
# ================================================================================
# Force-field QST analysis and momentum-equation curl projections (Stage 1 of the
# double-curl poloidal formulation; see
# docs/superpowers/specs/2026-06-10-poloidal-momentum-double-curl-design.md).
# NOT yet wired into the dynamics — verified standalone against an independent
# finite-difference curl reference in test/force_projection_reference.jl.
# ================================================================================

"""
    force_physical_to_qst!(force, Q, S, T)

Three-component spectral analysis of a (generally non-solenoidal) force field.
`Q` ← scalar analysis of the radial component; `(S, T)` ← spheroidal/toroidal
scalars of the tangential components (the same sphtor analysis the velocity
path uses — S is what that path stores as "poloidal").
"""
function force_physical_to_qst!(
        force::VectorFieldType{T},
        Q::SpectralFieldType{T},
        S::SpectralFieldType{T},
        T_out::SpectralFieldType{T}
) where {T}
    # Radial component → Q via the existing scalar analysis. The r_component
    # is a physical field on the same pencil; copy into a scalar physical
    # container to match scalar_physical_to_spectral!'s argument type.
    _force_r_scratch(Q.config, force) do fr_phys
        scalar_physical_to_spectral!(fr_phys, Q)
    end
    # Tangential components → (S, T) via the existing 2-component analysis
    # (argument order: vector, toroidal-out, poloidal-out; "poloidal" = S).
    vector_physical_to_spectral!(force, T_out, S)
    return Q, S, T_out
end

# Scratch physical container for the radial component (config-cached; built
# lazily the same way other per-config scratch is).
function _force_r_scratch(f::Function, config, force)
    fr = create_shtns_physical_field(
        eltype(parent(force.r_component.data)), config,
        nothing === nothing ? force.r_component.domain : nothing,  # see note
        config.pencils.r)
    copyto!(parent(fr.data), parent(force.r_component.data))
    return f(fr)
end
```

⚠️ Implementer note on `_force_r_scratch`: the goal is a `PhysicalFieldType`
wrapper around a copy of `force.r_component`'s data. Check the actual
constructor signature (`grep -n "function create_shtns_physical_field" src/`)
and the component's `domain` accessor; if `r_component` is ALREADY a
`PhysicalFieldType` (likely — same struct family), skip the copy entirely and
pass it straight to `scalar_physical_to_spectral!`, deleting the scratch
helper. Allocation cleanliness is NOT a Stage-1 goal; correctness is.

In `src/solver.jl`, after `include("physics/nonlinear.jl")` add:

```julia
include("physics/force_projection.jl")
```

- [ ] **Step 4: Run — expect PASS (all three recovered)**

Same command as Step 2. All `force_physical_to_qst!` asserts pass; calibration
still green.

- [ ] **Step 5: Commit**

```bash
git add src/physics/force_projection.jl src/solver.jl test/force_projection_reference.jl
git commit -m "feat(physics): QST analysis for force fields (stage 1, not yet wired)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Curl projections + reference verification (RED → GREEN)

**Files:**
- Modify: `src/physics/force_projection.jl`
- Test: `test/force_projection_reference.jl` (append testsets)

- [ ] **Step 1: Write the failing tests**

Three families. (1) NULL: gradient field ⇒ both projections vanish.
(2) BUOYANCY-LIKE: pure radial F ⇒ R_tor = 0 and R_pol matches the FD
reference (this is the identity that un-breaks convection). (3) GENERIC:
random QST mix ⇒ both projections match the FD reference.

```julia
# Project a physical scalar (the FD reference's radial-curl output) into
# spectral per-mode profiles for comparison:
function _fp_project(cfg, dom, phys)
    spec = _fp_spec(cfg, dom)
    GeoDynamo.scalar_physical_to_spectral!(phys, spec)
    return vcat(vec(parent(spec.data_real)), vec(parent(spec.data_imag)))
end

function _fp_random_force(cfg, dom; radial_only::Bool = false)
    Qin = _fp_spec(cfg, dom); Sin = _fp_spec(cfg, dom); Tin = _fp_spec(cfg, dom)
    # fill as in the Task-2 test (same loop, same band-limit margin), with
    # Sin/Tin left zero when radial_only=true
    # ... [identical fill loop to Task 2 — reproduce it here in full] ...
    F = _fp_vec(cfg, dom)
    GeoDynamo.vector_spectral_to_physical!(Tin, Sin, F)
    Fr = _fp_phys(cfg, dom)
    GeoDynamo.scalar_spectral_to_physical!(Qin, Fr)
    copyto!(parent(F.r_component.data), parent(Fr.data))
    return F
end

@testset "curl projections: gradient field is a double null" begin
    cfg, dom = _fp_setup()
    # F = ∇(g) with g a smooth random scalar: build g spectral, get physical
    # gradient components via the solver's scalar-gradient machinery
    # (the same _fp_angular_derivs path + compute_radial_gradient_spectral!),
    # assemble F from them. Then both projections must vanish.
    # ... build F = ∇g ...
    Q = _fp_spec(cfg, dom); S = _fp_spec(cfg, dom); T_ = _fp_spec(cfg, dom)
    GeoDynamo.force_physical_to_qst!(F, Q, S, T_)
    Rtor = _fp_spec(cfg, dom); Rpol = _fp_spec(cfg, dom)
    GeoDynamo.force_curl_projections!(Rtor, Rpol, Q, S, T_, dom)
    @test norm(parent(Rtor.data_real)) + norm(parent(Rtor.data_imag)) < 1e-6
    @test norm(parent(Rpol.data_real)) + norm(parent(Rpol.data_imag)) < 1e-6
end

@testset "curl projections: pure radial (buoyancy-like) force" begin
    cfg, dom = _fp_setup()
    Random.seed!(11)
    F = _fp_random_force(cfg, dom; radial_only = true)
    Q = _fp_spec(cfg, dom); S = _fp_spec(cfg, dom); T_ = _fp_spec(cfg, dom)
    GeoDynamo.force_physical_to_qst!(F, Q, S, T_)
    Rtor = _fp_spec(cfg, dom); Rpol = _fp_spec(cfg, dom)
    GeoDynamo.force_curl_projections!(Rtor, Rpol, Q, S, T_, dom)
    # toroidal projection of a radial field vanishes
    @test norm(parent(Rtor.data_real)) + norm(parent(Rtor.data_imag)) < 1e-6
    # poloidal projection: compare against FD double-curl reference
    G = _fp_curl(cfg, dom, F)            # ∇×F
    rr = _fp_radial_curl(cfg, dom, G)    # r̂·∇×∇×F on the grid
    ref = _fp_project(cfg, dom, rr)
    got = vcat(vec(parent(Rpol.data_real)), vec(parent(Rpol.data_imag)))
    # FD radial derivative limits accuracy; interior-mode agreement to ~1e-4
    # relative is the gate (banded D_r is spectral-accuracy on this grid, so
    # expect far better; loosen only with justification)
    @test isapprox(got, ref; rtol = 1e-4, atol = 1e-10 * max(norm(ref), 1.0))
    @test norm(got) > 1e-8   # NOT trivially zero — buoyancy must project
end

@testset "curl projections: generic force vs reference" begin
    cfg, dom = _fp_setup()
    Random.seed!(13)
    F = _fp_random_force(cfg, dom)
    Q = _fp_spec(cfg, dom); S = _fp_spec(cfg, dom); T_ = _fp_spec(cfg, dom)
    GeoDynamo.force_physical_to_qst!(F, Q, S, T_)
    Rtor = _fp_spec(cfg, dom); Rpol = _fp_spec(cfg, dom)
    GeoDynamo.force_curl_projections!(Rtor, Rpol, Q, S, T_, dom)
    rc = _fp_radial_curl(cfg, dom, F)
    @test isapprox(
        vcat(vec(parent(Rtor.data_real)), vec(parent(Rtor.data_imag))),
        _fp_project(cfg, dom, rc); rtol = 1e-4, atol = 1e-10)
    G = _fp_curl(cfg, dom, F)
    rr = _fp_radial_curl(cfg, dom, G)
    @test isapprox(
        vcat(vec(parent(Rpol.data_real)), vec(parent(Rpol.data_imag))),
        _fp_project(cfg, dom, rr); rtol = 1e-4, atol = 1e-10)
end
```

The `# ... build F = ∇g ...` and `# ... fill loop ...` ellipses must be
expanded by the implementer using the exact loops already given (Task 2 fill
loop; gradient assembly = `compute_all_gradients_spectral!` on a temperature
clone, then `scalar_spectral_to_physical!` of ∇θ/∇φ/∇r into the three F
components — mirroring `_fp_angular_derivs`).

- [ ] **Step 2: Run — expect FAIL (force_curl_projections! undefined)**

Same command. UndefVarError.

- [ ] **Step 3: Implement `force_curl_projections!`**

Append to `src/physics/force_projection.jl`:

```julia
"""
    force_curl_projections!(R_tor, R_pol, Q, S, T, domain)

Momentum-equation projections of a force field from its QST scalars:
`R_tor[lm](r) = [r̂·∇×F]_lm` and `R_pol[lm](r) = [r̂·∇×∇×F]_lm`.
Candidate identities (verified by test/force_projection_reference.jl —
adjust sign/r-power there if the reference disagrees, with a paper
re-derivation):

    R_tor = −(l(l+1)/r) · T
    R_pol = (l(l+1)/r²) · ( Q − ∂_r(r·S) )
"""
function force_curl_projections!(
        R_tor::SpectralFieldType{T},
        R_pol::SpectralFieldType{T},
        Q::SpectralFieldType{T},
        S::SpectralFieldType{T},
        T_in::SpectralFieldType{T},
        domain::RadialDomainType
) where {T}
    cfg = Q.config
    D1 = create_derivative_matrix(T, 1, domain)
    nr = domain.N
    rS = Vector{T}(undef, nr); drS = Vector{T}(undef, nr)
    qv = Vector{T}(undef, nr)
    r_range = local_range(Q.pencil, 3)

    for (dst_t, dst_p, src_q, src_s, src_t) in (
        (parent(R_tor.data_real), parent(R_pol.data_real),
         parent(Q.data_real), parent(S.data_real), parent(T_in.data_real)),
        (parent(R_tor.data_imag), parent(R_pol.data_imag),
         parent(Q.data_imag), parent(S.data_imag), parent(T_in.data_imag)),
    )
        for lm in 1:cfg.nlm
            slot = local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            l = cfg.l_values[lm]
            λ = T(l * (l + 1))
            if l == 0
                # no toroidal/poloidal projection for l=0
                for r_idx in r_range
                    lr = r_idx - first(r_range) + 1
                    set_local_spectral_value!(dst_t, slot, lr, zero(T))
                    set_local_spectral_value!(dst_p, slot, lr, zero(T))
                end
                continue
            end
            # gather full radial profiles (serial/spec-pencil keeps r local)
            for r_idx in 1:nr
                rS[r_idx] = T(domain.r[r_idx, 4]) *
                            local_spectral_value(src_s, slot, r_idx)
                qv[r_idx] = local_spectral_value(src_q, slot, r_idx)
            end
            LA.mul!(drS, D1, rS)
            for r_idx in r_range
                lr = r_idx - first(r_range) + 1
                r = T(domain.r[r_idx, 4])
                tval = local_spectral_value(src_t, slot, lr)
                set_local_spectral_value!(dst_t, slot, lr, -λ / r * tval)
                set_local_spectral_value!(dst_p, slot, lr,
                    λ / r^2 * (qv[r_idx] - drS[r_idx]))
            end
        end
    end
    return R_tor, R_pol
end
```

(`LA` is the `LinearAlgebra` alias already used in this part of the codebase —
check the includes at the top of `src/physics/nonlinear.jl` and import the
same way. If `BandedMatrix`'s 3-arg `mul!` is the available method, use it.)

- [ ] **Step 4: Run — iterate on the candidate constants until reference agrees**

Same command. Expected outcomes and required reactions:
- NULL test fails ⇒ structural error (a gradient field must be annihilated by
  any curl) — debug the harness or the analysis, NOT the constants.
- Buoyancy/generic tests fail by a uniform constant (same factor every mode,
  every radius — print `got ./ ref` to check) ⇒ convention bookkeeping: fix
  the sign/r-power in BOTH the formula and its docstring, re-derive on paper,
  re-run.
- Non-uniform mismatch ⇒ wrong structure (e.g. ∂_r(rS) vs r∂_rS vs S+r∂_rS):
  re-derive; do not ship.

All three testsets green before proceeding.

- [ ] **Step 5: Commit**

```bash
git add src/physics/force_projection.jl test/force_projection_reference.jl
git commit -m "feat(physics): reference-verified curl projections for force fields (stage 1)

R_tor=[r̂·∇×F]_lm and R_pol=[r̂·∇×∇×F]_lm from QST scalars, verified against
an independent FD/spectral-gradient curl reference: gradient-field null,
pure-radial (buoyancy) projection, and generic random force.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Register tests + full suite

**Files:**
- Modify: `test/runtests.jl` (append `"force_projection_reference.jl"` to the `additional_tests` list, same style as neighbours)

- [ ] **Step 1: Register**

Find the `additional_tests` array in `test/runtests.jl` and add the entry
`"force_projection_reference.jl",` in alphabetical-ish position with the
other physics tests.

- [ ] **Step 2: Full suite**

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()' > /tmp/stage1_suite.log 2>&1; echo "EXIT=$?"
grep -E "Testing GeoDynamo tests passed" /tmp/stage1_suite.log
grep -cE "Test Failed|Error During Test" /tmp/stage1_suite.log
```
Expected: `EXIT=0`, `tests passed`, `0`. Baseline broken count (18) unchanged
— Stage 1 adds code that nothing in the dynamics calls, so no behavioral test
may move.

- [ ] **Step 3: Commit**

```bash
git add test/runtests.jl
git commit -m "test: register force-projection reference tests (stage 1 complete)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (Stage 1 only):** QST analysis (Task 2 = spec's
`force_physical_to_qst!`), reference-verified projections incl. the buoyancy
identity (Task 3), brute-force independent reference (Task 1), nothing wired
into dynamics (all tasks — `finish_velocity_nonlinear!` untouched). Stages
2–5 are separate plans by design.

**Placeholder scan:** Two deliberate implementer-expansion points remain —
`_fp_curl`'s two remaining component loops and the test fill/gradient
assembly ellipses — each with the exact formulas and a same-file pattern to
copy (`_fp_radial_curl`, Task-2 fill loop). The `_force_r_scratch` note
explicitly licenses simplification if the component type already matches.
Constructor-name uncertainty (`create_radial_domain`) carries its own
verification grep. These are bounded look-ups, not open design.

**Type consistency:** `force_physical_to_qst!(F, Q, S, T_)` and
`force_curl_projections!(Rtor, Rpol, Q, S, T_, dom)` used identically in
src and tests; argument order of the existing
`vector_physical_to_spectral!(vec, toroidal, poloidal)` and
`vector_spectral_to_physical!(toroidal, poloidal, vec)` quoted from
numerics.jl:827/926 and used consistently (S rides the "poloidal" slot).

**Honesty gates:** candidate formulas are labelled candidates; the tests
arbitrate; constant-only corrections must be paper-re-derived; structural
mismatches block shipping.
