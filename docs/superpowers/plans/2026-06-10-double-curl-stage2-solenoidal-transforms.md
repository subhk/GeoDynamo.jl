# Double-Curl Stage 2: Consistent Solenoidal Velocity Transform Pair

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the (T,P)↔velocity transform pair a consistent solenoidal toroidal–poloidal representation — `u_r = l(l+1)·P/r²`, tangential spheroidal scalar `S = (1/r)·∂_r(r·P)` — with a REAL divergence diagnostic as the acceptance gate, unifying the two existing u_r conventions.

**Architecture:** Stage 2 of `docs/superpowers/specs/2026-06-10-poloidal-momentum-double-curl-design.md`. The synthesis change lives in the shared `vector_spectral_to_physical_disttranspose!` core (one site) plus its two `vr_factor` call sites; the analysis change makes P recovery Q-based (`P = r²·Q(u)/λ`) using the Stage-1 radial-analysis plumbing. Vorticity formulas re-derive under the new convention and are verified against the Stage-1 numerical curl reference (exact single-pass). Everything is gated by a new, real solenoidality check (the existing `compute_divergence_spectral` is a stub returning `(0.0, 0.0)` — the "Solenoidal Constraint Report" has been fake).

**Tech Stack:** Julia 1.11, SHTnsKit, PencilArrays. Direct binary `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=.`, repo root, single-rank.

**⚠️ Behavior-changing by design.** The stored meaning of P changes; every dynamic trajectory with nonzero velocity/magnetic field changes; restart files holding P are convention-incompatible (note in CHANGELOG-style commit). Tests pinning old behavior get UPDATED — each such update must name the old expectation and why the new one is right.

---

## Blast-radius survey (verified by grep, 2026-06-10)

- Shared synthesis core: `vector_spectral_to_physical_disttranspose!` (src/solver/numerics.jl:871) — tangential from `dist_synthesis_sphtor!(plan, Vt, Vp, Slm, Tlm)` with **Slm = raw P** (the inconsistency); v_r via `vr_factor`:
  - solver path numerics.jl:838: `l*(l+1)/r_val`
  - MIE path fields/transforms.jl:381: `l*(l+1)/(r_val*r_val)`
- Shared analysis core: `vector_physical_to_spectral_disttranspose!` (numerics.jl:946) — P ← raw S (the other half of the inconsistency).
- Synthesis callers: numerics.jl:1001/1012 (velocity+vorticity physical refresh), 1370/1381, diagnostics/solver.jl:68/70 (output writer; **magnetic too** — same pair, convention changes identically), Ball/Ball.jl:320, MIE wrappers transforms.jl:343/422.
- Vorticity: `compute_vorticity_spectral!` (numerics.jl:1059) AND `compute_vorticity_spectral_full!` (velocity/field.jl:307, 558) — per-mode radial differentiation; formulas tied to the convention.
- Divergence: `compute_divergence_spectral` (diagnostics/solver.jl:164) — **stub `(0.0, 0.0)`**.
- GPU: `gpu_vector_spectral_to_physical!` takes `(…, lfac, rscale)` — own convention args (gpu/vector_transform.jl:37, velocity_nonlinear.jl:34/40/46). NOT ported this stage — loud-gate it.
- Force projection (Stage 1): `force_physical_to_qst!` uses RAW sphtor scalars — unaffected by the P-convention. No change.
- Tests pinning old behavior (expect updates): `poloidal_solenoidality.jl` (documents non-solenoidal synthesis — flips to a hard gate), MIE/vector roundtrip tests, vorticity/energy tests if they assert old factors, GPU≈CPU equivalence gates (skip locally; flag for GPU-box rerun).

## File Structure

- **Modify** `src/solver/numerics.jl` — synthesis core (S-from-P derivative step, unified vr_factor), analysis core (Q-based P), vorticity.
- **Modify** `src/fields/transforms.jl` — MIE wrapper vr_factor (now identical to solver's; collapse the lambda to one shared constant function).
- **Modify** `src/physics/velocity/field.jl` — `compute_vorticity_spectral_full!` (both methods).
- **Modify** `src/diagnostics/solver.jl` — real `compute_divergence_spectral`.
- **Modify** `src/gpu/vector_transform.jl` — loud gate (error directing to CPU until ported).
- **Create** `test/solenoidal_transform_pair.jl` — the Stage-2 gate tests.
- **Modify** `test/poloidal_solenoidality.jl` — becomes a hard ∇·u≈0 gate.
- **Modify** `test/runtests.jl` — register.

Tasks below follow TDD; each has RED→GREEN→commit. Use the Stage-1 harness (`test/force_projection_reference.jl` fixtures) by `include`-ing its fixture block or duplicating the 20-line fixture — duplicate, to keep files independent (copy `_fp_setup`, `_fp_spec`, `_fp_vec`, `_fp_phys`, `_fp_angular_derivs`, `_fp_radial_deriv` into the new file with `_st_` prefixes).

---

### Task 1: Real divergence diagnostic (the gate itself)

**Files:**
- Modify: `src/diagnostics/solver.jl:164` (replace stub)
- Create: `test/solenoidal_transform_pair.jl`

The physical-space divergence of a vector field on the grid:
∇·u = (1/r²)∂_r(r²·u_r) + (1/(r·sinθ))∂θ(sinθ·u_θ) + (1/(r·sinθ))∂φ(u_φ)
computable exactly (band-limited) with the harness pieces. The SPECTRAL
diagnostic `compute_divergence_spectral(tor, pol, domain)` must return real
L2/L∞ norms of ∇·u for the synthesized field.

- [ ] **Step 1 (RED):** create `test/solenoidal_transform_pair.jl` with the `_st_*` fixtures plus:

```julia
# physical-space divergence via exact spectral angular derivatives + banded D1
function _st_divergence(cfg, dom, V)
    r_range = GeoDynamo.range_local(cfg.pencils.r, 3)
    sinθ = sin.(cfg.theta_grid)
    # r²·u_r
    r2ur = _st_phys(cfg, dom)
    a = parent(r2ur.data); ur = parent(V.r_component.data)
    for k in axes(a, 3), j in 1:cfg.nlon, i in 1:cfg.nlat
        r = dom.r[k + first(r_range) - 1, 4]
        a[i, j, k] = r^2 * ur[i, j, k]
    end
    d_r2ur = _st_radial_deriv(cfg, dom, r2ur)
    # sinθ·u_θ
    sut = _st_phys(cfg, dom)
    b = parent(sut.data); uθ = parent(V.θ_component.data)
    for k in axes(b, 3), j in 1:cfg.nlon, i in 1:cfg.nlat
        b[i, j, k] = sinθ[i] * uθ[i, j, k]
    end
    dθ_sut, _ = _st_angular_derivs(cfg, dom, sut)
    _, dφ_uφ = _st_angular_derivs(cfg, dom, V.φ_component)  # (1/sinθ)∂φuφ
    out = _st_phys(cfg, dom); o = parent(out.data)
    A = parent(d_r2ur.data); B = parent(dθ_sut.data); C = parent(dφ_uφ.data)
    for k in axes(o, 3), j in 1:cfg.nlon, i in 1:cfg.nlat
        r = dom.r[k + first(r_range) - 1, 4]
        o[i, j, k] = A[i, j, k] / r^2 + (B[i, j, k] / sinθ[i] + C[i, j, k]) / r
    end
    return out
end

@testset "compute_divergence_spectral is real (not a stub)" begin
    cfg, dom = _st_setup()
    Random.seed!(21)
    tor = _st_spec(cfg, dom); pol = _st_spec(cfg, dom)
    # random band-limited (l ≤ lmax-2) smooth profiles, same fill loop as the
    # Stage-1 tests (sinpi(x)*randn()*1e-2 per mode/radius, imag 0.7v for m>0)
    # ... [reproduce the fill loop from test/force_projection_reference.jl] ...
    l2, linf = GeoDynamo.compute_divergence_spectral(tor, pol, dom)
    V = _st_vec(cfg, dom)
    GeoDynamo.vector_spectral_to_physical!(tor, pol, V; domain = dom)
    div_phys = _st_divergence(cfg, dom, V)
    ref_linf = maximum(abs, parent(div_phys.data))
    # the diagnostic must (a) not be the stub, (b) agree with the physical
    # divergence to within discretization tolerance
    @test !(l2 == 0.0 && linf == 0.0) || ref_linf < 1e-10
    @test isapprox(linf, ref_linf; rtol = 0.2) || (linf < 1e-10 && ref_linf < 1e-10)
end
```

- [ ] **Step 2:** run, expect FAIL (stub returns zeros while the synthesized field is NOT divergence-free under the old convention, so `ref_linf` is large).

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using GeoDynamo; using Test; include("test/solenoidal_transform_pair.jl")' 2>&1 | tail -6
```

- [ ] **Step 3 (GREEN):** implement `compute_divergence_spectral` in `src/diagnostics/solver.jl` by synthesis + the same physical-space formula (synthesize u from (tor,pol) via `vector_spectral_to_physical!` into a scratch vector field, compute the divergence grid with banded D1 + the scalar-gradient spectral recurrences — mirror the test harness logic in src, using the existing solver scratch/buffers conventions; correctness over allocation-cleanliness, leave an optimization note).

- [ ] **Step 4:** run — the diagnostic now reports the TRUE (large) divergence of the old synthesis; the test's agreement clause passes. Commit:

```bash
git add src/diagnostics/solver.jl test/solenoidal_transform_pair.jl
git commit -m "feat(diagnostics): real divergence diagnostic (was a hardcoded-zero stub)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: New synthesis (S = (1/r)∂_r(rP), u_r = λP/r², one convention)

**Files:**
- Modify: `src/solver/numerics.jl` (`vector_spectral_to_physical_disttranspose!` + caller at :838)
- Modify: `src/fields/transforms.jl:381` (caller)
- Test: `test/solenoidal_transform_pair.jl` (append)

- [ ] **Step 1 (RED):** append the hard solenoidality + manufactured-field tests:

```julia
@testset "synthesized (T,P) field is solenoidal" begin
    cfg, dom = _st_setup()
    Random.seed!(23)
    tor = _st_spec(cfg, dom); pol = _st_spec(cfg, dom)
    # ... same band-limited random fill ...
    V = _st_vec(cfg, dom)
    GeoDynamo.vector_spectral_to_physical!(tor, pol, V; domain = dom)
    div_phys = _st_divergence(cfg, dom, V)
    scale = maximum(abs, parent(V.r_component.data)) +
            maximum(abs, parent(V.θ_component.data))
    # interior nodes (banded D1 endpoint stencils excluded via norm over 2:N-1)
    dd = parent(div_phys.data)
    interior = dd[:, :, 2:(size(dd, 3) - 1)]
    @test maximum(abs, interior) < 1e-6 * scale
end

@testset "manufactured single-mode synthesis: u_r = l(l+1)P/r²" begin
    cfg, dom = _st_setup()
    pol = _st_spec(cfg, dom); tor = _st_spec(cfg, dom)
    # seed (l=2,m=0), P(r) = r³ (smooth, nonzero at both ends)
    lm_seed = findfirst(i -> cfg.l_values[i] == 2 && cfg.m_values[i] == 0, 1:cfg.nlm)
    slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_seed)
    for r_idx in 1:dom.N
        GeoDynamo.set_local_spectral_value!(parent(pol.data_real), slot, r_idx,
            dom.r[r_idx, 4]^3)
    end
    V = _st_vec(cfg, dom)
    GeoDynamo.vector_spectral_to_physical!(tor, pol, V; domain = dom)
    # u_r grid must equal λ·P/r² × Y_lm-pattern: compare against scalar synthesis
    expect_spec = _st_spec(cfg, dom)
    for r_idx in 1:dom.N
        r = dom.r[r_idx, 4]
        GeoDynamo.set_local_spectral_value!(parent(expect_spec.data_real), slot, r_idx,
            6.0 * r^3 / r^2)   # λ=6 for l=2
    end
    expect_phys = _st_phys(cfg, dom)
    GeoDynamo.scalar_spectral_to_physical!(expect_spec, expect_phys)
    @test isapprox(parent(V.r_component.data), parent(expect_phys.data);
        rtol = 1e-8, atol = 1e-12)
    # u_θ must equal (1/r)d(rP)/dr·∂θY = (1/r)(4r³)·∂θY = 4r²·∂θY
    dY = _st_vec(cfg, dom); delta = _st_spec(cfg, dom); z = _st_spec(cfg, dom)
    for r_idx in 1:dom.N
        GeoDynamo.set_local_spectral_value!(parent(delta.data_real), slot, r_idx, 1.0)
    end
    GeoDynamo.vector_spectral_to_physical!(z, delta, dY)   # OLD convention call —
    # NOTE: after this task the tangential synthesis of `delta` carries the
    # derivative coupling, so build ∂θY instead from the Stage-1 trick on a
    # CONSTANT-in-r delta only if d(r·1)/dr/r = 1/r ≠ 1 — i.e. this shortcut
    # no longer yields bare ∂θY. Instead compare u_θ on INTERIOR nodes against
    # 4r²·(∂θY from the analytic recurrence is overkill) — simplest robust
    # check: form the RATIO u_θ(i,j,k)/u_θ(i,j,k′) across radii at fixed
    # (i,j): must equal 4r_k²/4r_k′² = (r_k/r_k′)² for interior nodes where
    # u_θ ≠ 0. Assert that ratio for several (i,j) and k-pairs (rtol 1e-6).
end
```

(The ratio check sidesteps needing bare ∂θY post-change; it pins the radial
structure (1/r)∂_r(rP) = 4r² for P=r³ exactly.)

- [ ] **Step 2:** run — solenoidality FAILS (old synthesis non-solenoidal), manufactured u_r FAILS on the solver path (old u_r = λP/r gives 6r² not 6r). RED confirmed for the right reasons.

- [ ] **Step 3 (GREEN):** in `vector_spectral_to_physical_disttranspose!`:
  1. After `spec_storage_to_solve!`/`from_spec_solve!` produce Slm from the stored P, ADD a per-mode radial transform: S_lm(r) ← (1/r)·∂_r(r·P_lm(r)), applied to the **Slm coefficient array** before `dist_synthesis_sphtor!`. Implementation: banded `D1 = create_derivative_matrix(T, 1, domain)` (build once per call; optimization note for later caching), loop modes (the Alm array is m-distributed (l, m_local, lev) — the radial axis is the batched `lev` dimension; gather each (l,m)'s radial profile across lev — on the spec-solve side the r-axis is local per the existing v_r fill pattern at numerics.jl:848-865 `_fill_vr_alm!`; mirror that loop structure: it already iterates (lev, mi, l) over the SAME array layout).
  2. `domain` becomes REQUIRED for tangential synthesis (the derivative needs radii): keep the `domain=nothing` signature but `error("synthesis requires domain under the solenoidal convention")` if any poloidal coefficient is nonzero and domain === nothing — adapt the two wrappers to pass their domain through (they already take it).
  3. Unify vr_factor: change numerics.jl:838 to `(l, r_val) -> l * (l + 1) / (r_val * r_val)`; transforms.jl:381 already has it — collapse both to a shared `_solenoidal_vr_factor(l, r_val)` defined next to the disttranspose core.
- [ ] **Step 4:** run — both new testsets green; Task-1 testset now reports near-zero divergence (its agreement clause still holds). Commit:

```bash
git add src/solver/numerics.jl src/fields/transforms.jl test/solenoidal_transform_pair.jl
git commit -m "feat(transforms)!: solenoidal vector synthesis (S=(1/r)d(rP)/dr, u_r=l(l+1)P/r2)

BREAKING: unifies the solver (l(l+1)/r) and MIE (l(l+1)/r^2) radial
conventions and adds the spheroidal derivative coupling; stored P now means
the standard poloidal potential. Restart files from older versions are
convention-incompatible.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: New analysis (Q-based P recovery) + roundtrip

**Files:**
- Modify: `src/solver/numerics.jl` (`vector_physical_to_spectral_disttranspose!`)
- Test: `test/solenoidal_transform_pair.jl` (append)

- [ ] **Step 1 (RED):**

```julia
@testset "synthesis→analysis roundtrip on (T,P)" begin
    cfg, dom = _st_setup()
    Random.seed!(29)
    tor = _st_spec(cfg, dom); pol = _st_spec(cfg, dom)
    # ... band-limited random fill (profiles vanish at endpoints via sinpi) ...
    V = _st_vec(cfg, dom)
    GeoDynamo.vector_spectral_to_physical!(tor, pol, V; domain = dom)
    tor2 = _st_spec(cfg, dom); pol2 = _st_spec(cfg, dom)
    GeoDynamo.vector_physical_to_spectral!(V, tor2, pol2; domain = dom)
    for (a, b) in ((tor2, tor), (pol2, pol))
        @test isapprox(
            vcat(vec(parent(a.data_real)), vec(parent(a.data_imag))),
            vcat(vec(parent(b.data_real)), vec(parent(b.data_imag)));
            rtol = 1e-8, atol = 1e-12)
    end
end
```

- [ ] **Step 2:** run — FAILS (old analysis recovers P from raw S; new synthesis emits derivative-coupled S).

- [ ] **Step 3 (GREEN):** in `vector_physical_to_spectral_disttranspose!`: T from the sphtor T-scalar (unchanged); P from the RADIAL component: scalar-analyze v_r (the exact plumbing Stage 1 added for Q in `force_physical_to_qst!` — `scalar_physical_to_spectral!(vector_field.r_component, Qtmp)`) then per-mode `P_lm(r) = r²·Q_lm(r)/(l(l+1))` (l=0 → 0). The S output of `dist_analysis_sphtor!` is no longer stored (keep computing T via sphtor; skip the S store). `domain` required (radii) — thread it through `vector_physical_to_spectral!`.
  ⚠️ The FORCE path (`force_physical_to_qst!`) calls `vector_physical_to_spectral!(force, T_out, S)` expecting RAW S — after this change that call would return Q-based "P" instead. FIX force_projection.jl in the same commit: it needs the raw sphtor S — add a kwarg `raw_sphtor::Bool = false` to `vector_physical_to_spectral!` (default new behavior; force path passes `raw_sphtor = true`), OR (cleaner) have the force path call the disttranspose sphtor analysis directly. Either way the Stage-1 tests must stay green unchanged — they pin the raw-S semantics.

- [ ] **Step 4:** run THIS file + `test/force_projection_reference.jl` — both fully green. Commit:

```bash
git add src/solver/numerics.jl src/physics/force_projection.jl test/solenoidal_transform_pair.jl
git commit -m "feat(transforms)!: Q-based poloidal analysis (P = r^2 Q/l(l+1)); raw-S path kept for forces

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Vorticity under the new convention

**Files:**
- Modify: `src/solver/numerics.jl:1059` and `src/physics/velocity/field.jl:307,558`
- Test: `test/solenoidal_transform_pair.jl` (append)

For u = P-poloidal + T-toroidal (standard convention), ω = ∇×u is exactly:
ζ_toroidal-potential = −D_l P (where D_l = ∂_rr + (2/r)∂_r − λ/r² is the same
scalar operator the solver's Laplacian builders use) and ζ_poloidal-potential
= T. (Curl swaps the chains: curl(T(t)) = P(t); curl(P(p)) = T(−D_l p).)

- [ ] **Step 1 (RED):** vorticity test against the Stage-1 numerical curl (single-pass — exact):

```julia
@testset "spectral vorticity equals numerical curl of synthesized u" begin
    cfg, dom = _st_setup()
    Random.seed!(31)
    # random band-limited (T,P), synthesize u, numerically curl it (exact
    # single application via the Stage-1 _fp_curl logic — duplicate as _st_curl),
    # then synthesize the solver's spectral vorticity (ζᵀ,ζᴾ) to physical and
    # compare grids on interior radial nodes.
    # The velocity-fields container carries ζᵀ/ζᴾ: build via the same
    # constructor the solver uses (grep create_shtns_velocity_fields) OR call
    # the per-field compute_vorticity_spectral! on a minimal container — if the
    # container is heavy, test numerics.jl's per-mode kernel directly by
    # seeding (T,P) spectral fields and applying the NEW formulas via a small
    # local reference implementation, then synthesizing both sides.
    # Gate: interior-node grid agreement rtol 1e-6.
end
```

(The implementer picks the lightest container route; the assertion structure
is fixed: solver vorticity synthesis ≡ numerical curl of solver velocity
synthesis.)

- [ ] **Step 2:** run — RED under old formulas.
- [ ] **Step 3 (GREEN):** update both vorticity implementations to ζᵀ = −D_l P, ζᴾ = T per-mode (reusing their existing per-mode radial-derivative loop structure and workspaces — only the formula lines change).
- [ ] **Step 4:** green; commit:

```bash
git add src/solver/numerics.jl src/physics/velocity/field.jl test/solenoidal_transform_pair.jl
git commit -m "fix(velocity)!: vorticity formulas under the solenoidal P convention

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Loud gates, test triage, full suite

**Files:**
- Modify: `src/gpu/vector_transform.jl` (loud gate), `test/poloidal_solenoidality.jl` (hard gate), `test/runtests.jl` (register), any test pinning the old convention.

- [ ] **Step 1:** GPU loud gate — at the top of `gpu_vector_spectral_to_physical!` and `gpu_vector_physical_to_spectral!` add:
```julia
error("GPU vector transforms have not been ported to the solenoidal P convention (Stage 2); use the CPU path. See docs/superpowers/specs/2026-06-10-poloidal-momentum-double-curl-design.md")
```
(If GPU tests exercise them on the Array backend, those tests get `@test_throws` updates or skip-marks with the same message.)
- [ ] **Step 2:** flip `test/poloidal_solenoidality.jl` to a hard gate: the synthesized poloidal field's physical divergence (via the `_st_divergence` logic — import or duplicate) must be < 1e-6·scale on interior nodes. Delete the "NOT divergence-free" expectation block.
- [ ] **Step 3:** register `solenoidal_transform_pair.jl` in `test/runtests.jl`.
- [ ] **Step 4:** full suite; triage failures ONE BY ONE: each updated test gets a comment naming the old expectation and the convention change. Expected hot spots: MIE/vector roundtrips, ball tests (analysis now Q-based — ball regularity loop reads P slots), energy spectra, GPU Array-backend tests, characterization snapshots (regenerate refs ONLY where velocity/magnetic content is nonzero — temperature-only snapshots survive).
- [ ] **Step 5:** suite green; commit:

```bash
git add -u test/ src/gpu/vector_transform.jl
git commit -m "test!: stage-2 triage — solenoidality hard gate, GPU loud-gate, convention updates

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** synthesis consistency + convention unification (Task 2), Q-based analysis (Task 3), vorticity (Task 4), hard ∇·u gate replacing the documented non-solenoidal expectation (Tasks 1+5), GPU loud-gate not silent divergence (Task 5), force-path raw-S preservation so Stage-1 stays green (Task 3). Magnetic shares the pair automatically; its insulating-BC audit is NOT here — flagged for Stage 3/4 prep (the BC builders act on P at boundaries; their derivation must be re-checked against the new P meaning before Stage 4 trusts them — recorded as an explicit follow-up).

**Placeholders:** the random fill loops reference the exact loop shipped in `test/force_projection_reference.jl` (committed, green) — reproduce verbatim with the `_st_` fixtures. Task 4's test skeleton fixes the assertion structure and gate while delegating the container choice — bounded.

**Type consistency:** `vector_spectral_to_physical!(tor, pol, V; domain)` / `vector_physical_to_spectral!(V, tor, pol; domain)` argument orders match numerics.jl:827/926. `D_l` matches the solver's Laplacian builder convention (implicit.jl:60-64).

**Honesty:** Task-1's gate must FAIL against the old synthesis before Task 2 (the fake-zero stub is replaced first so the failure is visible) — ordering is load-bearing.
