# Ball Geometry Full-MHD Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Full MHD in a full sphere (geometry = :ball): off-center radial grid, l-dependent Robin regularity rows at the inner boundary, unified Stage-4 projection nonlinear paths (legacy ball branches deleted), CNAB2 + ERK2, validated by spherical-Bessel decay rates, a constrained-eigenvalue probe, physics tests, and the Marti et al. (2014) benchmark.

**Architecture:** Ball flows through the SAME projection/transform code as the shell — possible because the new grid has no r=0 node so every 1/r is finite. Geometry differences are confined to: the domain builder, inner-boundary matrix rows (regularity instead of wall), and one suspected insulating-row fix. See `docs/superpowers/specs/2026-06-11-ball-geometry-mhd-design.md`.

**Tech Stack:** Julia 1.11, GeoDynamo.jl internal banded-matrix machinery (BandedMatrix/BandedLU, `create_derivative_matrix`, `solve_banded!`), SHTnsKit transforms (unchanged), existing ERK2 `SolverERK2BoundarySide` descriptor machinery.

---

## Critical context for the implementer

- **Base commit:** `0f103d0` on branch `test/gate-stage2-gpu-vector-and-eab2`. The repo's main checkout is driven by concurrent sessions — NEVER work in `/Users/subha/Documents/GitHub/GeoDynamo.jl` directly. Task 0 creates a dedicated worktree.
- **Julia:** the `julia` shim is broken. Use the direct binary:
  `~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=.`
  Never pipe `Pkg.test()` through `tail` (masks the exit code); redirect to a file.
- **Regularity exponents (derived in the spec, do not re-derive):** poloidal P and W use β = l+1; toroidal t (raw sphtor scalar) and scalars use β = l. Robin row: `f′(r₁) = β·f(r₁)/r₁`.
- **Radial domain layout:** `domain.r` is an N×7 matrix of r-powers; column p holds r^(p−4) — so `r[:,4]` = r, `r[:,3]` = 1/r, `r[:,2]` = 1/r². `domain.N` = N. Banded ops: `create_derivative_matrix(T, order, domain)` returns a BandedMatrix-like with `.data` (size (2bw+1)×N, row bw+1 = diagonal); entry (i,j) lives at `data[bw+1+i−j, j]`. `radial_bandwidth(domain)` = bw.
- **Boundary-row stamping idiom** (used everywhere; copy it exactly):
  ```julia
  # zero row 1 within the band, then stamp:
  @inbounds for j in 1:(1 + bw)
      system_data[bw + 1 + 1 - j, j] = d1_matrix.data[bw + 1 + 1 - j, j]
  end
  system_data[bw + 1, 1] -= T(beta * domain.r[1, 3])   # f′ − (β/r₁)f = 0
  ```
- **Test traps:** `test/static_checks.jl` pins source text — if a task's edit breaks it, repoint the assert to follow the new code (do not weaken it). The full suite has known flaky scalar-IC tests; re-run before attributing failures to your change. A single run showing ~3 IC failures may be a flake.

---

### Task 0: Worktree + branch + docs import

**Files:**
- Create: worktree `../GeoDynamo-ball` on new branch `feat/ball-geometry-mhd`

- [ ] **Step 1: Create the worktree from the base commit**

```bash
cd /Users/subha/Documents/GitHub/GeoDynamo.jl
git worktree add ../GeoDynamo-ball -b feat/ball-geometry-mhd 0f103d0
cd ../GeoDynamo-ball
```

- [ ] **Step 2: Import the spec + plan (they live on other branches)**

```bash
git -C /Users/subha/Documents/GitHub/GeoDynamo.jl show feat/r-dist-solenoidal-synthesis:docs/superpowers/specs/2026-06-11-ball-geometry-mhd-design.md > docs/superpowers/specs/2026-06-11-ball-geometry-mhd-design.md 2>/dev/null || true
```

If that file is empty (spec amendments may be uncommitted in the main checkout), copy it directly:
```bash
cp /Users/subha/Documents/GitHub/GeoDynamo.jl/docs/superpowers/specs/2026-06-11-ball-geometry-mhd-design.md docs/superpowers/specs/
cp /Users/subha/Documents/GitHub/GeoDynamo.jl/docs/superpowers/plans/2026-06-11-ball-geometry-mhd.md docs/superpowers/plans/
git add docs/superpowers/specs/2026-06-11-ball-geometry-mhd-design.md docs/superpowers/plans/2026-06-11-ball-geometry-mhd.md
git commit -m "docs: import ball geometry MHD spec + plan"
```

- [ ] **Step 3: Instantiate + baseline sanity**

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using GeoDynamo; include("test/ball_roundtrip.jl")' > /tmp/ball_base.log 2>&1; tail -5 /tmp/ball_base.log
```
Expected: existing ball roundtrip tests pass at base.

---

### Task 1: Off-center ball radial domain

**Files:**
- Modify: `src/Ball/Ball.jl` (`create_ball_radial_domain`, lines ~40–81)
- Create: `test/ball_domain.jl`
- Modify: `test/ball_finiteness.jl`, `test/ball_roundtrip.jl` (expectations referencing the r=0 node)

- [ ] **Step 1: Write the failing test** — `test/ball_domain.jl`:

```julia
using Test
using LinearAlgebra
using GeoDynamo
const Ball = GeoDynamo.GeoDynamoBall

@testset "ball off-center radial domain" begin
    N = 16
    dom = Ball.create_ball_radial_domain(N)
    rr = dom.r[1:N, 4]
    @test rr[N] ≈ 1.0
    @test rr[1] > 0.0                       # no node at the center
    @test rr[1] ≈ (1 - cos(pi / N)) / 2
    @test all(diff(rr) .> 0)
    @test all(isfinite, dom.r[1:N, 1:7])
    @test dom.r[1, 3] ≈ 1 / rr[1]           # honest 1/r at the innermost node
    @test dom.r[1, 2] ≈ 1 / rr[1]^2         # honest 1/r² (old code zeroed these)

    # operators finite and accurate mid-grid
    d1 = GeoDynamo.create_derivative_matrix(Float64, 1, dom)
    @test all(isfinite, d1.data)
    f = rr .^ 3
    df = similar(f)
    mul!(df, d1, f)
    @test isapprox(df[N ÷ 2], 3 * rr[N ÷ 2]^2; rtol = 1e-6)
end
```

- [ ] **Step 2: Run to verify it fails**

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using GeoDynamo; include("test/ball_domain.jl")'
```
Expected: FAIL on `rr[1] > 0.0` (old grid has r₁ = 0).

- [ ] **Step 3: Replace the node generation in `create_ball_radial_domain`**

Replace the body between the `N < 2` check and the `dr_matrices` allocation with:

```julia
    # Off-center cosine grid: r_n = (1 − cos(πn/N))/2, n = 1..N.
    # r_N = 1 exactly; r_1 = (1 − cos(π/N))/2 > 0 — no node at the center.
    # Regularity at r=0 is imposed through l-dependent Robin boundary rows in
    # the implicit matrices, not through grid values, so every 1/r, 1/r²
    # operator entry stays finite and honest.
    r = zeros(Float64, N, 7)
    for n in 1:N
        r[n, 4] = 0.5 * (1.0 - cos(pi * n / N))
    end
    for p in 1:7
        if p != 4
            power = p - 4
            for i in 1:N
                r[i, p] = r[i, 4]^power
            end
        end
    end
```

(The old `if r_val == 0.0 && power < 0` guard is deleted — nothing to regularize.)

- [ ] **Step 4: Run the new test — expect PASS; then run the two existing ball tests**

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using GeoDynamo; include("test/ball_domain.jl"); include("test/ball_finiteness.jl"); include("test/ball_roundtrip.jl")' > /tmp/t1.log 2>&1; tail -20 /tmp/t1.log
```

If `ball_finiteness.jl`/`ball_roundtrip.jl` fail: their failures will be assertions tied to the r=0 node (e.g., values exactly zero at the inner plane, or `dom.r[1,4] == 0`). Update those assertions to the off-center grid (`rr[1] > 0`, finiteness instead of exact zeros). Do NOT delete coverage — keep every finiteness/roundtrip check, only fix grid expectations.

- [ ] **Step 5: Commit**

```bash
git add src/Ball/Ball.jl test/ball_domain.jl test/ball_finiteness.jl test/ball_roundtrip.jl
git commit -m "feat(ball): off-center radial grid — no r=0 node, honest negative powers"
```

---

### Task 2: Scalar regularity rows + scalar Bessel-decay validation

**Files:**
- Modify: `src/bcs/scalar_bc.jl` (`_apply_scalar_boundary_rows!`, `create_scalar_matrices`)
- Modify: `src/bcs/thermal_bc.jl`, `src/bcs/compositional_bc.jl` (thread kwarg)
- Create: `test/ball_bessel_decay.jl`

- [ ] **Step 1: Write the failing test** — `test/ball_bessel_decay.jl`:

```julia
using Test
using LinearAlgebra
using GeoDynamo
const Ball = GeoDynamo.GeoDynamoBall

# Spherical Bessel closed forms + first zeros (no SpecialFunctions dep).
sph_j0(x) = x == 0 ? 1.0 : sin(x) / x
sph_j1(x) = x == 0 ? 0.0 : sin(x) / x^2 - cos(x) / x
const ALPHA_J0 = Float64(pi)            # first zero of j0
const ALPHA_J1 = 4.493409457909064      # first zero of j1

# CN-step a pure-diffusion radial profile through banded matrices `mats`
# (homogeneous BC rows) and return the measured decay rate from the second
# half of the run (first half discards the row-replacement transient).
function measured_decay_rate(mats, dom, l::Int, theta::Vector{Float64}; dt, nsteps)
    nr = dom.N
    idx = mats.lookup[l]
    A = mats.factorizations[idx]
    L = mats.linear_matrices[idx]
    rhs = similar(theta); Lf = similar(theta); out = similar(theta)
    inv_dt = 1 / dt
    mid = nr ÷ 2
    nhalf = nsteps ÷ 2
    vh = 0.0
    for s in 1:nsteps
        mul!(Lf, L, theta)
        @. rhs = inv_dt * theta + 0.5 * Lf
        rhs[1] = 0.0; rhs[nr] = 0.0          # homogeneous BC rows
        GeoDynamo.solve_banded!(out, A, rhs)
        copyto!(theta, out)
        s == nhalf && (vh = theta[mid])
    end
    return log(vh / theta[mid]) / ((nsteps - nhalf) * dt)
end

@testset "ball scalar Bessel decay (analytic anchor)" begin
    nr = 48; dt = 2e-4; nsteps = 200
    cfg = GeoDynamo.create_shtnskit_config(lmax = 4, mmax = 4,
        nlat = 12, nlon = 24, nr = nr)
    dom = Ball.create_ball_radial_domain(nr)
    mats = GeoDynamo.create_scalar_matrices(cfg, dom, 1.0, dt;
        scalar_bc_code = 1, inner_regularity = true)
    rr = dom.r[1:nr, 4]

    # l=0: Θ = j0(πr), σ = π² (inner row reduces to Θ′(r₁)=0 automatically)
    σ0 = measured_decay_rate(mats, dom, 0, [sph_j0(ALPHA_J0 * r) for r in rr];
        dt, nsteps)
    @test isapprox(σ0, ALPHA_J0^2; rtol = 5e-3)

    # l=1: Θ = j1(α₁r), σ = α₁²
    σ1 = measured_decay_rate(mats, dom, 1, [sph_j1(ALPHA_J1 * r) for r in rr];
        dt, nsteps)
    @test isapprox(σ1, ALPHA_J1^2; rtol = 5e-3)
end
```

- [ ] **Step 2: Run to verify it fails**

Expected: `MethodError`/`UndefKeywordError` — `create_scalar_matrices` has no `inner_regularity` kwarg.

- [ ] **Step 3: Implement.** In `src/bcs/scalar_bc.jl`:

(a) `_apply_scalar_boundary_rows!` gains `inner_regularity::Bool` (positional, after `N`); replace the inner branch:

```julia
    if inner_regularity
        # Ball center regularity: Θ ~ r^l ⇒ Θ′(r₁) = l·Θ(r₁)/r₁  (β = l;
        # l=0 reduces to Θ′(r₁)=0). Exact to leading order; consistency
        # error O(r₁²) shrinks as N⁻² (see ball design spec §5).
        @inbounds for j in 1:(1 + bw)
            system_data[bw + 1 + 1 - j, j] = d1_data[bw + 1 + 1 - j, j]
        end
        system_data[bw + 1, 1] -= T(l * r_inv_inner)
    elseif _scalar_inner_is_dirichlet(scalar_bc_code)
        system_data[bw + 1, 1] = one(T)
    else
        @inbounds for j in 1:(1 + bw)
            system_data[bw + 1 + 1 - j, j] = d1_data[bw + 1 + 1 - j, j]
        end
    end
```

`r_inv_inner` is a new argument (pass `domain.r[1, 3]` from the caller — the helper does not receive `domain` today). New signature:

```julia
function _apply_scalar_boundary_rows!(
        system_data::AbstractMatrix{T},
        d1_data::AbstractMatrix{T},
        scalar_bc_code::Int,
        l::Int,
        bw::Int,
        N::Int,
        inner_regularity::Bool,
        r_inv_inner::Float64
) where {T}
```

Keep the trailing `scalar_bc_code == 4 && l == 0` Dirichlet-pin block UNCHANGED and AFTER the new branch (NN at l=0 is singular regardless of geometry; the pin must still win).

(b) `create_scalar_matrices` gains `inner_regularity::Bool = false` kwarg; call site becomes:

```julia
        _apply_scalar_boundary_rows!(system_data, d1_matrix.data,
            scalar_bc_code, l, bw, N, inner_regularity, domain.r[1, 3])
```

(c) Thread the kwarg through the public wrappers: in `src/bcs/thermal_bc.jl` `create_temperature_matrices(...)` and `src/bcs/compositional_bc.jl` `create_composition_matrices(...)` add `inner_regularity::Bool = false` and pass it to `create_scalar_matrices`.

- [ ] **Step 4: Run the test — expect PASS.** Also run the shell-side scalar tests to prove no regression:

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using GeoDynamo; include("test/ball_bessel_decay.jl")'
```

If the rate is off by a few percent rather than <0.5%: increase `nr` to 64 and re-check — if the error shrinks ~quadratically it is the expected Robin truncation; loosen rtol to 1e-2 ONLY if the convergence trend is verified and note it in the test comment.

- [ ] **Step 5: Commit**

```bash
git add src/bcs/scalar_bc.jl src/bcs/thermal_bc.jl src/bcs/compositional_bc.jl test/ball_bessel_decay.jl
git commit -m "feat(ball): scalar inner regularity rows (beta=l) + Bessel decay anchor"
```

---

### Task 3: Toroidal + magnetic regularity rows; insulating-row audit

**Files:**
- Modify: `src/bcs/velocity_bc.jl` (`create_velocity_toroidal_matrices`, ~line 66)
- Modify: `src/bcs/magnetic_bc.jl` (`create_magnetic_toroidal_matrices` ~56, `create_magnetic_poloidal_matrices` ~147)
- Modify: `test/ball_bessel_decay.jl` (extend)

- [ ] **Step 1: Extend the test** — append to `test/ball_bessel_decay.jl`:

```julia
@testset "ball toroidal Bessel decay" begin
    nr = 48; dt = 2e-4; nsteps = 200
    cfg = GeoDynamo.create_shtnskit_config(lmax = 4, mmax = 4,
        nlat = 12, nlon = 24, nr = nr)
    dom = Ball.create_ball_radial_domain(nr)
    rr = dom.r[1:nr, 4]
    # velocity toroidal: Ek(∂t − Δ_l)t ⇒ rate independent of Ek; t ~ j_l(αr),
    # no-slip outer t(1)=0, regularity β=l inner.
    mats = GeoDynamo.create_velocity_toroidal_matrices(cfg, dom, 1.0, dt;
        velocity_bc_code = 1, mass_coeff = 1.0, inner_regularity = true)
    σ = measured_decay_rate(mats, dom, 1, [sph_j1(ALPHA_J1 * r) for r in rr];
        dt, nsteps)
    @test isapprox(σ, ALPHA_J1^2; rtol = 5e-3)
end

@testset "ball magnetic poloidal free decay — classic dipole rate pi^2" begin
    nr = 48; dt = 2e-4; nsteps = 200
    cfg = GeoDynamo.create_shtnskit_config(lmax = 4, mmax = 4,
        nlat = 12, nlon = 24, nr = nr)
    dom = Ball.create_ball_radial_domain(nr)
    rr = dom.r[1:nr, 4]
    mats = GeoDynamo.create_magnetic_poloidal_matrices(cfg, dom, 1.0, dt;
        inner_regularity = true)
    # Slowest l=1 insulating free-decay mode: P = r·j1(πr), σ = π²
    # (transcendental condition j_{l-1}(α)=0 under the B_r = λP/r² convention).
    σ = measured_decay_rate(mats, dom, 1,
        [r * sph_j1(Float64(pi) * r) for r in rr]; dt, nsteps)
    @test isapprox(σ, Float64(pi)^2; rtol = 5e-3)
end
```

- [ ] **Step 2: Run — expect FAIL** (no `inner_regularity` kwarg yet).

- [ ] **Step 3: Implement the three builders.**

(a) `create_velocity_toroidal_matrices`: add kwarg `inner_regularity::Bool = false`. The inner-BC block (currently `if velocity_bc_code == 1 || velocity_bc_code == 2` … identity / stress-free) becomes:

```julia
        if inner_regularity
            # Ball center regularity for the raw sphtor toroidal scalar:
            # t ~ r^l ⇒ t′(r₁) = l·t(r₁)/r₁ (β = l).
            @inbounds for j in 1:(1 + bw)
                system_data[bw + 1 + 1 - j, j] = d1_matrix.data[bw + 1 + 1 - j, j]
            end
            system_data[bw + 1, 1] -= T(l * domain.r[1, 3])
        elseif velocity_bc_code == 1 || velocity_bc_code == 2
            system_data[bw + 1, 1] = one(T)
        else
            ... existing stress-free block unchanged ...
        end
```

(b) `create_magnetic_toroidal_matrices`: add kwarg `inner_regularity::Bool = false`. The inner-row block (insulating identity / conducting Robin) gains a first branch:

```julia
        if inner_regularity
            # Ball center regularity (raw sphtor scalar): t′(r₁) = l·t(r₁)/r₁.
            @inbounds for j in 1:(1 + bw)
                system_data[bw + 1 + 1 - j, j] = d1_matrix.data[bw + 1 + 1 - j, j]
            end
            system_data[bw + 1, 1] -= T(l * domain.r[1, 3])
        elseif inner_alpha === nothing || !haskey(inner_alpha, l)
            ... existing ...
```

(c) `create_magnetic_poloidal_matrices`: add kwarg `inner_regularity::Bool = false`. Inner-row block gains:

```julia
        if inner_regularity
            # Ball center regularity: P ~ r^{l+1} ⇒ P′(r₁) = (l+1)·P(r₁)/r₁.
            @inbounds for j in 1:(1 + bw)
                system_data[bw + 1 + 1 - j, j] = d1_matrix.data[bw + 1 + 1 - j, j]
            end
            system_data[bw + 1, 1] -= T((l + 1) * domain.r[1, 3])
        elseif inner_alpha === nothing || !haskey(inner_alpha, l)
            ... existing insulating inner ...
```

- [ ] **Step 4: Run the magnetic free-decay test. Insulating-row audit.**

If σ ≈ π² → row is fine, done. If σ misses π² (expect ~10–20% off if the row is the suspected one-off): change the OUTER insulating row in `create_magnetic_poloidal_matrices` from `+ T((l + 1) * domain.r[N, 3])` to `+ T(l * domain.r[N, 3])` with this comment:

```julia
        # Insulating outer: under B_r = λP/r² the exterior vacuum solution is
        # P ∝ r^{−l} (B = −∇Φ, Φ ∝ r^{−(l+1)} ⇒ B_r ∝ r^{−(l+2)} = λP/r²).
        # Matching P′/P at r_o gives (∂r + l/r)P = 0. Verified by the classic
        # full-sphere dipole free-decay rate σ = π² (test/ball_bessel_decay.jl).
```

and re-run. If the fix is needed, ALSO audit the inner insulating row the same way (interior vacuum P ∝ r^{l+1} ⇒ row `(∂r − (l+1)/r)P = 0`, i.e. `l` → `l+1` at line ~215) and update `src/timestep/erk2/boundary.jl` `solver_create_insulating_inner_bc`/`solver_create_insulating_outer_bc` call sites in `build_solver_erk2_magnetic_pol_bc` to match (the descriptors take `r_inv` and apply `l_sign·l·r_inv` + `fixed_correction`; outer fix = set `fixed_correction` from `r_inv` to `zero(T)`; inner fix = `fixed_correction` from `zero(T)` to `−r_inv`). Then run the FULL magnetic test set:

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using GeoDynamo; for f in filter(x->occursin("magnetic", x), readdir("test")); include(joinpath("test", f)); end' > /tmp/t3mag.log 2>&1; tail -30 /tmp/t3mag.log
```

Shell magnetic tests asserting specific decay/equilibria may shift if the row was wrong and is now fixed — inspect each failure: tolerance-level shifts in physical-consistency tests are the fix working (update reference values with a comment); structural failures mean the audit conclusion is wrong — revert and investigate before proceeding.

- [ ] **Step 5: Commit**

```bash
git add src/bcs/velocity_bc.jl src/bcs/magnetic_bc.jl src/timestep/erk2/boundary.jl test/ball_bessel_decay.jl
git commit -m "feat(ball): toroidal/magnetic regularity rows + insulating-row audit via dipole free decay"
```

(Adjust the message if the audit found nothing: drop the audit clause.)

---

### Task 4: W-split ball — mixed influence, P-recovery regularity, CNAB2

**Files:**
- Modify: `src/solver/state.jl` (`PoloidalSplitMatrices`, ~line 264)
- Modify: `src/bcs/velocity_bc.jl` (`create_velocity_poloidal_split_matrices`, ~line 491)
- Modify: `src/physics/velocity/solver.jl` (`_apply_poloidal_wsplit_cnab2!`, ~line 265; `_get_or_build_poloidal_split!`, ~line 247)
- Create: `test/ball_wsplit_eigen.jl`

- [ ] **Step 1: Write the failing test** — `test/ball_wsplit_eigen.jl`:

```julia
using Test
using LinearAlgebra
using GeoDynamo
const Ball = GeoDynamo.GeoDynamoBall

# Dense N×N matrix from a banded operator via unit-vector matvecs.
function dense_from_banded(A, n)
    M = zeros(Float64, n, n)
    e = zeros(n); col = zeros(n)
    for j in 1:n
        fill!(e, 0.0); e[j] = 1.0
        mul!(col, A, e)
        M[:, j] = col
    end
    return M
end

@testset "ball W-split decay matches constrained eigenvalue" begin
    nr = 40; l = 2; Ek = 1.0; dt = 2e-5
    cfg = GeoDynamo.create_shtnskit_config(lmax = 4, mmax = 4,
        nlat = 12, nlon = 24, nr = nr)
    dom = Ball.create_ball_radial_domain(nr)
    split = GeoDynamo.create_velocity_poloidal_split_matrices(cfg, dom, Ek, dt;
        velocity_bc_code = 1, ball = true)
    @test split.ball
    rr = dom.r[1:nr, 4]
    r1inv = dom.r[1, 3]

    # ---- independent theory: σ·D_pol·p = D_pol²·p with 4 constraint rows
    # (rows 1: P-regularity Robin; 2: W-regularity Robin applied to D_pol·P;
    #  nr−1: outer no-slip P′(1)=0; nr: outer wall P(1)=0)
    D = dense_from_banded(split.dpol_op[split.lookup[l]], nr)
    d1 = GeoDynamo.create_derivative_matrix(Float64, 1, dom)
    D1 = dense_from_banded(d1, nr)
    A = D * D
    B = copy(D)
    preg = copy(D1[1, :]); preg[1] -= (l + 1) * r1inv
    wreg = vec(preg' * D)
    A[1, :] = preg;             B[1, :] .= 0.0
    A[2, :] = wreg;             B[2, :] .= 0.0
    A[nr - 1, :] = D1[nr, :];   B[nr - 1, :] .= 0.0
    A[nr, :] .= 0.0; A[nr, nr] = 1.0; B[nr, :] .= 0.0
    ev = eigen(A, B)
    finite_real = [real(v) for v in ev.values
                   if isfinite(v) && abs(imag(v)) < 1e-8 && real(v) < -1e-6]
    σ_th = maximum(finite_real)      # slowest decay rate (least negative)

    # ---- numeric: ball CNAB2 W-split kernel on one mode, zero forcing
    idx = split.lookup[l]
    P = @. rr^(l + 1) * (1 - rr^2)   # P(1)=0, regular leading behavior
    W = similar(P); LW = similar(P); rhs = similar(P)
    Wp = similar(P); Pp = similar(P)
    inv_dt = split.mass_coeff / dt
    om = 1 - split.theta
    nsteps = 4000; nhalf = 2000; mid = nr ÷ 2; vh = 0.0
    for s in 1:nsteps
        mul!(W, split.dpol_op[idx], P)
        mul!(LW, split.w_linear[idx], W)
        @. rhs = inv_dt * W + om * LW
        GeoDynamo.solve_banded!(Wp, split.w_factor[idx], rhs)
        rho1 = dot(split.d1_row_inner, Wp) - (l + 1) * split.reg_r_inv * Wp[1]
        Wp[1] = 0.0; Wp[nr] = 0.0
        GeoDynamo.solve_banded!(Pp, split.p_factor[idx], Wp)
        rho2 = dot(split.d1_row_outer, Pp)
        M = split.influence[idx]
        det = M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]
        a1 = (-rho1 * M[2, 2] + rho2 * M[1, 2]) / det
        a2 = (-rho2 * M[1, 1] + rho1 * M[2, 1]) / det
        @. P = Pp + a1 * split.h1[idx] + a2 * split.h2[idx]
        s == nhalf && (vh = P[mid])
    end
    σ_num = log(P[mid] / vh) / ((nsteps - nhalf) * dt)
    @test isapprox(σ_num, σ_th; rtol = 1e-2)
end
```

- [ ] **Step 2: Run — expect FAIL** (no `ball` kwarg / fields).

- [ ] **Step 3: Implement.**

(a) `src/solver/state.jl` — extend the struct (new fields LAST, before closing `end`, after `mass_coeff`):

```julia
struct PoloidalSplitMatrices{T}
    ... existing fields unchanged ...
    theta::Float64
    mass_coeff::Float64
    ball::Bool          # full-sphere: mixed influence rows + regularity recovery
    reg_r_inv::Float64  # 1/r₁ for the regularity Robin rows (0 for shell)
end
```

(b) `create_velocity_poloidal_split_matrices` — add kwarg `ball::Bool = false`; changes inside:

After `N = domain.N` add:
```julia
    reg_r_inv = ball ? domain.r[1, 3] : 0.0
```

Endpoint-row construction: for ball, `d1_row_inner` holds the PURE first-derivative row (the W-regularity residual row; the −(l+1)/r₁ diagonal correction is l-dependent and applied at dot time). Replace the `d1_row_inner[j] = ...` line with:

```julia
            d1_row_inner[j] = ball ? v1 :
                              (inner_noslip ? v1 :
                               d2.data[bw + 1 + 1 - j, j] - T(2 / domain.r[1, 4]) * v1)
```

P-recovery inner row: replace the unconditional `prec_data[bw + 1, 1] = one(T)` with:

```julia
        if ball
            # Center regularity: P ~ r^{l+1} ⇒ P′(r₁) = (l+1)·P(r₁)/r₁.
            @inbounds for j in 1:(1 + bw)
                prec_data[bw + 1 + 1 - j, j] = d1.data[bw + 1 + 1 - j, j]
            end
            prec_data[bw + 1, 1] -= T((l + 1) * domain.r[1, 3])
        else
            prec_data[bw + 1, 1] = one(T)
        end
        prec_data[bw + 1, N] = one(T)
```

(Keep the band-zeroing loops above this unchanged — they run for both.)

Influence matrix: replace the M assembly with:

```julia
        M = Matrix{T}(undef, 2, 2)
        if ball
            # Mixed rows: row 1 = inner W-regularity residual evaluated on the
            # W-space Green columns g_i (NOT on h_i — the condition lives on W);
            # row 2 = outer wall residual on the recovered P responses h_i.
            M[1, 1] = dot(d1_row_inner, gv1) - T((l + 1) * reg_r_inv) * gv1[1]
            M[1, 2] = dot(d1_row_inner, gv2) - T((l + 1) * reg_r_inv) * gv2[1]
        else
            M[1, 1] = dot(d1_row_inner, hv1)
            M[1, 2] = dot(d1_row_inner, hv2)
        end
        M[2, 1] = dot(d1_row_outer, hv1)
        M[2, 2] = dot(d1_row_outer, hv2)
        influence[idx] = M
```

Constructor call gains `ball, reg_r_inv` at the end.

(c) `_apply_poloidal_wsplit_cnab2!` — the residual computation becomes (ball residual must read Wp BEFORE wall-zeroing):

```julia
            solve_banded!(Wp, split.w_factor[idx], rhs)

            # Inner residual: ball evaluates the W-regularity Robin row on the
            # W solution (pre-zeroing); shell evaluates the inner wall row on P.
            rho1w = split.ball ?
                dot(split.d1_row_inner, Wp) -
                T((l + 1) * split.reg_r_inv) * Wp[1] : zero(T)

            Wp[1] = zero(T); Wp[nr] = zero(T)
            solve_banded!(Pp, split.p_factor[idx], Wp)

            rho1 = split.ball ? rho1w : dot(split.d1_row_inner, Pp)
            rho2 = dot(split.d1_row_outer, Pp)
```

(The rest of the influence solve is unchanged.)

(d) `_get_or_build_poloidal_split!` — pass geometry:

```julia
    split = create_velocity_poloidal_split_matrices(
        ...existing args...;
        velocity_bc_code = velocity_bc,
        theta = _timestepper_implicit_theta(state.parameters.timestepper, state.parameters),
        ball = state.parameters.geometry === :ball,
        T = T)
```

(e) Grep for other `PoloidalSplitMatrices{` constructor calls and `create_velocity_poloidal_split_matrices(` call sites (ERK2 integrate builds one too — `src/timestep/erk2/integrate.jl`); update every constructor call for the two new fields, and thread `ball = ` at the integrate site (Task 7 wires its behavior; the kwarg threads now so the struct stays consistent).

```bash
grep -rn "PoloidalSplitMatrices{\|create_velocity_poloidal_split_matrices(" src/ test/
```

- [ ] **Step 4: Run the new test — expect PASS. Then the shell W-split suite:**

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using GeoDynamo; include("test/ball_wsplit_eigen.jl"); include("test/poloidal_momentum_split.jl")' > /tmp/t4.log 2>&1; tail -15 /tmp/t4.log
```
Expected: both PASS (shell path untouched: `ball=false` default reproduces old behavior exactly).

- [ ] **Step 5: Commit**

```bash
git add src/solver/state.jl src/bcs/velocity_bc.jl src/physics/velocity/solver.jl src/timestep/erk2/integrate.jl test/ball_wsplit_eigen.jl
git commit -m "feat(ball): W-split center regularity — mixed 2x2 influence + Robin P-recovery"
```

---

### Task 5: Thread geometry into the backend matrix builders

**Files:**
- Modify: `src/solver/backend.jl` (`build_velocity_implicit_matrices` ~361, `build_magnetic_implicit_matrices` ~372, `solver_build_temperature_implicit_matrix` ~414, `solver_build_composition_implicit_matrix` ~420, `_build_implicit_matrices_dict` ~430)

- [ ] **Step 1: Implement (mechanical threading; the builders' behavior is already tested).**

`_build_implicit_matrices_dict` has `p::SolverParameters`. Define once at the top:

```julia
    inner_regularity = p.geometry === :ball
```

and thread a new trailing positional/kwarg into each wrapper:

```julia
function build_velocity_implicit_matrices(cfg, domain, E, dt, velocity_bc_code;
        inner_regularity::Bool = false)
    return (
        tor = SOLVER_VELOCITY_TOROIDAL_MATRIX_BUILDER(
            cfg, domain, E, dt; velocity_bc_code = velocity_bc_code,
            mass_coeff = E, inner_regularity = inner_regularity
        ),
        pol = SOLVER_VELOCITY_POLOIDAL_MATRIX_BUILDER(
            cfg, domain, E, dt; velocity_bc_code = velocity_bc_code,
            mass_coeff = E
        )
    )
end
```

NOTE: the legacy poloidal matrices (`SOLVER_VELOCITY_POLOIDAL_MATRIX_BUILDER` → `create_velocity_poloidal_matrices`) are NOT given the kwarg — they are dead on the active CNAB2 (W-split) and ERK2 paths; leave them untouched.

```julia
function build_magnetic_implicit_matrices(cfg, domain, dt; inner_regularity::Bool = false)
    return (
        tor = SOLVER_MAGNETIC_TOROIDAL_MATRIX_BUILDER(cfg, domain, 1.0, dt;
            inner_regularity = inner_regularity),
        pol = SOLVER_MAGNETIC_POLOIDAL_MATRIX_BUILDER(cfg, domain, 1.0, dt;
            inner_regularity = inner_regularity)
    )
end
```

`solver_build_temperature_implicit_matrix` / `solver_build_composition_implicit_matrix`: add a 6th positional arg `inner_regularity` passed as the kwarg. Update ALL call sites:

```bash
grep -rn "solver_build_temperature_implicit_matrix(\|solver_build_composition_implicit_matrix(\|build_velocity_implicit_matrices(\|build_magnetic_implicit_matrices(" src/
```

There is a dt-rebuild path that calls the same wrappers (the comment above `_build_implicit_matrices_dict` mentions it) — the grep finds it; thread `inner_regularity` there from its own `params`.

- [ ] **Step 2: Verify no behavior change for shell** — run the existing solver-facing tests:

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using GeoDynamo; include("test/poloidal_momentum_split.jl"); include("test/erk2_integration_step.jl")' > /tmp/t5.log 2>&1; tail -10 /tmp/t5.log
```

- [ ] **Step 3: Commit**

```bash
git add src/solver/backend.jl
git commit -m "feat(ball): thread geometry into implicit-matrix builders"
```

---

### Task 6: Unify nonlinear paths — delete legacy ball branches; ball stepping smoke

**Files:**
- Modify: `src/physics/velocity/solver.jl` (`finish_velocity_nonlinear!` ~37: delete ball branch)
- Modify: `src/solver/numerics.jl` (`apply_induction_nonlinear!` ~1611: delete ball branch; delete `solver_ball_vector_analysis!` ~1823)
- Modify: `src/Ball/Ball.jl` (remove plane-zeroing helpers + exports)
- Create: `test/ball_solver_physics.jl` (first sections)

- [ ] **Step 1: Write the failing test** — `test/ball_solver_physics.jl`:

```julia
using Test
using GeoDynamo

# Build ball solver params by mirroring the construction used in
# test/poloidal_momentum_split.jl (same fixture style), with geometry = :ball
# and radius_ratio = 0.0. Read that file's setup block and reuse its helper
# if one exists; otherwise construct SolverParameters directly with the same
# keyword set it uses, overriding:
#   geometry = :ball, radius_ratio = 0.0
# Small resolution: lmax = mmax = 8, nlat = 18, nlon = 36, nr = 16,
# timestepper = CNAB2, timestep = 1e-5, Ek = 1e-2, Ra = 1e4, Pr = 1.0.

@testset "ball CNAB2 stepping: finite + buoyancy alive" begin
    params = _ball_test_params(; Ra = 1e4)   # helper per above
    state = GeoDynamo.initialize_simulation(Float64, params)
    GeoDynamo.initialize_solver_fields!(state)
    # seed a thermal perturbation on (l=2,m=2) so buoyancy must source poloidal flow
    _seed_temperature_mode!(state, 2, 2, 1e-3)  # same seeding idiom as
        # test/poloidal_momentum_split.jl onset test — copy it
    for i in 1:10
        GeoDynamo.solver_step!(state)
    end
    vel = state.fields.velocity
    pol = parent(vel.poloidal.data_real)
    @test all(isfinite, pol)
    @test maximum(abs, pol) > 0          # buoyancy entered momentum in the ball
    nl = parent(vel.nl_poloidal.data_real)
    @test maximum(abs, nl) > 1e-14       # N_W assembled (projection path live)
end
```

The two helpers `_ball_test_params` / `_seed_temperature_mode!`: copy the exact construction + seeding code from `test/poloidal_momentum_split.jl` (its onset testset builds params and injects a temperature mode after `initialize_solver_fields!`) into this file, parameterized by `geometry = :ball, radius_ratio = 0.0`. Keep them file-local.

- [ ] **Step 2: Run — expect FAIL** at `maximum(abs, pol) > 0` or earlier: ball still routes nonlinear through `solver_ball_vector_analysis!` (legacy potentials ⇒ implicit/W-split mismatch; possibly non-finite or zero).

- [ ] **Step 3: Implement the deletions.**

(a) `finish_velocity_nonlinear!` (`src/physics/velocity/solver.jl:37`) — delete:

```julia
    if geometry === :ball
        return solver_ball_vector_analysis!(
            velocity_fields.advection_physical,
            velocity_fields.nl_toroidal,
            velocity_fields.nl_poloidal
        )
    end
```

(keep the `geometry` kwarg in the signature — callers pass it; add a comment `# geometry-blind since the ball grid has no r=0 node (off-center grid)`).

(b) `apply_induction_nonlinear!` (`src/solver/numerics.jl:1611`) — delete the `if geometry === :ball ... else` wrapper, keeping the solenoidal-convention body unconditionally (and its comment).

(c) Delete `solver_ball_vector_analysis!` (`src/solver/numerics.jl:1823`) entirely. Grep to confirm zero remaining references:

```bash
grep -rn "solver_ball_vector_analysis" src/ test/
```

(d) `src/Ball/Ball.jl`: delete `enforce_ball_scalar_regularity!`, `enforce_ball_vector_regularity!`, `apply_ball_temperature_regularity!`, `apply_ball_composition_regularity!`, `ball_physical_to_spectral!`, `ball_vector_analysis!` and their `export` lines (they zero an r=0 plane that no longer exists; regularity now lives in the matrix rows). Grep for users:

```bash
grep -rn "enforce_ball_\|apply_ball_temperature_regularity\|apply_ball_composition_regularity\|ball_physical_to_spectral!\|ball_vector_analysis!" src/ test/
```

Update `test/ball_finiteness.jl`/`test/ball_roundtrip.jl` if they call the removed helpers: replace the call + zero-plane assertion with plain transform finiteness (the surrounding checks stay).

- [ ] **Step 4: Run** the new test + ball tests + shell physics tests:

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using GeoDynamo; include("test/ball_solver_physics.jl"); include("test/ball_finiteness.jl"); include("test/ball_roundtrip.jl"); include("test/poloidal_momentum_split.jl")' > /tmp/t6.log 2>&1; tail -15 /tmp/t6.log
```
Expected: all PASS. If `static_checks.jl` pins text from `numerics.jl` near the deleted block, repoint it (run `include("test/static_checks.jl")` to check).

- [ ] **Step 5: Commit**

```bash
git add src/physics/velocity/solver.jl src/solver/numerics.jl src/Ball/Ball.jl test/ball_solver_physics.jl test/ball_finiteness.jl test/ball_roundtrip.jl
git commit -m "feat(ball)!: unify nonlinear paths — delete legacy ball analysis (dropped-Q defect dies)"
```

---

### Task 7: ERK2 ball

**Files:**
- Modify: `src/timestep/erk2/boundary.jl` (new regularity descriptor + bc-builder kwargs)
- Modify: `src/timestep/erk2/integrate.jl` (thread geometry at bc-builder + pol-split call sites)
- Modify: `src/physics/velocity/solver.jl` (`_erk2_poloidal_recover!` ball branch)
- Modify: `src/physics/scalar_field_solver_common.jl` (EAB2 scalar bc site — thread for consistency)
- Modify: `test/ball_solver_physics.jl` (add ERK2-vs-CNAB2 section)

- [ ] **Step 1: Write the failing test** — append to `test/ball_solver_physics.jl`:

```julia
@testset "ball ERK2 vs CNAB2 consistency" begin
    # Two states, identical IC, 20 steps each; relative trajectory gap < 5%.
    # Mirrors the shell consistency section in test/poloidal_momentum_split.jl —
    # copy its comparison loop verbatim, swapping params to the ball fixture.
    p_cn = _ball_test_params(; Ra = 1e4)
    p_rk = _ball_test_params(; Ra = 1e4, timestepper = GeoDynamo.ERK2())
    s_cn = GeoDynamo.initialize_simulation(Float64, p_cn)
    s_rk = GeoDynamo.initialize_simulation(Float64, p_rk)
    for s in (s_cn, s_rk)
        GeoDynamo.initialize_solver_fields!(s)
        _seed_temperature_mode!(s, 2, 2, 1e-3)
    end
    for i in 1:20
        GeoDynamo.solver_step!(s_cn)
        GeoDynamo.solver_step!(s_rk)
    end
    a = parent(s_cn.fields.velocity.poloidal.data_real)
    b = parent(s_rk.fields.velocity.poloidal.data_real)
    denom = max(maximum(abs, a), maximum(abs, b), 1e-30)
    @test maximum(abs, a .- b) / denom < 0.05
    @test all(isfinite, b)
end
```

(`GeoDynamo.ERK2()` — match the constructor used by the shell ERK2-vs-CNAB2 testset in `test/poloidal_momentum_split.jl`; if it takes arguments there, take them verbatim.)

- [ ] **Step 2: Run — expect FAIL** (ERK2 ball runs with shell wall descriptors + shell influence rows → inconsistent trajectory or error).

- [ ] **Step 3: Implement.**

(a) `src/timestep/erk2/boundary.jl` — new descriptor (after `solver_create_insulating_outer_bc`):

```julia
"""
    solver_create_regularity_bc(T, d1_row, r_inv; l_offset=1)

Ball-center regularity endpoint: f′(r₁) = (l + l_offset)·f(r₁)/r₁.
l_offset = 1 for poloidal potentials (f ~ r^{l+1}), 0 for raw-sphtor toroidal
scalars and scalar fields (f ~ r^l; l=0 reduces to f′(r₁)=0).
"""
function solver_create_regularity_bc(
        ::Type{T}, d1_row::Vector{T}, r_inv::T; l_offset::Int = 1) where {T}
    return SolverERK2BoundarySide{T}(
        :regularity,
        zero(T),
        copy(d1_row),
        r_inv,
        -one(T),                 # l_sign: self_coeff −= l/r₁
        true,                    # use_l_correction
        -T(l_offset) * r_inv,    # fixed_correction: −l_offset/r₁
        false
    )
end
```

(Field order verified against `solver_create_insulating_inner_bc`: `(type, value, stencil, r_inv, l_sign, use_l_correction, fixed_correction, l0_dirichlet)`; `solver_enforce_erk2_bc!` computes `self_coeff = stencil[b] + fixed_correction + l_sign·l·r_inv` — for offset 1 this gives `d1[1] − (l+1)/r₁` ✓.)

(b) bc-builder kwargs in the same file:

- `build_solver_erk2_scalar_bc(T, domain, boundary_condition; inner_regularity::Bool = false)` — when set, `inner = solver_create_regularity_bc(T, d1_inner, T(domain.r[1,3]); l_offset = 0)` (outer unchanged).
- `build_solver_erk2_velocity_tor_bc(...; inner_regularity::Bool = false)` — when set, `inner = solver_create_regularity_bc(T, d1_inner, r_inv_inner; l_offset = 0)`; skip the rotating-inner-core mode-value block when `inner_regularity` (no inner core in a ball).
- `build_solver_erk2_magnetic_tor_bc(T, nr)` → CHANGE SIGNATURE to `build_solver_erk2_magnetic_tor_bc(T, domain; inner_regularity::Bool = false)` (it needs the d1 row for ball): default path keeps the two Dirichlet sides built from `domain.N`; ball path uses `l_offset = 0` regularity inner. Update its call site (`src/timestep/erk2/integrate.jl` ~709, currently passes `nr`).
- `build_solver_erk2_magnetic_pol_bc(T, domain; inner_regularity::Bool = false)` — ball: `inner = solver_create_regularity_bc(T, d1_inner, r_inv_inner; l_offset = 1)` (outer insulating unchanged, as audited in Task 3).

(c) `src/timestep/erk2/integrate.jl` — at each bc-builder thunk (lines ~589, ~601, ~709, ~713, ~775) pass `inner_regularity = <params>.geometry === :ball` (the integrate scope has the solver parameters; grep for how `temperature_bc_code` reaches line 589 and use the same variable pathway). Also confirm the pol-split build site passes `ball = params.geometry === :ball` (threaded in Task 4).

(d) `_erk2_poloidal_recover!` (`src/physics/velocity/solver.jl` ~389) — ball branch. Replace the residual/influence block (from `Wv[1] = zero(T); Wv[nr] = zero(T)` through the `a1/a2` solve) with:

```julia
            # Ball: inner W-regularity residual must be read BEFORE wall-zeroing,
            # in W-space (Wv = V/Ek). The Green columns g are V-space; carry the
            # matching 1/Ek explicitly so row-1 of M and ρ share one scale.
            rho1w = split.ball ?
                dot(split.d1_row_inner, Wv) -
                T((l + 1) * split.reg_r_inv) * Wv[1] : zero(T)

            Wv[1] = zero(T); Wv[nr] = zero(T)
            solve_banded!(Pt, split.p_factor[idx], Wv)

            phi = phis[cidx]
            for r_idx in 1:nr
                g[r_idx] = c * phi[r_idx, 1]
            end
            m11b = split.ball ?
                T(invEk) * (dot(split.d1_row_inner, g) -
                            T((l + 1) * split.reg_r_inv) * g[1]) : zero(T)
            g[1] = zero(T); g[nr] = zero(T)
            solve_banded!(h1, split.p_factor[idx], g)
            for r_idx in 1:nr
                g[r_idx] = c * phi[r_idx, nr]
            end
            m12b = split.ball ?
                T(invEk) * (dot(split.d1_row_inner, g) -
                            T((l + 1) * split.reg_r_inv) * g[1]) : zero(T)
            g[1] = zero(T); g[nr] = zero(T)
            solve_banded!(h2, split.p_factor[idx], g)

            if split.ball
                m11 = m11b; m12 = m12b
                r1 = rho1w
            else
                m11 = dot(split.d1_row_inner, h1)
                m12 = dot(split.d1_row_inner, h2)
                r1 = dot(split.d1_row_inner, Pt)
            end
            m21 = dot(split.d1_row_outer, h1); m22 = dot(split.d1_row_outer, h2)
            r2 = dot(split.d1_row_outer, Pt)
            det = m11 * m22 - m12 * m21
            a1 = (-r1 * m22 + r2 * m12) / det
            a2 = (-r2 * m11 + r1 * m21) / det
```

NOTE the `g[nr > 1 ? 1 : 1]` is a transcription guard — it is simply `g[1]` (the regularity row acts at radial index 1 for BOTH Green columns); write `g[1]`.

(e) `src/physics/scalar_field_solver_common.jl` line ~137 (`build_solver_erk2_scalar_bc(T, runtime.outer_core_domain, bc_code)` in the EAB2 branch): add `inner_regularity = state.parameters.geometry === :ball`. (EAB2 is unreachable for a full solver run — the velocity gate fires first — but the call must not silently build shell walls for ball if the gate is ever lifted.)

- [ ] **Step 4: Run** the ball consistency test + full ERK2 + shell consistency:

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using GeoDynamo; include("test/ball_solver_physics.jl"); include("test/erk2_integration_step.jl"); include("test/poloidal_momentum_split.jl")' > /tmp/t7.log 2>&1; tail -15 /tmp/t7.log
```

- [ ] **Step 5: Commit**

```bash
git add src/timestep/erk2/boundary.jl src/timestep/erk2/integrate.jl src/physics/velocity/solver.jl src/physics/scalar_field_solver_common.jl test/ball_solver_physics.jl
git commit -m "feat(ball): ERK2 — regularity endpoint descriptors + mixed-row poloidal recovery"
```

---### Task 8: Ball physics tests — onset, subcritical, full-MHD stability; suite registration

**Files:**
- Modify: `test/ball_solver_physics.jl` (extend)
- Modify: `test/runtests.jl` (~line 211: register new files)

- [ ] **Step 1: Append the physics testsets:**

```julia
@testset "ball convective onset (supercritical growth)" begin
    params = _ball_test_params(; Ra = 1e4)
    state = GeoDynamo.initialize_simulation(Float64, params)
    GeoDynamo.initialize_solver_fields!(state)
    _seed_temperature_mode!(state, 2, 2, 1e-3)
    ke(s) = sum(abs2, parent(s.fields.velocity.poloidal.data_real)) +
            sum(abs2, parent(s.fields.velocity.toroidal.data_real))
    GeoDynamo.solver_step!(state)
    ke_early = ke(state)
    @test ke_early > 0
    for i in 1:40
        GeoDynamo.solver_step!(state)
    end
    @test all(isfinite, parent(state.fields.velocity.poloidal.data_real))
    @test ke(state) > ke_early          # growing above onset
end

@testset "ball subcritical decay (bounded transient, eventual decay)" begin
    params = _ball_test_params(; Ra = 1.0)
    state = GeoDynamo.initialize_simulation(Float64, params)
    GeoDynamo.initialize_solver_fields!(state)
    _seed_temperature_mode!(state, 2, 2, 1e-3)
    ke(s) = sum(abs2, parent(s.fields.velocity.poloidal.data_real)) +
            sum(abs2, parent(s.fields.velocity.toroidal.data_real))
    kes = Float64[]
    for i in 1:80
        GeoDynamo.solver_step!(state)
        push!(kes, ke(state))
    end
    ke_peak = maximum(kes)
    @test all(isfinite, kes)
    @test kes[end] < 0.9 * ke_peak      # transient growth allowed; net decay
end

@testset "ball full-MHD stability" begin
    params = _ball_test_params(; Ra = 1e4, include_magnetic = true)
    state = GeoDynamo.initialize_simulation(Float64, params)
    GeoDynamo.initialize_solver_fields!(state)
    _seed_temperature_mode!(state, 2, 2, 1e-3)
    for i in 1:30
        GeoDynamo.solver_step!(state)
    end
    for fld in (state.fields.velocity.poloidal, state.fields.velocity.toroidal,
                state.fields.magnetic.poloidal, state.fields.magnetic.toroidal,
                state.fields.temperature.spectral)
        @test all(isfinite, parent(fld.data_real))
        @test all(isfinite, parent(fld.data_imag))
    end
end
```

`_ball_test_params` extension: add `include_magnetic` passthrough (default false) and a `timestepper` kwarg (already used in Task 7). The magnetic field-path of the fixture follows the shell full-MHD fixture in `test/poloidal_momentum_split.jl`/`test/erk2_integration_step.jl` — copy the magnetic seeding/IC if those tests set one (a small (l=1,m=0) poloidal seed); otherwise the conductive-state induction from the velocity seed is enough for finiteness.

**Tuning recipe (run before asserting):** if onset shows no growth at Ra=1e4, raise Ra ×10 (cap 1e7); if any state goes non-finite within 40 steps, halve `timestep`. If subcritical at Ra=1.0 still grows after 80 steps, the inner regularity rows are wrong — debug, do not retune. Record final constants in the fixture with a comment.

- [ ] **Step 2: Register all new test files** — in `test/runtests.jl` after the `"ball_finiteness.jl",` entry (~line 211) add:

```julia
    "ball_domain.jl",
    "ball_bessel_decay.jl",
    "ball_wsplit_eigen.jl",
    "ball_solver_physics.jl",
```

- [ ] **Step 3: Run the whole ball set together:**

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using GeoDynamo; for f in ("ball_domain.jl","ball_roundtrip.jl","ball_finiteness.jl","ball_bessel_decay.jl","ball_wsplit_eigen.jl","ball_solver_physics.jl"); include(joinpath("test", f)); end' > /tmp/t8.log 2>&1; tail -20 /tmp/t8.log
```

- [ ] **Step 4: Commit**

```bash
git add test/ball_solver_physics.jl test/runtests.jl
git commit -m "test(ball): onset/subcritical/full-MHD physics gates + suite registration"
```

---

### Task 9: Marti et al. (2014) benchmark script

**Files:**
- Create: `scripts/marti_ball_benchmark.jl`

- [ ] **Step 1: Fetch the published targets (do NOT trust memory).** WebSearch:
  - `Marti et al 2014 "full sphere" benchmark Geophysical Journal International hydrodynamic case kinetic energy`
  - `Marti 2014 GJI full sphere benchmark table solution drift frequency`

  Extract and record IN THE SCRIPT HEADER: the hydro case's nondimensionalization (length/time scales, Ra and E definitions, internal-heating profile), boundary conditions (stress-free expected), parameter values, and the published steady kinetic energy + drift frequency with significant digits.

- [ ] **Step 2: Derive the parameter mapping** GeoDynamo ↔ paper, written as a comment block exactly like the Christensen mapping in `scripts/christensen_case0.jl` (factor-of-2 Coriolis, Ra conversion, Ekin normalization — re-derive for the paper's scheme; do not copy Case-0's numbers).

- [ ] **Step 3: Write the script** mirroring `scripts/christensen_case0.jl` structure: one `MARTI_LMAX` env knob deriving `lmax = mmax = LMAX, nlat = 2*LMAX, nlon = 4*LMAX`; `MARTI_NSTEPS`, `MARTI_DT` knobs; `geometry = :ball, radius_ratio = 0.0`; the paper's symmetry seed mode; per-200-step `step  t  Ekin  m{sym}frac  wall` lines; final `MARTI_DONE Ekin=…` marker. Reuse Case-0's gauss-weight physical-grid Ekin integral (adapting the radial trapezoid to the ball domain — inner radius contribution starts at r₁, plus the analytic r<r₁ contribution is O(r₁³) ≈ negligible; note this in a comment).

- [ ] **Step 4: Smoke it** (low res, finite check only):

```bash
MARTI_LMAX=16 MARTI_NSTEPS=200 ~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. scripts/marti_ball_benchmark.jl > /tmp/marti_smoke.log 2>&1; tail -5 /tmp/marti_smoke.log
```
Expected: finite Ekin lines, no NaN, exits 0.

- [ ] **Step 5: Commit; long benchmark run is a REPORT step, not a test** — launch from a frozen copy in the background (like Case-0), report the verdict against the published target when it lands.

```bash
git add scripts/marti_ball_benchmark.jl
git commit -m "bench(ball): Marti et al. 2014 full-sphere benchmark script"
```

---

### Task 10: Full suite + finish

- [ ] **Step 1: Full test suite** (NEVER through `tail` directly — file first):

```bash
~/.julia/juliaup/julia-1.11.1+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()' > /tmp/ball_suite.log 2>&1; echo "exit=$?"; grep -E "Test Summary|FAIL|Error" /tmp/ball_suite.log | tail -20
```

Known flakes: ~3 scalar-IC normalization failures can be spurious — re-run once before investigating. Baseline at branch point was green.

- [ ] **Step 2: Update the spec** — append a Status section to `docs/superpowers/specs/2026-06-11-ball-geometry-mhd-design.md` recording: what validated (Bessel rates, eigen probe, physics gates, audit outcome on the insulating rows), Marti run status, and any deviations from this plan. Commit.

- [ ] **Step 3: Finish** — use superpowers:finishing-a-development-branch (verify tests → present merge/PR options). Note for the merge decision: this branch is based on `test/gate-stage2-gpu-vector-and-eab2` (0f103d0), NOT main — the PR/merge target should be that branch or wherever the ERK2 W-split work is being integrated; flag this to the human.

---

## Self-review notes (already applied)

- Spec coverage: §4→Task 1, §5→Tasks 2–4 (+audit), §6→Task 6, §7→Tasks 4d/7, §8 items 1–2→Tasks 2–4, 3–6→Tasks 6–8, 7→Task 9, file map→all. EAB2 stays gated (no task — by design).
- Toroidal β = l (raw sphtor scalar), NOT l+1 — corrected in spec §5; Tasks 3/7 use l_offset 0 accordingly.
- The W inner-regularity condition lives in the INFLUENCE rows (both timesteppers), not in `w_factor` — one code shape; Task 4 test exercises exactly the production kernel algebra.
