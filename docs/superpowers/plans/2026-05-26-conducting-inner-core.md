# Conducting Inner Core (Magnetic) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a conducting inner core actually evolve the inner-core magnetic field (`𝒯ⁱᶜ/𝒫ⁱᶜ`) and couple it to the outer core at the ICB, for the CNAB2 timestepper with equal conductivity and no inner-core rotation.

**Architecture:** Schur-complement / admittance (spec Approach C). Per harmonic degree `l`, analytically eliminate the inner-core interior and impose its effect as a modified inner Robin row `(∂/∂r − α_l)S = φ0` on the outer-core implicit solve, reusing the verified matrix-embedded BC machinery. After the outer-core solve, reconstruct the inner-core field from the ICB value. Insulating is the special case ⇒ default path is byte-for-byte unchanged (opt-in).

**Tech Stack:** Julia, banded matrices (`src/numerics/banded_operators.jl`), spherical-harmonic spectral fields, MPI/PencilArrays. Tests via `Test`/`MPI`.

**Reference spec:** `docs/superpowers/specs/2026-05-26-conducting-inner-core-design.md`

**Runtime note:** juliaup shim is broken here. Run Julia via a direct binary with sandbox disabled:
`JL=~/.julia/juliaup/julia-1.11.8+0.x64.apple.darwin14/bin/julia`
Run one test file: `$JL --project=. -e 'using Test, GeoDynamo; include("test/<name>.jl")'`

**Conventions confirmed during audit:**
- `domain.r[:,4]=r`, `r[:,3]=1/r`, `r[:,2]=1/r²`; radial index increases outward (`r[1]=ri_inner`, `r[N]=ro`).
- Banded element `(i,j)` lives at `data[bw+1+i-j, j]`; diagonal row is `bw+1`.
- Outer-core insulating inner rows (existing, keep as default): poloidal `(∂/∂r − l/r)`, toroidal identity `T=0` (`src/bcs/magnetic_bc.jl`).
- Inner-core ball domain has r=0 regularity helpers in `src/Ball/Ball.jl`.

---

### Task 1: Opt-in enable flag (default insulating unchanged)

**Files:**
- Modify: `src/core/parameters.jl` (add field + validation)
- Modify: `src/api/model.jl` (thread kwarg through both constructors)
- Modify: `src/physics/magnetic/solver.jl:1` (`initialize_magnetic_field!` sets `bc_type_inner`)
- Test: `test/magnetic_conducting_inner_core.jl` (extend)

- [ ] **Step 1: Failing test — flag sets CONTINUITY_MAG on magnetic tor/pol**

Add to `test/magnetic_conducting_inner_core.jl` a `@testset` that builds a state with the new flag and asserts the bc type:

```julia
@testset "conducting flag sets CONTINUITY_MAG bc_type_inner" begin
    params = GeoDynamo.SolverParameters(
        architecture=:cpu, geometry=:shell, nr=16, nr_inner=8,
        lmax=4, mmax=4, nlat=12, nlon=16,
        include_magnetic_field=true, include_composition=false,
        timestepper=GeoDynamo.CNAB2(),
        magnetic_inner_bc=:conducting_inner_core,
    )
    state = GeoDynamo.initialize_simulation(Float64, params)
    GeoDynamo.initialize_fields!(state)
    mag = state.fields.magnetic
    @test all(==(Int(GeoDynamo.CONTINUITY_MAG)), mag.𝒯.bc_type_inner)
    @test all(==(Int(GeoDynamo.CONTINUITY_MAG)), mag.𝒫.bc_type_inner)
end
```

- [ ] **Step 2: Run — expect FAIL** (`MethodError`/`keyword argument magnetic_inner_bc not defined`). Command above.

- [ ] **Step 3: Implement**

In `src/core/parameters.jl` add to the `SolverParameters` struct (near other magnetic/Pm fields) and its keyword list:
```julia
magnetic_inner_bc::Symbol = :insulating   # :insulating | :conducting_inner_core
```
Add validation in the params validator alongside existing `geometry`/`nr_inner` checks:
```julia
if !(params.magnetic_inner_bc in (:insulating, :conducting_inner_core))
    push!(errors, "magnetic_inner_bc = $(params.magnetic_inner_bc) must be :insulating or :conducting_inner_core")
end
if params.magnetic_inner_bc === :conducting_inner_core && params.geometry !== :shell
    push!(errors, "magnetic_inner_bc=:conducting_inner_core requires geometry=:shell")
end
```
Thread `magnetic_inner_bc` through both `api/model.jl` constructors (add kwarg `magnetic_inner_bc::Symbol = :insulating`, pass to `SolverParameters`).
In `initialize_magnetic_field!` (`src/physics/magnetic/solver.jl`), after the `fill!` zeroing block, before returning:
```julia
if state.parameters.magnetic_inner_bc === :conducting_inner_core
    fill!(magnetic.𝒯.bc_type_inner, Int(CONTINUITY_MAG))
    fill!(magnetic.𝒫.bc_type_inner, Int(CONTINUITY_MAG))
end
```
(Confirm `CONTINUITY_MAG` is imported in that module — it is, via `physics/magnetic/field.jl:134`.)

- [ ] **Step 4: Run — expect PASS** for the new testset.

- [ ] **Step 5: (no commit — user gates commits)** Note completion.

---

### Task 2: Inner-core admittance module

**Files:**
- Create: `src/physics/magnetic/inner_core.jl`
- Modify: `src/physics/magnetic/field.jl` (add `include("inner_core.jl")` near the other includes)
- Test: `test/magnetic_inner_core_admittance.jl` (new)

Build, per unique `l`, the inner-core implicit diffusion operator on the ball domain and its ICB admittance. Inner-core grid index `1 = r=0`, index `Nic = ri`.

Operator: `M_ic = (1/dt)I − θ·η·∇²_l` where `∇²_l = ∂²/∂r² + (2/r)∂/∂r − l(l+1)/r²` (reuse `create_radial_laplacian` + the `l(l+1)/r²` diagonal subtraction exactly as in `create_magnetic_poloidal_matrices`).

Boundary rows of `M_ic`:
- Inner (r=0, row 1): regularity ⇒ Dirichlet `S=0` for `l≥1` (identity row). (l=0 magnetic skipped.)
- Outer (r=ri, row Nic): Dirichlet `S = g` (identity row) — we impose the ICB value here.

Admittance `α_l` = ICB radial derivative produced by a unit ICB value with zero interior source and regularity at r=0:
```
solve M_ic x = rhs   where rhs = e_{Nic}  (1 at ICB row, 0 elsewhere, 0 at r=0)
α_l = (d1_ic top row) · x        # one-sided ∂/∂r at r=ri
```
Here `d1_ic = create_derivative_matrix(T, 1, ic_domain)`, top row = row `Nic`.

- [ ] **Step 1: Failing test**

`test/magnetic_inner_core_admittance.jl`:
```julia
using Test, MPI, LinearAlgebra
@testset "inner-core admittance" begin
    if !MPI.Initialized(); MPI.Init(); end
    icdom = GeoDynamo.create_inner_core_domain_for_test(8)   # helper: ball [0,ri], nr=8
    η = 1.0; dt = 1e-3; θ = 0.5
    adm = GeoDynamo.create_inner_core_admittance(Float64, [1,2,3], icdom, η, dt; theta=θ)
    # α_l must be finite, real, and grow with l (stiffer interior response)
    α1 = GeoDynamo.inner_core_alpha(adm, 1)
    α2 = GeoDynamo.inner_core_alpha(adm, 2)
    @test isfinite(α1) && isfinite(α2)
    @test α2 > α1 > 0          # diffusive admittance positive & increasing in l
end
```
(If a public `create_inner_core_domain_for_test` is awkward, build the domain in the test via the same scaled ball builder used in `backend.jl:255-261`.)

- [ ] **Step 2: Run — expect FAIL** (functions undefined).

- [ ] **Step 3: Implement** `src/physics/magnetic/inner_core.jl`

```julia
struct InnerCoreAdmittance{T}
    factor::Vector{BandedLU{T}}     # M_ic LU per stored l
    alpha::Vector{T}                # ICB admittance per stored l
    d1_top::Vector{T}               # one-sided ∂/∂r row at r=ri (dense length Nic)
    lookup::Dict{Int,Int}
    Nic::Int
end

inner_core_alpha(a::InnerCoreAdmittance, l::Int) = a.alpha[a.lookup[l]]

function create_inner_core_admittance(::Type{T}, l_values, ic_domain,
                                      diffusivity::Float64, dt::Float64;
                                      theta::Float64=0.5) where T
    uniq = sort(unique(l_values)); filter!(>(0), uniq)
    Nic = ic_domain.N; bw = radial_bandwidth(ic_domain)
    lap = create_radial_laplacian(ic_domain)
    d1  = create_derivative_matrix(T, 1, ic_domain)
    r_inv_sq = @views ic_domain.r[1:Nic, 2]
    d1_top = T[ (1 <= bw+1+Nic-j <= 2bw+1) ? d1.data[bw+1+Nic-j, j] : zero(T)
                for j in 1:Nic ]
    base = T.(diffusivity .* lap.data)
    facs = Vector{BandedLU{T}}(); alphas = T[]; lk = Dict{Int,Int}()
    for (idx,l) in enumerate(uniq)
        data = copy(base)
        lf = Float64(l*(l+1))
        @inbounds for n in 1:Nic
            data[bw+1, n] -= T(diffusivity*lf*r_inv_sq[n])
        end
        data .*= -T(theta); data[bw+1, :] .+= T(1/dt)
        # row 1 (r=0) -> identity (regularity, l>=1)
        @inbounds for j in 1:(1+bw); data[bw+1+1-j, j] = zero(T); end
        data[bw+1, 1] = one(T)
        # row Nic (r=ri) -> identity (Dirichlet ICB value)
        @inbounds for j in (Nic-bw):Nic; data[bw+1+Nic-j, j] = zero(T); end
        data[bw+1, Nic] = one(T)
        M = BandedMatrix{T}(data, bw, Nic); lu = factorize_banded(M)
        rhs = zeros(T, Nic); rhs[Nic] = one(T)
        x = similar(rhs); solve_banded!(x, lu, rhs)
        push!(facs, lu); push!(alphas, dot(d1_top, x)); lk[l] = idx
    end
    return InnerCoreAdmittance{T}(facs, alphas, d1_top, lk, Nic)
end
```
Add `include("inner_core.jl")` to `src/physics/magnetic/field.jl` and export the two public names in `src/GeoDynamo.jl` if tests reference them as `GeoDynamo.*`.

- [ ] **Step 4: Run — expect PASS.** Numerical debugging note: if `α2 > α1 > 0` fails, check the one-sided `d1_top` sign and that `r[1]=0` is the regularity end (not `ri`).

- [ ] **Step 5: Note completion (no commit).**

---

### Task 3: Conducting outer-core matrix variants (Robin inner row)

**Files:**
- Modify: `src/bcs/magnetic_bc.jl` (add α-parameterized inner row)
- Test: `test/magnetic_conducting_matrix_rows.jl` (new)

Add `create_magnetic_{toroidal,poloidal}_matrices` variants (or an optional `inner_alpha::Union{Dict{Int,Float64},Nothing}=nothing` kwarg) that, when `inner_alpha` is supplied, replace the inner boundary row with `(∂/∂r − α_l)`:
```julia
# inner row: copy d1 row 1, then subtract α_l on diagonal
@inbounds for j in 1:(1+bw)
    system_data[bw+1+1-j, j] = d1_matrix.data[bw+1+1-j, j]
end
system_data[bw+1, 1] -= T(inner_alpha[l])
```
Outer (CMB) row stays insulating in BOTH tor and pol. For toroidal, the conducting inner row is the SAME `(∂/∂r − α_l)` form (replaces the `T=0` identity).

- [ ] **Step 1: Failing test** — extract inner row of the conducting matrix, assert it equals `d1_row1 − α_l·e_1`:
```julia
using Test, MPI, LinearAlgebra
include("magnetic_boundary_numerical.jl")  # reuse _magbc_banded_row, or redefine
@testset "conducting inner row = (∂/∂r − α)" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=4, mmax=4, nlat=12, nlon=16, nr=16)
    dom = GeoDynamo.create_radial_domain(16)
    α = Dict(1=>0.7, 2=>1.3, 3=>2.1, 4=>3.0)
    pol = GeoDynamo.create_magnetic_poloidal_matrices(cfg, dom, 1.0, 1e-3; theta=0.5, T=Float64, inner_alpha=α)
    d1 = GeoDynamo.create_derivative_matrix(1, dom)
    for (idx,l) in enumerate(pol.l_values)
        l==0 && continue
        row = _magbc_banded_row(pol.system_matrices[idx], 1)
        expected = _magbc_banded_row(d1, 1); expected[1] -= α[l]
        @test row ≈ expected atol=1e-12
    end
end
```

- [ ] **Step 2: Run — expect FAIL** (`inner_alpha` kwarg unknown).
- [ ] **Step 3: Implement** the kwarg + inner-row branch in both builders (default `nothing` ⇒ existing insulating rows untouched).
- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Note completion.**

---

### Task 4: Per-step history flux φ0 + RHS injection

**Files:**
- Modify: `src/physics/magnetic/inner_core.jl` (add `inner_core_history_flux`)
- Modify: `src/physics/magnetic/solver.jl` (use φ0 as the inner RHS for conducting)
- Test: `test/magnetic_inner_core_history.jl` (new)

`φ0` is the ICB derivative produced by the inner-core history with zero ICB value:
```
b_ic = (1/dt + (1−θ)·η·∇²_l) S_ic^n      # per (l,m) radial profile
solve M_ic y = b_ic  with y(r=0)=0 and y(ri)=0   (homogeneous BCs)
φ0 = d1_top · y
```
Implement `inner_core_history_flux(adm, l, S_ic_old_profile, lin_op_l)` returning the scalar φ0 for one mode (real and imag handled by the caller). Provide the linear operator `η∇²_l` per l (store it in `InnerCoreAdmittance` as `lin::Vector{BandedMatrix}` to apply `(1/dt + (1−θ)η∇²_l)`).

- [ ] **Step 1: Failing test** — φ0 is 0 for a zero history, nonzero for a nonzero history:
```julia
@testset "history flux" begin
    # build adm as in Task 2; profile zero -> φ0==0; ramped profile -> φ0!=0
    z = zeros(8); @test GeoDynamo.inner_core_history_flux(adm, 1, z) == 0.0
    p = collect(range(0,1;length=8)).^1   # ~r^1 regular profile
    @test GeoDynamo.inner_core_history_flux(adm, 1, p) != 0.0
end
```
- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** `inner_core_history_flux` (+ store `lin` per l in Task 2's struct; update its constructor and the Task 2 test if the signature changes).
- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Note completion.**

---

### Task 5: Inner-core reconstruction after the outer-core solve

**Files:**
- Modify: `src/physics/magnetic/inner_core.jl` (add `reconstruct_inner_core!`)
- Test: `test/magnetic_inner_core_reconstruct.jl` (new)

After the OC solve, for each mode: set `b_ic` from `S_ic^n` (same as Task 4), set ICB Dirichlet `= g = S_oc(ri)`, solve `M_ic S_ic = b_ic` (with `S_ic(ri)=g`, `S_ic(0)=0`), write into the `𝒯ⁱᶜ/𝒫ⁱᶜ` radial profile.

- [ ] **Step 1: Failing test** — reconstruction reproduces the ICB Dirichlet value and r=0 regularity:
```julia
@testset "reconstruct inner core" begin
    g = 0.37; Sold = zeros(8)
    Sic = GeoDynamo.reconstruct_inner_core(adm, 1, g, Sold)  # returns length-8 profile
    @test Sic[end] ≈ g atol=1e-12     # ICB value
    @test Sic[1]   ≈ 0.0 atol=1e-12   # regularity at r=0 (l≥1)
end
```
- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** `reconstruct_inner_core` (in-place `!` variant for the field path).
- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Note completion.**

---

### Task 6: Wire conducting path into the magnetic update + backend

**Files:**
- Modify: `src/solver/backend.jl` (build admittance + conducting matrices when enabled; store in state)
- Modify: `src/physics/magnetic/solver.jl` (`apply_magnetic_{toroidal,poloidal}_implicit_update!` conducting branch)
- Modify: `src/solver/state.jl` (carry `inner_core_admittance` if needed)
- Test: `test/magnetic_conducting_inner_core.jl` (the existing RED acceptance test → GREEN)

When `magnetic_inner_bc==:conducting_inner_core`:
- Backend builds `InnerCoreAdmittance` (tor and pol) from `inner_core_domain`, η=`Pm`-scaled magnetic diffusivity (match what the insulating builders use), dt, θ; builds `:magnetic_tor`/`:magnetic_pol` matrices with `inner_alpha = Dict(l=>α_l)`.
- In the CNAB2 branch of each update: after building the CNAB2 RHS, set the inner boundary RHS row to `φ0` (per mode, real+imag), solve, then `reconstruct_inner_core!` into `𝒯ⁱᶜ/𝒫ⁱᶜ`.
- Supersede `_magnetic_toroidal_inner_bc_increment` (the `−nl_pol` coupling) for the conducting case (guard it to only run for non-conducting, or remove its CONTINUITY_MAG trigger now that CONTINUITY_MAG means admittance coupling).

- [ ] **Step 1: Use the existing acceptance test** `test/magnetic_conducting_inner_core.jl`. Update it to enable via `magnetic_inner_bc=:conducting_inner_core` (Task 1) instead of manually setting bc_type. Add ICB continuity asserts:
```julia
ic_pol = parent(mag.𝒫ⁱᶜ.data_real); oc_pol = parent(mag.𝒫.data_real)
# 𝒫ⁱᶜ at its outer point (ri) ≈ 𝒫 at its inner point (ri), for owned modes
# (compare via local_spectral_value at IC index Nic and OC index 1)
@test maximum(abs, ic_pol) > 1e-12
@test maximum(abs, parent(mag.𝒯ⁱᶜ.data_real)) > 1e-12
# continuity (per representative mode) within 1e-6
```
- [ ] **Step 2: Run — expect FAIL** (IC still zero until wiring lands).
- [ ] **Step 3: Implement** backend build + the two conducting update branches + reconstruction calls.
- [ ] **Step 4: Run — expect PASS** (IC fields nonzero; continuity holds). Numerical debugging expected here: φ0 sign/θ-weighting, mode-index bookkeeping, MPI allreduce of boundary scalars if modes are distributed.
- [ ] **Step 5: Note completion.**

---

### Task 7: Regression + docs flip

**Files:**
- Test: `test/magnetic_boundary_numerical.jl`, `test/magnetic_boundary_static_checks.jl` (must still pass — insulating unchanged)
- Modify: `docs/src/boundary-conditions.md`, `docs/src/configuration.md` (flip status to implemented; document `magnetic_inner_bc=:conducting_inner_core`)

- [ ] **Step 1: Run insulating regression** — both magnetic_boundary tests. Expected: PASS unchanged (conducting is opt-in; `inner_alpha=nothing` default leaves rows identical).
- [ ] **Step 2: Run full new-feature tests** (Tasks 1–6) — all PASS.
- [ ] **Step 3: Update docs** — change the 🚧 "Not yet implemented" warnings for conducting IC to a usage example with `magnetic_inner_bc=:conducting_inner_core`; keep perfect-conductor as not implemented; note scope limits (equal σ, no IC rotation, CNAB2).
- [ ] **Step 4: Re-run** the doc-referenced acceptance test once more to confirm GREEN.
- [ ] **Step 5: Note completion; request user permission before any commit.**

---

## Self-Review

**Spec coverage:**
- Equal-σ pure-diffusion IC solve → Tasks 2,4,5. ✅
- ICB continuity (value + derivative) → value by reconstruction Dirichlet (Task 5), derivative by Robin α row (Task 3) + φ0 (Task 4). ✅
- CMB insulating unchanged → Tasks 3,7. ✅
- Opt-in, default unchanged → Task 1 flag + `inner_alpha=nothing` default (Task 3) + regression (Task 7). ✅
- Acceptance test GREEN → Task 6. ✅
- Follow-ups (variable σ, IC rotation, EAB2/ERK2) intentionally excluded. ✅

**Placeholder scan:** Numerical sign/stencil details are flagged as debugging notes (Tasks 2,6), not placeholders — the tests pin correctness. Test code is concrete. No "TODO/TBD".

**Type consistency:** `InnerCoreAdmittance{T}` fields (`factor`, `alpha`, `d1_top`, `lookup`, `Nic`, plus `lin` added in Task 4) and accessors (`inner_core_alpha`, `inner_core_history_flux`, `reconstruct_inner_core`) are used consistently across Tasks 2,4,5,6. `inner_alpha::Dict{Int,T}` kwarg name consistent in Tasks 3,6. If Task 4 adds `lin` to the struct, update the Task 2 constructor + test (noted inline).

**Known risk concentrated in Task 6** (per-step coupling wiring + numerics). Recommend executing Tasks 1–5 (unit-testable, low-risk) before Task 6.
