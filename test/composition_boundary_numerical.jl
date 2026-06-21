using Test
using MPI

# Numerical contract for the matrix-embedded composition boundary conditions.
#
# Composition reuses the SAME shared scalar operator/BC embedding as temperature
# (src/bcs/scalar_bc.jl): `create_composition_matrices` forwards to
# `create_scalar_matrices`, and `_composition_bc_code == _thermal_bc_code`. This
# file builds the real production operators, solves a radial profile per
# spherical-harmonic degree, and asserts the solved field satisfies each
# condition (codes 1–4 = DD / DN / ND / NN):
#
#   Dirichlet (FixedTemperature) : C        = value   (identity row)
#   Neumann   (FixedFlux)         : ∂C/∂r    = value   (first-derivative row)
#
# Boundary derivatives are evaluated with the same one-sided stencil the solver
# embeds. The only composition-specific scaling is diffusivity = Pm/Sc, which is
# irrelevant to the BC rows (identity / ∂r are diffusivity-independent). The
# Stefan condition (inner-core solidification) acts on temperature heat flux for
# topography evolution — it is NOT a composition boundary condition, so it is
# out of scope here.
#
# Special case: Neumann at BOTH boundaries (code 4) leaves the l=0 mean mode with
# a null-space gauge freedom, so the solver pins the inner row to Dirichlet for
# l=0 only.

function _compbc_banded_row(bm, i::Int)
    N = bm.size
    bw = bm.bandwidth
    row = zeros(eltype(bm.data), N)
    for j in max(1, i - bw):min(N, i + bw)
        band_row = bw + 1 + i - j
        if 1 <= band_row <= 2 * bw + 1
            row[j] = bm.data[band_row, j]
        end
    end
    return row
end

@testset "Composition boundary-condition numerical satisfaction" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping composition BC numerical tests"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    lmax = 4
    mmax = 4
    nlat = max(lmax + 2, 10)
    nlon = max(2lmax + 1, 16)
    nr = 24

    cfg = GeoDynamo.create_shtnskit_config(
        lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr)
    dom = GeoDynamo.create_radial_domain(nr)
    N = dom.N
    dt = 1.0e-3
    θ = 0.5
    diffusivity = 1.0     # BC rows (identity / ∂r) are diffusivity-independent

    d1 = GeoDynamo.create_derivative_matrix(Float64, 1, dom)

    solve_mode(mats, ll, rhs) = begin
        lu = mats.factorizations[mats.lookup[ll]]
        out = similar(rhs)
        GeoDynamo.solve_banded!(out, lu, copy(rhs))
        out
    end
    mat(code) = GeoDynamo.create_composition_matrices(cfg, dom, diffusivity, dt;
        composition_bc_code = code, theta = θ)

    # Smooth interior RHS; endpoints carry the prescribed boundary values.
    interior = Float64[0.3 * sin(1.7 * dom.r[i, 4]) for i in 1:N]
    rhs_with(inner, outer) = (b = copy(interior); b[1] = inner; b[N] = outer; b)

    val_atol = 1.0e-9    # Dirichlet value residuals enforced ~exactly
    der_atol = 1.0e-6    # Neumann derivative residuals amplify round-off by ~1/h

    @testset "DD (code 1) ⇒ C = value at both boundaries" begin
        C = solve_mode(mat(1), 2, rhs_with(0.8, -0.3))
        @test isapprox(C[1], 0.8; atol = val_atol)
        @test isapprox(C[N], -0.3; atol = val_atol)
    end

    @testset "NN (code 4), l ≥ 1 ⇒ ∂C/∂r = value at both boundaries" begin
        C = solve_mode(mat(4), 2, rhs_with(0.5, -0.2))
        dC = d1 * C
        @test isapprox(dC[1], 0.5; atol = der_atol)
        @test isapprox(dC[N], -0.2; atol = der_atol)
    end

    @testset "DN (code 2) ⇒ C(inner)=value, ∂C/∂r(outer)=value" begin
        C = solve_mode(mat(2), 2, rhs_with(0.8, -0.2))
        dC = d1 * C
        @test isapprox(C[1], 0.8; atol = val_atol)     # Dirichlet inner
        @test isapprox(dC[N], -0.2; atol = der_atol)     # Neumann outer
    end

    @testset "ND (code 3) ⇒ ∂C/∂r(inner)=value, C(outer)=value" begin
        # ND is the model default for composition (flux inner / fixed outer).
        C = solve_mode(mat(3), 2, rhs_with(0.5, -0.3))
        dC = d1 * C
        @test isapprox(dC[1], 0.5; atol = der_atol)     # Neumann inner
        @test isapprox(C[N], -0.3; atol = val_atol)      # Dirichlet outer
    end

    @testset "NN (code 4), l=0 mean mode ⇒ inner Neumann (no Dirichlet pin)" begin
        mats = mat(4)
        e1 = zeros(N);
        e1[1] = 1.0
        # Structural: the l=0 inner row is the SAME Neumann first-derivative
        # stencil as l ≥ 1 — NOT a Dirichlet identity pin on the mean mode.
        inner0 = _compbc_banded_row(mats.system_matrices[mats.lookup[0]], 1)
        inner2 = _compbc_banded_row(mats.system_matrices[mats.lookup[2]], 1)
        @test !(inner0 ≈ e1)
        @test inner0 ≈ inner2
        # Behavioral: the l=0 solve is well-posed — the (mass/dt)I shift lifts the
        # pure-Neumann constant null space — and honors the prescribed inner AND
        # outer flux (not a pinned value).
        C = solve_mode(mats, 0, rhs_with(0.5, -0.2))
        dC = d1 * C
        @test all(isfinite, C)
        @test isapprox(dC[1], 0.5; atol = der_atol)     # inner flux honored
        @test isapprox(dC[N], -0.2; atol = der_atol)     # outer flux honored
    end

    @testset "composition BC code mapping matches BoundaryConditions types" begin
        D = GeoDynamo.FixedTemperature(0.0)
        F = GeoDynamo.FixedFlux(0.0)
        @test GeoDynamo._composition_bc_code(GeoDynamo.BoundaryConditions(inner = D, outer = D)) ==
              1
        @test GeoDynamo._composition_bc_code(GeoDynamo.BoundaryConditions(inner = D, outer = F)) ==
              2
        @test GeoDynamo._composition_bc_code(GeoDynamo.BoundaryConditions(inner = F, outer = D)) ==
              3
        @test GeoDynamo._composition_bc_code(GeoDynamo.BoundaryConditions(inner = F, outer = F)) ==
              4
    end
end
