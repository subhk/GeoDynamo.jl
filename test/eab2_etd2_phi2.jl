using Test
using GeoDynamo
using MPI
using LinearAlgebra
using Random

MPI.Initialized() || MPI.Init()

# EAB2 / ETD2 repair.
#
# FIXED (φ₂ order): the second-order exponential Adams-Bashforth weights the
# previous-step nonlinear term by the DISTINCT matrix function φ₂(hL) =
# (eᶻ−1−z)/z², not φ₁. solver_phi2_action_krylov implements φ₂(dtA)v =
# (1/dt)A⁻¹(φ₁(dtA)v − v); the update is now
# u_{n+1} = e^{hL}u + h[φ₁·N_n + φ₂·(N_n − N_{n-1})].
#
# FIXED (operator): solver_build_banded_A carried NO boundary rows, so it was
# rank-deficient and the φ A⁻¹ solves hit a singular LU (EAB2 never completed a
# step). It now embeds a homogeneous-Dirichlet ETD operator (non-singular
# interior Dirichlet Laplacian; wall nodes decoupled and re-projected after the
# Krylov action), so the φ solves are well-posed for every l.

@testset "solver_phi2_action_krylov matches dense φ₂ (non-singular operator)" begin
    nr = 8
    # Distinct negative eigenvalues ⇒ non-singular diagonal banded operator.
    diagv = Float64[-(1.0 + 0.37 * i) for i in 1:nr]
    A = GeoDynamo.BandedOperator{Float64}(Matrix{Float64}(reshape(diagv, 1, nr)), 0, nr)
    Aop = GeoDynamo.SolverBandedAction(A)
    lu = GeoDynamo.solver_factorize_banded(A)
    Adense = Matrix(Diagonal(diagv))

    Random.seed!(3)
    v = randn(nr)
    dt = 0.2 / maximum(abs, diagv)   # keep dt·A small for the dense Taylor reference

    got1 = GeoDynamo.solver_phi1_action_krylov(Aop, lu, v, dt)
    got2 = GeoDynamo.solver_phi2_action_krylov(Aop, lu, v, dt)
    exp1 = GeoDynamo.phi1_series(dt .* Adense) * v
    exp2 = GeoDynamo.phi2_series(dt .* Adense) * v

    @test all(isfinite, got2)
    @test isapprox(got1, exp1; rtol = 1e-6, atol = 1e-10)   # φ₁ sanity
    @test isapprox(got2, exp2; rtol = 1e-6, atol = 1e-10)   # φ₂ correctness
end

# Helper: does an EAB2 full-MHD step complete with finite output?
function _eab2_step_runs()
    params = GeoDynamo.SolverParameters(
        geometry = :shell, lmax = 4, mmax = 4, nlat = 10, nlon = 20, nr = 10,
        nr_inner = 4, radial_bandwidth = 3, radius_ratio = 0.35,
        Ek = 1e-3, Ra = 1e4, Pm = 1.0, Pr = 1.0, timestep = 1e-4,
        include_magnetic = true, include_composition = true,
        timestepper = GeoDynamo.ExponentialAdamsBashforth2(),
    )
    try
        st = GeoDynamo.initialize_solver_state(Float64; params)
        GeoDynamo.solver_step!(st)
        GeoDynamo.solver_step!(st)
        return all(isfinite, parent(st.fields.temperature.spectral.data_real)) &&
               all(isfinite, parent(st.fields.magnetic.poloidal.data_real))
    catch
        return false
    end
end

@testset "EAB2 linear operator non-singular ⇒ φ A⁻¹ solves well-posed (l=0..5)" begin
    # The operator fix: solver_build_banded_A previously had no boundary rows and
    # was rank-deficient → solver_factorize_banded produced a singular LU and the
    # φ A⁻¹ solves crashed for EVERY l. The homogeneous-Dirichlet ETD operator is
    # full-rank, so factorization + φ₁/φ₂ actions are finite for every degree.
    dom = GeoDynamo.create_radial_domain(10)
    nr = dom.N
    Random.seed!(1)
    v = randn(nr)
    for l in (0, 1, 2, 5)
        A = GeoDynamo.solver_build_banded_A(Float64, dom, 1.0, l)
        Aop = GeoDynamo.SolverBandedAction(A)
        lu = GeoDynamo.solver_factorize_banded(A)   # previously a singular-LU crash
        p1 = GeoDynamo.solver_phi1_action_krylov(Aop, lu, v, 1e-3)
        p2 = GeoDynamo.solver_phi2_action_krylov(Aop, lu, v, 1e-3)
        @test all(isfinite, p1)
        @test all(isfinite, p2)
    end
end

@testset "EAB2 full-MHD step (velocity-poloidal W-split port still pending)" begin
    # φ₂ order + the singular operator are now FIXED. A full step is still gated
    # by apply_velocity_poloidal_implicit_update! (src/physics/velocity/solver.jl),
    # which is CNAB2-only until the exponential velocity-poloidal is ported to the
    # W-split (the documented Stage-4B follow-up). Known-pending, not a regression.
    @test_broken _eab2_step_runs()
end
