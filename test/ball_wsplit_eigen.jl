using Test
using LinearAlgebra
using MPI
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
    if !MPI.Initialized()
        MPI.Init()
    end

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
    idx = split.lookup[l]
    D = dense_from_banded(split.dpol_op[idx], nr)
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
    @test !isempty(finite_real)
    σ_th = maximum(finite_real)      # slowest decay rate (least negative)

    # ---- numeric: ball CNAB2 W-split kernel on one mode, zero forcing.
    # This loop mirrors the production ball branch of
    # _apply_poloidal_wsplit_cnab2! (src/physics/velocity/solver.jl):
    # ρ1 = W-regularity Robin row on Wp BEFORE wall-zeroing; ρ2 = outer
    # endpoint row on the recovered P; 2×2 influence solve of ρ + M·a = 0.
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
    @info "ball W-split decay" σ_num σ_th
    @test isapprox(σ_num, σ_th; rtol = 1e-2)
end
