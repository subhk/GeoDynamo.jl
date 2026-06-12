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
                   if isfinite(v) && abs(imag(v)) < 1e-6 * max(abs(real(v)), 1.0) && real(v) < -1e-6]
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
    # nhalf discards the transient from the non-eigenmode IC; second half is a clean single-mode decay window
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

@testset "ERK2 ball recovery zeroes the W-regularity residual (Ek≠1)" begin
    if !MPI.Initialized()
        MPI.Init()
    end

    nr = 16; l = 2; Ek = 1e-2; dt = 1e-5
    cfg = GeoDynamo.create_shtnskit_config(lmax = 4, mmax = 4,
        nlat = 12, nlon = 24, nr = nr)
    dom = Ball.create_ball_radial_domain(nr)
    split = GeoDynamo.create_velocity_poloidal_split_matrices(cfg, dom, Ek, dt;
        velocity_bc_code = 1, ball = true)
    idx = split.lookup[l]
    rr = dom.r[1:nr, 4]

    # Build the dense D_pol matrix (same operator ERK2 cache uses)
    D = zeros(nr, nr)
    e = zeros(nr); col = zeros(nr)
    for j in 1:nr
        fill!(e, 0.0); e[j] = 1.0
        mul!(col, split.dpol_op[idx], e); D[:, j] = col
    end

    # Compute φ1 on c·D_pol (c = dt/2 for the half-step)
    c = dt / 2
    A = c .* D
    phi = GeoDynamo.solver_compute_phi1_function(A, exp(A))

    # Smooth V satisfying regularity: V ~ r^(l+1) near origin, scaled by Ek
    V = @. (rr^(l + 1)) * (1.2 - rr) * Ek
    Wv = V ./ Ek

    # W-regularity Robin functional: ρ(x) = d1_row_inner · x − (l+1)/r₁ · x[1]
    rho(x) = dot(split.d1_row_inner, x) - (l + 1) * split.reg_r_inv * x[1]
    r1 = rho(Wv)

    # Pt = p_factor \ Wz  (wall-zeroed Wv)
    Wz = copy(Wv); Wz[1] = 0.0; Wz[nr] = 0.0
    Pt = similar(Wz); GeoDynamo.solve_banded!(Pt, split.p_factor[idx], Wz)

    # Build Green columns once (both paths use the same h1/h2/m21/m22/r2)
    g1_raw = c .* phi[:, 1]
    g2_raw = c .* phi[:, nr]

    g1z = copy(g1_raw); g1z[1] = 0.0; g1z[nr] = 0.0
    h1 = similar(g1z); GeoDynamo.solve_banded!(h1, split.p_factor[idx], g1z)
    g2z = copy(g2_raw); g2z[1] = 0.0; g2z[nr] = 0.0
    h2 = similar(g2z); GeoDynamo.solve_banded!(h2, split.p_factor[idx], g2z)
    m21 = dot(split.d1_row_outer, h1); m22 = dot(split.d1_row_outer, h2)
    r2 = dot(split.d1_row_outer, Pt)

    # Helper: given row-1 entries m11/m12, solve the 2x2 system and return residuals
    function recovery_residuals(m11, m12)
        det_m = m11 * m22 - m12 * m21
        a1 = (-r1 * m22 + r2 * m12) / det_m
        a2 = (-r2 * m11 + r1 * m21) / det_m
        # W-regularity on the implied RHS: Wv + a1*g1_raw + a2*g2_raw
        w_res = rho(Wv .+ a1 .* g1_raw .+ a2 .* g2_raw)
        # Outer P-endpoint on the recovered P output
        p_res = dot(split.d1_row_outer, Pt .+ a1 .* h1 .+ a2 .* h2)
        return w_res, p_res
    end

    # --- RAW-g convention (correct: no invEk factor on row 1) ---
    m11_raw = rho(g1_raw)
    m12_raw = rho(g2_raw)
    w_res_raw, p_res_raw = recovery_residuals(m11_raw, m12_raw)

    # --- BUGGY convention (spurious invEk on row 1 M entries) ---
    invEk = 1.0 / Ek
    m11_bug = invEk * rho(g1_raw)
    m12_bug = invEk * rho(g2_raw)
    w_res_bug, p_res_bug = recovery_residuals(m11_bug, m12_bug)

    @info "ERK2 ball recovery residuals" w_res_raw w_res_bug r1 Ek

    # The raw-g convention zeroes the W-regularity residual to machine precision.
    @test abs(w_res_raw) < 1e-10 * max(abs(r1), 1.0)
    @test abs(p_res_raw) < 1e-10 * max(abs(r2), 1.0)

    # The buggy invEk convention leaves a residual of order (1 - Ek)*r1 — clearly
    # non-zero when Ek << 1 (here Ek=1e-2, so ~99% of r1 is left uncorrected).
    # This assertion documents what the bug produces; the fix must pass the test
    # above while this characterizes the failure mode.
    @test abs(w_res_bug) > 0.5 * abs(r1) * abs(1.0 - Ek)
end
