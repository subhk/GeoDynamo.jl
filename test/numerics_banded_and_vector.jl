using Test
using LinearAlgebra

# Unit coverage for solver-layer numerics in src/solver/numerics.jl:
#   * solver_build_banded_A   — implicit radial banded operator for degree l
#   * solver_factorize_banded — banded LU factorization
#   * solve_banded!           — banded triangular solve (inverts the operator)
#   * apply_banded_full!      — banded matvec used to form RHS = A*x
#   * solver_phi1_action_krylov — Krylov action of the φ₁(dt·A) matrix function
#   * the live CPU vector copy-helpers exposed through the public dispatch
#     wrappers fill_vector_coeff_buffer! / store_vector_coefficients! /
#     extract_vector_component! / store_vector_components!.
#
# Every assertion checks a KNOWN-CORRECT value: the operator's exact size and
# bandwidth, a structural (zero-outside-band) property, a solve/round-trip
# identity, and the closed-form scalar φ₁ factor (exp(λΔt)-1)/(λΔt).
#
# Skipped (require a GPU device or live MPI ranks, not exercised here):
#   * the GPU branches of the dispatch wrappers (uses_gpu(config) is false on
#     the CPU test config, so the cpu_* path is taken — that IS what we cover);
#   * multi-rank reduction in extract_vector_component! (the in-place Allreduce
#     is a no-op on one rank, which is the configuration under test).

const _G = GeoDynamo

@testset "Numerics: banded operator + Krylov φ₁ + vector copy helpers" begin

    @testset "solver_build_banded_A structure" begin
        nr = 8
        ν = 0.7
        l = 2
        dom = _G.create_radial_domain(nr)
        A = _G.solver_build_banded_A(Float64, dom, ν, l)

        # Exact shape: square operator of order nr, default radial bandwidth 4.
        @test A isa _G.BandedOperator{Float64}
        @test A.size == nr
        @test A.bandwidth == 4
        @test size(A.data) == (2 * A.bandwidth + 1, nr)

        # Structural property: dense form is zero strictly outside the band.
        dense = _G.solver_banded_to_dense(A)
        @test size(dense) == (nr, nr)
        max_outside_band = 0.0
        for j in 1:nr, i in 1:nr
            if abs(i - j) > A.bandwidth
                max_outside_band = max(max_outside_band, abs(dense[i, j]))
            end
        end
        @test max_outside_band == 0.0
    end

    @testset "solver_factorize_banded + solve_banded! invert the operator" begin
        nr = 8
        ν = 0.7
        l = 2
        dom = _G.create_radial_domain(nr)
        A = _G.solver_build_banded_A(Float64, dom, ν, l)

        # The bare diffusion operator (no mass/BC terms) is singular, so form a
        # well-posed implicit operator M = A + σ·I — exactly the kind of shift the
        # time integrator applies — while still factorizing/solving a genuine
        # solver_build_banded_A operator via the production banded routines.
        σ = 1.0e3
        M = _G.BandedOperator{Float64}(copy(A.data), A.bandwidth, A.size)
        bw = M.bandwidth
        for n in 1:nr
            M.data[bw + 1, n] += σ   # diagonal lives at band row bw+1
        end

        # Known interior solution; b = M*x via the independent banded matvec.
        x = collect(1.0:nr) .+ 0.5
        b = zeros(Float64, nr)
        _G.apply_banded_full!(b, M, x)
        # apply_banded_full! must agree with the dense product.
        Mdense = _G.solver_banded_to_dense(M)
        @test b ≈ Mdense * x atol = 1.0e-10

        lu = _G.solver_factorize_banded(M)
        @test lu isa _G.BandedFactorization{Float64}
        @test lu.size == nr
        @test lu.bandwidth == bw

        recovered = zeros(Float64, nr)
        _G.solve_banded!(recovered, lu, b)
        @test recovered ≈ x atol = 1.0e-10
    end

    @testset "solver_phi1_action_krylov matches scalar φ₁ factor" begin
        # For A = λ·I the matrix function φ₁(dt·A) = ((exp(λΔt)-1)/(λΔt))·I, so the
        # action on any v is the scalar factor times v.
        n = 4
        λ = -0.3
        dt = 0.05
        data = zeros(Float64, 3, n)         # bandwidth 1, diagonal = λ
        for j in 1:n
            data[2, j] = λ
        end
        Aop = _G.BandedOperator{Float64}(data, 1, n)
        Aop_lu = _G.solver_factorize_banded(Aop)
        apply_A!(out, v) = _G.apply_banded_full!(out, Aop, v)

        v = [1.0, 2.0, -1.0, 0.5]
        result = _G.solver_phi1_action_krylov(apply_A!, Aop_lu, v, dt)
        factor = (exp(λ * dt) - 1) / (λ * dt)
        @test result ≈ factor .* v atol = 1.0e-12
        @test all(isfinite, result)

        # Zero input ⇒ exact zero output (norm-guard branch).
        z = _G.solver_phi1_action_krylov(apply_A!, Aop_lu, zeros(Float64, n), dt)
        @test z == zeros(Float64, n)
    end

    @testset "vector spectral coefficient round-trip (store ⇄ fill)" begin
        lmax = 3
        mmax = 3
        nlat = max(lmax + 2, 8)
        nlon = max(2lmax + 1, 8)
        nr = 4
        cfg = _G.create_shtnskit_config(
            lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon,
            nr = nr, optimize_decomp = false)
        dom = _G.create_radial_domain(nr)
        pencil = cfg.pencils.spec
        sf = _G.create_shtns_spectral_field(Float64, cfg, dom, pencil)
        sr = parent(sf.data_real)
        si = parent(sf.data_imag)

        # Known dense coefficient matrix indexed [l+1, m+1] over valid (l,m).
        cm = zeros(ComplexF64, lmax + 1, mmax + 1)
        for l in 0:lmax, m in 0:min(l, mmax)
            cm[l + 1, m + 1] = complex(0.1 * l + 0.3 * m + 1.0,
                m == 0 ? 0.7 : (0.2 * l - 0.1 * m))
        end

        r_local = 2
        # store_vector_coefficients! scatters the dense matrix into slot storage.
        _G.store_vector_coefficients!(sr, si, cm, r_local, cfg)
        # fill_vector_coeff_buffer! gathers slot storage back into a dense buffer.
        buf = ones(ComplexF64, lmax + 1, mmax + 1)   # pre-dirtied; must be overwritten
        _G.fill_vector_coeff_buffer!(buf, sr, si, r_local, cfg)

        # Every valid (l,m) entry must round-trip. The storage rule forces the
        # m==0 imaginary part to zero (the axisymmetric modes are real), so the
        # recovered buffer equals cm with its m==0 imaginary parts dropped.
        max_err = 0.0
        for l in 0:lmax, m in 0:min(l, mmax)
            expected = m == 0 ? complex(real(cm[l + 1, m + 1]), 0.0) : cm[l + 1, m + 1]
            max_err = max(max_err, abs(buf[l + 1, m + 1] - expected))
        end
        @test max_err == 0.0
        # m==0 imaginary part was explicitly zeroed on the way in.
        @test imag(buf[1, 1]) == 0.0
        @test real(buf[1, 1]) == real(cm[1, 1])

        # Untouched radial level stays exactly zero (store wrote only r_local).
        cm2 = zeros(ComplexF64, lmax + 1, mmax + 1)
        _G.fill_vector_coeff_buffer!(cm2, sr, si, 1, cfg)
        @test all(iszero, cm2)
    end

    @testset "physical vector component round-trip (store ⇄ extract)" begin
        lmax = 2
        mmax = 2
        nlat = max(lmax + 2, 8)
        nlon = max(2lmax + 1, 8)
        nr = 3
        cfg = _G.create_shtnskit_config(
            lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon,
            nr = nr, optimize_decomp = false)

        vt = zeros(Float64, nlat, nlon, nr)
        vp = zeros(Float64, nlat, nlon, nr)
        # Known distinct 2D fields.
        vt_field = [Float64(10 * i + j) for i in 1:nlat, j in 1:nlon]
        vp_field = [Float64(100 + i - 2 * j) for i in 1:nlat, j in 1:nlon]

        r_local = 2
        _G.store_vector_components!(vt, vp, vt_field, vp_field, r_local, cfg)

        # The stored slice equals the input field exactly.
        @test vt[:, :, r_local] == vt_field
        @test vp[:, :, r_local] == vp_field
        # Other radial levels remain untouched (exact zero).
        @test all(iszero, @view vt[:, :, 1])
        @test all(iszero, @view vp[:, :, nr])

        # extract_vector_component! reads the slice back (single-rank Allreduce
        # is the identity), recovering the original field.
        out_t = ones(Float64, nlat, nlon)   # pre-dirtied
        _G.extract_vector_component!(out_t, vt, r_local, cfg)
        @test out_t == vt_field

        out_p = ones(Float64, nlat, nlon)
        _G.extract_vector_component!(out_p, vp, r_local, cfg)
        @test out_p == vp_field
    end
end
