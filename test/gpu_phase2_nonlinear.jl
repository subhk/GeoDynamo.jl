using Test
using GeoDynamo

@testset "GPU Phase 2 — Nonlinear Kernels" begin
    nlat, nlon, nr = 6, 8, 3
    rnd() = rand(Float64, nlat, nlon, nr)

    @testset "scalar advection [LOCAL]" begin
        u_r, u_θ, u_φ = rnd(), rnd(), rnd()
        gr, gθ, gφ = rnd(), rnd(), rnd()
        out = zeros(Float64, nlat, nlon, nr)
        GeoDynamo.gpu_scalar_advection!(out, u_r, u_θ, u_φ, gr, gθ, gφ)
        ref = similar(out)
        @inbounds for i in eachindex(ref)
            ref[i] = -(u_r[i] * gr[i] + u_θ[i] * gθ[i] + u_φ[i] * gφ[i])
        end
        @test out == ref               # exact: same per-element arithmetic
    end

    @testset "cross product (overwrite + accumulate) [LOCAL]" begin
        a_r, a_θ, a_φ = rnd(), rnd(), rnd()
        b_r, b_θ, b_φ = rnd(), rnd(), rnd()
        coeff = 0.37
        or, oθ, oφ = zeros(nlat,nlon,nr), zeros(nlat,nlon,nr), zeros(nlat,nlon,nr)
        GeoDynamo.gpu_cross!(or, oθ, oφ, a_r, a_θ, a_φ, b_r, b_θ, b_φ, coeff)
        # reference: coeff·(a×b)  (cross order matches CPU Lorentz/u×ω/u×B)
        rr = similar(or); rθ = similar(oθ); rφ = similar(oφ)
        @inbounds for i in eachindex(rr)
            rr[i] = coeff * (a_θ[i]*b_φ[i] - a_φ[i]*b_θ[i])
            rθ[i] = coeff * (a_φ[i]*b_r[i] - a_r[i]*b_φ[i])
            rφ[i] = coeff * (a_r[i]*b_θ[i] - a_θ[i]*b_r[i])
        end
        @test or == rr && oθ == rθ && oφ == rφ

        # accumulate variant adds onto existing contents
        base_r, base_θ, base_φ = rnd(), rnd(), rnd()
        ar, aθ, aφ = copy(base_r), copy(base_θ), copy(base_φ)
        GeoDynamo.gpu_cross_add!(ar, aθ, aφ, a_r, a_θ, a_φ, b_r, b_θ, b_φ, coeff)
        @test ar == base_r .+ rr && aθ == base_θ .+ rθ && aφ == base_φ .+ rφ
    end

    @testset "Coriolis subtract [LOCAL]" begin
        u_r, u_θ, u_φ = rnd(), rnd(), rnd()
        sinθ = rand(Float64, nlat); cosθ = rand(Float64, nlat)
        or, oθ, oφ = rnd(), rnd(), rnd()
        base_r, base_θ, base_φ = copy(or), copy(oθ), copy(oφ)
        GeoDynamo.gpu_coriolis_sub!(or, oθ, oφ, u_r, u_θ, u_φ, sinθ, cosθ)
        rr = similar(or); rθ = similar(oθ); rφ = similar(oφ)
        @inbounds for k in 1:nr, j in 1:nlon, i in 1:nlat
            cr = -sinθ[i] * u_φ[i,j,k]
            cθ = -cosθ[i] * u_φ[i,j,k]
            cφ =  cosθ[i] * u_θ[i,j,k] + sinθ[i] * u_r[i,j,k]
            rr[i,j,k] = base_r[i,j,k] - cr
            rθ[i,j,k] = base_θ[i,j,k] - cθ
            rφ[i,j,k] = base_φ[i,j,k] - cφ
        end
        @test or == rr && oθ == rθ && oφ == rφ
    end

    @testset "buoyancy add [LOCAL]" begin
        s = rnd()
        r_vec = collect(range(0.5, 1.0; length = nr))
        factor = 1.7
        force_r = rnd(); base = copy(force_r)
        GeoDynamo.gpu_buoyancy_add!(force_r, s, r_vec, factor)
        ref = similar(force_r)
        @inbounds for k in 1:nr, j in 1:nlon, i in 1:nlat
            ref[i,j,k] = base[i,j,k] + factor * r_vec[k] * s[i,j,k]
        end
        @test force_r == ref
    end
end
