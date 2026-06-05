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
end
