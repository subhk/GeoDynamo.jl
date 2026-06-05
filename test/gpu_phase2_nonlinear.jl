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
end
