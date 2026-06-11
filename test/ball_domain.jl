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
