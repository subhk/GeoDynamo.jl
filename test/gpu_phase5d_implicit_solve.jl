using Test
using GeoDynamo
using Random

@testset "GPU Phase 5d — Implicit Solve" begin
    @testset "apply BC rows [LOCAL]" begin
        nl, nm, nr = 4, 3, 6
        x_r = rand(MersenneTwister(1), nl, nm, nr); x_i = rand(MersenneTwister(2), nl, nm, nr)
        bir = rand(MersenneTwister(3), nl, nm); bii = rand(MersenneTwister(4), nl, nm)
        bor = rand(MersenneTwister(5), nl, nm); boi = rand(MersenneTwister(6), nl, nm)
        x0r = copy(x_r); x0i = copy(x_i)
        GeoDynamo.gpu_apply_bc_rows!(x_r, x_i, bir, bii, bor, boi)
        @test x_r[:, :, 1] == bir && x_i[:, :, 1] == bii          # inner row set
        @test x_r[:, :, nr] == bor && x_i[:, :, nr] == boi        # outer row set
        @test x_r[:, :, 2:(nr-1)] == x0r[:, :, 2:(nr-1)]          # interior untouched
        @test x_i[:, :, 2:(nr-1)] == x0i[:, :, 2:(nr-1)]
    end
end
