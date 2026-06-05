using Test
using GeoDynamo
using Random

@testset "GPU Phase 5b — Scalar Gradient" begin
    @testset "phi gradient i·m·s [LOCAL]" begin
        nl, nm, nr = 5, 4, 3
        sr = rand(MersenneTwister(1), nl, nm, nr); si = rand(MersenneTwister(2), nl, nm, nr)
        mvals = Float64.(0:(nm - 1))                 # m per m-slot
        gφr = zeros(nl,nm,nr); gφi = zeros(nl,nm,nr)
        GeoDynamo.gpu_phi_gradient!(gφr, gφi, sr, si, mvals)
        for l in 1:nl, m in 1:nm, r in 1:nr
            @test gφr[l,m,r] == -mvals[m] * si[l,m,r]
            @test gφi[l,m,r] ==  mvals[m] * sr[l,m,r]
        end
    end
end
