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

    @testset "theta gradient (l±1,m) recurrence [LOCAL]" begin
        lmax, mmax, nr = 6, 6, 3
        nl, nm = lmax + 1, mmax + 1
        # band-limited source: only valid (l>=m) slots populated
        sr = zeros(nl,nm,nr); si = zeros(nl,nm,nr)
        rng = MersenneTwister(5)
        for mi in 1:nm, li in mi:nl, r in 1:nr     # l>=m
            sr[li,mi,r] = rand(rng); si[li,mi,r] = rand(rng)
        end
        gθr = fill(NaN, nl,nm,nr); gθi = fill(NaN, nl,nm,nr)
        GeoDynamo.gpu_theta_gradient!(gθr, gθi, sr, si, lmax)
        # independent reference: exact CPU recurrence
        for li in 1:nl, mi in 1:nm, r in 1:nr
            l = li - 1; m = mi - 1
            if l < m
                @test gθr[li,mi,r] == 0.0 && gθi[li,mi,r] == 0.0
                continue
            end
            dtr = 0.0; dti = 0.0
            if l < lmax
                ap = Float64(l) * sqrt(Float64((l+m+1)*(l-m+1)) / Float64((2l+1)*(2l+3)))
                dtr += ap * sr[li+1, mi, r]; dti += ap * si[li+1, mi, r]
            end
            if l > m
                am = -Float64(l+1) * sqrt(Float64((l+m)*(l-m)) / Float64((2l-1)*(2l+1)))
                dtr += am * sr[li-1, mi, r]; dti += am * si[li-1, mi, r]
            end
            @test gθr[li,mi,r] == dtr
            @test gθi[li,mi,r] == dti
        end
    end
end
