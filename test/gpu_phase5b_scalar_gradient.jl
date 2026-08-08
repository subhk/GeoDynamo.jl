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
        # Reference recurrence. sinθ·∂θY_l = A₊(l)·Y_{l+1} + A₋(l)·Y_{l-1}, so
        # collecting the Y_l term of Σ a_{l'}·sinθ∂θY_{l'} gives
        #   b_l = A₊(l-1)·a_{l-1} + A₋(l+1)·a_{l+1}
        # — the SOURCE a_{l∓1} carries A_±(l∓1), NOT A_±(l). This block previously
        # re-derived the A_±(l) form the kernel itself used, so it characterized the bug
        # instead of checking it (see test/code_review_batchC_fixes.jl for the
        # single-mode analytic anchor that pins the corrected weighting).
        for li in 1:nl, mi in 1:nm, r in 1:nr
            l = li - 1; m = mi - 1
            if l < m
                @test gθr[li,mi,r] == 0.0 && gθi[li,mi,r] == 0.0
                continue
            end
            dtr = 0.0; dti = 0.0
            if l < lmax
                # A₋(l+1)
                cp = -Float64(l+2) * sqrt(Float64((l+m+1)*(l-m+1)) / Float64((2l+1)*(2l+3)))
                dtr += cp * sr[li+1, mi, r]; dti += cp * si[li+1, mi, r]
            end
            if l > m
                # A₊(l−1)
                cm = Float64(l-1) * sqrt(Float64((l+m)*(l-m)) / Float64((2l-1)*(2l+1)))
                dtr += cm * sr[li-1, mi, r]; dti += cm * si[li-1, mi, r]
            end
            @test gθr[li,mi,r] == dtr
            @test gθi[li,mi,r] == dti
        end
    end

    @testset "scalar gradient assembly (∇r/∇θ/∇φ + 1/r) [LOCAL]" begin
        lmax, mmax, nr, bw = 5, 5, 4, 2
        nl, nm = lmax + 1, mmax + 1
        # reuse a banded matrix builder from Phase 5a's test idea
        function band(::Type{TT}, N, bw; seed) where {TT}
            rng = MersenneTwister(seed); d = zeros(TT, 2bw+1, N)
            for j in 1:N, i in max(1,j-bw):min(N,j+bw); d[bw+1+i-j,j] = rand(rng,TT)-TT(0.5); end
            GeoDynamo.BandedMatrix{TT}(d, bw, N)
        end
        d1 = band(Float64, nr, bw; seed = 31)
        sr = zeros(nl,nm,nr); si = zeros(nl,nm,nr); rng = MersenneTwister(33)
        for mi in 1:nm, li in mi:nl, r in 1:nr; sr[li,mi,r]=rand(rng); si[li,mi,r]=rand(rng); end
        mvals = Float64.(0:(nm-1)); rinv = [1.0/(0.5+0.1k) for k in 1:nr]
        grr=zeros(nl,nm,nr); gri=zeros(nl,nm,nr); gtr=zeros(nl,nm,nr); gti=zeros(nl,nm,nr); gpr=zeros(nl,nm,nr); gpi=zeros(nl,nm,nr)
        GeoDynamo.gpu_scalar_gradient!(grr,gri, gtr,gti, gpr,gpi, sr,si, d1.data, mvals, rinv, lmax, bw)
        # ∇r reference: d1·s per (l,m), NO 1/r
        for li in 1:nl, mi in 1:nm
            ref = zeros(nr); GeoDynamo.apply_radial_derivative!(ref, d1, collect(sr[li,mi,:]))
            @test grr[li,mi,:] == ref
        end
        # ∇φ reference: i·m·s, then ×1/r
        for li in 1:nl, mi in 1:nm, r in 1:nr
            @test gpr[li,mi,r] == (-(mvals[mi]) * si[li,mi,r]) * rinv[r]
            @test gpi[li,mi,r] == (mvals[mi] * sr[li,mi,r]) * rinv[r]
        end
        # ∇θ reference: recurrence × 1/r (spot-check a valid mode). Source a_{l∓1} is
        # weighted by A_±(l∓1) — see the recurrence note in the first testset.
        li, mi, r = 4, 2, 2; l = li-1; m = mi-1
        dtr = 0.0
        if l < lmax
            cp = -Float64(l+2)*sqrt(Float64((l+m+1)*(l-m+1))/Float64((2l+1)*(2l+3))); dtr += cp*sr[li+1,mi,r]
        end
        if l > m
            cm = Float64(l-1)*sqrt(Float64((l+m)*(l-m))/Float64((2l-1)*(2l+1))); dtr += cm*sr[li-1,mi,r]
        end
        @test gtr[li,mi,r] == dtr * rinv[r]
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5b gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            lmax, mmax, nr, bw = 6, 6, 4, 2
            nl, nm = lmax + 1, mmax + 1
            function band(::Type{TT}, N, bw; seed) where {TT}
                rng = MersenneTwister(seed); dd = zeros(TT, 2bw+1, N)
                for j in 1:N, i in max(1,j-bw):min(N,j+bw); dd[bw+1+i-j,j]=rand(rng,TT)-TT(0.5); end
                GeoDynamo.BandedMatrix{TT}(dd, bw, N)
            end
            d1 = band(Float64, nr, bw; seed = 41)
            sr = zeros(nl,nm,nr); si = zeros(nl,nm,nr); rng = MersenneTwister(43)
            for mi in 1:nm, li in mi:nl, r in 1:nr; sr[li,mi,r]=rand(rng); si[li,mi,r]=rand(rng); end
            mvals = Float64.(0:(nm-1)); rinv = [1.0/(0.5+0.1k) for k in 1:nr]
            z() = zeros(Float64, nl,nm,nr)
            c = (z(),z(),z(),z(),z(),z())
            GeoDynamo.gpu_scalar_gradient!(c..., sr,si, d1.data, mvals, rinv, lmax, bw)
            d(x) = GeoDynamo.on_architecture(GPU(), x)
            g = (d(z()),d(z()),d(z()),d(z()),d(z()),d(z()))
            GeoDynamo.gpu_scalar_gradient!(g..., d(sr),d(si), d(d1.data), d(mvals), d(rinv), lmax, bw)
            @test g[1] isa CUDA.CuArray
            for k in 1:6
                @test isapprox(Array(g[k]), c[k]; atol = 1e-12, rtol = 1e-10)
            end
        end
    end
end
