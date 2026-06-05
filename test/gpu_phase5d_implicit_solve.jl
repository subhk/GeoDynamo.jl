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

    @testset "implicit field solve == CPU implicit step [LOCAL]" begin
        nr, bw, nl, nm = 8, 2, 4, 3
        # per-l non-singular (diagonally dominant) banded matrices → LU factors
        function band(::Type{T}, N, bw; seed) where {T}
            rng = MersenneTwister(seed); d = zeros(T, 2bw+1, N)
            for j in 1:N, i in max(1,j-bw):min(N,j+bw)
                d[bw+1+i-j,j] = (i==j) ? (T(2bw)+rand(rng,T)) : (rand(rng,T)-T(0.5))
            end
            GeoDynamo.BandedMatrix{T}(d, bw, N)
        end
        lus = [GeoDynamo.factorize_banded(band(Float64, nr, bw; seed = 90 + l)) for l in 1:nl]
        lub = GeoDynamo.gpu_pack_banded_lu(lus, CPU())
        rng = MersenneTwister(91)
        rhs_r = rand(rng, nl, nm, nr); rhs_i = rand(rng, nl, nm, nr)
        bir = rand(rng, nl, nm); bii = rand(rng, nl, nm); bor = rand(rng, nl, nm); boi = rand(rng, nl, nm)
        x_r = copy(rhs_r); x_i = copy(rhs_i)
        GeoDynamo.gpu_implicit_solve_field!(x_r, x_i, lub, bir, bii, bor, boi, bw)
        # CPU reference: per mode, set BC rows then solve_banded!
        for l in 1:nl, m in 1:nm
            tr = collect(rhs_r[l, m, :]); ti = collect(rhs_i[l, m, :])
            tr[1] = bir[l, m]; ti[1] = bii[l, m]; tr[nr] = bor[l, m]; ti[nr] = boi[l, m]
            GeoDynamo.solve_banded!(tr, lus[l], tr); GeoDynamo.solve_banded!(ti, lus[l], ti)
            @test x_r[l, m, :] == tr
            @test x_i[l, m, :] == ti
        end
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5d gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            nr, bw, nl, nm = 10, 2, 4, 3
            function band(::Type{T}, N, bw; seed) where {T}
                rng = MersenneTwister(seed); dd = zeros(T, 2bw+1, N)
                for j in 1:N, i in max(1,j-bw):min(N,j+bw)
                    dd[bw+1+i-j,j] = (i==j) ? (T(2bw)+rand(rng,T)) : (rand(rng,T)-T(0.5))
                end
                GeoDynamo.BandedMatrix{T}(dd, bw, N)
            end
            lus = [GeoDynamo.factorize_banded(band(Float64, nr, bw; seed = 95 + l)) for l in 1:nl]
            rng = MersenneTwister(96)
            rhs_r = rand(rng,nl,nm,nr); rhs_i = rand(rng,nl,nm,nr)
            bir = rand(rng,nl,nm); bii = rand(rng,nl,nm); bor = rand(rng,nl,nm); boi = rand(rng,nl,nm)
            # CPU
            club = GeoDynamo.gpu_pack_banded_lu(lus, CPU())
            cxr = copy(rhs_r); cxi = copy(rhs_i)
            GeoDynamo.gpu_implicit_solve_field!(cxr, cxi, club, bir, bii, bor, boi, bw)
            # GPU
            d(x) = GeoDynamo.on_architecture(GPU(), x)
            glub = GeoDynamo.gpu_pack_banded_lu(lus, GPU())
            gxr = d(copy(rhs_r)); gxi = d(copy(rhs_i))
            GeoDynamo.gpu_implicit_solve_field!(gxr, gxi, glub, d(bir), d(bii), d(bor), d(boi), bw)
            @test gxr isa CUDA.CuArray
            @test isapprox(Array(gxr), cxr; atol = 1e-12, rtol = 1e-10)
            @test isapprox(Array(gxi), cxi; atol = 1e-12, rtol = 1e-10)
        end
    end
end
