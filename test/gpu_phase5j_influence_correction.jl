using Test
using GeoDynamo
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5j — Velocity Poloidal Influence Correction (2×2)" begin
    nl, nm, nr = 5, 4, 6          # degrees 0..4, orders 0..3
    rng = MersenneTwister(5)

    # Per-degree influence ops for degrees 1,2,3 (NOT 0, NOT 4 → those stay no-op).
    influence = Dict{Int, GeoDynamo.ERK2InfluenceOp{Float64}}()
    for l in (1, 2, 3)
        Gre  = rand(rng, nr, 2) .- 0.5
        invG = rand(rng, 2, 2) .- 0.5
        influence[l] = GeoDynamo.ERK2InfluenceOp{Float64}(Gre, invG, l)
    end

    # Random dense spectral field (real + imag), all slots filled.
    x_r0 = rand(rng, nl, nm, nr) .- 0.5
    x_i0 = rand(rng, nl, nm, nr) .- 0.5

    # CPU reference: per (li,mi) the degree is li-1; apply the op if present, else leave.
    function cpu_reference(x0)
        x = copy(x0)
        tmp = Vector{Float64}(undef, nr)
        for li in 1:nl, mi in 1:nm
            l = li - 1
            haskey(influence, l) || continue
            for ir in 1:nr; tmp[ir] = x[li, mi, ir]; end
            GeoDynamo.apply_solver_influence_matrix_correction!(tmp, influence[l], 0.0, 0.0)
            for ir in 1:nr; x[li, mi, ir] = tmp[ir]; end
        end
        return x
    end
    ref_r = cpu_reference(x_r0)
    ref_i = cpu_reference(x_i0)

    @testset "packer shape + zero-fill for missing degrees [LOCAL]" begin
        Gre_b, invG_b = GeoDynamo.gpu_pack_influence(influence, nl, nr, CPU())
        @test size(Gre_b) == (nr, 2, nl)
        @test size(invG_b) == (2, 2, nl)
        # degree 0 (slot 1) and degree 4 (slot 5) have no op → all-zero packed columns
        @test all(==(0.0), Gre_b[:, :, 1]) && all(==(0.0), invG_b[:, :, 1])
        @test all(==(0.0), Gre_b[:, :, 5]) && all(==(0.0), invG_b[:, :, 5])
        # degree 2 (slot 3) carries the op's data exactly
        @test Gre_b[:, :, 3]  == influence[2].Gre
        @test invG_b[:, :, 3] == influence[2].invG
    end

    @testset "correction == CPU reference (exact) [LOCAL]" begin
        Gre_b, invG_b = GeoDynamo.gpu_pack_influence(influence, nl, nr, CPU())
        x_r = copy(x_r0); x_i = copy(x_i0)
        GeoDynamo.gpu_velocity_poloidal_influence_correction!(x_r, x_i, Gre_b, invG_b)
        @test x_r == ref_r
        @test x_i == ref_i
        @test all(isfinite, x_r) && all(isfinite, x_i)
    end

    @testset "missing-degree modes are untouched [LOCAL]" begin
        Gre_b, invG_b = GeoDynamo.gpu_pack_influence(influence, nl, nr, CPU())
        x_r = copy(x_r0); x_i = copy(x_i0)
        GeoDynamo.gpu_velocity_poloidal_influence_correction!(x_r, x_i, Gre_b, invG_b)
        # degree 0 (slot 1) and degree 4 (slot 5) unchanged
        @test x_r[1, :, :] == x_r0[1, :, :]
        @test x_r[5, :, :] == x_r0[5, :, :]
        @test x_i[1, :, :] == x_i0[1, :, :]
        @test x_i[5, :, :] == x_i0[5, :, :]
    end

    @testset "packer rejects wrong-shaped op [LOCAL]" begin
        bad = Dict(1 => GeoDynamo.ERK2InfluenceOp{Float64}(rand(nr+1, 2), rand(2, 2), 1))
        @test_throws ArgumentError GeoDynamo.gpu_pack_influence(bad, nl, nr, CPU())
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5j gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            d(x) = GeoDynamo.on_architecture(GPU(), x)
            Gre_b, invG_b = GeoDynamo.gpu_pack_influence(influence, nl, nr, GPU())
            gx_r = d(copy(x_r0)); gx_i = d(copy(x_i0))
            GeoDynamo.gpu_velocity_poloidal_influence_correction!(gx_r, gx_i, Gre_b, invG_b)
            @test gx_r isa CUDA.CuArray
            @test gx_i isa CUDA.CuArray
            @test isapprox(Array(gx_r), ref_r; atol = 1e-12, rtol = 1e-10)
            @test isapprox(Array(gx_i), ref_i; atol = 1e-12, rtol = 1e-10)
        end
    end
end
