using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
using MPI
using Random
using LinearAlgebra: dot

MPI.Initialized() || MPI.Init()

@testset "GPU Phase 5l — Magnetic Conducting Inner Core (flux + reconstruct)" begin
    nl, nm, Nic = 6, 4, 7          # degrees 0..5, orders 0..3, inner-core points
    bw = 2
    dt = 5.0e-4; theta = 0.5
    inv_dt = 1.0 / dt; weight = 1.0 - theta
    rng = MersenneTwister(51)

    # synthetic per-degree operators for stored degrees 1..lmax (magnetic: no l=0)
    function banddata(N, b; seed, diagdom = false)
        r = MersenneTwister(seed); d = zeros(2b+1, N)
        for j in 1:N, i in max(1,j-b):min(N,j+b); d[b+1+i-j, j] = rand(r) - 0.5; end
        diagdom && (for j in 1:N; d[b+1, j] += 5.0; end)
        d
    end
    stored = collect(1:(nl-1))     # degrees 1..5
    facs = GeoDynamo.BandedLU{Float64}[]; lins = GeoDynamo.BandedMatrix{Float64}[]
    lookup = Dict{Int,Int}()
    for (idx, l) in enumerate(stored)
        M = GeoDynamo.BandedMatrix{Float64}(banddata(Nic, bw; seed = 100 + l, diagdom = true), bw, Nic)
        push!(facs, GeoDynamo.factorize_banded(M))
        push!(lins, GeoDynamo.BandedMatrix{Float64}(banddata(Nic, bw; seed = 200 + l), bw, Nic))
        lookup[l] = idx
    end
    d1_top = rand(rng, Nic) .- 0.5
    alphas = rand(rng, length(stored))     # unused by the two ported fns
    adm = GeoDynamo.InnerCoreAdmittance{Float64}(facs, alphas, d1_top, lookup, Nic, lins, dt, theta)

    # dense inner-core spectral state (real + imag), all (l,m) slots filled
    mk() = (a = zeros(nl, nm, Nic); for li in 1:nl, mi in 1:nm, r in 1:Nic; a[li,mi,r] = rand(rng) - 0.5; end; a)
    S_old_r = mk(); S_old_i = mk()
    g_r = rand(rng, nl, nm) .- 0.5; g_i = rand(rng, nl, nm) .- 0.5

    ic = GeoDynamo.gpu_pack_inner_core(adm, nl, CPU())

    @testset "packer bundle shape + identity-fill for non-stored degrees [LOCAL]" begin
        @test size(ic.lin_ic) == (2bw+1, Nic, nl)
        @test size(ic.lu_ic)  == (2bw+1, Nic, nl)
        @test length(ic.d1_top) == Nic
        @test ic.Nic == Nic && ic.bw == bw
        @test ic.inv_dt == inv_dt && ic.weight == weight
        # degree 0 (slot 1) is non-stored: zero L, identity LU (diag row == 1)
        @test all(==(0.0), ic.lin_ic[:, :, 1])
        @test all(==(1.0), ic.lu_ic[bw+1, :, 1])
        @test all(==(0.0), ic.lu_ic[1:bw, :, 1]) && all(==(0.0), ic.lu_ic[bw+2:2bw+1, :, 1])
        # stored degree 2 (slot 3) carries the operator data
        @test ic.lin_ic[:, :, 3] == lins[lookup[2]].data
        @test ic.lu_ic[:, :, 3]  == facs[lookup[2]].lu
    end

    @testset "history flux == CPU inner_core_history_flux (exact) [LOCAL]" begin
        φ0_r = zeros(nl, nm); φ0_i = zeros(nl, nm)
        GeoDynamo.gpu_inner_core_history_flux!(φ0_r, φ0_i, copy(S_old_r), copy(S_old_i), ic)
        for l in stored, mi in 1:nm
            li = l + 1
            ref_r = GeoDynamo.inner_core_history_flux(adm, l, S_old_r[li, mi, :])
            ref_i = GeoDynamo.inner_core_history_flux(adm, l, S_old_i[li, mi, :])
            @test φ0_r[li, mi] == ref_r
            @test φ0_i[li, mi] == ref_i
        end
        @test all(isfinite, φ0_r) && all(isfinite, φ0_i)
    end

    @testset "reconstruct == CPU reconstruct_inner_core (exact) [LOCAL]" begin
        S_new_r = similar(S_old_r); S_new_i = similar(S_old_i)
        GeoDynamo.gpu_reconstruct_inner_core!(S_new_r, S_new_i, copy(S_old_r), copy(S_old_i), g_r, g_i, ic)
        for l in stored, mi in 1:nm
            li = l + 1
            ref_r = GeoDynamo.reconstruct_inner_core(adm, l, g_r[li, mi], S_old_r[li, mi, :])
            ref_i = GeoDynamo.reconstruct_inner_core(adm, l, g_i[li, mi], S_old_i[li, mi, :])
            @test S_new_r[li, mi, :] == ref_r
            @test S_new_i[li, mi, :] == ref_i
        end
        @test all(isfinite, S_new_r) && all(isfinite, S_new_i)
    end

    @testset "GPU execution + GPU≈CPU parity (Phase-5l gate) [GPU-BOX]" begin
        if !GeoDynamo.gpu_functional()
            @test_skip "requires a functional CUDA GPU"
        else
            # CPU reference outputs
            cφ_r = zeros(nl, nm); cφ_i = zeros(nl, nm)
            GeoDynamo.gpu_inner_core_history_flux!(cφ_r, cφ_i, copy(S_old_r), copy(S_old_i), ic)
            cS_r = similar(S_old_r); cS_i = similar(S_old_i)
            GeoDynamo.gpu_reconstruct_inner_core!(cS_r, cS_i, copy(S_old_r), copy(S_old_i), g_r, g_i, ic)

            d(x) = GeoDynamo.on_architecture(GPU(), x)
            gic = GeoDynamo.gpu_pack_inner_core(adm, nl, GPU())
            gφ_r = d(zeros(nl, nm)); gφ_i = d(zeros(nl, nm))
            GeoDynamo.gpu_inner_core_history_flux!(gφ_r, gφ_i, d(copy(S_old_r)), d(copy(S_old_i)), gic)
            gS_r = d(similar(S_old_r)); gS_i = d(similar(S_old_i))
            GeoDynamo.gpu_reconstruct_inner_core!(gS_r, gS_i, d(copy(S_old_r)), d(copy(S_old_i)), d(g_r), d(g_i), gic)
            @test gφ_r isa CUDA.CuArray
            @test gφ_i isa CUDA.CuArray
            @test gS_r isa CUDA.CuArray
            @test gS_i isa CUDA.CuArray
            @test isapprox(Array(gφ_r), cφ_r; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gφ_i), cφ_i; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gS_r), cS_r; atol = 1e-9, rtol = 1e-8)
            @test isapprox(Array(gS_i), cS_i; atol = 1e-9, rtol = 1e-8)
        end
    end
end
