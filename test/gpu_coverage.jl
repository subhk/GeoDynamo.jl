using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
using MPI
using Random

MPI.Initialized() || MPI.Init()

# Coverage-gap tests for the GPU port: the error/guard paths and the Float32
# kernel path, neither of which the per-phase [LOCAL] tests exercised.

@testset "GPU coverage — error/guard paths + Float32 kernels" begin

    @testset "packer error paths [LOCAL]" begin
        # gpu_pack_banded_lu: empty input + mixed bandwidth/size
        @test_throws ArgumentError GeoDynamo.gpu_pack_banded_lu(GeoDynamo.BandedLU{Float64}[], CPU())
        lu_ok = GeoDynamo.BandedLU{Float64}(zeros(5, 4), 2, 4)
        lu_bad = GeoDynamo.BandedLU{Float64}(zeros(3, 4), 1, 4)        # different bandwidth
        @test_throws ArgumentError GeoDynamo.gpu_pack_banded_lu([lu_ok, lu_bad], CPU())

        # gpu_pack_influence: wrong invG shape (Gre ok, invG not 2×2)
        nr = 6
        bad_inf = Dict(1 => GeoDynamo.ERK2InfluenceOp{Float64}(rand(nr, 2), rand(3, 3), 1))
        @test_throws ArgumentError GeoDynamo.gpu_pack_influence(bad_inf, 4, nr, CPU())

        # gpu_pack_inner_core: factors with mismatched bandwidth
        Nic = 5
        facs = [GeoDynamo.BandedLU{Float64}(zeros(5, Nic), 2, Nic),    # bw 2
                GeoDynamo.BandedLU{Float64}(zeros(3, Nic), 1, Nic)]    # bw 1 (mismatch)
        lins = [GeoDynamo.BandedMatrix{Float64}(zeros(5, Nic), 2, Nic),
                GeoDynamo.BandedMatrix{Float64}(zeros(3, Nic), 1, Nic)]
        adm_bad = GeoDynamo.InnerCoreAdmittance{Float64}(
            facs, zeros(2), zeros(Nic), Dict(1 => 1, 2 => 2), Nic, lins, 1e-4, 0.5)
        @test_throws ArgumentError GeoDynamo.gpu_pack_inner_core(adm_bad, 4, CPU())
    end

    @testset "conducting inner core: CNAB2 packs it, ERK2/RK3 reject it [LOCAL]" begin
        # gpu_magnetic_field_step! takes the packed admittance via its `ic` argument,
        # so CNAB2 supports a conducting inner core end-to-end (parity gated in
        # gpu_phase5m2_magnetic_conducting.jl). The ERK2 and RungeKutta3 device steps
        # run their own magnetic update with no inner-core hook, so they must refuse
        # rather than silently drop the φ0 history-flux boundary condition.
        mk(ts) = GeoDynamo.initialize_solver_state(Float64;
            params = GeoDynamo.SolverParameters(
                geometry = :shell, lmax = 4, mmax = 4, nlat = 12, nlon = 24, nr = 8,
                nr_inner = 4, radial_bandwidth = 3, radius_ratio = 0.35,
                include_magnetic = true, include_composition = false,
                magnetic_inner_bc = :conducting_inner_core, timestepper = ts))

        st = mk(GeoDynamo.CNAB2())
        @test st.magnetic_ic_admittance !== nothing            # conducting → admittance set
        gst = GeoDynamo.build_gpu_solver_state(st)
        @test gst.ic !== nothing                               # ...and it reaches the bundle
        @test size(gst.ic.tor_ic_r, 3) == st.magnetic_ic_admittance.tor.Nic

        @test_throws ErrorException GeoDynamo.build_gpu_solver_state(
            mk(GeoDynamo.ExponentialRungeKutta2()))
        @test_throws ErrorException GeoDynamo.build_gpu_solver_state(
            mk(GeoDynamo.RungeKutta3()))
    end

    @testset "build_gpu_solver_state rejects time-dependent boundary data [LOCAL]" begin
        # The device bundle bakes boundary endpoint VALUES at pack time, while the
        # CPU refreshes them every step (bcs/integration.jl). A moving boundary must
        # be rejected loudly, like the conducting-IC / :ball / topography limits.
        function td_data(::Type{T}) where {T}
            GeoDynamo.bcs.BoundaryData{T}(
                nothing, nothing, T[0.0, 1.0], zeros(T, 4, 8, 2),
                "K", "synthetic time-dependent boundary", "", "temperature",
                true, 4, 8, 2, 1)
        end
        params = GeoDynamo.SolverParameters(
            geometry = :shell, lmax = 4, mmax = 4, nlat = 12, nlon = 24, nr = 8, nr_inner = 4,
            radial_bandwidth = 3, radius_ratio = 0.35,
            include_magnetic = false, include_composition = false)
        st = GeoDynamo.initialize_solver_state(Float64; params = params)
        GeoDynamo.solver_step!(st)
        @test GeoDynamo.build_gpu_solver_state(st) !== nothing      # static BCs: accepted
        st.fields.temperature.boundary_condition_set =
            GeoDynamo.bcs.BoundaryConditionSet{Float64}(
                td_data(Float64), td_data(Float64), "temperature",
                GeoDynamo.bcs.TEMPERATURE, 0.0)
        @test_throws ErrorException GeoDynamo.build_gpu_solver_state(st)
    end

    @testset "gpu_run!(::SolverState) rejects negative nsteps [LOCAL]" begin
        params = GeoDynamo.SolverParameters(
            geometry = :shell, lmax = 4, mmax = 4, nlat = 12, nlon = 24, nr = 8, nr_inner = 4,
            radial_bandwidth = 3, radius_ratio = 0.35,
            include_magnetic = true, include_composition = false)
        st = GeoDynamo.initialize_solver_state(Float64; params = params)
        @test_throws ArgumentError GeoDynamo.gpu_run!(st, -1)
    end

    @testset "Float32 kernels: exact broadcast + finite/≈ KA kernels [LOCAL]" begin
        rng = MersenneTwister(91)
        nlat, nlon, nr = 8, 12, 4
        f32(d...) = Float32.(rand(rng, d...) .- 0.5)

        # gpu_cross! — pure broadcast, exact in Float32
        ar = f32(nlat, nlon, nr); aθ = f32(nlat, nlon, nr); aφ = f32(nlat, nlon, nr)
        br = f32(nlat, nlon, nr); bθ = f32(nlat, nlon, nr); bφ = f32(nlat, nlon, nr)
        or = zeros(Float32, nlat, nlon, nr); oθ = similar(or); oφ = similar(or)
        coeff = 1.5f0
        GeoDynamo.gpu_cross!(or, oθ, oφ, ar, aθ, aφ, br, bθ, bφ, coeff)
        @test eltype(or) == Float32
        @test or == coeff .* (aθ .* bφ .- aφ .* bθ)
        @test oθ == coeff .* (aφ .* br .- ar .* bφ)
        @test oφ == coeff .* (ar .* bθ .- aθ .* br)

        # gpu_buoyancy_add! — force_r += factor·r·s, exact in Float32
        fr = f32(nlat, nlon, nr); fr0 = copy(fr)
        s = f32(nlat, nlon, nr); r_vec = Float32[0.5f0 + 0.1f0 * k for k in 1:nr]
        GeoDynamo.gpu_buoyancy_add!(fr, s, r_vec, 0.7f0)
        @test eltype(fr) == Float32
        @test fr == fr0 .+ 0.7f0 .* reshape(r_vec, 1, 1, :) .* s

        # gpu_batched_banded_solve! — KA substitution kernel, Float32 finite + ≈ Float64
        nl, nm, bw = 4, 3, 2
        function diagdom_lu(::Type{T}, seed) where {T}
            a = zeros(T, 2bw + 1, nr, nl); r = MersenneTwister(seed)
            for li in 1:nl, j in 1:nr, i in max(1, j - bw):min(nr, j + bw)
                a[bw + 1 + i - j, j, li] = T(rand(r) - 0.5)
            end
            for li in 1:nl, j in 1:nr; a[bw + 1, j, li] += T(5); end
            a
        end
        lu32 = diagdom_lu(Float32, 1); B32 = Float32.(rand(rng, nl, nm, nr) .- 0.5)
        X32 = copy(B32)
        GeoDynamo.gpu_batched_banded_solve!(X32, X32, lu32, bw)
        @test eltype(X32) == Float32 && all(isfinite, X32)
        X64 = Float64.(B32)
        GeoDynamo.gpu_batched_banded_solve!(X64, X64, Float64.(lu32), bw)
        @test isapprox(Float64.(X32), X64; atol = 1e-4, rtol = 1e-3)

        # gpu_spectral_curl! — KA stencil kernel, Float32 finite + ≈ Float64
        function band(::Type{T}, N, b, seed) where {T}
            d = zeros(T, 2b + 1, N); r = MersenneTwister(seed)
            for j in 1:N, i in max(1, j - b):min(N, j + b); d[b + 1 + i - j, j] = T(rand(r) - 0.5); end
            d
        end
        d1_32 = band(Float32, nr, bw, 2); d2_32 = band(Float32, nr, bw, 3)
        lfac32 = Float32[l * (l + 1) for l in 0:(nl - 1)]
        r32 = Float32[0.5f0 + 0.1f0k for k in 1:nr]
        rinv32 = 1f0 ./ r32; rinv2_32 = rinv32 .^ 2
        str = Float32.(rand(rng, nl, nm, nr)); sti = Float32.(rand(rng, nl, nm, nr))
        spr = Float32.(rand(rng, nl, nm, nr)); spi = Float32.(rand(rng, nl, nm, nr))
        dtr = similar(str); dti = similar(sti); dpr = similar(spr); dpi = similar(spi)
        GeoDynamo.gpu_spectral_curl!(dtr, dti, dpr, dpi, str, sti, spr, spi,
            d1_32, d2_32, lfac32, rinv32, rinv2_32, r32, bw)
        @test eltype(dtr) == Float32 && all(isfinite, dtr) && all(isfinite, dpr)
        dtr64 = zeros(nl, nm, nr); dti64 = zeros(nl, nm, nr); dpr64 = zeros(nl, nm, nr); dpi64 = zeros(nl, nm, nr)
        GeoDynamo.gpu_spectral_curl!(dtr64, dti64, dpr64, dpi64,
            Float64.(str), Float64.(sti), Float64.(spr), Float64.(spi),
            Float64.(d1_32), Float64.(d2_32), Float64.(lfac32), Float64.(rinv32), Float64.(rinv2_32),
            Float64.(r32), bw)
        @test isapprox(Float64.(dtr), dtr64; atol = 1e-4, rtol = 1e-3)
    end
end
