using Test
using MPI

const FINALIZE_MPI_PARALLEL = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

function fill_global_signature!(arr, pencil)
    θ_range = GeoDynamo.range_local(pencil, 1)
    φ_range = GeoDynamo.range_local(pencil, 2)
    r_range = GeoDynamo.range_local(pencil, 3)
    local_data = parent(arr)

    @inbounds for (i, θ) in enumerate(θ_range), (j, φ) in enumerate(φ_range), (k, r) in enumerate(r_range)
        local_data[i, j, k] = 1_000_000.0 * θ + 1_000.0 * φ + r
    end

    return arr
end

@testset "MPI Parallel Invariants" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping multi-rank invariants"
        return
    end

    if !MPI.Initialized()
        MPI.Init()
    end

    comm = GeoDynamo.get_comm()
    nprocs = GeoDynamo.get_nprocs()

    if nprocs == 1
        @test_skip "requires at least 2 MPI ranks"
    else
        lmax = 4
        mmax = 4
        nlat = max(lmax + 2, 10)
        nlon = max(2 * lmax + 1, 16)

        @testset "Pencil transpose roundtrip preserves global ordering" begin
            cfg = GeoDynamo.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon, nr=6)
            cfg_spec_m = length(GeoDynamo.range_local(cfg.pencils.spec, 2))
            cfg_spec_elements = prod(size(parent(GeoDynamo.create_pencil_array(Float64, cfg.pencils.spec; init=:zero))))
            min_cfg_spec_m = MPI.Allreduce(cfg_spec_m, MPI.MIN, comm)
            max_cfg_spec_m = MPI.Allreduce(cfg_spec_m, MPI.MAX, comm)
            min_cfg_spec_elements = MPI.Allreduce(cfg_spec_elements, MPI.MIN, comm)

            @test min_cfg_spec_m > 0
            @test max_cfg_spec_m <= mmax + 1
            @test min_cfg_spec_elements > 0

            pencils = GeoDynamo.create_pencil_topology(cfg; nr=6, optimize=true)
            plans = GeoDynamo.create_transpose_plans(pencils)

            local_spec_m = length(GeoDynamo.range_local(pencils.spec, 2))
            local_spec_elements = prod(size(parent(GeoDynamo.create_pencil_array(Float64, pencils.spec; init=:zero))))
            min_spec_m = MPI.Allreduce(local_spec_m, MPI.MIN, comm)
            max_spec_m = MPI.Allreduce(local_spec_m, MPI.MAX, comm)
            min_spec_elements = MPI.Allreduce(local_spec_elements, MPI.MIN, comm)

            @test min_spec_m > 0
            @test max_spec_m <= mmax + 1
            @test min_spec_elements > 0

            @test haskey(plans, :θ_to_φ)
            @test haskey(plans, :φ_to_θ)

            θ_src = GeoDynamo.create_pencil_array(Float64, pencils.θ; init=:zero)
            φ_mid = GeoDynamo.create_pencil_array(Float64, pencils.φ; init=:zero)
            θ_back = GeoDynamo.create_pencil_array(Float64, pencils.θ; init=:zero)

            fill_global_signature!(θ_src, pencils.θ)
            GeoDynamo.transpose_with_timer!(φ_mid, θ_src, :theta_to_phi_roundtrip)
            GeoDynamo.transpose_with_timer!(θ_back, φ_mid, :phi_to_theta_roundtrip)

            @test parent(θ_back) == parent(θ_src)

            local_sum_src = sum(parent(θ_src))
            local_sum_mid = sum(parent(φ_mid))
            local_sum_back = sum(parent(θ_back))

            global_sum_src = MPI.Allreduce(local_sum_src, MPI.SUM, comm)
            global_sum_mid = MPI.Allreduce(local_sum_mid, MPI.SUM, comm)
            global_sum_back = MPI.Allreduce(local_sum_back, MPI.SUM, comm)

            @test global_sum_mid == global_sum_src
            @test global_sum_back == global_sum_src
        end

        @testset "Shared topology keeps radial loops synchronized for uneven nr" begin
            uneven_cfg = GeoDynamo.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon, nr=5)
            local_r_count = length(GeoDynamo.range_local(uneven_cfg.pencils.r, 3))
            min_r_count = MPI.Allreduce(local_r_count, MPI.MIN, comm)
            max_r_count = MPI.Allreduce(local_r_count, MPI.MAX, comm)

            @test min_r_count == 5
            @test max_r_count == 5
            @test GeoDynamo.validate_radial_distribution(uneven_cfg.pencils; warn_uneven=false, strict=false)
            @test GeoDynamo.check_transform_synchronization(uneven_cfg; strict=false)
        end
    end

    if MPI.Initialized()
        MPI.Barrier(comm)
        if FINALIZE_MPI_PARALLEL && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
