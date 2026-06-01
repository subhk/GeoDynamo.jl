using Test
using MPI

const FINALIZE_MPI_PARALLEL = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

function fill_global_signature!(arr, pencil)
    θ_range = GeoDynamo.range_local(pencil, 1)
    φ_range = GeoDynamo.range_local(pencil, 2)
    r_range = GeoDynamo.range_local(pencil, 3)
    local_data = parent(arr)

    @inbounds for (i, θ) in enumerate(θ_range), (j, φ) in enumerate(φ_range),
        (k, r) in enumerate(r_range)
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
        # ----------------------------------------------------------------
        # Self-contained GEODYNAMO_PROC_GRID default: if the env var is
        # not already set (e.g. when driven directly by the multi-rank
        # command rather than the shell script), pick a sensible 2D split
        # so the test does not error out on read_proc_grid.
        # ----------------------------------------------------------------
        if !haskey(ENV, "GEODYNAMO_PROC_GRID")
            if nprocs == 4
                ENV["GEODYNAMO_PROC_GRID"] = "2x2"
            elseif nprocs == 2
                ENV["GEODYNAMO_PROC_GRID"] = "2x1"
            else
                # General fallback: θ_ranks = nprocs, r_ranks = 1
                ENV["GEODYNAMO_PROC_GRID"] = "$(nprocs)x1"
            end
        end

        # Parse the actual grid so assertions adapt to it.
        grid_spec   = ENV["GEODYNAMO_PROC_GRID"]
        parts       = split(grid_spec, 'x')
        θ_ranks     = parse(Int, parts[1])
        r_ranks     = parse(Int, parts[2])

        lmax = 4
        mmax = 4
        nlat = max(lmax + 2, 10)
        nlon = max(2 * lmax + 1, 16)
        nr_even   = 6
        nr_uneven = 5   # intentionally not divisible by r_ranks (when r_ranks≥2)

        @testset "Pencil transpose roundtrip preserves global ordering" begin
            cfg = GeoDynamo.create_shtnskit_config(
                lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr_even)

            # ---- spec pencil: l over θ_ranks, m over r_ranks, r LOCAL ----
            # Each rank should own at least 1 m-column.
            cfg_spec_m = length(GeoDynamo.range_local(cfg.pencils.spec, 2))
            cfg_spec_elements = prod(size(parent(GeoDynamo.create_pencil_array(
                Float64, cfg.pencils.spec; init = :zero))))
            min_cfg_spec_m = MPI.Allreduce(cfg_spec_m, MPI.MIN, comm)
            max_cfg_spec_m = MPI.Allreduce(cfg_spec_m, MPI.MAX, comm)
            min_cfg_spec_elements = MPI.Allreduce(cfg_spec_elements, MPI.MIN, comm)

            @test min_cfg_spec_m > 0
            @test max_cfg_spec_m <= mmax + 1
            @test min_cfg_spec_elements > 0

            # ---- r pencil: θ-dist / φ-local / r-dist (Phase 2) ----
            # φ is LOCAL — every rank sees the full longitude strip.
            cfg_r_phi = length(GeoDynamo.range_local(cfg.pencils.r, 2))
            @test cfg_r_phi == nlon

            # θ exact-covers 1:nlat across θ_ranks.
            cfg_r_theta = length(GeoDynamo.range_local(cfg.pencils.r, 1))
            min_theta   = MPI.Allreduce(cfg_r_theta, MPI.MIN, comm)
            max_theta   = MPI.Allreduce(cfg_r_theta, MPI.MAX, comm)
            sum_theta   = MPI.Allreduce(cfg_r_theta, MPI.SUM, comm)
            # Every rank owns at least 1 θ-row; sum across θ-ranks equals nlat (per
            # r-group). With the 2D grid, summing over ALL nprocs counts each θ-row
            # r_ranks times.
            @test min_theta >= 1
            @test sum_theta == nlat * r_ranks

            # r exact-covers 1:nr across r_ranks.
            cfg_r_local = length(GeoDynamo.range_local(cfg.pencils.r, 3))
            min_r       = MPI.Allreduce(cfg_r_local, MPI.MIN, comm)
            max_r       = MPI.Allreduce(cfg_r_local, MPI.MAX, comm)
            sum_r       = MPI.Allreduce(cfg_r_local, MPI.SUM, comm)
            # min>=1, sum across r_ranks equals nr_even (per θ-group), so total = nlat_groups * nr_even.
            @test min_r >= 1
            @test sum_r == nr_even * θ_ranks

            # ---- create_pencil_topology + transpose plans ----
            pencils = GeoDynamo.create_pencil_topology(cfg; nr = nr_even, optimize = true)
            plans = GeoDynamo.create_transpose_plans(pencils)

            local_spec_m = length(GeoDynamo.range_local(pencils.spec, 2))
            local_spec_elements = prod(size(parent(GeoDynamo.create_pencil_array(
                Float64, pencils.spec; init = :zero))))
            min_spec_m = MPI.Allreduce(local_spec_m, MPI.MIN, comm)
            max_spec_m = MPI.Allreduce(local_spec_m, MPI.MAX, comm)
            min_spec_elements = MPI.Allreduce(local_spec_elements, MPI.MIN, comm)

            @test min_spec_m > 0
            @test max_spec_m <= mmax + 1
            @test min_spec_elements > 0

            # spec is r-LOCAL: every rank has the full nr_even radial levels.
            local_spec_r = length(GeoDynamo.range_local(pencils.spec, 3))
            @test local_spec_r == nr_even

            # spec_transform pencil: l over θ_ranks, r over r_ranks, m LOCAL.
            local_st_m = length(GeoDynamo.range_local(pencils.spec_transform, 2))
            @test local_st_m == mmax + 1   # m is full/local in transform orientation

            @test haskey(plans, :θ_to_φ)
            @test haskey(plans, :φ_to_θ)

            θ_src  = GeoDynamo.create_pencil_array(Float64, pencils.θ; init = :zero)
            φ_mid  = GeoDynamo.create_pencil_array(Float64, pencils.φ; init = :zero)
            θ_back = GeoDynamo.create_pencil_array(Float64, pencils.θ; init = :zero)

            fill_global_signature!(θ_src, pencils.θ)
            GeoDynamo.transpose_with_timer!(φ_mid, θ_src, :theta_to_phi_roundtrip)
            GeoDynamo.transpose_with_timer!(θ_back, φ_mid, :phi_to_theta_roundtrip)

            @test parent(θ_back) == parent(θ_src)

            local_sum_src  = sum(parent(θ_src))
            local_sum_mid  = sum(parent(φ_mid))
            local_sum_back = sum(parent(θ_back))

            global_sum_src  = MPI.Allreduce(local_sum_src,  MPI.SUM, comm)
            global_sum_mid  = MPI.Allreduce(local_sum_mid,  MPI.SUM, comm)
            global_sum_back = MPI.Allreduce(local_sum_back, MPI.SUM, comm)

            @test global_sum_mid  == global_sum_src
            @test global_sum_back == global_sum_src
        end

        @testset "r<->lm transpose roundtrip is identity (multi-rank)" begin
            cfg = GeoDynamo.create_shtnskit_config(
                lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr_even)

            # Allocate in solve orientation (spec: l-dist / m-dist / r-local)
            a = GeoDynamo.create_pencil_array(ComplexF64, cfg.pencils.spec; init = :zero)
            rank_offset = ComplexF64(MPI.Comm_rank(comm) + 1)
            p = parent(a)
            for i in eachindex(p); p[i] = rank_offset + im * ComplexF64(i); end
            a0 = copy(parent(a))

            b = GeoDynamo.create_pencil_array(ComplexF64, cfg.pencils.spec_transform; init = :zero)
            GeoDynamo.transpose_solve_to_transform!(b, a)   # spec -> spec_transform
            GeoDynamo.transpose_transform_to_solve!(a, b)   # back to spec

            @test parent(a) == a0   # exact identity (no floating-point error)
        end

        @testset "spec pencil r-local: radial loops synchronized for uneven nr" begin
            # Phase 2 constraint: the SPEC pencil (solve orientation) must keep r LOCAL
            # (every rank sees all nr radial levels) so that per-(l,m)-mode banded radial
            # solves run without any collective synchronisation.  This must hold even when
            # nr is not divisible by r_ranks.
            uneven_cfg = GeoDynamo.create_shtnskit_config(
                lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr_uneven)

            # spec: r LOCAL (every rank = nr_uneven)
            local_r_spec = length(GeoDynamo.range_local(uneven_cfg.pencils.spec, 3))
            min_r_spec   = MPI.Allreduce(local_r_spec, MPI.MIN, comm)
            max_r_spec   = MPI.Allreduce(local_r_spec, MPI.MAX, comm)
            @test min_r_spec == nr_uneven
            @test max_r_spec == nr_uneven

            # pencils.r: r is DISTRIBUTED — each rank owns a slice; slices cover 1:nr.
            local_r_phys = length(GeoDynamo.range_local(uneven_cfg.pencils.r, 3))
            min_r_phys   = MPI.Allreduce(local_r_phys, MPI.MIN, comm)
            sum_r_phys   = MPI.Allreduce(local_r_phys, MPI.SUM, comm)
            @test min_r_phys >= 1                       # every rank has at least one r-level
            @test sum_r_phys == nr_uneven * θ_ranks     # all levels covered (each counted θ_ranks times)

            # validate_radial_distribution checks only r-LOCAL pencils (:spec, :mixed);
            # an uneven r-split on :r / :θ / :φ is expected and should NOT raise.
            @test GeoDynamo.validate_radial_distribution(
                uneven_cfg.pencils; warn_uneven = false, strict = false)
            @test GeoDynamo.check_transform_synchronization(uneven_cfg; strict = false)
        end
    end

    if MPI.Initialized()
        MPI.Barrier(comm)
        if FINALIZE_MPI_PARALLEL && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
