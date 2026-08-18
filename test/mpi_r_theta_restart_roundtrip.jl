using Test
using MPI
using NCDatasets
using LinearAlgebra

const FINALIZE_MPI_RT_RESTART = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

# Multi-rank restart round-trip under the r×θ 2D decomposition.
#
# `read_restart!` / `_load_restart_file` slice physical fields as
# `[θ_range, φ_range, :]` — taking the FULL radial extent. That is correct only
# when r is LOCAL (Phase 1). Under Phase 2 (`GEODYNAMO_PROC_GRID` with r_ranks>1)
# `pencils.r` distributes r, so each rank must read back only its local r-slab;
# reading all of r yields the wrong shape/data and silently corrupts a resumed
# run. This test writes a restart in which every physical cell encodes its
# GLOBAL (θ,φ,r) index, then each rank reads its slab back and asserts bit-exact
# equality with the slab it owns.
#
# At 1 rank it degrades to a serial round-trip (still a valid regression). The
# real r-distributed assertion needs `GEODYNAMO_PROC_GRID=<θ>x<r>` with r>1 plus
# `mpiexec`, driven by test/run_mpi_r_theta_restart_roundtrip.sh.
@testset "r×θ distributed restart round-trip" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping r×θ restart round-trip"
        return
    end
    MPI.Initialized() || MPI.Init()

    comm = GeoDynamo.output_comm()
    rank = MPI.Comm_rank(comm)

    # Probe parallel NetCDF (MPI-IO); skip cleanly if unavailable. The probe is
    # collective and picks a rank-identical path internally, so every rank must
    # reach it and every rank takes the same branch.
    parallel_ok = let probe_err = GeoDynamo.parallel_netcdf_probe(comm)
        probe_err === nothing ||
            @warn "Parallel NetCDF unavailable; skipping r×θ restart round-trip" error = probe_err
        probe_err === nothing
    end

    if parallel_ok
        lmax = 4
        mmax = 4
        nlat = max(lmax + 2, 10)
        nlon = max(2lmax + 1, 16)
        nr = 8

        cfg = GeoDynamo.create_shtnskit_config(
            lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr)
        pencils = cfg.pencils
        dom = GeoDynamo.create_radial_domain(nr)
        radial_nodes = Float64.(dom.r[1:nr, 4])

        # Local slab on the physical pencil (θ-dist / φ-local / r-dist in Phase 2).
        θr = GeoDynamo.range_local(pencils.r, 1)
        φr = GeoDynamo.range_local(pencils.r, 2)
        rr = GeoDynamo.range_local(pencils.r, 3)

        # Integer, exactly-representable, disjoint encoding of the GLOBAL index.
        encode(gθ, gφ, gr) = gθ * 1.0 + gφ * 1_000.0 + gr * 1_000_000.0
        localT = Array{Float64}(undef, length(θr), length(φr), length(rr))
        for (k, gr) in enumerate(rr), (j, gφ) in enumerate(φr), (i, gθ) in enumerate(θr)
            localT[i, j, k] = encode(gθ, gφ, gr)
        end
        fields = Dict{String, Any}("temperature" => localT)

        # Output dir must be identical on every rank (collective parallel write).
        tmpdir = get(ENV, "RTHETA_RESTART_TMPDIR",
            joinpath(tempdir(), "geodynamo_rtheta_restart_rt"))
        if rank == 0 && !isdir(tmpdir)
            mkpath(tmpdir)
        end
        MPI.Barrier(comm)

        base = GeoDynamo.default_config(Float64)
        config = GeoDynamo.OutputConfig(
            base.output_space, tmpdir, base.filename_prefix,
            base.include_metadata, base.include_grid, base.include_diagnostics,
            Float64, base.spectral_lmax_output, true,
            base.output_interval, base.restart_interval, base.max_output_time,
            base.time_tolerance)

        tracker = GeoDynamo.create_time_tracker(config, 0.0)
        tracker.output_count = 2
        metadata = Dict{String, Any}("current_time" => 1.0, "current_step" => 5)

        GeoDynamo.write_restart!(fields, tracker, metadata, config, pencils;
            shtns_config = cfg, geometry = :shell,
            radius_ratio = 0.35, radial_grid = radial_nodes)
        MPI.Barrier(comm)

        reader = GeoDynamo.create_time_tracker(config, 0.0)
        restart_data, md = GeoDynamo.read_restart!(
            reader, tmpdir, 1.0, config, pencils; shtns_config = cfg)

        @test md["current_step"] == 5
        @test reader.output_count == 2
        @test haskey(restart_data, "temperature")

        Tback = restart_data["temperature"]
        # each rank must recover exactly its own local (θ × φ × r) slab
        @test size(Tback) == size(localT)
        local_ok = (size(Tback) == size(localT) && Tback == localT) ? 1 : 0
        @test local_ok == 1
        # and every rank must agree (one wrong slab fails the whole grid)
        @test MPI.Allreduce(local_ok, MPI.MIN, comm) == 1

        MPI.Barrier(comm)
        rank == 0 && isdir(tmpdir) && rm(tmpdir; recursive = true, force = true)
    end

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_RT_RESTART && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
