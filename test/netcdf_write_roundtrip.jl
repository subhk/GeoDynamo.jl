using Test
using MPI
using NCDatasets
using LinearAlgebra

const FINALIZE_MPI_NCWRITE = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

# End-to-end NetCDF write contract.
#
# Exercises the *production* parallel-write path (create_parallel_netcdf →
# setup_* → write_*) and reads the file back. This guards three properties that
# were previously unverified because no test wrote a real file:
#
#   1. The parallel open call uses the NCDatasets MPI signature that actually
#      exists (`NCDataset(comm, path, mode)`), so output does not MethodError.
#   2. The radial coordinate written equals the true solver collocation nodes,
#      not a fabricated equispaced range.
#   3. The latitude coordinate is labelled truthfully, and field values /
#      spectral indices round-trip exactly.
@testset "NetCDF write round-trip contract" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping NetCDF write round-trip tests"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    comm = GeoDynamo.output_comm()

    # Probe whether this environment has parallel NetCDF (MPI-IO via HDF5). If
    # not, skip — but a MethodError from the production path below is NOT caught
    # here, so a regression in the open-call signature still fails the suite.
    parallel_ok = try
        probe = tempname() * ".nc"
        ds0 = NCDataset(comm, probe, "c"; info = MPI.Info())
        close(ds0)
        isfile(probe) && rm(probe; force = true)
        true
    catch err
        @warn "Parallel NetCDF unavailable; skipping write round-trip" error=err
        false
    end

    if parallel_ok
        lmax = 4
        mmax = 4
        nlat = max(lmax + 2, 10)
        nlon = max(2lmax + 1, 16)
        nr = 8

        cfg = GeoDynamo.create_shtnskit_config(
            lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr)
        dom = GeoDynamo.create_radial_domain(nr)
        radial_nodes = Float64.(dom.r[1:nr, 4])

        nlm = cfg.nlm
        T_in = randn(cfg.nlat, cfg.nlon, nr)
        btor_real = randn(nlm, nr)
        btor_imag = randn(nlm, nr)

        fields = Dict{String, Any}(
            "temperature" => copy(T_in),
            "magnetic_toroidal" =>
                Dict("real" => copy(btor_real), "imag" => copy(btor_imag))
        )

        out_cfg = GeoDynamo.output_config_from_parameters(output_precision = :float64)
        field_info = GeoDynamo.extract_field_info(fields, cfg, nothing;
            radius_ratio = 0.35, radial_grid = radial_nodes)

        filename = tempname() * ".nc"
        ds = GeoDynamo.create_parallel_netcdf(filename, out_cfg, field_info,
            Dict{String, Any}(), comm; geometry = :shell)
        try
            GeoDynamo.setup_dimensions!(ds, field_info, out_cfg)
            GeoDynamo.setup_variables!(ds, field_info, out_cfg, collect(keys(fields)))
            GeoDynamo.write_coordinate_data!(ds, field_info, out_cfg)
            GeoDynamo.write_field_data!(ds, fields, out_cfg, field_info)
            NCDatasets.sync(ds)
        finally
            close(ds)
        end

        @test isfile(filename)

        NCDataset(filename, "r") do d
            # (2) radial coordinate matches the true Chebyshev nodes, not [0.35,1].
            @test Array(d["r"][:]) ≈ radial_nodes atol = 1e-12
            @test !(Array(d["r"][:]) ≈ collect(range(0.35, 1.0, length = nr)))

            # (3a) latitude coordinate is labelled truthfully and matches config.
            @test d["theta"].attrib["long_name"] == "latitude"
            @test Array(d["theta"][:]) ≈ Float64.(cfg.theta_grid) atol = 1e-12

            # (3b) physical field values and shape round-trip exactly.
            T_out = Array(d["temperature"][:, :, :])
            @test size(T_out) == size(T_in)
            @test maximum(abs.(T_out .- T_in)) == 0.0

            # (3c) spectral coefficients and (l,m) indices round-trip exactly.
            @test maximum(abs.(Array(d["magnetic_toroidal_real"][:, :]) .- btor_real)) ==
                  0.0
            @test maximum(abs.(Array(d["magnetic_toroidal_imag"][:, :]) .- btor_imag)) ==
                  0.0
            @test Array(d["l_values"][:]) == Int32.(cfg.l_values[1:nlm])
            @test Array(d["m_values"][:]) == Int32.(cfg.m_values[1:nlm])
        end

        isfile(filename) && rm(filename; force = true)
    end

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_NCWRITE && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
