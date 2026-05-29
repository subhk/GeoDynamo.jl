using Test
using MPI
using NCDatasets

const FINALIZE_MPI_DIST_WRITE = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

# Multi-rank distributed NetCDF write→read assembly contract.
#
# `netcdf_write_roundtrip.jl` proves the writer is bit-exact on ONE rank, but a
# single rank owns the whole grid so it never exercises the distributed offset
# logic: each rank writing only its pencil slab at the correct GLOBAL position.
# That placement (`range_local` offsets for physical θ/φ, the local→global mode
# map for spectral coefficients) only matters at ≥2 ranks, and a wrong offset,
# an overlap, or a gap silently corrupts the file rather than erroring.
#
# Strategy: every rank fills ONLY its local slab, and the value in each cell
# encodes that cell's GLOBAL index (disjoint integer ranges → bit-exact compare).
# After the collective write, rank 0 reads the whole assembled file and checks
# every global cell equals its encoding. Any misplacement makes the readback
# disagree. The check covers both distributed axes: θ/φ (physical, r-pencil) and
# mode/r (spectral, spec-pencil).
#
# Runs at any rank count: at 1 rank it degrades to a serial roundtrip (still a
# valid regression); the real distributed assertion needs `mpiexec -n≥2`, driven
# by `test/run_mpi_write_roundtrip_smoke.sh`.
@testset "Distributed NetCDF write→read assembly" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping distributed write round-trip test"
        return
    end
    MPI.Initialized() || MPI.Init()

    comm = GeoDynamo.output_comm()
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # integer-valued, exactly-representable, disjoint ranges → bit-exact compare
    encode_phys(gθ, gφ, gr) = gθ * 1.0 + gφ * 1_000.0 + gr * 1_000_000.0
    encode_spec(glm, gr) = glm * 1.0 + gr * 100_000.0

    lmax = 4;
    mmax = 4
    nlat = max(lmax + 2, 10);
    nlon = max(2lmax + 1, 16);
    nr = 8

    cfg = GeoDynamo.create_shtnskit_config(
        lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr)
    dom = GeoDynamo.create_radial_domain(nr)
    radial_nodes = Float64.(dom.r[1:nr, 4])
    nlm = cfg.nlm

    # Probe collective MPI-IO. A missing parallel-HDF5 backend cannot satisfy a
    # ≥2-rank collective write, so skip rather than fail — but the production
    # open signature is still exercised, so a regression there still errors.
    probe = rank == 0 ? tempname() * ".nc" : ""
    probe = MPI.bcast(probe, 0, comm)
    parallel_ok = try
        ds0 = NCDataset(comm, probe, "c"; info = MPI.Info())
        close(ds0)
        MPI.Barrier(comm)
        rank == 0 && isfile(probe) && rm(probe; force = true)
        true
    catch err
        rank == 0 &&
            @warn "Parallel NetCDF unavailable; skipping distributed write test" error=err
        false
    end

    if parallel_ok
        # ---- physical temperature: r-pencil (θ,φ distributed, r local) ----
        Tarr = GeoDynamo.create_pencil_array(Float64, cfg.pencils.r; init = :zero)
        Tp = parent(Tarr)
        θr = GeoDynamo.range_local(cfg.pencils.r, 1)
        φr = GeoDynamo.range_local(cfg.pencils.r, 2)
        rr = GeoDynamo.range_local(cfg.pencils.r, 3)
        for (lk, gr) in enumerate(rr), (lj, gφ) in enumerate(φr), (li, gθ) in enumerate(θr)
            Tp[li, lj, lk] = encode_phys(gθ, gφ, gr)
        end

        # ---- spectral magnetic_toroidal: spec-pencil (modes and r distributed) ----
        specR = GeoDynamo.create_pencil_array(Float64, cfg.pencils.spec; init = :zero)
        specI = GeoDynamo.create_pencil_array(Float64, cfg.pencils.spec; init = :zero)
        Rp = parent(specR);
        Ip = parent(specI)
        lm_map = GeoDynamo.local_spectral_lm_map(cfg)
        spec_r = GeoDynamo.range_local(cfg.pencils.spec, 3)
        for slot in CartesianIndices(lm_map)
            glm = lm_map[slot]
            glm == 0 && continue
            for (lk, gr) in enumerate(spec_r)
                GeoDynamo.set_local_spectral_value!(Rp, slot, lk, encode_spec(glm, gr))
                GeoDynamo.set_local_spectral_value!(Ip, slot, lk, -encode_spec(glm, gr))
            end
        end

        fields = Dict{String, Any}(
            "temperature" => copy(Tp),
            "magnetic_toroidal" => Dict("real" => copy(Rp), "imag" => copy(Ip))
        )

        out_cfg = GeoDynamo.output_config_from_parameters(output_precision = :float64)
        field_info = GeoDynamo.extract_field_info(fields, cfg, cfg.pencils;
            radius_ratio = 0.35, radial_grid = radial_nodes)

        filename = rank == 0 ? tempname() * ".nc" : ""
        filename = MPI.bcast(filename, 0, comm)

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
        MPI.Barrier(comm)

        # Rank 0 owns the assembled global view; it alone reads back and asserts.
        if rank == 0
            @test isfile(filename)
            NCDataset(filename, "r") do d
                Tg = Array(d["temperature"][:, :, :])
                @test size(Tg) == (nlat, nlon, nr)
                phys_bad = 0
                for gr in 1:nr, gφ in 1:nlon, gθ in 1:nlat
                    Tg[gθ, gφ, gr] == encode_phys(gθ, gφ, gr) || (phys_bad += 1)
                end
                @test phys_bad == 0   # every physical cell at its correct global slab

                Rg = Array(d["magnetic_toroidal_real"][:, :])
                Ig = Array(d["magnetic_toroidal_imag"][:, :])
                @test size(Rg) == (nlm, nr)
                spec_bad = 0
                for gr in 1:nr, glm in 1:nlm

                    (Rg[glm, gr] == encode_spec(glm, gr) &&
                     Ig[glm, gr] == -encode_spec(glm, gr)) || (spec_bad += 1)
                end
                @test spec_bad == 0   # every spectral coeff at its correct global (mode,r)

                @test Array(d["l_values"][:]) == Int32.(cfg.l_values[1:nlm])
                @test Array(d["m_values"][:]) == Int32.(cfg.m_values[1:nlm])
                @test Array(d["r"][:]) ≈ radial_nodes atol = 1e-12
            end
            isfile(filename) && rm(filename; force = true)
            @info "Distributed write→read assembly verified" nranks physical_cells=nlat*nlon*nr spectral_coeffs=nlm*nr
        end
        MPI.Barrier(comm)
    end
end

if MPI.Initialized()
    MPI.Barrier(GeoDynamo.get_comm())
    if FINALIZE_MPI_DIST_WRITE && !MPI.Finalized()
        MPI.Finalize()
    end
end
