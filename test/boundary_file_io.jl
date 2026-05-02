using Test
using GeoDynamo
using NCDatasets

const BCS = GeoDynamo.bcs

mutable struct DummyBoundaryField
    boundary_interpolation_cache::Dict{String, Any}
end

function write_spectral_bc_file(path::AbstractString, bc_re, bc_im)
    NCDataset(path, "c") do ds
        defDim(ds, "boundary", size(bc_re, 1))
        defDim(ds, "mode", size(bc_re, 2))
        defVar(ds, "Cbc_Re", eltype(bc_re), ("boundary", "mode"))
        defVar(ds, "Cbc_Im", eltype(bc_im), ("boundary", "mode"))
        ds["Cbc_Re"][:, :] = bc_re
        ds["Cbc_Im"][:, :] = bc_im
    end
    return path
end

@testset "Boundary File IO" begin
    @testset "NetCDF boundary roundtrip" begin
        mktempdir() do tmpdir
            theta = collect(range(0.0, π; length=4))
            phi = collect(range(0.0, 2π * (1 - 1 / 5); length=5))
            values = reshape(collect(Float64, 1:20), 4, 5)
            boundary = BCS.create_boundary_data(
                values,
                "temperature";
                theta=theta,
                phi=phi,
                units="K",
                description="synthetic temperature boundary",
                file_path="programmatic",
            )

            filename = joinpath(tmpdir, "temperature_boundary.nc")
            BCS.write_netcdf_boundary_data(filename, boundary)

            @test BCS.validate_netcdf_boundary_file(filename)

            loaded = BCS.read_netcdf_boundary_data(filename)
            @test loaded.field_type == "temperature"
            @test loaded.units == "K"
            @test loaded.description == "synthetic temperature boundary"
            @test loaded.is_time_dependent == false
            @test loaded.nlat == 4
            @test loaded.nlon == 5
            @test loaded.ntime == 1
            @test loaded.ncomponents == 1
            @test loaded.theta ≈ theta
            @test loaded.phi ≈ phi
            @test loaded.values == values

            info = BCS.get_netcdf_file_info(filename)
            @test info["filename"] == filename
            @test "temperature" in info["variables"]
            @test info["nlat"] == 4
            @test info["nlon"] == 5
        end
    end

    @testset "NetCDF boundary validator catches malformed files" begin
        mktempdir() do tmpdir
            invalid_filename = joinpath(tmpdir, "invalid_boundary.nc")
            NCDataset(invalid_filename, "c") do ds
                defDim(ds, "theta", 2)
                defDim(ds, "phi", 3)
                defVar(ds, "theta", Float64, ("theta",))
                defVar(ds, "phi", Float64, ("phi",))
                defVar(ds, "temperature", Float64, ("theta", "phi"))
                ds["theta"][:] = [0.0, π + 0.1]
                ds["phi"][:] = [0.0, 1.0, 2.0]
                ds["temperature"][:, :] = zeros(2, 3)
            end

            @test_throws ArgumentError BCS.validate_netcdf_boundary_file(invalid_filename)

            missing_field_filename = joinpath(tmpdir, "missing_field.nc")
            NCDataset(missing_field_filename, "c") do ds
                defDim(ds, "theta", 2)
                defDim(ds, "phi", 2)
                defVar(ds, "theta", Float64, ("theta",))
                defVar(ds, "phi", Float64, ("phi",))
                ds["theta"][:] = [0.0, π]
                ds["phi"][:] = [0.0, π]
            end

            @test_throws ArgumentError BCS.read_netcdf_boundary_data(missing_field_filename)
        end
    end

    @testset "Spectral BC loading supports both stored orientations" begin
        cfg = GeoDynamo.create_shtnskit_config(lmax=2, mmax=2, nlat=8, nlon=16, nr=4)
        nlm = cfg.nlm

        mktempdir() do tmpdir
            bc_re = reshape(collect(Float64, 1:(2 * nlm)), 2, nlm)
            bc_im = -bc_re

            filename = write_spectral_bc_file(joinpath(tmpdir, "spectral_bc.nc"), bc_re, bc_im)
            coeffs = BCS.load_spectral_bc_from_file(filename, cfg; format=:spectral)
            @test coeffs.nlm == nlm
            @test coeffs.source_format == :spectral
            @test coeffs.bc_real == bc_re
            @test coeffs.bc_imag == bc_im

            transposed_filename = joinpath(tmpdir, "spectral_bc_transposed.nc")
            NCDataset(transposed_filename, "c") do ds
                defDim(ds, "mode", nlm)
                defDim(ds, "boundary", 2)
                defVar(ds, "Cbc_Re", Float64, ("mode", "boundary"))
                defVar(ds, "Cbc_Im", Float64, ("mode", "boundary"))
                ds["Cbc_Re"][:, :] = permutedims(bc_re)
                ds["Cbc_Im"][:, :] = permutedims(bc_im)
            end

            transposed = BCS.load_spectral_bc_from_file(transposed_filename, cfg; format=:spectral)
            @test transposed.bc_real == bc_re
            @test transposed.bc_imag == bc_im
        end
    end

    @testset "Physical BC loading and field cache storage" begin
        cfg = GeoDynamo.create_shtnskit_config(lmax=2, mmax=2, nlat=8, nlon=16, nr=4)

        mktempdir() do tmpdir
            filename = joinpath(tmpdir, "physical_bc.nc")
            NCDataset(filename, "c") do ds
                defDim(ds, "theta", cfg.nlat)
                defDim(ds, "phi", cfg.nlon)
                defVar(ds, "inner", Float64, ("theta", "phi"))
                defVar(ds, "outer", Float64, ("theta", "phi"))
                ds["inner"][:, :] = zeros(cfg.nlat, cfg.nlon)
                ds["outer"][:, :] = zeros(cfg.nlat, cfg.nlon)
            end

            coeffs = BCS.load_spectral_bc_from_file(filename, cfg; format=:physical)
            @test coeffs.source_format == :physical
            @test all(iszero, coeffs.bc_real)
            @test all(iszero, coeffs.bc_imag)

            field = DummyBoundaryField(Dict{String, Any}())
            BCS.store_bc_in_field!(field, coeffs)
            vectors = BCS.get_bc_vectors_from_field(field)
            @test vectors.inner_real !== nothing
            @test vectors.outer_real !== nothing
            @test length(vectors.inner_real) == cfg.nlm
            @test all(iszero, vectors.inner_real)
            @test all(iszero, vectors.outer_imag)
        end
    end
end
