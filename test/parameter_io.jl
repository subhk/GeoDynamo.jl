using Test

@testset "Parameter I/O" begin
    @testset "save_parameters roundtrip" begin
        params = GeoDynamo.SolverParameters(
            nr = 128, lmax = 64, mmax = 64, nlat = 128, nlon = 256,
            Ra = 1e8, Ek = 1e-5, Pr = 0.5, Pm = 2.0,
            timestep = 5e-5, timestepper = GeoDynamo.ERK2(),
            geometry = :shell, radius_ratio = 0.35,
            velocity_bcs = GeoDynamo.BoundaryConditions(
                inner = GeoDynamo.NoSlip(),
                outer = GeoDynamo.StressFree()
            )
        )

        tmpfile = tempname() * ".jl"
        try
            GeoDynamo.save_parameters(params, tmpfile)
            @test isfile(tmpfile)

            loaded = GeoDynamo.load_parameters_from_file(tmpfile)
            @test loaded.nr == params.nr
            @test loaded.lmax == params.lmax
            @test loaded.mmax == params.mmax
            @test loaded.Ra == params.Ra
            @test loaded.Ek == params.Ek
            @test loaded.Pr == params.Pr
            @test loaded.Pm == params.Pm
            @test loaded.timestep == params.timestep
            @test loaded.timestepper isa GeoDynamo.ERK2
            @test loaded.velocity_bcs isa typeof(params.velocity_bcs)
            @test loaded.geometry == params.geometry
            @test loaded.radius_ratio == params.radius_ratio
        finally
            rm(tmpfile, force = true)
        end
    end

    @testset "missing file: load_parameters errors, _from_file falls back" begin
        # An explicitly named missing file is a hard error at the load_parameters
        # level (silently using defaults hid typos / missing configs).
        @test_throws Exception GeoDynamo.load_parameters("nonexistent_file_12345.jl")
        # The lower-level helper still falls back to defaults — it backs the
        # optional default-file path where absence is allowed.
        params = GeoDynamo.load_parameters_from_file("nonexistent_file_12345.jl")
        @test params isa GeoDynamo.SolverParameters
        @test params.nr == 64
        @test params.lmax == 32
    end

    @testset "SolverParameters show method" begin
        params = GeoDynamo.SolverParameters()
        buf = IOBuffer()
        show(buf, MIME("text/plain"), params)
        output = String(take!(buf))
        @test contains(output, "SolverParameters")
        @test contains(output, "Grid")
        @test contains(output, "Physics")
        @test contains(output, "Timestepping")
        @test contains(output, "Boundary Conditions")
        @test contains(output, "Topography")
    end
end
