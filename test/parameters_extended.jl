using Test

@testset "Parameters Extended" begin
    @testset "get_parameters returns SolverParameters" begin
        original = GeoDynamo.get_parameters()
        try
            GeoDynamo.set_parameters!(GeoDynamo.SolverParameters(); validate = false)
            params = GeoDynamo.get_parameters()
            @test params isa GeoDynamo.SolverParameters
            @test params.nr >= 8
        finally
            GeoDynamo.set_parameters!(original; validate = false)
        end
    end

    @testset "set_parameters! and get_parameters roundtrip" begin
        original = GeoDynamo.get_parameters()
        try
            custom = GeoDynamo.SolverParameters(nr = 96, Ra = 5e7, Ek = 1e-3)
            GeoDynamo.set_parameters!(custom; validate = true, strict = false)
            retrieved = GeoDynamo.get_parameters()
            @test retrieved.nr == 96
            @test retrieved.Ra == 5e7
            @test retrieved.Ek == 1e-3
        finally
            GeoDynamo.set_parameters!(original; validate = false)
        end
    end

    @testset "set_parameters! with validate=false skips validation" begin
        original = GeoDynamo.get_parameters()
        try
            bad = GeoDynamo.SolverParameters(Ek = -999.0)
            GeoDynamo.set_parameters!(bad; validate = false)
            @test GeoDynamo.get_parameters().Ek == -999.0
        finally
            GeoDynamo.set_parameters!(original; validate = false)
        end
    end

    @testset "initialize_parameters returns params" begin
        params = GeoDynamo.initialize_parameters()
        @test params isa GeoDynamo.SolverParameters
    end

    @testset "create_parameter_template writes public names" begin
        tmpfile = tempname() * ".jl"
        try
            GeoDynamo.create_parameter_template(tmpfile)
            @test isfile(tmpfile)
            content = read(tmpfile, String)
            @test contains(content, "nr")
            @test contains(content, "lmax")
            @test contains(content, "Ek")
            @test !contains(content, "i_N")
        finally
            rm(tmpfile, force = true)
        end
    end

    @testset "find_package_root finds GeoDynamo root" begin
        root = GeoDynamo.find_package_root()
        @test isdir(root)
        @test isfile(joinpath(root, "Project.toml"))
    end

    @testset "topography parameter defaults" begin
        params = GeoDynamo.SolverParameters()
        @test params.topography_enabled == false
        @test params.topography_epsilon == 0.01
        @test params.topography_degree == -1
        @test params.include_topography_velocity == true
        @test params.include_topography_magnetic == true
        @test params.include_topography_thermal == true
        @test params.include_topography_slope_terms == true
        @test params.include_topography_shift_terms == true
    end

    @testset "Stefan condition parameter defaults" begin
        params = GeoDynamo.SolverParameters()
        @test params.stefan_enabled == false
        @test params.stefan_number == 1.0
        @test params.inner_core_conductivity_ratio == 1.0
        @test params.latent_heat == 1.0
    end

    @testset "restart parameter defaults" begin
        params = GeoDynamo.SolverParameters()
        @test params.restart_file == ""
        @test params.restart_dir == ""
        @test params.restart_time == 0.0
    end

    @testset "output parameter defaults" begin
        params = GeoDynamo.SolverParameters()
        @test params.output_precision == :float64
        @test params.independent_output_files == true
    end

    @testset "BC file parameter defaults" begin
        params = GeoDynamo.SolverParameters()
        @test params.temperature_bc_file == ""
        @test params.composition_bc_file == ""
        @test params.temperature_bc_format == :physical
        @test params.composition_bc_format == :physical
        @test params.poloidal_stress_iterations == 2
    end

    @testset "topography file parameter defaults" begin
        params = GeoDynamo.SolverParameters()
        @test params.icb_topography_file == ""
        @test params.ocb_topography_file == ""
    end
end
