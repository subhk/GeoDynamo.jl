using Test

const EXAMPLES_DIR = normpath(joinpath(@__DIR__, "..", "examples"))

read_example(name) = read(joinpath(EXAMPLES_DIR, name), String)

function example_uses_current_solver_api(source::String)
    return occursin("SolverParameters(", source) &&
           occursin("initialize_simulation(Float64, params)", source)
end

function example_uses_high_level_api(source::String)
    return occursin("Spherical", source) &&
           occursin("GeodynamoModel(", source) &&
           occursin("Simulation(", source) &&
           occursin("run!(simulation)", source)
end

@testset "Examples audit" begin
    stale_patterns = (
        r"\bcreate_optimized_config\b",
        r"\bload_temperature_boundaries\b",
        r"\bload_composition_boundaries\b",
        r"\bvalidate_netcdf_temperature_compatibility\b",
        r"\bvalidate_netcdf_composition_compatibility\b",
        r"\bapply_netcdf_temperature_boundaries!\b",
        r"\bapply_netcdf_composition_boundaries!\b",
        r"\bcpu_optimized_transform!\b",
        r"\bcompute_power_spectrum\b",
        r"\bevaluate_field_at_coordinates\b",
        r"\brotate_spherical_field\b",
        r"\breset_performance_stats!\b",
        r"\bprint_performance_report\b",
        r"include\(\"../src/InitialConditions\.jl\"\)",
        r"initialize_simulation\(Float64;",
        r"\bGeoDynamoParameters\("
    )

    for name in readdir(EXAMPLES_DIR)
        source = read_example(name)
        @testset "$name" begin
            for pattern in stale_patterns
                @test !occursin(pattern, source)
            end
        end
    end

    @test example_uses_high_level_api(read_example("ball_mhd_demo.jl"))
    @test example_uses_high_level_api(read_example("shell_dynamo_demo.jl"))
    @test !occursin("latitude_points =", read_example("ball_mhd_demo.jl"))
    @test !occursin("longitude_points =", read_example("ball_mhd_demo.jl"))
    @test !occursin("latitude_points =", read_example("shell_dynamo_demo.jl"))
    @test !occursin("longitude_points =", read_example("shell_dynamo_demo.jl"))

    create_sample = read_example("create_sample_netcdf_boundaries.jl")
    @test occursin("load_temperature_boundaries_from_files", create_sample)
    @test occursin("load_composition_boundaries_from_files", create_sample)
    @test !occursin("apply_netcdf_temperature_boundaries!", create_sample)
    @test !occursin("apply_netcdf_composition_boundaries!", create_sample)

    netcdf_demo = read_example("netcdf_boundary_demo.jl")
    @test occursin("load_temperature_boundaries_from_files", netcdf_demo)
    @test occursin("load_composition_boundaries_from_files", netcdf_demo)
    @test occursin("validate_boundary_compatibility", netcdf_demo)

    hybrid_demo = read_example("hybrid_boundary_demo.jl")
    @test occursin("create_hybrid_temperature_boundaries", hybrid_demo)
    @test occursin("create_hybrid_composition_boundaries", hybrid_demo)
    @test occursin("create_programmatic_boundary", hybrid_demo)

    shtns_demo = read_example("shtnskit_integration_demo.jl")
    @test occursin("create_shtnskit_config", shtns_demo)
    @test occursin("get_shtnskit_version_info", shtns_demo)
    @test occursin("compute_scalar_energy_spectrum", shtns_demo)
    @test occursin("rotate_field_z!", shtns_demo)

    initial_conditions_demo = read_example("initial_conditions_example.jl")
    @test occursin("generate_random_initial_conditions!", initial_conditions_demo)
    @test occursin("set_analytical_initial_conditions!", initial_conditions_demo)
    @test occursin("create_shtns_temperature_field", initial_conditions_demo)

    topography_demo = read_example("topography_example.jl")
    @test occursin("GeodynamoModel(", topography_demo)
    @test occursin("initialize_gaunt_cache!", topography_demo)
end
