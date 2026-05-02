using Test
using GeoDynamo

const EXAMPLES_DIR = normpath(joinpath(@__DIR__, "..", "examples"))

function load_example_module(name::AbstractString)
    modname = Symbol("Example_", replace(first(splitext(name)), r"[^A-Za-z0-9]" => "_"))
    mod = Module(modname)
    Core.eval(mod, :(include(path::AbstractString) = Base.include($mod, path)))
    Base.include(mod, joinpath(EXAMPLES_DIR, name))
    return mod
end

@testset "Examples smoke" begin
    @testset "NetCDF sample generator" begin
        mod = load_example_module("create_sample_netcdf_boundaries.jl")

        mktempdir() do tmp
            cd(tmp) do
                mod.create_sample_temperature_boundaries(nlat=8, nlon=16, time_dependent=false)
                mod.create_sample_temperature_boundaries(nlat=8, nlon=16, time_dependent=true)
                mod.create_sample_composition_boundaries(nlat=8, nlon=16)

                @test isfile("cmb_temp.nc")
                @test isfile("surface_temp.nc")
                @test isfile("cmb_temp_timedep.nc")
                @test isfile("surface_temp_timedep.nc")
                @test isfile("cmb_composition.nc")
                @test isfile("surface_composition.nc")
            end
        end
    end

    @testset "Boundary examples" begin
        netcdf_demo = load_example_module("netcdf_boundary_demo.jl")
        hybrid_demo = load_example_module("hybrid_boundary_demo.jl")

        netcdf_demo.ensure_sample_files!()
        cfg, temp_boundaries, comp_boundaries = netcdf_demo.demo_file_boundaries()
        @test cfg.nlm > 0
        @test temp_boundaries.boundary_set.inner_boundary.nlat > 0
        @test comp_boundaries.boundary_set.outer_boundary.nlon > 0
        netcdf_demo.demo_interpolation(temp_boundaries)
        netcdf_demo.demo_time_dependent_file()

        demo_cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=16, nlon=32, nr=8)
        hybrid_demo.ensure_sample_files!()
        hybrid_demo.demo_hybrid_sets(demo_cfg)
        hybrid_demo.demo_spherical_harmonic_patterns(demo_cfg)
        hybrid_demo.demo_time_dependent_pattern(demo_cfg)
        @test true
    end

    @testset "Transform and initial-condition examples" begin
        transforms_demo = load_example_module("shtnskit_integration_demo.jl")
        ic_demo = load_example_module("initial_conditions_example.jl")

        transforms_demo.main(lmax=8, mmax=8, nlat=16, nlon=32, nr=8)
        ic_demo.main(lmax=8, mmax=8, nlat=16, nlon=32, nr=8)
        @test true
    end

    @testset "Simulation setup examples" begin
        ball_demo = load_example_module("ball_mhd_demo.jl")
        shell_demo = load_example_module("shell_dynamo_demo.jl")
        topography_demo = load_example_module("topography_example.jl")

        ball_state = ball_demo.main(
            run=false,
            radial_points_outer=16,
            spherical_degree=4,
            spherical_order=4,
            latitude_points=12,
            longitude_points=16,
            max_steps=1,
        )
        @test ball_state.parameters.geometry === :ball

        shell_state = shell_demo.main(
            run=false,
            radial_points_outer=16,
            radial_points_inner=4,
            spherical_degree=4,
            spherical_order=4,
            latitude_points=12,
            longitude_points=16,
            max_steps=1,
        )
        @test shell_state.parameters.geometry === :shell

        params = topography_demo.setup_topography_via_params()
        topo = topography_demo.create_topography_examples()
        stefan = topography_demo.setup_stefan_evolution(0.35, 8)
        @test params.topography_enabled
        @test topo.cmb !== nothing
        @test topo.cmb.lmax > 0
        @test stefan !== nothing
    end
end
