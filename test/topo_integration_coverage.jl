using Test
using GeoDynamo
using MPI

MPI.Initialized() || MPI.Init()

# Coverage for the solver-level Stefan availability contract and for
# bcs/integration.jl's summary of a field with no loaded BC set.

@testset "Topography + BC-integration coverage" begin
    stefan_params = GeoDynamo.SolverParameters(
        geometry = :shell, lmax = 4, mmax = 4, nlat = 12, nlon = 24, nr = 8, nr_inner = 4,
        radial_bandwidth = 3, radius_ratio = 0.35,
        include_magnetic = false, include_composition = false,
        stefan_enabled = true, stefan_number = 2.0)

    @testset "solver rejects unavailable Stefan evolution [LOCAL]" begin
        valid, errors, _ = GeoDynamo.validate_parameters(stefan_params; strict = false)
        @test !valid
        @test any(contains(error, "stefan_enabled=true") for error in errors)
        @test_throws ArgumentError GeoDynamo.initialize_solver_state(
            Float64; params = stefan_params)
    end

    params = GeoDynamo.SolverParameters(
        geometry = :shell, lmax = 4, mmax = 4, nlat = 12, nlon = 24, nr = 8, nr_inner = 4,
        radial_bandwidth = 3, radius_ratio = 0.35,
        include_magnetic = false, include_composition = false)
    st = GeoDynamo.initialize_solver_state(Float64; params = params)

    @testset "get_boundary_condition_summary (no BC loaded) [LOCAL]" begin
        summ = GeoDynamo.bcs.get_boundary_condition_summary(
            st.fields.temperature, GeoDynamo.bcs.TEMPERATURE)
        @test summ isa Dict
        @test summ["field_type"] == string(GeoDynamo.bcs.TEMPERATURE)
        @test haskey(summ, "has_boundary_fields")
        # with no programmatic BC set loaded, the summary reports the reason
        @test haskey(summ, "reason") || summ["has_boundary_conditions"] == false
    end
end
