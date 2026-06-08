using Test
using GeoDynamo
using MPI

MPI.Initialized() || MPI.Init()

# Coverage for two under-tested files (line-coverage audit):
# - physics/topography.jl — the stefan-enabled solver-topography path
#   (create_solver_topography_state stefan branch, activate/apply/phase-change).
# - bcs/integration.jl — get_boundary_condition_summary on a field with no BC set.

@testset "Topography + BC-integration coverage" begin
    params = GeoDynamo.SolverParameters(
        geometry = :shell, lmax = 4, mmax = 4, nlat = 12, nlon = 24, nr = 8, nr_inner = 4,
        radial_bandwidth = 3, radius_ratio = 0.35,
        include_magnetic = false, include_composition = false,
        stefan_enabled = true, stefan_number = 2.0)
    st = GeoDynamo.initialize_solver_state(Float64; params = params)

    @testset "solver topography stefan path [LOCAL]" begin
        # stefan_enabled → create_solver_topography_state built a StefanState + data
        @test st.topography.stefan !== nothing
        @test st.topography.stefan.Stefan == 2.0
        @test st.topography.data !== nothing            # needs_topography true (stefan)

        # activate sets the global topography config; returns the topo state
        @test GeoDynamo.activate_solver_topography!(st.topography) === st.topography

        # apply (no topo files / config disabled → early return, no error)
        @test GeoDynamo.apply_solver_topography!(st) === st

        # phase-change update hits the not-yet-active @warn branch
        @test GeoDynamo.update_solver_icb_phase_change!(st) === st
    end

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
