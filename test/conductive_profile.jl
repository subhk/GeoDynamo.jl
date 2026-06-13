using Test
using GeoDynamo
const G = GeoDynamo

@testset "conductive IC: source params" begin
    p0 = G.SolverParameters(nr = 16, lmax = 4)
    @test p0.internal_heating === nothing
    @test p0.compositional_source === nothing
    p1 = G.SolverParameters(nr = 16, lmax = 4, internal_heating = 3.0)
    @test p1.internal_heating == 3.0
    p3 = G.SolverParameters(nr = 16, lmax = 4, compositional_source = 2.0)
    @test p3.compositional_source == 2.0
    p2 = G.SolverParameters(nr = 16, lmax = 4, internal_heating = (r -> 2r))
    @test p2.internal_heating isa Function

    # End-to-end: internal_heating propagates through GeodynamoModel public API
    grid = G.SphericalShellGrid(G.CPU(); lmax = 4, mmax = 4, nlat = 12, nlon = 16,
        nr = 16, nr_inner = 4)
    model = G.GeodynamoModel(grid; Ek = 1e-2, Ra = 1e4, include_magnetic = false,
        include_composition = false, internal_heating = 5.0)
    @test model.state.parameters.internal_heating == 5.0
end
