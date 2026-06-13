using Test
using GeoDynamo
const G = GeoDynamo

@testset "conductive IC: source params" begin
    p0 = G.SolverParameters(nr = 16, lmax = 4)
    @test p0.internal_heating === nothing
    @test p0.compositional_source === nothing
    p1 = G.SolverParameters(nr = 16, lmax = 4, internal_heating = 3.0)
    @test p1.internal_heating == 3.0
    p2 = G.SolverParameters(nr = 16, lmax = 4, internal_heating = (r -> 2r))
    @test p2.internal_heating isa Function
end
