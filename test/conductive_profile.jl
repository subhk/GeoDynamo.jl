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

@testset "bc-code mapping + source resolution" begin
    # DIRICHLET/NEUMANN are exported BoundaryType enum values (bcs/common.jl)
    DI = Int(GeoDynamo.DIRICHLET); NE = Int(GeoDynamo.NEUMANN)
    @test G._scalar_bc_code_from_types(DI, DI) == 1   # DD
    @test G._scalar_bc_code_from_types(DI, NE) == 2   # DN
    @test G._scalar_bc_code_from_types(NE, DI) == 3   # ND
    @test G._scalar_bc_code_from_types(NE, NE) == 4   # NN

    dom = G.create_radial_domain(8)
    r = [dom.r[k, 4] for k in 1:dom.N]
    @test G._resolve_source(nothing, dom, 0.0) == zeros(dom.N)      # default
    @test G._resolve_source(2.0, dom, 0.0) == fill(2.0, dom.N)      # uniform
    @test G._resolve_source(x -> x, dom, 0.0) ≈ r                   # function
    @test G._resolve_source(nothing, dom, 6.0) == fill(6.0, dom.N)  # geometry default
end
