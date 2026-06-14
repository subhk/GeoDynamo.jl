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

@testset "conductive_profile_solve" begin
    DI = Int(GeoDynamo.DIRICHLET); NE = Int(GeoDynamo.NEUMANN)
    # conductive_profile_solve is linear & unit-agnostic: this UNIT test uses raw
    # values (no √(4π)); callers pre-scale boundary values + source by √(4π).
    nr = 24
    dom = G.create_radial_domain(nr)
    r = [dom.r[k, 4] for k in 1:nr]; ri = r[1]; ro = r[end]

    # Shell, Dirichlet inner=1 outer=0, S=0 → a + b/r with T(ri)=1, T(ro)=0.
    c = G.conductive_profile_solve(; domain = dom,
        bc_code = G._scalar_bc_code_from_types(DI, DI),
        inner_value = 1.0, outer_value = 0.0,
        source = zeros(nr), inner_regularity = false)
    b = 1.0 / (1/ri - 1/ro); a = -b / ro
    @test c ≈ (a .+ b ./ r) atol = 1e-6 rtol = 1e-6

    # Shell + uniform S=4, Dirichlet 0/0 → T = a + b/r − S r²/6.
    S = 4.0
    c2 = G.conductive_profile_solve(; domain = dom,
        bc_code = G._scalar_bc_code_from_types(DI, DI),
        inner_value = 0.0, outer_value = 0.0, source = fill(S, nr),
        inner_regularity = false)
    part(rr) = -S*rr^2/6
    bb = (part(ro) - part(ri)) / (1/ri - 1/ro); aa = -part(ro) - bb/ro
    @test c2 ≈ (aa .+ bb ./ r .+ part.(r)) atol = 1e-5 rtol = 1e-5
end

@testset "conductive_profile_solve mixed + ball" begin
    DI = Int(GeoDynamo.DIRICHLET); NE = Int(GeoDynamo.NEUMANN)
    nr = 24
    dom = G.create_radial_domain(nr)
    r = [dom.r[k, 4] for k in 1:nr]; ri = r[1]; ro = r[end]
    bw = G.radial_bandwidth(dom)
    # `BandedMatrix` overloads `*` for vectors directly (there is no
    # `Matrix(::BandedMatrix)` conversion), so apply ∇²₀ via the `*` overload.
    lapB = G.BandedMatrix{Float64}(copy(G.create_radial_laplacian(dom).data), bw, nr)

    # Mixed: Dirichlet inner=1, Neumann outer flux=0; interior residual ∇²c+S ≈ 0.
    c = G.conductive_profile_solve(; domain = dom,
        bc_code = G._scalar_bc_code_from_types(DI, NE),
        inner_value = 1.0, outer_value = 0.0, source = fill(2.0, nr),
        inner_regularity = false)
    res = (lapB * c) .+ 2.0
    @test maximum(abs, res[3:nr-2]) < 1e-6

    # Ball regularity (inner Θ′(r₁)=0), outer Dirichlet 0, S=6.
    # NOTE: create_radial_domain(nr) is a SHELL grid (ri≈0.54, not a true ball),
    # so the textbook ball form ro²−r² does NOT hold (it needs ri→0). The
    # regularity row still yields a well-posed solve: assert (a) the interior
    # equation ∇²c+6≈0 and (b) the EXACT l=0 continuum solution for c′(ri)=0,
    # c(ro)=0 on this shell — c = a + b/r − r², b = −2ri³, a = ro² − b/ro.
    cb = G.conductive_profile_solve(; domain = dom,
        bc_code = G._scalar_bc_code_from_types(NE, DI),
        inner_value = 0.0, outer_value = 0.0, source = fill(6.0, nr),
        inner_regularity = true)
    resb = (lapB * cb) .+ 6.0
    @test maximum(abs, resb[3:nr-2]) < 1e-6
    bcoef = -2 * ri^3; acoef = ro^2 - bcoef / ro
    @test cb ≈ (acoef .+ bcoef ./ r .- r .^ 2) atol = 1e-3 rtol = 1e-3
end

@testset "temperature conductive IC == discrete equilibrium (shell)" begin
    # Ra must be positive (validator); velocity starts and stays at rest (buoyancy
    # does not enter momentum from rest), so the conductive equilibrium is
    # undisturbed regardless of Ra.
    params = G.SolverParameters(architecture = :cpu, geometry = :shell,
        nr = 16, nr_inner = 4, lmax = 4, mmax = 4, nlat = 12, nlon = 24,
        Ra = 1.0, Ek = 1e-2, Pr = 1.0, Pm = 1.0, timestep = 1e-3,
        include_magnetic = false, include_composition = false,
        internal_heating = 3.0)
    st = G.initialize_simulation(Float64, params)
    G.initialize_solver_fields!(st)
    tmp = st.fields.temperature
    dom = st.backend.outer_core_domain
    m00 = G.get_mode_index(tmp.config, 0, 0)
    slot = G.local_spectral_storage_slot(tmp.config, m00)
    # The equilibrium claim holds for the conductive (0,0) mode: it is built to
    # satisfy the SAME discrete Laplacian + boundary rows the implicit step uses,
    # with the source sustained via internal_sources, so one CNAB2 step leaves it
    # invariant to machine precision. (The IC's small random l>=1 seed modes are
    # NOT equilibria and do evolve under diffusion — by design — so compare the
    # l=0 conductive mode only.)
    before = [G.local_spectral_value(parent(tmp.spectral.data_real), slot, rk) for rk in 1:dom.N]
    G.solver_step!(st)
    after = [G.local_spectral_value(parent(tmp.spectral.data_real), slot, rk) for rk in 1:dom.N]
    rel = maximum(abs, after .- before) / max(maximum(abs, before), eps())
    @test rel < 1e-6
end

@testset "temperature default conductive backward-compat" begin
    s4pi = sqrt(4π)
    for (geom, rr) in ((:shell, 0.35), (:ball, 0.0))
        params = G.SolverParameters(architecture = :cpu, geometry = geom,
            nr = 24, nr_inner = 4, lmax = 2, mmax = 2, nlat = 8, nlon = 16,
            radius_ratio = rr, Ra = 1.0, Ek = 1e-2, Pr = 1.0, Pm = 1.0, timestep = 1e-3,
            include_magnetic = false, include_composition = false)
        st = G.initialize_simulation(Float64, params); G.initialize_solver_fields!(st)
        tmp = st.fields.temperature; dom = st.backend.outer_core_domain
        m00 = G.get_mode_index(tmp.config, 0, 0)
        slot = G.local_spectral_storage_slot(tmp.config, m00)
        ref = geom === :ball ? G._ball_conductive_temperature : G._shell_conductive_temperature
        for rk in 1:dom.N
            got = G.local_spectral_value(parent(tmp.spectral.data_real), slot, rk) / s4pi
            @test isapprox(got, ref(params, dom.r[rk, 4]); atol = 1e-3, rtol = 1e-3)
        end
    end
end
