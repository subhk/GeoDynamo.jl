using Test
using GeoDynamo
using MPI
using Random

MPI.Initialized() || MPI.Init()

# Coverage for two under-tested files (line-coverage audit targeted the actual
# uncovered functions, not the already-covered simple ones):
# - bcs/topography/stefan_condition.jl — the field-driven boundary flux / normal
#   velocity / initialization path (compute_boundary_heat_flux_spectral,
#   compute_normal_velocity_spectral, initialize_stefan_state!).
# - timestep/imex.jl — the EAB2 Krylov update (solver_eab2_update_krylov_cached!
#   + SolverBandedAction), exercised by a solver step with the EAB2 timestepper.

const Topo = GeoDynamo.bcs.topography

function _small_state(; timestepper = GeoDynamo.CNAB2(), include_magnetic = true)
    params = GeoDynamo.SolverParameters(
        geometry = :shell, lmax = 4, mmax = 4, nlat = 12, nlon = 24, nr = 8, nr_inner = 4,
        radial_bandwidth = 3, radius_ratio = 0.35,
        include_magnetic = include_magnetic, include_composition = false,
        timestepper = timestepper)
    st = GeoDynamo.initialize_solver_state(Float64; params = params)
    rng = MersenneTwister(31)
    seed = (st.fields.temperature.spectral, st.fields.velocity.toroidal, st.fields.velocity.poloidal)
    seed = include_magnetic ? (seed..., st.fields.magnetic.toroidal, st.fields.magnetic.poloidal) : seed
    for f in seed
        dr = parent(f.data_real); di = parent(f.data_imag)
        dr .+= 1e-3 .* (rand(rng, size(dr)...) .- 0.5)
        di .+= 1e-3 .* (rand(rng, size(di)...) .- 0.5)
    end
    return st
end

@testset "Stefan + IMEX coverage" begin

    @testset "Stefan boundary flux / normal velocity / init from real fields [LOCAL]" begin
        st = _small_state(include_magnetic = false)
        ri = 0.35

        # compute_boundary_heat_flux_spectral — spectral radial-derivative at a boundary
        flux_in = Topo.compute_boundary_heat_flux_spectral(st.fields.temperature, ri, :inner)
        flux_out = Topo.compute_boundary_heat_flux_spectral(st.fields.temperature, ri, :outer)
        nlm = st.fields.temperature.spectral.nlm
        @test length(flux_in) == nlm && length(flux_out) == nlm
        @test all(isfinite, abs.(flux_in)) && all(isfinite, abs.(flux_out))

        # compute_normal_velocity_spectral — uₙ at the ICB from the velocity field
        un = Topo.compute_normal_velocity_spectral(st.fields.velocity, ri, Topo.INNER_BOUNDARY)
        @test length(un) == nlm && all(isfinite, abs.(un))

        # initialize_stefan_state! drives both + flips is_initialized
        stefan = Topo.StefanState(lmax = 4, ri = ri)
        Topo.initialize_stefan_state!(stefan, st.fields.temperature, st.fields.temperature, st.fields.velocity)
        @test stefan.is_initialized == true
        @test length(stefan.heat_flux_ic) == stefan.topography.nlm
        @test all(isfinite, abs.(stefan.heat_flux_oc))
    end

    @testset "IMEX EAB2 Krylov step runs + stays finite [LOCAL]" begin
        st = _small_state(timestepper = GeoDynamo.EAB2(krylov_dimension = 8, tolerance = 1e-6),
            include_magnetic = true)
        step0 = st.step
        GeoDynamo.solver_step!(st)                       # drives solver_eab2_update_krylov_cached!
        @test st.step == step0 + 1
        @test all(isfinite, parent(st.fields.velocity.toroidal.data_real))
        @test all(isfinite, parent(st.fields.velocity.poloidal.data_real))
        @test all(isfinite, parent(st.fields.magnetic.toroidal.data_real))
        @test all(isfinite, parent(st.fields.temperature.spectral.data_real))
    end
end
