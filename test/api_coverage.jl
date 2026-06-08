using Test
using GeoDynamo
using MPI

MPI.Initialized() || MPI.Init()

# Coverage for two under-tested public-API surfaces (found via a line-coverage
# audit): the timestepper constructors/accessors/scheme resolution, and the
# NaN/Inf detection (including the abort path).

@testset "API coverage — timesteppers + NaN detection" begin

    @testset "timestepper constructors / accessors / scheme [LOCAL]" begin
        # CNAB2 theta vs implicit_theta reconciliation
        @test_throws ArgumentError GeoDynamo.CNAB2(theta = 0.3, implicit_theta = 0.7)
        @test GeoDynamo.CNAB2(theta = 0.4, implicit_theta = 0.4).implicit_theta == 0.4
        @test GeoDynamo.CNAB2(theta = 0.6).implicit_theta == 0.6
        @test GeoDynamo.CNAB2().implicit_theta == 0.5

        # ThetaMethod struct + ctor + theta accessor
        @test GeoDynamo.ThetaMethod().theta == 0.5
        @test GeoDynamo.ThetaMethod(theta = 0.25).theta == 0.25
        @test GeoDynamo._timestepper_implicit_theta(GeoDynamo.ThetaMethod(theta = 0.25), nothing) == 0.25
        @test GeoDynamo._timestepper_scheme(GeoDynamo.ThetaMethod()) === :theta

        # EAB2 / ETD krylov accessors
        @test GeoDynamo._timestepper_krylov_dimension(GeoDynamo.EAB2(krylov_dimension = 15), nothing) == 15
        @test GeoDynamo._timestepper_krylov_tolerance(GeoDynamo.EAB2(tolerance = 1e-6), nothing) == 1e-6
        @test GeoDynamo._timestepper_krylov_dimension(GeoDynamo.ETD(krylov_dimension = 12), nothing) == 12
        @test GeoDynamo._timestepper_krylov_tolerance(GeoDynamo.ETD(tolerance = 1e-7), nothing) == 1e-7
        @test GeoDynamo._timestepper_scheme(GeoDynamo.EAB2()) === :eab2
        @test GeoDynamo._timestepper_scheme(GeoDynamo.ETD()) === :etd
        @test GeoDynamo._timestepper_scheme(GeoDynamo.ERK2()) === :erk2

        # generic fallback accessors (non-krylov / non-theta schemes)
        @test GeoDynamo._timestepper_krylov_dimension(GeoDynamo.CNAB2(), nothing) == 20
        @test GeoDynamo._timestepper_krylov_tolerance(GeoDynamo.CNAB2(), nothing) == 1e-8

        # _timestepper_from_scheme for every scheme + unknown
        eab = GeoDynamo._timestepper_from_scheme(:eab2, nothing, 15, 1e-7)
        @test eab isa GeoDynamo.EAB2 && eab.krylov_dimension == 15 && eab.tolerance == 1e-7
        etd = GeoDynamo._timestepper_from_scheme(:etd, nothing, 12, 1e-6)
        @test etd isa GeoDynamo.ETD && etd.krylov_dimension == 12 && etd.tolerance == 1e-6
        @test GeoDynamo._timestepper_from_scheme(:theta, 0.3, nothing, nothing).theta == 0.3
        @test GeoDynamo._timestepper_from_scheme(:cnab2, 0.7, nothing, nothing).implicit_theta == 0.7
        @test GeoDynamo._timestepper_from_scheme(:erk2, nothing, nothing, nothing) isa GeoDynamo.ERK2
        @test_throws ArgumentError GeoDynamo._timestepper_from_scheme(:bogus, nothing, nothing, nothing)
    end

    @testset "NaN/Inf detection surface [LOCAL]" begin
        on_quiet = GeoDynamo.NaNDetectionConfig(true, 1, false, false)   # enabled, no-abort, quiet
        on_abort = GeoDynamo.NaNDetectionConfig(true, 1, true, false)    # enabled, abort, quiet
        on_loud = GeoDynamo.NaNDetectionConfig(true, 1, false, true)     # enabled, verbose (hits the @warn)
        off = GeoDynamo.NaNDetectionConfig(false, 1, true, false)        # disabled → no-op

        # check_field_for_nan on raw arrays
        @test GeoDynamo.check_field_for_nan([1.0, NaN, 3.0], "x", on_quiet, 1) == (true, false, 1, 0)
        @test GeoDynamo.check_field_for_nan([1.0, Inf, -Inf], "x", on_quiet, 1) == (false, true, 0, 2)
        @test GeoDynamo.check_field_for_nan([1.0, 2.0], "x", on_quiet, 1) == (false, false, 0, 0)
        @test GeoDynamo.check_field_for_nan([NaN], "x", off, 1) == (false, false, 0, 0)        # disabled
        every2 = GeoDynamo.NaNDetectionConfig(true, 2, false, false)
        @test GeoDynamo.check_field_for_nan([NaN], "x", every2, 1) == (false, false, 0, 0)      # 1 % 2 != 0 → skip
        @test GeoDynamo.check_field_for_nan([NaN], "x", on_loud, 1) == (true, false, 1, 0)       # exercises the @warn branch

        # comprehensive state check (magnetic + composition present)
        params = GeoDynamo.SolverParameters(
            geometry = :shell, lmax = 3, mmax = 3, nlat = 10, nlon = 20, nr = 8, nr_inner = 4,
            radial_bandwidth = 3, radius_ratio = 0.35,
            include_magnetic = true, include_composition = true)
        st = GeoDynamo.initialize_solver_state(Float64; params = params)
        @test GeoDynamo.check_simulation_state_for_nan(st, 1; config = on_quiet) == false   # clean (covers all field branches)
        @test GeoDynamo.check_simulation_state_for_nan(st, 1; config = off) == false         # disabled early-return

        # inject a NaN → detected (no-abort) then aborts (abort)
        parent(st.fields.magnetic.toroidal.data_real)[1] = NaN
        @test GeoDynamo.check_simulation_state_for_nan(st, 1; config = on_quiet) == true
        @test_throws ErrorException GeoDynamo.check_simulation_state_for_nan(st, 1; config = on_abort)
    end
end
