# ================================================================================
# Regression tests for batch D — the confirmed findings from the 8 candidates whose
# verifier agents died on the session limit.
# ================================================================================
#
#   D1 timestep/driver.jl:180   threaded-update safety rested on a hand-enumerated
#                              denylist of collective-issuing magnetic configs; any
#                              collective added elsewhere → silent multi-rank hang
#   D2 api/simulation.jl:492    the GPU sync predicate excluded stop-condition
#                              callbacks BY FUNCTION IDENTITY — the mechanism the
#                              surrounding comment says was abandoned — so a wrapped
#                              or user-supplied clock-only stop condition collapsed
#                              gpu_sync=:output back into :every
# ================================================================================

using Test
using MPI
using GeoDynamo

@testset "Max-effort review batch D" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping batch D fix tests"
        return
    end
    MPI.Initialized() || MPI.Init()

    # ── D1: a collective inside the threaded region must be caught, not hang ──
    @testset "D1 collectives inside a threaded update are rejected loudly" begin
        # The guard is only armed by the threaded path when it would actually be unsafe
        # (multi-rank); arming it by hand here lets it be tested at one rank. It is
        # task-scoped, so `_with_threaded_update_guard` is also what releases it — see
        # batch F for why a process-global Ref was the wrong home.
        @test GeoDynamo._in_threaded_implicit_update() == false
        @test GeoDynamo._assert_no_collective_in_threaded_update("probe") === nothing

        GeoDynamo._with_threaded_update_guard() do
            err = try
                GeoDynamo._assert_no_collective_in_threaded_update("probe")
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("probe", sprint(showerror, err))
            # the repo's own reduction helpers must route through the guard
            @test_throws ErrorException GeoDynamo.allreduce_sum(1.0)
            @test_throws ErrorException GeoDynamo.allreduce_sum_in_place!([1.0, 2.0])
        end
        # cleared again: the helpers work normally
        @test GeoDynamo.allreduce_sum(2.0) == 2.0
    end

    # ── D1b: the known-collective denylist must cover both components ─────────
    @testset "D1b magnetic collective predicate checks poloidal and topography" begin
        params = GeoDynamo.SolverParameters(
            geometry = :shell, lmax = 4, mmax = 4, nlat = 12, nlon = 24, nr = 8,
            nr_inner = 4, radial_bandwidth = 3, radius_ratio = 0.35,
            include_magnetic = true, include_composition = false)
        st = GeoDynamo.initialize_solver_state(Float64; params = params)
        mag = st.fields.magnetic
        @test mag !== nothing

        # The CONFIG predicate is rank-independent, which is what makes it testable
        # here; `_solver_multirank_magnetic_collective` gates it on rank count and so
        # is always false at one rank (asserted at the end).
        # baseline: insulating, no topography ⇒ no in-kernel collective
        @test GeoDynamo._solver_magnetic_config_has_collective(st) == false

        # a CONTINUITY_MAG entry on the POLOIDAL inner boundary must count too —
        # the predicate previously inspected only `toroidal.bc_type_inner`.
        mag.poloidal.bc_type_inner[1] = Int(GeoDynamo.CONTINUITY_MAG)
        @test GeoDynamo._solver_magnetic_config_has_collective(st) == true
        mag.poloidal.bc_type_inner[1] = Int(GeoDynamo.NEUMANN_MAG_INNER)
        @test GeoDynamo._solver_magnetic_config_has_collective(st) == false

        # toroidal still counts (unchanged behaviour)
        mag.toroidal.bc_type_inner[1] = Int(GeoDynamo.CONTINUITY_MAG)
        @test GeoDynamo._solver_magnetic_config_has_collective(st) == true
        mag.toroidal.bc_type_inner[1] = Int(GeoDynamo.DIRICHLET)

        # the rank gate keeps single-rank runs threading every scheme
        @test GeoDynamo._solver_multirank_magnetic_collective(st) == false

        # topography magnetic coupling issues its own collectives
        topo_params = GeoDynamo.SolverParameters(
            geometry = :shell, lmax = 4, mmax = 4, nlat = 12, nlon = 24, nr = 8,
            nr_inner = 4, radial_bandwidth = 3, radius_ratio = 0.35,
            include_magnetic = true, include_composition = false,
            topography_enabled = true, include_topography_magnetic = true)
        tst = GeoDynamo.initialize_solver_state(Float64; params = topo_params)
        @test GeoDynamo._solver_magnetic_config_has_collective(tst) == true
    end

    # ── D4: the two restart readers must share one implementation ────────────
    @testset "D4 read_restart! delegates to _load_restart_file" begin
        # Both entry points must produce byte-identical results for the same file, and
        # `read_restart!` must no longer carry its own copy of the read (the duplicate
        # bodies differed only in `&&`-chained vs nested-`if` form, so a field added to
        # one reader was silently skipped by the other).
        src = read(joinpath(normpath(joinpath(@__DIR__, "..")), "src", "io", "restart.jl"),
            String)
        i = findfirst("function read_restart!", src)
        j = findfirst("function _load_restart_file", src)
        body = src[last(i):first(j)]
        @test occursin("_load_restart_file(filename, tracker, config", body)
        # the field-reading logic must live in exactly ONE place now
        @test !occursin("restart_data[\"temperature\"]", body)
        @test !occursin("range_local(pencils.r", body)
    end

    # ── D2: the sync predicate must not key off function identity ─────────────
    @testset "D2 GPU sync predicate excludes clock-only callbacks by trait" begin
        grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU();
            lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
        model = GeoDynamo.GeodynamoModel(grid; Ek = 1e-2, Ra = 1e4,
            include_magnetic = false, include_composition = false)
        sim = GeoDynamo.Simulation(model; Δt = 1e-4, gpu = true, gpu_sync = :output)

        # The three built-in stop conditions are clock-only, so a default Simulation
        # (whose only callbacks are those three plus nan_checker) must not report a
        # pending host read on an iteration where nan_checker does not fire.
        @test GeoDynamo._gpu_clock_only(GeoDynamo.stop_time_exceeded)
        @test GeoDynamo._gpu_clock_only(GeoDynamo.stop_iteration_exceeded)
        @test GeoDynamo._gpu_clock_only(GeoDynamo.wall_time_limit_exceeded)
        @test GeoDynamo._gpu_clock_only(GeoDynamo.nan_checker) == false

        # A user stop condition wrapped in ClockOnlyCallback must be excluded too —
        # this is the extensibility the identity test could not provide.
        my_stop(sim) = (sim.model.clock.iteration >= 10 && (sim.running = false); nothing)
        @test GeoDynamo._gpu_clock_only(GeoDynamo.ClockOnlyCallback(my_stop))
        # and it still runs like a normal callback
        wrapped = GeoDynamo.ClockOnlyCallback(my_stop)
        @test wrapped(sim) === nothing

        # Registering it must NOT force a sync every iteration.
        GeoDynamo.add_callback!(sim, wrapped;
            schedule = GeoDynamo.IterationInterval(1), name = :my_stop)
        sim.model.clock.iteration = 3        # nan_checker (100) does not fire here
        @test GeoDynamo._gpu_host_read_pending(sim) == false

        # A closure around a built-in stop condition is likewise excluded when marked.
        GeoDynamo.add_callback!(sim,
            GeoDynamo.ClockOnlyCallback(s -> GeoDynamo.stop_time_exceeded(s));
            schedule = GeoDynamo.IterationInterval(1), name = :wrapped_stop)
        @test GeoDynamo._gpu_host_read_pending(sim) == false

        # An UNMARKED field-reading callback on every iteration still forces a sync.
        GeoDynamo.add_callback!(sim, GeoDynamo.nan_checker;
            schedule = GeoDynamo.IterationInterval(1), name = :nan_every)
        @test GeoDynamo._gpu_host_read_pending(sim) == true
    end
end
