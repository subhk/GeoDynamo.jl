# ================================================================================
# Regression tests for the round-3 codex findings on the public API layer
# ================================================================================
#
#   1. TimeInterval must fire on every crossed interval, not only when the time
#      lands exactly on a multiple (dt that doesn't divide the interval).
#   2. Simulation must reject invalid run controls (courant<=0, stop_iteration<1,
#      stop_time<=start_time), not silently accept them.
#   3. validate_parameters must report transform-invalid grids (nlat<lmax+1,
#      nlon<2*mmax+1) as INVALID, not merely warn.
#   4. Grid size defaults must be valid for small lmax (e.g. lmax=1).
#   5. FieldWriter must reject unknown field selectors at construction.
# ================================================================================

using Test
using MPI
using GeoDynamo

@testset "Codex round-3 validation fixes" begin
    S = GeoDynamo
    ctx(t) = S._ScheduleContext(Float64(t), 0, 0.0)

    # ── Finding 1: TimeInterval fires on crossed intervals ────────────────────
    @testset "TimeInterval fires on crossed intervals (dt=0.03, interval=0.1)" begin
        sched = S.TimeInterval(0.1)
        # Walk time in dt=0.03 steps; the 0.1 and 0.2 boundaries are crossed at
        # 0.12 and 0.21 (time never lands exactly on a multiple).
        fired = [t for t in 0.03:0.03:0.30 if S.should_fire(sched, ctx(t))]
        # Exactly one firing per crossed multiple (0.1, 0.2, 0.3) -> 3 firings.
        @test length(fired) == 3
        @test fired[1] ≈ 0.12
        @test fired[2] ≈ 0.21
        # A fresh schedule with an exact landing still fires.
        @test S.should_fire(S.TimeInterval(0.1), ctx(0.1))
        # interval <= 0 or time <= 0 never fires.
        @test S.should_fire(S.TimeInterval(0.0), ctx(0.5)) == false
        @test S.should_fire(S.TimeInterval(0.1), ctx(0.0)) == false
    end

    # ── Finding 3: validate_parameters rejects transform-invalid grids ────────
    @testset "validate_parameters rejects nlat<lmax+1 and nlon<2*mmax+1" begin
        ok, errs, _ = S.validate_parameters(
            S.SolverParameters(lmax = 4, mmax = 4, nlat = 1, nlon = 16))
        @test ok == false
        @test any(e -> occursin("nlat", e), errs)

        ok2, errs2, _ = S.validate_parameters(
            S.SolverParameters(lmax = 4, mmax = 4, nlat = 12, nlon = 4))
        @test ok2 == false
        @test any(e -> occursin("nlon", e), errs2)

        # A transform-valid grid still validates.
        ok3, _, _ = S.validate_parameters(
            S.SolverParameters(lmax = 4, mmax = 4, nlat = 12, nlon = 16))
        @test ok3 == true
    end

    # ── Finding 4: grid defaults valid for small lmax ─────────────────────────
    @testset "Grid defaults are valid for lmax=1" begin
        gs = S.SphericalShellGrid(lmax = 1, nr = 8, nr_inner = 2)
        @test gs isa S.SphericalShellGrid
        @test gs.nlat >= gs.lmax + 1
        @test gs.nlon >= 2 * gs.mmax + 1
        gb = S.SphericalBallGrid(lmax = 1, nr = 8)
        @test gb isa S.SphericalBallGrid
        @test gb.nlat >= gb.lmax + 1
    end

    # ── Finding 5: FieldWriter rejects unknown selectors ──────────────────────
    @testset "FieldWriter rejects unknown field selectors" begin
        @test_throws ArgumentError S.FieldWriter("out";
            schedule = S.IterationInterval(1), fields = [:not_a_field])
        # Known selectors still construct.
        fw = S.FieldWriter("out";
            schedule = S.IterationInterval(1), fields = [:velocity, :temperature])
        @test fw isa S.FieldWriter
    end

    # ── Finding 2: Simulation rejects invalid run controls ────────────────────
    @testset "Simulation rejects invalid run controls" begin
        if MPI.Finalized()
            @warn "MPI finalized; skipping Simulation control validation test"
        else
            MPI.Initialized() || MPI.Init()
            grid = S.SphericalShellGrid(S.CPU();
                lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
            model = S.GeodynamoModel(grid;
                Ek = 1e-2, Ra = 1e4, include_magnetic = false, include_composition = false)
            @test_throws ArgumentError S.Simulation(model; dt = 1e-4, courant = -1.0)
            @test_throws ArgumentError S.Simulation(model; dt = 1e-4, stop_iteration = 0)
            @test_throws ArgumentError S.Simulation(model; dt = 1e-4, stop_time = -1.0)
            # Valid controls still construct.
            @test S.Simulation(model; dt = 1e-4, stop_time = 1.0, stop_iteration = 5) isa
                  S.Simulation
        end
    end
end

@testset "Codex round-4 hardening" begin
    S = GeoDynamo

    # ── Finding 2: WallTimeInterval guards interval <= 0 ──────────────────────
    @testset "WallTimeInterval rejects interval <= 0" begin
        wctx(t) = S._ScheduleContext(0.0, 0, Float64(t))
        @test S.should_fire(S.WallTimeInterval(0.0), wctx(5.0)) == false
        @test S.should_fire(S.WallTimeInterval(-1.0), wctx(5.0)) == false
        # A positive interval still fires once enough wall time has elapsed.
        @test S.should_fire(S.WallTimeInterval(1.0), wctx(2.0)) == true
    end

    # ── Finding 3: explicit missing parameter file errors ─────────────────────
    @testset "load_parameters errors on explicit missing file" begin
        @test_throws Exception S.load_parameters("/definitely/missing_params_file_xyz.jl")
    end

    # ── Finding 4: load_parameters rejects invalid parameter values ───────────
    @testset "load_parameters rejects invalid parameter values" begin
        tmp = tempname() * ".jl"
        write(tmp, "timestep = -1e-4\nstop_iteration = 0\ncourant = -1.0\n")
        try
            @test_throws Exception S.load_parameters(tmp)
        finally
            rm(tmp; force = true)
        end
    end

    # ── Finding 5: initialize_solver_state rejects invalid params ─────────────
    @testset "initialize_solver_state rejects invalid params" begin
        @test_throws Exception S.initialize_solver_state(Float64;
            params = S.SolverParameters(timestep = -1e-5))
        @test_throws Exception S.initialize_solver_state(Float64;
            params = S.SolverParameters(courant = -1.0))
    end

    # ── Findings 1/6/7 need a built model (MPI-aware) ─────────────────────────
    @testset "model-driven round-4 (F1 fail-loud, F6 nan-check, F7 magnetic)" begin
        if MPI.Finalized()
            @warn "MPI finalized; skipping model-driven round-4 tests"
        else
            MPI.Initialized() || MPI.Init()
            grid = S.SphericalShellGrid(S.CPU();
                lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
            model = S.GeodynamoModel(grid;
                Ek = 1e-2, Ra = 1e4, include_magnetic = false, include_composition = false)
            state = model.state

            # Finding 7: include_magnetic=false must not emit magnetic field keys.
            allf = S.extract_all_fields(state)
            @test !haskey(allf, "magnetic_toroidal")
            @test !haskey(allf, "magnetic_poloidal")

            # Finding 6: NaN check must read the current SolverState layout
            # (state.fields.*) and not throw "no field velocity"; clean -> false.
            @test S.check_simulation_state_for_nan(state, 0) == false

            # Finding 1: a scheduled write that fails must be fail-loud, not warn
            # and continue. Point the writer at an existing FILE so the NetCDF
            # write cannot create its output and must raise.
            badfile = tempname()
            touch(badfile)
            fw = S.FieldWriter(badfile;
                schedule = S.IterationInterval(1), fields = [:temperature])
            sim = S.Simulation(model; dt = 1e-4, stop_iteration = 1)
            ctx1 = S._ScheduleContext(state.time, 1, 0.0)
            try
                @test_throws Exception S._run_output_writer!(fw, sim, ctx1)
            finally
                rm(badfile; force = true)
            end
        end
    end
end
