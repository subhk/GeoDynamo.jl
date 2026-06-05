# ================================================================================
# Regression tests for the 6 codex findings on the public API layer
# ================================================================================
#
#   1. Simulation(restart_from=...) must fail loud, not silently start from t=0
#   2. Simulation / time_step! must reject dt <= 0
#   3. EnergyDiagnostics / HealthCheck / SolenoidalMonitor must compute real
#      diagnostics, not log placeholders
#   4. An Inf output/restart interval must mean "never" — a FieldWriter must not
#      emit restart files
#   5. SolverParameters validation must catch nr_inner >= nr and courant <= 0
#   6. RandomPerturbation must reject a negative lmax
# ================================================================================

using Test
using MPI
using GeoDynamo

@testset "Codex API fixes" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping codex API fix tests"
        return
    end
    MPI.Initialized() || MPI.Init()

    # ── Finding 5: parameter validation (nr_inner < nr, courant > 0) ──────────
    @testset "validation catches nr_inner >= nr and courant <= 0" begin
        is_valid, errors, _ = GeoDynamo.validate_parameters(
            GeoDynamo.SolverParameters(nr = 16, nr_inner = 99, courant = -1.0))
        @test is_valid == false
        @test any(e -> occursin("nr_inner", e), errors)
        @test any(e -> occursin("courant", e), errors)

        ok, _, _ = GeoDynamo.validate_parameters(
            GeoDynamo.SolverParameters(nr = 32, nr_inner = 8, courant = 0.5))
        @test ok == true
    end

    # ── Finding 6: RandomPerturbation rejects negative lmax ───────────────────
    @testset "RandomPerturbation rejects negative lmax" begin
        @test_throws ArgumentError GeoDynamo.RandomPerturbation(amplitude = 0.1, lmax = -1)
        @test GeoDynamo.RandomPerturbation(amplitude = 0.1, lmax = 8) isa
              GeoDynamo.RandomPerturbation
    end

    # ── Finding 4: Inf interval means "never"; FieldWriter writes no restart ──
    @testset "Inf output/restart interval never fires" begin
        mkcfg(out, rst) = GeoDynamo.OutputConfig(
            GeoDynamo.MIXED_FIELDS, "./o", "geodynamo",
            true, true, true, Float64, -1, true,
            out, rst, Inf, 1e-10)

        # Both intervals Inf -> nothing ever fires (this is what "no schedule" means).
        cfg = mkcfg(Inf, Inf)
        tr = GeoDynamo.create_time_tracker(cfg, 5.0)
        @test GeoDynamo.should_output_now(tr, 6.0, cfg) == false
        @test GeoDynamo.should_restart_now(tr, 6.0, cfg) == false

        # FieldWriter layout: output every call (interval 0.0), never restart (Inf).
        cfg2 = mkcfg(0.0, Inf)
        tr2 = GeoDynamo.create_time_tracker(cfg2, 4.0)
        @test GeoDynamo.should_output_now(tr2, 5.0, cfg2) == true
        @test GeoDynamo.should_restart_now(tr2, 5.0, cfg2) == false
    end

    # ── Shared tiny model for the model-driven findings (1, 2, 3) ─────────────
    grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU();
        lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
    mkmodel() = GeoDynamo.GeodynamoModel(grid;
        Ek = 1e-2, Ra = 1e4, include_magnetic = false, include_composition = false)

    # ── Finding 2: dt <= 0 rejected ───────────────────────────────────────────
    @testset "Simulation / time_step! reject non-positive dt" begin
        m = mkmodel()
        @test_throws ArgumentError GeoDynamo.Simulation(m; dt = 0.0)
        @test_throws ArgumentError GeoDynamo.Simulation(m; dt = -1e-5)
        @test_throws ArgumentError GeoDynamo.time_step!(m, -1e-5)
        @test GeoDynamo.Simulation(m; dt = 1e-4) isa GeoDynamo.Simulation
    end

    # ── Finding 1: restart_from fails loud on a missing restart ───────────────
    @testset "Simulation restart_from errors on missing restart" begin
        m = mkmodel()
        missing_dir = joinpath(tempdir(), "geodynamo_no_such_restart_dir_xyz")
        @test_throws Exception GeoDynamo.Simulation(m; dt = 1e-4, restart_from = missing_dir)
    end

    # ── Finding 3: EnergyDiagnostics computes real kinetic energy ─────────────
    @testset "EnergyDiagnostics computes real kinetic energy" begin
        m = mkmodel()
        GeoDynamo.set_initial_condition!(m, :velocity,
            GeoDynamo.RandomPerturbation(amplitude = 1e-2, lmax = 4, seed = 1))
        e = GeoDynamo._energy_diagnostics(m)
        @test e.kinetic isa Float64
        @test isfinite(e.kinetic)
        @test e.kinetic > 0.0
        @test e.magnetic == 0.0   # model built without a magnetic field
    end

    # ── Finding 3: HealthCheck detects non-finite values and aborts ───────────
    @testset "HealthCheck detects non-finite values and aborts" begin
        m = mkmodel()
        @test GeoDynamo._health_check(m).has_issue == false

        parent(m.state.fields.velocity.toroidal.data_real)[1] = NaN
        r = GeoDynamo._health_check(m)
        @test r.has_issue == true
        @test !isempty(r.fields)

        cb = GeoDynamo.HealthCheck(schedule = GeoDynamo.IterationInterval(1), abort = true)
        sim = GeoDynamo.Simulation(m; dt = 1e-4)
        @test_throws Exception GeoDynamo._fire_callback!(cb, sim)
    end

    # ── Finding 3: SolenoidalMonitor runs the real solenoidal diagnostic ──────
    @testset "SolenoidalMonitor records a real solenoidal diagnostic" begin
        m = mkmodel()
        before = length(m.state.solenoidal_monitor.velocity_div_l2)
        d = GeoDynamo._solenoidal_diagnostics(m)
        @test d.velocity_linf isa Float64
        @test isfinite(d.velocity_linf)
        @test length(m.state.solenoidal_monitor.velocity_div_l2) == before + 1
    end
end
