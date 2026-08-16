# ================================================================================
# Regression tests for the 13 confirmed findings of the max-effort src/ review
# ================================================================================
#
#   F1  api/output_writers.jl:133  fresh TimeTracker per firing -> every write
#                                  reuses output #1 and deletes the previous file
#   F2  physics/velocity/solver.jl:323  poloidal W-split memo never invalidated
#                                  on a dt change (dt is baked into its LU)
#   F3  bcs/bcs.jl:565             BC spectral index built l-major while the
#                                  solver reads it m-major
#   F4  api/initial_conditions.jl:175  is_initialized set after ONE field's IC,
#                                  so every other field family is never initialized
#   F5  core/parameters.jl:489     a construction error silently returns ALL defaults
#   F6  solver/state.jl:635        absent checkpoint fields restore silently while
#                                  the clock is advanced
#   F7  Ball/Ball.jl:71            integration_weights left all zeros -> every
#                                  volume-integral diagnostic is 0 in :ball
#   F8  api/simulation.jl:83       nan_checker stops from a rank-LOCAL scan
#   F9  api/simulation.jl:75       wall_time_limit_exceeded uses rank-local time()
#   F10 api/output_writers.jl:110  WallTimeInterval gates a collective rank-locally
#   F11 api/simulation.jl:215      implicit_theta / krylov kwargs silently dropped
#   F12 api/simulation.jl:266      gpu=:auto omits the :ball / topography /
#                                  time-dependent-BC scope limits
#   F13 bcs/interpolation.jl:65    periodic wrap is local to find_grid_indices, so
#                                  get_interpolation_weights extrapolates wildly
# ================================================================================

using Test
using MPI
using GeoDynamo

# Whitespace-insensitive source matching (same convention as the *_static_checks
# files): used where the defect is a MISSING collective that a single-rank test
# cannot observe, so the contract has to be pinned in the source itself.
_crm_wsn(s) = replace(s, r"\s+" => "")
_crm_occ(pat::AbstractString, src) = occursin(_crm_wsn(pat), _crm_wsn(src))

const CRM_ROOT = normpath(joinpath(@__DIR__, ".."))
const CRM_SIMULATION_SRC = read(joinpath(CRM_ROOT, "src", "api", "simulation.jl"), String)
const CRM_CALLBACKS_SRC = read(joinpath(CRM_ROOT, "src", "api", "callbacks.jl"), String)
const CRM_WRITERS_SRC = read(joinpath(CRM_ROOT, "src", "api", "output_writers.jl"), String)

@testset "Max-effort code review fixes" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping max-effort code review fix tests"
        return
    end
    MPI.Initialized() || MPI.Init()

    crm_grid() = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU();
        lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
    crm_model(; kwargs...) = GeoDynamo.GeodynamoModel(crm_grid();
        Ek = 1e-2, Ra = 1e4, include_magnetic = false, include_composition = false,
        kwargs...)

    # ── F1: each firing must get its own output number ────────────────────────
    @testset "F1 output writers keep a persistent TimeTracker" begin
        fw = GeoDynamo.FieldWriter("out"; schedule = GeoDynamo.IterationInterval(1))
        cw = GeoDynamo.CheckpointWriter("out"; schedule = GeoDynamo.IterationInterval(1))

        # The tracker must live on the writer, not be rebuilt per firing.
        cfg = GeoDynamo.OutputConfig(
            GeoDynamo.MIXED_FIELDS, "out", "geodynamo", true, true, true,
            Float64, -1, true, 0.0, Inf, Inf, 1e-10)
        t1 = GeoDynamo._writer_tracker!(fw, cfg, 0.0)
        t2 = GeoDynamo._writer_tracker!(fw, cfg, 1.0)
        @test t1 === t2                      # same object -> counters survive
        t1.output_count += 1
        @test GeoDynamo._writer_tracker!(fw, cfg, 2.0).output_count == 1

        c1 = GeoDynamo._writer_tracker!(cw, cfg, 0.0)
        @test GeoDynamo._writer_tracker!(cw, cfg, 1.0) === c1

        # And the tracker must still authorize a write on every later firing.
        @test GeoDynamo.should_output_now(t1, 3.0, cfg) == true
        GeoDynamo.update_tracker!(t1, 3.0, cfg, true, false)
        @test GeoDynamo.should_output_now(t1, 4.0, cfg) == true
        @test t1.output_count == 2
    end

    @testset "F1b restart seeds writer counters without clobbering files" begin
        mktempdir() do dir
            first_fields = GeoDynamo.FieldWriter(dir;
                schedule = GeoDynamo.IterationInterval(1), fields = [:temperature])
            first_checkpoints = GeoDynamo.CheckpointWriter(dir;
                schedule = GeoDynamo.IterationInterval(1))
            first = GeoDynamo.Simulation(crm_model(); Δt = 1e-4, stop_iteration = 1,
                output_writers = (fields = first_fields, checkpoints = first_checkpoints))
            GeoDynamo.run!(first)

            hist1 = joinpath(dir, "geodynamo_shell_hist_1.nc")
            restart1 = joinpath(dir, "geodynamo_shell_restart_1.nc")
            @test isfile(hist1)
            @test isfile(restart1)

            resumed_fields = GeoDynamo.FieldWriter(dir;
                schedule = GeoDynamo.IterationInterval(1), fields = [:temperature])
            resumed_checkpoints = GeoDynamo.CheckpointWriter(dir;
                schedule = GeoDynamo.IterationInterval(1))
            resumed = GeoDynamo.Simulation(crm_model(); Δt = 1e-4, stop_iteration = 2,
                restart_from = dir,
                output_writers = (
                    fields = resumed_fields,
                    checkpoints = resumed_checkpoints,
                ))

            # The checkpoint writer knows restart #1, while the independent field
            # writer's count must be recovered from the existing history filename.
            fields_tracker = resumed_fields._tracker[]
            checkpoint_tracker = resumed_checkpoints._tracker[]
            @test fields_tracker !== nothing
            @test checkpoint_tracker !== nothing
            fields_tracker === nothing || @test fields_tracker.output_count == 1
            checkpoint_tracker === nothing || @test checkpoint_tracker.restart_count == 1

            GeoDynamo.run!(resumed)
            @test isfile(hist1)
            @test isfile(restart1)
            @test isfile(joinpath(dir, "geodynamo_shell_hist_2.nc"))
            @test isfile(joinpath(dir, "geodynamo_shell_restart_2.nc"))
        end
    end

    # ── F2: the poloidal W-split bakes dt, so a dt change must rebuild it ─────
    @testset "F2 poloidal W-split is invalidated on a dt change" begin
        model = crm_model()
        state = model.state
        sim = GeoDynamo.Simulation(model; Δt = 1e-4)
        GeoDynamo.time_step!(sim)
        split_a = state.timestep_caches.poloidal_split
        @test split_a !== nothing
        @test split_a.dt == 1e-4             # the split must record its own dt

        GeoDynamo.time_step!(model, 5e-4)    # dt change
        split_b = state.timestep_caches.poloidal_split
        @test split_b !== nothing
        @test split_b.dt == 5e-4             # rebuilt, not the stale 1e-4 LU
        @test split_b !== split_a
    end

    # ── F3: boundary spectral coefficients must use the canonical m-major index ─
    @testset "F3 boundary spectral index is m-major" begin
        lmax = mmax = 2
        nlat, nlon = 12, 16
        cfg = GeoDynamo.create_shtnskit_config(
            lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = 4)

        # Canonical (m-major) index 3 is (l=2, m=0); the old l-major index 3 was
        # (l=1, m=1). Synthesizing basis vector e_3 therefore gives an
        # AXISYMMETRIC field under the canonical ordering and a phi-dependent one
        # under the old ordering — an anchor independent of the transform pair.
        @test cfg.l_values[3] == 2
        @test cfg.m_values[3] == 0
        e3 = zeros(Float64, cfg.nlm)
        e3[3] = 1.0
        phys = GeoDynamo.shtns_spectral_to_physical(e3, cfg, nlat, nlon)
        for j in 2:nlon
            @test isapprox(phys[:, j], phys[:, 1]; atol = 1e-10)
        end

        # Forward direction, anchored the same way: a physical field built from
        # P_2(cos θ) is pure (l=2, m=0) and must land on canonical index 3.
        theta = cfg.theta_grid
        p2 = [0.5 * (3 * cos(t)^2 - 1) for t in theta]
        axisym = repeat(p2, 1, nlon)
        coeffs = GeoDynamo.shtns_physical_to_spectral(axisym, cfg)
        @test argmax(abs.(coeffs)) == 3
        # every other coefficient must be negligible
        others = [i for i in 1:cfg.nlm if i != 3]
        @test maximum(abs.(coeffs[others])) < 1e-8 * abs(coeffs[3])
    end

    # ── F4: a user IC must not suppress initialization of the other families ──
    @testset "F4 set_initial_condition! still initializes the other fields" begin
        model = crm_model(initial_conditions = (velocity = GeoDynamo.ZeroIC(),))
        tspec = parent(model.state.fields.temperature.spectral.data_real)
        # The conductive (0,0) background belongs to initialize_temperature_field!;
        # if the velocity IC short-circuited it, temperature is identically zero.
        @test any(!iszero, tspec)
        @test model.state.is_initialized

        # A later IC must not re-initialize (and so must not clobber) live fields.
        GeoDynamo.set_initial_condition!(model, :temperature, 0.25)
        before = copy(parent(model.state.fields.temperature.spectral.data_real))
        GeoDynamo.set_initial_condition!(model, :velocity, GeoDynamo.ZeroIC())
        @test parent(model.state.fields.temperature.spectral.data_real) == before
    end

    # ── F5: a bad parameter file must not silently become ALL defaults ────────
    @testset "F5 load_parameters_from_file fails loud on a bad value" begin
        mktempdir() do dir
            f = joinpath(dir, "params.jl")
            # geometry must be a Symbol; a String cannot convert -> the
            # SolverParameters constructor throws.
            write(f, "geometry = \"shell\"\nEk = 1e-6\nnr = 128\n")
            @test_throws ErrorException GeoDynamo.load_parameters_from_file(f)
            @test_throws ErrorException GeoDynamo.load_parameters(f)
        end
        # A missing implicit default file still falls back (unchanged behaviour).
        @test GeoDynamo.load_parameters_from_file("nonexistent_file_98765.jl") isa
              GeoDynamo.SolverParameters
    end

    # ── F6: a checkpoint missing an enabled field must not restore silently ───
    @testset "F6 restore_fields_from_restart! requires the enabled fields" begin
        model = crm_model()
        @test_throws ArgumentError GeoDynamo.restore_fields_from_restart!(
            model.state, Dict{String, Any}())

        # A dict missing only the magnetic pair, on a magnetic model, must throw
        # and must name the missing field.
        mag_grid = crm_grid()
        mag_model = GeoDynamo.GeodynamoModel(mag_grid; Ek = 1e-2, Ra = 1e4,
            include_magnetic = true, include_composition = false)
        full = GeoDynamo.extract_all_fields(mag_model.state)
        partial = Dict{String, Any}(k => v for (k, v) in full
        if !startswith(k, "magnetic"))
        err = try
            GeoDynamo.restore_fields_from_restart!(mag_model.state, partial)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("magnetic", sprint(showerror, err))

        # The complete dict restores without error.
        @test GeoDynamo.restore_fields_from_restart!(mag_model.state, full) ===
              mag_model.state
    end

    # ── F7: ball radial domain must carry real quadrature weights ────────────
    @testset "F7 ball integration weights are real" begin
        nr = 24
        dom = GeoDynamo.GeoDynamoBall.create_ball_radial_domain(nr)
        w = dom.integration_weights
        r = dom.r[1:nr, 4]
        @test length(w) == nr
        @test any(!iszero, w)
        @test all(>=(0.0), w)
        # The diagnostics integrand is r^2 * f(r); ∫_0^1 r^2 dr = 1/3.
        @test sum(w .* r .^ 2) ≈ 1 / 3 rtol = 1e-8
        # ∫_0^1 r^4 dr = 1/5 pins more than one moment.
        @test sum(w .* r .^ 4) ≈ 1 / 5 rtol = 1e-8
    end

    # ── F8: the NaN stop decision must be global ──────────────────────────────
    @testset "F8 nan_checker reduces its stop flag across ranks" begin
        # At one rank the reduction is the identity, so the helper's semantics are
        # tested directly and the call site is pinned in the source.
        @test GeoDynamo._any_rank_flag(true) == true
        @test GeoDynamo._any_rank_flag(false) == false
        @test _crm_occ("_any_rank_flag(r.has_issue)", CRM_SIMULATION_SRC)
        @test _crm_occ("_any_rank_flag", CRM_CALLBACKS_SRC)
    end

    # ── F9/F10: wall-clock decisions must be rank-consistent ──────────────────
    @testset "F9/F10 wall-clock schedules use a collective elapsed time" begin
        model = crm_model()
        sim = GeoDynamo.Simulation(model; Δt = 1e-4)
        @test GeoDynamo._collective_wtime(sim) == 0.0     # before run!
        sim._wall_start = time() - 5.0
        @test GeoDynamo._collective_wtime(sim) ≈ 5.0 atol = 1.0

        # No rank may derive a schedule- or stop-decision from a bare local clock.
        @test _crm_occ("_collective_wtime(sim)", CRM_CALLBACKS_SRC)
        @test _crm_occ("_collective_wtime(sim)", CRM_WRITERS_SRC)
        @test _crm_occ("_collective_wtime(sim)", CRM_SIMULATION_SRC)
        @test !_crm_occ("wtime = sim._wall_start > 0.0 ? time() - sim._wall_start : 0.0",
            CRM_CALLBACKS_SRC)
        @test !_crm_occ("wtime = sim._wall_start > 0.0 ? time() - sim._wall_start : 0.0",
            CRM_WRITERS_SRC)
        @test !_crm_occ("(time() - sim._wall_start) >= sim.wall_time_limit",
            CRM_SIMULATION_SRC)
    end

    # ── F11: implicit_theta / krylov kwargs must reach the solver ─────────────
    @testset "F11 implicit_theta and krylov kwargs are honoured" begin
        p = GeoDynamo.SolverParameters(nr = 16, nr_inner = 4)

        # struct + override: the override must be folded into the struct.
        opts = GeoDynamo._resolve_timestepper(
            GeoDynamo.CNAB2(), nothing, 1.0, nothing, nothing, p)
        @test opts.timestepper isa GeoDynamo.CNAB2
        @test opts.timestepper.implicit_theta == 1.0

        etd = GeoDynamo._resolve_timestepper(
            GeoDynamo.ETD(), nothing, nothing, 30, 1e-12, p)
        @test etd.timestepper.krylov_dimension == 30
        @test etd.timestepper.tolerance == 1e-12

        # default timestepper from params + override
        dflt = GeoDynamo._resolve_timestepper(nothing, nothing, 0.75, nothing, nothing, p)
        @test GeoDynamo._timestepper_implicit_theta(dflt.timestepper, p) == 0.75

        # A scheme that cannot carry the requested override must say so, not
        # silently ignore it.
        @test_throws ArgumentError GeoDynamo._resolve_timestepper(
            GeoDynamo.ExponentialRungeKutta2(), nothing, 1.0, nothing, nothing, p)

        # End to end through the public constructor.
        model = crm_model()
        GeoDynamo.Simulation(model; Δt = 1e-4, implicit_theta = 1.0)
        @test model.state.parameters.timestepper.implicit_theta == 1.0
    end

    # ── F12: gpu=:auto must decline unsupported configs, not hard-error later ─
    @testset "F12 gpu stepping pre-screens every device scope limit" begin
        model = crm_model()
        p = model.state.parameters

        # :ball geometry is CPU-only on the device path.
        model.state.parameters = GeoDynamo.SolverParameters(;
            (f => getfield(p, f) for f in fieldnames(GeoDynamo.SolverParameters))...,
            geometry = :ball)
        @test (@test_logs (:warn,) match_mode = :any GeoDynamo._resolve_gpu_stepping(
            true, model, GeoDynamo.CNAB2())) == false
        model.state.parameters = p

        # A supported shell config still resolves to true.
        @test GeoDynamo._resolve_gpu_stepping(true, model, GeoDynamo.CNAB2()) == true

        # Topography enabled -> CPU path.
        topo_model = GeoDynamo.GeodynamoModel(crm_grid(); Ek = 1e-2, Ra = 1e4,
            include_magnetic = false, include_composition = false,
            topography_enabled = true)
        @test (@test_logs (:warn,) match_mode = :any GeoDynamo._resolve_gpu_stepping(
            true, topo_model, GeoDynamo.CNAB2())) == false
    end

    # ── F13: the periodic wrap must reach the weight computation ──────────────
    @testset "F13 periodic interpolation weights see the wrapped target" begin
        bcs = GeoDynamo.bcs
        nlon = 64
        dphi = 2pi / nlon
        phi = collect(range(0.0, step = dphi, length = nlon))   # 0 .. 2pi-dphi

        # A target grid built as range(0, 2pi, length=nlon) ends EXACTLY at 2pi,
        # which wraps to 0.0 -> indices (1, 2) and weights (1, 0).
        idx = bcs.find_grid_indices(phi, 2pi; is_periodic = true)
        @test idx == (1, 2)
        w1, w2 = bcs.get_interpolation_weights(phi, 2pi, idx; is_periodic = true)
        @test w1 ≈ 1.0
        @test w2 ≈ 0.0

        # Interpolating a smooth field at 2pi must return its value at 0, not a
        # ~nlon-fold extrapolation.
        f = [cos(p) for p in phi]
        @test w1 * f[1] + w2 * f[2] ≈ f[1]

        # Genuinely wrapped targets in (phi[end], 2pi) still interpolate.
        mid = phi[end] + dphi / 2
        widx = bcs.find_grid_indices(phi, mid; is_periodic = true)
        @test widx == (nlon, 1)
        u1, u2 = bcs.get_interpolation_weights(phi, mid, widx; is_periodic = true)
        @test u1 ≈ 0.5
        @test u2 ≈ 0.5
    end
end
