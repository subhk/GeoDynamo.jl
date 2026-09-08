# ================================================================================
# Review 2026-09-05 — legacy checkpoints, ZeroIC history flag, atomic view swap
# ================================================================================
#
# Serial regressions for the 2026-09-05 review of the uncommitted working tree on
# `test/mpi-control-plane-invariants` (the SHTnsKit v2 migration + checkpoint
# history bookkeeping).
# ================================================================================

using Test
using MPI
using GeoDynamo

const topo0905 = GeoDynamo.bcs.topography

function _review0905_state(; include_magnetic = false, include_composition = false)
    params = GeoDynamo.SolverParameters(
        geometry = :shell, lmax = 3, mmax = 3, nlat = 10, nlon = 16, nr = 8,
        nr_inner = 4, radial_bandwidth = 3, radius_ratio = 0.35,
        Ek = 1e-3, Ra = 1e3, Pm = 1.0, Pr = 1.0, Sc = 1.0, timestep = 1e-4,
        timestepper = GeoDynamo.CNAB2(),
        include_magnetic = include_magnetic,
        include_composition = include_composition,
    )
    state = GeoDynamo.initialize_solver_state(Float64; params)
    GeoDynamo.initialize_solver_fields!(state)
    return state
end

_review0905_pair(field) = Dict(
    "real" => copy(parent(field.data_real)),
    "imag" => copy(parent(field.data_imag)))

@testset "Review 2026-09-05 fixes" begin
    MPI.Initialized() || MPI.Init()

    # ── F1: checkpoints written before the history keys existed still restore ──
    @testset "F1 legacy checkpoint without history keys restores and re-bootstraps" begin
        # Every released version wrote only the primary fields. Those files were
        # valid restart points (AB2 history was bootstrapped from the loaded
        # state). Hard-requiring the new history keys rejects all of them with a
        # remedy ("disable the missing field families") that cannot be followed.
        state = _review0905_state()
        fill!(parent(state.fields.velocity.toroidal.data_real), 3.0)
        legacy = Dict{String, Any}(
            "velocity_toroidal" => _review0905_pair(state.fields.velocity.toroidal),
            "velocity_poloidal" => _review0905_pair(state.fields.velocity.poloidal),
            "temperature" => copy(parent(state.fields.temperature.temperature.data)),
            "temperature_spectral" => _review0905_pair(state.fields.temperature.spectral),
        )

        target = _review0905_state()
        target.runtime.timestep_state.needs_ab2_bootstrap = false
        fill!(parent(target.fields.velocity.prev_nl_toroidal.data_real), 7.0)

        GeoDynamo.restore_fields_from_restart!(target, legacy)
        @test all(==(3.0), parent(target.fields.velocity.toroidal.data_real))
        # History is absent from the file, so the next step must rebuild it.
        @test target.runtime.timestep_state.needs_ab2_bootstrap === true

        # A checkpoint that is missing a PRIMARY field is still a hard error.
        broken = copy(legacy)
        delete!(broken, "temperature_spectral")
        @test_throws ArgumentError GeoDynamo.restore_fields_from_restart!(
            _review0905_state(), broken)
    end

    @testset "F1 partial history in a checkpoint forces a re-bootstrap" begin
        # A file that carries some history but not all of it (truncated write,
        # or a run whose enabled families changed) must not be trusted as an
        # AB2 history either.
        state = _review0905_state()
        partial = Dict{String, Any}(
            "velocity_toroidal" => _review0905_pair(state.fields.velocity.toroidal),
            "velocity_poloidal" => _review0905_pair(state.fields.velocity.poloidal),
            "temperature" => copy(parent(state.fields.temperature.temperature.data)),
            "temperature_spectral" => _review0905_pair(state.fields.temperature.spectral),
            "velocity_prev_nl_toroidal" =>
                _review0905_pair(state.fields.velocity.prev_nl_toroidal),
            "needs_ab2_bootstrap" => false,
        )
        target = _review0905_state()
        target.runtime.timestep_state.needs_ab2_bootstrap = false
        GeoDynamo.restore_fields_from_restart!(target, partial)
        @test target.runtime.timestep_state.needs_ab2_bootstrap === true
    end

    @testset "F1 complete history in a checkpoint is restored verbatim" begin
        state = _review0905_state()
        fill!(parent(state.fields.velocity.prev_nl_poloidal.data_imag), -5.0)
        state.runtime.timestep_state.needs_ab2_bootstrap = false
        full = GeoDynamo.extract_all_fields(state)
        target = _review0905_state()
        target.runtime.timestep_state.needs_ab2_bootstrap = true
        GeoDynamo.restore_fields_from_restart!(target, full)
        @test all(==(-5.0), parent(target.fields.velocity.prev_nl_poloidal.data_imag))
        @test target.runtime.timestep_state.needs_ab2_bootstrap === false
    end

    # ── F2: ZeroIC zeroes the AB2 history, so it must also ask for a bootstrap ──
    @testset "F2 ZeroIC mid-run re-arms the AB2 bootstrap" begin
        grid = GeoDynamo.SphericalShellGrid(GeoDynamo.CPU(); lmax = 4, mmax = 4,
            nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
        for field in (:velocity, :temperature)
            model = GeoDynamo.GeodynamoModel(grid; Ek = 1e-2, Ra = 1e4)
            GeoDynamo.initialize_fields!(model.state)
            # bootstrap_solver_history! has run: the flag is down.
            model.state.runtime.timestep_state.needs_ab2_bootstrap = false
            GeoDynamo.set_initial_condition!(model, field, GeoDynamo.ZeroIC())
            # prev_nl_* are now zero; CNAB2 would extrapolate 1.5*N^n - 0.5*0
            # unless the next step rebuilds the history.
            @test model.state.runtime.timestep_state.needs_ab2_bootstrap === true
        end
    end

    # ── F3: the lock-free cross-Gaunt view must be swapped atomically ─────────
    @testset "F3 G_cross_view is an atomic field" begin
        # `get_cross_gaunt` reads `cache.G_cross_view` without the lock while
        # `_publish_cross_gaunt_view!` replaces it. A plain field store/load pair
        # is a data race under Julia's memory model: the load may be hoisted out
        # of a caller's loop and a thread keeps serving (and re-locking on) the
        # stale view. An `@atomic` field makes the non-atomic access an error.
        cache = topo0905.GauntTensorCache{Float64}(4, 4)
        @test_throws ConcurrencyViolationError getfield(cache, :G_cross_view, :not_atomic)
        view = @atomic cache.G_cross_view
        @test view isa Dict{NTuple{6, Int}, Float64}

        # Publication and lock-free hits still work through the atomic field.
        nonzero = (2, 1, 1, 1, 2, 0)
        expected = topo0905.get_cross_gaunt(cache, nonzero...)
        @test abs(expected) > 1e-14
        topo0905.flush_cross_gaunt_view!(cache)
        @test haskey((@atomic cache.G_cross_view), nonzero)
        @test topo0905.get_cross_gaunt(cache, nonzero...) == expected
    end
end
