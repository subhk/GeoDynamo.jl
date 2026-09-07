# ================================================================================
# Review 2026-09-07 — regressions this branch introduced against main
# ================================================================================
#
# Three defects the 2026-09-07 review of the uncommitted working tree found, all
# of them behaviour that WORKED on `main` and stopped working here:
#
#   R1  restart hard-requires two keys the branch itself invented, so every
#       checkpoint written by any released version fails to restore a
#       conducting-inner-core run.
#   R2  `_ensure_output_directory_collectively!` was added and wired into the
#       FieldWriter path only, so a CheckpointWriter at a fresh path dies
#       mid-step instead of creating its directory.
#   R3  `BoundaryValueBase` became mutable and `reset_boundary_to_base!` now
#       WRITES the shared cache entry, but the writes stayed outside the lock
#       that `mark_boundary_applied!` takes.
# ================================================================================

using Test
using MPI
using GeoDynamo

const topo0907 = GeoDynamo.bcs.topography

function _review0907_state(; magnetic_inner_bc = :insulating)
    params = GeoDynamo.SolverParameters(
        geometry = :shell, lmax = 3, mmax = 3, nlat = 10, nlon = 16, nr = 8,
        nr_inner = 4, radial_bandwidth = 3, radius_ratio = 0.35,
        Ek = 1e-3, Ra = 1e3, Pm = 1.0, Pr = 1.0, Sc = 1.0, timestep = 1e-4,
        timestepper = GeoDynamo.CNAB2(),
        include_magnetic = true,
        include_composition = false,
        magnetic_inner_bc = magnetic_inner_bc,
    )
    state = GeoDynamo.initialize_solver_state(Float64; params)
    GeoDynamo.initialize_solver_fields!(state)
    return state
end

_review0907_pair(field) = Dict(
    "real" => copy(parent(field.data_real)),
    "imag" => copy(parent(field.data_imag)))

# An AbstractMatrix whose element access yields, so a concurrent task is
# guaranteed a scheduling window in the middle of any loop or `copy` over it.
# Without it, `@async` tasks never interleave on plain arrays and a lock-scope
# test cannot observe anything.
struct YieldMatrix{T} <: AbstractMatrix{T}
    data::Matrix{T}
end
Base.size(m::YieldMatrix) = size(m.data)
Base.getindex(m::YieldMatrix, i::Int) = (yield(); m.data[i])
Base.getindex(m::YieldMatrix, i::Int, j::Int) = (yield(); m.data[i, j])
Base.setindex!(m::YieldMatrix, v, i::Int) = (yield(); m.data[i] = v)
Base.setindex!(m::YieldMatrix, v, i::Int, j::Int) = (yield(); m.data[i, j] = v)

@testset "Review 2026-09-07 fixes" begin
    MPI.Initialized() || MPI.Init()

    # ── R1: checkpoints written before the inner-core keys existed still load ──
    @testset "R1 legacy checkpoint without inner-core keys restores" begin
        # No released version ever wrote `magnetic_toroidal_ic`. Putting those
        # keys on the REQUIRED list rejects every existing checkpoint of a
        # conducting-inner-core run, with a remedy ("disable the missing field
        # families") the user cannot follow without changing the physics.
        source = _review0907_state(magnetic_inner_bc = :conducting_inner_core)
        fill!(parent(source.fields.magnetic.toroidal.data_real), 3.0)
        legacy = Dict{String, Any}(
            "velocity_toroidal" => _review0907_pair(source.fields.velocity.toroidal),
            "velocity_poloidal" => _review0907_pair(source.fields.velocity.poloidal),
            "temperature" => copy(parent(source.fields.temperature.temperature.data)),
            "temperature_spectral" => _review0907_pair(source.fields.temperature.spectral),
            "magnetic_toroidal" => _review0907_pair(source.fields.magnetic.toroidal),
            "magnetic_poloidal" => _review0907_pair(source.fields.magnetic.poloidal),
        )

        target = _review0907_state(magnetic_inner_bc = :conducting_inner_core)
        fill!(parent(target.fields.magnetic.toroidal_ic.data_real), 9.0)

        GeoDynamo.restore_fields_from_restart!(target, legacy)
        @test all(==(3.0), parent(target.fields.magnetic.toroidal.data_real))
        # The file says nothing about the inner core, so the field the run
        # already built must be left exactly as it stands.
        @test all(==(9.0), parent(target.fields.magnetic.toroidal_ic.data_real))

        # A checkpoint that DOES carry them still restores them.
        fill!(parent(source.fields.magnetic.poloidal_ic.data_imag), -4.0)
        complete = copy(legacy)
        complete["magnetic_toroidal_ic"] =
            _review0907_pair(source.fields.magnetic.toroidal_ic)
        complete["magnetic_poloidal_ic"] =
            _review0907_pair(source.fields.magnetic.poloidal_ic)
        target2 = _review0907_state(magnetic_inner_bc = :conducting_inner_core)
        GeoDynamo.restore_fields_from_restart!(target2, complete)
        @test all(==(-4.0), parent(target2.fields.magnetic.poloidal_ic.data_imag))

        # A missing PRIMARY field is still a hard error.
        broken = copy(legacy)
        delete!(broken, "magnetic_poloidal")
        @test_throws ArgumentError GeoDynamo.restore_fields_from_restart!(
            _review0907_state(magnetic_inner_bc = :conducting_inner_core), broken)
    end

    # ── R2: a checkpoint writer creates its own directory ─────────────────────
    comm = GeoDynamo.output_comm()
    parallel_ok = let probe_err = GeoDynamo.parallel_netcdf_probe(comm)
        probe_err === nothing ||
            @warn "Parallel NetCDF unavailable; skipping checkpoint mkpath test" error = probe_err
        probe_err === nothing
    end

    if parallel_ok
        @testset "R2 write_restart! creates a missing output directory" begin
            # `_ensure_output_directory_collectively!` was added in this branch and
            # wired into `write_fields!` only. A CheckpointWriter routinely points
            # at a directory of its own that does not exist yet; without the
            # mkpath the parallel create fails with "Permission denied" and the
            # step's work is lost.
            lmax = mmax = 3
            nr = 8
            cfg = GeoDynamo.create_shtnskit_config(
                lmax = lmax, mmax = mmax, nlat = 10, nlon = 16, nr = nr)
            dom = GeoDynamo.create_radial_domain(nr)
            radial_nodes = Float64.(dom.r[1:nr, 4])
            nlm = cfg.nlm

            fields = Dict{String, Any}(
                "temperature" => randn(cfg.nlat, cfg.nlon, nr),
                "magnetic_toroidal" =>
                    Dict("real" => randn(nlm, nr), "imag" => randn(nlm, nr)),
            )

            base = GeoDynamo.default_config(Float64)
            fresh_dir = joinpath(mktempdir(), "checkpoints", "run_a")
            @test !isdir(fresh_dir)
            config = GeoDynamo.OutputConfig(
                base.output_space, fresh_dir, base.filename_prefix,
                base.include_metadata, base.include_grid, base.include_diagnostics,
                Float64, base.spectral_lmax_output, true,
                base.output_interval, base.restart_interval,
                base.max_output_time, base.time_tolerance)

            tracker = GeoDynamo.create_time_tracker(config, 0.0)
            metadata = Dict{String, Any}("current_time" => 1.0, "current_step" => 7)

            GeoDynamo.write_restart!(fields, tracker, metadata, config, nothing;
                shtns_config = cfg, geometry = :shell,
                radius_ratio = 0.35, radial_grid = radial_nodes)

            @test isdir(fresh_dir)
            @test length(GeoDynamo.find_restart_files(fresh_dir, 0.0)) == 1
        end
    end

    # ── R3: the boundary base cache is mutated under its lock ────────────────
    @testset "R3 reset_boundary_to_base! holds the lock while it writes" begin
        # `BoundaryValueBase` is mutable here and `reset_boundary_to_base!` now
        # assigns `entry.snapshot`/`entry.applied` and writes `snapshot[i]`
        # element-wise. `mark_boundary_applied!` mutates the same fields under
        # `_BOUNDARY_VALUE_BASE_LOCK`, and the GC finalizer can `delete!` the
        # entry at any safepoint. A reader that lands in the unlocked window
        # sees `applied === nothing`, skips the rebase and adopts the CORRECTED
        # array as the new base — the compounding this mechanism exists to stop.
        topo0907.clear_boundary_value_base_cache!()
        bv = YieldMatrix(zeros(2, 3))

        topo0907.reset_boundary_to_base!(bv)      # establish the entry
        bv[1, 1] = 1.0                            # a "correction"
        topo0907.mark_boundary_applied!(bv)       # entry.applied = corrected array

        finished = Ref(false)
        acquired_mid_flight = Ref(false)
        resetter = @async begin
            topo0907.reset_boundary_to_base!(bv)
            finished[] = true
        end
        observer = @async while !finished[]
            if trylock(topo0907._BOUNDARY_VALUE_BASE_LOCK)
                acquired_mid_flight[] = true
                unlock(topo0907._BOUNDARY_VALUE_BASE_LOCK)
            end
            yield()
        end
        wait(resetter)
        wait(observer)

        # No other task may hold the cache lock at any point between the call
        # starting and returning: every mutation belongs inside it.
        @test acquired_mid_flight[] === false
        # The rollback itself still works.
        @test bv[1, 1] == 0.0
        topo0907.clear_boundary_value_base_cache!()
    end
end
