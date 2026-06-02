# r×θ Step Equivalence Test — MHD (magnetic + composition)
#
# Gate: a full solver_step! on a 2D r×θ grid must produce an IDENTICAL global
# spectral state (temperature, composition, velocity tor/pol, magnetic tor/pol)
# to < 1e-10 absolute difference compared to the serial (1x1) reference, for:
#   1x1 (serial reference), 4x1 (theta-dist), 1x4 (r-dist), 2x2 (full r×θ)
#
# This test extends test/r_theta_equivalence.jl to include_magnetic=true and
# include_composition=true.  Both the magnetic and composition transforms use
# the same theta-subcomm / transpose path that was migrated in Phase 2; this
# test closes the gap left by the transport-only equivalence test (which only
# verified temperature + velocity).
#
# === Usage ===
#
# This file is the DRIVER.  Run it directly under mpiexec (or at np=1) with
# GEODYNAMO_PROC_GRID set:
#
#   # np=1 (serial reference):
#   julia --project=. test/r_theta_equivalence_mhd.jl
#   GEODYNAMO_PROC_GRID=1x1 julia --project=. test/r_theta_equivalence_mhd.jl
#
#   # np=4 (2x2 r×θ grid):
#   GEODYNAMO_PROC_GRID=2x2 mpiexec -n 4 julia --project=. test/r_theta_equivalence_mhd.jl
#
# The driver writes a binary snapshot to /tmp/rtheta_mhd_sig_<grid>.bin on rank 0.
# The shell script test/run_mpi_r_theta_equivalence_mhd.sh runs all four grids
# and does the cross-grid comparison.
#
# NOTE: This test is NOT wired into runtests.jl by default because it shells
# out to mpiexec — nesting MPI inside the np=1 suite would fail.  Use the
# shell runner instead.

using Test, GeoDynamo, MPI

MPI.Initialized() || MPI.Init()

const _RTHETA_MHD_FINALIZE = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

# ---------------------------------------------------------------------------
# Model parameters: small but non-trivial, magnetic + composition enabled
# ---------------------------------------------------------------------------

function _rtheta_mhd_params()
    GeoDynamo.SolverParameters(
        architecture        = :cpu,
        geometry            = :shell,
        nr                  = 8,
        nr_inner            = 2,
        lmax                = 4,
        mmax                = 4,
        nlat                = 10,
        nlon                = 16,
        Ra                  = 1e4,
        Ek                  = 1e-2,
        Pr                  = 1.0,
        Pm                  = 1.0,
        timestep            = 1e-4,
        start_time          = 0.0,
        end_time            = 1e-3,
        stop_iteration      = 10,
        include_magnetic    = true,
        include_composition = true,
        timestepper         = GeoDynamo.CNAB2(),
        topography_enabled  = false,
        stefan_enabled      = false,
    )
end

# ---------------------------------------------------------------------------
# Deterministic IC: values depend ONLY on global (lm_idx, r_idx), never rank.
# This makes the IC identical regardless of the MPI grid configuration.
#
# We override initialize_fields! by setting spectral arrays directly before
# marking is_initialized=true, so the default random initializers are never
# called.
# ---------------------------------------------------------------------------

function _rtheta_mhd_set_ic!(state, params)
    domain = state.backend.outer_core_domain

    η  = params.radius_ratio
    ri = η / (1.0 - η)
    ro = 1.0 / (1.0 - η)

    # ------------------------------------------------------------------
    # Temperature: conductive profile for l=0 + small deterministic l≥1
    # (identical to the non-MHD equivalence test)
    # ------------------------------------------------------------------
    begin
        temperature = state.fields.temperature
        cfg = temperature.config
        spec_real = parent(temperature.spectral.data_real)
        spec_imag = parent(temperature.spectral.data_imag)
        fill!(spec_real, zero(Float64))
        fill!(spec_imag, zero(Float64))

        lm_range = GeoDynamo.local_spectral_mode_indices(cfg)
        r_range  = GeoDynamo.range_local(cfg.pencils.spec, 3)

        for lm_idx in lm_range
            lm_idx <= cfg.nlm || continue
            l    = cfg.l_values[lm_idx]
            m    = cfg.m_values[lm_idx]
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_idx)
            slot === nothing && continue
            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                local_r > size(spec_real, 3) && continue
                r = domain.r[r_idx, 4]
                if l == 0 && m == 0
                    val = sqrt(4π) * (ri * ro / (ro - ri) * (1.0 / r - 1.0 / ro))
                    GeoDynamo.set_local_spectral_value!(spec_real, slot, local_r, val)
                elseif 1 <= l <= 4
                    amp   = 1e-3
                    val_r = amp * sinpi(Float64(0.3 * (lm_idx + r_idx * 7)))
                    val_i = m > 0 ? amp * cospi(Float64(0.3 * (lm_idx - r_idx * 5))) : 0.0
                    GeoDynamo.set_local_spectral_value!(spec_real, slot, local_r, val_r)
                    GeoDynamo.set_local_spectral_value!(spec_imag, slot, local_r, val_i)
                end
            end
        end
    end

    # ------------------------------------------------------------------
    # Composition: conductive profile for l=0 + small deterministic l≥1
    # Same pattern as temperature so both fields are non-trivial.
    # ------------------------------------------------------------------
    if state.fields.composition !== nothing
        comp = state.fields.composition
        cfg  = comp.config
        spec_real = parent(comp.spectral.data_real)
        spec_imag = parent(comp.spectral.data_imag)
        fill!(spec_real, zero(Float64))
        fill!(spec_imag, zero(Float64))

        lm_range = GeoDynamo.local_spectral_mode_indices(cfg)
        r_range  = GeoDynamo.range_local(cfg.pencils.spec, 3)

        for lm_idx in lm_range
            lm_idx <= cfg.nlm || continue
            l    = cfg.l_values[lm_idx]
            m    = cfg.m_values[lm_idx]
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_idx)
            slot === nothing && continue
            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                local_r > size(spec_real, 3) && continue
                r = domain.r[r_idx, 4]
                if l == 0 && m == 0
                    # Conductive-shell profile (matches default 0/0 BC wall values)
                    val = sqrt(4π) * (ri * ro / (ro - ri) * (1.0 / r - 1.0 / ro))
                    val *= 0.5   # scale so it differs from temperature
                    GeoDynamo.set_local_spectral_value!(spec_real, slot, local_r, val)
                elseif 1 <= l <= 4
                    amp   = 5e-4
                    # Use a different hash than temperature to avoid accidental cancellation
                    val_r = amp * sinpi(Float64(0.3 * (lm_idx * 3 + r_idx * 11)))
                    val_i = m > 0 ? amp * cospi(Float64(0.3 * (lm_idx * 2 - r_idx * 13))) : 0.0
                    GeoDynamo.set_local_spectral_value!(spec_real, slot, local_r, val_r)
                    GeoDynamo.set_local_spectral_value!(spec_imag, slot, local_r, val_i)
                end
            end
        end
    end

    # ------------------------------------------------------------------
    # Magnetic: deterministic l≥1 perturbations (NO rand()).
    # l=1,m=0 poloidal gets a dipole-ish r-profile; rest get small
    # deterministic tor+pol seeds so the induction term is non-trivial.
    # ------------------------------------------------------------------
    if state.fields.magnetic !== nothing
        mag = state.fields.magnetic
        cfg = mag.toroidal.config
        tor_real = parent(mag.toroidal.data_real)
        tor_imag = parent(mag.toroidal.data_imag)
        pol_real = parent(mag.poloidal.data_real)
        pol_imag = parent(mag.poloidal.data_imag)
        fill!(tor_real, zero(Float64))
        fill!(tor_imag, zero(Float64))
        fill!(pol_real, zero(Float64))
        fill!(pol_imag, zero(Float64))
        # Also zero inner-core fields
        fill!(parent(mag.toroidal_ic.data_real), zero(Float64))
        fill!(parent(mag.toroidal_ic.data_imag), zero(Float64))
        fill!(parent(mag.poloidal_ic.data_real), zero(Float64))
        fill!(parent(mag.poloidal_ic.data_imag), zero(Float64))

        lm_range = GeoDynamo.local_spectral_mode_indices(cfg)
        r_range  = GeoDynamo.range_local(cfg.pencils.spec, 3)

        for lm_idx in lm_range
            lm_idx <= cfg.nlm || continue
            l    = cfg.l_values[lm_idx]
            m    = cfg.m_values[lm_idx]
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_idx)
            slot === nothing && continue
            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                local_r > size(tor_real, 3) && continue
                r = domain.r[r_idx, 4]

                if l == 1 && m == 0
                    # Dipole seed: poloidal ∝ r²(1-r) — same as default init but deterministic
                    GeoDynamo.set_local_spectral_value!(pol_real, slot, local_r, r^2 * (1.0 - r))
                elseif 1 <= l <= 4
                    amp   = 1e-4
                    # Use a different hash than temperature/composition
                    val_tr = amp * sinpi(Float64(0.3 * (lm_idx * 5 + r_idx * 17)))
                    val_pr = amp * cospi(Float64(0.3 * (lm_idx * 7 + r_idx * 13)))
                    GeoDynamo.set_local_spectral_value!(tor_real, slot, local_r, val_tr)
                    GeoDynamo.set_local_spectral_value!(pol_real, slot, local_r, val_pr)
                    if m > 0
                        val_ti = amp * sinpi(Float64(0.3 * (lm_idx * 11 - r_idx * 19)))
                        val_pi = amp * cospi(Float64(0.3 * (lm_idx * 13 - r_idx * 7)))
                        GeoDynamo.set_local_spectral_value!(tor_imag, slot, local_r, val_ti)
                        GeoDynamo.set_local_spectral_value!(pol_imag, slot, local_r, val_pi)
                    end
                end
            end
        end
    end

    # ------------------------------------------------------------------
    # Velocity: deterministic small perturbation so Lorentz/advection coupling
    # is non-trivial (non-zero velocity exercises magnetic induction paths)
    # ------------------------------------------------------------------
    begin
        vel = state.fields.velocity
        cfg = vel.toroidal.config
        for (field, real_data, imag_data) in (
                (vel.toroidal, parent(vel.toroidal.data_real), parent(vel.toroidal.data_imag)),
                (vel.poloidal, parent(vel.poloidal.data_real), parent(vel.poloidal.data_imag)))
            fill!(real_data, zero(Float64))
            fill!(imag_data, zero(Float64))
        end

        lm_range = GeoDynamo.local_spectral_mode_indices(cfg)
        r_range  = GeoDynamo.range_local(cfg.pencils.spec, 3)

        tor_real = parent(vel.toroidal.data_real)
        tor_imag = parent(vel.toroidal.data_imag)
        pol_real = parent(vel.poloidal.data_real)
        pol_imag = parent(vel.poloidal.data_imag)

        for lm_idx in lm_range
            lm_idx <= cfg.nlm || continue
            l    = cfg.l_values[lm_idx]
            m    = cfg.m_values[lm_idx]
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_idx)
            slot === nothing && continue
            1 <= l <= 4 || continue
            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                local_r > size(tor_real, 3) && continue
                amp   = 1e-5
                val_tr = amp * sinpi(Float64(0.3 * (lm_idx * 2 + r_idx * 23)))
                val_pr = amp * cospi(Float64(0.3 * (lm_idx * 4 + r_idx * 29)))
                GeoDynamo.set_local_spectral_value!(tor_real, slot, local_r, val_tr)
                GeoDynamo.set_local_spectral_value!(pol_real, slot, local_r, val_pr)
                if m > 0
                    val_ti = amp * sinpi(Float64(0.3 * (lm_idx * 6 - r_idx * 31)))
                    val_pi = amp * cospi(Float64(0.3 * (lm_idx * 8 - r_idx * 37)))
                    GeoDynamo.set_local_spectral_value!(tor_imag, slot, local_r, val_ti)
                    GeoDynamo.set_local_spectral_value!(pol_imag, slot, local_r, val_pi)
                end
            end
        end
    end

    # Mark initialized so solver_step! does not overwrite our IC
    state.is_initialized = true
    return state
end

# ---------------------------------------------------------------------------
# Global spectral gather via MPI.Allreduce over COMM_WORLD.
# (Reused from r_theta_equivalence.jl — identical logic.)
#
# The spec pencil distributes (l,m) modes; r is LOCAL (1:nr on every rank).
# Each rank fills its owned modes in a global (nlm,nr) buffer; Allreduce(+)
# gives the complete global array on every rank.
# ---------------------------------------------------------------------------

function _rtheta_mhd_gather(spec_field, cfg, nr, comm)
    nlm       = cfg.nlm
    local_r   = zeros(Float64, nlm, nr)
    local_i   = zeros(Float64, nlm, nr)
    spec_real = parent(spec_field.data_real)
    spec_imag = parent(spec_field.data_imag)
    r_range   = GeoDynamo.range_local(cfg.pencils.spec, 3)

    for lm_idx in GeoDynamo.local_spectral_mode_indices(cfg)
        lm_idx <= nlm || continue
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_idx)
        slot === nothing && continue
        for r_idx in r_range
            lr = r_idx - first(r_range) + 1
            lr > size(spec_real, 3) && continue
            local_r[lm_idx, r_idx] = GeoDynamo.local_spectral_value(spec_real, slot, lr)
            local_i[lm_idx, r_idx] = GeoDynamo.local_spectral_value(spec_imag, slot, lr)
        end
    end

    MPI.Allreduce!(local_r, +, comm)
    MPI.Allreduce!(local_i, +, comm)
    return local_r, local_i
end

# ---------------------------------------------------------------------------
# Snapshot I/O (written on rank 0)
# Stores 6 field pairs (12 tensors):
#   [temp_r, temp_i, comp_r, comp_i, vel_tor_r, vel_tor_i,
#    vel_pol_r, vel_pol_i, mag_tor_r, mag_tor_i, mag_pol_r, mag_pol_i]
# ---------------------------------------------------------------------------

function _rtheta_mhd_snapshot_path(grid_tag::String)
    snap_dir = get(ENV, "RTHETA_TMPDIR", tempdir())
    joinpath(snap_dir, "rtheta_mhd_sig_$(grid_tag).bin")
end

function _rtheta_mhd_write_snapshot(path::String, nlm::Int, nr::Int, tensors)
    open(path, "w") do io
        write(io, Int64(nlm))
        write(io, Int64(nr))
        write(io, Int64(length(tensors)))
        for t in tensors; write(io, t); end
    end
end

function _rtheta_mhd_read_snapshot(path::String)
    open(path, "r") do io
        nlm      = read(io, Int64)
        nr       = read(io, Int64)
        ntensors = read(io, Int64)
        tensors  = [read!(io, Array{Float64}(undef, nlm, nr)) for _ in 1:ntensors]
        return nlm, nr, tensors
    end
end

# ---------------------------------------------------------------------------
# Main: build model, step, gather all fields, optionally write snapshot
# ---------------------------------------------------------------------------

function _rtheta_mhd_run(grid_tag::String; write_snapshot::Bool = true)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    params = _rtheta_mhd_params()
    state  = GeoDynamo.initialize_simulation(Float64, params)
    _rtheta_mhd_set_ic!(state, params)

    GeoDynamo.solver_step!(state)

    cfg = state.fields.temperature.config
    nr  = params.nr

    temp_r,    temp_i    = _rtheta_mhd_gather(state.fields.temperature.spectral,   cfg, nr, comm)
    comp_r,    comp_i    = _rtheta_mhd_gather(state.fields.composition.spectral,   cfg, nr, comm)
    vel_tor_r, vel_tor_i = _rtheta_mhd_gather(state.fields.velocity.toroidal,      cfg, nr, comm)
    vel_pol_r, vel_pol_i = _rtheta_mhd_gather(state.fields.velocity.poloidal,      cfg, nr, comm)
    mag_tor_r, mag_tor_i = _rtheta_mhd_gather(state.fields.magnetic.toroidal,      cfg, nr, comm)
    mag_pol_r, mag_pol_i = _rtheta_mhd_gather(state.fields.magnetic.poloidal,      cfg, nr, comm)

    if rank == 0
        all_data = vcat(
            vec(temp_r),    vec(temp_i),
            vec(comp_r),    vec(comp_i),
            vec(vel_tor_r), vec(vel_tor_i),
            vec(vel_pol_r), vec(vel_pol_i),
            vec(mag_tor_r), vec(mag_tor_i),
            vec(mag_pol_r), vec(mag_pol_i),
        )
        @test all(isfinite, all_data)

        if write_snapshot
            path = _rtheta_mhd_snapshot_path(grid_tag)
            _rtheta_mhd_write_snapshot(path, cfg.nlm, nr, [
                temp_r,    temp_i,
                comp_r,    comp_i,
                vel_tor_r, vel_tor_i,
                vel_pol_r, vel_pol_i,
                mag_tor_r, mag_tor_i,
                mag_pol_r, mag_pol_i,
            ])
            println("[r×θ-mhd-equiv grid=$(grid_tag)] snapshot written to $path; " *
                    "norm=$(round(sqrt(sum(abs2,all_data)); sigdigits=7))")
        end
    end

    MPI.Barrier(comm)
    return (temp_r, temp_i, comp_r, comp_i,
            vel_tor_r, vel_tor_i, vel_pol_r, vel_pol_i,
            mag_tor_r, mag_tor_i, mag_pol_r, mag_pol_i)
end

# ---------------------------------------------------------------------------
# @testset entry point
# ---------------------------------------------------------------------------

@testset "r×θ MHD step equivalence (magnetic+composition) to 1D layout" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping r×θ MHD equivalence test"
    else
        comm     = MPI.COMM_WORLD
        rank     = MPI.Comm_rank(comm)
        nprocs   = MPI.Comm_size(comm)
        grid_tag = get(ENV, "GEODYNAMO_PROC_GRID", "$(nprocs)x1")

        @testset "solver_step! runs at grid=$grid_tag (MHD)" begin
            write_snapshot = haskey(ENV, "GEODYNAMO_PROC_GRID") || nprocs > 1
            tensors = _rtheta_mhd_run(grid_tag; write_snapshot = write_snapshot)

            if nprocs == 1
                temp_r = tensors[1]
                @test temp_r[1, 1] != 0.0   # l=0,m=0 temperature mode must be non-zero
                @test maximum(abs, temp_r) > 0.0
                # magnetic poloidal l=1,m=0 dipole seed must survive the step
                mag_pol_r = tensors[11]
                @test maximum(abs, mag_pol_r) > 0.0
            end
        end

        # ------------------------------------------------------------------
        # Cross-grid comparison (only when snapshots from ALL four grids exist)
        # ------------------------------------------------------------------
        @testset "cross-grid MHD comparison (if all snapshots present)" begin
            grids = ["1x1", "4x1", "1x4", "2x2"]
            paths = [_rtheta_mhd_snapshot_path(g) for g in grids]

            if all(isfile, paths) && rank == 0
                nlm_ref, nr_ref, refs = _rtheta_mhd_read_snapshot(paths[1])
                tensor_names = [
                    "temp_real",    "temp_imag",
                    "comp_real",    "comp_imag",
                    "vel_tor_real", "vel_tor_imag",
                    "vel_pol_real", "vel_pol_imag",
                    "mag_tor_real", "mag_tor_imag",
                    "mag_pol_real", "mag_pol_imag",
                ]

                for (grid, path) in zip(grids[2:end], paths[2:end])
                    nlm, nr, tensors = _rtheta_mhd_read_snapshot(path)
                    @test nlm == nlm_ref
                    @test nr  == nr_ref
                    for (name, ref_t, got_t) in zip(tensor_names, refs, tensors)
                        maxdiff = maximum(abs.(ref_t .- got_t))
                        @test maxdiff < 1e-10
                        maxdiff < 1e-10 ||
                            @warn "grid=$grid tensor=$name maxdiff=$maxdiff exceeds 1e-10"
                    end
                    println("[r×θ-mhd-equiv] $grid vs 1x1: max diff ≤ 1e-10")
                end
            else
                @test_skip "MHD cross-grid snapshots not yet available (run run_mpi_r_theta_equivalence_mhd.sh)"
            end
        end
    end
end

if _RTHETA_MHD_FINALIZE && MPI.Initialized() && !MPI.Finalized()
    MPI.Finalize()
end
