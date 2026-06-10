const SOLVER_MAX_TRACKER_HISTORY = 10_000
const SOLVER_DEFAULT_NAN_CONFIG = getproperty(GeoDynamo, :DEFAULT_NAN_CONFIG)

function create_solver_energy_tracker()
    SolverEnergyTracker(
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Int[],
        true)
end

function create_solver_solenoidal_monitor()
    SolverSolenoidalMonitor(
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Int[],
        true)
end

function field_energy(field_data::Array{T, 3}) where {T}
    local_energy = 0.5 * sum(abs2, field_data)
    if mpi_initialized()
        return allreduce_sum(local_energy, mpi_comm())
    end
    return local_energy
end

function vector_energy(
        v_r::Array{T, 3},
        v_theta::Array{T, 3},
        v_phi::Array{T, 3}) where {T}
    local_energy = 0.5 * (sum(abs2, v_r) + sum(abs2, v_theta) + sum(abs2, v_phi))
    if mpi_initialized()
        return allreduce_sum(local_energy, mpi_comm())
    end
    return local_energy
end

function trim_energy_tracker!(tracker::SolverEnergyTracker)
    n = length(tracker.total_energy)
    if n > SOLVER_MAX_TRACKER_HISTORY
        keep = n - SOLVER_MAX_TRACKER_HISTORY ÷ 2
        deleteat!(tracker.kinetic_energy, 1:keep)
        deleteat!(tracker.magnetic_energy, 1:keep)
        deleteat!(tracker.thermal_energy, 1:keep)
        deleteat!(tracker.compositional_energy, 1:keep)
        deleteat!(tracker.total_energy, 1:keep)
        deleteat!(tracker.timestamps, 1:keep)
    end
    return tracker
end

function compute_total_energy!(state::SolverState{T, <:AbstractArchitecture}) where {T}
    tracker = state.energy_tracker
    tracker.enable_tracking || return nothing

    velocity = state.fields.velocity
    temperature = state.fields.temperature
    magnetic = state.fields.magnetic
    composition = state.fields.composition
    domain = state.backend.outer_core_domain

    vector_spectral_to_physical!(velocity.toroidal, velocity.poloidal, velocity.velocity; domain = domain)
    if magnetic !== nothing
        vector_spectral_to_physical!(magnetic.toroidal, magnetic.poloidal, magnetic.magnetic; domain = domain)
    end

    scalar_spectral_to_physical!(temperature.spectral, temperature.temperature)

    if composition !== nothing
        scalar_spectral_to_physical!(composition.spectral, composition.composition)
    end

    kinetic_e = vector_energy(
        parent(velocity.velocity.r_component.data),
        parent(velocity.velocity.θ_component.data),
        parent(velocity.velocity.φ_component.data)
    )

    magnetic_e = 0.0
    if magnetic !== nothing
        magnetic_e = vector_energy(
            parent(magnetic.magnetic.r_component.data),
            parent(magnetic.magnetic.θ_component.data),
            parent(magnetic.magnetic.φ_component.data)
        )
    end

    thermal_e = field_energy(parent(temperature.temperature.data))

    compositional_e = 0.0
    if composition !== nothing
        compositional_e = field_energy(parent(composition.composition.data))
    end

    total_e = kinetic_e + magnetic_e + thermal_e + compositional_e

    if mpi_rank() == 0
        push!(tracker.kinetic_energy, kinetic_e)
        push!(tracker.magnetic_energy, magnetic_e)
        push!(tracker.thermal_energy, thermal_e)
        push!(tracker.compositional_energy, compositional_e)
        push!(tracker.total_energy, total_e)
        push!(tracker.timestamps, state.runtime.timestep_state.step)
        trim_energy_tracker!(tracker)
    end

    return nothing
end

function report_energy_conservation(
        state::SolverState,
        step::Int;
        interval::Int = 100)
    tracker = state.energy_tracker
    tracker.enable_tracking || return nothing

    n_samples = length(tracker.total_energy)
    n_samples < 2 && return nothing

    if step % interval == 0 && mpi_rank() == 0
        E0 = tracker.total_energy[1]
        En = tracker.total_energy[end]
        ΔE = En - E0
        rel_error = E0 != 0.0 ? abs(ΔE / E0) : 0.0

        KE = tracker.kinetic_energy[end]
        ME = tracker.magnetic_energy[end]
        TE = tracker.thermal_energy[end]
        CE = tracker.compositional_energy[end]

        pct(x) = En != 0.0 ? x / En * 100 : 0.0

        @info """
        ╔══════════════════════════════════════════════════════════╗
        ║      Solver Energy Conservation Report (Step $step)
        ╠══════════════════════════════════════════════════════════╣
        ║ Total Energy:        $En
        ║ Initial Energy:      $E0
        ║ Energy Drift (ΔE):   $ΔE
        ║ Relative Error:      $(rel_error * 100)%
        ║
        ║ Energy Breakdown:
        ║   Kinetic:           $KE  ($(pct(KE))%)
        ║   Magnetic:          $ME  ($(pct(ME))%)
        ║   Thermal:           $TE  ($(pct(TE))%)
        ║   Compositional:     $CE  ($(pct(CE))%)
        ╚══════════════════════════════════════════════════════════╝
        """

        if rel_error > 0.01
            @warn "Solver energy conservation error exceeds 1%! Current: $(rel_error * 100)%"
        end
    end

    return nothing
end
"""
    compute_divergence_spectral(tor_spec, pol_spec, domain) -> (l2, linf)

Real divergence diagnostic: synthesizes the vector field from its (T, P)
potentials, re-analyzes it into QST scalars (Q from the radial component,
S/T from the tangential sphtor analysis), and evaluates the per-mode spectral
divergence

    div_lm(r) = (1/r²)·∂_r(r²·Q_lm) − l(l+1)·S_lm/r

returning RMS (L2) and L∞ norms over modes and interior radial nodes. Going
through physical space and back makes this a genuine check of the transform
chain (a (T,P) pair synthesizes solenoidally by construction under the
Stage-2 convention, so any nonzero divergence here is a chain defect). The
spectral formula avoids the grid-space aliasing that scalar analysis of
tangential components (1/sinθ structure) suffers beyond lmax for m>0 modes.

Replaces a stub that hardcoded (0.0, 0.0).
Allocation-heavy (builds scratch fields per call) — diagnostic path only.

# TODO(stage2): MPI.Allreduce the norms for multi-rank.
"""
function compute_divergence_spectral(
        tor_spec::SpectralFieldType{T},
        pol_spec::SpectralFieldType{T},
        domain::RadialDomainType) where {T}
    cfg = tor_spec.config

    # 1. Synthesize the vector field from its potentials.
    V = create_shtns_vector_field(
        T, cfg, domain, (cfg.pencils.θ, cfg.pencils.φ, cfg.pencils.r))
    vector_spectral_to_physical!(tor_spec, pol_spec, V; domain = domain)

    # 2. Re-analyze into QST scalars (Stage-1 machinery).
    Q = create_shtns_spectral_field(T, cfg, domain, cfg.pencils.spec)
    S = create_shtns_spectral_field(T, cfg, domain, cfg.pencils.spec)
    T_ = create_shtns_spectral_field(T, cfg, domain, cfg.pencils.spec)
    force_physical_to_qst!(V, Q, S, T_)

    # 3. Per-mode spectral divergence over interior radial nodes.
    D1 = create_derivative_matrix(Float64, 1, domain)
    nr = domain.N
    r2q = Vector{Float64}(undef, nr)
    d = Vector{Float64}(undef, nr)
    sumsq = 0.0
    linf = 0.0
    count = 0
    for lm in 1:cfg.nlm
        slot = local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue
        l = cfg.l_values[lm]
        λ = l * (l + 1)
        for (qsrc, ssrc) in ((parent(Q.data_real), parent(S.data_real)),
            (parent(Q.data_imag), parent(S.data_imag)))
            for r_idx in 1:nr
                r = domain.r[r_idx, 4]
                r2q[r_idx] = r^2 * local_spectral_value(qsrc, slot, r_idx)
            end
            mul!(d, D1, r2q)
            for r_idx in 2:(nr - 1)
                r = domain.r[r_idx, 4]
                dv = d[r_idx] / r^2 -
                     λ * local_spectral_value(ssrc, slot, r_idx) / r
                sumsq += dv^2
                linf = max(linf, abs(dv))
                count += 1
            end
        end
    end
    l2 = count > 0 ? sqrt(sumsq / count) : 0.0
    return (l2, linf)
end

function trim_solenoidal_monitor!(monitor::SolverSolenoidalMonitor)
    n = length(monitor.velocity_div_l2)
    if n > SOLVER_MAX_TRACKER_HISTORY
        keep = n - SOLVER_MAX_TRACKER_HISTORY ÷ 2
        deleteat!(monitor.velocity_div_l2, 1:keep)
        deleteat!(monitor.velocity_div_linf, 1:keep)
        deleteat!(monitor.magnetic_div_l2, 1:keep)
        deleteat!(monitor.magnetic_div_linf, 1:keep)
        deleteat!(monitor.timestamps, 1:keep)
    end
    return monitor
end

function check_solenoidal_constraint!(state::SolverState)
    monitor = state.solenoidal_monitor
    monitor.enable_monitoring || return nothing

    vel_l2,
    vel_linf = compute_divergence_spectral(
        state.fields.velocity.toroidal,
        state.fields.velocity.poloidal,
        state.backend.outer_core_domain
    )

    mag_l2, mag_linf = 0.0, 0.0
    if state.fields.magnetic !== nothing
        mag_l2,
        mag_linf = compute_divergence_spectral(
            state.fields.magnetic.toroidal,
            state.fields.magnetic.poloidal,
            state.backend.outer_core_domain
        )
    end

    push!(monitor.velocity_div_l2, vel_l2)
    push!(monitor.velocity_div_linf, vel_linf)
    push!(monitor.magnetic_div_l2, mag_l2)
    push!(monitor.magnetic_div_linf, mag_linf)
    push!(monitor.timestamps, state.runtime.timestep_state.step)
    trim_solenoidal_monitor!(monitor)

    return nothing
end

function report_solenoidal_constraint(
        state::SolverState,
        step::Int;
        interval::Int = 100
)
    monitor = state.solenoidal_monitor
    monitor.enable_monitoring || return nothing

    n_samples = length(monitor.velocity_div_l2)
    n_samples < 1 && return nothing

    if step % interval == 0 && mpi_rank() == 0
        vel_l2 = monitor.velocity_div_l2[end]
        vel_linf = monitor.velocity_div_linf[end]
        mag_l2 = monitor.magnetic_div_l2[end]
        mag_linf = monitor.magnetic_div_linf[end]

        @info """
        ╔══════════════════════════════════════════════════════════╗
        ║   Solver Solenoidal Constraint Report (Step $step)
        ╠══════════════════════════════════════════════════════════╣
        ║ Velocity Field (∇·v should be 0):
        ║   L2 norm:           $vel_l2
        ║   L∞ norm:           $vel_linf
        ║
        ║ Magnetic Field (∇·B should be 0):
        ║   L2 norm:           $mag_l2
        ║   L∞ norm:           $mag_linf
        ╚══════════════════════════════════════════════════════════╝
        """

        if vel_linf > 1e-10
            @warn "Solver velocity solenoidal constraint may be violated: L∞ = $vel_linf"
        end
        if mag_linf > 1e-10
            @warn "Solver magnetic solenoidal constraint may be violated: L∞ = $mag_linf"
        end
    end

    return nothing
end

function check_runtime_for_nan(
        state::SolverState;
        config::NaNConfigType = SOLVER_DEFAULT_NAN_CONFIG
)
    step = state.runtime.timestep_state.step
    if !config.enabled || step % config.check_every_n_steps != 0
        return false
    end

    any_issue = false

    has_nan, has_inf,
    _,
    _ = check_spectral_field_for_nan(
        state.fields.velocity.toroidal,
        "velocity_toroidal",
        config,
        step
    )
    any_issue |= (has_nan || has_inf)

    has_nan, has_inf,
    _,
    _ = check_spectral_field_for_nan(
        state.fields.velocity.poloidal,
        "velocity_poloidal",
        config,
        step
    )
    any_issue |= (has_nan || has_inf)

    if state.fields.magnetic !== nothing
        has_nan, has_inf,
        _,
        _ = check_spectral_field_for_nan(
            state.fields.magnetic.toroidal,
            "magnetic_toroidal",
            config,
            step
        )
        any_issue |= (has_nan || has_inf)

        has_nan, has_inf,
        _,
        _ = check_spectral_field_for_nan(
            state.fields.magnetic.poloidal,
            "magnetic_poloidal",
            config,
            step
        )
        any_issue |= (has_nan || has_inf)
    end

    has_nan, has_inf,
    _,
    _ = check_spectral_field_for_nan(
        state.fields.temperature.spectral,
        "temperature",
        config,
        step
    )
    any_issue |= (has_nan || has_inf)

    if state.fields.composition !== nothing
        has_nan, has_inf,
        _,
        _ = check_spectral_field_for_nan(
            state.fields.composition.spectral,
            "composition",
            config,
            step
        )
        any_issue |= (has_nan || has_inf)
    end

    comm = mpi_comm()
    if comm !== nothing && mpi_comm_size(comm) > 1
        local_flag = any_issue ? 1 : 0
        global_flag = allreduce_max(local_flag, comm)
        any_issue = global_flag > 0
    end

    if any_issue && config.abort_on_nan
        error(
            "NaN or Inf detected in solver fields at step $step. " *
            "Check timestep size, boundary values, initial conditions, and physical parameters.",
        )
    end

    return any_issue
end

function run_diagnostics!(state::SolverState; interval::Int = 100)
    step = state.runtime.timestep_state.step
    if step % interval == 0
        compute_total_energy!(state)
        report_energy_conservation(state, step; interval)
        check_solenoidal_constraint!(state)
        report_solenoidal_constraint(state, step; interval)
    end
    return nothing
end
