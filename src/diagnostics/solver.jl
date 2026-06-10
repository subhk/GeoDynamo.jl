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
potentials and evaluates ∇·u on the physical grid using a banded D1 radial
derivative and exact spectral angular derivatives (sphtor synthesis trick),
returning grid RMS (L2) and L∞ norms over the interior radial nodes.

Formula in spherical coordinates:
    ∇·u = (1/r²)∂_r(r²·u_r) + (1/(r·sinθ))[∂_θ(sinθ·u_θ) + ∂_φ(u_φ)]

Replaces a stub that hardcoded (0.0, 0.0).
Allocation-heavy (builds scratch fields per call) — diagnostic path only.

# TODO(stage2): MPI.Allreduce the norms for multi-rank.
"""
function compute_divergence_spectral(
        tor_spec::SpectralFieldType{T},
        pol_spec::SpectralFieldType{T},
        domain::RadialDomainType) where {T}
    cfg  = tor_spec.config
    nlat = cfg.nlat
    nlon = cfg.nlon

    # 1. Synthesize vector field from (toroidal, poloidal) potentials.
    V = create_shtns_vector_field(T, cfg, domain, (cfg.pencils.θ, cfg.pencils.φ, cfg.pencils.r))
    vector_spectral_to_physical!(tor_spec, pol_spec, V; domain = domain)

    # 2. Angular derivative harness: T=0, S=g_spec → sphtor synthesis.
    #    Returns (∂_θ g,  (1/sinθ)·∂_φ g) as physical fields on pencil_r.
    function _angular_derivs(g_phys)
        g_spec_tmp = create_shtns_spectral_field(T, cfg, domain, cfg.pencils.spec)
        scalar_physical_to_spectral!(g_phys, g_spec_tmp)
        zero_tor = create_shtns_spectral_field(T, cfg, domain, cfg.pencils.spec)
        W = create_shtns_vector_field(T, cfg, domain, (cfg.pencils.θ, cfg.pencils.φ, cfg.pencils.r))
        vector_spectral_to_physical!(zero_tor, g_spec_tmp, W; domain = nothing)
        return W.θ_component, W.φ_component
    end

    # 3. Radial derivative harness: banded D1 applied column-by-column.
    function _radial_deriv(g_phys)
        D1      = create_derivative_matrix(T, 1, domain)
        nr      = domain.N
        arr_in  = parent(g_phys.data)
        out     = create_shtns_physical_field(T, cfg, domain, cfg.pencils.r)
        arr_out = parent(out.data)
        prof    = Vector{T}(undef, nr)
        dprof   = Vector{T}(undef, nr)
        for j in 1:size(arr_in, 2), i in 1:size(arr_in, 1)
            for k in 1:nr; prof[k]    = arr_in[i, j, k]; end
            mul!(dprof, D1, prof)
            for k in 1:nr; arr_out[i, j, k] = dprof[k]; end
        end
        return out
    end

    r_range = range_local(cfg.pencils.r, 3)
    sinθ    = sin.(cfg.theta_grid)

    # 4. Build r²·u_r → differentiate radially → A = ∂_r(r²·u_r)
    r2ur = create_shtns_physical_field(T, cfg, domain, cfg.pencils.r)
    a    = parent(r2ur.data)
    ur   = parent(V.r_component.data)
    for k in axes(a, 3), j in 1:nlon, i in 1:nlat
        r          = domain.r[k + first(r_range) - 1, 4]
        a[i, j, k] = r^2 * ur[i, j, k]
    end
    d_r2ur = _radial_deriv(r2ur)

    # 5. Build sinθ·u_θ → angular derivative → B = ∂_θ(sinθ·u_θ)
    sut = create_shtns_physical_field(T, cfg, domain, cfg.pencils.r)
    b   = parent(sut.data)
    uθ  = parent(V.θ_component.data)
    for k in axes(b, 3), j in 1:nlon, i in 1:nlat
        b[i, j, k] = sinθ[i] * uθ[i, j, k]
    end
    dθ_sut, _ = _angular_derivs(sut)

    # 6. C = (1/sinθ)·∂_φ(u_φ) directly from angular derivative of u_φ
    _, dφ_uφ = _angular_derivs(V.φ_component)

    # 7. Accumulate L2 and L∞ over interior radial nodes (skip boundaries
    #    where D1 is less accurate).
    # TODO(stage2): MPI.Allreduce the norms for multi-rank
    A       = parent(d_r2ur.data)
    B       = parent(dθ_sut.data)
    C       = parent(dφ_uφ.data)
    n_int   = 0
    sum_sq  = zero(T)
    max_abs = zero(T)
    for k in axes(A, 3)
        r_glob = k + first(r_range) - 1
        (2 <= r_glob <= domain.N - 1) || continue
        n_int += 1
        r = domain.r[r_glob, 4]
        for j in 1:nlon, i in 1:nlat
            div_val = A[i, j, k] / r^2 + (B[i, j, k] / sinθ[i] + C[i, j, k]) / r
            sum_sq  += div_val^2
            abs_v    = abs(div_val)
            abs_v > max_abs && (max_abs = abs_v)
        end
    end

    l2   = n_int > 0 ? sqrt(sum_sq / (n_int * nlat * nlon)) : zero(T)
    linf = max_abs
    return (Float64(l2), Float64(linf))
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
