function initialize_magnetic_field!(state::SolverState{T, <:AbstractArchitecture}) where {T}
    magnetic = state.fields.magnetic
    magnetic === nothing && return state

    fill!(parent(magnetic.toroidal.data_real), zero(T))
    fill!(parent(magnetic.toroidal.data_imag), zero(T))
    fill!(parent(magnetic.poloidal.data_real), zero(T))
    fill!(parent(magnetic.poloidal.data_imag), zero(T))
    fill!(parent(magnetic.toroidal_ic.data_real), zero(T))
    fill!(parent(magnetic.toroidal_ic.data_imag), zero(T))
    fill!(parent(magnetic.poloidal_ic.data_real), zero(T))
    fill!(parent(magnetic.poloidal_ic.data_imag), zero(T))

    domain = state.backend.outer_core_domain
    lm_range = local_spectral_mode_indices(magnetic.toroidal.config)
    r_range = local_range(magnetic.toroidal.config.pencils.spec, 3)

    pol_real = parent(magnetic.poloidal.data_real)
    pol_imag = parent(magnetic.poloidal.data_imag)
    tor_real = parent(magnetic.toroidal.data_real)
    tor_imag = parent(magnetic.toroidal.data_imag)

    @inbounds for lm_idx in lm_range
        lm_idx <= magnetic.toroidal.config.nlm || continue
        l = magnetic.toroidal.config.l_values[lm_idx]
        m = magnetic.toroidal.config.m_values[lm_idx]
        slot = local_spectral_storage_slot(magnetic.toroidal.config, lm_idx)

        for r_idx in r_range
            if l == 1 && m == 0
                r = domain.r[r_idx, 4]
                set_local_spectral_value!(pol_real, slot, r_idx, T(r^2 * (1.0 - r)))
            elseif 1 <= l <= 3
                amplitude = T(1e-4)
                set_local_spectral_value!(
                    tor_real,
                    slot,
                    r_idx,
                    amplitude * (rand(T) - T(0.5))
                )
                set_local_spectral_value!(
                    pol_real,
                    slot,
                    r_idx,
                    amplitude * (rand(T) - T(0.5))
                )
                if m > 0
                    set_local_spectral_value!(
                        tor_imag,
                        slot,
                        r_idx,
                        amplitude * (rand(T) - T(0.5))
                    )
                    set_local_spectral_value!(
                        pol_imag,
                        slot,
                        r_idx,
                        amplitude * (rand(T) - T(0.5))
                    )
                end
            end
        end
    end

    if state.parameters.magnetic_inner_bc === :conducting_inner_core
        fill!(magnetic.toroidal.bc_type_inner, Int(CONTINUITY_MAG))
        fill!(magnetic.poloidal.bc_type_inner, Int(CONTINUITY_MAG))
    end

    return state
end

function prepare_magnetic_fields!(magnetic_fields, outer_domain; skip_refresh::Bool = false)
    # Induction terms need the magnetic field in physical space. Reset the work
    # buffers the nonlinear pass writes (work_tor/pol, work/induction_physical).
    reset_magnetic_work_arrays!(magnetic_fields)
    # When skip_refresh, the B (magnetic.magnetic) and current (magnetic.current)
    # physical buffers were already synthesized this step by the up-front refresh
    # in compute_solver_nonlinear_terms! and survive reset (it does not touch them).
    # The induction pass reads only B physical + u (never current / J-spectral), and
    # the spectral magnetic state is unchanged since the up-front refresh, so the
    # re-synthesis here is redundant. See mpi-efficiency-audit.
    if !skip_refresh
        refresh_magnetic_physical_fields!(magnetic_fields, outer_domain)
        refresh_current_physical_fields!(magnetic_fields, outer_domain)
    end
    return magnetic_fields
end

function apply_magnetic_nonlinear_terms!(
        magnetic_fields,
        velocity_fields;
        rotation_rate::Float64
)
    if velocity_fields !== nothing
        apply_induction_nonlinear!(
            magnetic_fields,
            velocity_fields
        )
    end
    if rotation_rate != 0.0
        apply_inner_core_rotation!(magnetic_fields, rotation_rate)
    end
    return magnetic_fields
end

function _magnetic_toroidal_inner_bc_increment(
        magnetic::GeoDynamo.SHTnsMagneticFields{T},
) where {T}
    continuity_code = Int(GeoDynamo.CONTINUITY_MAG)
    any(==(continuity_code), magnetic.toroidal.bc_type_inner) || return nothing

    # CONTINUITY_MAG couples the toroidal inner-boundary RHS to the poloidal
    # nonlinear term. Build mode-indexed real/imag vectors before the radial
    # solve so all ranks feed the same boundary values to matrix rows.
    bc_real = zeros(T, magnetic.toroidal.nlm)
    bc_imag = zeros(T, magnetic.toroidal.nlm)
    prev_real = zeros(T, magnetic.toroidal.nlm)
    prev_imag = zeros(T, magnetic.toroidal.nlm)

    nl_pol_real = parent(magnetic.nl_poloidal.data_real)
    nl_pol_imag = parent(magnetic.nl_poloidal.data_imag)
    prev_nl_pol_real = parent(magnetic.prev_nl_poloidal.data_real)
    prev_nl_pol_imag = parent(magnetic.prev_nl_poloidal.data_imag)

    lm_range = local_spectral_mode_indices(magnetic.toroidal.config)
    @inbounds for lm_idx in lm_range
        magnetic.toroidal.bc_type_inner[lm_idx] == continuity_code || continue
        slot = local_spectral_storage_slot(magnetic.toroidal.config, lm_idx)
        bc_real[lm_idx] = -local_spectral_value(nl_pol_real, slot, 1)
        bc_imag[lm_idx] = -local_spectral_value(nl_pol_imag, slot, 1)
        prev_real[lm_idx] = -local_spectral_value(prev_nl_pol_real, slot, 1)
        prev_imag[lm_idx] = -local_spectral_value(prev_nl_pol_imag, slot, 1)
    end

    comm = mpi_comm()
    if mpi_comm_size(comm) > 1
        # Fuse the four boundary-vector reductions into ONE collective.
        nlm = magnetic.toroidal.nlm
        packed = Vector{T}(undef, 4 * nlm)
        @inbounds copyto!(packed, 1, bc_real, 1, nlm)
        @inbounds copyto!(packed, nlm + 1, bc_imag, 1, nlm)
        @inbounds copyto!(packed, 2 * nlm + 1, prev_real, 1, nlm)
        @inbounds copyto!(packed, 3 * nlm + 1, prev_imag, 1, nlm)
        allreduce_sum_in_place!(packed, comm)
        @inbounds copyto!(bc_real, 1, packed, 1, nlm)
        @inbounds copyto!(bc_imag, 1, packed, nlm + 1, nlm)
        @inbounds copyto!(prev_real, 1, packed, 2 * nlm + 1, nlm)
        @inbounds copyto!(prev_imag, 1, packed, 3 * nlm + 1, nlm)
    end

    return (bc_real, prev_real, bc_imag, prev_imag)
end

"""
    _magnetic_conducting_history_flux(magnetic, ic_spec, adm) -> (φ0_real, φ0_imag)

Build the conducting-inner-core ICB history flux `φ0` for every local magnetic
mode `(l,m)` of `ic_spec` (the inner-core scalar `toroidal_ic` or `poloidal_ic`), using its OLD
radial profile over inner-core indices `1..Nic`.

`φ0_l = inner_core_history_flux(adm, l, S_ic_old)` is the ICB radial-derivative
contribution from the inner-core CNAB2 history with a zero ICB value; paired with
the admittance `α_l` baked into the outer-core Robin inner row it enforces
derivative continuity across the ICB. Returns mode-indexed real/imag vectors
(length `nlm`) that are MPI all-reduced so every rank feeds the same boundary
values to the matrix rows. `l=0` is skipped (magnetic has no `l=0` mode).
"""
function _magnetic_conducting_history_flux(
        magnetic::SHTnsMagneticFields{T},
        ic_spec::SHTnsSpecField{T},
        adm::InnerCoreAdmittance{T}
) where {T}
    nlm = ic_spec.nlm
    φ0_real = zeros(T, nlm)
    φ0_imag = zeros(T, nlm)

    ic_real = parent(ic_spec.data_real)
    ic_imag = parent(ic_spec.data_imag)
    Nic = adm.Nic

    S_old_real = Vector{T}(undef, Nic)
    S_old_imag = Vector{T}(undef, Nic)

    lm_range = local_spectral_mode_indices(ic_spec.config)
    @inbounds for lm_idx in lm_range
        slot = local_spectral_storage_slot(ic_spec.config, lm_idx)
        slot === nothing && continue
        l = ic_spec.config.l_values[lm_idx]
        l == 0 && continue
        haskey(adm.lookup, l) || continue

        for ir in 1:Nic
            S_old_real[ir] = local_spectral_value(ic_real, slot, ir)
            S_old_imag[ir] = local_spectral_value(ic_imag, slot, ir)
        end
        φ0_real[lm_idx] = inner_core_history_flux(adm, l, S_old_real)
        φ0_imag[lm_idx] = inner_core_history_flux(adm, l, S_old_imag)
    end

    comm = mpi_comm()
    if mpi_comm_size(comm) > 1
        # Fuse the real+imag reductions into ONE collective (this fires for both
        # toroidal and poloidal every step under a conducting inner core).
        packed = Vector{T}(undef, 2 * nlm)
        @inbounds copyto!(packed, 1, φ0_real, 1, nlm)
        @inbounds copyto!(packed, nlm + 1, φ0_imag, 1, nlm)
        allreduce_sum_in_place!(packed, comm)
        @inbounds copyto!(φ0_real, 1, packed, 1, nlm)
        @inbounds copyto!(φ0_imag, 1, packed, nlm + 1, nlm)
    end

    return φ0_real, φ0_imag
end

"""
    _magnetic_conducting_reconstruct!(oc_spec, ic_spec, adm)

After the outer-core solve, reconstruct the inner-core radial profile for every
local magnetic mode and write it into `ic_spec` (`toroidal_ic` or `poloidal_ic`).

For each mode the ICB value `g` is the outer-core solution `oc_spec` at radial
index 1 (the outer-core inner point coincides with the ICB), and
`S_ic_new = reconstruct_inner_core(adm, l, g, S_ic_old)` is solved with regularity
`S(0)=0` and ICB Dirichlet value `S(ri)=g`. Value continuity across the ICB holds
by construction (`g` is shared); the result is written over inner-core radial
indices `1..Nic`. `l=0` is skipped.
"""
function _magnetic_conducting_reconstruct!(
        oc_spec::SHTnsSpecField{T},
        ic_spec::SHTnsSpecField{T},
        adm::InnerCoreAdmittance{T}
) where {T}
    oc_real = parent(oc_spec.data_real)
    oc_imag = parent(oc_spec.data_imag)
    ic_real = parent(ic_spec.data_real)
    ic_imag = parent(ic_spec.data_imag)
    Nic = adm.Nic

    S_old_real = Vector{T}(undef, Nic)
    S_old_imag = Vector{T}(undef, Nic)

    lm_range = local_spectral_mode_indices(ic_spec.config)
    @inbounds for lm_idx in lm_range
        slot = local_spectral_storage_slot(ic_spec.config, lm_idx)
        slot === nothing && continue
        l = ic_spec.config.l_values[lm_idx]
        l == 0 && continue
        haskey(adm.lookup, l) || continue

        for ir in 1:Nic
            S_old_real[ir] = local_spectral_value(ic_real, slot, ir)
            S_old_imag[ir] = local_spectral_value(ic_imag, slot, ir)
        end
        g_real = local_spectral_value(oc_real, slot, 1)
        g_imag = local_spectral_value(oc_imag, slot, 1)

        S_new_real = reconstruct_inner_core(adm, l, g_real, S_old_real)
        S_new_imag = reconstruct_inner_core(adm, l, g_imag, S_old_imag)
        for ir in 1:Nic
            set_local_spectral_value!(ic_real, slot, ir, S_new_real[ir])
            set_local_spectral_value!(ic_imag, slot, ir, S_new_imag[ir])
        end
    end

    return ic_spec
end

function apply_magnetic_toroidal_implicit_update!(state::SolverState{
        T, <:AbstractArchitecture}) where {T}
    magnetic = state.fields.magnetic
    magnetic === nothing && return state

    runtime = state.runtime
    timestepper = state.parameters.timestepper
    dt = state.parameters.timestep

    # Conducting inner core (CNAB2): the outer-core matrices already carry the
    # Robin inner row (∂/∂r − α_l) S = φ0, so the inner boundary RHS is the
    # inner-core CNAB2 history flux φ0 — NOT the old CONTINUITY_MAG -nl_pol
    # coupling. We supersede `_magnetic_toroidal_inner_bc_increment` here.
    adm_set = state.magnetic_ic_admittance
    if adm_set !== nothing && timestepper isa CNAB2
        adm_tor = adm_set.tor::InnerCoreAdmittance{T}
        φ0_real,
        φ0_imag = _magnetic_conducting_history_flux(magnetic, magnetic.toroidal_ic, adm_tor)
        matrices = state.implicit_matrices[:magnetic_tor]
        radial_work = get_radial_work!(
            state.timestep_caches,
            :magnetic_toroidal,
            matrices.system_matrices[1].size
        )
        solver_build_rhs_cnab2!(
            magnetic.work_tor,
            magnetic.toroidal,
            magnetic.nl_toroidal,
            magnetic.prev_nl_toroidal,
            dt,
            matrices;
            work = radial_work
        )
        solver_solve_magnetic_implicit_step!(
            magnetic.toroidal,
            magnetic.work_tor,
            matrices,
            :toroidal;
            mag_bc_inner = φ0_real,
            mag_bc_inner_imag = φ0_imag,
            _topo_mag_bc(magnetic.toroidal)...,
            work = radial_work
        )
        # Reconstruct the inner-core profile from the new ICB value (g = toroidal[ICB]).
        _magnetic_conducting_reconstruct!(magnetic.toroidal, magnetic.toroidal_ic, adm_tor)
        return state
    end

    inner_bc = _magnetic_toroidal_inner_bc_increment(magnetic)

    if timestepper isa CNAB2
        matrices = state.implicit_matrices[:magnetic_tor]
        radial_work = get_radial_work!(
            state.timestep_caches,
            :magnetic_toroidal,
            matrices.system_matrices[1].size
        )
        solver_build_rhs_cnab2!(
            magnetic.work_tor,
            magnetic.toroidal,
            magnetic.nl_toroidal,
            magnetic.prev_nl_toroidal,
            dt,
            matrices;
            work = radial_work
        )
        solver_solve_magnetic_implicit_step!(
            magnetic.toroidal,
            magnetic.work_tor,
            matrices,
            :toroidal;
            mag_bc_inner = inner_bc === nothing ? nothing : inner_bc[1],
            prev_bc_inner = inner_bc === nothing ? nothing : inner_bc[2],
            mag_bc_inner_imag = inner_bc === nothing ? nothing : inner_bc[3],
            prev_bc_inner_imag = inner_bc === nothing ? nothing : inner_bc[4],
            _topo_mag_bc(magnetic.toroidal)...,
            work = radial_work
        )
    elseif timestepper isa ExponentialAdamsBashforth2
        if inner_bc !== nothing
            throw(ArgumentError("CONTINUITY_MAG toroidal magnetic inner-boundary increments are not implemented for ExponentialAdamsBashforth2() timestepping"))
        end
        alu_map = (state.timestep_caches.etd_magnetic_toroidal::EAB2CacheEntry{T}).map
        radial_work = get_radial_work!(
            state.timestep_caches,
            :magnetic_toroidal,
            runtime.outer_core_domain.N
        )
        bc_spec = build_solver_erk2_magnetic_tor_bc(T, runtime.outer_core_domain;
            inner_regularity = state.parameters.geometry === :ball)
        solver_eab2_update_krylov_cached!(
            magnetic.toroidal,
            magnetic.nl_toroidal,
            magnetic.prev_nl_toroidal,
            alu_map,
            runtime.outer_core_domain,
            1.0,
            runtime.shtns_config,
            dt;
            m = _timestepper_krylov_dimension(timestepper, state.parameters),
            tol = _timestepper_krylov_tolerance(timestepper, state.parameters),
            bc_spec = bc_spec,
            krylov_work = radial_work
        )
    else
        matrices = state.implicit_matrices[:magnetic_tor]
        radial_work = get_radial_work!(
            state.timestep_caches,
            :magnetic_toroidal,
            matrices.system_matrices[1].size
        )
        solver_solve_magnetic_implicit_step!(
            magnetic.toroidal,
            magnetic.nl_toroidal,
            matrices,
            :toroidal;
            mag_bc_inner = inner_bc === nothing ? nothing : inner_bc[1],
            prev_bc_inner = inner_bc === nothing ? nothing : inner_bc[2],
            mag_bc_inner_imag = inner_bc === nothing ? nothing : inner_bc[3],
            prev_bc_inner_imag = inner_bc === nothing ? nothing : inner_bc[4],
            _topo_mag_bc(magnetic.toroidal)...,
            work = radial_work
        )
    end

    return state
end

function apply_magnetic_poloidal_implicit_update!(state::SolverState{
        T, <:AbstractArchitecture}) where {T}
    magnetic = state.fields.magnetic
    magnetic === nothing && return state

    runtime = state.runtime
    timestepper = state.parameters.timestepper
    dt = state.parameters.timestep

    # Conducting inner core (CNAB2): same ICB coupling as the toroidal branch,
    # applied to the poloidal scalar with its own admittance (Robin inner row +
    # φ0 history flux + inner-core reconstruction).
    adm_set = state.magnetic_ic_admittance
    if adm_set !== nothing && timestepper isa CNAB2
        adm_pol = adm_set.pol::InnerCoreAdmittance{T}
        φ0_real,
        φ0_imag = _magnetic_conducting_history_flux(magnetic, magnetic.poloidal_ic, adm_pol)
        matrices = state.implicit_matrices[:magnetic_pol]
        radial_work = get_radial_work!(
            state.timestep_caches,
            :magnetic_poloidal,
            matrices.system_matrices[1].size
        )
        solver_build_rhs_cnab2!(
            magnetic.work_pol,
            magnetic.poloidal,
            magnetic.nl_poloidal,
            magnetic.prev_nl_poloidal,
            dt,
            matrices;
            work = radial_work
        )
        solver_solve_magnetic_implicit_step!(
            magnetic.poloidal,
            magnetic.work_pol,
            matrices,
            :poloidal;
            mag_bc_inner = φ0_real,
            mag_bc_inner_imag = φ0_imag,
            _topo_mag_bc(magnetic.poloidal)...,
            work = radial_work
        )
        _magnetic_conducting_reconstruct!(magnetic.poloidal, magnetic.poloidal_ic, adm_pol)
        return state
    end

    if timestepper isa CNAB2
        matrices = state.implicit_matrices[:magnetic_pol]
        radial_work = get_radial_work!(
            state.timestep_caches,
            :magnetic_poloidal,
            matrices.system_matrices[1].size
        )
        solver_build_rhs_cnab2!(
            magnetic.work_pol,
            magnetic.poloidal,
            magnetic.nl_poloidal,
            magnetic.prev_nl_poloidal,
            dt,
            matrices;
            work = radial_work
        )
        solver_solve_magnetic_implicit_step!(
            magnetic.poloidal,
            magnetic.work_pol,
            matrices,
            :poloidal;
            _topo_mag_bc(magnetic.poloidal)...,
            work = radial_work
        )
    elseif timestepper isa ExponentialAdamsBashforth2
        alu_map = (state.timestep_caches.etd_magnetic_poloidal::EAB2CacheEntry{T}).map
        radial_work = get_radial_work!(
            state.timestep_caches,
            :magnetic_poloidal,
            runtime.outer_core_domain.N
        )
        bc_spec = build_solver_erk2_magnetic_pol_bc(T, runtime.outer_core_domain;
            inner_regularity = state.parameters.geometry === :ball)
        solver_eab2_update_krylov_cached!(
            magnetic.poloidal,
            magnetic.nl_poloidal,
            magnetic.prev_nl_poloidal,
            alu_map,
            runtime.outer_core_domain,
            1.0,
            runtime.shtns_config,
            dt;
            m = _timestepper_krylov_dimension(timestepper, state.parameters),
            tol = _timestepper_krylov_tolerance(timestepper, state.parameters),
            bc_spec = bc_spec,
            krylov_work = radial_work
        )
    else
        matrices = state.implicit_matrices[:magnetic_pol]
        radial_work = get_radial_work!(
            state.timestep_caches,
            :magnetic_poloidal,
            matrices.system_matrices[1].size
        )
        solver_solve_magnetic_implicit_step!(
            magnetic.poloidal,
            magnetic.nl_poloidal,
            matrices,
            :poloidal;
            _topo_mag_bc(magnetic.poloidal)...,
            work = radial_work
        )
    end

    return state
end

function queue_magnetic_implicit_updates!(
        operations::Vector{Function},
        state::SolverState{T, <:AbstractArchitecture}
) where {T}
    state.fields.magnetic === nothing && return operations
    push!(operations, () -> apply_magnetic_toroidal_implicit_update!(state))
    push!(operations, () -> apply_magnetic_poloidal_implicit_update!(state))
    return operations
end
