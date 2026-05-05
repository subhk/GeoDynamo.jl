function initialize_magnetic_field!(state::SolverState{T,<:AbstractArchitecture}) where T
    magnetic = state.fields.magnetic
    magnetic === nothing && return state

    fill!(parent(magnetic.𝒯.data_real), zero(T))
    fill!(parent(magnetic.𝒯.data_imag), zero(T))
    fill!(parent(magnetic.𝒫.data_real), zero(T))
    fill!(parent(magnetic.𝒫.data_imag), zero(T))
    fill!(parent(magnetic.𝒯ⁱᶜ.data_real), zero(T))
    fill!(parent(magnetic.𝒯ⁱᶜ.data_imag), zero(T))
    fill!(parent(magnetic.𝒫ⁱᶜ.data_real), zero(T))
    fill!(parent(magnetic.𝒫ⁱᶜ.data_imag), zero(T))

    domain = state.backend.outer_core_domain
    lm_range = local_spectral_mode_indices(magnetic.𝒯.config)
    r_range  = local_range(magnetic.𝒯.config.pencils.spec, 3)

    pol_real = parent(magnetic.𝒫.data_real)
    pol_imag = parent(magnetic.𝒫.data_imag)
    tor_real = parent(magnetic.𝒯.data_real)
    tor_imag = parent(magnetic.𝒯.data_imag)

    @inbounds for lm_idx in lm_range
        lm_idx <= magnetic.𝒯.config.nlm || continue
        l = magnetic.𝒯.config.l_values[lm_idx]
        m = magnetic.𝒯.config.m_values[lm_idx]
        slot = local_spectral_storage_slot(magnetic.𝒯.config, lm_idx)

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
                    amplitude * (rand(T) - T(0.5)),
                )
                set_local_spectral_value!(
                    pol_real,
                    slot,
                    r_idx,
                    amplitude * (rand(T) - T(0.5)),
                )
                if m > 0
                    set_local_spectral_value!(
                        tor_imag,
                        slot,
                        r_idx,
                        amplitude * (rand(T) - T(0.5)),
                    )
                    set_local_spectral_value!(
                        pol_imag,
                        slot,
                        r_idx,
                        amplitude * (rand(T) - T(0.5)),
                    )
                end
            end
        end
    end

    return state
end

function solver_prepare_magnetic_fields!(magnetic_fields, outer_domain)
    # Induction terms need magnetic field and current in physical space. Refresh
    # both from the spectral toroidal/poloidal representation before use.
    solver_reset_magnetic_work_arrays!(magnetic_fields)
    solver_refresh_magnetic_physical_fields!(magnetic_fields, outer_domain)
    solver_refresh_current_physical_fields!(magnetic_fields, outer_domain)
    return magnetic_fields
end

function solver_apply_magnetic_nonlinear_terms!(
    magnetic_fields,
    velocity_fields;
    geometry::Symbol,
    rotation_rate::Float64,
)
    if velocity_fields !== nothing
        solver_apply_induction_nonlinear!(
            magnetic_fields,
            velocity_fields;
            geometry,
        )
    end
    if rotation_rate != 0.0
        solver_apply_inner_core_rotation!(magnetic_fields, rotation_rate)
    end
    return magnetic_fields
end

function _magnetic_toroidal_inner_bc_increment(
    magnetic::GeoDynamo.SHTnsMagneticFields{T},
) where T
    continuity_code = Int(GeoDynamo.CONTINUITY_MAG)
    any(==(continuity_code), magnetic.𝒯.bc_type_inner) || return nothing

    # CONTINUITY_MAG couples the toroidal inner-boundary RHS to the poloidal
    # nonlinear term. Build mode-indexed real/imag vectors before the radial
    # solve so all ranks feed the same boundary values to matrix rows.
    bc_real = zeros(T, magnetic.𝒯.nlm)
    bc_imag = zeros(T, magnetic.𝒯.nlm)
    prev_real = zeros(T, magnetic.𝒯.nlm)
    prev_imag = zeros(T, magnetic.𝒯.nlm)

    nl_pol_real = parent(magnetic.nlᴾ.data_real)
    nl_pol_imag = parent(magnetic.nlᴾ.data_imag)
    prev_nl_pol_real = parent(magnetic.prev_nlᴾ.data_real)
    prev_nl_pol_imag = parent(magnetic.prev_nlᴾ.data_imag)

    lm_range = local_spectral_mode_indices(magnetic.𝒯.config)
    @inbounds for lm_idx in lm_range
        magnetic.𝒯.bc_type_inner[lm_idx] == continuity_code || continue
        slot = local_spectral_storage_slot(magnetic.𝒯.config, lm_idx)
        bc_real[lm_idx] = -local_spectral_value(nl_pol_real, slot, 1)
        bc_imag[lm_idx] = -local_spectral_value(nl_pol_imag, slot, 1)
        prev_real[lm_idx] = -local_spectral_value(prev_nl_pol_real, slot, 1)
        prev_imag[lm_idx] = -local_spectral_value(prev_nl_pol_imag, slot, 1)
    end

    comm = mpi_comm()
    if mpi_comm_size(comm) > 1
        allreduce_sum_in_place!(bc_real, comm)
        allreduce_sum_in_place!(bc_imag, comm)
        allreduce_sum_in_place!(prev_real, comm)
        allreduce_sum_in_place!(prev_imag, comm)
    end

    return (bc_real, prev_real, bc_imag, prev_imag)
end

function apply_magnetic_toroidal_implicit_update!(state::SolverState{T,<:AbstractArchitecture}) where T
    magnetic = state.fields.magnetic
    magnetic === nothing && return state

    runtime = state.runtime
    timestepper = state.parameters.timestepper
    dt = state.parameters.timestep
    inner_bc = _magnetic_toroidal_inner_bc_increment(magnetic)

    if timestepper isa CNAB2
        matrices = state.implicit_matrices[:magnetic_tor]
        radial_work = solver_get_radial_work!(
            state.timestep_caches,
            :magnetic_toroidal,
            matrices.system_matrices[1].size,
        )
        solver_build_rhs_cnab2!(
            magnetic.work_tor,
            magnetic.𝒯,
            magnetic.nlᵀ,
            magnetic.prev_nlᵀ,
            dt,
            matrices;
            work=radial_work,
        )
        solver_solve_magnetic_implicit_step!(
            magnetic.𝒯,
            magnetic.work_tor,
            matrices,
            :toroidal;
            mag_bc_inner=inner_bc === nothing ? nothing : inner_bc[1],
            prev_bc_inner=inner_bc === nothing ? nothing : inner_bc[2],
            mag_bc_inner_imag=inner_bc === nothing ? nothing : inner_bc[3],
            prev_bc_inner_imag=inner_bc === nothing ? nothing : inner_bc[4],
            work=radial_work,
        )
    elseif timestepper isa EAB2
        if inner_bc !== nothing
            throw(ArgumentError("CONTINUITY_MAG toroidal magnetic inner-boundary increments are not implemented for EAB2() timestepping"))
        end
        alu_map = (state.timestep_caches.etd_magnetic_toroidal::EAB2CacheEntry{T}).map
        radial_work = solver_get_radial_work!(
            state.timestep_caches,
            :magnetic_toroidal,
            runtime.𝒟ᵒᶜ.N,
        )
        bc_spec = build_solver_erk2_magnetic_tor_bc(T, runtime.𝒟ᵒᶜ.N)
        solver_eab2_update_krylov_cached!(
            magnetic.𝒯,
            magnetic.nlᵀ,
            magnetic.prev_nlᵀ,
            alu_map,
            runtime.𝒟ᵒᶜ,
            1.0,
            runtime.shtns_config,
            dt;
            m=_timestepper_krylov_dimension(timestepper, state.parameters),
            tol=_timestepper_krylov_tolerance(timestepper, state.parameters),
            bc_spec=bc_spec,
            krylov_work=radial_work,
        )
    else
        solver_solve_magnetic_implicit_step!(
            magnetic.𝒯,
            magnetic.nlᵀ,
            state.implicit_matrices[:magnetic_tor],
            :toroidal;
            mag_bc_inner=inner_bc === nothing ? nothing : inner_bc[1],
            prev_bc_inner=inner_bc === nothing ? nothing : inner_bc[2],
            mag_bc_inner_imag=inner_bc === nothing ? nothing : inner_bc[3],
            prev_bc_inner_imag=inner_bc === nothing ? nothing : inner_bc[4],
        )
    end

    return state
end

function apply_magnetic_poloidal_implicit_update!(state::SolverState{T,<:AbstractArchitecture}) where T
    magnetic = state.fields.magnetic
    magnetic === nothing && return state

    runtime = state.runtime
    timestepper = state.parameters.timestepper
    dt = state.parameters.timestep

    if timestepper isa CNAB2
        matrices = state.implicit_matrices[:magnetic_pol]
        radial_work = solver_get_radial_work!(
            state.timestep_caches,
            :magnetic_poloidal,
            matrices.system_matrices[1].size,
        )
        solver_build_rhs_cnab2!(
            magnetic.work_pol,
            magnetic.𝒫,
            magnetic.nlᴾ,
            magnetic.prev_nlᴾ,
            dt,
            matrices;
            work=radial_work,
        )
        solver_solve_magnetic_implicit_step!(
            magnetic.𝒫,
            magnetic.work_pol,
            matrices,
            :poloidal,
            work=radial_work,
        )
    elseif timestepper isa EAB2
        alu_map = (state.timestep_caches.etd_magnetic_poloidal::EAB2CacheEntry{T}).map
        radial_work = solver_get_radial_work!(
            state.timestep_caches,
            :magnetic_poloidal,
            runtime.𝒟ᵒᶜ.N,
        )
        bc_spec = build_solver_erk2_magnetic_pol_bc(T, runtime.𝒟ᵒᶜ)
        solver_eab2_update_krylov_cached!(
            magnetic.𝒫,
            magnetic.nlᴾ,
            magnetic.prev_nlᴾ,
            alu_map,
            runtime.𝒟ᵒᶜ,
            1.0,
            runtime.shtns_config,
            dt;
            m=_timestepper_krylov_dimension(timestepper, state.parameters),
            tol=_timestepper_krylov_tolerance(timestepper, state.parameters),
            bc_spec=bc_spec,
            krylov_work=radial_work,
        )
    else
        solver_solve_magnetic_implicit_step!(
            magnetic.𝒫,
            magnetic.nlᴾ,
            state.implicit_matrices[:magnetic_pol],
            :poloidal,
        )
    end

    return state
end

function queue_magnetic_implicit_updates!(
    operations::Vector{Function},
    state::SolverState{T,<:AbstractArchitecture},
) where T
    state.fields.magnetic === nothing && return operations
    push!(operations, () -> apply_magnetic_toroidal_implicit_update!(state))
    push!(operations, () -> apply_magnetic_poloidal_implicit_update!(state))
    return operations
end
