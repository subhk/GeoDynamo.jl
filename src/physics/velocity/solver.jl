function initialize_velocity_field!(state::SolverState{T,<:AbstractArchitecture}) where T
    velocity = state.fields.velocity
    fill!(parent(velocity.𝒯.data_real), zero(T))
    fill!(parent(velocity.𝒯.data_imag), zero(T))
    fill!(parent(velocity.𝒫.data_real), zero(T))
    fill!(parent(velocity.𝒫.data_imag), zero(T))
    return state
end

function prepare_velocity_fields!(velocity_fields, domain)
    # Nonlinear velocity terms use the current velocity and vorticity in
    # physical space, so refresh both before accumulating body forces.
    reset_velocity_work_arrays!(velocity_fields)
    refresh_velocity_physical_fields!(velocity_fields, domain)
    refresh_vorticity_physical_fields!(velocity_fields, domain)
    return velocity_fields
end

function accumulate_velocity_nonlinear_terms!(
    velocity_fields,
    temperature_field,
    composition_field,
    magnetic_field,
    domain,
    params::SolverParameters,
)
    return compute_velocity_body_forces!(
        velocity_fields,
        temperature_field,
        composition_field,
        magnetic_field,
        domain,
        params,
    )
end

function finish_velocity_nonlinear!(velocity_fields; geometry::Symbol)
    if geometry === :ball
        return solver_ball_vector_analysis!(
            velocity_fields.advection_physical,
            velocity_fields.nlᵀ,
            velocity_fields.nlᴾ,
        )
    end
    return vector_physical_to_spectral!(
        velocity_fields.advection_physical,
        velocity_fields.nlᵀ,
        velocity_fields.nlᴾ,
    )
end

function apply_velocity_toroidal_implicit_update!(state::SolverState{T,<:AbstractArchitecture}) where T
    velocity = state.fields.velocity
    runtime = state.runtime
    timestepper = state.parameters.timestepper
    dt = state.parameters.timestep
    E = state.parameters.Ek
    velocity_bc = __velocity_bc_code(state.parameters.velocity_bcs)

    if timestepper isa CNAB2
        matrices = state.implicit_matrices[:velocity_tor]
        radial_work = get_radial_work!(
            state.timestep_caches,
            :velocity_toroidal,
            matrices.system_matrices[1].size,
        )
        solver_build_rhs_cnab2!(
            velocity.work_tor,
            velocity.𝒯,
            velocity.nlᵀ,
            velocity.prev_nlᵀ,
            dt,
            matrices;
            mass_coeff=E,
            work=radial_work,
        )
        solver_solve_velocity_implicit_step!(
            velocity.𝒯,
            velocity.work_tor,
            matrices,
            :toroidal;
            velocity_bc_code=velocity_bc,
            domain=runtime.𝒟ᵒᶜ,
            work=radial_work,
        )
    elseif timestepper isa EAB2
        alu_map = (state.timestep_caches.etd_velocity_toroidal::EAB2CacheEntry{T}).map
        radial_work = get_radial_work!(
            state.timestep_caches,
            :velocity_toroidal,
            runtime.𝒟ᵒᶜ.N,
        )
        bc_spec = build_solver_erk2_velocity_tor_bc(
            T,
            runtime.𝒟ᵒᶜ,
            velocity_bc;
            config=runtime.shtns_config,
            rot_omega=0.0,
        )
        solver_eab2_update_krylov_cached!(
            velocity.𝒯,
            velocity.nlᵀ,
            velocity.prev_nlᵀ,
            alu_map,
            runtime.𝒟ᵒᶜ,
            E,
            runtime.shtns_config,
            dt;
            m=__timestepper_krylov_dimension(timestepper, state.parameters),
            tol=__timestepper_krylov_tolerance(timestepper, state.parameters),
            mass_coeff=E,
            bc_spec=bc_spec,
            krylov_work=radial_work,
        )
    else
        matrices = state.implicit_matrices[:velocity_tor]
        radial_work = get_radial_work!(
            state.timestep_caches,
            :velocity_toroidal,
            matrices.system_matrices[1].size,
        )
        solver_solve_velocity_implicit_step!(
            velocity.𝒯,
            velocity.nlᵀ,
            matrices,
            :toroidal;
            velocity_bc_code=velocity_bc,
            domain=runtime.𝒟ᵒᶜ,
            work=radial_work,
        )
    end

    return state
end

function apply_velocity_poloidal_no_penetration!(
    state::SolverState{T,<:AbstractArchitecture},
    velocity_bc_code::Int,
) where T
    params = state.parameters
    runtime = state.runtime
    theta = __timestepper_implicit_theta(params.timestepper, params)
    effective_diffusivity = one(params.Ek)
    influence = get_solver_erk2_influence_matrices!(
        state.timestep_caches,
        :velocity_poloidal,
        T,
        runtime.shtns_config,
        runtime.𝒟ᵒᶜ,
        effective_diffusivity,
        params.timestep,
        velocity_bc_code;
        theta=theta,
    )
    apply_solver_velocity_poloidal_influence_correction!(
        state.fields.velocity.𝒫,
        influence,
        runtime.shtns_config,
    )
    return state
end

function apply_velocity_poloidal_implicit_update!(state::SolverState{T,<:AbstractArchitecture}) where T
    velocity = state.fields.velocity
    runtime = state.runtime
    timestepper = state.parameters.timestepper
    dt = state.parameters.timestep
    E = state.parameters.Ek
    velocity_bc = __velocity_bc_code(state.parameters.velocity_bcs)

    if timestepper isa CNAB2
        matrices = state.implicit_matrices[:velocity_pol]
        radial_work = get_radial_work!(
            state.timestep_caches,
            :velocity_poloidal,
            matrices.system_matrices[1].size,
        )
        solver_build_rhs_cnab2!(
            velocity.work_pol,
            velocity.𝒫,
            velocity.nlᴾ,
            velocity.prev_nlᴾ,
            dt,
            matrices;
            mass_coeff=E,
            work=radial_work,
        )
        solver_solve_velocity_implicit_step!(
            velocity.𝒫,
            velocity.work_pol,
            matrices,
            :poloidal;
            velocity_bc_code=velocity_bc,
            domain=runtime.𝒟ᵒᶜ,
            work=radial_work,
        )
    elseif timestepper isa EAB2
        alu_map = (state.timestep_caches.etd_velocity_poloidal::EAB2CacheEntry{T}).map
        radial_work = get_radial_work!(
            state.timestep_caches,
            :velocity_poloidal,
            runtime.𝒟ᵒᶜ.N,
        )
        bc_spec = build_solver_erk2_velocity_pol_bc(T, runtime.𝒟ᵒᶜ, velocity_bc)
        solver_eab2_update_krylov_cached!(
            velocity.𝒫,
            velocity.nlᴾ,
            velocity.prev_nlᴾ,
            alu_map,
            runtime.𝒟ᵒᶜ,
            E,
            runtime.shtns_config,
            dt;
            m=__timestepper_krylov_dimension(timestepper, state.parameters),
            tol=__timestepper_krylov_tolerance(timestepper, state.parameters),
            mass_coeff=E,
            bc_spec=bc_spec,
            krylov_work=radial_work,
        )
    else
        matrices = state.implicit_matrices[:velocity_pol]
        radial_work = get_radial_work!(
            state.timestep_caches,
            :velocity_poloidal,
            matrices.system_matrices[1].size,
        )
        solver_solve_velocity_implicit_step!(
            velocity.𝒫,
            velocity.nlᴾ,
            matrices,
            :poloidal;
            velocity_bc_code=velocity_bc,
            domain=runtime.𝒟ᵒᶜ,
            work=radial_work,
        )
    end

    apply_velocity_poloidal_no_penetration!(state, velocity_bc)
    return state
end

function queue_velocity_implicit_updates!(
    operations::Vector{Function},
    state::SolverState{T,<:AbstractArchitecture},
) where T
    push!(operations, () -> apply_velocity_toroidal_implicit_update!(state))
    push!(operations, () -> apply_velocity_poloidal_implicit_update!(state))
    return operations
end
