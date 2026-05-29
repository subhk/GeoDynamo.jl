const SOLVER_INNER_BOUNDARY = getproperty(GeoDynamo, :INNER_BOUNDARY)
const SOLVER_OUTER_BOUNDARY = getproperty(GeoDynamo, :OUTER_BOUNDARY)

function create_solver_topography_config(params::SolverParameters)
    return TopographyAPI.TopographyCouplingConfig(
        enabled = params.topography_enabled,
        velocity_coupling = params.include_topography_velocity,
        magnetic_coupling = params.include_topography_magnetic,
        thermal_coupling = params.include_topography_thermal,
        stefan_enabled = params.stefan_enabled,
        include_shift_terms = params.include_topography_shift_terms,
        include_slope_terms = params.include_topography_slope_terms,
        epsilon = params.topography_epsilon,
        lmax_topo = params.topography_degree
    )
end

function load_solver_topography_data(::Type{T}, params::SolverParameters) where {T}
    # Topography files may still matter when explicit coupling is off, because
    # Stefan evolution and diagnostics can consume the same loaded surfaces.
    needs_topography = params.topography_enabled ||
                       !isempty(params.icb_topography_file) ||
                       !isempty(params.ocb_topography_file) ||
                       params.stefan_enabled

    needs_topography || return nothing

    data = TopographyAPI.TopographyData{T}()
    data.epsilon = T(params.topography_epsilon)
    data.is_time_dependent = params.stefan_enabled

    if !isempty(params.icb_topography_file)
        data.icb = TopographyAPI.load_topography_from_file(
            params.icb_topography_file,
            SOLVER_INNER_BOUNDARY;
            radius = params.radius_ratio
        )
    end

    if !isempty(params.ocb_topography_file)
        data.cmb = TopographyAPI.load_topography_from_file(
            params.ocb_topography_file,
            SOLVER_OUTER_BOUNDARY;
            radius = 1.0
        )
    end

    if params.topography_enabled && (data.icb !== nothing || data.cmb !== nothing)
        lmax_topo = params.topography_degree > 0 ? params.topography_degree : params.lmax
        data.gaunt_cache = TopographyAPI.GauntTensorCache{T}(params.lmax, lmax_topo)
        TopographyAPI.precompute_gaunt_tensors!(data.gaunt_cache; verbose = false, use_wigner = true)
    end

    return data
end

function create_solver_topography_state(::Type{T}, params::SolverParameters) where {T}
    config = create_solver_topography_config(params)
    data = load_solver_topography_data(T, params)

    stefan = nothing
    if params.stefan_enabled
        stefan = TopographyAPI.StefanState{T}(
            lmax = params.lmax,
            ri = T(params.radius_ratio),
            k_ic = T(params.inner_core_conductivity_ratio),
            k_oc = one(T),
            rho = one(T),
            L = T(params.latent_heat)
        )
        stefan.Stefan = T(params.stefan_number)
        if data !== nothing && data.icb !== nothing
            stefan.topography = data.icb
        end
    end

    return SolverTopographyState{T}(config, data, stefan)
end

function activate_solver_topography!(topography::SolverTopographyState)
    TopographyAPI.set_topography_config!(topography.config)
    return topography
end

function apply_solver_topography!(state::SolverState)
    topo_data = state.topography.data
    topo_data === nothing && return state
    # Respect runtime enable/disable toggles even when topography data has
    # already been loaded into the solver state.
    state.topography.config.enabled || return state

    TopographyAPI.apply_all_topography_corrections!(
        state.fields,
        topo_data;
        config = state.topography.config
    )

    return state
end

function update_solver_icb_phase_change!(state::SolverState)
    stefan = state.topography.stefan
    stefan === nothing && return state

    # Keep the Stefan update visible in the new solver order. The actual
    # evaluation still depends on a dedicated inner-core thermal state that the
    # rewritten root solver path does not own yet.
    @warn "ICB phase-change update is not active yet in the rewritten solver path: a separate inner-core temperature field is still required." maxlog=1
    return state
end
