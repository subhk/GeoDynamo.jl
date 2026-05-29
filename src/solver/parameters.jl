@inline __solver_current_params() = get_parameters()
@inline solver_default_geometry() = __solver_current_params().geometry

"""
    create_solver_parameters([params])

Return a `SolverParameters` object for solver initialization.
"""
create_solver_parameters() = __solver_current_params()
create_solver_parameters(params::SolverParameters) = params

"""
    apply_solver_parameters!(params)

Record `params` as the active solver parameter set used by shared helpers.
"""
function apply_solver_parameters!(params::SolverParameters)
    set_parameters!(params; validate=false)
    set_timing_enabled!(solver_shared_timing_enabled())
    set_erk2_diagnostics!(
        solver_shared_erk2_diagnostics_enabled(),
        solver_shared_erk2_diagnostics_interval(),
    )
    return params
end

save_solver_parameters(params::SolverParameters, filename::String) =
    save_parameters(params, filename)

create_solver_parameter_template(filename::String="solver_parameters.jl") =
    create_parameter_template(filename)

load_solver_parameters_file(config_file::String="") =
    isempty(config_file) ? __solver_current_params() : load_parameters(config_file)

load_solver_parameters(config_file::String="") = load_solver_parameters_file(config_file)
