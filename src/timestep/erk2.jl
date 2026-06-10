#
# ERK2 staged exponential timestep support
#
# Developer map:
# - Boundary descriptors encode endpoint constraints and how to enforce them on
#   dense radial profiles.
# - Cache constructors precompute per-degree linear propagators (`E`, `phi1`,
#   `phi2`) for each active field family.
# - Cache getters rebuild only when physical parameters, timestep, grid, or
#   Krylov settings change.
# - Field buffers hold the provisional stage, current nonlinear terms, and
#   finalization work arrays for one field.
# - `integrate_solver_erk2_step!` orchestrates the full solver step across all
#   active fields.
#
# Forward-declare the public ERK2 API names so the `GeoDynamo.<name>` methods
# defined across the erk2/*.jl files below attach to bindings that already exist
# in the module (and the public names resolve regardless of which methods load).
@eval GeoDynamo begin
    function enforce_erk2_bc! end
    function create_dirichlet_bc end
    function create_neumann_bc end
    function create_stress_free_tor_bc end
    function create_noslip_pol_bc end
    function create_stress_free_pol_bc end
    function create_insulating_inner_bc end
    function create_insulating_outer_bc end
    function set_erk2_diagnostics_interval! end
    function enable_erk2_diagnostics! end
    function disable_erk2_diagnostics! end
    function erk2_diagnostics_enabled end
    function erk2_diagnostics_interval end
    function create_erk2_cache end
    function create_erk2_cache_scalar end
    function create_erk2_cache_temperature end
    function create_erk2_cache_composition end
    function create_erk2_cache_magnetic_toroidal end
    function create_erk2_cache_magnetic_poloidal end
    function compute_phi1_function end
    function compute_phi2_function end
    function reset_phi2_monitor! end
    function report_phi2_conditioning end
    function erk2_prepare_field! end
    function erk2_apply_stage! end
    function erk2_store_stage_nonlinear! end
    function erk2_finalize_field! end
    function erk2_stage_residual_stats end
    function maybe_log_erk2_stage_residual! end
    function create_velocity_poloidal_influence_matrices end
    function apply_influence_matrix_correction! end
    function apply_velocity_poloidal_influence_correction! end
    function save_erk2_cache_bundle end
    function load_erk2_cache_bundle end
    function install_erk2_cache_bundle! end
    function load_erk2_cache_bundle! end
end

include("erk2/common.jl")  # aliases first — other erk2/*.jl files depend on these consts
include("erk2/boundary.jl")
include("erk2/cache.jl")
include("erk2/influence.jl")
include("erk2/integrate.jl")
