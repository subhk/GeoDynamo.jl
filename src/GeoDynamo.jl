module GeoDynamo

using LinearAlgebra
using SparseArrays
using Random
using SHTnsKit   # Load SHTnsKit before MPI to avoid eager extension load during precompile
using MPI
using PencilArrays
using PencilFFTs
using NCDatasets
using OrderedCollections: OrderedDict
using Statistics
using Dates
using Printf
using JLD2

const ERK2_DIAGNOSTICS_ENABLED = Ref(false)
const ERK2_DIAGNOSTICS_INTERVAL = Ref(1)

struct GPUBackendState{
    SS <: Function,
    SA <: Function,
    VS <: Function,
    VA <: Function,
    SZ <: Function,
    FSC <: Function,
    SSC <: Function,
    EPS <: Function,
    SPS <: Function,
    FVC <: Function,
    SVC <: Function,
    EVC <: Function,
    SVCS <: Function
}
    loaded::Bool
    available::Bool
    device::Symbol
    scalar_synthesis::SS
    scalar_analysis::SA
    vector_synthesis::VS
    vector_analysis::VA
    scratch_zeros::SZ
    fill_scalar_coeff_buffer::FSC
    store_scalar_coefficients::SSC
    extract_physical_slice::EPS
    store_physical_slice::SPS
    fill_vector_coeff_buffer::FVC
    store_vector_coefficients::SVC
    extract_vector_component::EVC
    store_vector_components::SVCS
end

function _gpu_backend_not_loaded_error()
    return ArgumentError(
        "GPU support is not loaded. Load CUDA alongside GeoDynamo to enable architecture=:gpu.",
    )
end

function _gpu_backend_unavailable_error()
    return ArgumentError(
        "GPU support is loaded, but no functional CUDA device is available for architecture=:gpu.",
    )
end

function _default_gpu_scalar_synthesis(cfg, coeffs; real_output::Bool = true)
    throw(_gpu_backend_not_loaded_error())
end

function _default_gpu_scalar_analysis(cfg, spatial; real_output::Bool = true)
    throw(_gpu_backend_not_loaded_error())
end

function _default_gpu_vector_synthesis(cfg, sph_coeffs, tor_coeffs; real_output::Bool = true)
    throw(_gpu_backend_not_loaded_error())
end

function _default_gpu_vector_analysis(cfg, vtheta, vphi)
    throw(_gpu_backend_not_loaded_error())
end

function _default_gpu_scratch_zeros(::Type{T}, dims...) where {T}
    return zeros(T, dims...)
end

function _default_gpu_fill_scalar_coeff_buffer(
        coeffs_buffer, spec_real, spec_imag, r_local, config)
    return cpu_fill_scalar_coeff_buffer!(
        coeffs_buffer, spec_real, spec_imag, r_local, config)
end

function _default_gpu_store_scalar_coefficients(
        spec_real, spec_imag, coeffs_matrix, r_local, config)
    return cpu_store_scalar_coefficients!(
        spec_real, spec_imag, coeffs_matrix, r_local, config)
end

function _default_gpu_extract_physical_slice(
        slice_buffer, phys_data, r_local, config; axes_local = nothing)
    return cpu_extract_physical_slice!(
        slice_buffer, phys_data, r_local, config; axes_local = axes_local)
end

function _default_gpu_store_physical_slice(phys_data, phys_slice, r_local)
    return cpu_store_physical_slice!(phys_data, phys_slice, r_local)
end

function _default_gpu_fill_vector_coeff_buffer(
        coeffs_buffer, spec_real, spec_imag, r_local, config)
    return cpu_fill_vector_coeff_buffer!(
        coeffs_buffer, spec_real, spec_imag, r_local, config)
end

function _default_gpu_store_vector_coefficients(
        spec_real, spec_imag, coeffs_matrix, r_local, config)
    return cpu_store_vector_coefficients!(
        spec_real, spec_imag, coeffs_matrix, r_local, config)
end

function _default_gpu_extract_vector_component(
        component_buffer, v_data, r_local, config; axes_local = nothing)
    return cpu_extract_vector_component!(
        component_buffer,
        v_data,
        r_local,
        config;
        axes_local = axes_local
    )
end

function _default_gpu_store_vector_components(
        v_theta, v_phi, vt_field, vp_field, r_local, config; axes_local = nothing)
    return cpu_store_vector_components!(
        v_theta,
        v_phi,
        vt_field,
        vp_field,
        r_local,
        config;
        axes_local = axes_local
    )
end

const _GPU_BACKEND = Ref{GPUBackendState}(GPUBackendState(
    false, false, :cpu,
    _default_gpu_scalar_synthesis,
    _default_gpu_scalar_analysis,
    _default_gpu_vector_synthesis,
    _default_gpu_vector_analysis,
    _default_gpu_scratch_zeros,
    _default_gpu_fill_scalar_coeff_buffer,
    _default_gpu_store_scalar_coefficients,
    _default_gpu_extract_physical_slice,
    _default_gpu_store_physical_slice,
    _default_gpu_fill_vector_coeff_buffer,
    _default_gpu_store_vector_coefficients,
    _default_gpu_extract_vector_component,
    _default_gpu_store_vector_components
))

function register_gpu_backend!(;
        available::Bool,
        device::Symbol = :cuda,
        scalar_synthesis,
        scalar_analysis,
        vector_synthesis,
        vector_analysis,
        scratch_zeros = _default_gpu_scratch_zeros,
        fill_scalar_coeff_buffer = _default_gpu_fill_scalar_coeff_buffer,
        store_scalar_coefficients = _default_gpu_store_scalar_coefficients,
        extract_physical_slice = _default_gpu_extract_physical_slice,
        store_physical_slice = _default_gpu_store_physical_slice,
        fill_vector_coeff_buffer = _default_gpu_fill_vector_coeff_buffer,
        store_vector_coefficients = _default_gpu_store_vector_coefficients,
        extract_vector_component = _default_gpu_extract_vector_component,
        store_vector_components = _default_gpu_store_vector_components
)
    _GPU_BACKEND[] = GPUBackendState(
        true, available, available ? device : :cpu,
        scalar_synthesis, scalar_analysis,
        vector_synthesis, vector_analysis,
        scratch_zeros, fill_scalar_coeff_buffer,
        store_scalar_coefficients, extract_physical_slice,
        store_physical_slice, fill_vector_coeff_buffer,
        store_vector_coefficients, extract_vector_component,
        store_vector_components
    )
    return nothing
end

@inline gpu_backend_state() = _GPU_BACKEND[]

function restore_gpu_backend!(state::GPUBackendState)
    _GPU_BACKEND[] = state
    return nothing
end

function with_gpu_backend(
        f;
        available::Bool = true,
        device::Symbol = :cuda,
        scalar_synthesis,
        scalar_analysis,
        vector_synthesis,
        vector_analysis,
        scratch_zeros = _default_gpu_scratch_zeros,
        fill_scalar_coeff_buffer = _default_gpu_fill_scalar_coeff_buffer,
        store_scalar_coefficients = _default_gpu_store_scalar_coefficients,
        extract_physical_slice = _default_gpu_extract_physical_slice,
        store_physical_slice = _default_gpu_store_physical_slice,
        fill_vector_coeff_buffer = _default_gpu_fill_vector_coeff_buffer,
        store_vector_coefficients = _default_gpu_store_vector_coefficients,
        extract_vector_component = _default_gpu_extract_vector_component,
        store_vector_components = _default_gpu_store_vector_components
)
    state = gpu_backend_state()
    try
        register_gpu_backend!(
            available = available,
            device = device,
            scalar_synthesis = scalar_synthesis,
            scalar_analysis = scalar_analysis,
            vector_synthesis = vector_synthesis,
            vector_analysis = vector_analysis,
            scratch_zeros = scratch_zeros,
            fill_scalar_coeff_buffer = fill_scalar_coeff_buffer,
            store_scalar_coefficients = store_scalar_coefficients,
            extract_physical_slice = extract_physical_slice,
            store_physical_slice = store_physical_slice,
            fill_vector_coeff_buffer = fill_vector_coeff_buffer,
            store_vector_coefficients = store_vector_coefficients,
            extract_vector_component = extract_vector_component,
            store_vector_components = store_vector_components
        )
        return f()
    finally
        restore_gpu_backend!(state)
    end
end

@inline gpu_backend_loaded() = _GPU_BACKEND[].loaded
@inline gpu_backend_available() = _GPU_BACKEND[].available
@inline function gpu_backend_device()
    s = _GPU_BACKEND[]
    return s.available ? s.device : :cpu
end
@inline gpu_scalar_synthesis(cfg,
    coeffs;
    real_output::Bool = true) = _GPU_BACKEND[].scalar_synthesis(cfg, coeffs; real_output = real_output)
@inline gpu_scalar_analysis(cfg,
    spatial;
    real_output::Bool = true) = _GPU_BACKEND[].scalar_analysis(cfg, spatial; real_output = real_output)
@inline gpu_vector_synthesis(cfg,
    sph_coeffs,
    tor_coeffs;
    real_output::Bool = true) = _GPU_BACKEND[].vector_synthesis(
    cfg, sph_coeffs, tor_coeffs; real_output = real_output)
@inline gpu_vector_analysis(cfg, vtheta, vphi) = _GPU_BACKEND[].vector_analysis(cfg, vtheta, vphi)
@inline function gpu_scratch_zeros(::Type{T}, dims::Vararg{Int, N}) where {T, N}
    return _GPU_BACKEND[].scratch_zeros(T, dims...)::AbstractArray{T, N}
end
@inline gpu_fill_scalar_coeff_buffer(coeffs_buffer,
    spec_real,
    spec_imag,
    r_local,
    config) = _GPU_BACKEND[].fill_scalar_coeff_buffer(
    coeffs_buffer, spec_real, spec_imag, r_local, config)
@inline gpu_store_scalar_coefficients(spec_real,
    spec_imag,
    coeffs_matrix,
    r_local,
    config) = _GPU_BACKEND[].store_scalar_coefficients(
    spec_real, spec_imag, coeffs_matrix, r_local, config)
@inline gpu_extract_physical_slice(slice_buffer,
    phys_data,
    r_local,
    config;
    axes_local = nothing) = _GPU_BACKEND[].extract_physical_slice(
    slice_buffer, phys_data, r_local, config; axes_local = axes_local)
@inline gpu_store_physical_slice(phys_data, phys_slice,
    r_local) = _GPU_BACKEND[].store_physical_slice(phys_data, phys_slice, r_local)
@inline gpu_fill_vector_coeff_buffer(coeffs_buffer,
    spec_real,
    spec_imag,
    r_local,
    config) = _GPU_BACKEND[].fill_vector_coeff_buffer(
    coeffs_buffer, spec_real, spec_imag, r_local, config)
@inline gpu_store_vector_coefficients(spec_real,
    spec_imag,
    coeffs_matrix,
    r_local,
    config) = _GPU_BACKEND[].store_vector_coefficients(
    spec_real, spec_imag, coeffs_matrix, r_local, config)
@inline gpu_extract_vector_component(component_buffer,
    v_data,
    r_local,
    config;
    axes_local = nothing) = _GPU_BACKEND[].extract_vector_component(
    component_buffer, v_data, r_local, config; axes_local = axes_local)
@inline gpu_store_vector_components(v_theta,
    v_phi,
    vt_field,
    vp_field,
    r_local,
    config;
    axes_local = nothing) = _GPU_BACKEND[].store_vector_components(
    v_theta, v_phi, vt_field, vp_field, r_local, config; axes_local = axes_local)

# Public solver-facing generics live at module scope so extension methods do
# not depend on include order to create these names.
function initialize_simulation end
function run_simulation! end
function initialize_fields! end
function extract_all_fields end
function create_enhanced_metadata end
function load_solver_parameters end

# exports transforms/spectral.jl
export SHTnsKitConfig, create_shtnskit_config
export shtnskit_spectral_to_physical!, shtnskit_physical_to_spectral!
export shtnskit_vector_synthesis!, shtnskit_vector_analysis!
export batch_shtnskit_transforms!, get_shtnskit_performance_stats
export batch_spectral_to_physical!, optimize_erk2_transforms!
export enable_erk2_diagnostics!, disable_erk2_diagnostics!, set_erk2_diagnostics_interval!
export erk2_diagnostics_enabled, erk2_diagnostics_interval
export erk2_stage_residual_stats, save_erk2_cache_bundle, load_erk2_cache_bundle
export install_erk2_cache_bundle!, load_erk2_cache_bundle!
export validate_pencil_decomposition, create_erk2_config
# ERK2 influence matrix exports (for velocity poloidal no-penetration enforcement)
export ERK2InfluenceMatrix, create_velocity_poloidal_influence_matrices
export apply_influence_matrix_correction!, apply_velocity_poloidal_influence_correction!
# ERK2 magnetic cache exports (with embedded insulating BCs)
export create_erk2_cache_magnetic_toroidal, create_erk2_cache_magnetic_poloidal
# ERK2 scalar cache exports (temperature/composition with embedded Dirichlet/Neumann BCs)
export create_erk2_cache_scalar, create_erk2_cache_temperature,
       create_erk2_cache_composition

# exports SHTnsKit v1.1.15+ enhanced features
export SHTNSKIT_USE_DISTRIBUTED, SHTNSKIT_USE_QST, SHTNSKIT_USE_SCRATCH_BUFFERS
# Energy/spectrum analysis
export compute_scalar_energy_spectrum, compute_vector_energy_spectrum
export compute_total_scalar_energy, compute_total_vector_energy, compute_enstrophy
# Differential operators
export spectral_gradient!, extract_divergence_coefficients, extract_vorticity_coefficients
# QST transforms
export shtnskit_qst_to_spatial!, shtnskit_spatial_to_qst!
# In-place transforms
export shtnskit_synthesis_inplace!, shtnskit_analysis_inplace!
# Field rotation
export rotate_field_z!, rotate_field_y!, rotate_field_90y!, rotate_field_90x!,
       rotate_field_euler!
# Horizontal Laplacian
export apply_horizontal_laplacian!, apply_inverse_horizontal_laplacian!
export compute_horizontal_gradient_magnitude
# Spectral filtering
export apply_spectral_filter!, apply_exponential_filter!, truncate_spectral_modes!
# Threading and version info
export set_shtnskit_threads, get_shtnskit_version_info
# Fast index conversion
export index_to_lm_fast, build_lm_lookup_tables
# Thread-safe buffer cache utilities
export get_cached_buffer!, clear_buffer_cache!

# exports parallel/*.jl
export get_comm, get_rank, get_nprocs
export create_pencil_topology, create_transpose_plans
export transpose_with_timer!, print_transpose_statistics
export analyze_load_balance, estimate_memory_usage
export create_pencil_array
export print_pencil_info, print_pencil_axes, optimize_communication_order
export ENABLE_TIMING
# MPI validation for parallel transforms
export validate_radial_distribution, check_transform_synchronization

# exports field.jl
export SHTnsSpecField, SHTnsPhysField, SHTnsVectorField, SHTnsTorPolField
export RadialDomain, create_shtns_spectral_field, create_shtns_physical_field
export create_shtns_vector_field, create_radial_domain
export get_local_range, get_local_indices, local_data_size, get_local_data

# Legacy shtns_transforms.jl (deprecated - use SHTnsKit instead)
# Legacy SHTns exports removed in SHTnsKit migration

# exports numerics/banded_operators.jl
export BandedMatrix, create_derivative_matrix, create_radial_laplacian
export apply_∂r!

# exports timestep/implicit.jl
export TimestepState, SHTnsImplicitMatrices, create_shtns_timestepping_matrices
export create_velocity_toroidal_matrices, create_velocity_poloidal_matrices
export create_velocity_green_matrices, solve_velocity_implicit_step!
export apply_explicit_operator!, solve_implicit_step!, compute_timestep_error

# exports physics/velocity/field.jl
export SHTnsVelocityFields, create_shtns_velocity_fields
export VelocityWorkspace, create_velocity_workspace, set_velocity_workspace!
export compute_velocity_nonlinear!, compute_vorticity_spectral_full!
export compute_kinetic_energy, compute_reynolds_stress
export zero_velocity_work_arrays!
export add_thermal_buoyancy_force!
export add_buoyancy_force!, add_lorentz_force!, validate_velocity_configuration

# exports physics/magnetic/field.jl
export SHTnsMagneticFields, create_shtns_magnetic_fields, compute_magnetic_nonlinear!
export compute_current_density_spectral!
export create_magnetic_toroidal_matrices, create_magnetic_poloidal_matrices
export solve_magnetic_implicit_step!

# exports fields/scalar_operators.jl
export GradientWorkspace, create_gradient_workspace, zero_gradient_workspace!
export get_mode_index, clear_mode_index_cache!, clear_scalar_field_caches!

# exports physics/temperature/field.jl
export SHTnsTemperatureField, create_shtns_temperature_field
export compute_temperature_nonlinear!
export compute_nusselt_number, compute_thermal_energy
export compute_surface_flux, get_temperature_statistics
export zero_temperature_work_arrays!
export set_temperature_ic!, set_boundary_conditions!, set_internal_heating!
export create_temperature_matrices, solve_temperature_implicit_step!

# exports physics/composition/field.jl
export SHTnsCompositionField, create_shtns_composition_field
export compute_composition_nonlinear!
export compute_composition_rms, compute_composition_energy
export get_composition_statistics, zero_composition_work_arrays!
export set_composition_ic!, set_composition_boundary_conditions!
export create_composition_matrices, solve_composition_implicit_step!

# exports bcs module
export AbstractBoundaryCondition
export BoundaryLocation, INNER_BOUNDARY, OUTER_BOUNDARY
export BoundaryType, DIRICHLET, NEUMANN, MIXED, ROBIN
export NEUMANN_DERIV1, NEUMANN_DERIV2, NEUMANN_MAG_INNER, NEUMANN_MAG_OUTER, CONTINUITY_MAG
export FieldType, TEMPERATURE, COMPOSITION, VELOCITY, MAGNETIC
export load_boundary_conditions!, update_time_dependent_boundaries!
export validate_boundary_files, get_current_boundaries, print_boundary_summary
export get_boundary_module_info
# bcs SHTnsKit v1.1.15 caching utilities
export clear_bc_shtns_config_cache!, shtns_physical_to_spectral, shtns_spectral_to_physical
# File-based spectral BC loading
export SpectralBoundaryCoefficients, load_spectral_bc_from_file
export store_bc_in_field!, get_bc_vectors_from_field
# topography coupling exports
export enable_topography!, disable_topography!, is_topography_enabled
export TopographyCouplingConfig, get_topography_config, set_topography_config!
export TopographyData, TopographyField, GauntTensorCache
export precompute_gaunt_tensors!, apply_all_topography_corrections!
export gaunt_on_the_fly, gradient_gaunt_from_basic, compute_gaunt_tensor
export evaluate_spherical_harmonics_grid, evaluate_spherical_harmonic_gradient_grid
export create_topography_data, load_topography_from_file
export StefanState, initialize_stefan_state!, update_icb_topography!

# exports runtime workflows
export initialize_simulation, run_simulation!, sync_spectral_history!
export SolverBackend, SolverFields, SolverTopographyState, SolverTimestepState,
       SolverGradientWorkspace, TransformWorkspace, SolverRuntime, SolverState
export create_solver_backend, create_solver_topography_state
export initialize_solver_fields!, initialize_solver_state, solver_step!, advance_solver_step!, run_solver!
export GPUBackendState, register_gpu_backend!, gpu_backend_state, restore_gpu_backend!,
       with_gpu_backend
export gpu_backend_loaded, gpu_backend_available, gpu_backend_device

# exports io/writer.jl
export OutputConfig, FieldInfo, TimeTracker
export default_config, output_config_from_parameters, resolve_output_precision
export with_output_precision
export create_time_tracker, should_output_now, should_restart_now
export write_fields!, write_restart!, read_restart!
export create_shtns_aware_output_config, validate_output_compatibility
export check_parallel_netcdf_support, verify_all_ranks_wrote
export get_time_series, cleanup_old_files

# exports core/initial_conditions.jl
export randomize_scalar_field!, randomize_vector_field!, randomize_magnetic_field!
export load_initial_conditions!, save_initial_conditions
export generate_random_initial_conditions!, set_analytical_initial_conditions!

# exports core/parameters.jl
export load_parameters, save_parameters, create_parameter_template

# New Oceananigans-style user API
export AbstractArchitecture, CPU, GPU, arch_zeros, on_architecture, get_backend, gpu_functional, gpu_synchronize
export GPUPhysicalField, GPUSpectralField, allocate_gpu_physical_field,
       allocate_gpu_spectral_field, field_to_host, field_to_device
export gpu_scalar_spectral_to_physical!, gpu_scalar_physical_to_spectral!
export gpu_scalar_advection!, gpu_cross!, gpu_cross_add!, gpu_coriolis_sub!, gpu_buoyancy_add!
export gpu_vr_scale!, gpu_vector_spectral_to_physical!, gpu_vector_physical_to_spectral!
export gpu_pack_banded_lu, gpu_batched_banded_solve!
export gpu_batched_banded_matvec!, gpu_spectral_curl!
export gpu_phi_gradient!, gpu_theta_gradient!, gpu_scalar_gradient!
export gpu_batched_banded_matvec_perl!, gpu_build_rhs_cnab2!
export SphericalShellGrid, SphericalBallGrid
export NoSlip, StressFree, FixedTemperature, FixedFlux
export InsulatingMagnetic, ConductingMagnetic, BoundaryConditions
export Clock
export GeodynamoModel
export fields, prognostic_fields
export Simulation, run!, time_step!, add_callback!
export AbstractTimestepper, CNAB2, EAB2, ERK2, ETD, ThetaMethod
export TimeInterval, IterationInterval, WallTimeInterval
export FieldWriter, CheckpointWriter
export RandomPerturbation, AnalyticIC, FileIC, ZeroIC
export set_initial_condition!, set!
export Callback, EnergyDiagnostics, SolenoidalMonitor, SimulationProgress, HealthCheck

# ================================================================================
# Centralized Numerical Tolerance Constants
# ================================================================================
# Pivot/singularity detection in linear algebra (LU factorization, back-substitution)
const PIVOT_SINGULARITY_FACTOR = 100
"""Pivot singularity threshold: `abs(pivot) < eps(T) * PIVOT_SINGULARITY_FACTOR`"""
pivot_tol(::Type{T}) where {T} = eps(T) * T(PIVOT_SINGULARITY_FACTOR)

# Series convergence (matrix exponential/phi function Taylor series)
const SERIES_CONVERGENCE_FACTOR = 100
"""Series term convergence: `opnorm(term) < eps(T) * SERIES_CONVERGENCE_FACTOR`"""
series_tol(::Type{T}) where {T} = eps(T) * T(SERIES_CONVERGENCE_FACTOR)

# Condition number threshold for switching from direct solve to series expansion
"""Reciprocal condition number threshold for matrix function fallback"""
rcond_fallback_tol(::Type{T}) where {T} = sqrt(eps(T))

# Architecture type hierarchy — must be first so all subsequent includes can use AbstractArchitecture
include("core/architecture.jl")
include("gpu/device.jl")
include("gpu/fields.jl")
include("gpu/scalar_transform.jl")
include("gpu/nonlinear.jl")
include("gpu/vector_transform.jl")
include("gpu/banded_solve.jl")
include("gpu/spectral_curl.jl")
include("gpu/scalar_gradient.jl")
include("gpu/cnab2_rhs.jl")

# User-facing choice objects are needed by SolverParameters so the internal
# configuration stores the same objects users pass at construction time.
include("api/boundary_conditions.jl")
include("api/timesteppers.jl")

# Include Parameters system first
include("core/parameters.jl")

# Include base modules in dependency order
include("parallel/mpi.jl")
include("parallel/pencils.jl")
include("parallel/transposes.jl")
include("parallel/process_grid.jl")  # GEODYNAMO_PROC_GRID parser + θ/r subcommunicators
include("transforms/spectral.jl")  # SHTnsKit-based transforms (includes SHTnsKitConfig)
include("parallel/spectral_pencil_adapter.jl")  # dense (l,m) <-> SHTnsKit spectral seams
include("parallel/disttranspose_adapter.jl")    # Phase 3 DistTransposePlan plumbing (Task 1)
include("bcs/bcs.jl")  # Needed before field types import BC enums
using .bcs: AbstractBoundaryCondition
using .bcs: BoundaryLocation, INNER_BOUNDARY, OUTER_BOUNDARY
using .bcs: BoundaryType, DIRICHLET, NEUMANN, MIXED, ROBIN, NEUMANN_DERIV1, NEUMANN_DERIV2,
            NEUMANN_MAG_INNER, NEUMANN_MAG_OUTER, CONTINUITY_MAG
using .bcs: FieldType, TEMPERATURE, COMPOSITION, VELOCITY, MAGNETIC
using .bcs: load_boundary_conditions!, update_time_dependent_boundaries!
using .bcs: validate_boundary_files, get_current_boundaries, print_boundary_summary
using .bcs: get_boundary_module_info
using .bcs: clear_bc_shtns_config_cache!, shtns_physical_to_spectral,
            shtns_spectral_to_physical
using .bcs: BoundaryInterpolationCache, SpectralBoundaryCoefficients,
            load_spectral_bc_from_file
using .bcs: store_bc_in_field!, get_bc_vectors_from_field
using .bcs: enable_topography!, disable_topography!, is_topography_enabled
using .bcs: TopographyCouplingConfig, get_topography_config, set_topography_config!
using .bcs: TopographyData, TopographyField, GauntTensorCache
using .bcs: precompute_gaunt_tensors!, apply_all_topography_corrections!
using .bcs: gaunt_on_the_fly, gradient_gaunt_from_basic, compute_gaunt_tensor
using .bcs: evaluate_spherical_harmonics_grid, evaluate_spherical_harmonic_gradient_grid
using .bcs: create_topography_data, load_topography_from_file
using .bcs: StefanState, initialize_stefan_state!, update_icb_topography!
include("fields/containers.jl")  # Field/type definitions needed by subsequent modules
include("core/spectral_history.jl")
include("fields/transforms.jl")  # Field-dependent transform functions
include("numerics/banded_operators.jl")  # Requires RadialDomain/SHTns field types

# include("shtns_transforms.jl")  # Legacy - replaced by transforms/spectral.jl
# include("shtns_config.jl")      # Legacy - replaced by SHTnsKit configurations

include("fields/scalar_operators.jl")  # Depends on BandedMatrix definitions

# Timestepping modules
include("timestep/state.jl")
include("core/simulation_health.jl")
include("timestep/implicit.jl")
include("bcs/scalar_bc.jl")
include("physics/magnetic/field.jl")
include("physics/velocity/field.jl")
include("physics/temperature/field.jl")
include("physics/composition/field.jl")

# Include InitialConditions module
include("core/initial_conditions.jl")
using .InitialConditions
include("io/writer.jl")

# New Oceananigans-style user API
include("api/grids.jl")
include("api/schedules.jl")

# Geometry-specific convenience layers
include("Shell/Shell.jl")
include("Ball/Ball.jl")
include("solver.jl")
include("api/clock.jl")
include("api/model.jl")
include("api/initial_conditions.jl")
include("api/set.jl")
include("api/fields.jl")
include("api/callbacks.jl")
include("api/output_writers.jl")
include("api/simulation.jl")
include("api/show.jl")
# Import all Ball exports into GeoDynamo namespace
using .GeoDynamoBall

# Import all Shell exports into GeoDynamo namespace
using .GeoDynamoShell

# Backward-compatible alias for the flattened solver API.
const Solver = GeoDynamo

# Re-export Ball functions for convenience
export BallConfig, create_ball_pencils
export create_ball_radial_domain, create_ball_spectral_field
export create_ball_physical_field, create_ball_vector_field
export create_ball_velocity_fields, create_ball_magnetic_fields
export create_ball_temperature_field, create_ball_composition_field
export create_ball_hybrid_temperature_boundaries, create_ball_hybrid_composition_boundaries
export enforce_ball_scalar_regularity!, enforce_ball_vector_regularity!
export apply_ball_temperature_regularity!, apply_ball_composition_regularity!
export ball_physical_to_spectral!, ball_vector_analysis!

# Re-export Shell functions for convenience
export ShellConfig, create_shell_pencils
export create_shell_radial_domain, create_shell_spectral_field
export create_shell_physical_field, create_shell_vector_field
export create_shell_velocity_fields, create_shell_magnetic_fields
export create_shell_temperature_field, create_shell_composition_field
export create_shell_hybrid_temperature_boundaries,
       create_shell_hybrid_composition_boundaries
export apply_shell_temperature_boundaries!, apply_shell_composition_boundaries!

# Initialize parameters when module is loaded
function __init__()
    try
        # Load MPI at runtime if not already loaded
        # This is needed for SHTnsKit parallel extensions to work properly
        # Only import core MPI symbols that exist in all versions
        if !isdefined(Main, :MPI)
            try
                @eval using MPI: Allgather, Allreduce, Allreduce!, Barrier, COMM_WORLD,
                                 Cart_shift, Comm, Comm_rank, Comm_size, Finalize, Gather,
                                 Init, Initialized, Isend, MAX, MIN, Request, SUM, Waitall,
                                 Wtime, bcast
                @debug "GeoDynamo.jl loaded MPI at runtime"
            catch mpi_e
                @debug "Could not load MPI (continuing without MPI support): $mpi_e"
            end
        else
            try
                @eval using MPI: Allgather, Allreduce, Allreduce!, Barrier, COMM_WORLD,
                                 Cart_shift, Comm, Comm_rank, Comm_size, Finalize, Gather,
                                 Init, Initialized, Isend, MAX, MIN, Request, SUM, Waitall,
                                 Wtime, bcast
                @debug "GeoDynamo.jl detected MPI already available"
            catch mpi_e
                @debug "MPI appears loaded but importing symbols failed: $mpi_e"
            end
        end

        initialize_parameters()
        @info "GeoDynamo.jl initialized successfully"
    catch e
        @warn "Could not initialize GeoDynamo.jl properly: $e"
        try
            set_parameters!(SolverParameters())
            @info "Using default parameters"
        catch param_e
            @warn "Failed to set default parameters: $param_e"
        end
    end
end

end
