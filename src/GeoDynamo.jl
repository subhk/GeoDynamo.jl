module GeoDynamo

    using LinearAlgebra
    using SparseArrays
    using SHTnsKit   # Load SHTnsKit before MPI to avoid eager extension load during precompile
    using MPI
    using PencilArrays
    using PencilFFTs
    using HDF5
    using StaticArrays
    using NCDatasets
    using Statistics
    using Dates
    using Printf

    # exports shtnskit_transforms.jl (new SHTnsKit-based implementation)
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
    export rotate_field_z!, rotate_field_y!, rotate_field_90y!, rotate_field_90x!, rotate_field_euler!
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

    # exports pencil_decomps.jl
    export get_comm, get_rank, get_nprocs
    export create_pencil_topology, create_transpose_plans
    export transpose_with_timer!, print_transpose_statistics
    export analyze_load_balance, estimate_memory_usage
    export create_pencil_array, synchronize_halos!
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

    # exports linear_algebra.jl
    export BandedMatrix, create_derivative_matrix, create_radial_laplacian
    export apply_∂r!

    # exports timestep.jl
    export TimestepState, SHTnsImplicitMatrices, create_shtns_timestepping_matrices
    export create_velocity_toroidal_matrices, create_velocity_poloidal_matrices
    export create_velocity_green_matrices, solve_velocity_implicit_step!
    export apply_explicit_operator!, solve_implicit_step!, compute_timestep_error

    # exports velocity.jl
    export SHTnsVelocityFields, create_shtns_velocity_fields
    export VelocityWorkspace, create_velocity_workspace, set_velocity_workspace!
    export compute_velocity_nonlinear!, compute_vorticity_spectral_full!
    export compute_kinetic_energy, compute_reynolds_stress
    export zero_velocity_work_arrays!
    export add_thermal_buoyancy_force!
    export add_buoyancy_force!, add_lorentz_force!, validate_velocity_configuration

    # exports magnetic.jl
    export SHTnsMagneticFields, create_shtns_magnetic_fields, compute_magnetic_nonlinear!
    export compute_current_density_spectral!
    export create_magnetic_toroidal_matrices, create_magnetic_poloidal_matrices
    export solve_magnetic_implicit_step!

    # exports thermal.jl
    export SHTnsTemperatureField, create_shtns_temperature_field
    export compute_temperature_nonlinear!
    export compute_nusselt_number, compute_thermal_energy
    export compute_surface_flux, get_temperature_statistics
    export zero_temperature_work_arrays!
    export set_temperature_ic!, set_boundary_conditions!, set_internal_heating!
    export batch_transform_to_physical!
    export create_temperature_matrices, solve_temperature_implicit_step!

    # exports compositional.jl
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
    export create_topography_data, load_topography_from_file
    export StefanState, initialize_stefan_state!, update_icb_topography!

    # exports simulation.jl (single unified state)
    export SimulationState, initialize_simulation, run_simulation!

    # exports outputs_writer.jl
    export OutputConfig, FieldInfo, TimeTracker
    export default_config, output_config_from_parameters, resolve_output_precision
    export with_output_precision
    export create_time_tracker, should_output_now, should_restart_now
    export write_fields!, write_restart!, read_restart!
    export create_shtns_aware_output_config, validate_output_compatibility
    export check_parallel_netcdf_support, verify_all_ranks_wrote
    export get_time_series, cleanup_old_files

    # exports spectral_to_physical.jl (from extras)
    export SpectralToPhysicalConverter
    export create_spectral_converter, load_spectral_data!, convert_to_physical!
    export compute_global_diagnostics, save_physical_fields
    export convert_spectral_file, batch_convert_directory
    export main_convert_file, main_batch_convert

    # exports optimizations.jl (unified parallelization system)
    export AdvancedThreadManager, ThreadingAccelerator, SIMDOptimizer, TaskGraph, MemoryOptimizer
    export AsyncCommManager, DynamicLoadBalancer, ParallelIOOptimizer, PerformanceMonitor
    export HybridParallelizer, CPUParallelizer, MasterParallelizer
    export create_advanced_thread_manager, create_threading_accelerator, create_simd_optimizer
    export create_task_graph, create_memory_optimizer, create_async_comm_manager
    export create_dynamic_load_balancer, create_parallel_io_optimizer, create_performance_monitor
    export create_hybrid_parallelizer, create_cpu_parallelizer, create_master_parallelizer
    export hybrid_compute_nonlinear!, compute_nonlinear!, add_task!, execute_task_graph!
    export async_write_fields!, analyze_parallel_performance, adaptive_rebalance!
    export allocate_aligned_array, deallocate_aligned_array, optimize_memory_layout!

    # (deprecated) enhanced/master types removed in favor of unified SimulationState

    # exports InitialConditions.jl
    export set_temperature_initial_conditions!, set_velocity_initial_conditions!
    export set_magnetic_initial_conditions!, set_composition_initial_conditions!
    export randomize_scalar_field!, randomize_vector_field!, randomize_magnetic_field!
    export generate_random_field, generate_spherical_harmonic_field
    export load_initial_conditions!, save_initial_conditions

    # exports parameters.jl
    export GeoDynamoParameters, load_parameters, save_parameters, create_parameter_template
    export get_parameters, set_parameters!, initialize_parameters
    export @param  # Deprecated - use direct variable access instead

    # Include Parameters system first
    include("parameters.jl")

    # Include base modules in dependency order
    include("pencil_decomps.jl")
    include("shtnskit_transforms.jl")  # New SHTnsKit-based transforms (includes SHTnsKitConfig)
    include("bcs/bcs.jl")  # Needed before field types import BC enums
    include("fields.jl")  # Field/type definitions needed by subsequent modules
    include("shtnskit_field_functions.jl")  # Field-dependent transform functions
    include("linear_algebra.jl")  # Requires RadialDomain/SHTns field types

    # include("shtns_transforms.jl")  # Legacy - replaced by shtnskit_transforms.jl
    # include("shtns_config.jl")      # Legacy - replaced by SHTnsKit configurations

    include("scalar_field_common.jl")  # Depends on BandedMatrix definitions
    
    include("timestep.jl")
    include("magnetic.jl")
    include("velocity.jl")
    include("thermal.jl")
    include("compositional.jl")

    # Include InitialConditions module
    include("InitialConditions.jl")
    include("outputs_writer.jl")
    include("optimizations.jl")
    include("simulation.jl")

    include("../extras/spectral_to_physical.jl")
    include("combiner.jl")

    # Geometry-specific convenience layers
    include("Shell/Shell.jl")
    include("Ball/Ball.jl")

    # Import Ball functions into GeoDynamo namespace for direct use
    using .GeoDynamoBall: enforce_ball_vector_regularity!, apply_ball_temperature_regularity!,
                           apply_ball_composition_regularity!, ball_physical_to_spectral!,
                           ball_vector_analysis!, create_ball_radial_domain,
                           create_ball_velocity_fields, create_ball_magnetic_fields,
                           create_ball_temperature_field, create_ball_composition_field

    # Import Shell functions into GeoDynamo namespace for direct use
    using .GeoDynamoShell: create_shell_radial_domain, create_shell_velocity_fields,
                            create_shell_magnetic_fields, create_shell_temperature_field,
                            create_shell_composition_field

    # Re-export Ball functions for convenience
    export enforce_ball_vector_regularity!, apply_ball_temperature_regularity!,
           apply_ball_composition_regularity!, ball_physical_to_spectral!,
           ball_vector_analysis!, create_ball_radial_domain,
           create_ball_velocity_fields, create_ball_magnetic_fields,
           create_ball_temperature_field, create_ball_composition_field

    # Re-export Shell functions for convenience
    export create_shell_radial_domain, create_shell_velocity_fields,
           create_shell_magnetic_fields, create_shell_temperature_field,
           create_shell_composition_field

    # Expose combiner APIs under GeoDynamo namespace
    export FieldCombiner, CombinerConfig, create_combiner_config
    export combine_distributed_time, list_available_times
    export combine_time_series, save_combined_time_series
    export save_combined_fields

    # Initialize parameters when module is loaded
    function __init__()
        try
            # Load MPI at runtime if not already loaded
            # This is needed for SHTnsKit parallel extensions to work properly
            # Only import core MPI symbols that exist in all versions
            if !isdefined(Main, :MPI)
                try
                    @eval using MPI: Allgather, Allreduce, Allreduce!, Barrier, COMM_WORLD, Cart_shift, Comm, Comm_rank, Comm_size, Finalize, Gather, Init, Initialized, Isend, MAX, MIN, Request, SUM, Waitall, Wtime, bcast
                    @debug "GeoDynamo.jl loaded MPI at runtime"
                catch mpi_e
                    @debug "Could not load MPI (continuing without MPI support): $mpi_e"
                end
            else
                try
                    @eval using MPI: Allgather, Allreduce, Allreduce!, Barrier, COMM_WORLD, Cart_shift, Comm, Comm_rank, Comm_size, Finalize, Gather, Init, Initialized, Isend, MAX, MIN, Request, SUM, Waitall, Wtime, bcast
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
                set_parameters!(GeoDynamoParameters())
                @info "Using default parameters"
            catch param_e
                @warn "Failed to set default parameters: $param_e"
            end
        end
    end

end
