module Geodynamo

    using LinearAlgebra
    using SparseArrays
    using SHTnsKit   # Load SHTnsKit before MPI to avoid eager extension load during precompile
    using MPI
    using PencilArrays
    using PencilFFTs
    using HDF5
    using StaticArrays
    using NCDatasets

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

    # exports pencil_decomps.jl
    export get_comm, get_rank, get_nprocs
    export create_pencil_topology, create_transpose_plans
    export transpose_with_timer!, print_transpose_statistics
    export analyze_load_balance, estimate_memory_usage
    export create_pencil_array, synchronize_halos!
    export print_pencil_info, print_pencil_axes, optimize_communication_order
    export ENABLE_TIMING


    # exports field.jl
    export SHTnsSpectralField, SHTnsPhysicalField, SHTnsVectorField, SHTnsTorPolField
    export RadialDomain, create_shtns_spectral_field, create_shtns_physical_field
    export create_shtns_vector_field, create_radial_domain
    export get_local_range, get_local_indices, local_data_size, get_local_data

    # Legacy shtns_transforms.jl (deprecated - use SHTnsKit instead)
    # Legacy SHTns exports removed in SHTnsKit migration

    # exports linear_algebra.jl
    export BandedMatrix, create_derivative_matrix, create_radial_laplacian
    export apply_derivative_matrix!, apply_banded_matrix!

    # exports timestep.jl
    export TimestepState, SHTnsImplicitMatrices, create_shtns_timestepping_matrices
    export apply_explicit_operator!, solve_implicit_step!, compute_timestep_error

    # exports velocity.jl
    export SHTnsVelocityFields, create_shtns_velocity_fields
    export VelocityWorkspace, create_velocity_workspace, set_velocity_workspace!
    export compute_velocity_nonlinear!, compute_vorticity_spectral_full!
    export compute_kinetic_energy, compute_reynolds_stress
    export zero_velocity_work_arrays!
    export apply_velocity_boundary_conditions!, add_thermal_buoyancy_force!
    export add_buoyancy_force!, add_lorentz_force!, validate_velocity_configuration

    # exports magnetic.jl
    export SHTnsMagneticFields, create_shtns_magnetic_fields, compute_magnetic_nonlinear!
    export compute_current_density_spectral!

    # exports thermal.jl
    export SHTnsTemperatureField, create_shtns_temperature_field
    export compute_temperature_nonlinear!
    export compute_nusselt_number, compute_thermal_energy
    export compute_surface_flux, get_temperature_statistics
    export zero_temperature_work_arrays!
    export set_temperature_ic!, set_boundary_conditions!, set_internal_heating!
    export batch_transform_to_physical!, apply_temperature_boundary_conditions_spectral!

    # exports compositional.jl
    export SHTnsCompositionField, create_shtns_composition_field
    export compute_composition_nonlinear!
    export compute_composition_rms, compute_composition_energy
    export get_composition_statistics, zero_composition_work_arrays!
    export set_composition_ic!, set_composition_boundary_conditions!
    export apply_composition_boundary_conditions!, apply_composition_boundary_conditions_spectral!
    
    # exports BoundaryConditions module
    export AbstractBoundaryCondition
    export BoundaryLocation, INNER_BOUNDARY, OUTER_BOUNDARY
    export BoundaryType, DIRICHLET, NEUMANN, MIXED, ROBIN
    export FieldType, TEMPERATURE, COMPOSITION, VELOCITY, MAGNETIC
    export load_boundary_conditions!, update_time_dependent_boundaries!
    export validate_boundary_files, get_current_boundaries, print_boundary_summary
    export get_boundary_module_info

    # exports simulation.jl (single unified state)
    export SimulationState, initialize_simulation, run_simulation!

    # exports outputs_writer.jl
    export OutputConfig, FieldInfo, TimeTracker
    export default_config, output_config_from_parameters, resolve_output_precision
    export with_output_precision, with_independent_writes
    export create_time_tracker, should_output_now, should_restart_now
    export write_fields!, write_restart!, read_restart!
    export create_shtns_aware_output_config, validate_output_compatibility
    export get_time_series, find_files_in_time_range, cleanup_old_files

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
    export GeodynamoParameters, load_parameters, save_parameters, create_parameter_template
    export get_parameters, set_parameters!, initialize_parameters
    export @param  # Deprecated - use direct variable access instead

    # Include Parameters system first
    include("parameters.jl")

    # Include base modules in dependency order
    include("pencil_decomps.jl")
    include("shtnskit_transforms.jl")  # New SHTnsKit-based transforms (includes SHTnsKitConfig)
    include("fields.jl")
    include("shtnskit_field_functions.jl")  # Field-dependent transform functions
    include("linear_algebra.jl")
    # include("shtns_transforms.jl")  # Legacy - replaced by shtnskit_transforms.jl
    # include("shtns_config.jl")      # Legacy - replaced by SHTnsKit configurations
    include("scalar_field_common.jl")  # Include shared scalar field functions once
    include("BoundaryConditions/BoundaryConditions.jl")
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

    # Expose combiner APIs under Geodynamo namespace
    export FieldCombiner, CombinerConfig, create_combiner_config
    export combine_distributed_time, list_available_times
    export combine_time_series, save_combined_time_series
    export save_combined_fields

    # Initialize parameters when module is loaded
    function __init__()
        try
            # Load MPI at runtime if not already loaded
            # This is needed for SHTnsKit parallel extensions to work properly
            if !isdefined(Main, :MPI)
                try
                    @eval using MPI
                    @info "Geodynamo.jl loaded MPI at runtime"
                catch mpi_e
                    @warn "Could not load MPI (continuing without MPI support): $mpi_e"
                end
            else
                @info "Geodynamo.jl detected MPI already available"
            end

            initialize_parameters()
            @info "Geodynamo.jl initialized successfully"
        catch e
            @warn "Could not initialize Geodynamo.jl properly: $e"
            try
                set_parameters!(GeodynamoParameters())
                @info "Using default parameters"
            catch param_e
                @warn "Failed to set default parameters: $param_e"
            end
        end
    end

end
