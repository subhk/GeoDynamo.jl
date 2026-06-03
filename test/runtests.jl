using Test
include(joinpath(@__DIR__, "precompile_syntax.jl"))
include(joinpath(@__DIR__, "loop_macro_static_checks.jl"))
include(joinpath(@__DIR__, "pencil_decomposition_static_checks.jl"))
include(joinpath(@__DIR__, "io_writer_static_checks.jl"))
include(joinpath(@__DIR__, "velocity_boundary_static_checks.jl"))
include(joinpath(@__DIR__, "magnetic_boundary_static_checks.jl"))
include(joinpath(@__DIR__, "scalar_boundary_shared_static_checks.jl"))
include(joinpath(@__DIR__, "temperature_boundary_static_checks.jl"))
include(joinpath(@__DIR__, "composition_boundary_static_checks.jl"))
using GeoDynamo

const TEST_DIR = @__DIR__
const MPI_FINALIZE_KEY = "GEODYNAMO_TEST_MPI_FINALIZE"

@testset "GeoDynamo.jl" begin
    @testset "Package Loading" begin
        @test isdefined(GeoDynamo, :SHTnsKitConfig)
        @test isdefined(GeoDynamo, :SolverState)
        @test Base.isexported(GeoDynamo, :SolverState)
        @test !Base.isexported(GeoDynamo, :SimulationState)
        @test isdefined(GeoDynamo, :initialize_simulation)
        @test isdefined(GeoDynamo, :run_simulation!)
        @test isdefined(GeoDynamo, :initialize_fields!)
        @test isdefined(GeoDynamo, :extract_all_fields)
        @test isdefined(GeoDynamo, :create_enhanced_metadata)
        @test !isdefined(GeoDynamo, :GeoDynamoParameters)
        @test isdefined(GeoDynamo, :bcs)
        @test isdefined(GeoDynamo, :InitialConditions)
        @test isdefined(GeoDynamo, :GeoDynamoShell)
        @test isdefined(GeoDynamo, :GeoDynamoBall)
        # extras/ removed from core — guard against accidental re-export
        for sym in (:SpectralToPhysicalConverter, :FieldCombiner, :CombinerConfig,
            :create_spectral_converter, :create_combiner_config,
            :combine_distributed_time, :combine_time_series,
            :list_available_times, :save_combined_time_series, :save_combined_fields)
            @test !Base.isexported(GeoDynamo, sym)
            @test !isdefined(GeoDynamo, sym)
        end
        # performance/tools.jl deleted — guard against accidental re-export
        for sym in (:AdvancedThreadManager, :ThreadingAccelerator, :SIMDOptimizer,
            :TaskGraph, :MemoryOptimizer, :AsyncCommManager,
            :DynamicLoadBalancer, :ParallelIOOptimizer, :PerformanceMonitor,
            :CPUParallelizer, :MasterParallelizer,
            :create_advanced_thread_manager, :create_threading_accelerator,
            :create_simd_optimizer, :create_task_graph, :create_memory_optimizer,
            :create_async_comm_manager, :create_dynamic_load_balancer,
            :create_parallel_io_optimizer, :create_performance_monitor,
            :create_cpu_parallelizer, :create_master_parallelizer,
            :add_task!, :execute_task_graph!, :adaptive_rebalance!,
            :run_parallel_operations!)
            @test !Base.isexported(GeoDynamo, sym)
            @test !isdefined(GeoDynamo, sym)
        end
    end

    @testset "Basic Types" begin
        @test isdefined(GeoDynamo, :SHTnsSpecField)
        @test isdefined(GeoDynamo, :SHTnsPhysField)
        @test isdefined(GeoDynamo, :RadialDomain)

        # Test parameter system
        params = GeoDynamo.SolverParameters()
        @test params !== nothing
    end

    @testset "Submodules" begin
        @test isdefined(GeoDynamo.bcs, :FieldType)
        @test isdefined(GeoDynamo.bcs, :BoundaryLocation)
        @test isdefined(GeoDynamo.bcs, :BoundaryType)
    end

    @testset "No global parameter pull in library code" begin
        # Regression guard: active_parameters / get_parameters must not appear
        # in kernel/physics code. core/parameters.jl is the canonical home and
        # solver/parameters.jl is the file-backed bridge; both are excluded.
        src_dir = joinpath(@__DIR__, "..", "src")
        jl_files = String[]
        for (root, _, files) in walkdir(src_dir)
            for f in files
                endswith(f, ".jl") && push!(jl_files, joinpath(root, f))
            end
        end
        excluded = (joinpath("core", "parameters.jl"), joinpath("solver", "parameters.jl"))
        banned = ("active_parameters", "get_parameters")
        violations = String[]
        for path in jl_files
            any(contains(path, ex) for ex in excluded) && continue
            for (lineno, line) in enumerate(eachline(path))
                if any(contains(line, b) for b in banned)
                    push!(violations, "$path:$lineno: $line")
                end
            end
        end
        for v in violations
            @warn "Banned symbol found in kernel code" violation=v
        end
        @test isempty(violations)
    end

    @testset "Parameter symbols not in public exports" begin
        @test !Base.isexported(GeoDynamo, :get_parameters)
        @test !Base.isexported(GeoDynamo, :active_parameters)
        @test !Base.isexported(GeoDynamo, :set_parameters!)
        @test !Base.isexported(GeoDynamo, :initialize_parameters)
    end
end

additional_tests = (
    # Pure unit tests (no MPI)
    "banded_operators.jl",
    "banded_operations.jl",
    "index_mapping.jl",
    "output_scheduling.jl",
    "boundary_types.jl",
    "boundary_utilities.jl",
    "boundary_file_io.jl",
    "erk2_matrix_functions.jl",
    "numerics_phi_functions.jl",
    "process_grid_extended.jl",
    "krylov_exp.jl",
    "safe_parse.jl",
    "derived_parameters.jl",
    "stability_regressions.jl",
    "timestep_state.jl",
    "examples_audit.jl",
    "examples_smoke.jl",
    # Validation
    "parameter_validation.jl",
    "validation_edge_cases.jl",
    "parameter_io.jl",
    "nan_detection.jl",
    # New coverage tests (pure unit tests)
    "parameters_extended.jl",
    "output_writer_extended.jl",
    "tolerance_and_timestep.jl",
    "topography_data.jl",
    "gpu_backend.jl",
    "user_api.jl",
    "oceananigans_api.jl",
    # MPI-aware tests
    "radial_domain.jl",
    "field_containers.jl",
    "temperature_field.jl",
    "transform_collective_shortcuts.jl",
    "initial_conditions.jl",
    "composition_magnetic_fields.jl",
    "allocation_runtime_checks.jl",
    "magnetic_boundary_numerical.jl",
    "velocity_boundary_numerical.jl",
    "temperature_boundary_numerical.jl",
    "composition_boundary_numerical.jl",
    "mpi_parallel_invariants.jl",
    "mpi_threaded_step_smoke.jl",
    "mpi_distributed_write_roundtrip.jl",
    "mpi_r_theta_restart_roundtrip.jl",
    "derivative_accuracy.jl",
    "shell_geometry.jl",
    "programmatic_boundaries.jl",
    "erk2.jl",
    "erk2_cache_builders.jl",
    "cnab2_rhs_distributed_equivalence.jl",
    "shell_boundaries.jl",
    "shtnskit_roundtrip.jl",
    "theta_dist_adapter.jl",
    "theta_dist_transform.jl",
    "spectral_operators.jl",
    "ball_roundtrip.jl",
    "ball_finiteness.jl",
    "stefan_condition_sanity.jl",
    "stefan_condition_shift_sanity.jl",
    "check_gaunt_tensors.jl",
    "topography_coupling.jl",
    # Previously-orphaned coverage now wired in (MPI-aware unit/numerical tests).
    # NOTE: poloidal_solenoidality.jl stays unwired on purpose — it is a @test_broken
    # spec for a not-yet-implemented Mie-consistent poloidal reconstruction (see its
    # header + memory) and currently errors in setup.
    "analytical_blob_radial.jl",
    "composition_analytical_ic.jl",
    "composition_ic_bc_consistency.jl",
    "temperature_boundary_normalization.jl",
    "temperature_ic_normalization.jl",
    "nusselt_and_analytical_ic.jl",
    "solver_diagnostics.jl",
    "magnetic_conducting_inner_core.jl",
    "magnetic_conducting_matrix_rows.jl",
    "magnetic_inner_core_admittance.jl",
    "magnetic_inner_core_history.jl",
    "magnetic_inner_core_reconstruct.jl",
    # Integration tests
    "output_writer.jl",
    "netcdf_write_roundtrip.jl",
    "io_restart_roundtrip.jl",
    "io_subsystem.jl",
    "integration_simulation.jl",
    # r×θ decomposition tests (Phase 2)
    "r_theta_grid.jl",
    "r_theta_transpose.jl",
    "r_theta_equivalence.jl",
    # Legacy scalar transform migration (Phase 2 follow-up)
    "legacy_scalar_rtheta.jl",
    # Phase-3 DistTransposePlan gate (Task 5)
    # Runs as a np=1 smoke test inside the suite (Alm↔spec_solve roundtrip,
    # scalar+vector dist_* roundtrip, partition-neutrality).  Multi-rank gate
    # (2×2/2×1 p3_transpose + mpi_parallel_invariants) runs via the MPI shell runner.
    "p3_transpose.jl",
)

previous_setting = haskey(ENV, MPI_FINALIZE_KEY) ? ENV[MPI_FINALIZE_KEY] : nothing
ENV[MPI_FINALIZE_KEY] = "false"

try
    @testset "Extended GeoDynamo tests" begin
        for file in additional_tests
            include(joinpath(TEST_DIR, file))
        end
    end
finally
    if previous_setting === nothing
        delete!(ENV, MPI_FINALIZE_KEY)
    else
        ENV[MPI_FINALIZE_KEY] = previous_setting
    end
end

println("✓ GeoDynamo test suite completed")
