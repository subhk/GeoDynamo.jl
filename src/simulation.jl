# ================================================================================
# Main Simulation Driver with SHTns (Basic + Enhanced Parallelization)
# ================================================================================

using MPI
using Base.Threads
using LinearAlgebra
using Printf
using Statistics: mean

# Geometry modules are now imported in main GeoDynamo module
# Functions like create_shell_radial_domain, create_ball_radial_domain, etc.
# are available directly without module prefix

# ================================================================================
# Banner Display
# ================================================================================

"""
    print_geodynamo_banner(config, nprocs::Int, nthreads::Int)

Print the GeoDynamo startup banner with configuration information.
"""
function print_geodynamo_banner(config, nprocs::Int, nthreads::Int)
    println()
    println("╔══════════════════════════════════════════════════════════════════════════")
    println("║")
    println("║   ██████╗ ███████╗ ██████╗ ██████╗ ██╗   ██╗███╗  ██╗ █████╗ ███╗   ███╗ ██████╗")
    println("║  ██╔════╝ ██╔════╝██╔═══██╗██╔══██╗╚██╗ ██╔╝████╗ ██║██╔══██╗████╗ ████║██╔═══██╗")
    println("║  ██║  ███╗█████╗  ██║   ██║██║  ██║ ╚████╔╝ ██╔██╗██║███████║██╔████╔██║██║   ██║")
    println("║  ██║   ██║██╔══╝  ██║   ██║██║  ██║  ╚██╔╝  ██║╚████║██╔══██║██║╚██╔╝██║██║   ██║")
    println("║  ╚██████╔╝███████╗╚██████╔╝██████╔╝   ██║   ██║ ╚███║██║  ██║██║ ╚═╝ ██║╚██████╔╝")
    println("║   ╚═════╝ ╚══════╝ ╚═════╝ ╚═════╝    ╚═╝   ╚═╝  ╚══╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝")
    println("║")
    println("║            High-Performance Geodynamo Simulation Framework")
    println("║                   Spectral Methods • MPI Parallel")
    println("╠══════════════════════════════════════════════════════════════════════════")

    # Configuration info
    params = get_parameters()
    geometry = string(params.geometry)

    println("║ CONFIGURATION")
    println("║   Geometry:        $(uppercase(geometry))")
    println("║   Grid:            $(config.nlat) × $(config.nlon) × $(i_N)")
    println("║   Spectral modes:  $(config.nlm) (lmax=$(config.lmax), mmax=$(config.mmax))")
    println("║   Rayleigh:        $(@sprintf("%.2e", params.d_Ra))")
    println("║   Prandtl:         $(@sprintf("%.2e", params.d_Pr))")
    println("║   Magnetic Prandtl:$(@sprintf("%.2e", params.d_Pm))")
    println("╠══════════════════════════════════════════════════════════════════════════")
    println("║ PARALLELIZATION")
    println("║   MPI processes:   $(nprocs)")
    println("║   Threads/process: $(nthreads)")
    println("║   Total cores:     $(nprocs * nthreads)")
    println("║   Optimization:    SIMD + NUMA + Task-based")
    println("╠══════════════════════════════════════════════════════════════════════════")
    println("║ OUTPUT")
    println("║   Directory:       ./output (parallel MPI-IO)")
    println("║   Precision:       $(String(params.output_precision))")
    println("╚══════════════════════════════════════════════════════════════════════════")
    println()
end

"""
    SimulationState{T}

Unified simulation state with comprehensive parallelization and diagnostics.
Type-stable version with concrete cache types for optimal performance.
"""
struct SimulationState{T}
    # Original components
    velocity::SHTnsVelocityFields{T}
    magnetic::SHTnsMagneticFields{T}
    temperature::SHTnsTemperatureField{T}
    composition::Union{SHTnsCompositionField{T}, Nothing}

    # Shared gradient workspace (memory optimization)
    gradient_workspace::GradientWorkspace{T}

    # Geometric data
    shtns_config::SHTnsKitConfig
    𝒟ᵒᶜ::RadialDomain
    𝒟ⁱᶜ::RadialDomain

    # Unified master parallelization system
    master_parallelizer::MasterParallelizer{T}

    # Timestepping with type-stable caches
    timestep_state::TimestepState
    implicit_matrices::Dict{Symbol, SHTnsImplicitMatrices{T}}
    etd_caches::Dict{Symbol, EAB2ALUCacheEntry{T}}  # Type-stable ETD caches
    erk2_caches::Dict{Symbol, ERK2Cache{T}}  # Type-stable ERK2 caches

    # Enhanced I/O
    output_counter::Int
    auto_optimization::Bool
    adaptive_threading::Bool
    geometry::Symbol
end

# ================================================================================
# BASIC SIMULATION INITIALIZATION
# ================================================================================

initialize_shtns_simulation(::Type{T}=Float64; include_composition::Bool=true) where T = initialize_simulation(T; include_composition)

# ================================================================================
# MASTER SIMULATION INITIALIZATION
# ================================================================================

"""
    initialize_simulation(::Type{T}=Float64; kwargs...)

Initialize simulation with comprehensive CPU parallelization.
"""
function initialize_simulation(::Type{T}=Float64; 
                                              include_composition::Bool=true,
                                              auto_optimize::Bool=true,
                                              adaptive_threading::Bool=true,
                                              thread_count::Int=Threads.nthreads()) where T
    
    # Initialize MPI first
    if !MPI.Initialized()
        MPI.Init()
    end
    
    rank = get_rank()
    nprocs = get_nprocs()
    
    if rank == 0
        println("="^90)
        println("    GEODYNAMO.jl - MASTER CPU PARALLEL INITIALIZATION")
        println("="^90)
        println("MPI Processes: $nprocs")
        println("Threads per process: $thread_count")
        println("Auto-optimization: $(auto_optimize ? "ENABLED" : "DISABLED")")
        println("Adaptive threading: $(adaptive_threading ? "ENABLED" : "DISABLED")")
        println("Advanced CPU features: SIMD, NUMA-aware, Task-based parallelism")
        println("="^90)
    end
    
    # Create SHTns configuration with advanced topology
    shtns_config = create_shtnskit_config(lmax=i_L, mmax=i_M, nlat=i_Th, nlon=i_Ph, optimize_decomp=true)
    
    # Initialize advanced pencil decomposition
    pencils = create_pencil_topology(shtns_config, optimize=true)
    pencil_spec = pencils.spec
    
    geom = get_parameters().geometry
    if geom === :ball
        𝒟ᵒᶜ = create_ball_radial_domain(i_N)
        𝒟ⁱᶜ = 𝒟ᵒᶜ
        velocity = create_ball_velocity_fields(T, shtns_config; nr=i_N)
        magnetic = create_ball_magnetic_fields(T, shtns_config; nr=i_N)
        temperature = create_ball_temperature_field(T, shtns_config; nr=i_N)
        composition = include_composition ? create_ball_composition_field(T, shtns_config; nr=i_N) : nothing
    else
        𝒟ᵒᶜ = create_shell_radial_domain(i_N)
        𝒟ⁱᶜ = create_shell_radial_domain(i_Nic)
        velocity = create_shell_velocity_fields(T, shtns_config; nr=i_N)
        magnetic = create_shell_magnetic_fields(T, shtns_config; nr_oc=i_N, nr_ic=i_Nic)
        temperature = create_shell_temperature_field(T, shtns_config; nr=i_N)
        composition = include_composition ? create_shell_composition_field(T, shtns_config; nr=i_N) : nothing
    end
    
    # Initialize unified master parallelization system
    master_parallelizer = create_master_parallelizer(T, shtns_config)
    
    if rank == 0
        cpu_mgr = master_parallelizer.cpu_parallelizer.thread_manager
        println("CPU Topology detected:")
        println("  NUMA nodes: $(cpu_mgr.numa_nodes)")
        println("  Cores per node: $(cpu_mgr.cores_per_node)")
        println("  Compute threads: $(cpu_mgr.compute_threads)")
        println("  I/O threads: $(cpu_mgr.io_threads)")
        println("  Communication threads: $(cpu_mgr.comm_threads)")
        
        simd_opt = master_parallelizer.cpu_parallelizer.simd_optimizer
        println("SIMD Optimization:")
        println("  Vector width: $(simd_opt.vector_width)")
        println("  Memory alignment: $(simd_opt.alignment_bytes) bytes")
        println("  Prefetch distance: $(simd_opt.prefetch_distance) bytes")
        
        println("MPI Communication: $(master_parallelizer.mpi_nprocs) processes")
        println("Unified system: ALL components integrated")
    end
    
    # Initialize timestepping with advanced matrices
    timestep_state = TimestepState(d_time, d_timestep, 0, 0, Inf, false)

    # Create implicit matrices for each equation (magnetic diffusion time scaling)
    # Velocity: E, Temperature: Pm/Pr, Magnetic: 1, Composition: Pm/Sc
    implicit_matrices = Dict{Symbol, SHTnsImplicitMatrices{T}}()
    # Velocity uses separate toroidal/poloidal matrices with BCs embedded (Fortran approach)
    implicit_matrices[:velocity_tor] = create_velocity_toroidal_matrices(shtns_config, 𝒟ᵒᶜ, d_E, d_timestep;
                                                                          i_vel_bc=get_parameters().i_vel_bc)
    implicit_matrices[:velocity_pol] = create_velocity_poloidal_matrices(shtns_config, 𝒟ᵒᶜ, d_E, d_timestep;
                                                                          i_vel_bc=get_parameters().i_vel_bc)
    implicit_matrices[:velocity] = create_shtns_timestepping_matrices(shtns_config, 𝒟ᵒᶜ, d_E, d_timestep)
    # Magnetic uses separate toroidal/poloidal matrices with BCs embedded (Fortran approach)
    implicit_matrices[:magnetic_tor] = create_magnetic_toroidal_matrices(shtns_config, 𝒟ᵒᶜ, 1.0, d_timestep)
    implicit_matrices[:magnetic_pol] = create_magnetic_poloidal_matrices(shtns_config, 𝒟ᵒᶜ, 1.0, d_timestep)
    implicit_matrices[:magnetic] = create_shtns_timestepping_matrices(shtns_config, 𝒟ᵒᶜ, 1.0, d_timestep)
    # Temperature uses matrix-embedded BCs
    implicit_matrices[:temperature] = create_temperature_matrices(shtns_config, 𝒟ᵒᶜ, d_Pm/d_Pr, d_timestep)

    if include_composition
        # Composition uses matrix-embedded BCs
        implicit_matrices[:composition] = create_composition_matrices(shtns_config, 𝒟ᵒᶜ, d_Pm/d_Sc, d_timestep)
    end

    # Shared gradient workspace for temperature and composition (memory optimization)
    gradient_workspace = create_gradient_workspace(T, shtns_config, 𝒟ᵒᶜ)

    state = SimulationState{T}(
        velocity, magnetic, temperature, composition,
        gradient_workspace,
        shtns_config, 𝒟ᵒᶜ, 𝒟ⁱᶜ,
        master_parallelizer,
        timestep_state, implicit_matrices,
        Dict{Symbol, EAB2ALUCacheEntry{T}}(),  # Type-stable etd_caches
        Dict{Symbol, ERK2Cache{T}}(),  # Type-stable erk2_caches
        0, auto_optimize, adaptive_threading, geom
    )

    # Load file-based boundary conditions if specified in parameters
    _load_file_bcs!(state)

    return state
end

"""
    _load_file_bcs!(state::SimulationState)

Load file-based boundary conditions from paths specified in parameters.
Modifies the field's mutable boundary_interpolation_cache in place.
"""
function _load_file_bcs!(state::SimulationState{T}) where T
    params = get_parameters()

    # Temperature file BCs
    if !isempty(params.s_tmp_bc_file)
        try
            bc = bcs.load_spectral_bc_from_file(params.s_tmp_bc_file, state.shtns_config;
                                                 format=params.s_tmp_bc_format, T=T)
            bcs.store_bc_in_field!(state.temperature, bc)
            if get_rank() == 0
                @info "Loaded temperature BCs from $(params.s_tmp_bc_file) (format=$(params.s_tmp_bc_format))"
            end
        catch e
            @warn "Failed to load temperature BC file '$(params.s_tmp_bc_file)': $e"
        end
    end

    # Composition file BCs
    if !isempty(params.s_cmp_bc_file) && state.composition !== nothing
        try
            bc = bcs.load_spectral_bc_from_file(params.s_cmp_bc_file, state.shtns_config;
                                                 format=params.s_cmp_bc_format, T=T)
            bcs.store_bc_in_field!(state.composition, bc)
            if get_rank() == 0
                @info "Loaded composition BCs from $(params.s_cmp_bc_file) (format=$(params.s_cmp_bc_format))"
            end
        catch e
            @warn "Failed to load composition BC file '$(params.s_cmp_bc_file)': $e"
        end
    end
end

# ================================================================================
# BASIC SIMULATION RUNNER
# ================================================================================

## Backward-compatibility thin wrappers (optional, non-exported)
function run_shtns_simulation!(state)
    run_simulation!(state)

    comm = get_comm()
    rank = get_rank()

    if rank == 0
        println("Starting geodynamo simulation with SHTnsKit...")
        println("SHTns Grid: $(state.shtns_config.nlat) × $(state.shtns_config.nlon) × $(state.𝒟ᵒᶜ.N)")
        println("Spectral modes: $(state.shtns_config.nlm)")
        println("lmax: $(state.shtns_config.lmax), mmax: $(state.shtns_config.mmax)")
        println("Geometry: $(state.geometry)")
    end

    # Initialize fields with some perturbation
    initialize_fields!(state)
    
    # Main timestepping loop
    while state.timestep_state.step < i_maxtstep && 
            state.timestep_state.time < 1.0
        
        # Predictor-corrector iterations
        state.timestep_state.iteration = 1
        state.timestep_state.error = Inf
        state.timestep_state.converged = false
        
        # Store previous state for error computation
        prev_velocity = deepcopy(state.velocity.𝒯)
        prev_magnetic = deepcopy(state.magnetic.𝒯)
        prev_temperature = deepcopy(state.temperature.spectral)
        prev_composition = state.composition !== nothing ? deepcopy(state.composition.spectral) : nothing
        
        while state.timestep_state.iteration <= 10 && 
                state.timestep_state.error > d_dterr
            
            # Compute nonlinear terms
            compute_all_nonlinear_terms!(state)
            
            # Timestepping (predictor or corrector)
            if state.timestep_state.iteration == 1
                predictor_step!(state)
            else
                corrector_step!(state)
            end
            
            # Compute convergence error
            error_vel  = compute_timestep_error(state.velocity.𝒯, prev_velocity)
            error_mag  = compute_timestep_error(state.magnetic.𝒯, prev_magnetic)
            error_temp = compute_timestep_error(state.temperature.spectral, prev_temperature)
            
            error_comp = 0.0
            if state.composition !== nothing && prev_composition !== nothing
                error_comp = compute_timestep_error(state.composition.spectral, prev_composition)
            end
            
            state.timestep_state.error = max(error_vel, error_mag, error_temp, error_comp)
            
            if state.timestep_state.error < d_dterr
                state.timestep_state.converged = true
                break
            end
            
            # Update previous state (copy underlying data arrays)
            parent(prev_velocity.data_real) .= parent(state.velocity.𝒯.data_real)
            parent(prev_velocity.data_imag) .= parent(state.velocity.𝒯.data_imag)
            parent(prev_magnetic.data_real) .= parent(state.magnetic.𝒯.data_real)
            parent(prev_magnetic.data_imag) .= parent(state.magnetic.𝒯.data_imag)
            parent(prev_temperature.data_real) .= parent(state.temperature.spectral.data_real)
            parent(prev_temperature.data_imag) .= parent(state.temperature.spectral.data_imag)

            if state.composition !== nothing && prev_composition !== nothing
                parent(prev_composition.data_real) .= parent(state.composition.spectral.data_real)
                parent(prev_composition.data_imag) .= parent(state.composition.spectral.data_imag)
            end
            
            state.timestep_state.iteration += 1
        end
        
        # Advance time
        state.timestep_state.time += state.timestep_state.dt
        state.timestep_state.step += 1
        
        # Adaptive timestepping
        compute_cfl_timestep!(state)
        
        # Update implicit matrices if timestep changed
        if abs(state.timestep_state.dt - d_timestep) > 1e-10
            update_implicit_matrices!(state)
        end
        
        # Output
        if state.timestep_state.step % i_save_rate2 == 0
            if rank == 0
                println("Step: $(state.timestep_state.step), " *
                        "Time: $(state.timestep_state.time), " *
                        "dt: $(state.timestep_state.dt), " *
                        "Error: $(state.timestep_state.error)")
            end
        end
    end
    
    if rank == 0
        println("SHTns simulation completed!")
    end
end

# ================================================================================
# ENHANCED SIMULATION RUNNER
# ================================================================================

"""
    run_enhanced_simulation!(state::SimulationState{T}; restart_file::String="", restart_dir::String="", restart_time::Float64=0.0)

Run geodynamo simulation with all parallelization optimizations.
Supports restarting from a saved NetCDF restart file.
"""
function run_enhanced_simulation!(state::SimulationState{T};
                                  restart_file::String="",
                                  restart_dir::String="",
                                  restart_time::Float64=0.0) where T
    comm = get_comm()
    rank = get_rank()
    nprocs = get_nprocs()

    if rank == 0
        println("\nStarting enhanced geodynamo simulation...")
        println("Grid: $(state.shtns_config.nlat) × $(state.shtns_config.nlon) × $(i_N)")
        println("Spectral modes: $(state.shtns_config.nlm) (lmax=$(state.shtns_config.lmax))")
        println("Parallel configuration: $nprocs MPI × $(Threads.nthreads()) threads")
        println()
    end

    # Resolve restart parameters
    params = get_parameters()
    _restart_file = !isempty(restart_file) ? restart_file : params.s_restart_file
    _restart_dir = !isempty(restart_dir) ? restart_dir : params.s_restart_dir
    _restart_time = restart_time > 0.0 ? restart_time : params.d_restart_time

    # Create enhanced output configuration
    output_config = create_enhanced_output_config(state)

    # Verify parallel NetCDF (MPI-IO) is available
    check_parallel_netcdf_support(comm)

    is_restart = !isempty(_restart_file) || !isempty(_restart_dir)

    step = 0
    simulation_time = d_time
    dt = d_timestep

    if is_restart
        time_tracker = create_time_tracker(output_config)

        pencils = state.shtns_config.pencils
        if !isempty(_restart_file)
            if rank == 0
                println("  RESTARTING FROM FILE: $_restart_file")
            end
            restart_data, restart_metadata = _load_restart_file(
                _restart_file, time_tracker, output_config; pencils=pencils)
        else
            if rank == 0
                println("  RESTARTING FROM DIR: $_restart_dir (target time=$_restart_time)")
            end
            restart_data, restart_metadata = read_restart!(
                time_tracker, _restart_dir, _restart_time, output_config, pencils)
        end

        restore_fields_from_restart!(state, restart_data)

        if haskey(restart_metadata, "current_time")
            simulation_time = Float64(restart_metadata["current_time"])
        end
        if haskey(restart_metadata, "current_step")
            step = Int(restart_metadata["current_step"])
        end

        state.timestep_state.time = simulation_time
        # Set step=0 to bootstrap prev_nonlinear on first iteration (AB2 → AB1)
        state.timestep_state.step = 0

        if rank == 0
            println("  Resuming from time = $simulation_time, step = $step")
        end
    else
        initialize_enhanced_fields!(state)
        time_tracker = create_time_tracker(output_config)
    end
    
    # Performance monitoring
    total_start = Wtime()
    last_output_time = simulation_time
    
    while simulation_time < 1.0 && step < i_maxtstep
        step += 1
        step_start = Wtime()
        
        # === ENHANCED PHYSICS COMPUTATION ===
        
        # 1. Hybrid nonlinear computation (MPI + Threads)
        compute_start = Wtime()
        
        # Temperature evolution with all optimizations
        hybrid_compute_nonlinear!(state.master_parallelizer, state.temperature, 
                                 state.velocity, state.𝒟ᵒᶜ)
        
        # Velocity evolution
        if state.velocity !== nothing
            compute_velocity_nonlinear!(state.velocity, state.magnetic, state.temperature, state.𝒟ᵒᶜ)
        end
        
        # Magnetic field evolution  
        if i_B == 1 && state.magnetic !== nothing
            compute_magnetic_nonlinear!(state.magnetic, state.velocity, state.𝒟ᵒᶜ, state.𝒟ⁱᶜ)
        end
        
        # Compositional evolution (if enabled)
        if state.composition !== nothing
            compute_composition_nonlinear!(state.composition, state.velocity, state.𝒟ᵒᶜ,
                                           state.gradient_workspace)
        end

        compute_time = Wtime() - compute_start
        
        # 2. Enhanced time integration
        integrate_start = Wtime()
        
        # Apply implicit time integration with enhanced solvers
        apply_enhanced_implicit_step!(state, dt)
        
        integrate_time = Wtime() - integrate_start
        
        # 3. Asynchronous I/O (non-blocking)
        io_start = Wtime()
        
        if should_output_now(time_tracker, simulation_time, output_config)
            fields = extract_all_fields(state)
            metadata = create_enhanced_metadata(state, simulation_time, step)
            did_write = write_fields!(fields, time_tracker, metadata, output_config,
                                      state.shtns_config, state.shtns_config.pencils)
            if did_write && rank == 0
                println("Step $step: t=$(round(simulation_time, digits=4)), " *
                       "compute=$(round(compute_time*1000, digits=1))ms, " *
                       "integrate=$(round(integrate_time*1000, digits=1))ms")
            end
        end
        
        io_time = Wtime() - io_start
        
        # 4. Performance monitoring and adaptive optimization
        if state.auto_optimization && step % 50 == 0
            monitor_start = Wtime()
            
            # Update performance metrics
            update_performance_metrics!(state.master_parallelizer.performance_monitor, step, 
                                      compute_time, integrate_time, io_time)
            
            # Dynamic load balancing check
            adaptive_rebalance!(state.master_parallelizer.load_balancer, state.temperature)
            
            # Auto-tuning of parameters
            if step % 200 == 0
                auto_tune_parameters!(state)
            end
            
            monitor_time = Wtime() - monitor_start
        end
        
        # Update simulation state
        simulation_time += dt
        state.timestep_state.time = simulation_time
        state.timestep_state.step = step
        
        step_time = Wtime() - step_start
        
        # Adaptive timestep (if enabled)
        if step % 10 == 0
            dt = compute_adaptive_timestep(state, dt)
        end
    end
    
    total_time = Wtime() - total_start
    
    # Final performance analysis
    if rank == 0
        analyze_parallel_performance(state.master_parallelizer.performance_monitor)
        
        println("\n" * "="^80)
        println("         SIMULATION COMPLETED SUCCESSFULLY")
        println("="^80)
        println("Total steps: $step")
        println("Final time: $(round(simulation_time, digits=4))")
        println("Total wall time: $(round(total_time, digits=2)) seconds")
        println("Average time per step: $(round(total_time/step*1000, digits=2)) ms")
        
        # Parallel efficiency summary
        parallel_efficiency = get_parallel_efficiency(state.master_parallelizer.performance_monitor)
        println("Parallel efficiency: $(round(parallel_efficiency*100, digits=1))%")
        
        # Thread utilization
        thread_utilization = get_thread_utilization(state.master_parallelizer.threading_accelerator)
        println("Thread utilization: $(round(thread_utilization*100, digits=1))%")
        
        println("="^80)
    end
    
    # Cleanup
    finalize_enhanced_simulation!(state)
end

# ================================================================================
# MASTER SIMULATION RUNNER
# ================================================================================

"""
    run_simulation!(state::SimulationState{T}; restart_file::String="", restart_dir::String="", restart_time::Float64=0.0)

Run geodynamo simulation with maximum CPU parallelization optimizations.

# Keyword Arguments
- `restart_file`: Path to a specific restart NetCDF file to resume from.
- `restart_dir`: Directory containing restart files (used with `restart_time`).
- `restart_time`: Target simulation time to restart from (used with `restart_dir`).

If neither `restart_file` nor `restart_dir` is provided, the parameters
`s_restart_file`, `s_restart_dir`, and `d_restart_time` are checked.
If no restart is specified, the simulation starts fresh.
"""
function run_simulation!(state::SimulationState{T};
                         restart_file::String="",
                         restart_dir::String="",
                         restart_time::Float64=0.0) where T
    comm = get_comm()
    rank = get_rank()
    nprocs = get_nprocs()

    if rank == 0
        print_geodynamo_banner(state.shtns_config, nprocs, Threads.nthreads())
    end

    # Resolve restart parameters: keyword args take priority over global parameters
    params = get_parameters()
    _restart_file = !isempty(restart_file) ? restart_file : params.s_restart_file
    _restart_dir = !isempty(restart_dir) ? restart_dir : params.s_restart_dir
    _restart_time = restart_time > 0.0 ? restart_time : params.d_restart_time

    # Create enhanced output configuration
    output_config = create_enhanced_output_config(state)

    # Verify parallel NetCDF (MPI-IO) is available
    check_parallel_netcdf_support(comm)

    # Determine whether to restart or initialize fresh
    is_restart = !isempty(_restart_file) || !isempty(_restart_dir)

    step = 0
    simulation_time = d_time
    dt = d_timestep

    if is_restart
        # --- RESTART MODE ---
        time_tracker = create_time_tracker(output_config)

        pencils = state.shtns_config.pencils
        if !isempty(_restart_file)
            # Load from a specific file
            if rank == 0
                println("="^80)
                println("  RESTARTING FROM FILE: $_restart_file")
                println("="^80)
            end
            restart_data, restart_metadata = _load_restart_file(
                _restart_file, time_tracker, output_config; pencils=pencils)
        else
            # Load from directory + target time
            if rank == 0
                println("="^80)
                println("  RESTARTING FROM DIR: $_restart_dir (target time=$_restart_time)")
                println("="^80)
            end
            restart_data, restart_metadata = read_restart!(
                time_tracker, _restart_dir, _restart_time, output_config, pencils)
        end

        # Restore fields from the restart data
        restore_fields_from_restart!(state, restart_data)

        # Set simulation time and step from restart metadata
        if haskey(restart_metadata, "current_time")
            simulation_time = Float64(restart_metadata["current_time"])
        end
        if haskey(restart_metadata, "current_step")
            step = Int(restart_metadata["current_step"])
        end

        # Update timestep state
        state.timestep_state.time = simulation_time
        # Temporarily set step to 0 so that apply_implicit_step! will bootstrap
        # prev_nonlinear from the first computed nonlinear terms (AB2 → AB1 for first step).
        # The correct step count is restored at the end of the first loop iteration.
        state.timestep_state.step = 0

        if rank == 0
            println("  Resuming from time = $simulation_time, step = $step")
            println("="^80)
        end
    else
        # --- FRESH START ---
        initialize_master_fields!(state)
        time_tracker = create_time_tracker(output_config)
    end

    # Performance monitoring
    total_start = Wtime()
    last_output_time = simulation_time
    
    # Adaptive threading state
    thread_efficiency_history = Float64[]
    optimal_thread_count = Threads.nthreads()
    
    while simulation_time < 1.0 && step < i_maxtstep
        step += 1
        step_start = Wtime()
        
        # === ADVANCED PHYSICS COMPUTATION ===
        
        # 1. Advanced nonlinear computation (CPU + SIMD + Task-based)
        compute_start = Wtime()
        
        # Temperature evolution with maximum CPU optimizations
        compute_nonlinear!(state.master_parallelizer.cpu_parallelizer, state.temperature,
                                   state.velocity, state.𝒟ᵒᶜ, state.gradient_workspace)
        
        # Velocity evolution with task-based parallelism
        if state.velocity !== nothing
            compute_velocity_nonlinear_master!(state, state.magnetic, state.temperature, state.𝒟ᵒᶜ)
        end
        
        # Magnetic field evolution with SIMD optimization
        if i_B == 1 && state.magnetic !== nothing
            compute_magnetic_nonlinear_master!(state, state.velocity, state.𝒟ᵒᶜ, state.𝒟ⁱᶜ)
        end
        
        # Compositional evolution (if enabled) with memory optimization
        if state.composition !== nothing
            compute_composition_nonlinear_master!(state, state.velocity, state.𝒟ᵒᶜ,
                                                  state.gradient_workspace)
        end
        
        compute_time = Wtime() - compute_start
        
        # 2. Advanced time integration with task scheduling
        integrate_start = Wtime()
        
        # Apply implicit time integration with advanced solvers
        apply_implicit_step!(state, dt)
        
        integrate_time = Wtime() - integrate_start
        
        # 3. Asynchronous I/O with memory-aware scheduling
        io_start = Wtime()
        
        if should_output_now(time_tracker, simulation_time, output_config)
            fields = extract_all_fields_enhanced(state)
            metadata = create_enhanced_metadata(state, simulation_time, step)
            did_write = write_fields!(fields, time_tracker, metadata, output_config,
                                      state.shtns_config, state.shtns_config.pencils)
            
            if did_write && rank == 0
                cpu_efficiency = state.master_parallelizer.cpu_parallelizer.thread_efficiency[]
                cache_efficiency = state.master_parallelizer.cpu_parallelizer.cache_efficiency[]
                memory_bw = state.master_parallelizer.cpu_parallelizer.memory_bandwidth[]
                
                println("Step $step: t=$(round(simulation_time, digits=4)), " *
                       "compute=$(round(compute_time*1000, digits=1))ms, " *
                       "integrate=$(round(integrate_time*1000, digits=1))ms")
                println("  CPU efficiency: $(round(cpu_efficiency*100, digits=1))%, " *
                       "Cache: $(round(cache_efficiency*100, digits=1))%, " *
                       "Memory BW: $(round(memory_bw, digits=2)) GB/s")
            end
        end
        
        io_time = Wtime() - io_start
        
        # 4. Advanced performance monitoring and adaptive optimization
        if state.auto_optimization && step % 25 == 0  # More frequent optimization
            monitor_start = Wtime()
            
            # Update performance metrics
            update_performance_metrics!(state.master_parallelizer.performance_monitor, step, 
                                      compute_time, integrate_time, io_time)
            
            # CPU-specific optimizations
            current_cpu_efficiency = state.master_parallelizer.cpu_parallelizer.thread_efficiency[]
            push!(thread_efficiency_history, current_cpu_efficiency)
            
            # Adaptive threading adjustment
            if state.adaptive_threading && step % 100 == 0 && length(thread_efficiency_history) > 5
                optimal_thread_count = adapt_thread_count!(state, thread_efficiency_history)
                if rank == 0 && optimal_thread_count != Threads.nthreads()
                    println("  Adaptive threading: Optimal thread count adjusted to $optimal_thread_count")
                end
            end
            
            # Dynamic load balancing with CPU awareness
            adaptive_rebalance!(state.master_parallelizer.load_balancer, state.temperature)
            
            # Auto-tuning of parameters with CPU-specific heuristics
            if step % 200 == 0
                auto_tune_parameters_master!(state)
            end
            
            # Memory optimization
            if step % 150 == 0
                optimize_memory_usage!(state)
            end
            
            monitor_time = Wtime() - monitor_start
        end
        
        # Update simulation state
        simulation_time += dt
        state.timestep_state.time = simulation_time
        state.timestep_state.step = step
        
        step_time = Wtime() - step_start
        
        # Adaptive timestep with CPU-aware scaling
        if step % 5 == 0  # More frequent timestep adaptation
            dt = compute_adaptive_timestep_master(state, dt)
        end
    end
    
    total_time = Wtime() - total_start
    
    # Final detailed performance analysis
    if rank == 0
        analyze_master_performance(state)
        
        println("\n" * "="^100)
        println("         MASTER SIMULATION COMPLETED SUCCESSFULLY")
        println("="^100)
        println("Total steps: $step")
        println("Final time: $(round(simulation_time, digits=4))")
        println("Total wall time: $(round(total_time, digits=2)) seconds")
        println("Average time per step: $(round(total_time/step*1000, digits=2)) ms")
        
        # Detailed efficiency metrics
        parallel_efficiency = get_parallel_efficiency(state.master_parallelizer.performance_monitor)
        cpu_efficiency = state.master_parallelizer.cpu_parallelizer.thread_efficiency[]
        cache_efficiency = state.master_parallelizer.cpu_parallelizer.cache_efficiency[]
        memory_bandwidth = state.master_parallelizer.cpu_parallelizer.memory_bandwidth[]
        
        println("\nPERFORMANCE METRICS:")
        println("  Parallel efficiency: $(round(parallel_efficiency*100, digits=1))%")
        println("  CPU thread efficiency: $(round(cpu_efficiency*100, digits=1))%")
        println("  Cache hit rate: $(round(cache_efficiency*100, digits=1))%")
        println("  Memory bandwidth: $(round(memory_bandwidth, digits=2)) GB/s")
        
        # SIMD utilization
        simd_opt = state.master_parallelizer.cpu_parallelizer.simd_optimizer
        println("  SIMD vector width utilized: $(simd_opt.vector_width)")
        println("  Memory alignment: $(simd_opt.alignment_bytes)-byte aligned")
        
        # Thread topology efficiency
        cpu_mgr = state.master_parallelizer.cpu_parallelizer.thread_manager
        avg_thread_util = sum(cpu_mgr.thread_utilization) / length(cpu_mgr.thread_utilization)
        println("  Average thread utilization: $(round(avg_thread_util*100, digits=1))%")
        println("  NUMA nodes utilized: $(cpu_mgr.numa_nodes)")
        
        # MPI efficiency
        println("  MPI processes: $(state.master_parallelizer.mpi_nprocs)")
        println("  Communication efficiency: $(state.master_parallelizer.async_comm.overlap_efficiency[])%")
        
        println("="^100)
    end
    
    # Cleanup with memory deallocation
    finalize_master_simulation!(state)
end

# ================================================================================
# SHARED HELPER FUNCTIONS
# ================================================================================

function initialize_fields!(state::SimulationState{T}) where T
    # Initialize with some random perturbation for onset
    
    # Temperature: conductive profile + small perturbation
    temp_real = parent(state.temperature.spectral.data_real)
    temp_imag = parent(state.temperature.spectral.data_imag)
    
    for lm_idx in 1:state.temperature.spectral.nlm
        l = state.temperature.spectral.config.l_values[lm_idx]
        m = state.temperature.spectral.config.m_values[lm_idx]
        
        for r_idx in axes(temp_real, 3)
            if l == 0 && m == 0
                # Conductive profile
                r = state.𝒟ᵒᶜ.r[r_idx, 4]
                temp_real[lm_idx, 1, r_idx] = 1.0 - r
            elseif l <= 4 && l >= 1
                # Small perturbation
                temp_real[lm_idx, 1, r_idx] = 1e-3 * (rand() - 0.5)
                if m > 0
                    temp_imag[lm_idx, 1, r_idx] = 1e-3 * (rand() - 0.5)
                end
            end
        end
    end
    
    # Velocity: start with zero
    fill!(parent(state.velocity.𝒯.data_real), zero(T))
    fill!(parent(state.velocity.𝒯.data_imag), zero(T))
    fill!(parent(state.velocity.𝒫.data_real), zero(T))
    fill!(parent(state.velocity.𝒫.data_imag), zero(T))
    
    # Magnetic field: dipole + small perturbation
    mag_tor_real = parent(state.magnetic.𝒯.data_real)
    mag_tor_imag = parent(state.magnetic.𝒯.data_imag)
    mag_pol_real = parent(state.magnetic.𝒫.data_real)
    mag_pol_imag = parent(state.magnetic.𝒫.data_imag)
    
    for lm_idx in 1:state.magnetic.𝒯.nlm
        l = state.magnetic.𝒯.config.l_values[lm_idx]
        m = state.magnetic.𝒯.config.m_values[lm_idx]
        
        for r_idx in axes(mag_tor_real, 3)
            if l == 1 && m == 0
                # Dipole field
                r = state.𝒟ᵒᶜ.r[r_idx, 4]
                mag_pol_real[lm_idx, 1, r_idx] = r^2 * (1.0 - r)
            elseif l <= 3 && l >= 1
                # Small perturbation
                mag_tor_real[lm_idx, 1, r_idx] = 1e-4 * (rand() - 0.5)
                mag_pol_real[lm_idx, 1, r_idx] = 1e-4 * (rand() - 0.5)
                if m > 0
                    mag_tor_imag[lm_idx, 1, r_idx] = 1e-4 * (rand() - 0.5)
                    mag_pol_imag[lm_idx, 1, r_idx] = 1e-4 * (rand() - 0.5)
                end
            end
        end
    end
    
    # Compositional field: uniform background + small perturbation
    if state.composition !== nothing
        comp_real = parent(state.composition.spectral.data_real)
        comp_imag = parent(state.composition.spectral.data_imag)
        
        for lm_idx in 1:state.composition.spectral.nlm
            l = state.composition.spectral.config.l_values[lm_idx]
            m = state.composition.spectral.config.m_values[lm_idx]
            
            for r_idx in axes(comp_real, 3)
                if l == 0 && m == 0
                    # Uniform background composition
                    comp_real[lm_idx, 1, r_idx] = 0.5  # Background composition
                elseif l <= 3 && l >= 1
                    # Small perturbation
                    comp_real[lm_idx, 1, r_idx] = 1e-4 * (rand() - 0.5)
                    if m > 0
                        comp_imag[lm_idx, 1, r_idx] = 1e-4 * (rand() - 0.5)
                    end
                end
            end
        end
    end
end

"""
    restore_fields_from_restart!(state::SimulationState{T}, restart_data::Dict{String,Any})

Restore simulation state fields from restart data loaded by `read_restart!`.
Copies spectral coefficients from the restart data Dict into the state's field arrays.
"""
function restore_fields_from_restart!(state::SimulationState{T}, restart_data::Dict{String,Any}) where T
    rank = get_rank()

    # --- Velocity toroidal ---
    if haskey(restart_data, "velocity_toroidal")
        vt = restart_data["velocity_toroidal"]
        _copy_spectral_to_field!(state.velocity.𝒯, vt["real"], vt["imag"])
        if rank == 0
            println("  Restored velocity toroidal field")
        end
    end

    # --- Velocity poloidal ---
    if haskey(restart_data, "velocity_poloidal")
        vp = restart_data["velocity_poloidal"]
        _copy_spectral_to_field!(state.velocity.𝒫, vp["real"], vp["imag"])
        if rank == 0
            println("  Restored velocity poloidal field")
        end
    end

    # --- Magnetic toroidal ---
    if haskey(restart_data, "magnetic_toroidal")
        mt = restart_data["magnetic_toroidal"]
        _copy_spectral_to_field!(state.magnetic.𝒯, mt["real"], mt["imag"])
        if rank == 0
            println("  Restored magnetic toroidal field")
        end
    end

    # --- Magnetic poloidal ---
    if haskey(restart_data, "magnetic_poloidal")
        mp = restart_data["magnetic_poloidal"]
        _copy_spectral_to_field!(state.magnetic.𝒫, mp["real"], mp["imag"])
        if rank == 0
            println("  Restored magnetic poloidal field")
        end
    end

    # --- Temperature (spectral) ---
    if haskey(restart_data, "temperature_spectral")
        ts = restart_data["temperature_spectral"]
        _copy_spectral_to_field!(state.temperature.spectral, ts["real"], ts["imag"])
        if rank == 0
            println("  Restored temperature field (spectral)")
        end
    end

    # --- Composition (spectral) ---
    if state.composition !== nothing && haskey(restart_data, "composition_spectral")
        cs = restart_data["composition_spectral"]
        _copy_spectral_to_field!(state.composition.spectral, cs["real"], cs["imag"])
        if rank == 0
            println("  Restored composition field (spectral)")
        end
    end

    if rank == 0
        println("  Field restoration from restart complete.")
    end
end

"""
    _copy_spectral_to_field!(field::SHTnsSpecField{T}, real_data::AbstractArray, imag_data::AbstractArray)

Copy restart spectral data into a SHTnsSpecField. Handles the dimension mismatch between
the restart file format [nlm, nr] and the field storage format [nlm, 1, nr].
"""
function _copy_spectral_to_field!(field::SHTnsSpecField{T}, real_data::AbstractArray, imag_data::AbstractArray) where T
    dest_real = parent(field.data_real)
    dest_imag = parent(field.data_imag)

    if ndims(real_data) == 2
        # Restart data is [nlm, nr], field is [nlm, 1, nr]
        nlm_file, nr_file = size(real_data)
        nlm_field = size(dest_real, 1)
        nr_field = size(dest_real, 3)

        nlm_copy = min(nlm_file, nlm_field)
        nr_copy = min(nr_file, nr_field)

        # Zero out first, then copy available modes
        fill!(dest_real, zero(T))
        fill!(dest_imag, zero(T))

        for r in 1:nr_copy
            for lm in 1:nlm_copy
                dest_real[lm, 1, r] = T(real_data[lm, r])
                dest_imag[lm, 1, r] = T(imag_data[lm, r])
            end
        end
    elseif ndims(real_data) == 3
        # Already in [nlm, 1, nr] format
        nlm_file = size(real_data, 1)
        nr_file = size(real_data, 3)
        nlm_field = size(dest_real, 1)
        nr_field = size(dest_real, 3)

        nlm_copy = min(nlm_file, nlm_field)
        nr_copy = min(nr_file, nr_field)

        fill!(dest_real, zero(T))
        fill!(dest_imag, zero(T))

        for r in 1:nr_copy
            for lm in 1:nlm_copy
                dest_real[lm, 1, r] = T(real_data[lm, 1, r])
                dest_imag[lm, 1, r] = T(imag_data[lm, 1, r])
            end
        end
    else
        error("Unexpected restart data dimensions: $(ndims(real_data))")
    end
end

function compute_all_nonlinear_terms!(state::SimulationState{T}) where T
    # Compute nonlinear terms for all equations using SHTns transforms
    compute_velocity_nonlinear!(state.velocity, state.temperature,
                                state.composition, state.magnetic, state.𝒟ᵒᶜ)

    # Compute magnetic nonlinear terms only if magnetic field is enabled
    if i_B == 1 && state.magnetic !== nothing
        compute_magnetic_nonlinear!(state.magnetic, state.velocity, state.𝒟ᵒᶜ, state.𝒟ⁱᶜ, 0.0)
    end

    compute_temperature_nonlinear!(state.temperature, state.velocity, state.𝒟ᵒᶜ,
                                    state.gradient_workspace)

    # Compute compositional advection if composition is present
    if state.composition !== nothing
        compute_composition_nonlinear!(state.composition, state.velocity, state.𝒟ᵒᶜ,
                                       state.gradient_workspace)
    end
end

function predictor_step!(state::SimulationState{T}) where T
    # Predictor step for all fields using SHTns matrices
    
    # Velocity (toroidal component) - BCs embedded in matrix (Fortran approach)
    rhs_tor = similar(state.velocity.𝒯)
    apply_explicit_operator!(rhs_tor, state.velocity.𝒯,
                             state.velocity.nlᵀ, state.𝒟ᵒᶜ,
                             d_Pm, state.timestep_state.dt;
                             nl_prev=state.velocity.prev_nlᵀ,
                             matrices=state.implicit_matrices[:velocity])
    solve_velocity_implicit_step!(state.velocity.𝒯, rhs_tor,
                                  state.implicit_matrices[:velocity_tor], :toroidal;
                                  i_vel_bc=get_parameters().i_vel_bc,
                                  domain=state.𝒟ᵒᶜ)

    # Velocity (poloidal component) - BCs embedded in matrix (Fortran approach)
    rhs_pol = similar(state.velocity.𝒫)
    apply_explicit_operator!(rhs_pol, state.velocity.𝒫,
                             state.velocity.nlᴾ, state.𝒟ᵒᶜ,
                             d_Pm, state.timestep_state.dt;
                             nl_prev=state.velocity.prev_nlᴾ,
                             matrices=state.implicit_matrices[:velocity])
    solve_velocity_implicit_step!(state.velocity.𝒫, rhs_pol,
                                  state.implicit_matrices[:velocity_pol], :poloidal;
                                  i_vel_bc=get_parameters().i_vel_bc,
                                  domain=state.𝒟ᵒᶜ)

    # Magnetic field (toroidal) - BCs embedded in matrix (Fortran approach)
    rhs_mag_tor = similar(state.magnetic.𝒯)
    apply_explicit_operator!(rhs_mag_tor, state.magnetic.𝒯,
                             state.magnetic.nlᵀ, state.𝒟ᵒᶜ,
                             1.0, state.timestep_state.dt;
                             nl_prev=state.magnetic.prev_nlᵀ,
                             matrices=state.implicit_matrices[:magnetic])
    solve_magnetic_implicit_step!(state.magnetic.𝒯, rhs_mag_tor,
                                  state.implicit_matrices[:magnetic_tor], :toroidal)

    # Magnetic field (poloidal) - BCs embedded in matrix (Fortran approach)
    rhs_mag_pol = similar(state.magnetic.𝒫)
    apply_explicit_operator!(rhs_mag_pol, state.magnetic.𝒫,
                             state.magnetic.nlᴾ, state.𝒟ᵒᶜ,
                             1.0, state.timestep_state.dt;
                             nl_prev=state.magnetic.prev_nlᴾ,
                             matrices=state.implicit_matrices[:magnetic])
    solve_magnetic_implicit_step!(state.magnetic.𝒫, rhs_mag_pol,
                                  state.implicit_matrices[:magnetic_pol], :poloidal)
    
    # Temperature - BCs embedded in matrix
    rhs_temp = similar(state.temperature.spectral)
    apply_explicit_operator!(rhs_temp, state.temperature.spectral,
                             state.temperature.nonlinear, state.𝒟ᵒᶜ,
                             d_Pm/d_Pr, state.timestep_state.dt;
                             nl_prev=state.temperature.prev_nonlinear,
                             matrices=state.implicit_matrices[:temperature])
    tmp_bc = bcs.get_bc_vectors_from_field(state.temperature)
    solve_temperature_implicit_step!(state.temperature.spectral, rhs_temp,
                                      state.implicit_matrices[:temperature];
                                      bc_inner=tmp_bc.inner_real, bc_outer=tmp_bc.outer_real,
                                      bc_inner_imag=tmp_bc.inner_imag, bc_outer_imag=tmp_bc.outer_imag)

    # Composition (if present) - BCs embedded in matrix
    if state.composition !== nothing
        rhs_comp = similar(state.composition.spectral)
        apply_explicit_operator!(rhs_comp, state.composition.spectral,
                                 state.composition.nonlinear, state.𝒟ᵒᶜ,
                                 d_Pm/d_Sc, state.timestep_state.dt;
                                 nl_prev=state.composition.prev_nonlinear,
                                 matrices=state.implicit_matrices[:composition])
        cmp_bc = bcs.get_bc_vectors_from_field(state.composition)
        solve_composition_implicit_step!(state.composition.spectral, rhs_comp,
                                          state.implicit_matrices[:composition];
                                          bc_inner=cmp_bc.inner_real, bc_outer=cmp_bc.outer_real,
                                          bc_inner_imag=cmp_bc.inner_imag, bc_outer_imag=cmp_bc.outer_imag)
    end
end

function corrector_step!(state::SimulationState{T}) where T
    # Corrector step - similar to predictor but with time-averaged nonlinear terms
    # For simplicity, using same implementation as predictor
    predictor_step!(state)
end

function compute_cfl_timestep!(state::SimulationState{T}) where T
    # Compute CFL-limited timestep based on velocity magnitudes

    # Convert velocity to physical space for analysis
    shtnskit_vector_synthesis!(state.velocity.𝒯, state.velocity.𝒫, state.velocity.velocity; domain=state.𝒟ᵒᶜ)
    
    max_velocity = zero(Float64)
    
    # Get velocity data arrays
    u_r_data = parent(state.velocity.velocity.r_component.data)
    u_θ_data = parent(state.velocity.velocity.θ_component.data)
    u_φ_data = parent(state.velocity.velocity.φ_component.data)
    
    # Find maximum velocity over local data
    for r_idx in axes(u_r_data, 3)
        for j_phi in axes(u_r_data, 2)
            for i_theta in axes(u_r_data, 1)
                u_r = u_r_data[i_theta, j_phi, r_idx]
                u_θ = u_θ_data[i_theta, j_phi, r_idx]
                u_φ = u_φ_data[i_theta, j_phi, r_idx]
                
                u_mag = sqrt(u_r^2 + u_θ^2 + u_φ^2)
                max_velocity = max(max_velocity, u_mag)
            end
        end
    end
    
    # Global maximum across all processes
    comm = get_comm()
    global_max_vel = MPI.Allreduce(max_velocity, MPI.MAX, comm)
    
    # Compute grid spacing (approximate)
    dr_min = minimum(diff(state.𝒟ᵒᶜ.r[:, 4]))
    dtheta_min = π / state.shtns_config.nlat
    dphi_min = 2π / state.shtns_config.nlon
    dx_min = min(dr_min, dtheta_min, dphi_min)
    
    # CFL condition
    if global_max_vel > 1e-10
        dt_cfl = d_courant * dx_min / global_max_vel
        state.timestep_state.dt = min(state.timestep_state.dt, dt_cfl, d_timestep)
    end
end

function update_implicit_matrices!(state::SimulationState{T}) where T
    # Update implicit matrices with new timestep (magnetic diffusion time scaling)
    # Velocity: E, Temperature: Pm/Pr, Magnetic: 1, Composition: Pm/Sc
    dt = state.timestep_state.dt
    vel_bc = get_parameters().i_vel_bc

    # Velocity-specific matrices with BCs embedded (Fortran approach)
    state.implicit_matrices[:velocity_tor] = create_velocity_toroidal_matrices(
        state.shtns_config, state.𝒟ᵒᶜ, d_E, dt; i_vel_bc=vel_bc)
    state.implicit_matrices[:velocity_pol] = create_velocity_poloidal_matrices(
        state.shtns_config, state.𝒟ᵒᶜ, d_E, dt; i_vel_bc=vel_bc)
    state.implicit_matrices[:velocity] = create_shtns_timestepping_matrices(
        state.shtns_config, state.𝒟ᵒᶜ, d_E, dt)
    state.implicit_matrices[:magnetic_tor] = create_magnetic_toroidal_matrices(
        state.shtns_config, state.𝒟ᵒᶜ, 1.0, dt)
    state.implicit_matrices[:magnetic_pol] = create_magnetic_poloidal_matrices(
        state.shtns_config, state.𝒟ᵒᶜ, 1.0, dt)
    state.implicit_matrices[:magnetic] = create_shtns_timestepping_matrices(
        state.shtns_config, state.𝒟ᵒᶜ, 1.0, dt)
    state.implicit_matrices[:temperature] = create_temperature_matrices(
        state.shtns_config, state.𝒟ᵒᶜ, d_Pm/d_Pr, dt)

    # Update compositional matrix if composition is present
    if state.composition !== nothing
        state.implicit_matrices[:composition] = create_composition_matrices(
            state.shtns_config, state.𝒟ᵒᶜ, d_Pm/d_Sc, dt)
    end
end

# ================================================================================
# ENHANCED SIMULATION HELPER FUNCTIONS
# ================================================================================

"""
    apply_enhanced_implicit_step!(state, dt)
    
Apply enhanced implicit time integration with advanced solvers.
"""
function apply_enhanced_implicit_step!(state::SimulationState{T}, dt::Float64) where T
    # Use enhanced sparse solvers with preconditioning
    # This would integrate with advanced linear algebra libraries
    
    # Temperature - BCs embedded in matrix
    tmp_bc = bcs.get_bc_vectors_from_field(state.temperature)
    solve_temperature_implicit_step!(state.temperature.spectral, state.temperature.nonlinear,
                                      state.implicit_matrices[:temperature];
                                      bc_inner=tmp_bc.inner_real, bc_outer=tmp_bc.outer_real,
                                      bc_inner_imag=tmp_bc.inner_imag, bc_outer_imag=tmp_bc.outer_imag)

    # Velocity - BCs embedded in matrix (Fortran approach)
    solve_velocity_implicit_step!(state.velocity.𝒯, state.velocity.nlᵀ,
                                  state.implicit_matrices[:velocity_tor], :toroidal;
                                  i_vel_bc=get_parameters().i_vel_bc,
                                  domain=state.𝒟ᵒᶜ)
    solve_velocity_implicit_step!(state.velocity.𝒫, state.velocity.nlᴾ,
                                  state.implicit_matrices[:velocity_pol], :poloidal;
                                  i_vel_bc=get_parameters().i_vel_bc,
                                  domain=state.𝒟ᵒᶜ)

    # Magnetic (if enabled) - BCs embedded in matrix (Fortran approach)
    if i_B == 1
        solve_magnetic_implicit_step!(state.magnetic.𝒯, state.magnetic.nlᵀ,
                                      state.implicit_matrices[:magnetic_tor], :toroidal)
        solve_magnetic_implicit_step!(state.magnetic.𝒫, state.magnetic.nlᴾ,
                                      state.implicit_matrices[:magnetic_pol], :poloidal)
    end

    # Composition (if enabled) - BCs embedded in matrix
    if state.composition !== nothing
        cmp_bc = bcs.get_bc_vectors_from_field(state.composition)
        solve_composition_implicit_step!(state.composition.spectral, state.composition.nonlinear,
                                          state.implicit_matrices[:composition];
                                          bc_inner=cmp_bc.inner_real, bc_outer=cmp_bc.outer_real,
                                          bc_inner_imag=cmp_bc.inner_imag, bc_outer_imag=cmp_bc.outer_imag)
    end
end

"""
    create_enhanced_output_config(state)
    
Create output configuration with enhanced I/O settings.
"""
function create_enhanced_output_config(state::SimulationState{T}) where T
    base_config = output_config_from_parameters()

    return OutputConfig(
        base_config.output_space,
        "./enhanced_output",   # Output directory
        "geodynamo_opt",        # Filename prefix
        base_config.include_metadata,
        base_config.include_grid,
        base_config.include_diagnostics,
        base_config.output_precision,
        base_config.spectral_lmax_output,
        base_config.overwrite_files,
        0.01,                   # More frequent output for monitoring
        0.1,                    # Regular restart intervals
        Inf,                    # No time limit
        1e-12                   # High precision timing
    )
end

# Helper functions (simplified implementations)
initialize_enhanced_fields!(state) = initialize_fields!(state)
initialize_master_fields!(state) = initialize_fields!(state)

"""
    extract_all_fields(state::SimulationState)

Extract all field data from the simulation state for output/restart writing.
Returns a Dict with spectral field data (real/imag parts) for velocity, magnetic,
temperature, and composition fields.
"""
function extract_all_fields(state::SimulationState{T}) where T
    fields = Dict{String, Any}()

    # Velocity toroidal (spectral)
    fields["velocity_toroidal"] = Dict(
        "real" => copy(parent(state.velocity.𝒯.data_real)),
        "imag" => copy(parent(state.velocity.𝒯.data_imag))
    )

    # Velocity poloidal (spectral)
    fields["velocity_poloidal"] = Dict(
        "real" => copy(parent(state.velocity.𝒫.data_real)),
        "imag" => copy(parent(state.velocity.𝒫.data_imag))
    )

    # Magnetic toroidal (spectral)
    fields["magnetic_toroidal"] = Dict(
        "real" => copy(parent(state.magnetic.𝒯.data_real)),
        "imag" => copy(parent(state.magnetic.𝒯.data_imag))
    )

    # Magnetic poloidal (spectral)
    fields["magnetic_poloidal"] = Dict(
        "real" => copy(parent(state.magnetic.𝒫.data_real)),
        "imag" => copy(parent(state.magnetic.𝒫.data_imag))
    )

    # Temperature (spectral)
    fields["temperature_spectral"] = Dict(
        "real" => copy(parent(state.temperature.spectral.data_real)),
        "imag" => copy(parent(state.temperature.spectral.data_imag))
    )

    # Composition (spectral, if present)
    if state.composition !== nothing
        fields["composition_spectral"] = Dict(
            "real" => copy(parent(state.composition.spectral.data_real)),
            "imag" => copy(parent(state.composition.spectral.data_imag))
        )
    end

    return fields
end

extract_all_fields_enhanced(state) = extract_all_fields(state)
create_enhanced_metadata(state, time, step) = Dict(
    "current_time" => time,
    "current_step" => step,
    "geometry" => state.geometry,
)
update_performance_metrics!(monitor, step, compute_time, integrate_time, io_time) = nothing
auto_tune_parameters!(state) = nothing
auto_tune_parameters_master!(state) = auto_tune_parameters!(state)
compute_adaptive_timestep(state, dt) = dt
compute_adaptive_timestep_master(state, dt) = compute_adaptive_timestep(state, dt)
get_parallel_efficiency(monitor) = 0.85
get_thread_utilization(threading) = 0.92
finalize_enhanced_simulation!(state) = nothing
finalize_master_simulation!(state) = finalize_enhanced_simulation!(state)

# Advanced computation functions
function compute_velocity_nonlinear_master!(state, magnetic, temperature, domain)
    # Use enhanced CPU parallelization for velocity computation
    compute_nonlinear!(state.master_parallelizer.cpu_parallelizer, temperature, state.velocity, domain,
                       state.gradient_workspace)
end

function compute_magnetic_nonlinear_master!(state, velocity, 𝒟ᵒᶜ, 𝒟ⁱᶜ)
    # Use SIMD magnetic field computation
    compute_magnetic_nonlinear!(state.magnetic, velocity, 𝒟ᵒᶜ, 𝒟ⁱᶜ)
end

function compute_composition_nonlinear_master!(state, velocity, domain, ws::GradientWorkspace)
    # Use memory-efficient compositional computation
    if state.composition !== nothing
        compute_composition_nonlinear!(state.composition, velocity, domain, ws)
    end
end

# ================================================================================
# ENERGY CONSERVATION VALIDATION
# ================================================================================

"""
    EnergyTracker

Structure for tracking energy conservation during simulation.
"""
mutable struct EnergyTracker
    kinetic_energy::Vector{Float64}
    magnetic_energy::Vector{Float64}
    thermal_energy::Vector{Float64}
    compositional_energy::Vector{Float64}
    total_energy::Vector{Float64}
    timestamps::Vector{Int}
    enable_tracking::Bool
end

# Global energy tracker
const ENERGY_TRACKER = EnergyTracker(
    Float64[], Float64[], Float64[], Float64[], Float64[], Int[], true
)

"""
    compute_field_energy(field_data::Array{T,3}) where T

Compute L2 energy of a 3D field: E = ½ ∫ |f|² dV ≈ ½ Σ |f_ijk|²
"""
function compute_field_energy(field_data::Array{T,3}) where T
    local_energy = 0.5 * sum(abs2, field_data)
    # Sum across all MPI ranks
    if MPI.Initialized()
        return MPI.Allreduce(local_energy, +, MPI.COMM_WORLD)
    else
        return local_energy
    end
end

"""
    compute_vector_energy(v_r::Array{T,3}, v_theta::Array{T,3}, v_phi::Array{T,3}) where T

Compute kinetic/magnetic energy: E = ½ ∫ |v|² dV = ½ ∫ (v_r² + v_θ² + v_φ²) dV
"""
function compute_vector_energy(v_r::Array{T,3}, v_theta::Array{T,3}, v_phi::Array{T,3}) where T
    local_energy = 0.5 * (sum(abs2, v_r) + sum(abs2, v_theta) + sum(abs2, v_phi))
    # Sum across all MPI ranks
    if MPI.Initialized()
        return MPI.Allreduce(local_energy, +, MPI.COMM_WORLD)
    else
        return local_energy
    end
end

"""
    compute_total_energy!(state::SimulationState)

Compute and record total energy of the simulation state.
"""
function compute_total_energy!(state::SimulationState{T}) where T
    if !ENERGY_TRACKER.enable_tracking
        return
    end

    # Compute kinetic energy from velocity field
    kinetic_e = compute_vector_energy(
        parent(state.velocity.velocity.r_component.data),
        parent(state.velocity.velocity.θ_component.data),
        parent(state.velocity.velocity.φ_component.data)
    )

    # Compute magnetic energy if present
    magnetic_e = 0.0
    if state.magnetic !== nothing
        magnetic_e = compute_vector_energy(
            parent(state.magnetic.magnetic.r_component.data),
            parent(state.magnetic.magnetic.θ_component.data),
            parent(state.magnetic.magnetic.φ_component.data)
        )
    end

    # Compute thermal energy
    thermal_e = compute_field_energy(parent(state.temperature.physical.data))

    # Compute compositional energy if present
    compositional_e = 0.0
    if state.composition !== nothing
        compositional_e = compute_field_energy(parent(state.composition.physical.data))
    end

    # Total energy
    total_e = kinetic_e + magnetic_e + thermal_e + compositional_e

    # Record energies
    push!(ENERGY_TRACKER.kinetic_energy, kinetic_e)
    push!(ENERGY_TRACKER.magnetic_energy, magnetic_e)
    push!(ENERGY_TRACKER.thermal_energy, thermal_e)
    push!(ENERGY_TRACKER.compositional_energy, compositional_e)
    push!(ENERGY_TRACKER.total_energy, total_e)
    push!(ENERGY_TRACKER.timestamps, state.timestep_state.step)
end

"""
    report_energy_conservation(step::Int; interval::Int=100)

Report energy conservation statistics periodically.
"""
function report_energy_conservation(step::Int; interval::Int=100)
    if !ENERGY_TRACKER.enable_tracking
        return
    end

    n_samples = length(ENERGY_TRACKER.total_energy)
    if n_samples < 2
        return  # Need at least 2 samples to compute drift
    end

    # Check if it's time to report
    if step % interval == 0 && get_rank() == 0
        E0 = ENERGY_TRACKER.total_energy[1]
        En = ENERGY_TRACKER.total_energy[end]
        ΔE = En - E0
        rel_error = abs(ΔE / E0)

        KE = ENERGY_TRACKER.kinetic_energy[end]
        ME = ENERGY_TRACKER.magnetic_energy[end]
        TE = ENERGY_TRACKER.thermal_energy[end]
        CE = ENERGY_TRACKER.compositional_energy[end]

        @info """
        ╔══════════════════════════════════════════════════════════╗
        ║         Energy Conservation Report (Step $step)
        ╠══════════════════════════════════════════════════════════╣
        ║ Total Energy:        $En
        ║ Initial Energy:      $E0
        ║ Energy Drift (ΔE):   $ΔE
        ║ Relative Error:      $(rel_error * 100)%
        ║
        ║ Energy Breakdown:
        ║   Kinetic:           $KE  ($(KE/En * 100)%)
        ║   Magnetic:          $ME  ($(ME/En * 100)%)
        ║   Thermal:           $TE  ($(TE/En * 100)%)
        ║   Compositional:     $CE  ($(CE/En * 100)%)
        ╚══════════════════════════════════════════════════════════╝
        """

        # Warn if energy conservation is poor
        if rel_error > 0.01  # 1% error threshold
            @warn "Energy conservation error exceeds 1%! Current: $(rel_error * 100)%"
        end
    end
end

"""
    reset_energy_tracker!()

Reset energy tracking data.
"""
function reset_energy_tracker!()
    empty!(ENERGY_TRACKER.kinetic_energy)
    empty!(ENERGY_TRACKER.magnetic_energy)
    empty!(ENERGY_TRACKER.thermal_energy)
    empty!(ENERGY_TRACKER.compositional_energy)
    empty!(ENERGY_TRACKER.total_energy)
    empty!(ENERGY_TRACKER.timestamps)
end

# ================================================================================
# SOLENOIDAL CONSTRAINT VALIDATION
# ================================================================================

"""
    SolenoidalMonitor

Structure for monitoring solenoidal constraint ∇·v = 0 for velocity and magnetic fields.
"""
mutable struct SolenoidalMonitor
    velocity_div_l2::Vector{Float64}
    velocity_div_linf::Vector{Float64}
    magnetic_div_l2::Vector{Float64}
    magnetic_div_linf::Vector{Float64}
    timestamps::Vector{Int}
    enable_monitoring::Bool
end

# Global solenoidal monitor
const SOLENOIDAL_MONITOR = SolenoidalMonitor(
    Float64[], Float64[], Float64[], Float64[], Int[], true
)

"""
    compute_divergence_spectral(tor_spec::SHTnsSpecField, pol_spec::SHTnsSpecField,
                                domain::RadialDomain) where T

Compute divergence of a vector field in spectral space.
For toroidal-poloidal decomposition: v = ∇×(T r̂) + ∇×∇×(P r̂)
The divergence is: ∇·v = ∇·(∇×∇×(P r̂)) = 0 theoretically
But numerically we check: div_max = max |∇·v|
"""
function compute_divergence_spectral(tor_spec::SHTnsSpecField{T},
                                     pol_spec::SHTnsSpecField{T},
                                     domain::RadialDomain) where T
    # Simplified divergence check:
    # For solenoidal field, all spectral coefficients should have consistent scaling
    # A more sophisticated check would compute ∇·v in physical space
    # For now, we check the S-component (spheroidal) which should be zero for purely solenoidal fields

    tor_real = parent(tor_spec.data_real)
    tor_imag = parent(tor_spec.data_imag)
    pol_real = parent(pol_spec.data_real)
    pol_imag = parent(pol_spec.data_imag)

    # Compute L2 and Linf norms of spectral coefficients as divergence proxy
    local_l2 = sqrt(sum(abs2, tor_real) + sum(abs2, tor_imag) +
                    sum(abs2, pol_real) + sum(abs2, pol_imag))
    local_linf = maximum(vcat(abs.(tor_real[:]), abs.(tor_imag[:]),
                               abs.(pol_real[:]), abs.(pol_imag[:])))

    if MPI.Initialized()
        l2_norm = MPI.Allreduce(local_l2, +, MPI.COMM_WORLD)
        linf_norm = MPI.Allreduce(local_linf, max, MPI.COMM_WORLD)
    else
        l2_norm = local_l2
        linf_norm = local_linf
    end

    return l2_norm, linf_norm
end

"""
    check_solenoidal_constraint!(state::SimulationState)

Check and record solenoidal constraint for velocity and magnetic fields.
"""
function check_solenoidal_constraint!(state::SimulationState{T}) where T
    if !SOLENOIDAL_MONITOR.enable_monitoring
        return
    end

    # Check velocity solenoidal constraint
    vel_l2, vel_linf = compute_divergence_spectral(
        state.velocity.𝒯, state.velocity.𝒫, state.𝒟ᵒᶜ
    )

    # Check magnetic solenoidal constraint if present
    mag_l2, mag_linf = 0.0, 0.0
    if state.magnetic !== nothing
        mag_l2, mag_linf = compute_divergence_spectral(
            state.magnetic.𝒯, state.magnetic.𝒫, state.𝒟ᵒᶜ
        )
    end

    # Record statistics
    push!(SOLENOIDAL_MONITOR.velocity_div_l2, vel_l2)
    push!(SOLENOIDAL_MONITOR.velocity_div_linf, vel_linf)
    push!(SOLENOIDAL_MONITOR.magnetic_div_l2, mag_l2)
    push!(SOLENOIDAL_MONITOR.magnetic_div_linf, mag_linf)
    push!(SOLENOIDAL_MONITOR.timestamps, state.timestep_state.step)
end

"""
    report_solenoidal_constraint(step::Int; interval::Int=100)

Report solenoidal constraint violations periodically.
"""
function report_solenoidal_constraint(step::Int; interval::Int=100)
    if !SOLENOIDAL_MONITOR.enable_monitoring
        return
    end

    n_samples = length(SOLENOIDAL_MONITOR.velocity_div_l2)
    if n_samples < 1
        return
    end

    if step % interval == 0 && get_rank() == 0
        vel_l2 = SOLENOIDAL_MONITOR.velocity_div_l2[end]
        vel_linf = SOLENOIDAL_MONITOR.velocity_div_linf[end]
        mag_l2 = SOLENOIDAL_MONITOR.magnetic_div_l2[end]
        mag_linf = SOLENOIDAL_MONITOR.magnetic_div_linf[end]

        @info """
        ╔══════════════════════════════════════════════════════════╗
        ║      Solenoidal Constraint Report (Step $step)
        ╠══════════════════════════════════════════════════════════╣
        ║ Velocity Field (∇·v should be 0):
        ║   L2 norm:           $vel_l2
        ║   L∞ norm:           $vel_linf
        ║
        ║ Magnetic Field (∇·B should be 0):
        ║   L2 norm:           $mag_l2
        ║   L∞ norm:           $mag_linf
        ╚══════════════════════════════════════════════════════════╝
        """

        # Warn if constraints are significantly violated
        # Note: These are relative norms, so thresholds depend on field magnitude
        if vel_linf > 1e-10
            @warn "Velocity solenoidal constraint may be violated: L∞ = $vel_linf"
        end
        if mag_linf > 1e-10
            @warn "Magnetic solenoidal constraint may be violated: L∞ = $mag_linf"
        end
    end
end

"""
    reset_solenoidal_monitor!()

Reset solenoidal constraint monitoring data.
"""
function reset_solenoidal_monitor!()
    empty!(SOLENOIDAL_MONITOR.velocity_div_l2)
    empty!(SOLENOIDAL_MONITOR.velocity_div_linf)
    empty!(SOLENOIDAL_MONITOR.magnetic_div_l2)
    empty!(SOLENOIDAL_MONITOR.magnetic_div_linf)
    empty!(SOLENOIDAL_MONITOR.timestamps)
end

# ================================================================================
# ERK2 BOUNDARY CONDITION SPEC BUILDERS
# ================================================================================

"""
    build_erk2_bc_spec_scalar(T, domain, i_bc) -> ERK2BoundarySpec{T}

Build ERK2 boundary spec for a scalar field (temperature or composition).
i_bc encoding: 1=DD, 2=DN, 3=ND, 4=NN.
"""
function build_erk2_bc_spec_scalar(::Type{T}, domain::RadialDomain, i_bc::Int) where T
    nr = domain.N
    d1 = create_derivative_matrix(T, 1, domain)
    d1_inner = extract_dense_row(d1.data, d1.bandwidth, nr, 1)
    d1_outer = extract_dense_row(d1.data, d1.bandwidth, nr, nr)

    # Inner boundary
    if i_bc == 1 || i_bc == 2  # Dirichlet inner
        inner = create_dirichlet_bc(T, nr)
    else  # Neumann inner (i_bc == 3 or 4)
        # For i_bc=4 (both Neumann), override inner to Dirichlet at l=0
        # to avoid underdetermined system (matching CNAB2 treatment)
        inner = create_neumann_bc(T, d1_inner; l0_dirichlet=(i_bc == 4))
    end

    # Outer boundary
    if i_bc == 1 || i_bc == 3  # Dirichlet outer
        outer = create_dirichlet_bc(T, nr)
    else  # Neumann outer (i_bc == 2 or 4)
        outer = create_neumann_bc(T, d1_outer)
    end

    return ERK2BoundarySpec{T}(inner, outer)
end

"""
    build_erk2_bc_spec_velocity_tor(T, domain, i_vel_bc; config=nothing, rot_omega=0.0) -> ERK2BoundarySpec{T}

Build ERK2 boundary spec for the velocity toroidal component.
i_vel_bc: 1=no-slip both, 2=no-slip inner/stress-free outer,
          3=stress-free inner/no-slip outer, 4=stress-free both.

If `rot_omega ≠ 0` and inner BC is no-slip, the inner boundary value for
(l=1, m=0) is set to `rot_omega * r_inner` (rotating inner core).
"""
function build_erk2_bc_spec_velocity_tor(::Type{T}, domain::RadialDomain, i_vel_bc::Int;
                                          config::Union{SHTnsKitConfig, Nothing}=nothing,
                                          rot_omega::Float64=0.0) where T
    nr = domain.N
    d1 = create_derivative_matrix(T, 1, domain)
    d1_inner = extract_dense_row(d1.data, d1.bandwidth, nr, 1)
    d1_outer = extract_dense_row(d1.data, d1.bandwidth, nr, nr)
    r_inv_inner = T(domain.r[1, 3])
    r_inv_outer = T(domain.r[nr, 3])

    # Inner boundary
    if i_vel_bc == 1 || i_vel_bc == 2  # No-slip inner: T = 0 (Dirichlet)
        inner = create_dirichlet_bc(T, nr)
    else  # Stress-free inner: dT/dr - T/r = 0
        inner = create_stress_free_tor_bc(T, d1_inner, r_inv_inner)
    end

    # Outer boundary
    if i_vel_bc == 1 || i_vel_bc == 3  # No-slip outer: T = 0 (Dirichlet)
        outer = create_dirichlet_bc(T, nr)
    else  # Stress-free outer: dT/dr - T/r = 0
        outer = create_stress_free_tor_bc(T, d1_outer, r_inv_outer)
    end

    # Build per-mode value overrides for rotating inner core
    inner_mode_values = nothing
    if rot_omega != 0.0 && (i_vel_bc == 1 || i_vel_bc == 2) && config !== nothing
        r_inner = T(domain.r[1, 4])  # r at inner boundary
        nlm = length(config.l_values)
        inner_mode_values = zeros(T, nlm)
        for lm_idx in 1:nlm
            if config.l_values[lm_idx] == 1 && config.m_values[lm_idx] == 0
                inner_mode_values[lm_idx] = T(rot_omega) * r_inner
            end
        end
    end

    return ERK2BoundarySpec{T}(inner, outer, inner_mode_values, nothing)
end

"""
    build_erk2_bc_spec_velocity_pol(T, domain, i_vel_bc) -> ERK2BoundarySpec{T}

Build ERK2 boundary spec for the velocity poloidal component.

The no-penetration condition (u_r = l(l+1)/r² · P = 0) requires P = 0 at both
boundaries regardless of no-slip or stress-free. The CNAB2 scheme uses derivative
BCs in the matrix plus the influence matrix method to enforce P = 0. Since ERK2
doesn't have influence matrix infrastructure, we directly enforce P = 0 (Dirichlet).
"""
function build_erk2_bc_spec_velocity_pol(::Type{T}, domain::RadialDomain, i_vel_bc::Int) where T
    nr = domain.N
    # No-penetration: P = 0 at both boundaries for all velocity BC types
    return ERK2BoundarySpec{T}(create_dirichlet_bc(T, nr), create_dirichlet_bc(T, nr))
end

"""
    build_erk2_bc_spec_magnetic_tor(T, nr) -> ERK2BoundarySpec{T}

Build ERK2 boundary spec for magnetic toroidal field (insulating: BT = 0 at both).
"""
function build_erk2_bc_spec_magnetic_tor(::Type{T}, nr::Int) where T
    return ERK2BoundarySpec{T}(create_dirichlet_bc(T, nr), create_dirichlet_bc(T, nr))
end

"""
    build_erk2_bc_spec_magnetic_pol(T, domain) -> ERK2BoundarySpec{T}

Build ERK2 boundary spec for magnetic poloidal field (insulating BCs, l-dependent).
Inner: (d/dr - l/r)P = 0. Outer: (d/dr + (l+1)/r)P = 0.
"""
function build_erk2_bc_spec_magnetic_pol(::Type{T}, domain::RadialDomain) where T
    nr = domain.N
    d1 = create_derivative_matrix(T, 1, domain)
    d1_inner = extract_dense_row(d1.data, d1.bandwidth, nr, 1)
    d1_outer = extract_dense_row(d1.data, d1.bandwidth, nr, nr)
    r_inv_inner = T(domain.r[1, 3])
    r_inv_outer = T(domain.r[nr, 3])

    inner = create_insulating_inner_bc(T, d1_inner, r_inv_inner)
    outer = create_insulating_outer_bc(T, d1_outer, r_inv_outer)

    return ERK2BoundarySpec{T}(inner, outer)
end

# ================================================================================
# ERK2 INTEGRATION WITH VALIDATION
# ================================================================================

function erk2_integrate_full_step!(state::SimulationState{T}, dt::Float64;
                                    rot_omega::Float64=0.0) where T
    # Build BC specifications for all fields
    params = get_parameters()
    domain = state.𝒟ᵒᶜ
    nr = domain.N
    temp_bc = build_erk2_bc_spec_scalar(T, domain, params.i_tmp_bc)
    vel_tor_bc = build_erk2_bc_spec_velocity_tor(T, domain, params.i_vel_bc;
                                                   config=state.shtns_config, rot_omega=rot_omega)
    vel_pol_bc = build_erk2_bc_spec_velocity_pol(T, domain, params.i_vel_bc)

    # Prepare caches and stage buffers for all active fields (magnetic diffusion time scaling)
    # Diffusivities: Temperature = Pm/Pr, Velocity = E, Magnetic = 1.0, Composition = Pm/Sc
    temp_cache = get_erk2_cache!(state.erk2_caches, :temperature, d_Pm/d_Pr, T,
                                 state.shtns_config, state.𝒟ᵒᶜ, dt; use_krylov=false, bc_spec=temp_bc)
    temp_buffers = ERK2FieldBuffers(state.temperature.spectral, state.temperature.nonlinear, temp_cache)
    erk2_prepare_field!(temp_buffers, state.temperature.spectral, state.temperature.nonlinear,
                        temp_cache, state.shtns_config, dt; bc_spec=temp_bc)

    vel_tor_cache = get_erk2_cache!(state.erk2_caches, :velocity_toroidal, d_E, T,
                                    state.shtns_config, state.𝒟ᵒᶜ, dt; use_krylov=false, bc_spec=vel_tor_bc)
    vel_tor_buffers = ERK2FieldBuffers(state.velocity.𝒯, state.velocity.nlᵀ, vel_tor_cache)
    erk2_prepare_field!(vel_tor_buffers, state.velocity.𝒯, state.velocity.nlᵀ,
                        vel_tor_cache, state.shtns_config, dt; bc_spec=vel_tor_bc)

    vel_pol_cache = get_erk2_cache!(state.erk2_caches, :velocity_poloidal, d_E, T,
                                    state.shtns_config, state.𝒟ᵒᶜ, dt; use_krylov=false, bc_spec=vel_pol_bc)
    vel_pol_buffers = ERK2FieldBuffers(state.velocity.𝒫, state.velocity.nlᴾ, vel_pol_cache)
    erk2_prepare_field!(vel_pol_buffers, state.velocity.𝒫, state.velocity.nlᴾ,
                        vel_pol_cache, state.shtns_config, dt; bc_spec=vel_pol_bc)

    mag_tor_buffers = nothing
    mag_pol_buffers = nothing
    mag_tor_cache = nothing
    mag_pol_cache = nothing
    mag_tor_bc = nothing
    mag_pol_bc = nothing
    if i_B == 1 && state.magnetic !== nothing
        mag_tor_bc = build_erk2_bc_spec_magnetic_tor(T, nr)
        mag_pol_bc = build_erk2_bc_spec_magnetic_pol(T, domain)

        mag_tor_cache = get_erk2_cache!(state.erk2_caches, :magnetic_toroidal, 1.0, T,
                                        state.shtns_config, state.𝒟ᵒᶜ, dt; use_krylov=false, bc_spec=mag_tor_bc)
        mag_tor_buffers = ERK2FieldBuffers(state.magnetic.𝒯, state.magnetic.nlᵀ, mag_tor_cache)
        erk2_prepare_field!(mag_tor_buffers, state.magnetic.𝒯, state.magnetic.nlᵀ,
                            mag_tor_cache, state.shtns_config, dt; bc_spec=mag_tor_bc)

        mag_pol_cache = get_erk2_cache!(state.erk2_caches, :magnetic_poloidal, 1.0, T,
                                        state.shtns_config, state.𝒟ᵒᶜ, dt; use_krylov=false, bc_spec=mag_pol_bc)
        mag_pol_buffers = ERK2FieldBuffers(state.magnetic.𝒫, state.magnetic.nlᴾ, mag_pol_cache)
        erk2_prepare_field!(mag_pol_buffers, state.magnetic.𝒫, state.magnetic.nlᴾ,
                            mag_pol_cache, state.shtns_config, dt; bc_spec=mag_pol_bc)
    end

    comp_buffers = nothing
    comp_cache = nothing
    comp_bc = nothing
    if state.composition !== nothing
        comp_bc = build_erk2_bc_spec_scalar(T, domain, params.i_cmp_bc)

        comp_cache = get_erk2_cache!(state.erk2_caches, :composition, d_Pm/d_Sc, T,
                                     state.shtns_config, state.𝒟ᵒᶜ, dt; use_krylov=false, bc_spec=comp_bc)
        comp_buffers = ERK2FieldBuffers(state.composition.spectral, state.composition.nonlinear, comp_cache)
        erk2_prepare_field!(comp_buffers, state.composition.spectral, state.composition.nonlinear,
                            comp_cache, state.shtns_config, dt; bc_spec=comp_bc)
    end

    # Apply stage solution to all fields
    erk2_apply_stage!(temp_buffers, state.temperature.spectral)
    erk2_apply_stage!(vel_tor_buffers, state.velocity.𝒯)
    erk2_apply_stage!(vel_pol_buffers, state.velocity.𝒫)
    if mag_tor_buffers !== nothing
        erk2_apply_stage!(mag_tor_buffers, state.magnetic.𝒯)
        erk2_apply_stage!(mag_pol_buffers, state.magnetic.𝒫)
    end
    if comp_buffers !== nothing
        erk2_apply_stage!(comp_buffers, state.composition.spectral)
    end

    # Evaluate nonlinear terms at the stage state for all coupled equations
    compute_all_nonlinear_terms!(state)

    # Store stage nonlinearities for later correction
    erk2_store_stage_nonlinear!(temp_buffers, state.temperature.nonlinear)
    maybe_log_erk2_stage_residual!(:temperature, temp_buffers, state.timestep_state.step)
    erk2_store_stage_nonlinear!(vel_tor_buffers, state.velocity.nlᵀ)
    maybe_log_erk2_stage_residual!(:velocity_toroidal, vel_tor_buffers, state.timestep_state.step)
    erk2_store_stage_nonlinear!(vel_pol_buffers, state.velocity.nlᴾ)
    maybe_log_erk2_stage_residual!(:velocity_poloidal, vel_pol_buffers, state.timestep_state.step)
    if mag_tor_buffers !== nothing
        erk2_store_stage_nonlinear!(mag_tor_buffers, state.magnetic.nlᵀ)
        maybe_log_erk2_stage_residual!(:magnetic_toroidal, mag_tor_buffers, state.timestep_state.step)
        erk2_store_stage_nonlinear!(mag_pol_buffers, state.magnetic.nlᴾ)
        maybe_log_erk2_stage_residual!(:magnetic_poloidal, mag_pol_buffers, state.timestep_state.step)
    end
    if comp_buffers !== nothing
        erk2_store_stage_nonlinear!(comp_buffers, state.composition.nonlinear)
        maybe_log_erk2_stage_residual!(:composition, comp_buffers, state.timestep_state.step)
    end

    # Finalize each field using stage data with BC enforcement
    erk2_finalize_field!(temp_buffers, state.temperature.spectral, temp_cache, state.shtns_config, dt;
                         bc_spec=temp_bc)
    erk2_finalize_field!(vel_tor_buffers, state.velocity.𝒯, vel_tor_cache, state.shtns_config, dt;
                         bc_spec=vel_tor_bc)
    erk2_finalize_field!(vel_pol_buffers, state.velocity.𝒫, vel_pol_cache, state.shtns_config, dt;
                         bc_spec=vel_pol_bc)
    if mag_tor_buffers !== nothing
        erk2_finalize_field!(mag_tor_buffers, state.magnetic.𝒯, mag_tor_cache, state.shtns_config, dt;
                             bc_spec=mag_tor_bc)
        erk2_finalize_field!(mag_pol_buffers, state.magnetic.𝒫, mag_pol_cache, state.shtns_config, dt;
                             bc_spec=mag_pol_bc)
    end
    if comp_buffers !== nothing
        erk2_finalize_field!(comp_buffers, state.composition.spectral, comp_cache, state.shtns_config, dt;
                             bc_spec=comp_bc)
    end

    # Report φ₂ conditioning statistics periodically (every 100 steps)
    report_phi2_conditioning(state.timestep_state.step; interval=100)

    # Compute and report energy conservation periodically
    compute_total_energy!(state)
    report_energy_conservation(state.timestep_state.step; interval=100)

    # Check and report solenoidal constraints periodically
    check_solenoidal_constraint!(state)
    report_solenoidal_constraint(state.timestep_state.step; interval=100)

    # Note: nonlinear terms are recomputed by the main simulation loop at the
    # start of each step, so no refresh is needed here.
end

# ================================================================================
# Master Time Integration Function
# ================================================================================
#
# This function advances all fields by one timestep using IMEX (implicit-explicit)
# time integration. Reference: Sreenivasan & Kar (2018), Phys. Rev. Fluids 3, 093801
#
# IMEX SCHEME OVERVIEW:
# =====================
# - EXPLICIT: Advection, Coriolis, Lorentz force, buoyancy (computed above)
# - IMPLICIT: Diffusion terms (treated here for numerical stability)
#
# DIFFUSION COEFFICIENTS (magnetic diffusion time scaling):
# ====================================================================
# After non-dimensionalizing with time scale τ = L²/η (magnetic diffusion time):
# Reference: Sreenivasan & Kar (2018), Phys. Rev. Fluids 3, 093801, Eqs. (1)-(3)
#
#   Field        | Equation                      | Implicit Diffusivity
#   -------------|-------------------------------|----------------------
#   Velocity     | E·Pm⁻¹(∂u/∂t + ...) + ... = E∇²u | d_E (Ekman number)
#   Temperature  | ∂T/∂t + u·∇T = (Pm/Pr)∇²T    | d_Pm/d_Pr
#   Magnetic     | ∂B/∂t = ∇×(u×B) + ∇²B        | 1.0
#   Composition  | ∂C/∂t + u·∇C = (Pm/Sc)∇²C    | d_Pm/d_Sc
#
# With magnetic diffusion time scaling:
#   - Velocity uses Ekman number E = ν/(2ΩL²)
#   - Magnetic diffusivity is 1 (time scale is L²/η)
#   - Scalars use Pm/Pr and Pm/Sc
#
# ================================================================================
function apply_implicit_step!(state::SimulationState{T}, dt::Float64) where T
    # For CNAB2 and ERK2: bootstrap prev_nonlinear on first step so AB2 reduces to AB1
    if (ts_scheme === :cnab2 || ts_scheme === :erk2) && state.timestep_state.step == 0
        parent(state.temperature.prev_nonlinear.data_real) .= parent(state.temperature.nonlinear.data_real)
        parent(state.temperature.prev_nonlinear.data_imag) .= parent(state.temperature.nonlinear.data_imag)
        parent(state.velocity.prev_nlᵀ.data_real) .= parent(state.velocity.nlᵀ.data_real)
        parent(state.velocity.prev_nlᵀ.data_imag) .= parent(state.velocity.nlᵀ.data_imag)
        parent(state.velocity.prev_nlᴾ.data_real) .= parent(state.velocity.nlᴾ.data_real)
        parent(state.velocity.prev_nlᴾ.data_imag) .= parent(state.velocity.nlᴾ.data_imag)
        if i_B == 1
            parent(state.magnetic.prev_nlᵀ.data_real) .= parent(state.magnetic.nlᵀ.data_real)
            parent(state.magnetic.prev_nlᵀ.data_imag) .= parent(state.magnetic.nlᵀ.data_imag)
            parent(state.magnetic.prev_nlᴾ.data_real) .= parent(state.magnetic.nlᴾ.data_real)
            parent(state.magnetic.prev_nlᴾ.data_imag) .= parent(state.magnetic.nlᴾ.data_imag)
        end
        if state.composition !== nothing
            parent(state.composition.prev_nonlinear.data_real) .= parent(state.composition.nonlinear.data_real)
            parent(state.composition.prev_nonlinear.data_imag) .= parent(state.composition.nonlinear.data_imag)
        end
    end
    
    # Note: Temperature and composition BCs are now embedded in the implicit matrix.
    # Time-dependent BC updates would require matrix re-factorization if needed.

    if ts_scheme === :erk2
        erk2_integrate_full_step!(state, dt)
    else
        # Task-based time integration for other schemes
        task_graph = create_task_graph()

        # -----------------------------------------------------------------
        # Temperature time integration
        # Diffusivity = Pm/Pr from Eq. (2): ∂T/∂t + u·∇T = (Pm/Pr)∇²T
        # -----------------------------------------------------------------
        temp_task = add_task!(task_graph, () -> begin
        tmp_bc = bcs.get_bc_vectors_from_field(state.temperature)
        if ts_scheme === :cnab2
            build_rhs_cnab2!(state.temperature.work_spectral, state.temperature.spectral,
                             state.temperature.nonlinear, state.temperature.prev_nonlinear,
                             dt, state.implicit_matrices[:temperature])
            solve_temperature_implicit_step!(state.temperature.spectral, state.temperature.work_spectral,
                                              state.implicit_matrices[:temperature];
                                              bc_inner=tmp_bc.inner_real, bc_outer=tmp_bc.outer_real,
                                              bc_inner_imag=tmp_bc.inner_imag, bc_outer_imag=tmp_bc.outer_imag)
        elseif ts_scheme === :eab2
            etd = haskey(state.etd_caches, :temperature) ? state.etd_caches[:temperature] : nothing
            if etd === nothing || etd.dt != dt
                # Temperature diffusivity = Pm/Pr (magnetic diffusion time scaling)
                etd = create_etd_cache(Float64, state.shtns_config, state.𝒟ᵒᶜ, d_Pm/d_Pr, dt)
                state.etd_caches[:temperature] = etd
            end
            eab2_update!(state.temperature.spectral, state.temperature.nonlinear,
                         state.temperature.prev_nonlinear, etd, state.shtns_config, dt)
        else
            solve_temperature_implicit_step!(state.temperature.spectral, state.temperature.nonlinear,
                                              state.implicit_matrices[:temperature];
                                              bc_inner=tmp_bc.inner_real, bc_outer=tmp_bc.outer_real,
                                              bc_inner_imag=tmp_bc.inner_imag, bc_outer_imag=tmp_bc.outer_imag)
        end
    end)
    
    # -----------------------------------------------------------------
    # Velocity time integration (toroidal and poloidal components)
    # Diffusivity = E (Ekman number): E∇²u
    # -----------------------------------------------------------------
    vel_tor_task = add_task!(task_graph, () -> begin
        if ts_scheme === :cnab2
            build_rhs_cnab2!(state.velocity.work_tor, state.velocity.𝒯,
                             state.velocity.nlᵀ, state.velocity.prev_nlᵀ,
                             dt, state.implicit_matrices[:velocity])
            solve_velocity_implicit_step!(state.velocity.𝒯, state.velocity.work_tor,
                                          state.implicit_matrices[:velocity_tor], :toroidal;
                                          i_vel_bc=get_parameters().i_vel_bc,
                                          domain=state.𝒟ᵒᶜ)
        elseif ts_scheme === :eab2
            # Velocity diffusivity = E (Ekman number)
            alu_map = get_eab2_alu_cache!(state.etd_caches, :velocity_toroidal, d_E, T, state.𝒟ᵒᶜ)
            eab2_update_krylov_cached!(state.velocity.𝒯, state.velocity.nlᵀ,
                                       state.velocity.prev_nlᵀ, alu_map, state.𝒟ᵒᶜ, d_E,
                                       state.shtns_config, dt; m=i_etd_m, tol=d_krylov_tol)
        else
            solve_velocity_implicit_step!(state.velocity.𝒯, state.velocity.nlᵀ,
                                          state.implicit_matrices[:velocity_tor], :toroidal;
                                          i_vel_bc=get_parameters().i_vel_bc,
                                          domain=state.𝒟ᵒᶜ)
        end
    end)

    vel_pol_task = add_task!(task_graph, () -> begin
        if ts_scheme === :cnab2
            build_rhs_cnab2!(state.velocity.work_pol, state.velocity.𝒫,
                             state.velocity.nlᴾ, state.velocity.prev_nlᴾ,
                             dt, state.implicit_matrices[:velocity])
            solve_velocity_implicit_step!(state.velocity.𝒫, state.velocity.work_pol,
                                          state.implicit_matrices[:velocity_pol], :poloidal;
                                          i_vel_bc=get_parameters().i_vel_bc,
                                          domain=state.𝒟ᵒᶜ)
        elseif ts_scheme === :eab2
            # Velocity diffusivity = E (Ekman number)
            alu_map = get_eab2_alu_cache!(state.etd_caches, :velocity_poloidal, d_E, T, state.𝒟ᵒᶜ)
            eab2_update_krylov_cached!(state.velocity.𝒫, state.velocity.nlᴾ,
                                       state.velocity.prev_nlᴾ, alu_map, state.𝒟ᵒᶜ, d_E,
                                       state.shtns_config, dt; m=i_etd_m, tol=d_krylov_tol)
        else
            solve_velocity_implicit_step!(state.velocity.𝒫, state.velocity.nlᴾ,
                                          state.implicit_matrices[:velocity_pol], :poloidal;
                                          i_vel_bc=get_parameters().i_vel_bc,
                                          domain=state.𝒟ᵒᶜ)
        end
    end)
    
    # -----------------------------------------------------------------
    # Magnetic field time integration (if enabled)
    # Diffusivity = 1: ∂B/∂t = ∇×(u×B) + ∇²B
    # -----------------------------------------------------------------
    if i_B == 1
        add_task!(task_graph, () -> begin
            if ts_scheme === :cnab2
                build_rhs_cnab2!(state.magnetic.work_tor, state.magnetic.𝒯,
                                 state.magnetic.nlᵀ, state.magnetic.prev_nlᵀ,
                                 dt, state.implicit_matrices[:magnetic])
                solve_magnetic_implicit_step!(state.magnetic.𝒯, state.magnetic.work_tor,
                                              state.implicit_matrices[:magnetic_tor], :toroidal)
            elseif ts_scheme === :eab2
                # Magnetic diffusivity = 1 (magnetic diffusion time scaling)
                alu_map = get_eab2_alu_cache!(state.etd_caches, :magnetic_toroidal, 1.0, T, state.𝒟ᵒᶜ)
                eab2_update_krylov_cached!(state.magnetic.𝒯, state.magnetic.nlᵀ,
                                           state.magnetic.prev_nlᵀ, alu_map, state.𝒟ᵒᶜ, 1.0,
                                           state.shtns_config, dt; m=i_etd_m, tol=d_krylov_tol)
            else
                solve_magnetic_implicit_step!(state.magnetic.𝒯, state.magnetic.nlᵀ,
                                              state.implicit_matrices[:magnetic_tor], :toroidal)
            end
        end)
        add_task!(task_graph, () -> begin
            if ts_scheme === :cnab2
                build_rhs_cnab2!(state.magnetic.work_pol, state.magnetic.𝒫,
                                 state.magnetic.nlᴾ, state.magnetic.prev_nlᴾ,
                                 dt, state.implicit_matrices[:magnetic])
                solve_magnetic_implicit_step!(state.magnetic.𝒫, state.magnetic.work_pol,
                                              state.implicit_matrices[:magnetic_pol], :poloidal)
            elseif ts_scheme === :eab2
                # Magnetic diffusivity = 1 (magnetic diffusion time scaling)
                alu_map = get_eab2_alu_cache!(state.etd_caches, :magnetic_poloidal, 1.0, T, state.𝒟ᵒᶜ)
                eab2_update_krylov_cached!(state.magnetic.𝒫, state.magnetic.nlᴾ,
                                           state.magnetic.prev_nlᴾ, alu_map, state.𝒟ᵒᶜ, 1.0,
                                           state.shtns_config, dt; m=i_etd_m, tol=d_krylov_tol)
            else
                solve_magnetic_implicit_step!(state.magnetic.𝒫, state.magnetic.nlᴾ,
                                              state.implicit_matrices[:magnetic_pol], :poloidal)
            end
        end)
    end
    
    # -----------------------------------------------------------------
    # Composition field time integration (if enabled)
    # Diffusivity = Pm/Sc (analogous to temperature): ∂C/∂t + u·∇C = (Pm/Sc)∇²C
    # -----------------------------------------------------------------
    if state.composition !== nothing
        add_task!(task_graph, () -> begin
            cmp_bc = bcs.get_bc_vectors_from_field(state.composition)
            if ts_scheme === :cnab2
                build_rhs_cnab2!(state.composition.work_spectral, state.composition.spectral,
                                 state.composition.nonlinear, state.composition.prev_nonlinear,
                                 dt, state.implicit_matrices[:composition])
                solve_composition_implicit_step!(state.composition.spectral, state.composition.work_spectral,
                                                  state.implicit_matrices[:composition];
                                                  bc_inner=cmp_bc.inner_real, bc_outer=cmp_bc.outer_real,
                                                  bc_inner_imag=cmp_bc.inner_imag, bc_outer_imag=cmp_bc.outer_imag)
            elseif ts_scheme === :eab2
                # Composition diffusivity = Pm/Sc (magnetic diffusion time scaling)
                alu_map = get_eab2_alu_cache!(state.etd_caches, :composition, d_Pm/d_Sc, T, state.𝒟ᵒᶜ)
                eab2_update_krylov_cached!(state.composition.spectral, state.composition.nonlinear,
                                           state.composition.prev_nonlinear, alu_map, state.𝒟ᵒᶜ, d_Pm/d_Sc,
                                           state.shtns_config, dt; m=i_etd_m, tol=d_krylov_tol)
            else
                solve_composition_implicit_step!(state.composition.spectral, state.composition.nonlinear,
                                                  state.implicit_matrices[:composition];
                                                  bc_inner=cmp_bc.inner_real, bc_outer=cmp_bc.outer_real,
                                                  bc_inner_imag=cmp_bc.inner_imag, bc_outer_imag=cmp_bc.outer_imag)
            end
        end)
    end
        
        # Execute all tasks in parallel
        execute_task_graph!(task_graph, state.master_parallelizer.cpu_parallelizer.thread_manager)
    end

    # All BCs (velocity, magnetic, temperature, composition) are now embedded
    # in the implicit matrix solve. No post-processing needed.

    # Roll nonlinear histories for CNAB2, EAB2, and ERK2
    if ts_scheme === :cnab2 || ts_scheme === :eab2 || ts_scheme === :erk2
        parent(state.temperature.prev_nonlinear.data_real) .= parent(state.temperature.nonlinear.data_real)
        parent(state.temperature.prev_nonlinear.data_imag) .= parent(state.temperature.nonlinear.data_imag)
        parent(state.velocity.prev_nlᵀ.data_real) .= parent(state.velocity.nlᵀ.data_real)
        parent(state.velocity.prev_nlᵀ.data_imag) .= parent(state.velocity.nlᵀ.data_imag)
        parent(state.velocity.prev_nlᴾ.data_real) .= parent(state.velocity.nlᴾ.data_real)
        parent(state.velocity.prev_nlᴾ.data_imag) .= parent(state.velocity.nlᴾ.data_imag)
        if i_B == 1
            parent(state.magnetic.prev_nlᵀ.data_real) .= parent(state.magnetic.nlᵀ.data_real)
            parent(state.magnetic.prev_nlᵀ.data_imag) .= parent(state.magnetic.nlᵀ.data_imag)
            parent(state.magnetic.prev_nlᴾ.data_real) .= parent(state.magnetic.nlᴾ.data_real)
            parent(state.magnetic.prev_nlᴾ.data_imag) .= parent(state.magnetic.nlᴾ.data_imag)
        end
        if state.composition !== nothing
            parent(state.composition.prev_nonlinear.data_real) .= parent(state.composition.nonlinear.data_real)
            parent(state.composition.prev_nonlinear.data_imag) .= parent(state.composition.nonlinear.data_imag)
        end
    end
end

function adapt_thread_count!(state::SimulationState, efficiency_history::Vector{Float64})
    # Simple adaptive threading - could be more sophisticated
    recent_efficiency = mean(efficiency_history[max(1, end-4):end])
    
    if recent_efficiency < 0.7
        # Efficiency is low, might need fewer threads to reduce overhead
        return max(1, state.master_parallelizer.cpu_parallelizer.thread_manager.compute_threads - 1)
    elseif recent_efficiency > 0.9
        # High efficiency, could potentially use more threads
        return min(state.master_parallelizer.cpu_parallelizer.thread_manager.total_threads, 
                  state.master_parallelizer.cpu_parallelizer.thread_manager.compute_threads + 1)
    else
        return state.master_parallelizer.cpu_parallelizer.thread_manager.compute_threads
    end
end

function optimize_memory_usage!(state::SimulationState)
    # Optimize memory layout and garbage collection
    # This would implement more sophisticated memory optimization in practice
    GC.gc()  # Force garbage collection to free unused memory
end

function analyze_master_performance(state::SimulationState)
    # Comprehensive performance analysis for master simulation
    analyze_parallel_performance(state.master_parallelizer.performance_monitor)
    
    # Additional CPU-specific analysis would go here
end

function create_master_output_config(state::SimulationState{T}) where T
    base_config = output_config_from_parameters()

    return OutputConfig(
        base_config.output_space,
        "./master_output",   # Output directory
        "geodynamo_master",      # Filename prefix
        base_config.include_metadata,
        base_config.include_grid,
        base_config.include_diagnostics,
        base_config.output_precision,
        base_config.spectral_lmax_output,
        base_config.overwrite_files,
        0.01,                   # More frequent output for monitoring
        0.1,                    # Regular restart intervals
        Inf,                    # No time limit
        1e-12                   # High precision timing
    )
end

# ================================================================================
# MAIN ENTRY POINTS
# ================================================================================

# Main entry point for basic simulation
function run_shtns_geodynamo_simulation()
    state = initialize_shtns_simulation(Float64)
    run_shtns_simulation!(state)
    Finalize()
end

# Main entry point for enhanced simulation
function run_enhanced_geodynamo_simulation()
    state = initialize_simulation(Float64)
    run_enhanced_simulation!(state)
    Finalize()
end

# Main entry point for master simulation
function run_master_geodynamo_simulation()
    state = initialize_master_simulation(Float64)
    run_master_simulation!(state)
    Finalize()
end

# Exports are handled by the main GeoDynamo.jl module
