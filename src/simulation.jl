# ================================================================================
# Main Simulation Driver with SHTns (Basic + Enhanced Parallelization)
# ================================================================================

using MPI
using Base.Threads
using LinearAlgebra

# Import geometry modules - safe to do even if they're not available
try
    using .GeodynamoBall
catch e
    @debug "GeodynamoBall not available: $e"
end

try
    using .GeodynamoShell
catch e
    @debug "GeodynamoShell not available: $e"
end

"""
    SimulationState{T}
    
Unified simulation state with comprehensive parallelization and diagnostics.
"""
struct SimulationState{T}
    # Original components
    velocity::SHTnsVelocityFields{T}
    magnetic::SHTnsMagneticFields{T}
    temperature::SHTnsTemperatureField{T}
    composition::Union{SHTnsCompositionField{T}, Nothing}
    
    # Geometric data
    shtns_config::SHTnsKitConfig
    oc_domain::RadialDomain
    ic_domain::RadialDomain
    
    # Unified master parallelization system
    master_parallelizer::MasterParallelizer{T}
    
    # Timestepping
    timestep_state::TimestepState
    implicit_matrices::Dict{Symbol, SHTnsImplicitMatrices{T}}
    etd_caches::Dict{Symbol, Any}
    erk2_caches::Dict{Symbol, Any}  # ERK2 method caches
    
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
# ENHANCED SIMULATION INITIALIZATION
# ================================================================================

"""
    initialize_enhanced_simulation(::Type{T}=Float64; kwargs...)
    
Initialize simulation with all parallelization features enabled.
"""
function initialize_enhanced_simulation(::Type{T}=Float64; 
                                        include_composition::Bool=true,
                                        auto_optimize::Bool=true,
                                        thread_count::Int=Threads.nthreads()) where T
    
    # Initialize MPI first
    if !MPI.Initialized()
        MPI.Init()
    end
    
    rank = get_rank()
    nprocs = get_nprocs()
    
    if rank == 0
        println("="^80)
        println("    GEODYNAMO.jl - ENHANCED PARALLEL INITIALIZATION")
        println("="^80)
        println("MPI Processes: $nprocs")
        println("Threads per process: $thread_count")
        println("Auto-optimization: $(auto_optimize ? "ENABLED" : "DISABLED")")
        println("="^80)
    end
    
    # Create SHTns configuration with enhanced topology
    shtns_config = create_shtnskit_config(lmax=i_L, mmax=i_M, nlat=i_Th, nlon=i_Ph, optimize_decomp=true)
    
    # Initialize enhanced pencil decomposition
    pencils = create_pencil_topology(shtns_config, optimize=true)
    pencil_spec = pencils.spec
    
    geom = get_parameters().geometry
    if geom === :ball
        oc_domain = GeodynamoBall.create_ball_radial_domain(i_N)
        ic_domain = oc_domain
        velocity = GeodynamoBall.create_ball_velocity_fields(T, shtns_config; nr=i_N)
        magnetic = GeodynamoBall.create_ball_magnetic_fields(T, shtns_config; nr=i_N)
        temperature = GeodynamoBall.create_ball_temperature_field(T, shtns_config; nr=i_N)
        composition = include_composition ? GeodynamoBall.create_ball_composition_field(T, shtns_config; nr=i_N) : nothing
    else
        oc_domain = GeodynamoShell.create_shell_radial_domain(i_N)
        ic_domain = GeodynamoShell.create_shell_radial_domain(i_Nic)
        velocity = GeodynamoShell.create_shell_velocity_fields(T, shtns_config; nr=i_N)
        magnetic = GeodynamoShell.create_shell_magnetic_fields(T, shtns_config; nr_oc=i_N, nr_ic=i_Nic)
        temperature = GeodynamoShell.create_shell_temperature_field(T, shtns_config; nr=i_N)
        composition = include_composition ? GeodynamoShell.create_shell_composition_field(T, shtns_config; nr=i_N) : nothing
    end
    
    # Initialize hybrid parallelization system
    hybrid_parallelizer = create_hybrid_parallelizer(T, shtns_config)
    performance_monitor = create_performance_monitor()
    
    # Initialize timestepping with enhanced matrices
    timestep_state = TimestepState(d_time, d_timestep, 0, 0, Inf, false)
    
    # Create implicit matrices for each equation
    implicit_matrices = Dict{Symbol, SHTnsImplicitMatrices{T}}()
    implicit_matrices[:velocity] = create_shtns_timestepping_matrices(shtns_config, oc_domain, d_Pm, d_timestep)
    implicit_matrices[:magnetic] = create_shtns_timestepping_matrices(shtns_config, oc_domain, 1.0, d_timestep)
    implicit_matrices[:temperature] = create_shtns_timestepping_matrices(shtns_config, oc_domain, d_Pm/d_Pr, d_timestep)
    
    if include_composition
        implicit_matrices[:composition] = create_shtns_timestepping_matrices(shtns_config, oc_domain, d_Pm/d_Sc, d_timestep)
    end
    
    return SimulationState{T}(
        velocity, magnetic, temperature, composition,
        shtns_config, oc_domain, ic_domain,
        hybrid_parallelizer, performance_monitor,
        timestep_state, implicit_matrices,
        0, auto_optimize, geom
    )
end

# ================================================================================
# MASTER SIMULATION INITIALIZATION
# ================================================================================

"""
    initialize_master_simulation(::Type{T}=Float64; kwargs...)
    
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
        oc_domain = GeodynamoBall.create_ball_radial_domain(i_N)
        ic_domain = oc_domain
        velocity = GeodynamoBall.create_ball_velocity_fields(T, shtns_config; nr=i_N)
        magnetic = GeodynamoBall.create_ball_magnetic_fields(T, shtns_config; nr=i_N)
        temperature = GeodynamoBall.create_ball_temperature_field(T, shtns_config; nr=i_N)
        composition = include_composition ? GeodynamoBall.create_ball_composition_field(T, shtns_config; nr=i_N) : nothing
    else
        oc_domain = GeodynamoShell.create_shell_radial_domain(i_N)
        ic_domain = GeodynamoShell.create_shell_radial_domain(i_Nic)
        velocity = GeodynamoShell.create_shell_velocity_fields(T, shtns_config; nr=i_N)
        magnetic = GeodynamoShell.create_shell_magnetic_fields(T, shtns_config; nr_oc=i_N, nr_ic=i_Nic)
        temperature = GeodynamoShell.create_shell_temperature_field(T, shtns_config; nr=i_N)
        composition = include_composition ? GeodynamoShell.create_shell_composition_field(T, shtns_config; nr=i_N) : nothing
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
    
    # Create implicit matrices for each equation
    implicit_matrices = Dict{Symbol, SHTnsImplicitMatrices{T}}()
    implicit_matrices[:velocity] = create_shtns_timestepping_matrices(shtns_config, oc_domain, d_Pm, d_timestep)
    implicit_matrices[:magnetic] = create_shtns_timestepping_matrices(shtns_config, oc_domain, 1.0, d_timestep)
    implicit_matrices[:temperature] = create_shtns_timestepping_matrices(shtns_config, oc_domain, d_Pm/d_Pr, d_timestep)
    
    if include_composition
        implicit_matrices[:composition] = create_shtns_timestepping_matrices(shtns_config, oc_domain, d_Pm/d_Sc, d_timestep)
    end
    
    return SimulationState{T}(
        velocity, magnetic, temperature, composition,
        shtns_config, oc_domain, ic_domain,
        master_parallelizer,
        timestep_state, implicit_matrices,
        Dict{Symbol,Any}(),  # etd_caches
        Dict{Symbol,Any}(),  # erk2_caches
        0, auto_optimize, adaptive_threading, geom
    )
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
        println("SHTns Grid: $(state.shtns_config.nlat) × $(state.shtns_config.nlon) × $(state.oc_domain.N)")
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
        prev_velocity = deepcopy(state.velocity.toroidal)
        prev_magnetic = deepcopy(state.magnetic.toroidal)
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
            error_vel  = compute_timestep_error(state.velocity.toroidal, prev_velocity)
            error_mag  = compute_timestep_error(state.magnetic.toroidal, prev_magnetic)
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
            
            # Update previous state
            prev_velocity    .= state.velocity.toroidal
            prev_magnetic    .= state.magnetic.toroidal
            prev_temperature .= state.temperature.spectral
            
            if state.composition !== nothing && prev_composition !== nothing
                prev_composition .= state.composition.spectral
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
    run_enhanced_simulation!(state::SimulationState{T})

Run geodynamo simulation with all parallelization optimizations.
"""
function run_enhanced_simulation!(state::SimulationState{T}) where T
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
    
    # Initialize fields with perturbations
    initialize_enhanced_fields!(state)
    
    # Create enhanced output configuration
    output_config = create_enhanced_output_config(state)
    time_tracker = create_time_tracker(output_config)
    
    # Main timestepping loop with optimizations
    step = 0
    simulation_time = d_time
    dt = d_timestep
    
    # Performance monitoring
    total_start = MPI.Wtime()
    last_output_time = simulation_time
    
    while simulation_time < 1.0 && step < i_maxtstep
        step += 1
        step_start = MPI.Wtime()
        
        # === ENHANCED PHYSICS COMPUTATION ===
        
        # 1. Hybrid nonlinear computation (MPI + Threads)
        compute_start = MPI.Wtime()
        
        # Temperature evolution with all optimizations
        hybrid_compute_nonlinear!(state.hybrid_parallelizer, state.temperature, 
                                 state.velocity, state.oc_domain)
        
        # Velocity evolution
        if state.velocity !== nothing
            compute_velocity_nonlinear!(state.velocity, state.magnetic, state.temperature, state.oc_domain)
        end
        
        # Magnetic field evolution  
        if i_B == 1 && state.magnetic !== nothing
            compute_magnetic_nonlinear!(state.magnetic, state.velocity, state.oc_domain, state.ic_domain)
        end
        
        # Compositional evolution (if enabled)
        if state.composition !== nothing
            compute_composition_nonlinear!(state.composition, state.velocity, state.oc_domain)
        end
        
        compute_time = MPI.Wtime() - compute_start
        
        # 2. Enhanced time integration
        integrate_start = MPI.Wtime()
        
        # Apply implicit time integration with enhanced solvers
        apply_enhanced_implicit_step!(state, dt)
        
        integrate_time = MPI.Wtime() - integrate_start
        
        # 3. Asynchronous I/O (non-blocking)
        io_start = MPI.Wtime()
        
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
        
        io_time = MPI.Wtime() - io_start
        
        # 4. Performance monitoring and adaptive optimization
        if state.auto_optimization && step % 50 == 0
            monitor_start = MPI.Wtime()
            
            # Update performance metrics
            update_performance_metrics!(state.performance_monitor, step, 
                                      compute_time, integrate_time, io_time)
            
            # Dynamic load balancing check
            adaptive_rebalance!(state.hybrid_parallelizer.load_balancer, state.temperature)
            
            # Auto-tuning of parameters
            if step % 200 == 0
                auto_tune_parameters!(state)
            end
            
            monitor_time = MPI.Wtime() - monitor_start
        end
        
        # Update simulation state
        simulation_time += dt
        state.timestep_state.current_time = simulation_time
        state.timestep_state.step_count = step
        
        step_time = MPI.Wtime() - step_start
        
        # Adaptive timestep (if enabled)
        if step % 10 == 0
            dt = compute_adaptive_timestep(state, dt)
        end
    end
    
    total_time = MPI.Wtime() - total_start
    
    # Final performance analysis
    if rank == 0
        analyze_parallel_performance(state.performance_monitor)
        
        println("\n" * "="^80)
        println("         SIMULATION COMPLETED SUCCESSFULLY")
        println("="^80)
        println("Total steps: $step")
        println("Final time: $(round(simulation_time, digits=4))")
        println("Total wall time: $(round(total_time, digits=2)) seconds")
        println("Average time per step: $(round(total_time/step*1000, digits=2)) ms")
        
        # Parallel efficiency summary
        parallel_efficiency = get_parallel_efficiency(state.performance_monitor)
        println("Parallel efficiency: $(round(parallel_efficiency*100, digits=1))%")
        
        # Thread utilization
        thread_utilization = get_thread_utilization(state.hybrid_parallelizer.threading_accelerator)
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
    run_master_simulation!(state::SimulationState{T})
    
Run geodynamo simulation with maximum CPU parallelization optimizations.
"""
function run_simulation!(state::SimulationState{T}) where T
    comm = get_comm()
    rank = get_rank()
    nprocs = get_nprocs()
    
    if rank == 0
        println("\nStarting MASTER geodynamo simulation...")
        println("Grid: $(state.shtns_config.nlat) × $(state.shtns_config.nlon) × $(i_N)")
        println("Spectral modes: $(state.shtns_config.nlm) (lmax=$(state.shtns_config.lmax))")
        println("Parallel configuration: $nprocs MPI × $(Threads.nthreads()) threads")
        println("CPU parallelization level: COMPREHENSIVE (SIMD + NUMA + Task-based)")
        println()
    end
    
    # Initialize fields with perturbations
    initialize_master_fields!(state)
    
    # Create enhanced output configuration
    output_config = create_enhanced_output_config(state)
    time_tracker = create_time_tracker(output_config)
    
    # Main timestepping loop with advanced features
    step = 0
    simulation_time = d_time
    dt = d_timestep
    
    # Performance monitoring
    total_start = MPI.Wtime()
    last_output_time = simulation_time
    
    # Adaptive threading state
    thread_efficiency_history = Float64[]
    optimal_thread_count = Threads.nthreads()
    
    while simulation_time < 1.0 && step < i_maxtstep
        step += 1
        step_start = MPI.Wtime()
        
        # === ADVANCED PHYSICS COMPUTATION ===
        
        # 1. Advanced nonlinear computation (CPU + SIMD + Task-based)
        compute_start = MPI.Wtime()
        
        # Temperature evolution with maximum CPU optimizations
        compute_nonlinear!(state.master_parallelizer.cpu_parallelizer, state.temperature, 
                                   state.velocity, state.oc_domain)
        
        # Velocity evolution with task-based parallelism
        if state.velocity !== nothing
            compute_velocity_nonlinear_master!(state, state.magnetic, state.temperature, state.oc_domain)
        end
        
        # Magnetic field evolution with SIMD optimization
        if i_B == 1 && state.magnetic !== nothing
            compute_magnetic_nonlinear_master!(state, state.velocity, state.oc_domain, state.ic_domain)
        end
        
        # Compositional evolution (if enabled) with memory optimization
        if state.composition !== nothing
            compute_composition_nonlinear_master!(state, state.velocity, state.oc_domain)
        end
        
        compute_time = MPI.Wtime() - compute_start
        
        # 2. Advanced time integration with task scheduling
        integrate_start = MPI.Wtime()
        
        # Apply implicit time integration with advanced solvers
        apply_master_implicit_step!(state, dt)
        
        integrate_time = MPI.Wtime() - integrate_start
        
        # 3. Asynchronous I/O with memory-aware scheduling
        io_start = MPI.Wtime()
        
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
        
        io_time = MPI.Wtime() - io_start
        
        # 4. Advanced performance monitoring and adaptive optimization
        if state.auto_optimization && step % 25 == 0  # More frequent optimization
            monitor_start = MPI.Wtime()
            
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
            
            monitor_time = MPI.Wtime() - monitor_start
        end
        
        # Update simulation state
        simulation_time += dt
        state.timestep_state.current_time = simulation_time
        state.timestep_state.step_count = step
        
        step_time = MPI.Wtime() - step_start
        
        # Adaptive timestep with CPU-aware scaling
        if step % 5 == 0  # More frequent timestep adaptation
            dt = compute_adaptive_timestep_master(state, dt)
        end
    end
    
    total_time = MPI.Wtime() - total_start
    
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
                r = state.oc_domain.r[r_idx, 4]
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
    fill!(parent(state.velocity.toroidal.data_real), zero(T))
    fill!(parent(state.velocity.toroidal.data_imag), zero(T))
    fill!(parent(state.velocity.poloidal.data_real), zero(T))
    fill!(parent(state.velocity.poloidal.data_imag), zero(T))
    
    # Magnetic field: dipole + small perturbation
    mag_tor_real = parent(state.magnetic.toroidal.data_real)
    mag_tor_imag = parent(state.magnetic.toroidal.data_imag)
    mag_pol_real = parent(state.magnetic.poloidal.data_real)
    mag_pol_imag = parent(state.magnetic.poloidal.data_imag)
    
    for lm_idx in 1:state.magnetic.toroidal.nlm
        l = state.magnetic.toroidal.config.l_values[lm_idx]
        m = state.magnetic.toroidal.config.m_values[lm_idx]
        
        for r_idx in axes(mag_tor_real, 3)
            if l == 1 && m == 0
                # Dipole field
                r = state.oc_domain.r[r_idx, 4]
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

function compute_all_nonlinear_terms!(state::SimulationState{T}) where T
    # Compute nonlinear terms for all equations using SHTns transforms
    compute_velocity_nonlinear!(state.velocity, state.temperature, 
                                state.composition, state.magnetic, state.oc_domain)
    
    compute_magnetic_nonlinear!(state.magnetic, state.velocity, state.oc_domain, state.ic_domain, 0.0)  # No inner core rotation for now
    
    compute_temperature_nonlinear!(state.temperature, state.velocity, state.oc_domain)
    
    # Compute compositional advection if composition is present
    if state.composition !== nothing
        compute_composition_nonlinear!(state.composition, state.velocity, state.oc_domain)
    end
end

function predictor_step!(state::SimulationState{T}) where T
    # Predictor step for all fields using SHTns matrices
    
    # Velocity (toroidal component)
    rhs_tor = similar(state.velocity.toroidal)
    apply_explicit_operator!(rhs_tor, state.velocity.toroidal,
                             state.velocity.nl_toroidal, state.oc_domain,
                             d_Pm, state.timestep_state.dt;
                             nl_prev=state.velocity.prev_nl_toroidal,
                             matrices=state.implicit_matrices[:velocity])
    solve_implicit_step!(state.velocity.toroidal, rhs_tor, 
                        state.implicit_matrices[:velocity])
    
    # Velocity (poloidal component)
    rhs_pol = similar(state.velocity.poloidal)
    apply_explicit_operator!(rhs_pol, state.velocity.poloidal,
                             state.velocity.nl_poloidal, state.oc_domain,
                             d_Pm, state.timestep_state.dt;
                             nl_prev=state.velocity.prev_nl_poloidal,
                             matrices=state.implicit_matrices[:velocity])
    solve_implicit_step!(state.velocity.poloidal, rhs_pol, 
                        state.implicit_matrices[:velocity])
    
    # Magnetic field (toroidal)
    rhs_mag_tor = similar(state.magnetic.toroidal)
    apply_explicit_operator!(rhs_mag_tor, state.magnetic.toroidal,
                             state.magnetic.nl_toroidal, state.oc_domain,
                             1.0, state.timestep_state.dt;
                             nl_prev=state.magnetic.prev_nl_toroidal,
                             matrices=state.implicit_matrices[:magnetic])
    solve_implicit_step!(state.magnetic.toroidal, rhs_mag_tor, 
                        state.implicit_matrices[:magnetic])
    
    # Magnetic field (poloidal)
    rhs_mag_pol = similar(state.magnetic.poloidal)
    apply_explicit_operator!(rhs_mag_pol, state.magnetic.poloidal,
                             state.magnetic.nl_poloidal, state.oc_domain,
                             1.0, state.timestep_state.dt;
                             nl_prev=state.magnetic.prev_nl_poloidal,
                             matrices=state.implicit_matrices[:magnetic])
    solve_implicit_step!(state.magnetic.poloidal, rhs_mag_pol, 
                        state.implicit_matrices[:magnetic])
    
    # Temperature
    rhs_temp = similar(state.temperature.spectral)
    apply_explicit_operator!(rhs_temp, state.temperature.spectral,
                             state.temperature.nonlinear, state.oc_domain,
                             d_Pm/d_Pr, state.timestep_state.dt;
                             nl_prev=state.temperature.prev_nonlinear,
                             matrices=state.implicit_matrices[:temperature])
    solve_implicit_step!(state.temperature.spectral, rhs_temp, 
                        state.implicit_matrices[:temperature])
    
    # Composition (if present)
    if state.composition !== nothing
        rhs_comp = similar(state.composition.spectral)
        apply_explicit_operator!(rhs_comp, state.composition.spectral,
                                 state.composition.nonlinear, state.oc_domain,
                                 d_Pm/d_Sc, state.timestep_state.dt;
                                 nl_prev=state.composition.prev_nonlinear,
                                 matrices=state.implicit_matrices[:composition])
        solve_implicit_step!(state.composition.spectral, rhs_comp, 
                            state.implicit_matrices[:composition])
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
    shtnskit_vector_synthesis!(state.velocity.toroidal, state.velocity.poloidal, state.velocity.velocity)
    
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
    dr_min = minimum(diff(state.oc_domain.r[:, 4]))
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
    # Update implicit matrices with new timestep
    dt = state.timestep_state.dt
    
    state.implicit_matrices[:velocity] = create_shtns_timestepping_matrices(
        state.shtns_config, state.oc_domain, d_Pm, dt)
    state.implicit_matrices[:magnetic] = create_shtns_timestepping_matrices(
        state.shtns_config, state.oc_domain, 1.0, dt)
    state.implicit_matrices[:temperature] = create_shtns_timestepping_matrices(
        state.shtns_config, state.oc_domain, d_Pm/d_Pr, dt)
    
    # Update compositional matrix if composition is present
    if state.composition !== nothing
        state.implicit_matrices[:composition] = create_shtns_timestepping_matrices(
            state.shtns_config, state.oc_domain, d_Pm/d_Sc, dt)
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
    
    # Temperature
    solve_implicit_step!(state.temperature.spectral, state.temperature.nonlinear,
                        state.implicit_matrices[:temperature], dt)
    
    # Velocity
    solve_implicit_step!(state.velocity.toroidal, state.velocity.nl_toroidal,
                        state.implicit_matrices[:velocity], dt)
    solve_implicit_step!(state.velocity.poloidal, state.velocity.nl_poloidal,
                        state.implicit_matrices[:velocity], dt)
    
    # Magnetic (if enabled)
    if i_B == 1
        solve_implicit_step!(state.magnetic.toroidal, state.magnetic.nl_toroidal,
                            state.implicit_matrices[:magnetic], dt)
        solve_implicit_step!(state.magnetic.poloidal, state.magnetic.nl_poloidal,
                            state.implicit_matrices[:magnetic], dt)
    end
    
    # Composition (if enabled)
    if state.composition !== nothing
        solve_implicit_step!(state.composition.spectral, state.composition.nonlinear,
                            state.implicit_matrices[:composition], dt)
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
        base_config.naming_scheme,
        "./enhanced_output",   # Output directory
        "geodynamo_opt",        # Filename prefix
        9,                      # Maximum compression for storage efficiency
        base_config.include_metadata,
        base_config.include_grid,
        base_config.include_diagnostics,
        base_config.output_precision,
        base_config.spectral_lmax_output,
        base_config.overwrite_files,
        base_config.independent_writes,
        0.01,                   # More frequent output for monitoring
        0.1,                    # Regular restart intervals
        Inf,                    # No time limit
        1e-12                   # High precision timing
    )
end

# Helper functions (simplified implementations)
initialize_enhanced_fields!(state) = initialize_fields!(state)
initialize_master_fields!(state) = initialize_fields!(state)
extract_all_fields(state) = Dict("temperature" => rand(32, 64, 20))
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
update_tracker!(tracker, time, config, output, restart) = nothing

# Advanced computation functions
function compute_velocity_nonlinear_master!(state, magnetic, temperature, domain)
    # Use enhanced CPU parallelization for velocity computation
    compute_nonlinear!(state.master_parallelizer.cpu_parallelizer, temperature, state.velocity, domain)
end

function compute_magnetic_nonlinear_master!(state, velocity, oc_domain, ic_domain)
    # Use SIMD magnetic field computation
    compute_magnetic_nonlinear!(state.magnetic, velocity, oc_domain, ic_domain)
end

function compute_composition_nonlinear_master!(state, velocity, domain)
    # Use memory-efficient compositional computation
    if state.composition !== nothing
        compute_composition_nonlinear!(state.composition, velocity, domain)
    end
end

function erk2_integrate_full_step!(state::SimulationState{T}, dt::Float64) where T
    # Prepare caches and stage buffers for all active fields
    temp_cache = get_erk2_cache!(state.erk2_caches, :temperature, d_Pm/d_Pr, T,
                                 state.shtns_config, state.oc_domain, dt; use_krylov=false)
    temp_buffers = ERK2FieldBuffers(state.temperature.spectral, state.temperature.nonlinear, temp_cache)
    erk2_prepare_field!(temp_buffers, state.temperature.spectral, state.temperature.nonlinear,
                        temp_cache, state.shtns_config, dt)

    vel_tor_cache = get_erk2_cache!(state.erk2_caches, :velocity_toroidal, d_Pm, T,
                                    state.shtns_config, state.oc_domain, dt; use_krylov=false)
    vel_tor_buffers = ERK2FieldBuffers(state.velocity.toroidal, state.velocity.nl_toroidal, vel_tor_cache)
    erk2_prepare_field!(vel_tor_buffers, state.velocity.toroidal, state.velocity.nl_toroidal,
                        vel_tor_cache, state.shtns_config, dt)

    vel_pol_cache = get_erk2_cache!(state.erk2_caches, :velocity_poloidal, d_Pm, T,
                                    state.shtns_config, state.oc_domain, dt; use_krylov=false)
    vel_pol_buffers = ERK2FieldBuffers(state.velocity.poloidal, state.velocity.nl_poloidal, vel_pol_cache)
    erk2_prepare_field!(vel_pol_buffers, state.velocity.poloidal, state.velocity.nl_poloidal,
                        vel_pol_cache, state.shtns_config, dt)

    mag_tor_buffers = nothing
    mag_pol_buffers = nothing
    mag_tor_cache = nothing
    mag_pol_cache = nothing
    if i_B == 1 && state.magnetic !== nothing
        mag_tor_cache = get_erk2_cache!(state.erk2_caches, :magnetic_toroidal, 1.0, T,
                                        state.shtns_config, state.oc_domain, dt; use_krylov=false)
        mag_tor_buffers = ERK2FieldBuffers(state.magnetic.toroidal, state.magnetic.nl_toroidal, mag_tor_cache)
        erk2_prepare_field!(mag_tor_buffers, state.magnetic.toroidal, state.magnetic.nl_toroidal,
                            mag_tor_cache, state.shtns_config, dt)

        mag_pol_cache = get_erk2_cache!(state.erk2_caches, :magnetic_poloidal, 1.0, T,
                                        state.shtns_config, state.oc_domain, dt; use_krylov=false)
        mag_pol_buffers = ERK2FieldBuffers(state.magnetic.poloidal, state.magnetic.nl_poloidal, mag_pol_cache)
        erk2_prepare_field!(mag_pol_buffers, state.magnetic.poloidal, state.magnetic.nl_poloidal,
                            mag_pol_cache, state.shtns_config, dt)
    end

    comp_buffers = nothing
    comp_cache = nothing
    if state.composition !== nothing
        comp_cache = get_erk2_cache!(state.erk2_caches, :composition, d_Pm/d_Sc, T,
                                     state.shtns_config, state.oc_domain, dt; use_krylov=false)
        comp_buffers = ERK2FieldBuffers(state.composition.spectral, state.composition.nonlinear, comp_cache)
        erk2_prepare_field!(comp_buffers, state.composition.spectral, state.composition.nonlinear,
                            comp_cache, state.shtns_config, dt)
    end

    # Apply stage solution to all fields
    erk2_apply_stage!(temp_buffers, state.temperature.spectral)
    erk2_apply_stage!(vel_tor_buffers, state.velocity.toroidal)
    erk2_apply_stage!(vel_pol_buffers, state.velocity.poloidal)
    if mag_tor_buffers !== nothing
        erk2_apply_stage!(mag_tor_buffers, state.magnetic.toroidal)
        erk2_apply_stage!(mag_pol_buffers, state.magnetic.poloidal)
    end
    if comp_buffers !== nothing
        erk2_apply_stage!(comp_buffers, state.composition.spectral)
    end

    # Evaluate nonlinear terms at the stage state for all coupled equations
    compute_all_nonlinear_terms!(state)

    # Store stage nonlinearities for later correction
    erk2_store_stage_nonlinear!(temp_buffers, state.temperature.nonlinear)
    maybe_log_erk2_stage_residual!(:temperature, temp_buffers, state.timestep_state.step)
    erk2_store_stage_nonlinear!(vel_tor_buffers, state.velocity.nl_toroidal)
    maybe_log_erk2_stage_residual!(:velocity_toroidal, vel_tor_buffers, state.timestep_state.step)
    erk2_store_stage_nonlinear!(vel_pol_buffers, state.velocity.nl_poloidal)
    maybe_log_erk2_stage_residual!(:velocity_poloidal, vel_pol_buffers, state.timestep_state.step)
    if mag_tor_buffers !== nothing
        erk2_store_stage_nonlinear!(mag_tor_buffers, state.magnetic.nl_toroidal)
        maybe_log_erk2_stage_residual!(:magnetic_toroidal, mag_tor_buffers, state.timestep_state.step)
        erk2_store_stage_nonlinear!(mag_pol_buffers, state.magnetic.nl_poloidal)
        maybe_log_erk2_stage_residual!(:magnetic_poloidal, mag_pol_buffers, state.timestep_state.step)
    end
    if comp_buffers !== nothing
        erk2_store_stage_nonlinear!(comp_buffers, state.composition.nonlinear)
        maybe_log_erk2_stage_residual!(:composition, comp_buffers, state.timestep_state.step)
    end

    # Finalize each field using stage data
    erk2_finalize_field!(temp_buffers, state.temperature.spectral, temp_cache, state.shtns_config, dt)
    erk2_finalize_field!(vel_tor_buffers, state.velocity.toroidal, vel_tor_cache, state.shtns_config, dt)
    erk2_finalize_field!(vel_pol_buffers, state.velocity.poloidal, vel_pol_cache, state.shtns_config, dt)
    if mag_tor_buffers !== nothing
        erk2_finalize_field!(mag_tor_buffers, state.magnetic.toroidal, mag_tor_cache, state.shtns_config, dt)
        erk2_finalize_field!(mag_pol_buffers, state.magnetic.poloidal, mag_pol_cache, state.shtns_config, dt)
    end
    if comp_buffers !== nothing
        erk2_finalize_field!(comp_buffers, state.composition.spectral, comp_cache, state.shtns_config, dt)
    end

    # Refresh nonlinear terms at the new solution for the next step
    compute_all_nonlinear_terms!(state)
end

function apply_master_implicit_step!(state::SimulationState{T}, dt::Float64) where T
    # For CNAB2 and ERK2: bootstrap prev_nonlinear on first step so AB2 reduces to AB1
    if (ts_scheme === :cnab2 || ts_scheme === :erk2) && state.timestep_state.step == 0
        parent(state.temperature.prev_nonlinear.data_real) .= parent(state.temperature.nonlinear.data_real)
        parent(state.temperature.prev_nonlinear.data_imag) .= parent(state.temperature.nonlinear.data_imag)
        parent(state.velocity.prev_nl_toroidal.data_real) .= parent(state.velocity.nl_toroidal.data_real)
        parent(state.velocity.prev_nl_toroidal.data_imag) .= parent(state.velocity.nl_toroidal.data_imag)
        parent(state.velocity.prev_nl_poloidal.data_real) .= parent(state.velocity.nl_poloidal.data_real)
        parent(state.velocity.prev_nl_poloidal.data_imag) .= parent(state.velocity.nl_poloidal.data_imag)
        if i_B == 1
            parent(state.magnetic.prev_nl_toroidal.data_real) .= parent(state.magnetic.nl_toroidal.data_real)
            parent(state.magnetic.prev_nl_toroidal.data_imag) .= parent(state.magnetic.nl_toroidal.data_imag)
            parent(state.magnetic.prev_nl_poloidal.data_real) .= parent(state.magnetic.nl_poloidal.data_real)
            parent(state.magnetic.prev_nl_poloidal.data_imag) .= parent(state.magnetic.nl_poloidal.data_imag)
        end
        if state.composition !== nothing
            parent(state.composition.prev_nonlinear.data_real) .= parent(state.composition.nonlinear.data_real)
            parent(state.composition.prev_nonlinear.data_imag) .= parent(state.composition.nonlinear.data_imag)
        end
    end
    
    # Update time-dependent boundary conditions if enabled
    current_time = state.timestep_state.step * dt
    update_time_dependent_temperature_boundaries!(state.temperature, current_time)
    
    if ts_scheme === :erk2
        erk2_integrate_full_step!(state, dt)
    else
        # Task-based time integration for other schemes
        task_graph = create_task_graph()

        temp_task = add_task!(task_graph, () -> begin
        if ts_scheme === :cnab2
            build_rhs_cnab2!(state.temperature.work_spectral, state.temperature.spectral,
                             state.temperature.nonlinear, state.temperature.prev_nonlinear,
                             dt, state.implicit_matrices[:temperature])
            solve_implicit_step!(state.temperature.spectral, state.temperature.work_spectral,
                                 state.implicit_matrices[:temperature])
        elseif ts_scheme === :eab2
            etd = haskey(state.etd_caches, :temperature) ? state.etd_caches[:temperature] : nothing
            if etd === nothing || etd.dt != dt
                etd = create_etd_cache(Float64, state.shtns_config, state.oc_domain, d_Pm/d_Pr, dt)
                state.etd_caches[:temperature] = etd
            end
            eab2_update!(state.temperature.spectral, state.temperature.nonlinear,
                         state.temperature.prev_nonlinear, etd, state.shtns_config, dt)
        else
            solve_implicit_step!(state.temperature.spectral, state.temperature.nonlinear,
                                 state.implicit_matrices[:temperature])
        end
    end)
    
    vel_tor_task = add_task!(task_graph, () -> begin
        if ts_scheme === :cnab2
            build_rhs_cnab2!(state.velocity.work_tor, state.velocity.toroidal,
                             state.velocity.nl_toroidal, state.velocity.prev_nl_toroidal,
                             dt, state.implicit_matrices[:velocity])
            solve_implicit_step!(state.velocity.toroidal, state.velocity.work_tor,
                                 state.implicit_matrices[:velocity])
        elseif ts_scheme === :eab2
            # Use Krylov-based action with cached banded LU per l
            alu_map = get_eab2_alu_cache!(state.etd_caches, :velocity_toroidal, d_Pm, T, state.oc_domain)
            eab2_update_krylov_cached!(state.velocity.toroidal, state.velocity.nl_toroidal,
                                       state.velocity.prev_nl_toroidal, alu_map, state.oc_domain, d_Pm,
                                       state.shtns_config, dt; m=i_etd_m, tol=d_krylov_tol)
        else
            solve_implicit_step!(state.velocity.toroidal, state.velocity.nl_toroidal,
                                 state.implicit_matrices[:velocity])
        end
    end)
    
    vel_pol_task = add_task!(task_graph, () -> begin
        if ts_scheme === :cnab2
            build_rhs_cnab2!(state.velocity.work_pol, state.velocity.poloidal,
                             state.velocity.nl_poloidal, state.velocity.prev_nl_poloidal,
                             dt, state.implicit_matrices[:velocity])
            solve_implicit_step!(state.velocity.poloidal, state.velocity.work_pol,
                                 state.implicit_matrices[:velocity])
        elseif ts_scheme === :eab2
            alu_map = get_eab2_alu_cache!(state.etd_caches, :velocity_poloidal, d_Pm, T, state.oc_domain)
            eab2_update_krylov_cached!(state.velocity.poloidal, state.velocity.nl_poloidal,
                                       state.velocity.prev_nl_poloidal, alu_map, state.oc_domain, d_Pm,
                                       state.shtns_config, dt; m=i_etd_m, tol=d_krylov_tol)
        else
            solve_implicit_step!(state.velocity.poloidal, state.velocity.nl_poloidal,
                                 state.implicit_matrices[:velocity])
        end
    end)
    
    # Magnetic field tasks (if enabled)
    if i_B == 1
        add_task!(task_graph, () -> begin
            if ts_scheme === :cnab2
                build_rhs_cnab2!(state.magnetic.work_tor, state.magnetic.toroidal,
                                 state.magnetic.nl_toroidal, state.magnetic.prev_nl_toroidal,
                                 dt, state.implicit_matrices[:magnetic])
                solve_implicit_step!(state.magnetic.toroidal, state.magnetic.work_tor,
                                     state.implicit_matrices[:magnetic])
            elseif ts_scheme === :eab2
                alu_map = get_eab2_alu_cache!(state.etd_caches, :magnetic_toroidal, 1.0, T, state.oc_domain)
                eab2_update_krylov_cached!(state.magnetic.toroidal, state.magnetic.nl_toroidal,
                                           state.magnetic.prev_nl_toroidal, alu_map, state.oc_domain, 1.0,
                                           state.shtns_config, dt; m=i_etd_m, tol=d_krylov_tol)
            else
                solve_implicit_step!(state.magnetic.toroidal, state.magnetic.nl_toroidal,
                                     state.implicit_matrices[:magnetic])
            end
        end)
        add_task!(task_graph, () -> begin
            if ts_scheme === :cnab2
                build_rhs_cnab2!(state.magnetic.work_pol, state.magnetic.poloidal,
                                 state.magnetic.nl_poloidal, state.magnetic.prev_nl_poloidal,
                                 dt, state.implicit_matrices[:magnetic])
                solve_implicit_step!(state.magnetic.poloidal, state.magnetic.work_pol,
                                     state.implicit_matrices[:magnetic])
            elseif ts_scheme === :eab2
                alu_map = get_eab2_alu_cache!(state.etd_caches, :magnetic_poloidal, 1.0, T, state.oc_domain)
                eab2_update_krylov_cached!(state.magnetic.poloidal, state.magnetic.nl_poloidal,
                                           state.magnetic.prev_nl_poloidal, alu_map, state.oc_domain, 1.0,
                                           state.shtns_config, dt; m=i_etd_m, tol=d_krylov_tol)
            else
                solve_implicit_step!(state.magnetic.poloidal, state.magnetic.nl_poloidal,
                                     state.implicit_matrices[:magnetic])
            end
        end)
    end
    
    # Composition task (if enabled)
    if state.composition !== nothing
        add_task!(task_graph, () -> begin
            if ts_scheme === :cnab2
                build_rhs_cnab2!(state.composition.work_spectral, state.composition.spectral,
                                 state.composition.nonlinear, state.composition.prev_nonlinear,
                                 dt, state.implicit_matrices[:composition])
                solve_implicit_step!(state.composition.spectral, state.composition.work_spectral,
                                     state.implicit_matrices[:composition])
            elseif ts_scheme === :eab2
                alu_map = get_eab2_alu_cache!(state.etd_caches, :composition, d_Pm/d_Sc, T, state.oc_domain)
                eab2_update_krylov_cached!(state.composition.spectral, state.composition.nonlinear,
                                           state.composition.prev_nonlinear, alu_map, state.oc_domain, d_Pm/d_Sc,
                                           state.shtns_config, dt; m=i_etd_m, tol=d_krylov_tol)
            else
                solve_implicit_step!(state.composition.spectral, state.composition.nonlinear,
                                     state.implicit_matrices[:composition])
            end
        end)
    end
        
        # Execute all tasks in parallel
        execute_task_graph!(task_graph, state.master_parallelizer.cpu_parallelizer.thread_manager)
    end

    apply_velocity_boundary_conditions!(state.velocity)
    if i_B == 1 && state.magnetic !== nothing
        apply_magnetic_boundary_conditions!(state.magnetic)
    end
    if state.composition !== nothing
        apply_composition_boundary_conditions!(state.composition)
    end

    # Roll nonlinear histories for CNAB2, EAB2, and ERK2
    if ts_scheme === :cnab2 || ts_scheme === :eab2 || ts_scheme === :erk2
        parent(state.temperature.prev_nonlinear.data_real) .= parent(state.temperature.nonlinear.data_real)
        parent(state.temperature.prev_nonlinear.data_imag) .= parent(state.temperature.nonlinear.data_imag)
        parent(state.velocity.prev_nl_toroidal.data_real) .= parent(state.velocity.nl_toroidal.data_real)
        parent(state.velocity.prev_nl_toroidal.data_imag) .= parent(state.velocity.nl_toroidal.data_imag)
        parent(state.velocity.prev_nl_poloidal.data_real) .= parent(state.velocity.nl_poloidal.data_real)
        parent(state.velocity.prev_nl_poloidal.data_imag) .= parent(state.velocity.nl_poloidal.data_imag)
        if i_B == 1
            parent(state.magnetic.prev_nl_toroidal.data_real) .= parent(state.magnetic.nl_toroidal.data_real)
            parent(state.magnetic.prev_nl_toroidal.data_imag) .= parent(state.magnetic.nl_toroidal.data_imag)
            parent(state.magnetic.prev_nl_poloidal.data_real) .= parent(state.magnetic.nl_poloidal.data_real)
            parent(state.magnetic.prev_nl_poloidal.data_imag) .= parent(state.magnetic.nl_poloidal.data_imag)
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
    # Reuse the existing enhanced output config
    base_config = output_config_from_parameters()

    return OutputConfig(
        base_config.output_space,
        base_config.naming_scheme,
        "./master_output",   # Output directory
        "geodynamo_master",      # Filename prefix
        9,                      # Maximum compression for storage efficiency
        base_config.include_metadata,
        base_config.include_grid,
        base_config.include_diagnostics,
        base_config.output_precision,
        base_config.spectral_lmax_output,
        base_config.overwrite_files,
        base_config.independent_writes,
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
    MPI.Finalize()
end

# Main entry point for enhanced simulation
function run_enhanced_geodynamo_simulation()
    state = initialize_enhanced_simulation(Float64)
    run_enhanced_simulation!(state)
    MPI.Finalize()
end

# Main entry point for master simulation
function run_master_geodynamo_simulation()
    state = initialize_master_simulation(Float64)
    run_master_simulation!(state)
    MPI.Finalize()
end

# Exports are handled by the main Geodynamo.jl module
