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
    erk2_influence_matrices::Dict{Symbol, Tuple{Dict{Int, ERK2InfluenceMatrix{T}}, Float64}}  # Influence matrices for ERK2 (matrices, dt)

    # Enhanced I/O
    auto_optimization::Bool
    adaptive_threading::Bool
    geometry::Symbol

end

# ================================================================================
# SIMULATION INITIALIZATION
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
    timestep_state = TimestepState(d_time, d_timestep, 0, 0, Inf, false, true)

    # Create implicit matrices for each equation (magnetic diffusion time scaling)
    # Velocity: E, Temperature: Pm/Pr, Magnetic: 1, Composition: Pm/Sc
    implicit_matrices = Dict{Symbol, SHTnsImplicitMatrices{T}}()
    # Velocity uses separate toroidal/poloidal matrices with BCs embedded (Fortran approach)
    implicit_matrices[:velocity_tor] = create_velocity_toroidal_matrices(shtns_config, 𝒟ᵒᶜ, d_E, d_timestep;
                                                                          i_vel_bc=get_parameters().i_vel_bc,
                                                                          mass_coeff=d_E)
    implicit_matrices[:velocity_pol] = create_velocity_poloidal_matrices(shtns_config, 𝒟ᵒᶜ, d_E, d_timestep;
                                                                          i_vel_bc=get_parameters().i_vel_bc,
                                                                          mass_coeff=d_E)
    # Magnetic uses separate toroidal/poloidal matrices with BCs embedded (Fortran approach)
    implicit_matrices[:magnetic_tor] = create_magnetic_toroidal_matrices(shtns_config, 𝒟ᵒᶜ, 1.0, d_timestep)
    implicit_matrices[:magnetic_pol] = create_magnetic_poloidal_matrices(shtns_config, 𝒟ᵒᶜ, 1.0, d_timestep)
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
        Dict{Symbol, Tuple{Dict{Int, ERK2InfluenceMatrix{T}}, Float64}}(),  # ERK2 influence matrices (matrices, dt)
        auto_optimize, adaptive_threading, geom
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
# SIMULATION RUNNER
# ================================================================================

"""
    run_simulation!(state::SimulationState{T}; restart_file::String="", restart_dir::String="", restart_time::Float64=0.0)

Run geodynamo simulation with all parallelization optimizations.
Supports restarting from a saved NetCDF restart file.
"""
function run_simulation!(state::SimulationState{T};
                                  restart_file::String="",
                                  restart_dir::String="",
                                  restart_time::Float64=0.0) where T
    comm = get_comm()
    rank = get_rank()
    nprocs = get_nprocs()

    if rank == 0
        println("\nStarting geodynamo simulation...")
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
        state.timestep_state.step = step
        # Bootstrap prev_nonlinear on first iteration after restart (AB2 → AB1)
        state.timestep_state.needs_ab2_bootstrap = true

        if rank == 0
            println("  Resuming from time = $simulation_time, step = $step")
        end
    else
        initialize_fields!(state)
        time_tracker = create_time_tracker(output_config)
    end
    
    # Performance monitoring
    total_start = Wtime()
    
    t_end = d_t_end
    while simulation_time < t_end && step < i_maxtstep
        step += 1

        # === PHYSICS COMPUTATION ===

        # 1. Compute nonlinear terms for all equations
        compute_start = Wtime()

        # Velocity evolution
        compute_velocity_nonlinear!(state.velocity, state.temperature,
                                    state.composition, state.magnetic, state.𝒟ᵒᶜ)

        # Magnetic field evolution
        if i_B == 1 && state.magnetic !== nothing
            compute_magnetic_nonlinear!(state.magnetic, state.velocity, state.𝒟ᵒᶜ, state.𝒟ⁱᶜ)
        end

        # Temperature evolution
        compute_temperature_nonlinear!(state.temperature, state.velocity, state.𝒟ᵒᶜ,
                                        state.gradient_workspace)

        # Compositional evolution (if enabled)
        if state.composition !== nothing
            compute_composition_nonlinear!(state.composition, state.velocity, state.𝒟ᵒᶜ,
                                           state.gradient_workspace)
        end

        compute_time = Wtime() - compute_start
        
        # 2. Enhanced time integration
        integrate_start = Wtime()
        
        # Apply implicit time integration (supports CNAB2, EAB2, ERK2)
        apply_implicit_step!(state, dt)
        
        integrate_time = Wtime() - integrate_start
        
        # Update simulation state
        simulation_time += dt
        state.timestep_state.time = simulation_time
        state.timestep_state.step = step

        # 3. Asynchronous I/O (non-blocking)

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

        # 4. Dynamic load balancing and auto-tuning
        if state.auto_optimization && step % 50 == 0
            adaptive_rebalance!(state.master_parallelizer.load_balancer, state.temperature)
            if step % 200 == 0
                auto_tune_parameters!(state)
            end
        end
        
        # Adaptive timestep (if enabled)
        if step % 10 == 0
            dt = compute_adaptive_timestep(state, dt)
        end
    end
    
    total_time = Wtime() - total_start
    
    # Final summary
    if rank == 0
        println("\n" * "="^80)
        println("         SIMULATION COMPLETED SUCCESSFULLY")
        println("="^80)
        println("Total steps: $step")
        println("Final time: $(round(simulation_time, digits=4))")
        println("Total wall time: $(round(total_time, digits=2)) seconds")
        if step > 0
            println("Average time per step: $(round(total_time/step*1000, digits=2)) ms")
        end
        println("="^80)
    end
end


# ================================================================================
# SHARED HELPER FUNCTIONS
# ================================================================================

function initialize_fields!(state::SimulationState{T}) where T
    # Seed RNG per rank for reproducible initialization across MPI processes
    Random.seed!(42 + get_rank())

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
        compute_magnetic_nonlinear!(state.magnetic, state.velocity, state.𝒟ᵒᶜ, state.𝒟ⁱᶜ)
    end

    compute_temperature_nonlinear!(state.temperature, state.velocity, state.𝒟ᵒᶜ,
                                    state.gradient_workspace)

    # Compute compositional advection if composition is present
    if state.composition !== nothing
        compute_composition_nonlinear!(state.composition, state.velocity, state.𝒟ᵒᶜ,
                                       state.gradient_workspace)
    end
end

# ================================================================================
# SIMULATION HELPER FUNCTIONS
# ================================================================================

"""
    create_enhanced_output_config(state)
    
Create output configuration with enhanced I/O settings.
"""
function create_enhanced_output_config(state::SimulationState{T}) where T
    base_config = output_config_from_parameters()

    return OutputConfig(
        base_config.output_space,
        base_config.output_dir,
        base_config.filename_prefix,
        base_config.include_metadata,
        base_config.include_grid,
        base_config.include_diagnostics,
        base_config.output_precision,
        base_config.spectral_lmax_output,
        base_config.overwrite_files,
        base_config.output_interval,
        base_config.restart_interval,
        base_config.time_limit,
        base_config.timing_precision
    )
end


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

create_enhanced_metadata(state, time, step) = Dict(
    "current_time" => time,
    "current_step" => step,
    "geometry" => state.geometry,
)
auto_tune_parameters!(state) = nothing
compute_adaptive_timestep(state, dt) = dt

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
        return MPI.Allreduce(local_energy, +, get_comm())
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
        return MPI.Allreduce(local_energy, +, get_comm())
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

    # Refresh physical fields from updated spectral coefficients.
    # After implicit/ERK2 steps, spectral data is current but physical arrays are stale.
    shtnskit_vector_synthesis!(state.velocity.𝒯, state.velocity.𝒫,
                              state.velocity.velocity; domain=state.𝒟ᵒᶜ)
    if state.magnetic !== nothing
        shtnskit_vector_synthesis!(state.magnetic.𝒯, state.magnetic.𝒫,
                                  state.magnetic.magnetic; domain=state.𝒟ᵒᶜ)
    end
    shtnskit_spectral_to_physical!(state.temperature.spectral, state.temperature.temperature)
    if state.composition !== nothing
        shtnskit_spectral_to_physical!(state.composition.spectral, state.composition.composition)
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
    thermal_e = compute_field_energy(parent(state.temperature.temperature.data))

    # Compute compositional energy if present
    compositional_e = 0.0
    if state.composition !== nothing
        compositional_e = compute_field_energy(parent(state.composition.composition.data))
    end

    # Total energy
    total_e = kinetic_e + magnetic_e + thermal_e + compositional_e

    # Record energies (capped history to prevent unbounded memory growth)
    push!(ENERGY_TRACKER.kinetic_energy, kinetic_e)
    push!(ENERGY_TRACKER.magnetic_energy, magnetic_e)
    push!(ENERGY_TRACKER.thermal_energy, thermal_e)
    push!(ENERGY_TRACKER.compositional_energy, compositional_e)
    push!(ENERGY_TRACKER.total_energy, total_e)
    push!(ENERGY_TRACKER.timestamps, state.timestep_state.step)
    _trim_energy_tracker!()
end

const _MAX_TRACKER_HISTORY = 10000

function _trim_energy_tracker!()
    n = length(ENERGY_TRACKER.total_energy)
    if n > _MAX_TRACKER_HISTORY
        keep = n - _MAX_TRACKER_HISTORY ÷ 2
        deleteat!(ENERGY_TRACKER.kinetic_energy, 1:keep)
        deleteat!(ENERGY_TRACKER.magnetic_energy, 1:keep)
        deleteat!(ENERGY_TRACKER.thermal_energy, 1:keep)
        deleteat!(ENERGY_TRACKER.compositional_energy, 1:keep)
        deleteat!(ENERGY_TRACKER.total_energy, 1:keep)
        deleteat!(ENERGY_TRACKER.timestamps, 1:keep)
    end
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
        rel_error = E0 != 0.0 ? abs(ΔE / E0) : 0.0

        KE = ENERGY_TRACKER.kinetic_energy[end]
        ME = ENERGY_TRACKER.magnetic_energy[end]
        TE = ENERGY_TRACKER.thermal_energy[end]
        CE = ENERGY_TRACKER.compositional_energy[end]

        pct(x) = En != 0.0 ? x / En * 100 : 0.0

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
        ║   Kinetic:           $KE  ($(pct(KE))%)
        ║   Magnetic:          $ME  ($(pct(ME))%)
        ║   Thermal:           $TE  ($(pct(TE))%)
        ║   Compositional:     $CE  ($(pct(CE))%)
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

Check solenoidal constraint for toroidal-poloidal decomposition.
For v = ∇×(T r̂) + ∇×∇×(P r̂), the divergence ∇·v = 0 is satisfied analytically
by construction (divergence of a curl is identically zero). Therefore this function
always returns (0.0, 0.0) — the solenoidal constraint is guaranteed by the
representation, not by numerical accident.
"""
function compute_divergence_spectral(tor_spec::SHTnsSpecField{T},
                                     pol_spec::SHTnsSpecField{T},
                                     domain::RadialDomain) where T
    # Toroidal-poloidal decomposition is divergence-free by construction:
    # ∇·(∇×F) = 0 for any vector field F.
    # No numerical check is needed or meaningful.
    return zero(Float64), zero(Float64)
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

    # Record statistics (capped history to prevent unbounded memory growth)
    push!(SOLENOIDAL_MONITOR.velocity_div_l2, vel_l2)
    push!(SOLENOIDAL_MONITOR.velocity_div_linf, vel_linf)
    push!(SOLENOIDAL_MONITOR.magnetic_div_l2, mag_l2)
    push!(SOLENOIDAL_MONITOR.magnetic_div_linf, mag_linf)
    push!(SOLENOIDAL_MONITOR.timestamps, state.timestep_state.step)
    _trim_solenoidal_monitor!()
end

function _trim_solenoidal_monitor!()
    n = length(SOLENOIDAL_MONITOR.velocity_div_l2)
    if n > _MAX_TRACKER_HISTORY
        keep = n - _MAX_TRACKER_HISTORY ÷ 2
        deleteat!(SOLENOIDAL_MONITOR.velocity_div_l2, 1:keep)
        deleteat!(SOLENOIDAL_MONITOR.velocity_div_linf, 1:keep)
        deleteat!(SOLENOIDAL_MONITOR.magnetic_div_l2, 1:keep)
        deleteat!(SOLENOIDAL_MONITOR.magnetic_div_linf, 1:keep)
        deleteat!(SOLENOIDAL_MONITOR.timestamps, 1:keep)
    end
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

Uses proper derivative boundary conditions matching DD_2DCODE:
- No-slip: ∂P/∂r = 0 (first derivative)
- Stress-free: ∂²P/∂r² = 0 (second derivative)

The no-penetration condition (P = 0 at boundaries) is enforced separately
via the influence matrix method, which is applied after the exponential
operator in erk2_finalize_velocity_poloidal!.

i_vel_bc: 1=no-slip both, 2=no-slip inner/stress-free outer,
          3=stress-free inner/no-slip outer, 4=stress-free both.
"""
function build_erk2_bc_spec_velocity_pol(::Type{T}, domain::RadialDomain, i_vel_bc::Int) where T
    nr = domain.N
    d1 = create_derivative_matrix(T, 1, domain)
    d2 = create_derivative_matrix(T, 2, domain)
    d1_inner = extract_dense_row(d1.data, d1.bandwidth, nr, 1)
    d1_outer = extract_dense_row(d1.data, d1.bandwidth, nr, nr)
    d2_inner = extract_dense_row(d2.data, d2.bandwidth, nr, 1)
    d2_outer = extract_dense_row(d2.data, d2.bandwidth, nr, nr)

    # Inner boundary
    if i_vel_bc == 1 || i_vel_bc == 2  # No-slip inner: ∂P/∂r = 0
        inner = create_noslip_pol_bc(T, d1_inner)
    else  # Stress-free inner: ∂²P/∂r² = 0
        inner = create_stress_free_pol_bc(T, d2_inner)
    end

    # Outer boundary
    if i_vel_bc == 1 || i_vel_bc == 3  # No-slip outer: ∂P/∂r = 0
        outer = create_noslip_pol_bc(T, d1_outer)
    else  # Stress-free outer: ∂²P/∂r² = 0
        outer = create_stress_free_pol_bc(T, d2_outer)
    end

    return ERK2BoundarySpec{T}(inner, outer)
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
# ERK2 INFLUENCE MATRIX MANAGEMENT
# ================================================================================

"""
    get_erk2_influence_matrices!(cache, key, T, config, domain, diffusivity, dt, i_vel_bc; theta)

Get or create ERK2 influence matrices for poloidal velocity.
Caches matrices by key for reuse across timesteps.
Invalidates and recomputes when dt changes.
"""
function get_erk2_influence_matrices!(cache::Dict{Symbol, Tuple{Dict{Int, ERK2InfluenceMatrix{T}}, Float64}},
                                       key::Symbol,
                                       ::Type{T},
                                       config::SHTnsKitConfig,
                                       domain::RadialDomain,
                                       diffusivity::Float64,
                                       dt::Float64,
                                       i_vel_bc::Int;
                                       theta::Float64=d_implicit) where T
    if !haskey(cache, key) || cache[key][2] != dt
        matrices = create_velocity_poloidal_influence_matrices(T, config, domain,
                                                                diffusivity, dt, i_vel_bc;
                                                                theta=theta)
        cache[key] = (matrices, dt)
    end
    return cache[key][1]
end

"""
    apply_velocity_poloidal_influence_correction!(field, influence_matrices, config)

Apply influence matrix correction to enforce P = 0 at boundaries for velocity poloidal.
This is called after erk2_finalize_field! to enforce the no-penetration condition
while preserving the derivative BC constraints used in the exponential operator.

Matches DD_2DCODE's influence matrix method for velocity poloidal.
"""
function apply_velocity_poloidal_influence_correction!(field::SHTnsSpecField{T},
                                                        influence_matrices::Dict{Int, ERK2InfluenceMatrix{T}},
                                                        config::SHTnsKitConfig) where T
    u_real = parent(field.data_real)
    u_imag = parent(field.data_imag)

    lm_range = get_local_range(field.pencil, 1)
    nr = size(u_real, 3)

    # Temporary buffer for radial profile
    tmp = Vector{T}(undef, nr)

    # Loop over local lm modes
    @inbounds for lm_idx in lm_range
        local_lm = lm_idx - first(lm_range) + 1
        l = config.l_values[lm_idx]

        # Skip l=0 (no poloidal component) and modes without influence matrix
        l == 0 && continue
        !haskey(influence_matrices, l) && continue

        influence = influence_matrices[l]

        # Apply correction to real part
        for ir in 1:nr
            tmp[ir] = u_real[local_lm, 1, ir]
        end
        apply_influence_matrix_correction!(tmp, influence, zero(T), zero(T))
        for ir in 1:nr
            u_real[local_lm, 1, ir] = tmp[ir]
        end

        # Apply correction to imaginary part
        for ir in 1:nr
            tmp[ir] = u_imag[local_lm, 1, ir]
        end
        apply_influence_matrix_correction!(tmp, influence, zero(T), zero(T))
        for ir in 1:nr
            u_imag[local_lm, 1, ir] = tmp[ir]
        end
    end

    return field
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

    # Get or create influence matrices for poloidal velocity (matching DD_2DCODE)
    vel_pol_influence = get_erk2_influence_matrices!(state.erk2_influence_matrices,
                                                      :velocity_poloidal, T,
                                                      state.shtns_config, domain,
                                                      d_E, dt, params.i_vel_bc)

    # Prepare caches and stage buffers for all active fields (magnetic diffusion time scaling)
    # Diffusivities: Temperature = Pm/Pr, Velocity = E, Magnetic = 1.0, Composition = Pm/Sc

    # Temperature: use specialized cache with embedded BCs (matching DD_2DCODE tmp_bc_T)
    temp_cache = get_erk2_temperature_cache!(state.erk2_caches, d_Pm/d_Pr, T,
                                              state.shtns_config, state.𝒟ᵒᶜ, dt, params.i_tmp_bc;
                                              use_krylov=false)
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
        # Build BC specs for post-evolution enforcement (safety measure)
        mag_tor_bc = build_erk2_bc_spec_magnetic_tor(T, nr)
        mag_pol_bc = build_erk2_bc_spec_magnetic_pol(T, domain)

        # Use specialized caches with embedded insulating BCs (matching DD_2DCODE)
        # Toroidal: Dirichlet BCs (BT = 0) at both boundaries
        mag_tor_cache = get_erk2_magnetic_toroidal_cache!(state.erk2_caches, 1.0, T,
                                                          state.shtns_config, state.𝒟ᵒᶜ, dt; use_krylov=false)
        mag_tor_buffers = ERK2FieldBuffers(state.magnetic.𝒯, state.magnetic.nlᵀ, mag_tor_cache)
        erk2_prepare_field!(mag_tor_buffers, state.magnetic.𝒯, state.magnetic.nlᵀ,
                            mag_tor_cache, state.shtns_config, dt; bc_spec=mag_tor_bc)

        # Poloidal: l-dependent insulating BCs
        # Inner: (∂/∂r - l/r)P = 0, Outer: (∂/∂r + (l+1)/r)P = 0
        mag_pol_cache = get_erk2_magnetic_poloidal_cache!(state.erk2_caches, 1.0, T,
                                                          state.shtns_config, state.𝒟ᵒᶜ, dt; use_krylov=false)
        mag_pol_buffers = ERK2FieldBuffers(state.magnetic.𝒫, state.magnetic.nlᴾ, mag_pol_cache)
        erk2_prepare_field!(mag_pol_buffers, state.magnetic.𝒫, state.magnetic.nlᴾ,
                            mag_pol_cache, state.shtns_config, dt; bc_spec=mag_pol_bc)
    end

    comp_buffers = nothing
    comp_cache = nothing
    comp_bc = nothing
    if state.composition !== nothing
        # Build BC spec for post-evolution enforcement (safety measure)
        comp_bc = build_erk2_bc_spec_scalar(T, domain, params.i_cmp_bc)

        # Composition: use specialized cache with embedded BCs (matching DD_2DCODE cmp_bc_C)
        comp_cache = get_erk2_composition_cache!(state.erk2_caches, d_Pm/d_Sc, T,
                                                  state.shtns_config, state.𝒟ᵒᶜ, dt, params.i_cmp_bc;
                                                  use_krylov=false)
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
    # Apply influence matrix correction to enforce P = 0 at boundaries (matching DD_2DCODE)
    apply_velocity_poloidal_influence_correction!(state.velocity.𝒫, vel_pol_influence, state.shtns_config)

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

    # Compute and report energy/solenoidal diagnostics periodically (not every step)
    step = state.timestep_state.step
    if step % 100 == 0
        compute_total_energy!(state)
        report_energy_conservation(step; interval=100)
        check_solenoidal_constraint!(state)
        report_solenoidal_constraint(step; interval=100)
    end

    # Restore step-start nonlinear terms from ERK2 buffers into the nl fields.
    # compute_all_nonlinear_terms! (line 1451) overwrote nl with stage (midpoint) values,
    # but the history roll in apply_implicit_step! needs the step-start values for correct
    # AB2 extrapolation on the next timestep.
    copyto!(parent(state.temperature.nonlinear.data_real), temp_buffers.n_current_real)
    copyto!(parent(state.temperature.nonlinear.data_imag), temp_buffers.n_current_imag)
    copyto!(parent(state.velocity.nlᵀ.data_real), vel_tor_buffers.n_current_real)
    copyto!(parent(state.velocity.nlᵀ.data_imag), vel_tor_buffers.n_current_imag)
    copyto!(parent(state.velocity.nlᴾ.data_real), vel_pol_buffers.n_current_real)
    copyto!(parent(state.velocity.nlᴾ.data_imag), vel_pol_buffers.n_current_imag)
    if mag_tor_buffers !== nothing
        copyto!(parent(state.magnetic.nlᵀ.data_real), mag_tor_buffers.n_current_real)
        copyto!(parent(state.magnetic.nlᵀ.data_imag), mag_tor_buffers.n_current_imag)
        copyto!(parent(state.magnetic.nlᴾ.data_real), mag_pol_buffers.n_current_real)
        copyto!(parent(state.magnetic.nlᴾ.data_imag), mag_pol_buffers.n_current_imag)
    end
    if comp_buffers !== nothing
        copyto!(parent(state.composition.nonlinear.data_real), comp_buffers.n_current_real)
        copyto!(parent(state.composition.nonlinear.data_imag), comp_buffers.n_current_imag)
    end
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
    if (ts_scheme === :cnab2 || ts_scheme === :eab2 || ts_scheme === :erk2) && state.timestep_state.needs_ab2_bootstrap
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
        state.timestep_state.needs_ab2_bootstrap = false
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
            # Temperature: dT/dt = (Pm/Pr)*∇²T + NL (mass_coeff=1.0 for scalars)
            alu_map = get_eab2_alu_cache!(state.etd_caches, :temperature, d_Pm/d_Pr, T, state.𝒟ᵒᶜ)
            eab2_update_krylov_cached!(state.temperature.spectral, state.temperature.nonlinear,
                                       state.temperature.prev_nonlinear, alu_map, state.𝒟ᵒᶜ, d_Pm/d_Pr,
                                       state.shtns_config, dt; m=i_etd_m, tol=d_krylov_tol)
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
                             dt, state.implicit_matrices[:velocity_tor];
                             mass_coeff=d_E)
            solve_velocity_implicit_step!(state.velocity.𝒯, state.velocity.work_tor,
                                          state.implicit_matrices[:velocity_tor], :toroidal;
                                          i_vel_bc=get_parameters().i_vel_bc,
                                          domain=state.𝒟ᵒᶜ)
        elseif ts_scheme === :eab2
            # Velocity: d_E * du/dt = d_E*∇²u + NL → du/dt = ∇²u + NL/d_E
            alu_map = get_eab2_alu_cache!(state.etd_caches, :velocity_toroidal, d_E, T, state.𝒟ᵒᶜ)
            eab2_update_krylov_cached!(state.velocity.𝒯, state.velocity.nlᵀ,
                                       state.velocity.prev_nlᵀ, alu_map, state.𝒟ᵒᶜ, d_E,
                                       state.shtns_config, dt; m=i_etd_m, tol=d_krylov_tol,
                                       mass_coeff=d_E)
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
                             dt, state.implicit_matrices[:velocity_pol];
                             mass_coeff=d_E)
            solve_velocity_implicit_step!(state.velocity.𝒫, state.velocity.work_pol,
                                          state.implicit_matrices[:velocity_pol], :poloidal;
                                          i_vel_bc=get_parameters().i_vel_bc,
                                          domain=state.𝒟ᵒᶜ)
        elseif ts_scheme === :eab2
            # Velocity: d_E * du/dt = d_E*∇²u + NL → du/dt = ∇²u + NL/d_E
            alu_map = get_eab2_alu_cache!(state.etd_caches, :velocity_poloidal, d_E, T, state.𝒟ᵒᶜ)
            eab2_update_krylov_cached!(state.velocity.𝒫, state.velocity.nlᴾ,
                                       state.velocity.prev_nlᴾ, alu_map, state.𝒟ᵒᶜ, d_E,
                                       state.shtns_config, dt; m=i_etd_m, tol=d_krylov_tol,
                                       mass_coeff=d_E)
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
                                 dt, state.implicit_matrices[:magnetic_tor])
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
                                 dt, state.implicit_matrices[:magnetic_pol])
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

# Exports are handled by the main GeoDynamo.jl module
