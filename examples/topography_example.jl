# ================================================================================
# Topography Coupling Example
# ================================================================================
#
# This example demonstrates how to use the boundary topography coupling feature
# in Geodynamo.jl simulations.
#
# Topography effects are linearized assuming small deviations from a spherical
# boundary:  r = r_b + ε*h(θ,φ)  where ε << 1
#
# ================================================================================

using GeoDynamo

# ================================================================================
# QUICK START: Enable topography in your simulation
# ================================================================================

# Method 1: Using parameters (recommended for production runs)
function setup_topography_via_params()
    params = GeoDynamoParameters(
        # ... your other parameters ...

        # Enable topography coupling
        b_topography_enabled = true,
        d_topo_epsilon = 0.01,           # Topography amplitude parameter ε

        # Choose which fields get topography corrections
        b_topo_velocity = true,          # Velocity BCs
        b_topo_magnetic = true,          # Magnetic BCs
        b_topo_thermal = true,           # Temperature BCs

        # Term selection (for performance tuning)
        b_topo_slope_terms = true,       # Include ∇h coupling
        b_topo_shift_terms = true,       # Include h shift terms

        # Optional: Stefan condition for ICB evolution
        b_stefan_enabled = false,
        d_stefan_number = 1.0,

        # Topography files (NetCDF)
        s_topo_cmb_file = "cmb_topography.nc",  # Or "" for no CMB topo
        s_topo_icb_file = "",                   # Or "" for no ICB topo
    )

    return params
end

# Method 2: Using the runtime API (for interactive use)
function setup_topography_runtime()
    # Enable with default settings
    enable_topography!()

    # Or customize
    enable_topography!(
        epsilon = 0.02,        # 2% amplitude
        velocity = true,
        magnetic = true,
        thermal = true,
        stefan = false,
        slope_terms = true,
        shift_terms = true
    )

    # Check if enabled
    if is_topography_enabled()
        println("Topography coupling is active")
        config = get_topography_config()
        println("  ε = $(config.epsilon)")
    end

    # Disable when needed
    # disable_topography!()
end

# ================================================================================
# Creating Topography Data
# ================================================================================

function create_topography_examples()
    # Example 1: Uniform topography (constant offset)
    cmb_uniform = create_uniform_topography(
        0.1,                    # Amplitude
        1.0,                    # CMB radius
        OUTER_BOUNDARY;
        lmax = 32
    )

    # Example 2: Single spherical harmonic (e.g., Y_2^0 for ellipsoidal shape)
    cmb_y20 = create_spherical_harmonic_topography(
        2, 0,                   # l=2, m=0
        0.05,                   # Amplitude
        1.0,                    # Radius
        OUTER_BOUNDARY;
        lmax = 32
    )

    # Example 3: Random topography with power spectrum
    # Power spectrum ~ l^(-2) (red spectrum, common for planetary surfaces)
    cmb_random = create_random_topography(
        l -> 0.01 / max(l, 1)^2,  # Power spectrum function
        1.0,                       # Radius
        OUTER_BOUNDARY;
        lmax = 32,
        seed = 42                  # For reproducibility
    )

    # Example 4: Load from file
    # cmb_from_file = load_topography_from_file("my_topography.nc", OUTER_BOUNDARY)

    # Create TopographyData combining ICB and CMB
    topo_data = create_topography_data(
        cmb_coeffs = get_topography_coefficients(cmb_y20),
        cmb_radius = 1.0,
        # icb_coeffs = ...,  # Optional ICB topography
        # icb_radius = 0.35,
        lmax = 32,
        epsilon = 0.01
    )

    return topo_data
end

# ================================================================================
# Gaunt Tensor Precomputation (for efficiency)
# ================================================================================

function setup_gaunt_tensors(lmax_field::Int, lmax_topo::Int)
    # Create Gaunt tensor cache
    gaunt_cache = GauntTensorCache{Float64}(lmax_field, lmax_topo)

    # Precompute all non-zero Gaunt integrals
    # This is expensive but only needs to be done once!
    println("Precomputing Gaunt tensors (this may take a while for large lmax)...")
    @time precompute_gaunt_tensors!(gaunt_cache; verbose=true)

    return gaunt_cache
end

# ================================================================================
# Applying Topography Corrections During Simulation
# ================================================================================

function apply_topography_in_timestep(sim_state, topo_data, gaunt_cache)
    # Get configuration
    config = get_topography_config()

    if !config.enabled
        return
    end

    # Option 1: Apply all corrections at once
    apply_all_topography_corrections!(
        (velocity = sim_state.velocity,
         magnetic = sim_state.magnetic,
         temperature = sim_state.temperature),
        topo_data
    )

    # Option 2: Apply individually (for more control)
    # apply_velocity_topography_correction!(sim_state.velocity, topo_data, config)
    # apply_magnetic_topography_correction!(sim_state.magnetic, topo_data, config)
    # apply_thermal_topography_correction!(sim_state.temperature, topo_data, config)
end

# ================================================================================
# Stefan Condition for ICB Evolution (Advanced)
# ================================================================================

function setup_stefan_evolution(ri::Float64, lmax::Int)
    # Create Stefan state for ICB phase change
    stefan = StefanState(
        lmax = lmax,
        ri = ri,
        k_ic = 1.0,           # Inner core conductivity
        k_oc = 1.0,           # Outer core conductivity
        rho = 1.0,            # Density
        L = 1.0,              # Latent heat
        use_clapeyron = false # Optional Clapeyron correction
    )

    return stefan
end

function evolve_icb_topography!(stefan, dt, velocity, T_ic, T_oc; topo_data=nothing, gaunt=nothing, config=nothing)
    # Update ICB topography based on Stefan condition:
    # ε ∂_t h = u_n + (k_ic ∂_n T_ic - k ∂_n T) / (ρ L)

    update_icb_topography!(
        stefan, dt, velocity, T_ic, T_oc;
        topo_data = topo_data,
        gaunt = gaunt,
        config = config
    )

    # Print diagnostics
    diag = get_stefan_diagnostics(stefan)
    println("ICB topography RMS: $(diag["topography_rms"])")
    println("Growth rate RMS: $(diag["growth_rate_rms"])")
end

# ================================================================================
# Complete Example: Setting up a simulation with topography
# ================================================================================

function example_simulation_with_topography()
    println("="^60)
    println("TOPOGRAPHY COUPLING EXAMPLE")
    println("="^60)

    # 1. Create parameters with topography enabled
    params = GeoDynamoParameters(
        i_L = 32, i_M = 32,
        i_N = 64,
        d_rratio = 0.35,
        b_topography_enabled = true,
        d_topo_epsilon = 0.01,
    )

    println("\n1. Parameters configured with topography:")
    println("   ε = $(params.d_topo_epsilon)")
    println("   Velocity coupling: $(params.b_topo_velocity)")
    println("   Magnetic coupling: $(params.b_topo_magnetic)")
    println("   Thermal coupling: $(params.b_topo_thermal)")

    # 2. Create topography data
    println("\n2. Creating CMB topography (Y_2^0 pattern)...")
    topo = create_topography_data(
        cmb_radius = 1.0,
        lmax = params.i_L,
        epsilon = params.d_topo_epsilon
    )

    # Add Y_2^0 topography manually
    cmb_field = create_spherical_harmonic_topography(
        2, 0, 0.1, 1.0, OUTER_BOUNDARY; lmax=params.i_L
    )
    topo.cmb = cmb_field

    println("   CMB topography RMS: $(topo.cmb.rms_amplitude)")

    # 3. Initialize Gaunt tensors
    println("\n3. Initializing Gaunt tensor cache...")
    initialize_gaunt_cache!(topo, params.i_L; precompute=true)

    # 4. Print summary
    println("\n4. Topography summary:")
    print_topography_summary(topo)

    println("\n" * "="^60)
    println("Setup complete! Topography corrections will be applied to BCs.")
    println("="^60)

    return params, topo
end

# ================================================================================
# Run example
# ================================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Run the example when this file is executed directly
    example_simulation_with_topography()
end
