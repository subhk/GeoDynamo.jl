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
using GeoDynamo.bcs:
    INNER_BOUNDARY,
    OUTER_BOUNDARY,
    GauntTensorCache,
    StefanState,
    apply_all_topography_corrections!,
    enable_topography!,
    get_topography_config,
    is_topography_enabled,
    precompute_gaunt_tensors!,
    update_icb_topography!
using GeoDynamo.bcs.topography:
    create_random_topography,
    create_spherical_harmonic_topography,
    create_topography_data,
    create_uniform_topography,
    get_stefan_diagnostics,
    get_topography_coefficients,
    initialize_gaunt_cache!,
    print_topography_summary

# ================================================================================
# QUICK START: Enable topography in your simulation
# ================================================================================

# Method 1: Set topography directly on the high-level model.
function setup_topography_model(;
    nr=64,
    nr_inner=16,
    lmax=32,
    mmax=32,
    nlat=64,
    nlon=128,
    radius_ratio=0.35,
    topography_epsilon=0.01,
    kwargs...,
)
    grid = SphericalShellGrid(
        lmax=lmax,
        mmax=mmax,
        nlat=nlat,
        nlon=nlon,
        nr=nr,
        nr_inner=nr_inner,
        r_inner=radius_ratio,
    )

    return GeodynamoModel(
        grid;
        topography_enabled=true,
        topography_epsilon=topography_epsilon,
        topography_degree=lmax,
        include_topography_velocity=true,
        include_topography_magnetic=true,
        include_topography_thermal=true,
        include_topography_slope_terms=true,
        include_topography_shift_terms=true,
        stefan_enabled=false,
        stefan_number=1.0,
        ocb_topography_file="",
        icb_topography_file="",
        kwargs...,
    )
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

function example_simulation_with_topography(; run=false)
    println("="^60)
    println("TOPOGRAPHY COUPLING EXAMPLE")
    println("="^60)

    # 1. Create a model with topography enabled
    model = setup_topography_model()
    simulation = Simulation(model; Δt=1e-4, max_steps=10)
    params = model.state.parameters

    println("\n1. Model configured with topography:")
    println("   ε = $(params.topography_epsilon)")
    println("   Velocity coupling: $(params.include_topography_velocity)")
    println("   Magnetic coupling: $(params.include_topography_magnetic)")
    println("   Thermal coupling: $(params.include_topography_thermal)")

    # 2. Create topography data
    println("\n2. Creating CMB topography (Y_2^0 pattern)...")
    topo = create_topography_data(
        cmb_radius = 1.0,
        lmax = params.lmax,
        epsilon = params.topography_epsilon
    )

    # Add Y_2^0 topography manually
    cmb_field = create_spherical_harmonic_topography(
        2, 0, 0.1, 1.0, OUTER_BOUNDARY; lmax=params.lmax
    )
    topo.cmb = cmb_field

    println("   CMB topography RMS: $(topo.cmb.rms_amplitude)")

    # 3. Initialize Gaunt tensors
    println("\n3. Initializing Gaunt tensor cache...")
    initialize_gaunt_cache!(topo, params.lmax; precompute=true)

    # 4. Print summary
    println("\n4. Topography summary:")
    print_topography_summary(topo)

    println("\n" * "="^60)
    println("Setup complete! Topography corrections will be applied to BCs.")
    println("="^60)

    if run
        run!(simulation)
    end

    return simulation, topo
end

# ================================================================================
# Run example
# ================================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Run the example when this file is executed directly
    example_simulation_with_topography()
end

main() = example_simulation_with_topography()
