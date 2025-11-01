#!/usr/bin/env julia

"""
Example: Setting Initial Conditions for GeoDynamo Simulations

This example demonstrates how to set initial conditions for temperature,
magnetic field, velocity, and composition fields in geodynamo simulations.

Three methods are shown:
1. Loading from NetCDF files (spectral coefficients)
2. Generating random initial conditions
3. Setting analytical patterns

Run this example to see all three approaches in action.
"""

using LinearAlgebra
using Random

# Include the initial conditions module directly
include("../src/InitialConditions.jl")

println("="^70)
println("GEODYNAMO INITIAL CONDITIONS EXAMPLE")
println("="^70)

# ============================================================================
# Mock Field Structures for Testing
# ============================================================================

"""
Create mock field structures for demonstration.
In real simulations, these would be created by the main geodynamo code.
"""

# Mock spectral field structure
struct MockSpectralField{T}
    data::Matrix{T}  # [nr, nlm] - radial points × spectral modes
end

# Mock temperature field
struct MockTemperatureField{T}
    spectral::MockSpectralField{T}
    config::NamedTuple
end

# Mock magnetic field
struct MockMagneticField{T}
    toroidal::MockSpectralField{T}
    poloidal::MockSpectralField{T}
    config::NamedTuple
end

# Mock velocity field
struct MockVelocityField{T}
    toroidal::MockSpectralField{T}
    poloidal::MockSpectralField{T}
    config::NamedTuple
end

# Mock composition field
struct MockCompositionField{T}
    spectral::MockSpectralField{T}
    config::NamedTuple
end

# Create field aliases to match expected types in InitialConditions module
const SHTnsTemperatureField = MockTemperatureField
const SHTnsMagneticFields = MockMagneticField
const SHTnsVelocityFields = MockVelocityField

# ============================================================================
# Example Configuration
# ============================================================================

# Simulation parameters
nr = 64           # Number of radial points
lmax = 32         # Maximum spherical harmonic degree
nlm = (lmax + 1)^2  # Number of spectral modes
T = Float64       # Precision

# Configuration
config = (nr=nr, lmax=lmax, nlm=nlm, T=T)

println("Simulation Configuration:")
println("  Radial points: $nr")
println("  Maximum l: $lmax")
println("  Spectral modes: $nlm")
println("  Precision: $T")

# ============================================================================
# Create Mock Fields
# ============================================================================

println("\n" * "="^70)
println("CREATING FIELD STRUCTURES")
println("="^70)

# Create field structures
temp_field = MockTemperatureField(
    MockSpectralField(zeros(T, nr, nlm)),
    config
)

magnetic_field = MockMagneticField(
    MockSpectralField(zeros(T, nr, nlm)),  # toroidal
    MockSpectralField(zeros(T, nr, nlm)),  # poloidal
    config
)

velocity_field = MockVelocityField(
    MockSpectralField(zeros(T, nr, nlm)),  # toroidal
    MockSpectralField(zeros(T, nr, nlm)),  # poloidal
    config
)

composition_field = MockCompositionField(
    MockSpectralField(zeros(T, nr, nlm)),
    config
)

println("Field structures created")

# ============================================================================
# Method 1: Random Initial Conditions
# ============================================================================

println("\n" * "="^70)
println("METHOD 1: RANDOM INITIAL CONDITIONS")
println("="^70)

println("\n1.1 Random Temperature Field")
generate_random_initial_conditions!(temp_field, :temperature,
                                   amplitude=0.1,
                                   modes_range=1:10,
                                   seed=42)

# Check the results
temp_max = maximum(abs.(temp_field.spectral.data))
temp_l0 = temp_field.spectral.data[end÷2, 1]  # Middle radial point, l=0 mode
println("  Max amplitude: $(round(temp_max, digits=4))")
println("  l=0 mode (mid-radius): $(round(temp_l0, digits=4))")

println("\n1.2 Random Magnetic Field")
generate_random_initial_conditions!(magnetic_field, :magnetic,
                                   amplitude=0.01,
                                   modes_range=1:15,
                                   seed=123)

# Check the results
mag_tor_max = maximum(abs.(magnetic_field.toroidal.data))
mag_pol_max = maximum(abs.(magnetic_field.poloidal.data))
println("  Toroidal max: $(round(mag_tor_max, digits=4))")
println("  Poloidal max: $(round(mag_pol_max, digits=4))")

println("\n1.3 Random Velocity Field")
generate_random_initial_conditions!(velocity_field, :velocity,
                                   amplitude=0.001,
                                   modes_range=1:20)

vel_tor_max = maximum(abs.(velocity_field.toroidal.data))
vel_pol_max = maximum(abs.(velocity_field.poloidal.data))
println("  Toroidal max: $(round(vel_tor_max, digits=6))")
println("  Poloidal max: $(round(vel_pol_max, digits=6))")

println("\n1.4 Random Composition Field")
generate_random_initial_conditions!(composition_field, :composition,
                                   amplitude=0.05,
                                   modes_range=1:8)

comp_max = maximum(abs.(composition_field.spectral.data))
comp_l0 = composition_field.spectral.data[nr÷2, 1]
println("  Max amplitude: $(round(comp_max, digits=4))")
println("  l=0 mode (mid-radius): $(round(comp_l0, digits=4))")

# ============================================================================
# Method 2: Analytical Initial Conditions
# ============================================================================

println("\n" * "="^70)
println("METHOD 2: ANALYTICAL INITIAL CONDITIONS")
println("="^70)

# Reset fields
fill!(temp_field.spectral.data, 0.0)
fill!(magnetic_field.toroidal.data, 0.0)
fill!(magnetic_field.poloidal.data, 0.0)
fill!(velocity_field.toroidal.data, 0.0)
fill!(velocity_field.poloidal.data, 0.0)
fill!(composition_field.spectral.data, 0.0)

println("\n2.1 Conductive Temperature Profile")
set_analytical_initial_conditions!(temp_field, :temperature, :conductive,
                                  amplitude=1.0)

temp_bottom = temp_field.spectral.data[1, 1]    # Bottom (r=0)
temp_top = temp_field.spectral.data[end, 1]     # Top (r=1)
println("  Bottom temperature: $(round(temp_bottom, digits=3))")
println("  Top temperature: $(round(temp_top, digits=3))")

println("\n2.2 Hot Thermal Blob")
set_analytical_initial_conditions!(temp_field, :temperature, :hot_blob,
                                  amplitude=0.5,
                                  r_center=0.3, blob_width=0.15)

temp_blob_center = temp_field.spectral.data[Int(0.3*nr)+1, 1]
println("  Temperature at blob center: $(round(temp_blob_center, digits=3))")

println("\n2.3 Dipolar Magnetic Field")
set_analytical_initial_conditions!(magnetic_field, :magnetic, :dipole,
                                  amplitude=1.0)

dipole_strength = magnetic_field.poloidal.data[nr÷2, 3]  # l=1, m=0 mode
println("  Dipole strength (l=1,m=0): $(round(dipole_strength, digits=3))")

println("\n2.4 Uniform Magnetic Field")
# Reset and set uniform field
fill!(magnetic_field.poloidal.data, 0.0)
set_analytical_initial_conditions!(magnetic_field, :magnetic, :uniform_field,
                                  amplitude=0.5, direction=:z)

uniform_strength = magnetic_field.poloidal.data[nr÷2, 1]  # l=0, m=0 mode
println("  Uniform field strength: $(round(uniform_strength, digits=3))")

println("\n2.5 Small Convective Velocities")
set_analytical_initial_conditions!(velocity_field, :velocity, :convective,
                                  amplitude=0.01)

conv_tor = maximum(abs.(velocity_field.toroidal.data[nr÷2, 2:10]))
conv_pol = maximum(abs.(velocity_field.poloidal.data[nr÷2, 2:10]))
println("  Max convective toroidal: $(round(conv_tor, digits=6))")
println("  Max convective poloidal: $(round(conv_pol, digits=6))")

println("\n2.6 Stratified Composition")
set_analytical_initial_conditions!(composition_field, :composition, :stratified,
                                  bottom_composition=0.3, top_composition=0.1)

comp_bottom = composition_field.spectral.data[1, 1]
comp_top = composition_field.spectral.data[end, 1]
println("  Bottom composition: $(round(comp_bottom, digits=3))")
println("  Top composition: $(round(comp_top, digits=3))")

println("\n2.7 Compositional Blob")
set_analytical_initial_conditions!(composition_field, :composition, :blob,
                                  r_center=0.2, blob_width=0.1,
                                  blob_composition=0.8)

blob_comp = composition_field.spectral.data[Int(0.2*nr)+1, 1]
println("  Blob composition: $(round(blob_comp, digits=3))")

# ============================================================================
# Method 3: Loading from Files (Demonstration)
# ============================================================================

println("\n" * "="^70)
println("METHOD 3: LOADING FROM FILES")
println("="^70)

println("\nNote: NetCDF loading is not fully implemented in this example.")
println("The following shows the interface that would be used:")
println()

# This would load actual initial conditions from NetCDF files
example_files = [
    ("temperature_IC.nc", :temperature),
    ("magnetic_IC.nc", :magnetic),
    ("velocity_IC.nc", :velocity),
    ("composition_IC.nc", :composition)
]

for (filename, field_type) in example_files
    println("  load_initial_conditions!($(field_type)_field, :$field_type, \"$filename\")")
end

println("\nExpected NetCDF file format:")
println("  - For scalar fields: spectral_coefficients[nr, nlm]")
println("  - For vector fields: toroidal_coefficients[nr, nlm], poloidal_coefficients[nr, nlm]")
println("  - Coordinate arrays: radial_grid[nr], lm_indices[nlm]")
println("  - Metadata: lmax, time, description, units")

# ============================================================================
# Example Output and Validation
# ============================================================================

println("\n" * "="^70)
println("SUMMARY AND VALIDATION")
println("="^70)

println("\nField Energy Summary:")

# Calculate simple energy measures
temp_energy = sum(abs2.(temp_field.spectral.data))
mag_energy = sum(abs2.(magnetic_field.toroidal.data)) + sum(abs2.(magnetic_field.poloidal.data))
vel_energy = sum(abs2.(velocity_field.toroidal.data)) + sum(abs2.(velocity_field.poloidal.data))
comp_energy = sum(abs2.(composition_field.spectral.data))

println("  Temperature energy: $(round(temp_energy, digits=6))")
println("  Magnetic energy: $(round(mag_energy, digits=6))")
println("  Velocity energy: $(round(vel_energy, digits=8))")
println("  Composition energy: $(round(comp_energy, digits=6))")

println("\nSpectral Content Analysis:")

# Check mode distribution
function analyze_spectral_content(data, name)
    total_energy = sum(abs2.(data))
    l0_energy = sum(abs2.(data[:, 1]))  # l=0, m=0 mode
    low_l_energy = sum(abs2.(data[:, 1:min(10, end)]))  # First 10 modes

    println("  $name:")
    println("    Total energy: $(round(total_energy, digits=6))")
    println("    l=0 fraction: $(round(l0_energy/total_energy*100, digits=1))%")
    println("    Low-l fraction: $(round(low_l_energy/total_energy*100, digits=1))%")
end

analyze_spectral_content(temp_field.spectral.data, "Temperature")
analyze_spectral_content(magnetic_field.poloidal.data, "Magnetic (poloidal)")
analyze_spectral_content(velocity_field.toroidal.data, "Velocity (toroidal)")
analyze_spectral_content(composition_field.spectral.data, "Composition")

# ============================================================================
# Saving Initial Conditions (Demonstration)
# ============================================================================

println("\n" * "="^70)
println("SAVING INITIAL CONDITIONS")
println("="^70)

println("The following demonstrates how to save generated initial conditions:")
println()

save_files = [
    ("my_temperature_IC.nc", :temperature),
    ("my_magnetic_IC.nc", :magnetic),
    ("my_velocity_IC.nc", :velocity),
    ("my_composition_IC.nc", :composition)
]

for (filename, field_type) in save_files
    field = field_type == :temperature ? temp_field :
            field_type == :magnetic ? magnetic_field :
            field_type == :velocity ? velocity_field :
            composition_field

    save_initial_conditions(field, field_type, filename)
end

# ============================================================================
# Usage Tips and Best Practices
# ============================================================================

println("\n" * "="^70)
println("USAGE TIPS AND BEST PRACTICES")
println("="^70)

tips = [
    "1. Random Amplitudes: Start with small amplitudes (0.001-0.1) to avoid instability",
    "2. Mode Range: Limit random modes to low l (1-20) for realistic initial conditions",
    "3. Magnetic Fields: Use dipolar patterns (l=1) as base, add small random perturbations",
    "4. Temperature: Include both conductive profile + small convective perturbations",
    "5. Composition: Keep within [0,1] range, use stratified background + blobs",
    "6. Velocity: Start with very small amplitudes (< 0.01) to allow natural development",
    "7. Reproducibility: Use seeds for random generation in production runs",
    "8. Validation: Check energy levels and spectral content before starting simulation"
]

for tip in tips
    println("  $tip")
end

println("\n" * "="^70)
println("INITIAL CONDITIONS EXAMPLE COMPLETE")
println("="^70)

println("\nAll initial conditions methods demonstrated successfully!")
println("\nThis example shows how to:")
println("  • Generate random initial conditions with controlled amplitude and modes")
println("  • Set analytical patterns (conductive, dipolar, stratified, etc.)")
println("  • Load and save initial conditions from/to NetCDF files")
println("  • Validate and analyze spectral content")
println("\nYou can now use these methods in your geodynamo simulations!")

println("\nNext steps:")
println("  1. Integrate InitialConditions module into your main simulation")
println("  2. Create NetCDF initial condition files for your specific cases")
println("  3. Experiment with different amplitudes and mode ranges")
println("  4. Validate that initial conditions lead to stable simulations")

# Example function showing how to integrate with main simulation
println("\n" * "-"^50)
println("INTEGRATION EXAMPLE")
println("-"^50)

print("""
# Example integration in main simulation code:

using InitialConditions

function setup_simulation()
    # Create fields (normally done by main geodynamo code)
    temp_field, magnetic_field, velocity_field = create_fields(config)

    # Method 1: Random initial conditions
    generate_random_initial_conditions!(temp_field, :temperature, amplitude=0.1, seed=42)
    generate_random_initial_conditions!(magnetic_field, :magnetic, amplitude=0.01, modes_range=1:10)
    generate_random_initial_conditions!(velocity_field, :velocity, amplitude=0.001)

    # Method 2: Analytical + random perturbations
    set_analytical_initial_conditions!(temp_field, :temperature, :conductive)
    generate_random_initial_conditions!(temp_field, :temperature, amplitude=0.01)  # Add perturbations

    # Method 3: Load from files
    if isfile("initial_conditions.nc")
        load_initial_conditions!(magnetic_field, :magnetic, "initial_conditions.nc")
    end

    return temp_field, magnetic_field, velocity_field
end
""")

println()
println("="^70)