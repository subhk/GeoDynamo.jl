#!/usr/bin/env julia

"""
SHTnsKit integration demo using the current GeoDynamo.jl transform API.

This example demonstrates:
1. creating an SHTnsKit-backed transform configuration
2. querying feature/version information
3. scalar synthesis/analysis on a temperature field
4. vector synthesis/analysis on a velocity field
5. energy spectrum and rotation utilities on raw coefficient matrices

Run with:
    julia --project examples/shtnskit_integration_demo.jl
"""

using GeoDynamo
using LinearAlgebra
using Random

function local_roundtrip_error(before_real, before_imag, after_real, after_imag)
    return norm(after_real .- before_real) + norm(after_imag .- before_imag)
end

function demo_scalar_transforms(cfg, domain)
    println("=== Scalar transform roundtrip ===")

    field = create_shtns_temperature_field(Float64, cfg, domain)
    set_temperature_ic!(field, domain; perturbation_amplitude = 1e-3)

    before_real = copy(parent(field.spectral.data_real))
    before_imag = copy(parent(field.spectral.data_imag))

    shtnskit_spectral_to_physical!(field.spectral, field.work_physical)
    shtnskit_physical_to_spectral!(field.work_physical, field.work_spectral)

    error = local_roundtrip_error(
        before_real,
        before_imag,
        parent(field.work_spectral.data_real),
        parent(field.work_spectral.data_imag)
    )

    println("Local scalar roundtrip error: $(round(error, sigdigits=4))")
end

function demo_vector_transforms(cfg, domain)
    println("\n=== Vector transform roundtrip ===")

    fields = create_shtns_velocity_fields(Float64, cfg, domain)
    Random.seed!(42)
    randomize_vector_field!(fields; amplitude = 1e-4, lmax = 4, domain = domain)

    before_t_real = copy(parent(fields.𝒯.data_real))
    before_t_imag = copy(parent(fields.𝒯.data_imag))
    before_p_real = copy(parent(fields.𝒫.data_real))
    before_p_imag = copy(parent(fields.𝒫.data_imag))

    shtnskit_vector_synthesis!(fields.𝒯, fields.𝒫, fields.velocity; domain = domain)
    shtnskit_vector_analysis!(fields.velocity, fields.work_tor, fields.work_pol)

    tor_error = local_roundtrip_error(
        before_t_real,
        before_t_imag,
        parent(fields.work_tor.data_real),
        parent(fields.work_tor.data_imag)
    )
    pol_error = local_roundtrip_error(
        before_p_real,
        before_p_imag,
        parent(fields.work_pol.data_real),
        parent(fields.work_pol.data_imag)
    )

    println("Local toroidal roundtrip error: $(round(tor_error, sigdigits=4))")
    println("Local poloidal roundtrip error: $(round(pol_error, sigdigits=4))")
end

function demo_spectrum_and_rotation(cfg)
    println("\n=== Spectrum and rotation helpers ===")

    alm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    alm[1, 1] = 1.0 + 0im
    alm[3, 1] = 0.15 + 0im
    alm[4, 3] = 0.05 - 0.02im

    spectrum = compute_scalar_energy_spectrum(cfg, alm)
    rotated = similar(alm)
    rotate_field_z!(cfg, alm, π / 4; alm_out = rotated)

    println("Total scalar energy: $(round(sum(spectrum), sigdigits=4))")
    println("Energy in l=0       : $(round(spectrum[1], sigdigits=4))")
    println("Rotation preserved norm: $(isapprox(norm(alm), norm(rotated); rtol=1e-10, atol=1e-12))")
end

function main(; lmax = 32, mmax = 32, nlat = 64, nlon = 128, nr = 16)
    cfg = GeoDynamo.create_shtnskit_config(
        lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr)
    domain = create_shell_radial_domain(nr)

    info = get_shtnskit_version_info()
    println("SHTnsKit version: $(info.version)")
    println("Distributed transforms available: $(info.has_distributed_transforms)")
    println("QST transforms available        : $(info.has_qst_transforms)")
    println("Energy helpers available        : $(info.has_energy_functions)")
    println("Rotation helpers available      : $(info.has_rotation_functions)")

    demo_scalar_transforms(cfg, domain)
    demo_vector_transforms(cfg, domain)
    demo_spectrum_and_rotation(cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
