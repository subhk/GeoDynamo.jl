# Simple sanity check for stress-free flux corrections (including topography RHS).

using GeoDynamo

function check_tau_flux_correction()
    domain = GeoDynamo.create_radial_domain(nr=8, rratio=0.35, kl=2)
    ∂r = GeoDynamo.create_derivative_matrix(1, domain)

    nr = domain.N
    spec_real = zeros(Float64, 1, 1, nr)
    spec_imag = zeros(Float64, 1, 1, nr)

    r = domain.r[:, 4]
    spec_real[1, 1, :] = r .^ 2

    boundary_values = zeros(Float64, 2, 1)
    boundary_values[1, 1] = 0.25
    boundary_values[2, 1] = -0.1

    GeoDynamo.apply_velocity_flux_bc_tau!(spec_real, spec_imag, 1, 1,
                                          true, true, boundary_values,
                                          ∂r, domain, 1:nr, true)

    profile = vec(spec_real[1, 1, :])
    dprofile = ∂r * profile

    target_inner = (r[1] < 1e-14 ? 0.0 : profile[1] / r[1]) + boundary_values[1, 1]
    target_outer = profile[end] / r[end] + boundary_values[2, 1]

    println("inner flux error: ", abs(dprofile[1] - target_inner))
    println("outer flux error: ", abs(dprofile[end] - target_outer))
end

check_tau_flux_correction()
