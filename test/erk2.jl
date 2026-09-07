using Test
using LinearAlgebra

@testset "ERK2 staged update" begin
    # Need nr >= 3 because ERK2 applies zero BCs at indices 1 and nr.
    # With nr=2, both points are boundary points and get zeroed.
    # With nr=3, index 2 is an interior point that retains its value.
    nr = 3
    dom = GeoDynamo.create_radial_domain(nr)
    cfg = GeoDynamo.create_shtnskit_config(
        lmax = 0, mmax = 0, nlat = 2, nlon = 2, nr = dom.N, optimize_decomp = false)

    u_field = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    nl_field = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)

    u0 = 0.3
    c = 0.2
    # Initialize interior radial point (index 2) - boundary points (1 and nr) get zeroed by BCs
    # Array dimensions are (lmax+1, mmax+1, nr) = (1, 1, 3) for lmax=0, mmax=0, nr=3
    parent(u_field.data_real)[1, 1, 1] = 0.0
    parent(u_field.data_imag)[1, 1, 1] = 0.0
    parent(nl_field.data_real)[1, 1, 1] = 0.0
    parent(nl_field.data_imag)[1, 1, 1] = 0.0
    # Interior point with test values
    parent(u_field.data_real)[1, 1, 2] = u0
    parent(u_field.data_imag)[1, 1, 2] = 0.0
    parent(nl_field.data_real)[1, 1, 2] = c
    parent(nl_field.data_imag)[1, 1, 2] = 0.0
    # Third radial point (outer boundary)
    parent(u_field.data_real)[1, 1, 3] = 0.0
    parent(u_field.data_imag)[1, 1, 3] = 0.0
    parent(nl_field.data_real)[1, 1, 3] = 0.0
    parent(nl_field.data_imag)[1, 1, 3] = 0.0

    dt = 0.1
    lambda = 0.5
    z = -lambda * dt

    E_half_scalar = exp(z / 2)
    E_full_scalar = exp(z)
    phi1_half_scalar = (E_half_scalar - 1) / (z / 2)
    phi1_full_scalar = (E_full_scalar - 1) / z
    phi2_full_scalar = (E_full_scalar - 1 - z) / (z^2)

    # Create nr×nr diagonal matrices (each radial point evolves independently with same eigenvalue)
    E_half_mat = E_half_scalar * Matrix{Float64}(I, nr, nr)
    E_full_mat = E_full_scalar * Matrix{Float64}(I, nr, nr)
    phi1_half_mat = phi1_half_scalar * Matrix{Float64}(I, nr, nr)
    phi1_full_mat = phi1_full_scalar * Matrix{Float64}(I, nr, nr)
    phi2_full_mat = phi2_full_scalar * Matrix{Float64}(I, nr, nr)

    cache = GeoDynamo.ERK2Cache{Float64}(
        dt,
        [cfg.l_values[1]],
        [E_half_mat],
        [E_full_mat],
        [phi1_half_mat],
        [phi1_full_mat],
        [phi2_full_mat],
        false,
        20,
        1e-8,
        true
    )

    buffers = GeoDynamo.ERK2FieldBuffers(u_field, nl_field, cache)
    GeoDynamo.erk2_prepare_field!(buffers, u_field, nl_field, cache, cfg, dt)
    GeoDynamo.erk2_apply_stage!(buffers, u_field)
    GeoDynamo.erk2_store_stage_nonlinear!(buffers, nl_field)
    GeoDynamo.erk2_finalize_field!(buffers, u_field, cache, cfg, dt)

    # Test interior point (index 2) - boundary points are zeroed by BCs
    u_real = parent(u_field.data_real)[1, 1, 2]
    expected = exp(-lambda * dt) * u0 + (1 - exp(-lambda * dt)) * c / lambda
    @test u_real ≈ expected atol=1e-4
    @test parent(u_field.data_imag)[1, 1, 2] ≈ 0.0 atol=1e-4

    # Some governing equations carry a mass coefficient on the time
    # derivative.  After dividing by that coefficient, ERK2 must apply the
    # same scale to both nonlinear evaluations while retaining the raw values
    # in its buffers (the solver restores those raw nonlinear fields after the
    # staged evaluation).
    forcing_scale = 4.0
    parent(u_field.data_real) .= 0.0
    parent(u_field.data_imag) .= 0.0
    parent(nl_field.data_real) .= 0.0
    parent(nl_field.data_imag) .= 0.0
    parent(u_field.data_real)[1, 1, 2] = u0
    parent(nl_field.data_real)[1, 1, 2] = c

    buffers_scaled = GeoDynamo.ERK2FieldBuffers(u_field, nl_field, cache)
    GeoDynamo.erk2_prepare_field!(
        buffers_scaled, u_field, nl_field, cache, cfg, dt;
        nonlinear_scale = forcing_scale)
    @test buffers_scaled.n_current_real[1, 1, 2] == c
    GeoDynamo.erk2_apply_stage!(buffers_scaled, u_field)
    GeoDynamo.erk2_store_stage_nonlinear!(buffers_scaled, nl_field)
    GeoDynamo.erk2_finalize_field!(
        buffers_scaled, u_field, cache, cfg, dt;
        nonlinear_scale = forcing_scale)

    expected_scaled = exp(-lambda * dt) * u0 +
                      (1 - exp(-lambda * dt)) * forcing_scale * c / lambda
    @test parent(u_field.data_real)[1, 1, 2] ≈ expected_scaled atol=1e-4

    # Scenario 2: linear nonlinearity N(u) = beta * u requiring stage recomputation
    u0_linear = 0.45
    beta = 0.15

    # Initialize boundary points to zero
    parent(u_field.data_real)[1, 1, 1] = 0.0
    parent(u_field.data_imag)[1, 1, 1] = 0.0
    parent(nl_field.data_real)[1, 1, 1] = 0.0
    parent(nl_field.data_imag)[1, 1, 1] = 0.0
    # Interior point with test values
    parent(u_field.data_real)[1, 1, 2] = u0_linear
    parent(u_field.data_imag)[1, 1, 2] = 0.0
    parent(nl_field.data_real)[1, 1, 2] = beta * u0_linear
    parent(nl_field.data_imag)[1, 1, 2] = 0.0
    # Outer boundary point
    parent(u_field.data_real)[1, 1, 3] = 0.0
    parent(u_field.data_imag)[1, 1, 3] = 0.0
    parent(nl_field.data_real)[1, 1, 3] = 0.0
    parent(nl_field.data_imag)[1, 1, 3] = 0.0

    buffers_linear = GeoDynamo.ERK2FieldBuffers(u_field, nl_field, cache)

    GeoDynamo.erk2_prepare_field!(buffers_linear, u_field, nl_field, cache, cfg, dt)
    GeoDynamo.erk2_apply_stage!(buffers_linear, u_field)

    # Emulate stage nonlinear evaluation: N(u_stage) = beta * u_stage
    u_stage = parent(u_field.data_real)[1, 1, 2]
    parent(nl_field.data_real)[1, 1, 2] = beta * u_stage
    # Boundary points remain zero
    parent(nl_field.data_real)[1, 1, 1] = 0.0
    parent(nl_field.data_real)[1, 1, 3] = 0.0
    GeoDynamo.erk2_store_stage_nonlinear!(buffers_linear, nl_field)

    GeoDynamo.erk2_finalize_field!(buffers_linear, u_field, cache, cfg, dt)
    u_linear = parent(u_field.data_real)[1, 1, 2]
    expected_linear = exp((beta - lambda) * dt) * u0_linear

    @test u_linear ≈ expected_linear atol=1e-4
    @test parent(u_field.data_imag)[1, 1, 2] ≈ 0.0 atol=1e-4

    # Repeat the state-dependent forcing case with a non-unit mass scaling.
    # Unlike the constant-forcing case above, N(stage) - N(initial) is nonzero,
    # so this specifically guards the scaled phi2 correction in finalize.
    parent(u_field.data_real) .= 0.0
    parent(u_field.data_imag) .= 0.0
    parent(nl_field.data_real) .= 0.0
    parent(nl_field.data_imag) .= 0.0
    parent(u_field.data_real)[1, 1, 2] = u0_linear
    parent(nl_field.data_real)[1, 1, 2] = beta * u0_linear

    buffers_linear_scaled = GeoDynamo.ERK2FieldBuffers(u_field, nl_field, cache)
    GeoDynamo.erk2_prepare_field!(
        buffers_linear_scaled, u_field, nl_field, cache, cfg, dt;
        nonlinear_scale = forcing_scale)
    GeoDynamo.erk2_apply_stage!(buffers_linear_scaled, u_field)
    u_stage_scaled = parent(u_field.data_real)[1, 1, 2]
    parent(nl_field.data_real)[1, 1, 2] = beta * u_stage_scaled
    GeoDynamo.erk2_store_stage_nonlinear!(buffers_linear_scaled, nl_field)
    GeoDynamo.erk2_finalize_field!(
        buffers_linear_scaled, u_field, cache, cfg, dt;
        nonlinear_scale = forcing_scale)

    expected_linear_scaled = exp((forcing_scale * beta - lambda) * dt) * u0_linear
    @test parent(u_field.data_real)[1, 1, 2] ≈ expected_linear_scaled atol=1e-4
    @test parent(u_field.data_imag)[1, 1, 2] ≈ 0.0 atol=1e-4
end
