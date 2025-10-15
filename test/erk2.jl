using Test
using Geodynamo

@testset "ERK2 staged update" begin
    dom = Geodynamo.create_radial_domain(1)
    cfg = Geodynamo.create_shtnskit_config(lmax=0, mmax=0, nlat=2, nlon=2, nr=dom.N, optimize_decomp=false)

    u_field = Geodynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
    nl_field = Geodynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)

    u0 = 0.3
    c = 0.2
    parent(u_field.data_real)[1, 1, 1] = u0
    parent(u_field.data_imag)[1, 1, 1] = 0.0
    parent(nl_field.data_real)[1, 1, 1] = c
    parent(nl_field.data_imag)[1, 1, 1] = 0.0

    dt = 0.1
    lambda = 0.5
    z = -lambda * dt

    E_half = exp(z / 2)
    E_full = exp(z)
    phi1_half = (E_half - 1) / (z / 2)
    phi1_full = (E_full - 1) / z
    phi2_full = (E_full - 1 - z) / (z^2)

    cache = Geodynamo.ERK2Cache{Float64}(
        dt,
        [cfg.l_values[1]],
        [fill(E_half, 1, 1)],
        [fill(E_full, 1, 1)],
        [fill(phi1_half, 1, 1)],
        [fill(phi1_full, 1, 1)],
        [fill(phi2_full, 1, 1)],
        false,
        20,
        1e-8,
        true,
    )

    buffers = Geodynamo.ERK2FieldBuffers(u_field, nl_field, cache)
    Geodynamo.erk2_prepare_field!(buffers, u_field, nl_field, cache, cfg, dt)
    Geodynamo.erk2_apply_stage!(buffers, u_field)
    Geodynamo.erk2_store_stage_nonlinear!(buffers, nl_field)
    Geodynamo.erk2_finalize_field!(buffers, u_field, cache, cfg, dt)

    u_real = parent(u_field.data_real)[1, 1, 1]
    expected = exp(-lambda * dt) * u0 + (1 - exp(-lambda * dt)) * c / lambda
    @test u_real ≈ expected atol=1e-4
    @test parent(u_field.data_imag)[1, 1, 1] ≈ 0.0 atol=1e-4

    # Scenario 2: linear nonlinearity N(u) = beta * u requiring stage recomputation
    u0_linear = 0.45
    beta = 0.15
    
    parent(u_field.data_real)[1, 1, 1] = u0_linear
    parent(u_field.data_imag)[1, 1, 1] = 0.0
    parent(nl_field.data_real)[1, 1, 1] = beta * u0_linear
    parent(nl_field.data_imag)[1, 1, 1] = 0.0

    buffers_linear = Geodynamo.ERK2FieldBuffers(u_field, nl_field, cache)

    Geodynamo.erk2_prepare_field!(buffers_linear, u_field, nl_field, cache, cfg, dt)
    Geodynamo.erk2_apply_stage!(buffers_linear, u_field)

    # Emulate stage nonlinear evaluation: N(u_stage) = beta * u_stage
    u_stage = parent(u_field.data_real)[1, 1, 1]
    parent(nl_field.data_real)[1, 1, 1] = beta * u_stage
    Geodynamo.erk2_store_stage_nonlinear!(buffers_linear, nl_field)

    Geodynamo.erk2_finalize_field!(buffers_linear, u_field, cache, cfg, dt)
    u_linear = parent(u_field.data_real)[1, 1, 1]
    expected_linear = exp((beta - lambda) * dt) * u0_linear

    @test u_linear ≈ expected_linear atol=1e-4
    @test parent(u_field.data_imag)[1, 1, 1] ≈ 0.0 atol=1e-4
end
