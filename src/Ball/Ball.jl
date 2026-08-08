module GeoDynamoBall

"""
Ball-specific convenience API.
Provides a radial domain and field constructors appropriate for a full
sphere (no inner boundary; off-center radial grid). Transforms reuse the
core SHTnsKit machinery.
"""

using ..GeoDynamo
using LinearAlgebra

"""
    BallConfig

Alias for [`SHTnsKitConfig`](@ref), re-exported under a ball-specific name for
clarity when working in full-sphere geometry (no inner boundary; off-center
radial grid). It carries the spherical-harmonic transform plans, parallel
layout, and buffers.
"""
const BallConfig = GeoDynamo.SHTnsKitConfig

export BallConfig
export create_ball_pencils
export create_ball_radial_domain
export create_ball_spectral_field, create_ball_physical_field, create_ball_vector_field
export create_ball_velocity_fields, create_ball_temperature_field
export create_ball_composition_field, create_ball_magnetic_fields
export create_ball_hybrid_temperature_boundaries, create_ball_hybrid_composition_boundaries

"""
    create_ball_radial_domain(nr) -> RadialDomain

Create a radial domain for a solid sphere using an off-center cosine grid:

    r_n = (1 − cos(πn/N)) / 2,  n = 1, …, N

The outermost node r_N = 1 exactly (cos(π) = −1). The innermost node
r_1 = (1 − cos(π/N))/2 > 0 — there is **no node at the centre r = 0**.
All negative-power columns (1/r, 1/r²) are therefore finite and honest;
no regularisation guard is needed.  Regularity at r = 0 is imposed
separately through l-dependent Robin boundary rows in the implicit
matrices.
"""
function create_ball_radial_domain(nr::Int; radial_bandwidth::Int = 4)
    N = nr
    if N < 2
        error("Ball radial domain requires nr >= 2, got nr=$N")
    end

    # Off-center cosine grid: r_n = (1 − cos(πn/N))/2, n = 1..N.
    # r_N = 1 exactly; r_1 = (1 − cos(π/N))/2 > 0 — no node at the center.
    # Regularity at r=0 is imposed through l-dependent Robin boundary rows in
    # the implicit matrices, not through grid values, so every 1/r, 1/r²
    # operator entry stays finite and honest.
    r = zeros(Float64, N, 7)
    for n in 1:N
        r[n, 4] = 0.5 * (1.0 - cos(pi * n / N))
    end
    for p in 1:7
        if p != 4
            power = p - 4
            for i in 1:N
                r[i, p] = r[i, 4]^power
            end
        end
    end

    dr_matrices = [zeros(2*radial_bandwidth+1, N) for _ in 1:3]
    radial_laplacian = zeros(2*radial_bandwidth+1, N)

    # Clenshaw-Curtis weights for the Chebyshev-Lobatto grid θ_n = πn/N,
    # r = (1 − cos θ)/2, mapped to r ∈ [0, 1] (dr = dx/2, half-interval 0.5).
    # `_populate_radial_operators!` fills only the derivative operators, so these
    # have to be built here: leaving them zero makes every r²-weighted volume
    # integral (kinetic/magnetic/scalar energy, Reynolds stress) identically zero
    # in :ball geometry.
    # The n = 0 node (r = 0) is deliberately NOT part of this grid, so its
    # Lobatto weight is dropped. Every consumer integrates r²·f(r), which
    # vanishes at r = 0, so the dropped term contributes nothing; only the outer
    # node r_N = 1 keeps the usual endpoint halving.
    integration_weights = zeros(Float64, N)
    for n in 1:N
        theta_n = pi * n / N
        w = 0.0
        for k in 0:div(N, 2)
            bk = (k == 0 || (k == div(N, 2) && iseven(N))) ? 1.0 : 2.0
            w += bk * cos(2 * k * theta_n) / (1 - 4 * k^2)
        end
        w *= 2.0 / N
        if n == N
            w *= 0.5  # Lobatto endpoint correction at r = 1
        end
        integration_weights[n] = w * 0.5  # scale by half-interval h = 0.5
    end

    domain = GeoDynamo.RadialDomain(
        N, 1:N, r, dr_matrices, radial_laplacian, integration_weights)
    GeoDynamo._populate_radial_operators!(domain)
    return domain
end

"""
    create_ball_spectral_field(T, cfg::BallConfig, domain::GeoDynamo.RadialDomain, pencil)
"""
function create_ball_spectral_field(
        ::Type{T}, cfg::BallConfig, domain::GeoDynamo.RadialDomain, pencil) where {T}
    GeoDynamo.create_shtns_spectral_field(T, cfg, domain, pencil)
end

"""
    create_ball_physical_field(T, cfg::BallConfig, domain::GeoDynamo.RadialDomain, pencil)
"""
function create_ball_physical_field(
        ::Type{T}, cfg::BallConfig, domain::GeoDynamo.RadialDomain, pencil) where {T}
    GeoDynamo.create_shtns_physical_field(T, cfg, domain, pencil)
end

"""
    create_ball_vector_field(T, cfg::BallConfig, domain::GeoDynamo.RadialDomain, pencils)
"""
function create_ball_vector_field(::Type{T}, cfg::BallConfig, domain::GeoDynamo.RadialDomain, pencils) where {T}
    GeoDynamo.create_shtns_vector_field(T, cfg, domain, pencils)
end

"""
    create_ball_pencils(cfg::BallConfig; optimize=true)
"""
function create_ball_pencils(cfg::BallConfig; nr::Int, optimize::Bool = true)
    GeoDynamo.create_pencil_topology(cfg; nr, optimize)
end

"""
    create_ball_velocity_fields(T, cfg::BallConfig; nr)
"""
function create_ball_velocity_fields(::Type{T}, cfg::BallConfig; nr::Int) where {T}
    domain = create_ball_radial_domain(nr)
    pencils = create_ball_pencils(cfg; nr)
    # geometry = :ball cannot be inferred from the off-center grid (r_1 > 0),
    # so pass it explicitly; the defaulted params then get geometry/:ball,
    # radius_ratio 0.0, and nr_inner 0.
    return GeoDynamo.create_shtns_velocity_fields(
        T, cfg, domain, pencils, pencils.spec; geometry = :ball)
end

"""
    create_ball_temperature_field(T, cfg::BallConfig; nr)
"""
function create_ball_temperature_field(::Type{T}, cfg::BallConfig; nr::Int) where {T}
    domain = create_ball_radial_domain(nr)
    return GeoDynamo.create_shtns_temperature_field(T, cfg, domain)
end

"""
    create_ball_composition_field(T, cfg::BallConfig; nr)
"""
function create_ball_composition_field(::Type{T}, cfg::BallConfig; nr::Int) where {T}
    domain = create_ball_radial_domain(nr)
    return GeoDynamo.create_shtns_composition_field(T, cfg, domain)
end

"""
    create_ball_magnetic_fields(T, cfg::BallConfig; nr)

Create magnetic fields for a solid sphere. Since a "core" split is not
used in a ball, we pass the same domain for both oc and ic to reuse the
core implementation.
"""
function create_ball_magnetic_fields(::Type{T}, cfg::BallConfig; nr::Int) where {T}
    domain = create_ball_radial_domain(nr)
    pencils = create_ball_pencils(cfg; nr)
    return GeoDynamo.create_shtns_magnetic_fields(
        T, cfg, domain, domain, pencils, pencils.spec)
end

const BC_Ball = GeoDynamo.bcs

function create_ball_hybrid_temperature_boundaries(
        inner_spec::Tuple, outer_spec::Tuple, cfg::BallConfig;
        precision::Type{T} = Float64) where {T}
    return BC_Ball.create_programmatic_temperature_boundaries(inner_spec, outer_spec, cfg)
end
function create_ball_hybrid_temperature_boundaries(
        inner_spec::String, outer_spec::Tuple, cfg::BallConfig;
        precision::Type{T} = Float64) where {T}
    return BC_Ball.create_hybrid_temperature_boundaries(inner_spec, outer_spec, cfg; swap_boundaries = false)
end
function create_ball_hybrid_temperature_boundaries(
        inner_spec::Tuple, outer_spec::String, cfg::BallConfig;
        precision::Type{T} = Float64) where {T}
    return BC_Ball.create_hybrid_temperature_boundaries(outer_spec, inner_spec, cfg; swap_boundaries = true)
end
function create_ball_hybrid_temperature_boundaries(
        inner_spec::String, outer_spec::String, cfg::BallConfig;
        precision::Type{T} = Float64) where {T}
    return BC_Ball.load_temperature_boundaries_from_files(inner_spec, outer_spec, cfg)
end

function create_ball_hybrid_composition_boundaries(
        inner_spec::Tuple, outer_spec::Tuple, cfg::BallConfig;
        precision::Type{T} = Float64) where {T}
    return BC_Ball.create_programmatic_composition_boundaries(inner_spec, outer_spec, cfg)
end
function create_ball_hybrid_composition_boundaries(
        inner_spec::String, outer_spec::Tuple, cfg::BallConfig;
        precision::Type{T} = Float64) where {T}
    return BC_Ball.create_hybrid_composition_boundaries(inner_spec, outer_spec, cfg; swap_boundaries = false)
end
function create_ball_hybrid_composition_boundaries(
        inner_spec::Tuple, outer_spec::String, cfg::BallConfig;
        precision::Type{T} = Float64) where {T}
    return BC_Ball.create_hybrid_composition_boundaries(outer_spec, inner_spec, cfg; swap_boundaries = true)
end
function create_ball_hybrid_composition_boundaries(
        inner_spec::String, outer_spec::String, cfg::BallConfig;
        precision::Type{T} = Float64) where {T}
    return BC_Ball.load_composition_boundaries_from_files(inner_spec, outer_spec, cfg)
end

end # module
