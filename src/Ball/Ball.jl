module GeoDynamoBall

"""
Ball-specific convenience API.
Provides a radial domain and field constructors appropriate for a solid
sphere (inner radius = 0). Transforms reuse the core SHTnsKit machinery.
"""

using ..GeoDynamo
using LinearAlgebra

# Re-export the core config type for clarity in ball context
const BallConfig = GeoDynamo.SHTnsKitConfig

export BallConfig
export create_ball_pencils
export create_ball_radial_domain
export create_ball_spectral_field, create_ball_physical_field, create_ball_vector_field
export create_ball_velocity_fields, create_ball_temperature_field
export create_ball_composition_field, create_ball_magnetic_fields
export create_ball_hybrid_temperature_boundaries, create_ball_hybrid_composition_boundaries
export enforce_ball_scalar_regularity!, enforce_ball_vector_regularity!
export apply_ball_temperature_regularity!, apply_ball_composition_regularity!
export ball_physical_to_spectral!, ball_vector_analysis!

"""
    create_ball_radial_domain(nr=i_N) -> RadialDomain

Create a radial domain for a solid sphere (inner radius = 0).
Uses a cosine-stretched grid similar to the shell for compatibility,
but sets the inner radius to zero and adjusts the coordinate columns
to match expectations of downstream operators.
"""
function create_ball_radial_domain(nr::Int = GeoDynamo.i_N)
    N = nr
    # r[:,4] holds the base radius coordinate in existing code
    r = zeros(Float64, N, 7)
    # Cosine clustering towards r=0 and r=1 like Chebyshev nodes mapped to [0,1]
    for n in 1:N
        # x in [-1,1]
        x = cos(pi * (N - n) / (N - 1))
        # map to [0,1]
        r[n, 4] = 0.5 * (1.0 + x)
    end
    # Optionally scale to a physical outer radius R>0
    R = try
        GeoDynamo.get_parameters().d_R_outer
    catch
        1.0
    end
    if !(R > 0)
        error("d_R_outer must be > 0 for ball geometry (got $(R))")
    end
    if R != 1.0
        r[:, 4] .*= R
    end

    # Fill powers of r in other columns for compatibility (after any scaling)
    # Guard against Inf at r=0 (ball center) for negative powers
    for p in 1:7
        if p != 4
            power = p - 4
            for i in 1:N
                r_val = r[i, 4]
                if r_val == 0.0 && power < 0
                    r[i, p] = 0.0  # Regularize: treat 1/r^n as 0 at the origin
                else
                    r[i, p] = r_val ^ power
                end
            end
        end
    end

    dr_matrices         = [zeros(2*GeoDynamo.i_KL+1, N) for _ in 1:3]
    radial_laplacian    = zeros(2*GeoDynamo.i_KL+1, N)
    integration_weights = zeros(Float64, N)

    return GeoDynamo.RadialDomain(N, 1:N, r, dr_matrices, radial_laplacian, integration_weights)
end

"""
    create_ball_spectral_field(T, cfg::BallConfig, domain::GeoDynamo.RadialDomain, pencil)
"""
create_ball_spectral_field(::Type{T}, cfg::BallConfig, domain::GeoDynamo.RadialDomain, pencil) where {T} =
    GeoDynamo.create_shtns_spectral_field(T, cfg, domain, pencil)

"""
    create_ball_physical_field(T, cfg::BallConfig, domain::GeoDynamo.RadialDomain, pencil)
"""
create_ball_physical_field(::Type{T}, cfg::BallConfig, domain::GeoDynamo.RadialDomain, pencil) where {T} =
    GeoDynamo.create_shtns_physical_field(T, cfg, domain, pencil)

"""
    create_ball_vector_field(T, cfg::BallConfig, domain::GeoDynamo.RadialDomain, pencils)
"""
create_ball_vector_field(::Type{T}, cfg::BallConfig, domain::GeoDynamo.RadialDomain, pencils) where {T} =
    GeoDynamo.create_shtns_vector_field(T, cfg, domain, pencils)

"""
    create_ball_pencils(cfg::BallConfig; optimize=true)
"""
create_ball_pencils(cfg::BallConfig; optimize::Bool=true) = GeoDynamo.create_pencil_topology(cfg; optimize)

"""
    create_ball_velocity_fields(T, cfg::BallConfig; nr=GeoDynamo.i_N)
"""
function create_ball_velocity_fields(::Type{T}, cfg::BallConfig; nr::Int=GeoDynamo.i_N) where {T}
    domain = create_ball_radial_domain(nr)
    pencils = create_ball_pencils(cfg)
    return GeoDynamo.create_shtns_velocity_fields(T, cfg, domain, pencils, pencils.spec)
end

"""
    create_ball_temperature_field(T, cfg::BallConfig; nr=GeoDynamo.i_N)
"""
function create_ball_temperature_field(::Type{T}, cfg::BallConfig; nr::Int=GeoDynamo.i_N) where {T}
    domain = create_ball_radial_domain(nr)
    return GeoDynamo.create_shtns_temperature_field(T, cfg, domain)
end

"""
    create_ball_composition_field(T, cfg::BallConfig; nr=GeoDynamo.i_N)
"""
function create_ball_composition_field(::Type{T}, cfg::BallConfig; nr::Int=GeoDynamo.i_N) where {T}
    domain = create_ball_radial_domain(nr)
    return GeoDynamo.create_shtns_composition_field(T, cfg, domain)
end

"""
    create_ball_magnetic_fields(T, cfg::BallConfig; nr=GeoDynamo.i_N)

Create magnetic fields for a solid sphere. Since a "core" split is not
used in a ball, we pass the same domain for both oc and ic to reuse the
core implementation.
"""
function create_ball_magnetic_fields(::Type{T}, cfg::BallConfig; nr::Int=GeoDynamo.i_N) where {T}
    domain = create_ball_radial_domain(nr)
    pencils = create_ball_pencils(cfg)
    return GeoDynamo.create_shtns_magnetic_fields(T, cfg, domain, domain, pencils, pencils.spec)
end

"""
    create_ball_hybrid_temperature_boundaries(inner_spec, outer_spec, cfg::BallConfig; precision=Float64)
"""
create_ball_hybrid_temperature_boundaries(inner_spec, outer_spec, cfg::BallConfig; precision::Type{T}=Float64) where {T} =
    GeoDynamo.create_hybrid_temperature_boundaries(inner_spec, outer_spec, cfg; precision)

"""
    create_ball_hybrid_composition_boundaries(inner_spec, outer_spec, cfg::BallConfig; precision=Float64)
"""
create_ball_hybrid_composition_boundaries(inner_spec, outer_spec, cfg::BallConfig; precision::Type{T}=Float64) where {T} =
    GeoDynamo.create_hybrid_composition_boundaries(inner_spec, outer_spec, cfg; precision)

"""
    enforce_ball_scalar_regularity!(spec::GeoDynamo.SHTnsSpecField)

Enforce scalar regularity at r=0 for solid sphere: for l>0, the scalar
amplitude must vanish at r=0. Sets inner radial plane to zero for all
nonzero l modes (both real and imaginary parts).
"""
function enforce_ball_scalar_regularity!(spec::GeoDynamo.SHTnsSpecField)
    cfg = spec.config
    spec_real = parent(spec.data_real)
    spec_imag = parent(spec.data_imag)

    lm_range = GeoDynamo.range_local(cfg.pencils.spec, 1)
    r_range = GeoDynamo.range_local(cfg.pencils.spec, 3)

    # Only proceed if this rank owns the inner boundary (r=0, global index 1)
    if !(1 in r_range)
        return spec  # This rank doesn't own r=0
    end

    # Convert global radial index 1 to local index
    r_local_idx = 1 - first(r_range) + 1

    T = eltype(spec_real)

    @inbounds for (k, lm_idx) in enumerate(lm_range)
        if lm_idx <= cfg.nlm
            l = cfg.l_values[lm_idx]
            if l > 0
                spec_real[k, 1, r_local_idx] = zero(T)
                spec_imag[k, 1, r_local_idx] = zero(T)
            end
        end
    end
    return spec
end

"""
    enforce_ball_vector_regularity!(tor_spec::GeoDynamo.SHTnsSpecField,
                                    pol_spec::GeoDynamo.SHTnsSpecField)

Enforce vector-field regularity at r=0 for solid sphere. For smooth
fields, both toroidal and poloidal potentials behave like r^{l+1}, so
they vanish at r=0 for all l ≥ 1. Zeros the inner radial plane for l≥1.
"""
function enforce_ball_vector_regularity!(tor_spec::GeoDynamo.SHTnsSpecField,
                                         pol_spec::GeoDynamo.SHTnsSpecField)
    cfg = tor_spec.config

    lm_range = GeoDynamo.range_local(cfg.pencils.spec, 1)
    r_range = GeoDynamo.range_local(cfg.pencils.spec, 3)

    # Only proceed if this rank owns the inner boundary (r=0, global index 1)
    if !(1 in r_range)
        return tor_spec, pol_spec  # This rank doesn't own r=0
    end

    # Convert global radial index 1 to local index
    r_local_idx = 1 - first(r_range) + 1

    for sp in (tor_spec, pol_spec)
        sreal = parent(sp.data_real)
        simag = parent(sp.data_imag)
        T = eltype(sreal)

        @inbounds for (k, lm_idx) in enumerate(lm_range)
            if lm_idx <= cfg.nlm
                l = cfg.l_values[lm_idx]
                if l >= 1
                    sreal[k, 1, r_local_idx] = zero(T)
                    simag[k, 1, r_local_idx] = zero(T)
                end
            end
        end
    end
    return tor_spec, pol_spec
end

"""
    apply_ball_temperature_regularity!(temp_field)

Convenience to enforce scalar regularity on the temperature spectral field.
Call after assembling or updating temp_field.spectral.
"""
function apply_ball_temperature_regularity!(temp_field)
    return enforce_ball_scalar_regularity!(temp_field.spectral)
end

"""
    apply_ball_composition_regularity!(comp_field)
"""
function apply_ball_composition_regularity!(comp_field)
    return enforce_ball_scalar_regularity!(comp_field.spectral)
end

"""
    ball_physical_to_spectral!(phys::GeoDynamo.SHTnsPhysField,
                               spec::GeoDynamo.SHTnsSpecField)

Wrapper for transforms in a solid sphere that enforces scalar regularity at r=0
after analysis. Use this for scalar fields (temperature, composition, etc.).
"""
function ball_physical_to_spectral!(phys::GeoDynamo.SHTnsPhysField{T},
                                    spec::GeoDynamo.SHTnsSpecField{T}) where {T}
    GeoDynamo.shtnskit_physical_to_spectral!(phys, spec)
    enforce_ball_scalar_regularity!(spec)
    return spec
end

"""
    ball_vector_analysis!(vec::GeoDynamo.SHTnsVectorField,
                          tor::GeoDynamo.SHTnsSpecField,
                          pol::GeoDynamo.SHTnsSpecField)

Wrapper for vector analysis in a solid sphere; enforces vector regularity at r=0
after transforming to spectral toroidal/poloidal.
"""
function ball_vector_analysis!(vec::GeoDynamo.SHTnsVectorField{T},
                               tor::GeoDynamo.SHTnsSpecField{T},
                               pol::GeoDynamo.SHTnsSpecField{T}) where {T}
    GeoDynamo.shtnskit_vector_analysis!(vec, tor, pol)
    enforce_ball_vector_regularity!(tor, pol)
    return tor, pol
end

end # module
