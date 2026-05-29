module GeoDynamoBall

"""
Ball-specific convenience API.
Provides a radial domain and field constructors appropriate for a solid
sphere (inner radius = 0). Transforms reuse the core SHTnsKit machinery.
"""

using ..GeoDynamo
using LinearAlgebra

"""
    BallConfig

Alias for [`SHTnsKitConfig`](@ref), re-exported under a ball-specific name for
clarity when working in full-sphere geometry (inner radius = 0). It carries the
spherical-harmonic transform plans, parallel layout, and buffers.
"""
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
    create_ball_radial_domain(nr) -> RadialDomain

Create a radial domain for a solid sphere (inner radius = 0).
Uses a cosine-stretched grid similar to the shell for compatibility,
but sets the inner radius to zero and adjusts the coordinate columns
to match expectations of downstream operators.
"""
function create_ball_radial_domain(nr::Int; radial_bandwidth::Int=4)
    N = nr
    if N < 2
        error("Ball radial domain requires nr >= 2, got nr=$N")
    end
    # r[:,4] holds the base radius coordinate in existing code
    r = zeros(Float64, N, 7)
    # Cosine clustering towards r=0 and r=1 like Chebyshev nodes mapped to [0,1]
    for n in 1:N
        # x in [-1,1]
        x = cos(pi * (N - n) / (N - 1))
        # map to [0,1]
        r[n, 4] = 0.5 * (1.0 + x)
    end
    # Outer radius is always normalized to 1.0
    R = 1.0

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

    dr_matrices         = [zeros(2*radial_bandwidth+1, N) for _ in 1:3]
    radial_laplacian    = zeros(2*radial_bandwidth+1, N)
    integration_weights = zeros(Float64, N)

    domain = GeoDynamo.RadialDomain(N, 1:N, r, dr_matrices, radial_laplacian, integration_weights)
    GeoDynamo.__populate_radial_operators!(domain)
    return domain
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
create_ball_pencils(cfg::BallConfig; nr::Int, optimize::Bool=true) = GeoDynamo.create_pencil_topology(cfg; nr, optimize)

"""
    create_ball_velocity_fields(T, cfg::BallConfig; nr)
"""
function create_ball_velocity_fields(::Type{T}, cfg::BallConfig; nr::Int) where {T}
    domain = create_ball_radial_domain(nr)
    pencils = create_ball_pencils(cfg; nr)
    return GeoDynamo.create_shtns_velocity_fields(T, cfg, domain, pencils, pencils.spec)
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
    return GeoDynamo.create_shtns_magnetic_fields(T, cfg, domain, domain, pencils, pencils.spec)
end

const BC_Ball = GeoDynamo.bcs

function create_ball_hybrid_temperature_boundaries(inner_spec::Tuple, outer_spec::Tuple, cfg::BallConfig; precision::Type{T}=Float64) where {T}
    return BC_Ball.create_programmatic_temperature_boundaries(inner_spec, outer_spec, cfg)
end
function create_ball_hybrid_temperature_boundaries(inner_spec::String, outer_spec::Tuple, cfg::BallConfig; precision::Type{T}=Float64) where {T}
    return BC_Ball.create_hybrid_temperature_boundaries(inner_spec, outer_spec, cfg; swap_boundaries=false)
end
function create_ball_hybrid_temperature_boundaries(inner_spec::Tuple, outer_spec::String, cfg::BallConfig; precision::Type{T}=Float64) where {T}
    return BC_Ball.create_hybrid_temperature_boundaries(outer_spec, inner_spec, cfg; swap_boundaries=true)
end
function create_ball_hybrid_temperature_boundaries(inner_spec::String, outer_spec::String, cfg::BallConfig; precision::Type{T}=Float64) where {T}
    return BC_Ball.load_temperature_boundaries_from_files(inner_spec, outer_spec, cfg)
end

function create_ball_hybrid_composition_boundaries(inner_spec::Tuple, outer_spec::Tuple, cfg::BallConfig; precision::Type{T}=Float64) where {T}
    return BC_Ball.create_programmatic_composition_boundaries(inner_spec, outer_spec, cfg)
end
function create_ball_hybrid_composition_boundaries(inner_spec::String, outer_spec::Tuple, cfg::BallConfig; precision::Type{T}=Float64) where {T}
    return BC_Ball.create_hybrid_composition_boundaries(inner_spec, outer_spec, cfg; swap_boundaries=false)
end
function create_ball_hybrid_composition_boundaries(inner_spec::Tuple, outer_spec::String, cfg::BallConfig; precision::Type{T}=Float64) where {T}
    return BC_Ball.create_hybrid_composition_boundaries(outer_spec, inner_spec, cfg; swap_boundaries=true)
end
function create_ball_hybrid_composition_boundaries(inner_spec::String, outer_spec::String, cfg::BallConfig; precision::Type{T}=Float64) where {T}
    return BC_Ball.load_composition_boundaries_from_files(inner_spec, outer_spec, cfg)
end

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

    lm_range = GeoDynamo.local_spectral_mode_indices(cfg)
    r_range = GeoDynamo.range_local(cfg.pencils.spec, 3)

    # Only proceed if this rank owns the inner boundary (r=0, global index 1)
    if !(1 in r_range)
        return spec  # This rank doesn't own r=0
    end

    # Convert global radial index 1 to local index
    r_local_idx = 1 - first(r_range) + 1

    T = eltype(spec_real)

    @inbounds for lm_idx in lm_range
        if lm_idx <= cfg.nlm
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_idx)
            slot === nothing && continue
            l = cfg.l_values[lm_idx]
            if l > 0
                GeoDynamo.set_local_spectral_value!(spec_real, slot, r_local_idx, zero(T))
                GeoDynamo.set_local_spectral_value!(spec_imag, slot, r_local_idx, zero(T))
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

    lm_range = GeoDynamo.local_spectral_mode_indices(cfg)
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

        @inbounds for lm_idx in lm_range
            if lm_idx <= cfg.nlm
                slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_idx)
                slot === nothing && continue
                l = cfg.l_values[lm_idx]
                if l >= 1
                    GeoDynamo.set_local_spectral_value!(sreal, slot, r_local_idx, zero(T))
                    GeoDynamo.set_local_spectral_value!(simag, slot, r_local_idx, zero(T))
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
