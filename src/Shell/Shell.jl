module GeoDynamoShell

"""
Shell-specific convenience API.
This module provides thin wrappers around the core SHTnsKit-based
implementations to work with a spherical shell geometry (inner radius > 0).
"""

using ..GeoDynamo

"""
    ShellConfig

Alias for [`SHTnsKitConfig`](@ref), re-exported under a shell-specific name for
clarity when working in spherical-shell geometry (inner radius > 0). It carries
the spherical-harmonic transform plans, parallel layout, and buffers.
"""
const ShellConfig = GeoDynamo.SHTnsKitConfig

export ShellConfig
export create_shell_pencils
export create_shell_radial_domain
export create_shell_spectral_field, create_shell_physical_field, create_shell_vector_field
export create_shell_velocity_fields, create_shell_temperature_field
export create_shell_composition_field, create_shell_magnetic_fields
export create_shell_hybrid_temperature_boundaries,
       create_shell_hybrid_composition_boundaries
export apply_shell_temperature_boundaries!, apply_shell_composition_boundaries!

const BC = GeoDynamo.bcs

"""
    create_shell_radial_domain(nr; radius_ratio=0.35, radial_bandwidth=4) -> RadialDomain

Create a radial domain suitable for a spherical shell.
This is a thin wrapper over `GeoDynamo.create_radial_domain`.
"""
function create_shell_radial_domain(nr::Int; radius_ratio::Real = 0.35, radial_bandwidth::Int = 4)
    return GeoDynamo.create_radial_domain(nr; radius_ratio = radius_ratio, radial_bandwidth = radial_bandwidth)
end

"""
    create_shell_spectral_field(T, cfg::ShellConfig, domain::GeoDynamo.RadialDomain, pencil)
"""
function create_shell_spectral_field(
        ::Type{T}, cfg::ShellConfig, domain::GeoDynamo.RadialDomain, pencil) where {T}
    GeoDynamo.create_shtns_spectral_field(T, cfg, domain, pencil)
end

"""
    create_shell_physical_field(T, cfg::ShellConfig, domain::GeoDynamo.RadialDomain, pencil)
"""
function create_shell_physical_field(
        ::Type{T}, cfg::ShellConfig, domain::GeoDynamo.RadialDomain, pencil) where {T}
    GeoDynamo.create_shtns_physical_field(T, cfg, domain, pencil)
end

"""
    create_shell_vector_field(T, cfg::ShellConfig, domain::GeoDynamo.RadialDomain, pencils)
"""
function create_shell_vector_field(
        ::Type{T}, cfg::ShellConfig, domain::GeoDynamo.RadialDomain, pencils) where {T}
    GeoDynamo.create_shtns_vector_field(T, cfg, domain, pencils)
end

"""
    create_shell_pencils(cfg::ShellConfig; optimize=true)

Create a shell-oriented pencil decomposition using the core topology helper.
"""
function create_shell_pencils(cfg::ShellConfig; nr::Int, optimize::Bool = true)
    GeoDynamo.create_pencil_topology(cfg; nr, optimize)
end

"""
    create_shell_velocity_fields(T, cfg::ShellConfig; nr)
"""
function create_shell_velocity_fields(::Type{T}, cfg::ShellConfig; nr::Int) where {T}
    domain = create_shell_radial_domain(nr)
    pencils = create_shell_pencils(cfg; nr)
    return GeoDynamo.create_shtns_velocity_fields(T, cfg, domain, pencils, pencils.spec)
end

"""
    create_shell_temperature_field(T, cfg::ShellConfig; nr)
"""
function create_shell_temperature_field(::Type{T}, cfg::ShellConfig; nr::Int) where {T}
    domain = create_shell_radial_domain(nr)
    return GeoDynamo.create_shtns_temperature_field(T, cfg, domain)
end

"""
    create_shell_composition_field(T, cfg::ShellConfig; nr)
"""
function create_shell_composition_field(::Type{T}, cfg::ShellConfig; nr::Int) where {T}
    domain = create_shell_radial_domain(nr)
    return GeoDynamo.create_shtns_composition_field(T, cfg, domain)
end

"""
    create_shell_magnetic_fields(T, cfg::ShellConfig; nr_oc, nr_ic)
"""
function create_shell_magnetic_fields(::Type{T}, cfg::ShellConfig; nr_oc::Int, nr_ic::Int) where {T}
    outer_core_domain = create_shell_radial_domain(nr_oc)
    inner_core_domain = create_shell_radial_domain(nr_ic)
    pencils = create_shell_pencils(cfg; nr = nr_oc)
    return GeoDynamo.create_shtns_magnetic_fields(T, cfg, outer_core_domain, inner_core_domain, pencils, pencils.spec)
end

"""
    create_shell_hybrid_temperature_boundaries(inner_spec, outer_spec, cfg::ShellConfig; precision=Float64)

Create temperature boundaries for shell geometry. Dispatches to appropriate function based on spec types:
- Both Tuple: fully programmatic boundaries
- String + Tuple: hybrid (one from file, one programmatic)
- Both String: both from files
"""
function create_shell_hybrid_temperature_boundaries(
        inner_spec::Tuple, outer_spec::Tuple, cfg::ShellConfig;
        precision::Type{T} = Float64) where {T}
    return BC.create_programmatic_temperature_boundaries(inner_spec, outer_spec, cfg)
end

function create_shell_hybrid_temperature_boundaries(
        inner_spec::String, outer_spec::Tuple, cfg::ShellConfig;
        precision::Type{T} = Float64) where {T}
    return BC.create_hybrid_temperature_boundaries(inner_spec, outer_spec, cfg; swap_boundaries = false)
end

function create_shell_hybrid_temperature_boundaries(
        inner_spec::Tuple, outer_spec::String, cfg::ShellConfig;
        precision::Type{T} = Float64) where {T}
    return BC.create_hybrid_temperature_boundaries(outer_spec, inner_spec, cfg; swap_boundaries = true)
end

function create_shell_hybrid_temperature_boundaries(
        inner_spec::String, outer_spec::String, cfg::ShellConfig;
        precision::Type{T} = Float64) where {T}
    return BC.load_temperature_boundaries_from_files(inner_spec, outer_spec, cfg)
end

"""
    create_shell_hybrid_composition_boundaries(inner_spec, outer_spec, cfg::ShellConfig; precision=Float64)

Create composition boundaries for shell geometry. Dispatches to appropriate function based on spec types:
- Both Tuple: fully programmatic boundaries
- String + Tuple: hybrid (one from file, one programmatic)
- Both String: both from files
"""
function create_shell_hybrid_composition_boundaries(
        inner_spec::Tuple, outer_spec::Tuple, cfg::ShellConfig;
        precision::Type{T} = Float64) where {T}
    return BC.create_programmatic_composition_boundaries(inner_spec, outer_spec, cfg)
end

function create_shell_hybrid_composition_boundaries(
        inner_spec::String, outer_spec::Tuple, cfg::ShellConfig;
        precision::Type{T} = Float64) where {T}
    return BC.create_hybrid_composition_boundaries(inner_spec, outer_spec, cfg; swap_boundaries = false)
end

function create_shell_hybrid_composition_boundaries(
        inner_spec::Tuple, outer_spec::String, cfg::ShellConfig;
        precision::Type{T} = Float64) where {T}
    return BC.create_hybrid_composition_boundaries(outer_spec, inner_spec, cfg; swap_boundaries = true)
end

function create_shell_hybrid_composition_boundaries(
        inner_spec::String, outer_spec::String, cfg::ShellConfig;
        precision::Type{T} = Float64) where {T}
    return BC.load_composition_boundaries_from_files(inner_spec, outer_spec, cfg)
end

"""
    apply_shell_temperature_boundaries!(temp_field, boundary_set; time=0)

Wrapper around core boundary application for shell geometry.
"""
function apply_shell_temperature_boundaries!(temp_field, boundary_set; time = 0.0)
    BC.apply_temperature_boundaries!(temp_field, boundary_set; time = time)
end

"""
    apply_shell_composition_boundaries!(comp_field, boundary_set; time=0)
"""
function apply_shell_composition_boundaries!(comp_field, boundary_set; time = 0.0)
    BC.apply_composition_boundaries!(comp_field, boundary_set; time = time)
end

end # module
