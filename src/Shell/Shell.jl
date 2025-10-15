module GeodynamoShell

"""
Shell-specific convenience API.
This module provides thin wrappers around the core SHTnsKit-based
implementations to work with a spherical shell geometry (inner radius > 0).
"""

using ..Geodynamo

# Re-export the core config type for clarity in shell context
const ShellConfig = Geodynamo.SHTnsKitConfig

export ShellConfig
export create_shell_pencils
export create_shell_radial_domain
export create_shell_spectral_field, create_shell_physical_field, create_shell_vector_field
export create_shell_velocity_fields, create_shell_temperature_field
export create_shell_composition_field, create_shell_magnetic_fields
export create_shell_hybrid_temperature_boundaries, create_shell_hybrid_composition_boundaries
export apply_shell_temperature_boundaries!, apply_shell_composition_boundaries!

const BC = Geodynamo.BoundaryConditions

"""
    create_shell_radial_domain(nr=i_N) -> RadialDomain

Create a radial domain suitable for a spherical shell using the
global parameter `d_rratio` for inner radius ratio.
This is a thin wrapper over `Geodynamo.create_radial_domain`.
"""
function create_shell_radial_domain(nr::Int = Geodynamo.i_N)
    return Geodynamo.create_radial_domain(nr)
end

"""
    create_shell_spectral_field(T, cfg::ShellConfig, domain::Geodynamo.RadialDomain, pencil)
"""
create_shell_spectral_field(::Type{T}, cfg::ShellConfig, domain::Geodynamo.RadialDomain, pencil) where {T} =
    Geodynamo.create_shtns_spectral_field(T, cfg, domain, pencil)

"""
    create_shell_physical_field(T, cfg::ShellConfig, domain::Geodynamo.RadialDomain, pencil)
"""
create_shell_physical_field(::Type{T}, cfg::ShellConfig, domain::Geodynamo.RadialDomain, pencil) where {T} =
    Geodynamo.create_shtns_physical_field(T, cfg, domain, pencil)

"""
    create_shell_vector_field(T, cfg::ShellConfig, domain::Geodynamo.RadialDomain, pencils)
"""
create_shell_vector_field(::Type{T}, cfg::ShellConfig, domain::Geodynamo.RadialDomain, pencils) where {T} =
    Geodynamo.create_shtns_vector_field(T, cfg, domain, pencils)

"""
    create_shell_pencils(cfg::ShellConfig; optimize=true)

Create a shell-oriented pencil decomposition using the core topology helper.
"""
create_shell_pencils(cfg::ShellConfig; optimize::Bool=true) = Geodynamo.create_pencil_topology(cfg; optimize)

"""
    create_shell_velocity_fields(T, cfg::ShellConfig; nr=Geodynamo.i_N)
"""
function create_shell_velocity_fields(::Type{T}, cfg::ShellConfig; nr::Int=Geodynamo.i_N) where {T}
    domain = create_shell_radial_domain(nr)
    pencils = create_shell_pencils(cfg)
    return Geodynamo.create_shtns_velocity_fields(T, cfg, domain, pencils, pencils.spec)
end

"""
    create_shell_temperature_field(T, cfg::ShellConfig; nr=Geodynamo.i_N)
"""
function create_shell_temperature_field(::Type{T}, cfg::ShellConfig; nr::Int=Geodynamo.i_N) where {T}
    domain = create_shell_radial_domain(nr)
    return Geodynamo.create_shtns_temperature_field(T, cfg, domain)
end

"""
    create_shell_composition_field(T, cfg::ShellConfig; nr=Geodynamo.i_N)
"""
function create_shell_composition_field(::Type{T}, cfg::ShellConfig; nr::Int=Geodynamo.i_N) where {T}
    domain = create_shell_radial_domain(nr)
    return Geodynamo.create_shtns_composition_field(T, cfg, domain)
end

"""
    create_shell_magnetic_fields(T, cfg::ShellConfig; nr_oc=Geodynamo.i_N, nr_ic=Geodynamo.i_Nic)
"""
function create_shell_magnetic_fields(::Type{T}, cfg::ShellConfig; nr_oc::Int=Geodynamo.i_N, nr_ic::Int=Geodynamo.i_Nic) where {T}
    domain_oc = create_shell_radial_domain(nr_oc)
    domain_ic = create_shell_radial_domain(nr_ic)
    pencils = create_shell_pencils(cfg)
    return Geodynamo.create_shtns_magnetic_fields(T, cfg, domain_oc, domain_ic, pencils, pencils.spec)
end

"""
    create_shell_hybrid_temperature_boundaries(inner_spec, outer_spec, cfg::ShellConfig; precision=Float64)

Create temperature boundaries for shell geometry. Dispatches to appropriate function based on spec types:
- Both Tuple: fully programmatic boundaries
- String + Tuple: hybrid (one from file, one programmatic)
- Both String: both from files
"""
function create_shell_hybrid_temperature_boundaries(inner_spec::Tuple, outer_spec::Tuple, cfg::ShellConfig; precision::Type{T}=Float64) where {T}
    return BC.create_programmatic_temperature_boundaries(inner_spec, outer_spec, cfg)
end

function create_shell_hybrid_temperature_boundaries(inner_spec::String, outer_spec::Tuple, cfg::ShellConfig; precision::Type{T}=Float64) where {T}
    return BC.create_hybrid_temperature_boundaries(inner_spec, outer_spec, cfg; swap_boundaries=false)
end

function create_shell_hybrid_temperature_boundaries(inner_spec::Tuple, outer_spec::String, cfg::ShellConfig; precision::Type{T}=Float64) where {T}
    return BC.create_hybrid_temperature_boundaries(outer_spec, inner_spec, cfg; swap_boundaries=true)
end

function create_shell_hybrid_temperature_boundaries(inner_spec::String, outer_spec::String, cfg::ShellConfig; precision::Type{T}=Float64) where {T}
    return BC.load_temperature_boundaries_from_files(inner_spec, outer_spec, cfg)
end

"""
    create_shell_hybrid_composition_boundaries(inner_spec, outer_spec, cfg::ShellConfig; precision=Float64)

Create composition boundaries for shell geometry. Dispatches to appropriate function based on spec types:
- Both Tuple: fully programmatic boundaries
- String + Tuple: hybrid (one from file, one programmatic)
- Both String: both from files
"""
function create_shell_hybrid_composition_boundaries(inner_spec::Tuple, outer_spec::Tuple, cfg::ShellConfig; precision::Type{T}=Float64) where {T}
    return BC.create_programmatic_composition_boundaries(inner_spec, outer_spec, cfg)
end

function create_shell_hybrid_composition_boundaries(inner_spec::String, outer_spec::Tuple, cfg::ShellConfig; precision::Type{T}=Float64) where {T}
    return BC.create_hybrid_composition_boundaries(inner_spec, outer_spec, cfg; swap_boundaries=false)
end

function create_shell_hybrid_composition_boundaries(inner_spec::Tuple, outer_spec::String, cfg::ShellConfig; precision::Type{T}=Float64) where {T}
    return BC.create_hybrid_composition_boundaries(outer_spec, inner_spec, cfg; swap_boundaries=true)
end

function create_shell_hybrid_composition_boundaries(inner_spec::String, outer_spec::String, cfg::ShellConfig; precision::Type{T}=Float64) where {T}
    return BC.load_composition_boundaries_from_files(inner_spec, outer_spec, cfg)
end

"""
    apply_shell_temperature_boundaries!(temp_field, boundary_set; time=0)

Wrapper around core boundary application for shell geometry.
"""
apply_shell_temperature_boundaries!(temp_field, boundary_set; time=0.0) =
    BC.apply_temperature_boundaries!(temp_field, boundary_set; time=time)

"""
    apply_shell_composition_boundaries!(comp_field, boundary_set; time=0)
"""
apply_shell_composition_boundaries!(comp_field, boundary_set; time=0.0) =
    BC.apply_composition_boundaries!(comp_field, boundary_set; time=time)

end # module
