# ================================================================================
# ERK2 Cache I/O and Cache Management Functions
# ================================================================================

"""
    get_erk2_cache!(caches, key, diffusivity, config, domain, dt; use_krylov=false)

Retrieve or create ERK2 cache with automatic invalidation when parameters change.
Type-stable version using concrete ERK2Cache{T} type.
"""
function get_erk2_cache!(caches::Dict{Symbol, ERK2Cache{T}}, key::Symbol, diffusivity::Float64,
                        ::Type{T}, config::SHTnsKitConfig, domain::RadialDomain, dt::Float64;
                        use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8,
                        bc_spec::Union{ERK2BoundarySpec{T}, Nothing}=nothing) where T
    nr = domain.N
    cache = get(caches, key, nothing)

    needs_rebuild = cache === nothing ||
                    cache.diffusivity != diffusivity ||
                    cache.nr != nr ||
                    cache.dt != dt ||
                    cache.use_krylov != use_krylov ||
                    !cache.mpi_consistent ||
                    cache.l_values != config.l_values

    if needs_rebuild
        if get_rank() == 0
            @info "Creating new ERK2 cache for $key (ν=$diffusivity, nr=$nr, dt=$dt)"
        end

        cache = create_erk2_cache(T, config, domain, diffusivity, dt;
                                  use_krylov=use_krylov, m=m, tol=tol, bc_spec=bc_spec)
    end

    caches[key] = cache
    return cache::ERK2Cache{T}  # Type assertion for compiler optimization
end

"""
    get_erk2_magnetic_toroidal_cache!(caches, diffusivity, T, config, domain, dt; use_krylov=false, m=20, tol=1e-8)

Retrieve or create ERK2 cache for magnetic toroidal field with embedded insulating BCs.
Uses Dirichlet BCs (BT = 0) at both boundaries, matching DD_2DCODE's mag_bc_Tor.
"""
function get_erk2_magnetic_toroidal_cache!(caches::Dict{Symbol, ERK2Cache{T}}, diffusivity::Float64,
                                           ::Type{T}, config::SHTnsKitConfig, domain::RadialDomain, dt::Float64;
                                           use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T
    key = :magnetic_toroidal_embedded
    nr = domain.N
    cache = get(caches, key, nothing)

    needs_rebuild = cache === nothing ||
                    cache.diffusivity != diffusivity ||
                    cache.nr != nr ||
                    cache.dt != dt ||
                    cache.use_krylov != use_krylov ||
                    !cache.mpi_consistent ||
                    cache.l_values != config.l_values

    if needs_rebuild
        if get_rank() == 0
            @info "Creating magnetic toroidal ERK2 cache with embedded insulating BCs (ν=$diffusivity, nr=$nr, dt=$dt)"
        end
        cache = create_erk2_cache_magnetic_toroidal(T, config, domain, diffusivity, dt;
                                                     use_krylov=use_krylov, m=m, tol=tol)
    end

    caches[key] = cache
    return cache::ERK2Cache{T}
end

"""
    get_erk2_magnetic_poloidal_cache!(caches, diffusivity, T, config, domain, dt; use_krylov=false, m=20, tol=1e-8)

Retrieve or create ERK2 cache for magnetic poloidal field with embedded insulating BCs.
Uses l-dependent insulating BCs matching DD_2DCODE's mag_bc_Pol:
- Inner boundary: (∂/∂r - l/r)P = 0
- Outer boundary: (∂/∂r + (l+1)/r)P = 0
"""
function get_erk2_magnetic_poloidal_cache!(caches::Dict{Symbol, ERK2Cache{T}}, diffusivity::Float64,
                                           ::Type{T}, config::SHTnsKitConfig, domain::RadialDomain, dt::Float64;
                                           use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T
    key = :magnetic_poloidal_embedded
    nr = domain.N
    cache = get(caches, key, nothing)

    needs_rebuild = cache === nothing ||
                    cache.diffusivity != diffusivity ||
                    cache.nr != nr ||
                    cache.dt != dt ||
                    cache.use_krylov != use_krylov ||
                    !cache.mpi_consistent ||
                    cache.l_values != config.l_values

    if needs_rebuild
        if get_rank() == 0
            @info "Creating magnetic poloidal ERK2 cache with embedded insulating BCs (ν=$diffusivity, nr=$nr, dt=$dt)"
        end
        cache = create_erk2_cache_magnetic_poloidal(T, config, domain, diffusivity, dt;
                                                     use_krylov=use_krylov, m=m, tol=tol)
    end

    caches[key] = cache
    return cache::ERK2Cache{T}
end

"""
    get_erk2_temperature_cache!(caches, diffusivity, T, config, domain, dt, i_tmp_bc; use_krylov=false, m=20, tol=1e-8)

Retrieve or create ERK2 cache for temperature field with embedded BCs.
Uses BC type specified by i_tmp_bc (1=DD, 2=DN, 3=ND, 4=NN), matching DD_2DCODE's tmp_bc_T.
"""
function get_erk2_temperature_cache!(caches::Dict{Symbol, ERK2Cache{T}}, diffusivity::Float64,
                                      ::Type{T}, config::SHTnsKitConfig, domain::RadialDomain,
                                      dt::Float64, i_tmp_bc::Int;
                                      use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T
    key = Symbol(:temperature_embedded_bc, i_tmp_bc)
    nr = domain.N
    cache = get(caches, key, nothing)

    needs_rebuild = cache === nothing ||
                    cache.diffusivity != diffusivity ||
                    cache.nr != nr ||
                    cache.dt != dt ||
                    cache.use_krylov != use_krylov ||
                    !cache.mpi_consistent ||
                    cache.l_values != config.l_values

    if needs_rebuild
        bc_desc = ["DD", "DN", "ND", "NN"][clamp(i_tmp_bc, 1, 4)]
        if get_rank() == 0
            @info "Creating temperature ERK2 cache with embedded BCs (type=$bc_desc, ν=$diffusivity, nr=$nr, dt=$dt)"
        end
        cache = create_erk2_cache_temperature(T, config, domain, diffusivity, dt, i_tmp_bc;
                                               use_krylov=use_krylov, m=m, tol=tol)
    end

    caches[key] = cache
    return cache::ERK2Cache{T}
end

"""
    get_erk2_composition_cache!(caches, diffusivity, T, config, domain, dt, i_cmp_bc; use_krylov=false, m=20, tol=1e-8)

Retrieve or create ERK2 cache for composition field with embedded BCs.
Uses BC type specified by i_cmp_bc (1=DD, 2=DN, 3=ND, 4=NN), matching DD_2DCODE's cmp_bc_C.
"""
function get_erk2_composition_cache!(caches::Dict{Symbol, ERK2Cache{T}}, diffusivity::Float64,
                                      ::Type{T}, config::SHTnsKitConfig, domain::RadialDomain,
                                      dt::Float64, i_cmp_bc::Int;
                                      use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T
    key = Symbol(:composition_embedded_bc, i_cmp_bc)
    nr = domain.N
    cache = get(caches, key, nothing)

    needs_rebuild = cache === nothing ||
                    cache.diffusivity != diffusivity ||
                    cache.nr != nr ||
                    cache.dt != dt ||
                    cache.use_krylov != use_krylov ||
                    !cache.mpi_consistent ||
                    cache.l_values != config.l_values

    if needs_rebuild
        bc_desc = ["DD", "DN", "ND", "NN"][clamp(i_cmp_bc, 1, 4)]
        if get_rank() == 0
            @info "Creating composition ERK2 cache with embedded BCs (type=$bc_desc, ν=$diffusivity, nr=$nr, dt=$dt)"
        end
        cache = create_erk2_cache_composition(T, config, domain, diffusivity, dt, i_cmp_bc;
                                               use_krylov=use_krylov, m=m, tol=tol)
    end

    caches[key] = cache
    return cache::ERK2Cache{T}
end

function _normalize_erk2_cache_entry(entry)
    if entry isa ERK2Cache
        return entry
    elseif entry isa Dict
        cache = get(entry, :cache, nothing)
        return cache isa ERK2Cache ? cache : nothing
    elseif entry === nothing
        return nothing
    else
        return nothing
    end
end

"""
    save_erk2_cache_bundle(path, caches; metadata=Dict())

Persist a dictionary of ERK2 caches to disk along with optional metadata.
"""
function save_erk2_cache_bundle(path::AbstractString,
                                caches::Dict{Symbol,<:ERK2Cache};
                                metadata::Dict{String,Any}=Dict{String,Any}())
    meta = Dict{String,Any}(metadata)
    meta["created_at"] = get(meta, "created_at", string(now()))
    jldopen(path, "w") do file
        file["caches"] = caches
        file["metadata"] = meta
    end
    return path
end

function save_erk2_cache_bundle(path::AbstractString,
                                caches::Dict{Symbol,Any};
                                metadata::Dict{String,Any}=Dict{String,Any}())
    bundle = Dict{Symbol,ERK2Cache}()
    for (key, value) in caches
        cache = _normalize_erk2_cache_entry(value)
        cache === nothing && continue
        bundle[key] = cache
    end
    return save_erk2_cache_bundle(path, bundle; metadata=metadata)
end

"""
    load_erk2_cache_bundle(path) -> (caches, metadata)

Load ERK2 caches and associated metadata from disk.
"""
function load_erk2_cache_bundle(path::AbstractString)
    caches = Dict{Symbol,Any}()
    metadata = Dict{String,Any}()
    jldopen(path, "r") do file
        caches = Dict{Symbol,Any}(file["caches"])
        metadata = haskey(file, "metadata") ? Dict{String,Any}(file["metadata"]) : Dict{String,Any}()
    end
    return caches, metadata
end

"""
    install_erk2_cache_bundle!(target, bundle)

Copy ERK2 caches from `bundle` into the target cache dictionary.
"""
function install_erk2_cache_bundle!(target::Dict{Symbol,Any},
                                    bundle::Dict{Symbol,<:Any})
    for (key, value) in bundle
        cache = _normalize_erk2_cache_entry(value)
        cache === nothing && continue
        target[key] = cache
    end
    return target
end

"""
    load_erk2_cache_bundle!(target, path) -> metadata

Load caches from `path` and install them into `target`, returning metadata.
"""
function load_erk2_cache_bundle!(target::Dict{Symbol,Any}, path::AbstractString)
    bundle, metadata = load_erk2_cache_bundle(path)
    install_erk2_cache_bundle!(target, bundle)
    return metadata
end
