# ================================================================================
# Pencil Topology Utilities
# ================================================================================

using PencilArrays
using PencilArrays: Pencil, PencilArray
using Statistics: std

# ================================
# Optimized Process Topology
# ================================
"""
    optimize_process_topology(nprocs::Int, dims::Tuple{Int,Int,Int}, spectral_dims=nothing)
    
Find optimal 2D process grid for given number of processes and problem dimensions.
Minimizes communication volume.
"""
function optimize_process_topology(nprocs::Int,
        dims::Tuple{Int, Int, Int},
        spectral_dims::Union{Nothing, Tuple{Int, Int, Int}} = nothing)
    nlat, nlon, nr = dims

    # Find all valid 2D decompositions
    decompositions = Tuple{Int, Int}[]
    for p1 in 1:nprocs
        if nprocs % p1 == 0
            p2 = nprocs ÷ p1
            push!(decompositions, (p1, p2))
        end
    end

    # Score each decomposition based on communication patterns
    best_score = Inf
    best_decomp = (nprocs, 1)

    for (p1, p2) in decompositions
        # Estimate communication volume for different pencil orientations
        # Prefer decompositions that balance load and minimize surface/volume ratio

        # Check if decomposition is valid for physical and spectral storage.
        if nlat ÷ p1 < 2 || nlon ÷ p2 < 2
            continue
        end
        if spectral_dims !== nothing &&
           (spectral_dims[1] ÷ p1 < 1 || spectral_dims[2] ÷ p2 < 1)
            continue
        end

        # Score based on:
        # 1. Load balance (prefer square-ish decompositions)
        # 2. Communication volume (proportional to surface area)
        # 3. Cache efficiency (prefer contiguous dimensions)

        aspect_ratio = max(p1/p2, p2/p1)
        comm_volume = (nlat/p1 + nlon/p2) * nr  # Simplified communication estimate
        cache_penalty = abs(p1 - p2)  # Penalty for non-square decomposition

        score = comm_volume * aspect_ratio * (1.0 + 0.1 * cache_penalty)

        if score < best_score
            best_score = score
            best_decomp = (p1, p2)
        end
    end

    # Validate that the best decomposition satisfies the minimum grid-per-process constraint
    p1, p2 = best_decomp
    if nlat ÷ p1 < 2 || nlon ÷ p2 < 2 ||
       (spectral_dims !== nothing &&
        (spectral_dims[1] ÷ p1 < 1 || spectral_dims[2] ÷ p2 < 1))
        @warn "No valid 2D pencil decomposition found for nlat=$nlat, nlon=$nlon with $nprocs processes. " *
              "Best candidate ($p1, $p2) gives $(nlat÷p1) × $(nlon÷p2) points per process " *
              "(minimum 2×2 physical and 1×1 spectral required). Consider reducing the number of MPI processes or increasing resolution."
    end

    return best_decomp
end

@inline spectral_mode_grid_dims(lmax::Int, mmax::Int, nr::Int) = (lmax + 1, mmax + 1, nr)
@inline spectral_mode_grid_dims(config, nr::Int) = spectral_mode_grid_dims(config.lmax, config.mmax, nr)

"""
    create_inner_core_spectral_pencil(config, reference_spec_pencil, nr_inner) -> Pencil

Spectral pencil for inner-core fields. Reuses the (l, m) process topology and
decomposition of the outer-core spectral pencil but sizes the (local) radial
dimension to the inner-core grid (`nr_inner`) instead of the outer-core `nr`,
so `toroidal_ic/poloidal_ic` own exactly their physical radial extent rather than padding to `nr`.
The mode-slot (l, m) layout is identical to the outer-core spectral pencil.
"""
function create_inner_core_spectral_pencil(config, reference_spec_pencil, nr_inner::Int)
    spec_dims = spectral_mode_grid_dims(config, nr_inner)
    return Pencil(topology(reference_spec_pencil), spec_dims, (1, 2))
end

"""
    create_pencil_topology(shtns_config; nr, optimize=true)

Create enhanced pencil decomposition for SHTns grids.
Phase 2 (r×θ 2D topology): reads the process grid from GEODYNAMO_PROC_GRID
(`θ_ranks×r_ranks`), giving θ-distributed / φ-local / r-distributed physical
pencils and a `theta_phys` 2D prototype built on the θ-subcommunicator.
The `optimize` keyword is accepted for API compatibility.
Accepts an object with fields `nlat`, `nlon`, `nlm`, `lmax`, and `mmax`
(e.g., `SHTnsKitConfig`).
"""
function create_pencil_topology(shtns_config; nr::Int, optimize::Bool = true)
    comm = get_comm()
    rank = get_rank()
    nprocs = get_nprocs()

    # Get SHTns grid dimensions
    nlat = shtns_config.nlat
    nlon = shtns_config.nlon
    dims = (nlat, nlon, nr)

    # Phase 2 (r×θ 2D topology): read process grid from environment.
    # At nprocs==1 returns (1,1) without requiring the env var.
    θ_ranks, r_ranks = read_proc_grid(nprocs)
    proc_dims = (θ_ranks, r_ranks)

    # Create PencilArrays topology
    # Construct MPI-aware topology (modern PencilArrays exports MPITopology)
    TopoCtor = getproperty(PencilArrays, Symbol("MPITopology"))
    topology = TopoCtor(comm, proc_dims)

    if rank == 0
        println("═══════════════════════════════════════════════════════")
        println(" Pencil Decomposition Setup")
        println("═══════════════════════════════════════════════════════")
        println(" MPI Configuration:")
        println("   Processes:        $nprocs")
        println("   Process grid:     $(proc_dims[1]) × $(proc_dims[2]) (θ×r)")
        println(" Grid dimensions:")
        println("   Physical:         $nlat × $nlon × $nr")
        println("   Spectral modes:   $(shtns_config.nlm)")
        println("═══════════════════════════════════════════════════════")
    end

    # Create pencils for different computational stages (subcomms derived inside,
    # from pencil_r's actual distribution).
    pencils = create_computation_pencils(topology, dims, shtns_config, comm, θ_ranks)

    return pencils
end

"""
    create_computation_pencils(topology, dims, config, θ_comm, r_comm, θ_ranks)

Create specialized pencils for different stages of computation.

Phase 2 (r×θ): `pencil_r` uses decomp `(1,3)` so θ(dim1) is distributed over
θ_ranks and r(dim3) is distributed over r_ranks, leaving φ(dim2) LOCAL.
`theta_phys` is a 2D `(nlat,nlon)` pencil on the θ-subcommunicator so its
θ-split matches `pencil_r`'s θ-split within each r-group.
`θ_comm` and `r_comm` are forwarded in the returned NamedTuple for sub-collective
operations.
"""
function create_computation_pencils(topology, dims::Tuple{Int, Int, Int}, config,
        comm, θ_ranks::Int)
    nlat, nlon, nr = dims

    # Each pencil keeps one axis local and distributes the other two. The
    # selected local axis matches the operation that should run without MPI
    # communication in that orientation.
    pencil_θ = Pencil(topology, dims, (2, 3))  # Contiguous in θ (latitude)
    pencil_φ = Pencil(topology, dims, (1, 3))  # Contiguous in φ (longitude)

    # Phase 2 r×θ: decomp (1,3) → θ(dim1) over axis1(θ_ranks), r(dim3) over
    # axis2(r_ranks), φ(dim2) LOCAL.  At (θ_ranks,r_ranks)==(1,1) trivially local.
    pencil_r = Pencil(topology, dims, (1, 3))

    # Derive subcomms from pencil_r's actual distribution (robust for any grid/ordering).
    θ_comm, r_comm = make_subcomms(comm, pencil_r)

    # Spectral space is represented as a rectangular (l, m, r) grid rather than
    # the compact SHTns `nlm` list.  Phase-3 layout: m (dim2) over θ_ranks, l (dim1)
    # over r_ranks, r (dim3) LOCAL — i.e. decomp (2,1) — matching the spec_solve
    # partition (l-dist/r_comm, m-dist/θ_comm).  The generic (l,m)->packed-index map
    # reads range_local(spec,1/2), so the radial-solve indexing adapts automatically.
    spec_dims = spectral_mode_grid_dims(config, nr)
    pencil_spec = Pencil(topology, spec_dims, (2, 1))

    # Compatibility alias for older call sites. Mixed spectral storage must obey
    # the same rectangular (l, m, r) ownership contract as the spectral pencil.

    # 2D (nlat,nlon) θ-distributed prototype for SHTnsKit's per-level transforms.
    # Built on θ_comm (ranks sharing an r-slab that split θ) so its θ-split matches
    # pencil_r's θ-split per rank.
    TopoCtor = getproperty(PencilArrays, Symbol("MPITopology"))
    topo1d_θ = TopoCtor(θ_comm, (θ_ranks,))
    pencil_theta_phys = Pencil(topo1d_θ, (nlat, nlon), (1,))

    return (θ = pencil_θ,
        φ = pencil_φ,
        r = pencil_r,
        spec = pencil_spec,
        mixed = pencil_spec,
        theta_phys = pencil_theta_phys,
        θ_comm = θ_comm,
        r_comm = r_comm)
end

# ===============================
# Load Balancing Analysis
# ===============================
"""
    analyze_load_balance(pencil::Pencil)
    
Analyze and report load balance for a given pencil decomposition.
"""
function analyze_load_balance(pencil::Pencil)::Float64
    comm = get_comm()
    rank = get_rank()
    nprocs = get_nprocs()

    local_size::Tuple{Int, Int, Int} = size_local(pencil)
    local_elements::Int = prod(local_size)

    min_size = MPI.Allreduce(local_elements, MPI.MIN, comm)
    max_size = MPI.Allreduce(local_elements, MPI.MAX, comm)
    total_size = MPI.Allreduce(local_elements, MPI.SUM, comm)
    avg_size = total_size / nprocs
    imbalance = (max_size - min_size) / avg_size * 100

    # Collective: every rank must call MPI.Gather; only root receives the result.
    all_sizes = MPI.Gather(local_elements, comm; root = 0)

    if rank == 0
        std_size = std(all_sizes)

        println("\nLoad Balance Analysis:")
        println("  Min elements: $min_size")
        println("  Max elements: $max_size")
        println("  Average:      $avg_size")
        println("  Std dev:      $(round(std_size, digits=2))")
        println("  Imbalance:    $(round(imbalance, digits=1))%")

        if imbalance > 10
            println("  Warning: Load imbalance exceeds 10%")
        end
    end

    return imbalance
end

# ===================================
# Memory-Aware Pencil Creation
# ===================================
"""
    estimate_memory_usage(pencils, field_count::Int, precision::Type)
    
Estimate memory usage for given pencil configuration.
"""
function estimate_memory_usage(pencils, field_count::Int, precision::Type)
    bytes_per_element = sizeof(precision)
    total_bytes = 0

    # Calculate memory for each pencil orientation
    for (name, pencil) in pairs(pencils)
        pencil isa Pencil || continue   # skip θ_comm, r_comm entries
        local_size = size_local(pencil)
        local_bytes = prod(local_size) * bytes_per_element * field_count
        total_bytes += local_bytes
    end

    # Add overhead for transpose buffers (typically 2x largest pencil)
    max_pencil_size = maximum([prod(size_local(p)) for p in values(pencils) if p isa Pencil])
    buffer_bytes = 2 * max_pencil_size * bytes_per_element
    total_bytes += buffer_bytes

    # Convert to human-readable format
    if total_bytes < 1024^2
        memory_str = "$(round(total_bytes/1024, digits=1)) KB"
    elseif total_bytes < 1024^3
        memory_str = "$(round(total_bytes/1024^2, digits=1)) MB"
    else
        memory_str = "$(round(total_bytes/1024^3, digits=2)) GB"
    end

    return total_bytes, memory_str
end

# =============================
# Pencil Array Utilities
# =============================
"""
    create_pencil_array(::Type{T}, pencil::Pencil; init=:zero) where T
    
Create a PencilArray with specified initialization.
"""
function create_pencil_array(::Type{T}, pencil::Pencil; init = :zero) where {T}
    arr = PencilArray{T}(undef, pencil)

    if init == :zero
        fill!(parent(arr), zero(T))
    elseif init == :random
        parent(arr) .= randn(T, size(parent(arr)))
    elseif init == :ones
        fill!(parent(arr), one(T))
    end

    return arr
end

# ===========================
# Diagnostic Functions
# ===========================
"""
    print_pencil_info(pencils)
    
Print detailed information about pencil decomposition.
"""
function print_pencil_info(pencils)
    rank = get_rank()

    if rank == 0
        println("\n═══════════════════════════════════════════════════════")
        println(" Pencil Decomposition Information")
        println("═══════════════════════════════════════════════════════")
    end

    for (name, pencil) in pairs(pencils)
        pencil isa Pencil || continue   # skip θ_comm, r_comm entries
        global_size = size_global(pencil)
        local_size = size_local(pencil)
        local_range = range_local(pencil)

        # Gather info from all ranks
        all_local_sizes = MPI.Gather(prod(local_size), get_comm(); root = 0)

        if rank == 0
            println("\n Pencil: $name")
            println("   Global size:  $(global_size)")
            println("   Decomposed:   $(decomposition(pencil))")

            if get_nprocs() > 1
                min_local = minimum(all_local_sizes)
                max_local = maximum(all_local_sizes)
                balance = max_local / min_local
                println("   Load balance: $(round(balance, digits=2))x")
            end
        end
    end

    if rank == 0
        println("═══════════════════════════════════════════════════════")
    end
end

"""
    print_pencil_axes(pencils)

Print the `axes_local` tuple for each pencil, showing the local index ranges
for all three axes. This helps verify which axes are distributed (those with
nontrivial subranges across ranks) and which axis is contiguous locally.
"""
function print_pencil_axes(pencils)
    rank = get_rank()
    if rank == 0
        println("\nPencil axes_local (local index ranges per axis):")
    end
    for (name, pencil) in pairs(pencils)
        pencil isa Pencil || continue   # skip θ_comm, r_comm entries
        # Use range_local accessor for version compatibility
        axes_in = range_local(pencil)
        if rank == 0
            println(rpad("  " * String(name), 14), " => ", axes_in)
        end
    end
end

"""
    validate_radial_distribution(pencils; warn_uneven::Bool=true, strict::Bool=true) -> Bool

Validate that radial dimension has compatible distribution across all pencils.

# MPI Synchronization Requirement
The SHTnsKit transforms use MPI.Allreduce inside per-radial-level loops.
All processes must have the SAME number of local radial levels, otherwise
processes will enter/exit the loop at different times causing **MPI DEADLOCK**.

# Arguments
- `pencils`: Named tuple of pencil configurations
- `warn_uneven`: If true, emit warning for uneven distribution
- `strict`: If true (default), throw an error instead of just warning

# Returns
`true` if distribution is valid (all processes have same local radial count).
`false` if there's a potential synchronization issue.

# Default Behavior
Strict mode is enabled by default to prevent MPI deadlock in production runs.
Set `strict=false` only for debugging purposes.

# Example
```julia
# Recommended: use default strict mode
validate_radial_distribution(pencils)

# For debugging only: disable strict mode
validate_radial_distribution(pencils; strict=false)
```
"""
function validate_radial_distribution(pencils; warn_uneven::Bool = true, strict::Bool = true)
    comm = get_comm()
    rank = get_rank()
    nprocs = get_nprocs()

    if nprocs == 1
        return true  # No distribution issues with single process
    end

    # Check radial distribution for each pencil type
    valid = true
    problematic_pencils = Symbol[]
    distribution_info = Dict{Symbol, Tuple{Int, Int}}()
    checked_pencils = Pencil[]

    for (name, pencil) in pairs(pencils)
        # Only the r-local compute pencils (:spec, :mixed) require synchronized
        # radial counts across ranks — their per-radial collective sync depends on
        # every rank holding the full radial profile for its (l,m) mode subset.
        # Phase 2: :r is now r-DISTRIBUTED (decomp (1,3)), so uneven r-splits across
        # r-groups are expected and harmless (PencilArrays handles Alltoallv).
        # The :θ/:φ transpose pencils also distribute r by design (FFT orientations).
        # Skip all of these to avoid false "uneven radial distribution" alarms.
        name in (:spec, :mixed) || continue
        # :mixed aliases the :spec pencil (mixed = pencil_spec), so checking both
        # would run the identical MPI.Allgather twice. De-duplicate by object
        # identity — the alias relationship is the same on every rank, so the
        # collective count stays consistent across ranks.
        any(p -> p === pencil, checked_pencils) && continue
        push!(checked_pencils, pencil)
        # Use range_local accessor for version compatibility
        local_axes = range_local(pencil)
        if length(local_axes) >= 3
            # Get number of local radial levels
            local_r_count = length(local_axes[3])

            # Gather counts from all processes
            all_r_counts = MPI.Allgather(local_r_count, comm)

            # Check if all counts are equal
            min_count = minimum(all_r_counts)
            max_count = maximum(all_r_counts)

            if min_count != max_count
                valid = false
                push!(problematic_pencils, name)
                distribution_info[name] = (min_count, max_count)
            end
        end
    end

    if !valid
        msg = """
        CRITICAL: Uneven radial distribution detected!

        Affected pencils: $(join(problematic_pencils, ", "))
        Distribution: $(join(["$k: min=$(v[1]), max=$(v[2])" for (k,v) in distribution_info], "; "))
        MPI processes: $nprocs

        This WILL cause MPI deadlock in SHTnsKit transforms because
        MPI.Allreduce is called inside per-radial-level loops.

        SOLUTION: Ensure nr (radial grid points) is evenly divisible by nprocs.
        For example: if nprocs=4, use nr=64, 128, 256, etc.
        """

        if strict
            error(msg)
        elseif warn_uneven && rank == 0
            @warn msg
        end
    end

    return valid
end

"""
    check_transform_synchronization(config; strict::Bool=false) -> Bool

Verify that the SHTnsKit transform configuration is safe for parallel execution.
"""
function check_transform_synchronization(config; strict::Bool = false)
    nprocs = get_nprocs()

    if nprocs == 1
        return true
    end

    pencils = if config isa AbstractDict
        get(config, :pencils, nothing)
    elseif hasproperty(config, :pencils)
        getproperty(config, :pencils)
    else
        nothing
    end

    if pencils !== nothing &&
       !validate_radial_distribution(pencils; warn_uneven = true, strict = strict)
        return false
    end

    return true
end
