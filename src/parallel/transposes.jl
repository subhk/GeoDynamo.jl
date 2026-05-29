# ================================================================================
# Pencil Transpose Planning and Diagnostics
# ================================================================================

using PencilArrays
using PencilArrays: Pencil, PencilArray

@inline function __pencil_make_transpose(pair)
    trans_mod = PencilArrays.Transpositions
    src = first(pair)
    dest = last(pair)

    src_array = PencilArray{Float64}(undef, src)
    dest_array = PencilArray{Float64}(undef, dest)

    return trans_mod.Transposition(dest_array, src_array)
end

"""
    create_transpose_plans(pencils)

Create enhanced transpose plans between pencil orientations.
Includes caching and communication optimization.
"""
function create_transpose_plans(pencils)
    plans = Dict{Symbol, Any}()

    function try_create_transpose(name::Symbol, pair)
        try
            plans[name] = __pencil_make_transpose(pair)
        catch e
            if e isa ArgumentError
                @debug "Skipping transpose $name: $e"
            else
                rethrow(e)
            end
        end
    end

    try_create_transpose(:θ_to_φ, pencils.θ => pencils.φ)
    try_create_transpose(:φ_to_r, pencils.φ => pencils.r)
    try_create_transpose(:r_to_θ, pencils.r => pencils.θ)

    try_create_transpose(:φ_to_θ, pencils.φ => pencils.θ)
    try_create_transpose(:r_to_φ, pencils.r => pencils.φ)
    try_create_transpose(:θ_to_r, pencils.θ => pencils.r)

    try_create_transpose(:r_to_spec, pencils.r => pencils.spec)
    try_create_transpose(:spec_to_r, pencils.spec => pencils.r)

    try_create_transpose(:mixed_to_r, pencils.mixed => pencils.r)
    try_create_transpose(:r_to_mixed, pencils.r => pencils.mixed)

    return plans
end

const ENABLE_TIMING = Ref(false)
const TRANSPOSE_TIMES = Dict{Symbol, Float64}()
const TRANSPOSE_COUNTS = Dict{Symbol, Int}()
const __TIMING_LOCK = ReentrantLock()

"""
    transpose_with_timer!(dest, src, label=:default)

Perform transpose with optional timing and statistics.
"""
function transpose_with_timer!(dest::PencilArray, src::PencilArray, label::Symbol=:default)
    if ENABLE_TIMING[]
        t_start = MPI.Wtime()
        PencilArrays.transpose!(dest, src)
        t_end = MPI.Wtime()

        lock(__TIMING_LOCK) do
            TRANSPOSE_TIMES[label] = get(TRANSPOSE_TIMES, label, 0.0) + (t_end - t_start)
            TRANSPOSE_COUNTS[label] = get(TRANSPOSE_COUNTS, label, 0) + 1
        end
    else
        PencilArrays.transpose!(dest, src)
    end
end

"""
    print_transpose_statistics()

Print accumulated transpose timing statistics.
"""
function print_transpose_statistics()
    if get_rank() == 0 && !isempty(TRANSPOSE_TIMES)
        println("\n═══════════════════════════════════════════════════════")
        println(" Transpose Operation Statistics")
        println("═══════════════════════════════════════════════════════")

        for (label, total_time) in sort(collect(TRANSPOSE_TIMES), by=x->x[2], rev=true)
            count = TRANSPOSE_COUNTS[label]
            avg_time = total_time / count
            println(" $label:")
            println("   Total time:  $(round(total_time, digits=3)) s")
            println("   Calls:       $count")
            println("   Average:     $(round(avg_time*1000, digits=3)) ms")
        end
        println("═══════════════════════════════════════════════════════")
    end
end

"""
    optimize_communication_order(plans::Dict)

Determine optimal order for transpose operations to minimize communication.
"""
function optimize_communication_order(plans::Dict)
    comm_costs = Dict{Symbol, Float64}()

    for (name, plan) in plans
        src_pencil, dest_pencil = infer_pencils_from_transpose_name(name, plan)

        if src_pencil !== nothing && dest_pencil !== nothing
            data_volume = prod(size_global(src_pencil))
            comm_distance = estimate_communication_distance(src_pencil, dest_pencil)
            comm_costs[name] = data_volume * comm_distance
        else
            comm_costs[name] = 1.0
        end
    end

    return sort(collect(comm_costs), by=x->x[2])
end

"""
    infer_pencils_from_transpose_name(name::Symbol, plan) -> (src_pencil, dest_pencil)

Infer source and destination pencils from transpose operation name and plan.
"""
function infer_pencils_from_transpose_name(name::Symbol, plan)
    if hasfield(typeof(plan), :src_pencil) && hasfield(typeof(plan), :dest_pencil)
        return plan.src_pencil, plan.dest_pencil
    elseif hasfield(typeof(plan), :src) && hasfield(typeof(plan), :dest)
        return plan.src, plan.dest
    else
        @debug "Cannot infer pencils from TransposeOperator $name - communication optimization disabled"
        return nothing, nothing
    end
end

function estimate_communication_distance(src::Pencil, dest::Pencil)
    src_decomp = decomposition(src)
    dest_decomp = decomposition(dest)
    distance = sum(src_decomp .!= dest_decomp)
    return Float64(distance)
end
