# ================================================================================
# Pencil Transpose Planning and Diagnostics
# ================================================================================

using PencilArrays
using PencilArrays: Pencil, PencilArray

@inline function _pencil_make_transpose(pair)
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
            plans[name] = _pencil_make_transpose(pair)
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
const _TIMING_LOCK = ReentrantLock()

"""
    transpose_with_timer!(dest, src, label=:default)

Perform transpose with optional timing and statistics.
"""
function transpose_with_timer!(dest::PencilArray, src::PencilArray, label::Symbol = :default)
    if ENABLE_TIMING[]
        t_start = MPI.Wtime()
        PencilArrays.transpose!(dest, src)
        t_end = MPI.Wtime()

        lock(_TIMING_LOCK) do
            TRANSPOSE_TIMES[label] = get(TRANSPOSE_TIMES, label, 0.0) + (t_end - t_start)
            TRANSPOSE_COUNTS[label] = get(TRANSPOSE_COUNTS, label, 0) + 1
        end
    else
        PencilArrays.transpose!(dest, src)
    end
end

# Cached intermediate PencilArrays for the two-step spec↔spec_transform transpose
# (keyed by the (src_pencil, dst_pencil) pair).  Needed because the Phase-3 spec
# pencil is decomp (2,1) while spec_transform is (1,3): they differ in BOTH
# decomposed axes, and PencilArrays.transpose! permits at most one differing axis
# per call.  We route through an intermediate that shares one axis with each.
const _SPEC_TRANSPOSE_INTERMEDIATE_CACHE = IdDict{Any, Any}()
const _SPEC_TRANSPOSE_LOCK = ReentrantLock()

# Build an intermediate decomposition that differs from both `a` (src) and `b`
# (dst) in at most one decomposed axis and has no repeated dimension.  We try the
# two natural candidates — keep a's axis-1 and take b's axis-2, or keep a's axis-2
# and take b's axis-1 — and return the first that is a valid (non-repeating)
# decomposition.  For (2,1)↔(1,3) both directions yield (2,3).
@inline function _intermediate_decomp(a::NTuple{2,Int}, b::NTuple{2,Int})
    c1 = (a[1], b[2])   # keep src axis-1, take dst axis-2
    c2 = (b[1], a[2])   # take dst axis-1, keep src axis-2
    if c1[1] != c1[2]
        return c1
    elseif c2[1] != c2[2]
        return c2
    else
        error("no valid intermediate decomposition between $a and $b")
    end
end

function _spec_transpose!(dst::PencilArray, src::PencilArray, label::Symbol)
    src_p = pencil(src)
    dst_p = pencil(dst)
    sd = decomposition(src_p)
    dd = decomposition(dst_p)
    if sum(sd .!= dd) <= 1
        transpose_with_timer!(dst, src, label)
        return dst
    end
    # Two-step via a cached intermediate (single differing axis per leg).
    inter = lock(_SPEC_TRANSPOSE_LOCK) do
        get!(_SPEC_TRANSPOSE_INTERMEDIATE_CACHE, (src_p, dst_p)) do
            mid_decomp = _intermediate_decomp(Tuple(sd), Tuple(dd))
            mid_pencil = Pencil(src_p, decomp_dims = mid_decomp)
            PencilArray{eltype(src)}(undef, mid_pencil)
        end
    end::PencilArray
    transpose_with_timer!(inter, src, label)
    transpose_with_timer!(dst, inter, label)
    return dst
end

"""
    transpose_solve_to_transform!(dst::PencilArray, src::PencilArray)

Transpose spectral data from the solve orientation (`spec`, Phase-3 decomp (2,1):
m over θ_ranks, l over r_ranks, r local) to the transform orientation
(`spec_transform`, decomp (1,3): l over θ_ranks, r over r_ranks, m LOCAL).

Because (2,1) and (1,3) differ in both decomposed axes, this routes through a
cached intermediate `(2,3)` pencil (two single-axis `PencilArrays.transpose!`
calls), preserving exact, identity-invertible data movement.  After this call,
each rank holds a full-m × l-subset slab for each of its local r levels, enabling
the Phase-2 per-level distributed SH vector calls.
"""
transpose_solve_to_transform!(dst::PencilArray, src::PencilArray) =
    _spec_transpose!(dst, src, :spec_solve_to_transform)

"""
    transpose_transform_to_solve!(dst::PencilArray, src::PencilArray)

Inverse of `transpose_solve_to_transform!`.  Transposes spectral data from the
transform orientation (`spec_transform`, decomp (1,3)) back to the solve
orientation (`spec`, Phase-3 decomp (2,1)); also routed through the cached
intermediate.  Exact identity roundtrip.
"""
transpose_transform_to_solve!(dst::PencilArray, src::PencilArray) =
    _spec_transpose!(dst, src, :spec_transform_to_solve)

"""
    print_transpose_statistics()

Print accumulated transpose timing statistics.
"""
function print_transpose_statistics()
    if get_rank() == 0 && !isempty(TRANSPOSE_TIMES)
        println("\n═══════════════════════════════════════════════════════")
        println(" Transpose Operation Statistics")
        println("═══════════════════════════════════════════════════════")

        for (label, total_time) in sort(collect(TRANSPOSE_TIMES), by = x->x[2], rev = true)
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

    return sort(collect(comm_costs), by = x->x[2])
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
