"""
    ParityFixtures

Shared `SolverState` builder for the parity harness.

Extracted from the near-identical `_sbp_state` (scalar_bc_timestepper_parity.jl)
and `_vm_state` (velocity_magnetic_bc_timestepper_parity.jl).

The grid is the one both of those already use: small enough that a wide matrix is
affordable, and already demonstrated to exercise the ERK2 magnetic Robin, ERK2
scalar Neumann, and ERK2 l=0 NN bugs.
"""
module ParityFixtures

using GeoDynamo
using MPI
using Random

export ParityCase, build_state, evolve!

const VAL = GeoDynamo.ValueBoundaryCondition
const FLX = GeoDynamo.FluxBoundaryCondition

# scalar_code => BoundaryConditions.  1 = DD, 2 = DN, 3 = ND, 4 = NN.
const SCALAR_BCS = Dict(
    1 => GeoDynamo.BoundaryConditions(inner = VAL(1.0), outer = VAL(0.0)),
    2 => GeoDynamo.BoundaryConditions(inner = VAL(1.0), outer = FLX(0.0)),
    3 => GeoDynamo.BoundaryConditions(inner = FLX(1.0), outer = VAL(0.0)),
    4 => GeoDynamo.BoundaryConditions(inner = FLX(1.0), outer = FLX(0.0)),
)

# wall_code => BoundaryConditions(inner, outer).  1 = NS/NS, 2 = NS/SF, 3 = SF/NS, 4 = SF/SF.
#
# DEVIATION FROM BRIEF: the brief's WALL_BCS stored bare
# `(inner_marker, outer_marker)` Tuples. SolverParameters.velocity_bcs is typed
# `BoundaryConditions` (= FieldBoundaryConditions, src/core/parameters.jl:54),
# not a Tuple, so that construction throws
# `MethodError: Cannot convert ... Tuple{NoSlip,NoSlip} to ... FieldBoundaryConditions`
# on the very first build_state call. The existing probe this was extracted from
# (velocity_magnetic_bc_timestepper_parity.jl:141) wraps the same two markers in
# `GeoDynamo.BoundaryConditions(inner = ..., outer = ...)` — mirroring that here.
const WALL_BCS = Dict(
    1 => GeoDynamo.BoundaryConditions(inner = GeoDynamo.NoSlip(), outer = GeoDynamo.NoSlip()),
    2 => GeoDynamo.BoundaryConditions(inner = GeoDynamo.NoSlip(), outer = GeoDynamo.StressFree()),
    3 => GeoDynamo.BoundaryConditions(inner = GeoDynamo.StressFree(), outer = GeoDynamo.NoSlip()),
    4 => GeoDynamo.BoundaryConditions(inner = GeoDynamo.StressFree(), outer = GeoDynamo.StressFree()),
)

const TIMESTEPPERS = [
    ("CNAB2", GeoDynamo.CNAB2()),
    ("ERK2", GeoDynamo.ExponentialRungeKutta2()),
    ("RK3", GeoDynamo.RungeKutta3()),
]

struct ParityCase
    timestepper_name::String
    timestepper::Any
    scalar_code::Int
    wall_code::Int
    magnetic::Bool
    composition::Bool
end

function Base.show(io::IO, c::ParityCase)
    print(io, "$(c.timestepper_name)/scalar$(c.scalar_code)/wall$(c.wall_code)",
        c.magnetic ? "/mag" : "", c.composition ? "/comp" : "")
end

"""
    build_state(case; seed = 11)

Build a `SolverState`, initialize its fields, then perturb them deterministically.

The explicit `initialize_solver_fields!` call is load-bearing. `solver_step!`
does `state.is_initialized || initialize_solver_fields!(state)`
(src/solver/mainloop.jl:92), and `SolverState` is constructed with
`is_initialized = false` (mainloop.jl:56). Perturbing before that flag is set
means the first step silently erases the perturbation and every seed produces an
identical trajectory.
"""
function build_state(case::ParityCase; seed::Int = 11)
    kw = Dict{Symbol, Any}(
        :geometry => :shell, :lmax => 4, :mmax => 4, :nlat => 12, :nlon => 24,
        :nr => 16, :nr_inner => 4, :radial_bandwidth => 3, :radius_ratio => 0.35,
        :Ek => 1e-3, :Ra => 1e3, :Pm => 1.0, :Pr => 1.0, :timestep => 1e-5,
        :include_magnetic => case.magnetic,
        :include_composition => case.composition,
        :timestepper => case.timestepper,
        :temperature_bcs => SCALAR_BCS[case.scalar_code],
        :velocity_bcs => WALL_BCS[case.wall_code],
    )
    case.composition && (kw[:composition_bcs] = SCALAR_BCS[case.scalar_code])

    st = GeoDynamo.initialize_solver_state(
        Float64; params = GeoDynamo.SolverParameters(; kw...))

    GeoDynamo.initialize_solver_fields!(st)

    rng = MersenneTwister(seed)
    for f in _perturbable(st)
        dr = parent(f.data_real)
        di = parent(f.data_imag)
        dr .+= 1e-3 .* (rand(rng, size(dr)...) .- 0.5)
        di .+= 1e-3 .* (rand(rng, size(di)...) .- 0.5)
    end
    return st
end

function _perturbable(st)
    fs = Any[st.fields.temperature.spectral,
        st.fields.velocity.toroidal,
        st.fields.velocity.poloidal]
    st.fields.magnetic === nothing ||
        append!(fs, (st.fields.magnetic.toroidal, st.fields.magnetic.poloidal))
    st.fields.composition === nothing ||
        push!(fs, st.fields.composition.spectral)
    return fs
end

"""
    evolve!(state; nsteps = 4)

Step the real solver. Four steps, not one: CNAB2's `prev_nonlinear` history does
not participate until the second step, so a shorter trajectory is blind to
exactly the corruption the digest captures it for.
"""
function evolve!(state; nsteps::Int = 4)
    for _ in 1:nsteps
        GeoDynamo.solver_step!(state)
    end
    return state
end

const PARITY_MATRIX_FULL = [
    ParityCase(tsname, ts, sc, wc, mag, comp)
    for (tsname, ts) in TIMESTEPPERS
    for sc in 1:4
    for wc in 1:4
    for mag in (false, true)
    for comp in (false, true)
]

# Pairwise-covering subset: every level of every factor appears, and every pair of
# factors is exercised at least once. 12 cases against the full matrix's 192.
const PARITY_MATRIX_DEFAULT = [
    ParityCase("CNAB2", TIMESTEPPERS[1][2], 1, 1, false, false),
    ParityCase("CNAB2", TIMESTEPPERS[1][2], 2, 2, true, true),
    ParityCase("CNAB2", TIMESTEPPERS[1][2], 3, 3, true, false),
    ParityCase("CNAB2", TIMESTEPPERS[1][2], 4, 4, false, true),
    ParityCase("ERK2", TIMESTEPPERS[2][2], 1, 2, true, false),
    ParityCase("ERK2", TIMESTEPPERS[2][2], 2, 1, false, true),
    ParityCase("ERK2", TIMESTEPPERS[2][2], 3, 4, false, false),
    ParityCase("ERK2", TIMESTEPPERS[2][2], 4, 3, true, true),
    ParityCase("RK3", TIMESTEPPERS[3][2], 1, 3, false, true),
    ParityCase("RK3", TIMESTEPPERS[3][2], 2, 4, true, false),
    ParityCase("RK3", TIMESTEPPERS[3][2], 3, 1, true, true),
    ParityCase("RK3", TIMESTEPPERS[3][2], 4, 2, false, false),
]

"""
    select_matrix()

The default subset, or all 192 when `GEODYNAMO_PARITY_FULL=1`. The full matrix is
for a once-per-sub-project pre-PR run, not for routine use.
"""
select_matrix() = get(ENV, "GEODYNAMO_PARITY_FULL", "0") == "1" ?
                  PARITY_MATRIX_FULL : PARITY_MATRIX_DEFAULT

end # module
