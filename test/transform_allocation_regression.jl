# Regression: the per-step SH transform path must stay type-stable (no per-operand
# boxing of the cached scratch/plan).
#
# The Phase-3 scratch + plan caches live in `Union{Any, Nothing}` fields on
# SHTnsBuffers (disttranspose_plan, p3_scalar_scratch, p3_vector_scratch == `Any`).
# The transform entry points fetch `plan`/`sc` from those fields, so without a
# function barrier the synthesis/analysis bodies box every `sc.<field>`/`plan`
# operand. Funnelling `(config, plan, sc, …)` into per-transform kernels
# (`_vector_synthesis_kernel!` etc.) restores specialization. Measured effect on
# the tiny CNAB2 fixture (nr=16, lmax=4, mag+comp):
#   * whole solver_step!: ~587 KB/step (boxing) -> ~385 KB/step
#   * vector_spectral_to_physical!: ~75 KB/call -> ~36 KB/call
# The remaining allocation is upstream (SHTnsKit/PencilArrays transform internals)
# plus the `::Any` plan handle itself, and is out of scope here.
#
# These guards pin the type-stable path: a regression that drops the barriers
# (reintroducing the boxing) pushes both figures back above the thresholds.

using Test
using MPI
using GeoDynamo

# Function barriers so the figures reflect the callee's own heap traffic, not
# call-site dispatch of the non-const test locals.
_alloc_step(s)            = @allocated GeoDynamo.solver_step!(s)
_alloc_vec_synth(t, p, v, d) = @allocated GeoDynamo.vector_spectral_to_physical!(t, p, v; domain = d)

@testset "SH transform path stays type-stable (no per-operand boxing)" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping transform allocation regression"
        return
    end
    MPI.Initialized() || MPI.Init()

    params = GeoDynamo.SolverParameters(
        architecture = :cpu, geometry = :shell,
        nr = 16, nr_inner = 4, lmax = 4, mmax = 4, nlat = 12, nlon = 16,
        timestep = 1e-4, stop_iteration = 1000,
        include_magnetic = true, include_composition = true,
        timestepper = GeoDynamo.CNAB2(), topography_enabled = false, stefan_enabled = false,
    )
    state = GeoDynamo.initialize_simulation(Float64, params)
    GeoDynamo.initialize_fields!(state)
    # Warm: compile hot paths and populate the per-config transform/scratch caches.
    for _ in 1:6
        GeoDynamo.solver_step!(state)
    end

    # Whole-step steady-state allocation (robust aggregate; min removes GC noise).
    per_step = minimum(_alloc_step(state) for _ in 1:8)

    # Vector synthesis per call (the dominant boxing site the barrier targets).
    vel    = state.fields.velocity
    domain = state.runtime.outer_core_domain
    GeoDynamo.vector_spectral_to_physical!(vel.toroidal, vel.poloidal, vel.velocity; domain = domain)
    vec_synth = minimum(_alloc_vec_synth(vel.toroidal, vel.poloidal, vel.velocity, domain) for _ in 1:5)

    # Boxing baseline was ~587 KB/step and ~75 KB/call; the type-stable path is
    # ~385 KB/step and ~36 KB/call. Thresholds sit between, so dropping the
    # barriers (regression) fails, while the current path passes with margin.
    @test per_step  < 480_000
    @test vec_synth <  50_000
end
