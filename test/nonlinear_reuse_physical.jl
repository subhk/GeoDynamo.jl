using Test
using GeoDynamo
using MPI

MPI.Initialized() || MPI.Init()

# Efficiency fix: compute_solver_nonlinear_terms! refreshes the scalar/magnetic
# physical fields once up front (for buoyancy + Lorentz), then each per-field
# nonlinear pass used to RE-synthesize the same fields from unchanged spectral.
# The optimization reuses the up-front buffers (reuse_physical=true) instead.
#
# This MUST be behavior-preserving: the nonlinear outputs with reuse_physical=true
# must equal those computed the redundant way (reuse_physical=false). Both calls
# read the SAME (unmodified) spectral state, so any difference is a bug in the
# skip logic.

function _nl_snapshot(state)
    v = state.fields.velocity
    b = state.fields.magnetic
    t = state.fields.temperature
    c = state.fields.composition
    snap = Dict{String,Array{Float64}}()
    snap["v_tor_r"] = copy(parent(v.nl_toroidal.data_real))
    snap["v_tor_i"] = copy(parent(v.nl_toroidal.data_imag))
    snap["v_pol_r"] = copy(parent(v.nl_poloidal.data_real))
    snap["v_pol_i"] = copy(parent(v.nl_poloidal.data_imag))
    if b !== nothing
        snap["b_tor_r"] = copy(parent(b.nl_toroidal.data_real))
        snap["b_pol_r"] = copy(parent(b.nl_poloidal.data_real))
        snap["b_tor_i"] = copy(parent(b.nl_toroidal.data_imag))
        snap["b_pol_i"] = copy(parent(b.nl_poloidal.data_imag))
    end
    snap["t_nl_r"] = copy(parent(t.nonlinear.data_real))
    snap["t_nl_i"] = copy(parent(t.nonlinear.data_imag))
    if c !== nothing
        snap["c_nl_r"] = copy(parent(c.nonlinear.data_real))
        snap["c_nl_i"] = copy(parent(c.nonlinear.data_imag))
    end
    return snap
end

@testset "compute_solver_nonlinear_terms! reuse_physical is behavior-preserving" begin
    params = GeoDynamo.SolverParameters(
        geometry = :shell, lmax = 6, mmax = 6, nlat = 12, nlon = 24, nr = 12,
        nr_inner = 4, radial_bandwidth = 3, radius_ratio = 0.35,
        Ek = 1e-3, Ra = 1e5, Pm = 1.0, Pr = 1.0, timestep = 1e-4,
        include_magnetic = true, include_composition = true,
        timestepper = GeoDynamo.CNAB2(),
    )
    st = GeoDynamo.initialize_solver_state(Float64; params)
    GeoDynamo.initialize_solver_fields!(st)

    # Redundant path (full re-synthesis in each per-field pass).
    GeoDynamo.compute_solver_nonlinear_terms!(st; reuse_physical = false)
    ref = _nl_snapshot(st)

    # Optimized path — reads the same (unchanged) spectral state, reuses up-front
    # physical buffers. Must match the redundant path bit-for-bit.
    GeoDynamo.compute_solver_nonlinear_terms!(st; reuse_physical = true)
    opt = _nl_snapshot(st)

    for k in keys(ref)
        @test opt[k] == ref[k]
    end
end
