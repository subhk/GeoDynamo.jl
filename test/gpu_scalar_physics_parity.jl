using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
using MPI
using Random

MPI.Initialized() || MPI.Init()

# =============================================================================
# GPU≈CPU parity for scalar physics the default device gates CANNOT see, because
# every other GPU gate runs Pm=Pr=1 with no internal source:
#
#   (1) Scalar mass coefficient. The CPU scalar implicit matrices fold the
#       diffusivity into L and use a 1/dt mass term (scalar_bc.jl, mass_coeff=1).
#       The GPU reuses those CPU LUs, so `inv_dt_temp`/`inv_dt_comp` MUST be 1/dt,
#       NOT (Pm/Pr)/dt — otherwise temperature integrates a wrong coefficient
#       whenever Pm/Pr ≠ 1 (the normal regime). This test runs Pr ≠ Pm.
#
#   (2) Internal heating. The GPU scalar nonlinear must add the internal-source
#       profile into the advection before the spectral analysis, like the CPU
#       (solver_add_internal_sources_local!). The source profile is set directly
#       on the field so the parity genuinely exercises the source-add (the
#       internal_heating IC plumbing populates it lazily, off this code path).
#
# Both run on the Array backend (the GPU≈CPU gate); two CNAB2 steps.
# =============================================================================

function _scalar_phys_state(; Pr)
    params = GeoDynamo.SolverParameters(
        geometry = :shell, lmax = 6, mmax = 6, nlat = 14, nlon = 28, nr = 8, nr_inner = 4,
        radial_bandwidth = 3, radius_ratio = 0.35,
        Ek = 1e-3, Ra = 1e5, Pm = 1.0, Pr = Pr, timestep = 1e-4,
        include_magnetic = false, include_composition = false,
        temperature_bcs = GeoDynamo.BoundaryConditions(
            inner = GeoDynamo.ValueBoundaryCondition(1.0),
            outer = GeoDynamo.ValueBoundaryCondition(0.0)))
    st = GeoDynamo.initialize_solver_state(Float64; params = params)
    rng = MersenneTwister(7)
    for f in (st.fields.temperature.spectral,
              st.fields.velocity.toroidal, st.fields.velocity.poloidal)
        parent(f.data_real) .+= 1e-3 .* (rand(rng, size(parent(f.data_real))...) .- 0.5)
        parent(f.data_imag) .+= 1e-3 .* (rand(rng, size(parent(f.data_imag))...) .- 0.5)
    end
    return st
end

# `st` must already be warmed up (one solver_step!) so the velocity lag is set
# and build_gpu_solver_state captures a consistent state.
function _scalar_parity(st; nsteps = 2)
    gst = GeoDynamo.build_gpu_solver_state(st)
    cfg = st.backend.shtns_config
    nr = st.runtime.outer_core_domain.N
    for _ in 1:nsteps
        GeoDynamo.solver_step!(st)
        GeoDynamo.gpu_solver_step!(gst)
    end
    worst = 0.0
    for (spec, gr, gi) in [
            (st.fields.temperature.spectral, gst.temperature.spec_r, gst.temperature.spec_i),
            (st.fields.velocity.toroidal, gst.velocity.tor.spec_r, gst.velocity.tor.spec_i),
            (st.fields.velocity.poloidal, gst.velocity.pol.spec_r, gst.velocity.pol.spec_i)]
        cr, ci = GeoDynamo.cpu_spectral_to_dense(spec, cfg, nr, Float64)
        worst = max(worst, maximum(abs, cr .- gr), maximum(abs, ci .- gi))
    end
    finite = all(isfinite, gst.temperature.spec_r) && all(isfinite, gst.velocity.pol.spec_r)
    return worst, finite
end

# Measured absolute agreement floor for these two configs (Pr=2 mass coefficient,
# nonzero internal heating): worst = 2.5e-13. 1e-11 keeps ~40x headroom and is
# 100x tighter than the 1e-9 it replaced. Re-measure before lowering.
const SCALAR_PHYS_ATOL = 1e-11

@testset "GPU≈CPU scalar physics parity (Pm/Pr≠1, internal heating)" begin
    @testset "Pm/Pr ≠ 1 (Pr = 2): scalar mass coefficient" begin
        st = _scalar_phys_state(; Pr = 2.0)
        @test st.parameters.Pm / st.parameters.Pr != 1.0    # the regime the old (Pm/Pr)/dt bug corrupted
        GeoDynamo.solver_step!(st)                           # warm-up
        worst, finite = _scalar_parity(st)
        @test worst < SCALAR_PHYS_ATOL
        @test finite
    end

    @testset "internal heating ≠ 0" begin
        st = _scalar_phys_state(; Pr = 1.5)
        GeoDynamo.solver_step!(st)                           # warm-up (is_initialized ⇒ no later IC clobber)
        # Force a nonzero radial source profile so the parity exercises the GPU
        # source-add. The CPU reads field.internal_sources every step; build_gpu_solver_state
        # captures it into the device bundle.
        fill!(st.fields.temperature.internal_sources, 2.0)
        @test !all(iszero, st.fields.temperature.internal_sources)
        worst, finite = _scalar_parity(st)
        @test worst < SCALAR_PHYS_ATOL
        @test finite
    end
end
