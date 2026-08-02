using Test
using GeoDynamo
include(joinpath(@__DIR__, "gpu_test_preamble.jl"))
using MPI
using Random

MPI.Initialized() || MPI.Init()

# =============================================================================
# GPU≈CPU parity across the boundary-condition combinations.
#
# The main device gates (5n2 / phase6 / gpu_erk2_step) run the default config
# (no-slip walls, Dirichlet-Dirichlet temperature). The packed BC machinery is
# shared across all combos — velocity codes 1–4 change the W-split endpoint
# rows, scalar DD/DN/ND/NN change the implicit BC rows (CNAB2) and the
# boundary packs (ERK2) — but each combo deserves its own gate. Velocity +
# temperature configs only (composition reuses the thermal machinery;
# insulating magnetic is gated elsewhere). Two steps per combo.
# =============================================================================

const _NS = GeoDynamo.NoSlip()
const _SF = GeoDynamo.StressFree()
const _VAL = GeoDynamo.ValueBoundaryCondition
const _FLX = GeoDynamo.FluxBoundaryCondition

function _combo_state(vel_bcs, temp_bcs, timestepper)
    params = GeoDynamo.SolverParameters(
        geometry = :shell, lmax = 6, mmax = 6, nlat = 14, nlon = 28, nr = 8, nr_inner = 4,
        radial_bandwidth = 3, radius_ratio = 0.35,
        Ek = 1e-3, Ra = 1e5, Pm = 1.0, Pr = 1.0, timestep = 1e-4,
        include_magnetic = false, include_composition = false,
        timestepper = timestepper,
        velocity_bcs = vel_bcs,
        temperature_bcs = temp_bcs)
    st = GeoDynamo.initialize_solver_state(Float64; params = params)
    rng = MersenneTwister(7)
    for f in (st.fields.temperature.spectral,
              st.fields.velocity.toroidal, st.fields.velocity.poloidal)
        parent(f.data_real) .+= 1e-3 .* (rand(rng, size(parent(f.data_real))...) .- 0.5)
        parent(f.data_imag) .+= 1e-3 .* (rand(rng, size(parent(f.data_imag))...) .- 0.5)
    end
    return st
end

function _combo_parity(vel_bcs, temp_bcs, timestepper; nsteps = 2)
    st = _combo_state(vel_bcs, temp_bcs, timestepper)
    GeoDynamo.solver_step!(st)                  # warm-up
    gst = GeoDynamo.build_gpu_solver_state(st)
    erk = timestepper isa GeoDynamo.ExponentialRungeKutta2 ? GeoDynamo.build_gpu_erk2_state(st) : nothing
    cfg = st.backend.shtns_config
    nr = st.runtime.outer_core_domain.N
    for _ in 1:nsteps
        GeoDynamo.solver_step!(st)
        erk === nothing ? GeoDynamo.gpu_solver_step!(gst) :
                          GeoDynamo.gpu_erk2_solver_step!(gst, erk)
    end
    worst = 0.0
    for (spec, gr, gi) in [
            (st.fields.temperature.spectral, gst.temperature.spec_r, gst.temperature.spec_i),
            (st.fields.velocity.toroidal, gst.velocity.tor.spec_r, gst.velocity.tor.spec_i),
            (st.fields.velocity.poloidal, gst.velocity.pol.spec_r, gst.velocity.pol.spec_i)]
        cr, ci = GeoDynamo.cpu_spectral_to_dense(spec, cfg, nr, Float64)
        worst = max(worst, maximum(abs, cr .- gr), maximum(abs, ci .- gi))
    end
    finite = all(isfinite, gst.velocity.pol.spec_r) && all(isfinite, gst.temperature.spec_r)
    return worst, finite
end

# Absolute agreement floor for these combos, measured across all 16 of them
# (velocity codes 1-4 x DD/DN/ND/NN, CNAB2 and ERK2): worst = 3.7e-12, ~3 orders
# above the plain full-step gates in gpu_test_preamble.jl (GPU_LOCAL_ATOL) because
# ERK2 and the inhomogeneous BC values are less well conditioned. 1e-10 keeps ~27x
# headroom while being 10x tighter than the 1e-9 it replaced. Do not lower without
# re-measuring: a shared constant across gate families does NOT hold here.
const COMBO_PARITY_ATOL = 1e-10

@testset "GPU≈CPU parity across BC combinations" begin
    dd = GeoDynamo.BoundaryConditions(inner = _VAL(1.0), outer = _VAL(0.0))

    @testset "velocity walls (codes 1–4), DD temperature" begin
        for (label, bcs) in [
                ("NS/NS", GeoDynamo.BoundaryConditions(inner = _NS, outer = _NS)),
                ("NS/SF", GeoDynamo.BoundaryConditions(inner = _NS, outer = _SF)),
                ("SF/NS", GeoDynamo.BoundaryConditions(inner = _SF, outer = _NS)),
                ("SF/SF", GeoDynamo.BoundaryConditions(inner = _SF, outer = _SF))]
            for ts in (GeoDynamo.CNAB2(), GeoDynamo.ExponentialRungeKutta2())
                worst, finite = _combo_parity(bcs, dd, ts)
                @testset "$label $(nameof(typeof(ts)))" begin
                    @test worst < COMBO_PARITY_ATOL
                    @test finite
                end
            end
        end
    end

    @testset "temperature combos (DD/DN/ND/NN), no-slip walls" begin
        ns = GeoDynamo.BoundaryConditions(inner = _NS, outer = _NS)
        for (label, bcs) in [
                ("DN", GeoDynamo.BoundaryConditions(inner = _VAL(1.0), outer = _FLX(0.5))),
                ("ND", GeoDynamo.BoundaryConditions(inner = _FLX(1.0), outer = _VAL(0.0))),
                ("NN", GeoDynamo.BoundaryConditions(inner = _FLX(1.0), outer = _FLX(0.0)))]
            for ts in (GeoDynamo.CNAB2(), GeoDynamo.ExponentialRungeKutta2())
                worst, finite = _combo_parity(ns, bcs, ts)
                @testset "$label $(nameof(typeof(ts)))" begin
                    @test worst < COMBO_PARITY_ATOL
                    @test finite
                end
            end
        end
    end
end
