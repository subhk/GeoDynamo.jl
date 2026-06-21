using Test
using GeoDynamo
using MPI
using LinearAlgebra

MPI.Initialized() || MPI.Init()

# CB3 / RungeKutta3 diffusion consistency.
#
# The Cavaglieri-Bewley IMEX-RK3 must integrate the implicit (diffusion) operator
# L with the companion Crank-Nicolson split coefficients α/β whose stage weights
# sum to 1. The buggy implementation treated L fully implicitly with the EXPLICIT
# stage coefficients γ (Σγ = 1.7) and dropped the explicit α·L term, so the
# diffusion operator was over-integrated by ~70% per step — not even first-order
# consistent for the linear term.
#
# Test (pure diffusion): with no velocity (u = 0) and Ra = 0 the temperature
# equation reduces to ∂T/∂t = (1/Pr)∇²T. CB3 and the independently order-verified
# CNAB2 are both 2nd-order schemes, so on this linear problem they must agree to
# O(dt²). The bug makes CB3 decay the field dramatically faster than CNAB2.

const DT = 5.0e-4
const KSTEPS = 20

function _zero_field!(f)
    parent(f.data_real) .= 0
    parent(f.data_imag) .= 0
    return f
end

function build_pure_diffusion_state(stepper)
    params = GeoDynamo.SolverParameters(
        geometry = :shell,
        lmax = 4, mmax = 4, nlat = 10, nlon = 20, nr = 12, nr_inner = 4,
        radial_bandwidth = 3, radius_ratio = 0.35,
        Ek = 1e-3, Ra = 1e-10, Pm = 1.0, Pr = 1.0,  # Ra→0⁺: buoyancy negligible, u stays ≈0
        timestep = DT,
        include_magnetic = false, include_composition = false,
        timestepper = stepper,
        temperature_bcs = GeoDynamo.BoundaryConditions(
            inner = GeoDynamo.FixedTemperature(0.0),
            outer = GeoDynamo.FixedTemperature(0.0),
        ),
    )
    st = GeoDynamo.initialize_solver_state(Float64; params)
    GeoDynamo.initialize_solver_fields!(st)   # consume one-shot IC regeneration

    # Kill velocity entirely so temperature is purely diffusive and stays so
    # (Ra = 0 ⇒ no buoyancy forcing ⇒ u remains 0 across steps).
    v = st.fields.velocity
    for f in (v.toroidal, v.poloidal, v.nl_toroidal, v.prev_nl_toroidal,
              v.nl_poloidal, v.prev_nl_poloidal)
        _zero_field!(f)
    end

    # Deterministic smooth temperature seed: a single radial half-sine bump
    # (zero at both Dirichlet walls) on every owned (l,m) mode.
    cfg = st.backend.shtns_config
    tr = parent(st.fields.temperature.spectral.data_real)
    _zero_field!(st.fields.temperature.spectral)
    nr = size(tr, 3)
    for lm in 1:cfg.nlm
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue
        for r in 1:nr
            val = sin(pi * (r - 1) / (nr - 1)) * 1e-3 / lm
            GeoDynamo.set_local_spectral_value!(tr, slot, r, val)
        end
    end
    _zero_field!(st.fields.temperature.nonlinear)
    _zero_field!(st.fields.temperature.prev_nonlinear)
    return st
end

@testset "CB3 pure-diffusion consistency (matches CNAB2 to O(dt²))" begin
    cb3 = build_pure_diffusion_state(GeoDynamo.RungeKutta3())
    cnab = build_pure_diffusion_state(GeoDynamo.CNAB2())

    for _ in 1:KSTEPS
        GeoDynamo.solver_step!(cb3)
        GeoDynamo.solver_step!(cnab)
    end

    Tcb_r = parent(cb3.fields.temperature.spectral.data_real)
    Tcn_r = parent(cnab.fields.temperature.spectral.data_real)

    @test all(isfinite, Tcb_r)
    # Sanity: CNAB2 actually diffused the field (it did not vanish or blow up).
    @test 0.05 < norm(Tcn_r) / (1e-3) < 1e3

    reldiff = norm(Tcb_r .- Tcn_r) / max(norm(Tcn_r), eps())
    @info "CB3 vs CNAB2 pure-diffusion relative difference" reldiff
    # Two 2nd-order schemes on a linear problem agree to O(dt²); the ~70%
    # over-integration bug pushes this to order 1.
    @test reldiff < 0.02
end
