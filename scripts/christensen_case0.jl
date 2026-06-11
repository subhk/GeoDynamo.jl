# Christensen et al. (2001) dynamo benchmark, Case 0 (non-magnetic rotating
# convection). Their scaling: E(∂t + u·∇ − ∇²)u + 2ẑ×u + ∇P = Ra·(r/r_o)·T,
# E = 1e-3, Ra = 100, Pr = 1, η = 0.35, no-slip, fixed ΔT.
#
# GeoDynamo mapping (momentum Ek·∂t u = −Ek(u×ω) − ẑ×u + Ra_geo·r·Θ + Ek∇²u):
#   Coriolis 1 vs their 2  ⇒ Ek_geo = E/2     = 5.0e-4
#   buoyancy r vs r/r_o    ⇒ Ra_geo = Ra/(2r_o) = 100/(2·1.5384615…) = 32.5
#   Pm = Pr = 1 (magnetic-diffusion time ≡ viscous time), η = 0.35.
#
# Benchmark target: mean kinetic energy density Ekin = (1/2V)∫u²dV = 58.348,
# quasi-stationary m=4 drifting wave.
using GeoDynamo
using MPI
using Printf
using LinearAlgebra

MPI.Initialized() || MPI.Init()

const LMAX = 32
const NR = 33
const DT = 1.0e-4
const NSTEPS = parse(Int, get(ENV, "CASE0_NSTEPS", "20000"))
const REPORT_EVERY = 200

params = GeoDynamo.SolverParameters(architecture = :cpu, 
                    geometry = :shell,
                    nr = NR, nr_inner = 8, 
                    lmax = LMAX, mmax = LMAX, 
                    nlat = 48, nlon = 96,
                    Ra = 25.0, Ek = 5.0e-4, Pr = 1.0, Pm = 1.0, Sc = 1.0,
                    timestep = DT, start_time = 0.0, end_time = 10.0, stop_iteration = 10_000_000,
                    include_magnetic = false, include_composition = false,
                    timestepper = GeoDynamo.CNAB2(), topography_enabled = false, stefan_enabled = false,
                )
                
state = GeoDynamo.initialize_simulation(Float64, params)
GeoDynamo.initialize_solver_fields!(state)

# IC: keep the code's conductive l=0 profile; REPLACE the random l≥1 content
# with the benchmark's m=4 seed — radial profile (1−3x²+3x⁴−x⁶), x = 2r−ri−ro
# (vanishes at both walls; BC-compatible), on the (l=4, m=4) mode only.
temp = state.fields.temperature
cfg = temp.config
dom = state.backend.outer_core_domain
ri = dom.r[1, 4]; ro = dom.r[dom.N, 4]
sr = parent(temp.spectral.data_real); si = parent(temp.spectral.data_imag)
for lm in 1:cfg.nlm
    l = cfg.l_values[lm]
    l == 0 && continue
    slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
    slot === nothing && continue
    seed = (l == 4 && cfg.m_values[lm] == 4)
    for r_idx in 1:dom.N
        x = 2 * dom.r[r_idx, 4] - ri - ro
        amp = seed ? 0.1 * (1 - 3x^2 + 3x^4 - x^6) : 0.0
        GeoDynamo.set_local_spectral_value!(sr, slot, r_idx, amp)
        GeoDynamo.set_local_spectral_value!(si, slot, r_idx, 0.0)
    end
end

# ---- diagnostics ------------------------------------------------------------
const V_SHELL = (4π / 3) * (ro^3 - ri^3)

# radial trapezoid weights on the (non-uniform) grid
const RW = let
    w = zeros(dom.N)
    for k in 1:(dom.N - 1)
        h = dom.r[k + 1, 4] - dom.r[k, 4]
        w[k] += h / 2; w[k + 1] += h / 2
    end
    w
end

# physical-grid kinetic energy: Ekin = (1/2V)∫u² dV
# (Gauss weights over θ sum to 2 and absorb sinθ; φ uniform.)
function ekin(state)
    v = state.fields.velocity
    Vph = GeoDynamo.create_shtns_vector_field(Float64, cfg, dom,
        (cfg.pencils.θ, cfg.pencils.φ, cfg.pencils.r))
    GeoDynamo.vector_spectral_to_physical!(v.toroidal, v.poloidal, Vph; domain = dom)
    ur = parent(Vph.r_component.data); ut = parent(Vph.θ_component.data)
    up = parent(Vph.φ_component.data)
    gw = cfg.gauss_weights
    dφ = 2π / cfg.nlon
    s = 0.0
    @inbounds for k in 1:dom.N, j in 1:cfg.nlon, i in 1:cfg.nlat
        u2 = ur[i, j, k]^2 + ut[i, j, k]^2 + up[i, j, k]^2
        s += u2 * gw[i] * dφ * RW[k] * dom.r[k, 4]^2
    end
    return s / (2 * V_SHELL)
end

# energy fraction in m ≡ 0 (mod 4) of the velocity (m=4 wave check), from the
# spectral storage m-columns (slot m-index = m+1)
function m4_fraction(state)
    v = state.fields.velocity
    tot = 0.0; m4 = 0.0
    for spec in (v.toroidal, v.poloidal)
        a = parent(spec.data_real); b = parent(spec.data_imag)
        for mi in 1:size(a, 2)
            e = sum(abs2, @view a[:, mi, :]) + sum(abs2, @view b[:, mi, :])
            tot += e
            ((mi - 1) % 4 == 0) && (m4 += e)
        end
    end
    return tot > 0 ? m4 / tot : 0.0
end

println("# Case 0: Ek_geo=5e-4 Ra_geo=25.0 (criticality-calibrated: Rac_geo=13.9 measured, benchmark Ra/Rac=1.79) lmax=$LMAX nr=$NR dt=$DT steps=$NSTEPS")
println("# target Ekin = 58.348 (Christensen et al. 2001)")
flush(stdout)
t0 = time()
for step in 1:NSTEPS
    GeoDynamo.solver_step!(state)
    if step % REPORT_EVERY == 0
        E = ekin(state)
        f4 = m4_fraction(state)
        @printf("step %6d  t=%.4f  Ekin=%.4f  m4frac=%.3f  wall=%.0fs\n",
            step, step * DT, E, f4, time() - t0)
        flush(stdout)
        isfinite(E) || (println("DIVERGED"); break)
    end
end
println("CASE0_DONE")
