# Marti et al. (2014) full-sphere Benchmark 1 (rotating thermal convection,
# hydrodynamic). "Full sphere hydrodynamic and dynamo benchmarks", Geophys. J.
# Int. 197(1):119-134, doi:10.1093/gji/ggt518.
#
# Sources (fetched 2026-06-12):
#   - paper PDF:  https://eprints.gla.ac.uk/88879/1/88879.pdf
#   - OUP page:   https://academic.oup.com/gji/article/197/1/119/686457
#
# Published problem definition (paper Section 2.1, verbatim values):
#   gravity g = g·r/r_o (linear); homogeneous heat source S; basic state
#   T_b = (β/2)(r_o² − r²), β = S/(3κ). Nondimensionalization: length r_o,
#   time r_o²/ν (viscous), temperature β·r_o².
#     E  = ν/(2Ω r_o²) = 3e-4          (NOTE: factor 2 in their Ekman number)
#     Pr = ν/κ = 1
#     Ra = g α β r_o³/(2Ω κ) = 95,  S = 3
#   Momentum (eq 6):  E(∂t − ∇²)u = E u∧(∇∧u) + Ra·T·r − ẑ∧u − ∇π
#   Temperature (eq 8): (Pr ∂t − ∇²)T = S − Pr u·∇T;  T_b = (1−r²)/2 (eq 9)
#   BCs: stress-free + impenetrable wall, fixed temperature at r_o.
#   IC (eq 10): T = (1−r²)/2 + (1e-5/8)√(35/π)·r³(1−r²)(cos3φ+sin3φ)sin³θ,
#   u = 0 — the (l=3,m=3) perturbation selects the m=3 branch (the alternative
#   branch is m=4). Solution: quasi-stationary drifting wave, 3-fold azimuthal
#   symmetry, equatorially antisymmetric velocity.
#
# Published targets (paper Table 2, "Requested values"):
#   E_k = (1/2)∫_V u² dV  = 29.1206 ± 1e-4    (NOT volume-normalized)
#   f_d                   = 12.3862 ± 8e-4
# Drift definition (paper eqs 13-14 + text): f_d is the frequency at which the
# flow pattern passes a fixed point in space; the whole pattern rotates at
# f̂_d = f_d/3. Hence the (l,m)=(3,3) spectral coefficient's complex phase
# advances at the angular rate 2π·f_d per (viscous) time unit — we measure f_d
# as |d(arg c₃₃)/dt|/(2π).
#
# GeoDynamo mapping (momentum, src/physics/velocity/field.jl docstring, mass
# coefficient Ek in the timestepping matrices, magnetic-diffusion time unit):
#   Ek(∂t + advection)u + ẑ×u = −∇p + (Pm/Pr)·Ra·T·r·r̂ + Ek∇²u
# With Pm = Pr = 1 the magnetic-diffusion time ≡ the viscous time and the
# equation matches Marti's form TERM FOR TERM (Coriolis coefficient 1 in both;
# Marti's E already contains the 2Ω):
#   Ek_geo = E_marti           = 3.0e-4   (no factor 2, unlike Christensen)
#   Ra_geo = Ra_marti          = 95       (buoyancy Ra·T·r ≡ Ra·T·r, r_o = 1)
#   internal heating amplitude = S/Pr     = 3  (enters ∂t T directly:
#       GeoDynamo: ∂tT + u·∇T = (Pm/Pr)∇²T + src;
#       Marti /Pr: ∂tT + u·∇T = (1/Pr)∇²T + S/Pr)
#   temperature units anchored by the SOURCE + κ_nondim = Pm/Pr = 1 + fixed
#       T(r_o) = 0, so (unlike the Christensen Case-0 mapping, which had a
#       measured ×1.3 residual traced to the IC ΔT scale) no IC-amplitude
#       ambiguity survives the thermal transient. The mapping SHOULD be exact;
#       MARTI_RA_SCALE is exposed for sensitivity studies anyway, and the
#       verdict line reports the raw values so both can be compared.
#   velocity/time units (Pm=1): u in ν/r_o, t in r_o²/ν ⇒ Ekin and f_d are
#       directly comparable to the published numbers.
# BCs: ball ⇒ inner wall replaced by center-regularity Robin rows everywhere
# (scalar, toroidal, W-split poloidal); the OUTER rows follow velocity_bcs.
# velocity_bcs = (inner=NoSlip [ignored: regularity], outer=StressFree) ⇒
# bc code 2; W-split builds the stress-free outer residual P″ − (2/r)P′ = 0
# and the toroidal solve uses T′ − T/r = 0 (src/bcs/velocity_bc.jl).
# CAVEAT: no explicit angular-momentum correction is applied for the
# stress-free wall (Marti et al. note several codes correct Lz); from the
# zero-net-rotation IC the l=1,m=0 toroidal mode should stay ~0 numerically.
#
# IC here: the exact conductive state T₀ = (1−r²)/2 on l=0 (steady solution of
# ∂tT = ∇²T + 3 with T(1) = 0) + an (l=3,m=3) seed with the BC/regularity-
# compatible radial profile r³(1−r²). Seed amplitude 1e-3 (larger than the
# paper's ~4e-6 to shorten the transient; the saturated state is independent
# of it — only the branch selection matters). u = 0 (initialize zeroes it).
using GeoDynamo
using MPI
using Printf
using LinearAlgebra

MPI.Initialized() || MPI.Init()

const LMAX = parse(Int, get(ENV, "MARTI_LMAX", "32"))
const NR = parse(Int, get(ENV, "MARTI_NR", "48"))
const NSTEPS = parse(Int, get(ENV, "MARTI_NSTEPS", "20000"))
# NOTE: reliable f_d needs the post-transient window; default NSTEPS=20000 gives t=2.0 which may still graze the transient (paper Fig 3) — use 50000+ for publication-grade drift values.
const DT = parse(Float64, get(ENV, "MARTI_DT", "1.0e-4"))
const RA_SCALE = parse(Float64, get(ENV, "MARTI_RA_SCALE", "1.0"))
const RA = 95.0 * RA_SCALE
const EK = 3.0e-4
const S_HEAT = 3.0
const REPORT_EVERY = 200
const DRIFT_EVERY = 20    # coefficient-phase sampling; 2π·f_d·(20·dt) ≪ π

params = GeoDynamo.SolverParameters(architecture = :cpu,
    geometry = :ball, radius_ratio = 0.0,
    nr = NR, lmax = LMAX, mmax = LMAX,
    nlat = 2 * LMAX, nlon = 4 * LMAX,
    Ra = RA, Ek = EK, Pr = 1.0, Pm = 1.0, Sc = 1.0,
    timestep = DT, start_time = 0.0, end_time = 100.0,
    stop_iteration = 10_000_000,
    include_magnetic = false, include_composition = false,
    timestepper = GeoDynamo.CNAB2(),
    velocity_bcs = GeoDynamo.BoundaryConditions(
        inner = GeoDynamo.NoSlip(), outer = GeoDynamo.StressFree()),
    temperature_bcs = GeoDynamo.BoundaryConditions(
        inner = GeoDynamo.FixedTemperature(0.0),
        outer = GeoDynamo.FixedTemperature(0.0)),
    topography_enabled = false, stefan_enabled = false,
)

state = GeoDynamo.initialize_simulation(Float64, params)
GeoDynamo.initialize_solver_fields!(state)   # consume one-shot init BEFORE seeding

temp = state.fields.temperature
cfg = temp.spectral.config
dom = state.backend.outer_core_domain
ri = dom.r[1, 4]; ro = dom.r[dom.N, 4]

# Marti's uniform internal heating S = 3 (steady-state balance ∇²T₀ = −3).
GeoDynamo.set_internal_heating!(temp, dom; heating_type = :uniform,
    amplitude = S_HEAT)

# IC: REPLACE the code's default temperature with the benchmark's — l=0
# conductive state (1−r²)/2 (orthonormal Y₀⁰ = 1/√4π ⇒ store mean·√4π) plus
# the (l=3,m=3) branch-selection seed; everything else zero.
let sr = parent(temp.spectral.data_real), si = parent(temp.spectral.data_imag)
    for lm in 1:cfg.nlm
        l = cfg.l_values[lm]
        m = cfg.m_values[lm]
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue
        for r_idx in 1:dom.N
            r = dom.r[r_idx, 4]
            amp = if l == 0 && m == 0
                sqrt(4π) * 0.5 * (1.0 - r^2)
            elseif l == 3 && m == 3
                1.0e-3 * r^3 * (1.0 - r^2)
            else
                0.0
            end
            GeoDynamo.set_local_spectral_value!(sr, slot, r_idx, amp)
            GeoDynamo.set_local_spectral_value!(si, slot, r_idx, 0.0)
        end
    end
end

# ---- diagnostics ------------------------------------------------------------

# radial trapezoid weights on the (non-uniform, off-center) grid; the omitted
# core r < r₁ contributes O(r₁³) ~ 1e-9 of the volume — negligible.
const RW = let
    w = zeros(dom.N)
    for k in 1:(dom.N - 1)
        h = dom.r[k + 1, 4] - dom.r[k, 4]
        w[k] += h / 2; w[k + 1] += h / 2
    end
    w
end

# physical-grid kinetic energy: Ekin = (1/2)∫u² dV — Marti's eq (12), NOT
# volume-normalized. (Gauss weights over θ sum to 2 and absorb sinθ; φ uniform.)
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
    return s / 2
end

# energy fraction in m ≡ 0 (mod 3) of the velocity (3-fold-symmetry check),
# from the spectral storage m-columns (slot m-index = m+1)
function m3_fraction(state)
    v = state.fields.velocity
    tot = 0.0; m3 = 0.0
    for spec in (v.toroidal, v.poloidal)
        a = parent(spec.data_real); b = parent(spec.data_imag)
        for mi in 1:size(a, 2)
            e = sum(abs2, @view a[:, mi, :]) + sum(abs2, @view b[:, mi, :])
            tot += e
            ((mi - 1) % 3 == 0) && (m3 += e)
        end
    end
    return tot > 0 ? m3 / tot : 0.0
end

# drift diagnostic: complex (l=3,m=3) temperature coefficient at mid-radius;
# its phase advances at 2π·f_d per time unit (see header).
const LM33 = let idx = findfirst(
        i -> cfg.l_values[i] == 3 && cfg.m_values[i] == 3, 1:cfg.nlm)
    idx === nothing && error("(l=3,m=3) not in config — need LMAX ≥ 3")
    idx
end
const SLOT33 = GeoDynamo.local_spectral_storage_slot(cfg, LM33)
const RMID = argmin(abs.(dom.r[1:dom.N, 4] .- 0.5))

function drift_sample(state)
    SLOT33 === nothing && return complex(NaN, NaN)   # mode not local (multi-rank run)
    sr = parent(state.fields.temperature.spectral.data_real)
    si = parent(state.fields.temperature.spectral.data_imag)
    return complex(GeoDynamo.local_spectral_value(sr, SLOT33, RMID),
        GeoDynamo.local_spectral_value(si, SLOT33, RMID))
end

# unwrapped-phase linear fit over the LAST QUARTER of the samples → f_d
function drift_frequency(ts::Vector{Float64}, cs::Vector{ComplexF64})
    n = length(ts)
    (n < 8 || abs(cs[end]) < 1e-14) && return NaN  # mode not yet excited
    ph = Vector{Float64}(undef, n)
    ph[1] = angle(cs[1])
    for k in 2:n
        dp = angle(cs[k]) - angle(cs[k - 1])
        dp > π && (dp -= 2π)
        dp < -π && (dp += 2π)
        ph[k] = ph[k - 1] + dp
    end
    i0 = max(1, n - n ÷ 4 + 1)
    t = @view ts[i0:n]; p = @view ph[i0:n]
    tm = sum(t) / length(t); pm = sum(p) / length(p)
    slope = sum((t .- tm) .* (p .- pm)) / sum(abs2, t .- tm)
    return slope / (2π)   # signed; |f_d| is the benchmark number
end

println("# Marti et al. (2014) Benchmark 1 (full sphere, hydro): " *
        "Ek=$EK Ra=$RA (=95×$RA_SCALE) Pr=1 S=$S_HEAT stress-free, fixed T(r_o)=0")
println("# lmax=mmax=$LMAX nlat=$(2LMAX) nlon=$(4LMAX) nr=$NR (r∈[$(round(ri, sigdigits=3)),$ro]) dt=$DT steps=$NSTEPS CNAB2")
println("# targets (Table 2): Ekin = 29.1206 ± 1e-4, f_d = 12.3862 ± 8e-4 (m=3 drifting wave)")
flush(stdout)

drift_t = Float64[]
drift_c = ComplexF64[]
diverged = false
t0 = time()
for step in 1:NSTEPS
    GeoDynamo.solver_step!(state)
    if step % DRIFT_EVERY == 0
        push!(drift_t, step * DT)
        push!(drift_c, drift_sample(state))
    end
    if step % REPORT_EVERY == 0
        E = ekin(state)
        f3 = m3_fraction(state)
        @printf("step %6d  t=%.4f  Ekin=%.6f  m3frac=%.3f  wall=%.0fs\n",
            step, step * DT, E, f3, time() - t0)
        flush(stdout)
        isfinite(E) || (println("DIVERGED"); global diverged = true; break)
    end
end

E_final = diverged ? NaN : ekin(state)
fd_signed = drift_frequency(drift_t, drift_c)
@printf("MARTI_DONE Ekin=%.6f drift=%.6f drift_signed=%.6f c33_amp=%.3e\n",
    E_final, abs(fd_signed), fd_signed, isempty(drift_c) ? NaN : abs(drift_c[end]))
