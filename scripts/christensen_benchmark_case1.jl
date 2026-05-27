# =============================================================================
# Christensen et al. (2001) numerical dynamo benchmark — Case 1
#   "A numerical dynamo benchmark", Phys. Earth Planet. Inter. 128 (2001) 25–34.
#
# STATUS: SCAFFOLD / WORK IN PROGRESS — NOT a validated, passing benchmark.
#   This file captures the full authoritative Case-1 specification + reference
#   values and provides a runnable harness, but three things must be resolved
#   before it constitutes a real validation:
#     (1) the GeoDynamo↔Christensen nondimensionalization mapping (esp. Ra and
#         the energy normalization — see "Convention mapping" below);
#     (2) the prescribed magnetic initial condition, whose construction goes
#         through the vector synthesis/analysis path currently affected by the
#         poloidal-reconstruction issue (see memory
#         proj_geodynamo_poloidal_synthesis_nonsolenoidal);
#     (3) an hours-long run to the drifting quasi-steady state (~15 viscous
#         diffusion times).
#   It is expected to EXPOSE the poloidal-reconstruction discrepancy rather than
#   pass, until that is fixed. Treat the comparison output as diagnostic.
#
# -----------------------------------------------------------------------------
# Benchmark specification (verbatim from the paper)
# -----------------------------------------------------------------------------
# Geometry: spherical shell, radius ratio r_i/r_o = 0.35. Length scale D=r_o-r_i,
#   so r_i = 7/13 ≈ 0.538462, r_o = 20/13 ≈ 1.538462 (matches GeoDynamo's domain).
# Scales: time D²/ν (VISCOUS), velocity ν/D, temperature ΔT, B by (ρμηΩ)^{1/2},
#   pressure ρνΩ. Boussinesq; gravity ∝ r.
#
# Equations (viscous-time nondimensionalization):
#   E(∂u/∂t + u·∇u − ∇²u) + 2 ẑ×u + ∇P = Ra (r/r_o) T + (1/Pm)(∇×B)×B   (1)
#   ∂B/∂t = ∇×(u×B) + (1/Pm)∇²B                                          (2)
#   ∂T/∂t + u·∇T = (1/Pr)∇²T                                             (3)
#   ∇·u = 0,  ∇·B = 0                                                    (4)
#
# Parameters:  E = ν/(ΩD²),  Ra = α g_o ΔT D /(ν Ω),  Pr = ν/κ,  Pm = ν/η.
#   CASE 1:  E = 1e-3,  Ra = 100,  Pr = 1,  Pm = 5.
# Boundary conditions: no-slip; fixed temperature (T=0 outer, T=ΔT=1 inner);
#   insulating magnetic at both boundaries; inner core insulating, co-rotating.
#
# Initial conditions (the solution is steady in a drifting frame; IC chosen for
# reproducibility). Initial velocity = 0. Initial temperature (A=0.1, x=2r−r_i−r_o):
#   T = r_o r_i / r − r_i
#       + (210 A / √(17920π)) (1 − 3x² + 3x⁴ − x⁶) sin⁴θ cos 4φ              (9)
# Case-1 initial magnetic field (r_i ≤ r ≤ r_o):
#   B_r = (5/8)(8 r_o − 6 r − 2 r_i⁴/r³) cosθ                               (10)
#   B_θ = (5/8)(9 r − 8 r_o − r_i⁴/r³) sinθ                                 (11)
#   B_φ = 5 sin(π (r − r_i)) sin 2θ                                         (12)
#   (dipolar poloidal from a φ-current uniform in r + an l=2 toroidal field;
#    max|B_r|=max|B_φ|=5.)
#
# Diagnostics:
#   E_kin = (1/2V_s) ∫ u² dV                                               (17)
#   E_mag = (1/(2 V_s E Pm)) ∫ B² dV                                       (18)
#   Local point: mid-depth r=(r_i+r_o)/2, equator θ=π/2, φ where u_r=0 and
#     ∂u_r/∂φ>0. Report T, u_φ, B_θ. Plus drift frequency ω.
#
# Reference values (converged CWG, Table 2):
#   E_kin = 30.773 ,  E_mag = 626.41 ,  T = 0.37337 ,
#   u_φ = −7.625 ,  B_θ = −4.928 ,  ω = −3.1016
#
# -----------------------------------------------------------------------------
# Convention mapping  (GeoDynamo ⟶ Christensen)  — VERIFY before trusting output
# -----------------------------------------------------------------------------
#   • Ekman: GeoDynamo defines E = ν/(2ΩD²) (Coriolis term ẑ×u, factor 2 absorbed),
#     Christensen E = ν/(ΩD²) (Coriolis 2ẑ×u). Same physics ⇒  E_geo = E_chr/2.
#         CASE 1 ⇒  E_geo = 5e-4.
#   • Time scale: GeoDynamo nondimensionalizes by the MAGNETIC diffusion time
#     (∂_t carries E/Pm), Christensen by the VISCOUS time. Buoyancy enters
#     GeoDynamo as (Pm/Pr)·Ra·T·r (radial gravity, no 1/r_o) vs Christensen
#     Ra·(r/r_o)·T. The physical modified-Ra should be identical, but the input
#     `Ra` value GeoDynamo expects may differ by 1/r_o and/or a Pm time-scale
#     factor.  >>> OPEN: derive Ra_geo from src/physics/velocity/field.jl buoyancy
#         and the magnetic-time momentum equation; do NOT assume Ra_geo=100. <<<
#   • Energy normalization: GeoDynamo's diagnostics use `vector_energy`/`field_energy`
#     (sum/integral of squares, see src/diagnostics/solver.jl) which is NOT the
#     benchmark's (1/2V_s)∫u² (E_kin) nor (1/(2 V_s E Pm))∫B² (E_mag). Convert
#     before comparing (need V_s = (4π/3)(r_o³−r_i³), and the E·Pm factor for B).

using GeoDynamo
const G = GeoDynamo

# ---- Benchmark constants -----------------------------------------------------
const R_I = 7 / 13
const R_O = 20 / 13
const V_S = (4π / 3) * (R_O^3 - R_I^3)               # fluid-shell volume
const REF = (Ekin = 30.773, Emag = 626.41, T = 0.37337, uφ = -7.625, Bθ = -4.928, ω = -3.1016)

# ---- Case-1 parameters in GeoDynamo conventions ------------------------------
# resolution: `:smoke` = tiny grid / few steps (executes in seconds, checks the
# harness runs and produces finite diagnostics); `:full` = benchmark resolution
# and a long run to quasi-steady (~15 viscous times) — hours of compute.
function case1_parameters(; mode::Symbol = :smoke)
    nr, lmax = mode === :full ? (48, 48) : (16, 8)
    return G.SolverParameters(
        geometry      = :shell,
        radius_ratio  = 0.35,
        nr            = nr,
        lmax          = lmax,  mmax = lmax,
        nlat          = max(lmax + 2, 12),
        nlon          = max(2lmax + 1, 24),
        Ek            = 5e-4,            # = E_chr/2  (Christensen E=1e-3)
        Ra            = 100.0,           # OPEN: verify GeoDynamo's Ra convention (see header)
        Pr            = 1.0,
        Pm            = 5.0,
        include_magnetic_field = true,
        include_composition    = false,
        velocity_bcs    = G.BoundaryConditions(inner = G.NoSlip(),  outer = G.NoSlip()),
        temperature_bcs = G.BoundaryConditions(inner = G.FixedTemperature(1.0),
                                               outer = G.FixedTemperature(0.0)),
        # magnetic BCs: insulating/insulating (Case 1) — set via the magnetic BC API.
        timestepper   = G.CNAB2(),
    )
end

# Christensen Case 0: non-magnetic rotating convection (same E, Ra=100, Pr=1; no
# field). NOT blocked by the magnetic IC, so it is the tractable FIRST validation
# — and it still exercises the velocity poloidal reconstruction, so it is the
# right evolution-level check for the reconstruction fix. Converged ref (Table 1):
const REF_CASE0 = (Ekin = 58.348, T = 0.42812, uφ = -10.157, ω = 0.18241)

function case0_parameters(; mode::Symbol = :smoke)
    nr, lmax = mode === :full ? (48, 48) : (16, 8)
    return G.SolverParameters(
        geometry = :shell, radius_ratio = 0.35,
        nr = nr, lmax = lmax, mmax = lmax,
        nlat = max(lmax + 2, 12), nlon = max(2lmax + 1, 24),
        Ek = 5e-4, Ra = 100.0, Pr = 1.0,          # same OPEN Ra-mapping caveat as Case 1
        include_magnetic_field = false, include_composition = false,
        velocity_bcs    = G.BoundaryConditions(inner = G.NoSlip(), outer = G.NoSlip()),
        temperature_bcs = G.BoundaryConditions(inner = G.FixedTemperature(1.0),
                                               outer = G.FixedTemperature(0.0)),
        timestepper = G.CNAB2(),
    )
end

# ---- Initial conditions ------------------------------------------------------
# Temperature: conductive background + the Eq.(9) perturbation. The perturbation
# is a physical pattern (sin⁴θ cos4φ); the faithful route is to build T(r,θ,φ) on
# the grid and analyze to spectral (shtnskit_physical_to_spectral!). The conductive
# l=0 part is installed by initialize_temperature_field! (now √4π-correct).
# TODO: add the Eq.(9) perturbation via physical construction + analysis.
#
# Magnetic: Eqs.(10–12) give B in PHYSICAL components. Installing it requires the
# vector analysis (physical→tor/pol), which currently goes through the poloidal
# reconstruction under investigation — so a faithful magnetic IC is BLOCKED until
# that is resolved. TODO once the reconstruction is fixed/validated.
function christensen_temperature_perturbation(r, θ, φ; A = 0.1)
    x = 2r - R_I - R_O
    return (210A / sqrt(17920π)) * (1 - 3x^2 + 3x^4 - x^6) * sin(θ)^4 * cos(4φ)
end

# ---- Diagnostics (benchmark normalization) -----------------------------------
# Convert GeoDynamo's raw vector energy to E_kin=(1/2V_s)∫u² and
# E_mag=(1/(2 V_s E Pm))∫B². The exact ∫(·)dV from GeoDynamo's tracker must be
# reconciled with vector_energy's normalization first (see header).
# TODO: implement once vector_energy's normalization is confirmed.

# ---- Driver ------------------------------------------------------------------
function run_case1(; mode::Symbol = :smoke, steps::Int = mode === :full ? 100_000 : 50)
    params = case1_parameters(; mode)
    state  = G.initialize_simulation(params)
    # G.run_simulation!(state; nsteps=steps)   # long for :full
    @info "Christensen Case-1 scaffold initialized" mode nr=params.nr lmax=params.lmax Ek=params.Ek Ra=params.Ra Pm=params.Pm
    @info "Reference (converged)" REF
    @warn """SCAFFOLD: parameter Ra mapping, magnetic IC, energy normalization, and a
             full run are still required (see header). Output is diagnostic, not a pass."""
    return state
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_case1(; mode = :smoke)
end
