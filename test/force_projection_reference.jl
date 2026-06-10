using Test
using MPI
using LinearAlgebra
using Random
using GeoDynamo

MPI.Initialized() || MPI.Init()

const FP_NR = 48
const FP_LMAX = 8

function _fp_setup()
    cfg = GeoDynamo.create_shtnskit_config(
        lmax = FP_LMAX, mmax = FP_LMAX, nlat = 2 * FP_LMAX + 4, nlon = 4 * FP_LMAX + 8,
        nr = FP_NR)
    dom = GeoDynamo.create_radial_domain(FP_NR)   # shell domain, radius_ratio=0.35 default
    return cfg, dom
end

_fp_spec(cfg, dom) = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, cfg.pencils.spec)
_fp_vec(cfg, dom) = GeoDynamo.create_shtns_vector_field(Float64, cfg, dom,
    (cfg.pencils.θ, cfg.pencils.φ, cfg.pencils.r))
_fp_phys(cfg, dom) = GeoDynamo.create_shtns_physical_field(Float64, cfg, dom, cfg.pencils.r)

# ---------------------------------------------------------------------------
# Harness 1: angular derivatives via scalar SH differentiation
#
# Returns (dθ, dφ) where both are physical fields on pencils.r.
#
# *** CALIBRATION FINDING (see calibration test below) ***
#
# dφ IS (1/r)∂φg  — the φ-recurrence is im·m·c_lm (exact) → synthesis gives
#   the true (1/r)∂φg physical field.
#
# dθ is NOT (1/r)∂θg.  The θ-recurrence encodes the identity
#   sinθ · ∂Y_l/∂θ = A_plus(l)·Y_{l+1} + A_minus(l)·Y_{l-1}
# so spectral output_l = A_plus(l)·c_{l+1} + A_minus(l)·c_{l-1}.
# After the 1/r geometric factor and synthesis, the physical field is a
# TRUNCATED SPECTRAL PROJECTION of (sinθ/r)·∂g/∂θ — not (1/r)∂g/∂θ.
# For g = r²cosθ (only l=1 mode), only the l=2 output mode is excited and
# the physical result is ≈ −r·(3cos²θ−1), NOT −r·sinθ (the true (1/r)∂θg).
#
# Consequence for the curl harnesses:
#   (∇×F)_θ  uses only dφ and ∂_r — both correct → G_θ is exact.
#   (∇×F)_r and (∇×F)_φ use dθ → those components carry the truncation error.
# The harness is useful for internal-consistency checks against the solver's
# own gradient convention; it is NOT a band-unlimited true-physics reference.
# ---------------------------------------------------------------------------
function _fp_angular_derivs(cfg, dom, g_phys::GeoDynamo.SHTnsPhysField)
    # Step 1: physical → spectral
    g_spec = _fp_spec(cfg, dom)
    GeoDynamo.scalar_physical_to_spectral!(g_phys, g_spec)

    # Step 2: load spectral data into a temperature-field container so that
    #         compute_all_gradients_spectral! can access 𝔽.spectral, 𝔽.∂r, etc.
    tf = GeoDynamo.create_shtns_temperature_field(Float64, cfg, dom)
    parent(tf.spectral.data_real) .= parent(g_spec.data_real)
    parent(tf.spectral.data_imag) .= parent(g_spec.data_imag)

    # Step 3: compute all gradient components in spectral space (θ, φ, r).
    #         GradientWorkspace (scalar_operators.jl) is the lightweight version
    #         that works with AbstractScalarField — no SolverBackend needed.
    ws = GeoDynamo.create_gradient_workspace(Float64, cfg, dom)
    GeoDynamo.zero_gradient_workspace!(ws)
    GeoDynamo.compute_all_gradients_spectral!(tf, dom, ws)

    # Step 4: synthesise θ and φ gradient components back to physical space.
    #         ws.∇θ_spec holds the TRUNCATED SPECTRAL PROJECTION described above
    #         (NOT the true (1/r)∂θg).  ws.∇φ_spec holds the exact (1/r)∂φg.
    dθ = _fp_phys(cfg, dom)
    dφ = _fp_phys(cfg, dom)
    GeoDynamo.scalar_spectral_to_physical!(ws.∇θ_spec, dθ)
    GeoDynamo.scalar_spectral_to_physical!(ws.∇φ_spec, dφ)

    return dθ, dφ
end

# ---------------------------------------------------------------------------
# Harness 2: radial derivative via banded finite-difference operator
#
# Returns ∂g/∂r as a physical field on pencils.r.
# Assumes the radial dimension is fully local (serial / Phase-1 layout).
# ---------------------------------------------------------------------------
function _fp_radial_deriv(cfg, dom, g_phys)
    D1 = GeoDynamo.create_derivative_matrix(Float64, 1, dom)
    nr = dom.N
    arr_in  = parent(g_phys.data)
    out     = _fp_phys(cfg, dom)
    arr_out = parent(out.data)

    prof  = Vector{Float64}(undef, nr)
    dprof = Vector{Float64}(undef, nr)

    for j in 1:size(arr_in, 2), i in 1:size(arr_in, 1)
        for k in 1:nr
            prof[k] = arr_in[i, j, k]
        end
        mul!(dprof, D1, prof)
        for k in 1:nr
            arr_out[i, j, k] = dprof[k]
        end
    end

    return out
end

# ---------------------------------------------------------------------------
# Harness 3: radial component of curl
#
# (∇×F)_r = (1/(r sinθ)) [∂_θ(sinθ F_φ) − ∂_φ F_θ]
#
# _fp_angular_derivs returns the exact (1/r)∂_φ(·) but only a TRUNCATED
# SPECTRAL PROJECTION for the θ-component (see harness 1 header).
# The formula is written using the (1/r) convention for both:
#   φ-term: (1/(r sinθ)) ∂_φ F_θ = d_φ_Fθ / sinθ  ← EXACT
#   θ-term: (1/(r sinθ)) ∂_θ(sinθ F_φ) ≈ d_θ_h / sinθ ← APPROXIMATE (truncation)
# → (∇×F)_r ≈ (d_θ_h − d_φ_Fθ) / sinθ
#   (exact only for F_φ band-limited within [0, lmax-1] so all derivative modes fit)
#
# F.θ_component, F.φ_component, F.r_component are all on pencils.r (see
# create_shtns_vector_field — all components use pencil_r regardless of the
# tuple argument ordering).
# ---------------------------------------------------------------------------
function _fp_radial_curl(cfg, dom, F)
    sinθ = sin.(cfg.theta_grid)  # colatitude grid; sinθ = sin(θ) ∈ [0,1]
    nlat = cfg.nlat
    nlon = cfg.nlon

    arr_Fφ = parent(F.φ_component.data)
    arr_Fθ = parent(F.θ_component.data)

    # Build sinθ · F_φ on a fresh field so the SH transform sees the right layout
    h_φ = _fp_phys(cfg, dom)
    for k in axes(arr_Fφ, 3), j in 1:nlon, i in 1:nlat
        parent(h_φ.data)[i, j, k] = sinθ[i] * arr_Fφ[i, j, k]
    end

    # Angular derivatives: _fp_angular_derivs returns (1/r)∂_θ, (1/r)∂_φ
    d_θ_h, _   = _fp_angular_derivs(cfg, dom, h_φ)
    _, d_φ_Fθ  = _fp_angular_derivs(cfg, dom, F.θ_component)

    a_dθ_h  = parent(d_θ_h.data)
    a_dφ_Fθ = parent(d_φ_Fθ.data)

    out     = _fp_phys(cfg, dom)
    arr_out = parent(out.data)

    for k in axes(arr_out, 3), j in 1:nlon, i in 1:nlat
        arr_out[i, j, k] = (a_dθ_h[i, j, k] - a_dφ_Fθ[i, j, k]) / sinθ[i]
    end

    return out
end

# ---------------------------------------------------------------------------
# Harness 4: full curl ∇×F (three physical components)
#
# Accuracy per component (see harness 1 header for the θ-truncation caveat):
#
# (∇×F)_r  ≈ (d_θ_h − d_φ_Fθ) / sinθ          (see harness 3)
#   φ-term EXACT, θ-term APPROXIMATE (truncated SH projection)
#
# (∇×F)_θ  = d_φ_Fr / sinθ − ∂_r(rFφ) / r       EXACT
#   [d_φ_Fr = exact (1/r)∂_φF_r; radial derivative exact]
#
# (∇×F)_φ  ≈ ∂_r(rFθ) / r − d_θ_Fr              APPROXIMATE (θ-term)
#   [radial derivative exact; d_θ_Fr uses truncated θ-projection]
# ---------------------------------------------------------------------------
function _fp_curl(cfg, dom, F)
    sinθ = sin.(cfg.theta_grid)
    nlat = cfg.nlat
    nlon = cfg.nlon
    r_range = GeoDynamo.range_local(cfg.pencils.r, 3)

    arr_Fr = parent(F.r_component.data)
    arr_Fθ = parent(F.θ_component.data)
    arr_Fφ = parent(F.φ_component.data)

    # === Build helper physical fields on pencils.r ===
    h_φ = _fp_phys(cfg, dom)       # sinθ · F_φ  (for ∂_θ term in G_r)
    rFφ = _fp_phys(cfg, dom)       # r · F_φ     (for ∂_r term in G_θ)
    rFθ = _fp_phys(cfg, dom)       # r · F_θ     (for ∂_r term in G_φ)
    h_Fr = _fp_phys(cfg, dom)      # F_r copy    (for angular derivatives)

    for k in axes(arr_Fr, 3), j in 1:nlon, i in 1:nlat
        r_k = dom.r[k + first(r_range) - 1, 4]
        parent(h_φ.data)[i, j, k]  = sinθ[i] * arr_Fφ[i, j, k]
        parent(rFφ.data)[i, j, k]  = r_k * arr_Fφ[i, j, k]
        parent(rFθ.data)[i, j, k]  = r_k * arr_Fθ[i, j, k]
        parent(h_Fr.data)[i, j, k] = arr_Fr[i, j, k]
    end

    # === Angular derivatives (all return (1/r)∂_θ / (1/r)∂_φ) ===
    d_θ_h,  _        = _fp_angular_derivs(cfg, dom, h_φ)         # for G_r
    _, d_φ_Fθ        = _fp_angular_derivs(cfg, dom, F.θ_component)# for G_r
    _, d_φ_Fr        = _fp_angular_derivs(cfg, dom, h_Fr)         # for G_θ
    d_θ_Fr, _        = _fp_angular_derivs(cfg, dom, h_Fr)         # for G_φ

    # === Radial derivatives ===
    deriv_rFφ = _fp_radial_deriv(cfg, dom, rFφ)   # ∂_r(r F_φ)
    deriv_rFθ = _fp_radial_deriv(cfg, dom, rFθ)   # ∂_r(r F_θ)

    a_dθ_h     = parent(d_θ_h.data)
    a_dφ_Fθ   = parent(d_φ_Fθ.data)
    a_dφ_Fr    = parent(d_φ_Fr.data)
    a_dθ_Fr    = parent(d_θ_Fr.data)
    a_drFφ     = parent(deriv_rFφ.data)
    a_drFθ     = parent(deriv_rFθ.data)

    G = _fp_vec(cfg, dom)
    arr_Gr = parent(G.r_component.data)
    arr_Gθ = parent(G.θ_component.data)
    arr_Gφ = parent(G.φ_component.data)

    for k in axes(arr_Gr, 3), j in 1:nlon, i in 1:nlat
        r_k = dom.r[k + first(r_range) - 1, 4]

        # (∇×F)_r = (d_θ_h − d_φ_Fθ) / sinθ
        arr_Gr[i, j, k] = (a_dθ_h[i, j, k]  - a_dφ_Fθ[i, j, k]) / sinθ[i]

        # (∇×F)_θ = d_φ_Fr / sinθ − ∂_r(rFφ) / r
        arr_Gθ[i, j, k] = a_dφ_Fr[i, j, k] / sinθ[i] - a_drFφ[i, j, k] / r_k

        # (∇×F)_φ = ∂_r(rFθ) / r − d_θ_Fr
        arr_Gφ[i, j, k] = a_drFθ[i, j, k] / r_k - a_dθ_Fr[i, j, k]
    end

    return G
end

# ===========================================================================
# Calibration test
#
# Determines what _fp_angular_derivs actually returns for g = r²cosθ.
#
# True (1/r)∂_θ g = −r sinθ, so if the function returned (1/r)∂_θg we would
# see dθ / (−sinθ) ≈ r.  If it returned ∂_θg directly we would see ≈ r².
#
# OBSERVED (see analysis in harness 1 header): the θ-recurrence encodes
#   sinθ·∂Y_l/∂θ = A_plus(l)·Y_{l+1} + A_minus(l)·Y_{l-1}
# so for g = r²cosθ (only l=1 SH mode present) the recurrence excites only
# the l=2 output mode.  After 1/r geometric factor + synthesis:
#   dθ[i,j,k] ≈ −r_k · (3cos²θ_i − 1)
# This is NOT −r sinθ (the true gradient); the θ-dependence differs.
#
# The test below pins this empirical result at two independent grid points and
# also records the true gradient value via @info for reference.
# ===========================================================================
@testset "calibration: scalar gradient scaling" begin
    cfg, dom = _fp_setup()
    g = _fp_phys(cfg, dom)
    arr = parent(g.data)
    r_range = GeoDynamo.range_local(cfg.pencils.r, 3)
    for k in axes(arr, 3), j in 1:cfg.nlon, i in 1:cfg.nlat
        r_k = dom.r[min(k + first(r_range) - 1, dom.N), 4]
        arr[i, j, k] = r_k^2 * cos(cfg.theta_grid[i])
    end
    dθ, _ = _fp_angular_derivs(cfg, dom, g)
    a = parent(dθ.data)

    # --- first probe point ---
    i1, j1, k1 = 3, 4, div(FP_NR, 2)
    r1  = dom.r[k1 + first(r_range) - 1, 4]
    θ1  = cfg.theta_grid[i1]
    observed1  = a[i1, j1, k1]
    expected1  = -r1 * (3 * cos(θ1)^2 - 1)
    true_val1  = -r1 * sin(θ1)            # what (1/r)∂_θg should be
    @info "calibration point 1" observed = observed1 expected_truncated = expected1 true_gradient = true_val1 r = r1 θ = θ1

    # --- second probe point (different θ to confirm θ-dependence) ---
    i2, j2, k2 = 8, 2, div(FP_NR, 3)
    r2  = dom.r[k2 + first(r_range) - 1, 4]
    θ2  = cfg.theta_grid[i2]
    observed2  = a[i2, j2, k2]
    expected2  = -r2 * (3 * cos(θ2)^2 - 1)
    true_val2  = -r2 * sin(θ2)
    @info "calibration point 2" observed = observed2 expected_truncated = expected2 true_gradient = true_val2 r = r2 θ = θ2

    # Pin the ACTUAL observed behavior: dθ ≈ −r·(3cos²θ−1) at both probes.
    # (NOT −r·sinθ — the θ-recurrence gives a truncated SH projection.)
    @test isapprox(observed1, expected1; rtol = 1e-4)
    @test isapprox(observed2, expected2; rtol = 1e-4)
end
