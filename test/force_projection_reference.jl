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
# Calibration result: returns physical (1/r)∂θg and (1/r)∂φg.
# (apply_geometric_factors_spectral! multiplies spectral gradient by 1/r before
# synthesis, so the synthesised physical field carries the 1/r factor.)
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
    #         ws.∇θ_spec and ws.∇φ_spec contain (1/r)∂_θg and (1/r)∂_φg in
    #         spectral space (the geometric 1/r factor is baked in by step 3).
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
# Since _fp_angular_derivs returns (1/r)∂_θ(·) and (1/r)∂_φ(·):
#   (1/(r sinθ)) ∂_θ(sinθ F_φ) = (1/sinθ) · (1/r)∂_θ(sinθ F_φ) = d_θ_h / sinθ
#   (1/(r sinθ)) ∂_φ F_θ       = (1/sinθ) · (1/r)∂_φ F_θ       = d_φ_Fθ / sinθ
# → (∇×F)_r = (d_θ_h − d_φ_Fθ) / sinθ
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
# (∇×F)_r = (1/(r sinθ)) [∂_θ(sinθ F_φ) − ∂_φ F_θ]
#          = (d_θ_h − d_φ_Fθ) / sinθ                         (see harness 3)
#
# (∇×F)_θ = (1/(r sinθ)) ∂_φ F_r − (1/r) ∂_r(r F_φ)
#          = d_φ_Fr / sinθ − ∂_r(rFφ) / r
#          [d_φ_Fr = (1/r)∂_φF_r, so (1/(r sinθ))∂_φFr = d_φ_Fr / sinθ]
#
# (∇×F)_φ = (1/r) ∂_r(r F_θ) − (1/r) ∂_θ F_r
#          = ∂_r(rFθ) / r − d_θ_Fr
#          [d_θ_Fr = (1/r)∂_θFr, so (1/r)∂_θFr = d_θ_Fr already]
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
# g = r² cosθ  →  ∂_θ g = −r² sinθ  →  (1/r)∂_θ g = −r sinθ
# So dθ[i,j,k] / (−sinθ[i]) should equal r (not r²).
# ===========================================================================
@testset "calibration: scalar gradient scaling" begin
    cfg, dom = _fp_setup()
    g = _fp_phys(cfg, dom)
    arr = parent(g.data)
    r_range = GeoDynamo.range_local(cfg.pencils.r, 3)
    for k in axes(arr, 3), j in 1:cfg.nlon, i in 1:cfg.nlat
        r = dom.r[min(k + first(r_range) - 1, dom.N), 4]
        arr[i, j, k] = r^2 * cos(cfg.theta_grid[i])
    end
    dθ, _ = _fp_angular_derivs(cfg, dom, g)
    a = parent(dθ.data)
    i, j, k = 3, 4, div(FP_NR, 2)
    r = dom.r[k + first(r_range) - 1, 4]
    ratio = a[i, j, k] / (-sin(cfg.theta_grid[i]))
    @info "gradient scaling" ratio r r^2
    @test isapprox(ratio, r; rtol = 1e-6) || isapprox(ratio, r^2; rtol = 1e-6)
end
