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
# Harness 1: exact angular derivatives via sphtor spectral synthesis
#
# Returns (dθ_bare, dφ_oversin) — two physical fields on pencils.r:
#   dθ_bare    = ∂θ g   (bare colatitude derivative, unit-sphere, no 1/r)
#   dφ_oversin = (1/sinθ)·∂φ g
#
# Mechanism: set toroidal = 0, poloidal/spheroidal = spectral(g), call
# vector_spectral_to_physical! (sphtor synthesis via SHTnsKit
# dist_synthesis_sphtor!).  Exact for band-limited fields.
#
# The old implementation routed through compute_all_gradients_spectral!, which
# returned a truncated SH projection of (sinθ/r)·∂θg rather than ∂θg, making
# the (∇×F)_r and (∇×F)_φ components only approximate.  Replaced.
# ---------------------------------------------------------------------------
function _fp_angular_derivs(cfg, dom, g_phys::GeoDynamo.SHTnsPhysField)
    # Step 1: physical → spectral
    g_spec = _fp_spec(cfg, dom)
    GeoDynamo.scalar_physical_to_spectral!(g_phys, g_spec)

    # Step 2: zero toroidal, g_spec as spheroidal/poloidal → sphtor synthesis.
    #         With domain=nothing the v_r component is zero-filled (not needed).
    zero_spec = _fp_spec(cfg, dom)   # all-zero toroidal
    V = _fp_vec(cfg, dom)
    GeoDynamo.vector_spectral_to_physical!(zero_spec, g_spec, V; domain = nothing)

    # V.θ_component = ∂θg  (bare)
    # V.φ_component = (1/sinθ)∂φg
    return V.θ_component, V.φ_component
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
# Harness 3: radial component of curl  (EXACT for band-limited fields)
#
# (∇×F)_r = (1/(r sinθ)) [∂_θ(sinθ F_φ) − ∂_φ F_θ]
#          = ( dθ_bare(sinθ·F_φ) / sinθ  −  dφ_oversin(F_θ) ) / r
#
# dθ_bare    = bare ∂_θ from sphtor synthesis (harness 1)
# dφ_oversin = (1/sinθ)∂_φ from sphtor synthesis (harness 1)
#
# F.θ_component, F.φ_component, F.r_component are all on pencils.r (see
# create_shtns_vector_field — all components use pencil_r regardless of the
# tuple argument ordering).
# ---------------------------------------------------------------------------
function _fp_radial_curl(cfg, dom, F)
    sinθ   = sin.(cfg.theta_grid)  # colatitude grid; sinθ = sin(θ) ∈ [0,1]
    nlat   = cfg.nlat
    nlon   = cfg.nlon
    r_range = GeoDynamo.range_local(cfg.pencils.r, 3)

    arr_Fφ = parent(F.φ_component.data)
    arr_Fθ = parent(F.θ_component.data)

    # Build sinθ · F_φ on a fresh field so the SH transform sees the right layout
    h_φ = _fp_phys(cfg, dom)
    for k in axes(arr_Fφ, 3), j in 1:nlon, i in 1:nlat
        parent(h_φ.data)[i, j, k] = sinθ[i] * arr_Fφ[i, j, k]
    end

    # Exact angular derivatives via sphtor synthesis:
    #   first return  = dθ_bare(·)    = ∂θ(·)
    #   second return = dφ_oversin(·) = (1/sinθ)∂φ(·)
    dθ_h, _   = _fp_angular_derivs(cfg, dom, h_φ)           # ∂θ(sinθ·F_φ)
    _, dφ_Fθ  = _fp_angular_derivs(cfg, dom, F.θ_component)  # (1/sinθ)∂φ(F_θ)

    a_dθ_h  = parent(dθ_h.data)
    a_dφ_Fθ = parent(dφ_Fθ.data)

    out     = _fp_phys(cfg, dom)
    arr_out = parent(out.data)

    for k in axes(arr_out, 3), j in 1:nlon, i in 1:nlat
        r_k = dom.r[k + first(r_range) - 1, 4]
        # (∇×F)_r = ( ∂θ(sinθ·F_φ)/sinθ  −  (1/sinθ)∂φ(F_θ) ) / r
        arr_out[i, j, k] = (a_dθ_h[i, j, k] / sinθ[i] - a_dφ_Fθ[i, j, k]) / r_k
    end

    return out
end

# ---------------------------------------------------------------------------
# Harness 4: full curl ∇×F (three physical components)  — EXACT for band-limited fields
#
# With exact angular derivatives from sphtor synthesis all three components are
# spectrally accurate (radial derivative = banded D1, spectral-grade too):
#
# (∇×F)_r = ( dθ_bare(sinθ·F_φ)/sinθ  −  dφ_oversin(F_θ) ) / r
# (∇×F)_θ = ( dφ_oversin(F_r)  −  ∂_r(r·F_φ) ) / r
# (∇×F)_φ = ( ∂_r(r·F_θ)  −  dθ_bare(F_r) ) / r
#
# dθ_bare    = bare ∂_θ from sphtor synthesis (harness 1)
# dφ_oversin = (1/sinθ)∂_φ from sphtor synthesis (harness 1)
# ---------------------------------------------------------------------------
function _fp_curl(cfg, dom, F)
    sinθ    = sin.(cfg.theta_grid)
    nlat    = cfg.nlat
    nlon    = cfg.nlon
    r_range = GeoDynamo.range_local(cfg.pencils.r, 3)

    arr_Fr = parent(F.r_component.data)
    arr_Fθ = parent(F.θ_component.data)
    arr_Fφ = parent(F.φ_component.data)

    # === Build helper physical fields on pencils.r ===
    h_φ  = _fp_phys(cfg, dom)       # sinθ · F_φ  (for ∂_θ term in G_r)
    rFφ  = _fp_phys(cfg, dom)       # r · F_φ     (for ∂_r term in G_θ)
    rFθ  = _fp_phys(cfg, dom)       # r · F_θ     (for ∂_r term in G_φ)
    h_Fr = _fp_phys(cfg, dom)       # F_r copy    (for angular derivatives)

    for k in axes(arr_Fr, 3), j in 1:nlon, i in 1:nlat
        r_k = dom.r[k + first(r_range) - 1, 4]
        parent(h_φ.data)[i, j, k]  = sinθ[i] * arr_Fφ[i, j, k]
        parent(rFφ.data)[i, j, k]  = r_k * arr_Fφ[i, j, k]
        parent(rFθ.data)[i, j, k]  = r_k * arr_Fθ[i, j, k]
        parent(h_Fr.data)[i, j, k] = arr_Fr[i, j, k]
    end

    # === Exact angular derivatives via sphtor synthesis ===
    #   first return  = dθ_bare(·)    = ∂θ(·)
    #   second return = dφ_oversin(·) = (1/sinθ)∂φ(·)
    dθ_h, _        = _fp_angular_derivs(cfg, dom, h_φ)          # ∂θ(sinθ·F_φ) for G_r
    _, dφ_Fθ       = _fp_angular_derivs(cfg, dom, F.θ_component) # (1/sinθ)∂φ(F_θ) for G_r
    dθ_Fr, dφ_Fr   = _fp_angular_derivs(cfg, dom, h_Fr)          # ∂θ(F_r) and (1/sinθ)∂φ(F_r)

    # === Radial derivatives ===
    deriv_rFφ = _fp_radial_deriv(cfg, dom, rFφ)   # ∂_r(r F_φ)
    deriv_rFθ = _fp_radial_deriv(cfg, dom, rFθ)   # ∂_r(r F_θ)

    a_dθ_h   = parent(dθ_h.data)
    a_dφ_Fθ  = parent(dφ_Fθ.data)
    a_dθ_Fr  = parent(dθ_Fr.data)
    a_dφ_Fr  = parent(dφ_Fr.data)
    a_drFφ   = parent(deriv_rFφ.data)
    a_drFθ   = parent(deriv_rFθ.data)

    G = _fp_vec(cfg, dom)
    arr_Gr = parent(G.r_component.data)
    arr_Gθ = parent(G.θ_component.data)
    arr_Gφ = parent(G.φ_component.data)

    for k in axes(arr_Gr, 3), j in 1:nlon, i in 1:nlat
        r_k = dom.r[k + first(r_range) - 1, 4]

        # (∇×F)_r = ( ∂θ(sinθ·F_φ)/sinθ − (1/sinθ)∂φ(F_θ) ) / r
        arr_Gr[i, j, k] = (a_dθ_h[i, j, k] / sinθ[i] - a_dφ_Fθ[i, j, k]) / r_k

        # (∇×F)_θ = ( (1/sinθ)∂φ(F_r) − ∂_r(r·F_φ) ) / r
        arr_Gθ[i, j, k] = (a_dφ_Fr[i, j, k] - a_drFφ[i, j, k]) / r_k

        # (∇×F)_φ = ( ∂_r(r·F_θ) − ∂θ(F_r) ) / r
        arr_Gφ[i, j, k] = (a_drFθ[i, j, k] - a_dθ_Fr[i, j, k]) / r_k
    end

    return G
end

# ===========================================================================
# Calibration test — exact angular derivatives
#
# Verifies that _fp_angular_derivs returns the TRUE bare derivatives:
#
#   g  = r²·cosθ           → dθ_bare = ∂θ(r²cosθ) = −r²·sinθ
#   g2 = r·sinθ·cosφ       → dφ_oversin = (1/sinθ)∂φ(r·sinθ·cosφ) = −r·sinφ
#
# Both should match to ~rtol 1e-6 (only SH transform roundtrip error).
# ===========================================================================
@testset "calibration: exact angular derivatives" begin
    cfg, dom = _fp_setup()
    r_range  = GeoDynamo.range_local(cfg.pencils.r, 3)

    # -----------------------------------------------------------------------
    # Test A: g = r²·cosθ  →  dθ_bare = −r²·sinθ
    # -----------------------------------------------------------------------
    g = _fp_phys(cfg, dom)
    arr = parent(g.data)
    for k in axes(arr, 3), j in 1:cfg.nlon, i in 1:cfg.nlat
        r_k = dom.r[min(k + first(r_range) - 1, dom.N), 4]
        arr[i, j, k] = r_k^2 * cos(cfg.theta_grid[i])
    end
    dθ, _ = _fp_angular_derivs(cfg, dom, g)
    a_dθ  = parent(dθ.data)

    i1, j1, k1 = 3, 4, div(FP_NR, 2)
    r1 = dom.r[k1 + first(r_range) - 1, 4]
    θ1 = cfg.theta_grid[i1]
    obs1 = a_dθ[i1, j1, k1]
    exp1 = -r1^2 * sin(θ1)
    @info "dθ_bare probe 1" observed = obs1 expected = exp1 r = r1 θ = θ1

    i2, j2, k2 = 8, 2, div(FP_NR, 3)
    r2 = dom.r[k2 + first(r_range) - 1, 4]
    θ2 = cfg.theta_grid[i2]
    obs2 = a_dθ[i2, j2, k2]
    exp2 = -r2^2 * sin(θ2)
    @info "dθ_bare probe 2" observed = obs2 expected = exp2 r = r2 θ = θ2

    @test isapprox(obs1, exp1; rtol = 1e-6)
    @test isapprox(obs2, exp2; rtol = 1e-6)

    # -----------------------------------------------------------------------
    # Test B: g2 = r·sinθ·cosφ  →  dφ_oversin = −r·sinφ
    # -----------------------------------------------------------------------
    g2   = _fp_phys(cfg, dom)
    arr2 = parent(g2.data)
    for k in axes(arr2, 3), j in 1:cfg.nlon, i in 1:cfg.nlat
        r_k = dom.r[min(k + first(r_range) - 1, dom.N), 4]
        arr2[i, j, k] = r_k * sin(cfg.theta_grid[i]) * cos(cfg.phi_grid[j])
    end
    _, dφ = _fp_angular_derivs(cfg, dom, g2)
    a_dφ  = parent(dφ.data)

    i3, j3, k3 = 4, 5, div(FP_NR, 2)
    r3  = dom.r[k3 + first(r_range) - 1, 4]
    φ3  = cfg.phi_grid[j3]
    obs3 = a_dφ[i3, j3, k3]
    exp3 = -r3 * sin(φ3)
    @info "dφ_oversin probe 1" observed = obs3 expected = exp3 r = r3 φ = φ3

    i4, j4, k4 = 7, 10, div(FP_NR, 3)
    r4  = dom.r[k4 + first(r_range) - 1, 4]
    φ4  = cfg.phi_grid[j4]
    obs4 = a_dφ[i4, j4, k4]
    exp4 = -r4 * sin(φ4)
    @info "dφ_oversin probe 2" observed = obs4 expected = exp4 r = r4 φ = φ4

    @test isapprox(obs3, exp3; rtol = 1e-6)
    @test isapprox(obs4, exp4; rtol = 1e-6)
end
