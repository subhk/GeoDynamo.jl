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

@testset "force_physical_to_qst! recovers Q, S, T" begin
    cfg, dom = _fp_setup()
    Random.seed!(7)
    Qin = _fp_spec(cfg, dom); Sin = _fp_spec(cfg, dom); Tin = _fp_spec(cfg, dom)
    for spec in (Qin, Sin, Tin)
        sr = parent(spec.data_real); si = parent(spec.data_imag)
        for lm in 1:cfg.nlm
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            l = cfg.l_values[lm]; m = cfg.m_values[lm]
            (1 <= l <= FP_LMAX - 2) || continue
            for r_idx in 1:dom.N
                x = (dom.r[r_idx, 4] - dom.r[1, 4]) / (dom.r[dom.N, 4] - dom.r[1, 4])
                v = sinpi(x) * randn() * 1e-2
                GeoDynamo.set_local_spectral_value!(sr, slot, r_idx, v)
                m > 0 && GeoDynamo.set_local_spectral_value!(si, slot, r_idx, 0.7v)
            end
        end
    end
    # synthesize physical F: tangential from (S,T) via the existing sphtor
    # synthesis (argument order: toroidal, poloidal-which-is-S, out-vector);
    # radial from Q via the scalar synthesis
    F = _fp_vec(cfg, dom)
    GeoDynamo.vector_spectral_to_physical!(Tin, Sin, F)
    Fr_phys = _fp_phys(cfg, dom)
    GeoDynamo.scalar_spectral_to_physical!(Qin, Fr_phys)
    copyto!(parent(F.r_component.data), parent(Fr_phys.data))

    Q = _fp_spec(cfg, dom); S = _fp_spec(cfg, dom); T_ = _fp_spec(cfg, dom)
    GeoDynamo.force_physical_to_qst!(F, Q, S, T_)

    for (out, ref, name) in ((Q, Qin, "Q"), (S, Sin, "S"), (T_, Tin, "T"))
        a = vcat(vec(parent(out.data_real)), vec(parent(out.data_imag)))
        b = vcat(vec(parent(ref.data_real)), vec(parent(ref.data_imag)))
        @test isapprox(a, b; rtol = 1e-8, atol = 1e-12)
    end
end

# ---------------------------------------------------------------------------
# Helpers for curl-projection testsets
# ---------------------------------------------------------------------------

# Find the 1-based lm index in cfg for mode (l_target, m_target)
function _fp_lm_index(cfg, l_target, m_target)
    for lm in 1:cfg.nlm
        if cfg.l_values[lm] == l_target && cfg.m_values[lm] == m_target
            return lm
        end
    end
    error("Mode (l=$l_target, m=$m_target) not found in config")
end

# Project a physical field to a flat real+imag spectral vector
function _fp_project(cfg, dom, phys)
    spec = _fp_spec(cfg, dom)
    GeoDynamo.scalar_physical_to_spectral!(phys, spec)
    return vcat(vec(parent(spec.data_real)), vec(parent(spec.data_imag)))
end

# Spectral L2 norm summing all modes EXCEPT lm_seed
function _fp_off_seed_norm(spec_field, cfg, dom, lm_seed)
    s = 0.0
    for lm2 in 1:cfg.nlm
        lm2 == lm_seed && continue
        sl = GeoDynamo.local_spectral_storage_slot(cfg, lm2)
        sl === nothing && continue
        for r_idx in 1:dom.N
            s += GeoDynamo.local_spectral_value(parent(spec_field.data_real), sl, r_idx)^2
            s += GeoDynamo.local_spectral_value(parent(spec_field.data_imag), sl, r_idx)^2
        end
    end
    return sqrt(s)
end

# Build a unit spectral delta at (lm_seed, all r) — used to synthesize Y_lm
function _fp_delta_spec(cfg, dom, lm_seed)
    delta = _fp_spec(cfg, dom)
    slot  = GeoDynamo.local_spectral_storage_slot(cfg, lm_seed)
    for r_idx in 1:dom.N
        GeoDynamo.set_local_spectral_value!(parent(delta.data_real), slot, r_idx, 1.0)
    end
    return delta
end

# ===========================================================================
# Test: curl projections — pure radial (buoyancy) force
#
# Family 1: F = q(r)·Y_lm·r̂  ⟹  R_tor ≡ 0,  R_pol[lm](r) = λ·q(r)/r²
#
# Seeds:
#   (l=1,m=0), q(r)=r  → R_pol = 2/r  (constant × 1/r)
#   (l=2,m=1), q(r)=r² → R_pol = 6    (constant)
# ===========================================================================
@testset "curl projections: pure radial (buoyancy) force" begin
    cfg, dom = _fp_setup()
    r_range  = GeoDynamo.range_local(cfg.pencils.r, 3)

    for (l_s, m_s, q_fn) in [(1, 0, r -> r), (2, 1, r -> r^2)]
        lm_seed = _fp_lm_index(cfg, l_s, m_s)
        slot    = GeoDynamo.local_spectral_storage_slot(cfg, lm_seed)
        λ       = Float64(l_s * (l_s + 1))

        # Build Ygrid = Y_lm on the physical grid (same at every r)
        delta = _fp_delta_spec(cfg, dom, lm_seed)
        Ygrid = _fp_phys(cfg, dom)
        GeoDynamo.scalar_spectral_to_physical!(delta, Ygrid)

        # Build force F with only radial component = q(r_k)*Y_lm
        F = _fp_vec(cfg, dom)
        for k in axes(parent(F.r_component.data), 3)
            r_k = dom.r[k + first(r_range) - 1, 4]
            parent(F.r_component.data)[:, :, k] .= q_fn(r_k) .*
                                                   parent(Ygrid.data)[:, :, k]
        end
        # θ and φ components stay zero

        # QST analysis
        Q  = _fp_spec(cfg, dom); S  = _fp_spec(cfg, dom); T_ = _fp_spec(cfg, dom)
        GeoDynamo.force_physical_to_qst!(F, Q, S, T_)

        # Curl projections
        Rtor = _fp_spec(cfg, dom); Rpol = _fp_spec(cfg, dom)
        GeoDynamo.force_curl_projections!(Rtor, Rpol, Q, S, T_, dom)

        # R_tor must be (nearly) zero
        all_rtor = vcat(vec(parent(Rtor.data_real)), vec(parent(Rtor.data_imag)))
        @test norm(all_rtor) < 1e-8

        # R_pol at the seed mode must match λ·q(r)/r²
        got      = [GeoDynamo.local_spectral_value(parent(Rpol.data_real), slot, r_idx)
                    for r_idx in 1:dom.N]
        expected = [λ * q_fn(dom.r[r_idx, 4]) / dom.r[r_idx, 4]^2 for r_idx in 1:dom.N]
        @test isapprox(got[2:dom.N-1], expected[2:dom.N-1]; rtol = 1e-6)

        # Off-seed R_pol norm must be tiny
        @test _fp_off_seed_norm(Rpol, cfg, dom, lm_seed) < 1e-8

        # Projection is non-trivial (buoyancy does project)
        @test norm(got) > 1e-8
    end
end

# ===========================================================================
# Test: curl projections — spheroidal force
#
# Family 2: F_tan = s(r)·∇₁Y_lm  ⟹  R_tor ≡ 0,
#           R_pol = −(λ/r²)·d(r·s)/dr
#
# Seed (l=1,m=0), s(r)=r²:
#   d(r·r²)/dr = d(r³)/dr = 3r²
#   R_pol = −λ·3r²/r² = −3λ  (constant)
# ===========================================================================
@testset "curl projections: spheroidal force" begin
    cfg, dom = _fp_setup()
    r_range  = GeoDynamo.range_local(cfg.pencils.r, 3)

    lm_seed = _fp_lm_index(cfg, 1, 0)
    slot    = GeoDynamo.local_spectral_storage_slot(cfg, lm_seed)
    λ       = 2.0  # l=1 → λ = 1·2 = 2

    # Build dY via sphtor synthesis (T=0, S=delta) → θ and φ angular derivs
    delta     = _fp_delta_spec(cfg, dom, lm_seed)
    zero_spec = _fp_spec(cfg, dom)
    dY        = _fp_vec(cfg, dom)
    GeoDynamo.vector_spectral_to_physical!(zero_spec, delta, dY)
    # dY.θ_component = ∂θY;  dY.φ_component = (1/sinθ)∂φY

    # Build force F_tan = s(r)·∇₁Y  with s(r)=r²
    F = _fp_vec(cfg, dom)
    for k in axes(parent(F.θ_component.data), 3)
        r_k  = dom.r[k + first(r_range) - 1, 4]
        s_rk = r_k^2
        parent(F.θ_component.data)[:, :, k] .= s_rk .* parent(dY.θ_component.data)[:, :, k]
        parent(F.φ_component.data)[:, :, k] .= s_rk .* parent(dY.φ_component.data)[:, :, k]
    end
    # F.r_component stays zero

    # QST analysis
    Q  = _fp_spec(cfg, dom); S  = _fp_spec(cfg, dom); T_ = _fp_spec(cfg, dom)
    GeoDynamo.force_physical_to_qst!(F, Q, S, T_)

    # Curl projections
    Rtor = _fp_spec(cfg, dom); Rpol = _fp_spec(cfg, dom)
    GeoDynamo.force_curl_projections!(Rtor, Rpol, Q, S, T_, dom)

    # R_tor must be (nearly) zero
    all_rtor = vcat(vec(parent(Rtor.data_real)), vec(parent(Rtor.data_imag)))
    @test norm(all_rtor) < 1e-8

    # R_pol at seed = −3λ (constant, analytically exact)
    got      = [GeoDynamo.local_spectral_value(parent(Rpol.data_real), slot, r_idx)
                for r_idx in 1:dom.N]
    expected = fill(-3.0 * λ, dom.N)
    @test isapprox(got[2:dom.N-1], expected[2:dom.N-1]; rtol = 1e-6)

    # Off-seed R_pol norm must be tiny
    @test _fp_off_seed_norm(Rpol, cfg, dom, lm_seed) < 1e-8
end

# ===========================================================================
# Test: curl projections — toroidal force
#
# Family 3: F_tan = r̂×∇₁(t(r)Y_lm)  ⟹  R_pol ≡ 0,
#           R_tor[lm](r) = −λ·t(r)/r
#
# Seed (l=2,m=1), t(r)=r:
#   R_tor = −λ·r/r = −λ  (constant, λ=6)
# ===========================================================================
@testset "curl projections: toroidal force" begin
    cfg, dom = _fp_setup()
    r_range  = GeoDynamo.range_local(cfg.pencils.r, 3)

    lm_seed = _fp_lm_index(cfg, 2, 1)
    slot    = GeoDynamo.local_spectral_storage_slot(cfg, lm_seed)
    λ       = 6.0  # l=2 → λ = 2·3 = 6

    # Build dY via sphtor synthesis (T=delta, S=0) → toroidal components
    delta     = _fp_delta_spec(cfg, dom, lm_seed)
    zero_spec = _fp_spec(cfg, dom)
    dY        = _fp_vec(cfg, dom)
    GeoDynamo.vector_spectral_to_physical!(delta, zero_spec, dY)
    # dY.θ_component = -(1/sinθ)∂φY; dY.φ_component = ∂θY  (toroidal)

    # Build force with t(r)=r (toroidal structure)
    F = _fp_vec(cfg, dom)
    for k in axes(parent(F.θ_component.data), 3)
        r_k  = dom.r[k + first(r_range) - 1, 4]
        t_rk = r_k  # t(r) = r
        parent(F.θ_component.data)[:, :, k] .= t_rk .* parent(dY.θ_component.data)[:, :, k]
        parent(F.φ_component.data)[:, :, k] .= t_rk .* parent(dY.φ_component.data)[:, :, k]
    end

    # QST analysis
    Q  = _fp_spec(cfg, dom); S  = _fp_spec(cfg, dom); T_ = _fp_spec(cfg, dom)
    GeoDynamo.force_physical_to_qst!(F, Q, S, T_)

    # Curl projections
    Rtor = _fp_spec(cfg, dom); Rpol = _fp_spec(cfg, dom)
    GeoDynamo.force_curl_projections!(Rtor, Rpol, Q, S, T_, dom)

    # R_pol must be (nearly) zero
    all_rpol = vcat(vec(parent(Rpol.data_real)), vec(parent(Rpol.data_imag)))
    @test norm(all_rpol) < 1e-8

    # R_tor at seed = −λ (constant, analytically exact)
    got      = [GeoDynamo.local_spectral_value(parent(Rtor.data_real), slot, r_idx)
                for r_idx in 1:dom.N]
    expected = fill(-λ, dom.N)
    @test isapprox(got[2:dom.N-1], expected[2:dom.N-1]; rtol = 1e-6)

    # Off-seed R_tor norm must be tiny
    @test _fp_off_seed_norm(Rtor, cfg, dom, lm_seed) < 1e-8
end

# ===========================================================================
# Test: curl projections — generic force vs direct spectral reference
#
# Strategy: seed random (Qin, Sin, Tin), build physical F, recover (Q, S, T_)
# via force_physical_to_qst!, apply force_curl_projections!, then compare
# against the same projection applied directly to the original spectral inputs
# (Qin, Sin, Tin).
#
# This tests both (a) QST roundtrip accuracy and (b) correct formula
# application for all modes simultaneously, without the ~2 % aliasing error
# that _fp_radial_curl accumulates from the 1/sinθ coupling in the F_θ path.
# Accuracy is near machine-epsilon (QST roundtrip ~1e-14, formula ~1e-15).
# ===========================================================================
@testset "curl projections: generic force vs spectral reference" begin
    cfg, dom = _fp_setup()

    Random.seed!(13)
    Qin = _fp_spec(cfg, dom); Sin = _fp_spec(cfg, dom); Tin = _fp_spec(cfg, dom)
    for spec in (Qin, Sin, Tin)
        sr = parent(spec.data_real); si = parent(spec.data_imag)
        for lm in 1:cfg.nlm
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            l = cfg.l_values[lm]; m = cfg.m_values[lm]
            (1 <= l <= FP_LMAX - 2) || continue
            for r_idx in 1:dom.N
                x = (dom.r[r_idx, 4] - dom.r[1, 4]) / (dom.r[dom.N, 4] - dom.r[1, 4])
                v = sinpi(x) * randn() * 1e-2
                GeoDynamo.set_local_spectral_value!(sr, slot, r_idx, v)
                m > 0 && GeoDynamo.set_local_spectral_value!(si, slot, r_idx, 0.7v)
            end
        end
    end

    # Synthesize physical F from (Tin, Sin, Qin)
    F = _fp_vec(cfg, dom)
    GeoDynamo.vector_spectral_to_physical!(Tin, Sin, F)
    Fr_phys = _fp_phys(cfg, dom)
    GeoDynamo.scalar_spectral_to_physical!(Qin, Fr_phys)
    copyto!(parent(F.r_component.data), parent(Fr_phys.data))

    # QST analysis (round-trip: should recover Qin/Sin/Tin to ~1e-14)
    Q  = _fp_spec(cfg, dom); S  = _fp_spec(cfg, dom); T_ = _fp_spec(cfg, dom)
    GeoDynamo.force_physical_to_qst!(F, Q, S, T_)

    # Curl projections on recovered scalars
    Rtor = _fp_spec(cfg, dom); Rpol = _fp_spec(cfg, dom)
    GeoDynamo.force_curl_projections!(Rtor, Rpol, Q, S, T_, dom)

    # Spectral reference: same projection applied directly to the original inputs.
    # Because QST roundtrip is ~1e-14 exact, Rtor ≈ Rtor_ref and Rpol ≈ Rpol_ref
    # to near machine precision (rtol ~ 1e-10).
    Rtor_ref = _fp_spec(cfg, dom); Rpol_ref = _fp_spec(cfg, dom)
    GeoDynamo.force_curl_projections!(Rtor_ref, Rpol_ref, Qin, Sin, Tin, dom)

    @test isapprox(parent(Rtor.data_real), parent(Rtor_ref.data_real); rtol = 1e-10)
    @test isapprox(parent(Rtor.data_imag), parent(Rtor_ref.data_imag); rtol = 1e-10)
    @test isapprox(parent(Rpol.data_real), parent(Rpol_ref.data_real); rtol = 1e-10)
    @test isapprox(parent(Rpol.data_imag), parent(Rpol_ref.data_imag); rtol = 1e-10)
end

# ===========================================================================
# Legacy compute_velocity_nonlinear! must pass `domain` to the vector analysis,
# so the Stage-2 solenoidal recovery folds the RADIAL force component (buoyancy /
# Lorentz) into the poloidal nonlinear. Without it the radial force is silently
# dropped (the poloidal reduces to the raw spheroidal scalar of the tangential
# part) — the "convection-can't-start-from-rest" bug.
# ===========================================================================
@testset "legacy velocity nonlinear: radial force recovered only with domain" begin
    cfg, dom = _fp_setup()
    r_range = GeoDynamo.range_local(cfg.pencils.r, 3)

    # Purely radial force F = r·Y_{1,0} r̂ (tangential components zero).
    lm_seed = _fp_lm_index(cfg, 1, 0)
    delta = _fp_delta_spec(cfg, dom, lm_seed)
    Ygrid = _fp_phys(cfg, dom)
    GeoDynamo.scalar_spectral_to_physical!(delta, Ygrid)
    F = _fp_vec(cfg, dom)
    for k in axes(parent(F.r_component.data), 3)
        r_k = dom.r[k + first(r_range) - 1, 4]
        parent(F.r_component.data)[:, :, k] .= r_k .* parent(Ygrid.data)[:, :, k]
    end

    # WITHOUT domain: the radial force is discarded ⇒ poloidal ≈ 0.
    tor0 = _fp_spec(cfg, dom); pol0 = _fp_spec(cfg, dom)
    GeoDynamo.shtnskit_vector_analysis!(F, tor0, pol0)
    pol0_norm = norm(vcat(vec(parent(pol0.data_real)), vec(parent(pol0.data_imag))))

    # WITH domain: Stage-2 recovery folds the radial force into the poloidal.
    tor1 = _fp_spec(cfg, dom); pol1 = _fp_spec(cfg, dom)
    GeoDynamo.shtnskit_vector_analysis!(F, tor1, pol1; domain = dom)
    pol1_norm = norm(vcat(vec(parent(pol1.data_real)), vec(parent(pol1.data_imag))))

    @test pol0_norm < 1e-8     # radial force lost without domain
    @test pol1_norm > 1e-6     # radial force recovered with domain

    # The fix: compute_velocity_nonlinear! must pass the domain to the analysis.
    src = read(joinpath(@__DIR__, "..", "src", "physics", "velocity", "field.jl"), String)
    @test occursin("𝒰.nl_poloidal;\n        domain = outer_core_domain)", src)
end
