# =============================================================================
# GPU ERK2 — device-state pack for the staged exponential RK2 integrator.
# Packs the CPU ERK2 machinery into dense batched arrays:
#   - per-field, per-degree DENSE propagators E_half/E_full/φ1_half/φ1_full/
#     φ2_full as (nr, nr, nl) stacks (slot l+1), from the memoized CPU caches
#     (built on demand via the same getters integrate_solver_erk2_step! uses);
#   - per-field boundary packs mirroring SolverERK2BoundarySide/-Spec
#     (endpoint kind, stencil row, l-correction scalars, dense (nl, nm)
#     per-mode endpoint values for real and imag);
#   - the velocity-poloidal W-split recovery constants: the φ1-column Green
#     responses h₁/h₂ and 2×2 influence matrices for BOTH the stage (dt/2,
#     φ1_half) and the finalize (dt, φ1_full), precomputed once (they depend
#     only on dt and l — the CPU path re-solves them every step).
#
# Scope (mirrors the CNAB2 device builder): shell geometry, insulating
# magnetic, boundary VALUES constant in time (they are baked at pack time).
# =============================================================================

# One boundary side → NamedTuple of plain arrays/scalars.
# kind: 0 = zero-endpoint (bc_spec === nothing), 1 = Dirichlet, 2 = stencil row.
function _pack_erk2_bc_side(side::SolverERK2DirichletSide{T}, mode_vals, mode_vals_im,
        config, nl::Int, nm::Int, nr::Int) where {T}
    return _pack_erk2_bc_side_common(Int32(1), zeros(T, nr), side.value,
        zero(T), zero(T), false, zero(T), mode_vals, mode_vals_im, config, nl, nm)
end

function _pack_erk2_bc_side(side::SolverERK2StencilSide{T}, mode_vals, mode_vals_im,
        config, nl::Int, nm::Int, nr::Int) where {T}
    stencil = zeros(T, nr)
    length(side.stencil) == nr && (stencil .= side.stencil)
    return _pack_erk2_bc_side_common(Int32(2), stencil, side.target,
        side.r_inv, side.l_sign, side.use_l_correction, side.fixed_correction,
        mode_vals, mode_vals_im, config, nl, nm)
end

# Shared tail: scatter the scalar endpoint target over (l, m) and overlay any
# per-mode overrides. `kind` is the device tag (1 = assign, 2 = solve the row).
function _pack_erk2_bc_side_common(kind::Int32, stencil::Vector{T}, target::T,
        r_inv::T, l_sign::T, use_l_correction::Bool, fixed_correction::T,
        mode_vals, mode_vals_im, config, nl::Int, nm::Int) where {T}
    val_r = fill(target, nl, nm)
    val_i = fill(zero(T), nl, nm)
    if mode_vals !== nothing
        @inbounds for lm in 1:config.nlm
            lm <= length(mode_vals) || continue
            l = config.l_values[lm]; m = config.m_values[lm]
            (0 <= l <= nl - 1 && 0 <= m <= nm - 1) || continue
            val_r[l + 1, m + 1] = mode_vals[lm]
        end
    end
    if mode_vals_im !== nothing
        @inbounds for lm in 1:config.nlm
            lm <= length(mode_vals_im) || continue
            l = config.l_values[lm]; m = config.m_values[lm]
            (0 <= l <= nl - 1 && 0 <= m <= nm - 1) || continue
            val_i[l + 1, m + 1] = mode_vals_im[lm]
        end
    end
    return (; kind, stencil, r_inv, l_sign, use_l_correction, fixed_correction,
        val_r, val_i)
end

# bc_spec === nothing → both endpoints forced to zero (kind 0).
function _pack_erk2_bc(::Nothing, config, nl::Int, nm::Int, nr::Int, ::Type{T}) where {T}
    z = (; kind = Int32(0), stencil = zeros(T, nr),
        r_inv = zero(T), l_sign = zero(T), use_l_correction = false,
        fixed_correction = zero(T),
        val_r = zeros(T, nl, nm), val_i = zeros(T, nl, nm))
    return (; inner = z, outer = z)
end

function _pack_erk2_bc(spec::SolverERK2BoundarySpec{T}, config, nl::Int, nm::Int,
        nr::Int, ::Type{T2}) where {T, T2}
    return (;
        inner = _pack_erk2_bc_side(spec.inner, spec.inner_mode_values,
            spec.inner_mode_values_imag, config, nl, nm, nr),
        outer = _pack_erk2_bc_side(spec.outer, spec.outer_mode_values,
            spec.outer_mode_values_imag, config, nl, nm, nr))
end

# Dense (nr, nr, nl) stack from a per-unique-l Vector{Matrix}, slotted by l+1.
function _pack_erk2_dense(mats::Vector{Matrix{T}}, l_values::Vector{Int},
        nl::Int, nr::Int) where {T}
    out = zeros(T, nr, nr, nl)
    for (i, l) in enumerate(l_values)
        0 <= l <= nl - 1 || continue
        out[:, :, l + 1] .= mats[i]
    end
    return out
end

function _pack_erk2_cache(cache::ERK2StageCache{T}, nl::Int, nr::Int) where {T}
    cache.use_krylov && error("GPU ERK2 pack: Krylov caches are not supported")
    # Low-storage fold: the finalize  u⁺ = E_f·u₀ + dt·φ1_f·n₀ + 2dt·φ2·(n_st − n₀)
    # is regrouped as  u⁺ = acc + 2dt·φ2·n_st  with  acc = E_f·u₀ + dt·M1·n₀ and
    # M1 := φ1_f − 2·φ2 precomputed here — φ1_full never ships to the device and
    # the per-field linear/k1/n₀ stage buffers collapse into the single `acc`.
    M1 = [cache.phi1_full[i] .- 2 .* cache.phi2_full[i] for i in eachindex(cache.phi1_full)]
    return (;
        Eh = _pack_erk2_dense(cache.E_half, cache.l_values, nl, nr),
        Ef = _pack_erk2_dense(cache.E_full, cache.l_values, nl, nr),
        p1h = _pack_erk2_dense(cache.phi1_half, cache.l_values, nl, nr),
        M1 = _pack_erk2_dense(M1, cache.l_values, nl, nr),
        p2f = _pack_erk2_dense(cache.phi2_full, cache.l_values, nl, nr))
end

# φ1-column Green responses for the W-split P-recovery, per degree and per
# stage half: h_i = A_P⁻¹ R(c·φ1(cA)·e_i) with R zeroing the wall rows, plus
# the 2×2 endpoint-residual matrices M[j,i] = d1_row_j · h_i. Mirrors
# _erk2_poloidal_recover! exactly (which recomputes these constants per step).
function _pack_erk2_recovery(split, cache::ERK2StageCache{T}, dt::Float64,
        nl::Int, nr::Int) where {T}
    h1h = zeros(T, nl, nr); h2h = zeros(T, nl, nr)
    h1f = zeros(T, nl, nr); h2f = zeros(T, nl, nr)
    Mh = zeros(T, 2, 2, nl); Mf = zeros(T, 2, 2, nl)
    g = Vector{T}(undef, nr); h = Vector{T}(undef, nr)
    for (ci, l) in enumerate(cache.l_values)
        (1 <= l <= nl - 1) || continue          # l = 0 carries no poloidal flow
        haskey(split.lookup, l) || continue
        idx = split.lookup[l]
        s = l + 1
        for (c, phi, h1, h2, M) in (
            (dt / 2, cache.phi1_half[ci], h1h, h2h, Mh),
            (dt, cache.phi1_full[ci], h1f, h2f, Mf),
        )
            for (col, hdst) in ((1, h1), (nr, h2))
                @inbounds for r in 1:nr
                    g[r] = T(c) * phi[r, col]
                end
                g[1] = zero(T); g[nr] = zero(T)
                solve_banded!(h, split.p_factor[idx], g)
                hdst[s, :] .= h
                j = col == 1 ? 1 : 2
                M[1, j, s] = dot(split.d1_row_inner, h)
                M[2, j, s] = dot(split.d1_row_outer, h)
            end
        end
    end
    return (; h1h, h2h, h1f, h2f, Mh, Mf)
end

"""
    build_gpu_erk2_state(cpu_state) -> NamedTuple

Assemble the ERK2 device-state pack from a CPU `SolverState` configured with
the ERK2 timestepper. Builds (or reuses) the same memoized propagator caches
and boundary specs the CPU `integrate_solver_erk2_step!` uses, then packs them
into dense batched arrays for [`gpu_erk2_solver_step!`](@ref). Combine with the
base bundle from [`build_gpu_solver_state`](@ref):

    st  = ...                      # CPU SolverState, ERK2 timestepper
    solver_step!(st)               # warm-up (physical buffers, caches)
    gst = build_gpu_solver_state(st)
    erk = build_gpu_erk2_state(st)
    gpu_erk2_solver_step!(gst, erk)

Arrays are host `Array`s; move with `gpu_to_device` alongside the base bundle.
Boundary endpoint VALUES are baked at pack time (time-constant BCs only).
"""
function build_gpu_erk2_state(st)
    T = Float64
    params = st.parameters
    params.timestepper isa ExponentialRungeKutta2 || error(
        "build_gpu_erk2_state: state is configured with $(typeof(params.timestepper)), not ExponentialRungeKutta2")
    params.geometry === :shell || error(
        "build_gpu_erk2_state: GPU ERK2 supports only :shell geometry, got $(params.geometry) " *
        "(the velocity-poloidal recovery + boundary packs hard-code the shell layout)")
    # The boundary packs below bake endpoint values at pack time; a time-dependent
    # boundary would be frozen at t=0. Same scope limit as build_gpu_solver_state.
    _gpu_assert_static_bcs(st)
    runtime = st.runtime
    cfg = runtime.shtns_config
    domain = runtime.outer_core_domain
    nl = cfg.lmax + 1
    nm = cfg.mmax + 1
    nr = domain.N
    dt = params.timestep
    caches = st.timestep_caches

    velocity_bc_code = _velocity_bc_code(params.velocity_bcs)
    temperature_bc_code = _thermal_bc_code(params.temperature_bcs)
    composition_bc_code = _composition_bc_code(params.composition_bcs)

    # --- boundary specs (same builders + mode-value attachment as the CPU step) ---
    temp_spec = _get_or_build_erk2_boundary_spec!(
        caches, :temperature, temperature_bc_code,
        () -> build_solver_erk2_scalar_bc(T, domain, temperature_bc_code))
    tv = get_bc_vectors(st.fields.temperature)
    temp_spec = with_boundary_mode_values(temp_spec,
        tv.inner_real, tv.outer_real, tv.inner_imag, tv.outer_imag)

    vel_tor_spec = _get_or_build_erk2_boundary_spec!(
        caches, :velocity_tor, velocity_bc_code,
        () -> build_solver_erk2_velocity_tor_bc(T, domain, velocity_bc_code;
            config = cfg, rot_omega = 0.0))

    # --- propagator caches (memoized; identical getter calls to the CPU step) ---
    temp_cache = get_solver_erk2_temperature_cache!(
        caches, params.Pm / params.Pr, T, cfg, domain, dt, temperature_bc_code;
        bc_spec = temp_spec, use_krylov = false)
    vel_tor_cache = get_solver_erk2_cache!(
        caches, :velocity_toroidal, params.Ek, T, cfg, domain, dt;
        use_krylov = false, bc_spec = vel_tor_spec)
    vel_pol_cache = get_solver_erk2_cache!(
        caches, :velocity_poloidal, 1.0, T, cfg, domain, dt;
        use_krylov = false, bc_spec = nothing, dpol_operator = true)

    split = _get_or_build_poloidal_split!(st, velocity_bc_code)

    # Per-field packs as concretely-typed locals (no Dict{Symbol,Any} indirection,
    # so each `erk.<field>` infers its concrete NamedTuple type). Optional fields
    # are `nothing` unless the corresponding physics is active.
    temperature = (;
        _pack_erk2_cache(temp_cache, nl, nr)...,
        bc = _pack_erk2_bc(temp_spec, cfg, nl, nm, nr, T))
    velocity_tor = (;
        _pack_erk2_cache(vel_tor_cache, nl, nr)...,
        bc = _pack_erk2_bc(vel_tor_spec, cfg, nl, nm, nr, T))
    velocity_pol = (;
        _pack_erk2_cache(vel_pol_cache, nl, nr)...,
        bc = _pack_erk2_bc(nothing, cfg, nl, nm, nr, T))

    magnetic_tor = nothing
    magnetic_pol = nothing
    if params.include_magnetic && st.fields.magnetic !== nothing
        mag_tor_spec = _get_or_build_erk2_boundary_spec!(
            caches, :magnetic_tor, 0, () -> build_solver_erk2_magnetic_tor_bc(T, domain))
        mag_pol_spec = _get_or_build_erk2_boundary_spec!(
            caches, :magnetic_pol, 0, () -> build_solver_erk2_magnetic_pol_bc(T, domain))
        mag_tor_cache = get_solver_erk2_magnetic_toroidal_cache!(
            caches, 1.0, T, cfg, domain, dt;
            bc_spec = mag_tor_spec, use_krylov = false)
        mag_pol_cache = get_solver_erk2_magnetic_poloidal_cache!(
            caches, 1.0, T, cfg, domain, dt;
            bc_spec = mag_pol_spec, use_krylov = false)
        magnetic_tor = (;
            _pack_erk2_cache(mag_tor_cache, nl, nr)...,
            bc = _pack_erk2_bc(mag_tor_spec, cfg, nl, nm, nr, T))
        magnetic_pol = (;
            _pack_erk2_cache(mag_pol_cache, nl, nr)...,
            bc = _pack_erk2_bc(mag_pol_spec, cfg, nl, nm, nr, T))
    end

    composition = nothing
    if st.fields.composition !== nothing
        comp_spec = _get_or_build_erk2_boundary_spec!(
            caches, :composition, composition_bc_code,
            () -> build_solver_erk2_scalar_bc(T, domain, composition_bc_code))
        cv = get_bc_vectors(st.fields.composition)
        comp_spec = with_boundary_mode_values(comp_spec,
            cv.inner_real, cv.outer_real, cv.inner_imag, cv.outer_imag)
        comp_cache = get_solver_erk2_composition_cache!(
            caches, params.Pm / params.Sc, T, cfg, domain, dt, composition_bc_code;
            bc_spec = comp_spec, use_krylov = false)
        composition = (;
            _pack_erk2_cache(comp_cache, nl, nr)...,
            bc = _pack_erk2_bc(comp_spec, cfg, nl, nm, nr, T))
    end

    recovery = _pack_erk2_recovery(split, vel_pol_cache, dt, nl, nr)

    return (;
        dt = dt,
        temperature = temperature,
        velocity_tor = velocity_tor,
        velocity_pol = velocity_pol,
        magnetic_tor = magnetic_tor,
        magnetic_pol = magnetic_pol,
        composition = composition,
        recovery = recovery)
end
