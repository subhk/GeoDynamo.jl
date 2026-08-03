# =============================================================================
# GPU Phase 5n2 — build the gpu_solver_step! device-state bundle from a real CPU
# SolverState, + the GPU≈CPU full-step validation gate.  The CPU spectral storage
# is slot-packed (not dense (lmax+1,mmax+1,nr)); `cpu_spectral_to_dense` scatters
# it.  lin/lu come from `state.implicit_matrices[field].{linear_matrices,
# factorizations}`; operators from the velocity field + outer-core domain; factors
# from SolverParameters.  Insulating, homogeneous-BC config for the first gate.
#
# Verified CPU accessors (worktree):
#   - spectral: SHTnsSpecField.{data_real,data_imag} (PencilArray) → parent(...);
#     slot via local_spectral_storage_slot(config, lm_idx); l/m via
#     config.l_values/.m_values; config.nlm/.lmax/.mmax.
#   - velocity SHTnsVelocityFields: .toroidal/.poloidal (spectral), .nl_*/.prev_nl_*,
#     .∂r/.∂²r (BandedMatrix, .data is (2bw+1,nr)), .coriolis_factors ((2,nlat):
#     row1=sinθ, row2=cosθ), .velocity (physical SHTnsVectorField).
#   - temperature SHTnsTemperatureField: .spectral, .nonlinear/.prev_nonlinear,
#     .temperature (physical SHTnsPhysField, parent(.data) = (nlat,nlon,nr)).
#   - composition: .spectral, .nonlinear/.prev_nonlinear, .composition (physical).
#   - magnetic SHTnsMagneticFields: .toroidal/.poloidal, .nl_*/.prev_nl_*,
#     .magnetic (physical B vector), .current (physical J vector).
#   - ImplicitMatrixSet: .linear_matrices[i].data, .factorizations[i].lu,
#     .l_values, .system_matrices[1].{bandwidth,size}.
#   - domain outer_core_domain.r columns: col2=r⁻², col3=r⁻¹, col4=r.
#   - velocity-poloidal influence: Dict{Int,ERK2InfluenceOp} in
#     state.timestep_caches.erk2_influence_velocity_poloidal.matrices (built lazily
#     during a step; effective diffusivity=1.0, θ from the timestepper).
# =============================================================================

"""
    cpu_spectral_to_dense(field_spec, config, nr, ::Type{T}) -> (dense_r, dense_i)

Scatter the CPU slot-packed spectral storage of `field_spec` into dense
`(lmax+1, mmax+1, nr)` real/imag arrays (mode `(l,m)` → slot `(l+1, m+1)`).
Modes absent from the local storage map are left at zero.
"""
function cpu_spectral_to_dense(field_spec, config, nr::Int, ::Type{T}) where {T}
    nl = config.lmax + 1
    nm = config.mmax + 1
    dr = zeros(T, nl, nm, nr)
    di = zeros(T, nl, nm, nr)
    pr = parent(field_spec.data_real)
    pim = parent(field_spec.data_imag)
    nr_local = size(pr, 3)
    @inbounds for lm_idx in 1:config.nlm
        slot = local_spectral_storage_slot(config, lm_idx)
        slot === nothing && continue
        l = config.l_values[lm_idx]
        m = config.m_values[lm_idx]
        (0 <= l <= config.lmax && 0 <= m <= config.mmax) || continue
        for k in 1:min(nr, nr_local)
            dr[l + 1, m + 1, k] = local_spectral_value(pr, slot, k)
            di[l + 1, m + 1, k] = local_spectral_value(pim, slot, k)
        end
    end
    return dr, di
end

"""
    cpu_bc_to_dense(bc_vec, config, ::Type{T}) -> dense

Scatter a CPU per-mode (length `nlm`) boundary vector into a dense `(lmax+1,
mmax+1)` array (mode `(l,m)` → `(l+1, m+1)`).  `nothing` → all zeros.
"""
function cpu_bc_to_dense(bc_vec, config, ::Type{T}) where {T}
    nl = config.lmax + 1
    nm = config.mmax + 1
    dense = zeros(T, nl, nm)
    bc_vec === nothing && return dense
    @inbounds for lm_idx in 1:config.nlm
        lm_idx <= length(bc_vec) || continue
        l = config.l_values[lm_idx]
        m = config.m_values[lm_idx]
        (0 <= l <= config.lmax && 0 <= m <= config.mmax) || continue
        dense[l + 1, m + 1] = T(bc_vec[lm_idx])
    end
    return dense
end

# Batched (2bw+1,nr,nl) lin + lu from an ImplicitMatrixSet, indexed by dim-3 = l+1.
# Degrees absent from the set leave zero columns (matching dense-degree slotting).
function _pack_implicit(mset, nl::Int, ::Type{T}) where {T}
    bw = mset.system_matrices[1].bandwidth
    nr = mset.system_matrices[1].size
    lin = zeros(T, 2bw + 1, nr, nl)
    lu = zeros(T, 2bw + 1, nr, nl)
    @inbounds for (i, l) in enumerate(mset.l_values)
        (0 <= l <= nl - 1) || continue
        lin[:, :, l + 1] .= mset.linear_matrices[i].data
        lu[:, :, l + 1] .= mset.factorizations[i].lu
    end
    return lin, lu, bw
end

# Physical scalar array (nlat,nlon,nr) for a temperature/composition field.
_phys_scalar(field) = Array(parent(field.data))

# Physical (r,θ,φ) component arrays for a vector field (SHTnsVectorField).
_phys_vector(vecfield) = (
    Array(parent(vecfield.r_component.data)),
    Array(parent(vecfield.θ_component.data)),
    Array(parent(vecfield.φ_component.data)),
)

# Velocity-poloidal influence operators (Dict{Int,ERK2InfluenceOp}) packed for the
# GPU correction kernel.  Prefer the live cache populated during a CPU step; if it
# is empty (no step taken yet) build it via the same getter the CPU update uses
# (effective diffusivity = 1.0, θ from the timestepper, the velocity BC code).
function _build_influence_pack(st, nl::Int, nr::Int, ::Type{T}) where {T}
    cache = st.timestep_caches
    params = st.parameters
    runtime = st.runtime
    velocity_bc = _velocity_bc_code(params.velocity_bcs)
    theta = _timestepper_implicit_theta(params.timestepper, params)
    entry = cache.erk2_influence_velocity_poloidal
    matrices = if entry !== nothing
        entry.matrices
    else
        get_solver_erk2_influence_matrices!(
            cache, :velocity_poloidal, T,
            runtime.shtns_config, runtime.outer_core_domain,
            one(Float64), params.timestep, velocity_bc; theta = theta,
        )
    end
    Gre_b, invG_b = gpu_pack_influence(matrices, nl, nr, CPU())
    return (; Gre_b, invG_b)
end

# Stage-4B poloidal W-split operators packed per degree (slot l+1), mirroring
# PoloidalSplitMatrices (state.jl) for the batched GPU step:
#   dpol/wlin: per-l banded operators (2bw+1, nr, nl)
#   wlu/plu:   per-l banded LU factors (same layout, BandedLU.lu payload)
#   h1/h2:     no-slip influence Green responses, (nl, nr) so a zero-copy
#              reshape to (nl, 1, nr) broadcasts against (nl, nm) corrections
#   M:         2×2 endpoint-influence matrices, (2, 2, nl)
#   d1_inner/d1_outer: endpoint first-derivative residual rows (length nr)
function _build_wsplit_pack(st, nl::Int, nr::Int, bw::Int, ::Type{T}) where {T}
    velocity_bc = _velocity_bc_code(st.parameters.velocity_bcs)
    split = _get_or_build_poloidal_split!(st, velocity_bc)
    split_bw = split.dpol_op[1].bandwidth
    split_bw == bw || error(
        "W-split bandwidth $split_bw ≠ velocity operator bandwidth $bw")
    dpol = zeros(T, 2bw + 1, nr, nl); wlin = zeros(T, 2bw + 1, nr, nl)
    wlu = zeros(T, 2bw + 1, nr, nl); plu = zeros(T, 2bw + 1, nr, nl)
    h1 = zeros(T, nl, nr); h2 = zeros(T, nl, nr)
    M = zeros(T, 2, 2, nl)
    for (i, l) in enumerate(split.l_values)
        s = l + 1
        s <= nl || continue
        dpol[:, :, s] .= split.dpol_op[i].data
        wlin[:, :, s] .= split.w_linear[i].data
        wlu[:, :, s] .= split.w_factor[i].lu
        plu[:, :, s] .= split.p_factor[i].lu
        h1[s, :] .= split.h1[i]
        h2[s, :] .= split.h2[i]
        M[:, :, s] .= split.influence[i]
    end
    return (; dpol, wlin, wlu, plu, h1, h2, M,
        d1_inner = Vector{T}(split.d1_row_inner),
        d1_outer = Vector{T}(split.d1_row_outer))
end

function _pack_wsplit(split, nl::Int, nr::Int, bw::Int, ::Type{T}) where {T}
    split_bw = split.dpol_op[1].bandwidth
    split_bw == bw || error(
        "W-split bandwidth $split_bw ≠ velocity operator bandwidth $bw")
    dpol = zeros(T, 2bw + 1, nr, nl); wlin = zeros(T, 2bw + 1, nr, nl)
    wlu = zeros(T, 2bw + 1, nr, nl); plu = zeros(T, 2bw + 1, nr, nl)
    h1 = zeros(T, nl, nr); h2 = zeros(T, nl, nr)
    M = zeros(T, 2, 2, nl)
    for (i, l) in enumerate(split.l_values)
        s = l + 1
        s <= nl || continue
        dpol[:, :, s] .= split.dpol_op[i].data
        wlin[:, :, s] .= split.w_linear[i].data
        wlu[:, :, s] .= split.w_factor[i].lu
        plu[:, :, s] .= split.p_factor[i].lu
        h1[s, :] .= split.h1[i]
        h2[s, :] .= split.h2[i]
        M[:, :, s] .= split.influence[i]
    end
    return (; dpol, wlin, wlu, plu, h1, h2, M,
        d1_inner = Vector{T}(split.d1_row_inner),
        d1_outer = Vector{T}(split.d1_row_outer))
end

# `map` over the CB3_GAMMA tuple returns a concrete `NTuple{3, NamedTuple}` (the three
# stage packs are homogeneously typed), so `state.cb3[stage]` field reads infer concretely
# — unlike the old `Any[]` + `push!` + `Tuple(packs)`, which produced an abstract `Tuple`.
function _build_cb3_stage_pack(st, nl::Int, nr::Int, bw::Int, ::Type{T}) where {T}
    # SMR/Cavaglieri-Bewley IMEX-RK3: the per-stage system matrix is
    # (mass/dt) I − β·L built with the FULL step dt and the companion CN
    # coefficient β (NOT γ·dt with full-implicit θ=1). The bare per-l linear
    # operators (lin / wlin) are retained so the step can add the explicit α·L
    # term, mirroring the CPU path in src/timestep/cb3.jl. `map` over CB3_BETA
    # returns a concrete NTuple{3} so `state.cb3[stage]` field reads infer.
    return map(CB3_BETA) do beta
        matrices, magnetic_ic_admittance = _build_implicit_matrices_dict(
            T,
            st.backend.shtns_config,
            st.backend.outer_core_domain,
            st.backend.inner_core_domain,
            st.parameters,
            st.parameters.timestep;
            theta = beta,
        )
        magnetic_ic_admittance === nothing || throw(ArgumentError(
            "RungeKutta3 GPU path does not yet support magnetic_inner_bc=:conducting_inner_core"))
        store = create_solver_implicit_matrix_store(matrices)
        vtor_lin, vtor_lu, _ = _pack_implicit(store[:velocity_tor], nl, T)
        mt_lin, mt_lu, _ = _pack_implicit(store[:magnetic_tor], nl, T)
        mp_lin, mp_lu, _ = _pack_implicit(store[:magnetic_pol], nl, T)
        tt_lin, tt_lu, _ = _pack_implicit(store[:temperature], nl, T)
        cc = haskey(store, :composition) ? _pack_implicit(store[:composition], nl, T) : nothing
        split = create_velocity_poloidal_split_matrices(
            st.runtime.shtns_config,
            st.runtime.outer_core_domain,
            st.parameters.Ek,
            st.parameters.timestep;
            velocity_bc_code = _velocity_bc_code(st.parameters.velocity_bcs),
            theta = beta,
            T = T,
        )
        (;
            velocity_tor_lin = vtor_lin,
            velocity_tor_lu = vtor_lu,
            magnetic_tor_lin = mt_lin,
            magnetic_tor_lu = mt_lu,
            magnetic_pol_lin = mp_lin,
            magnetic_pol_lu = mp_lu,
            temperature_lin = tt_lin,
            temperature_lu = tt_lu,
            composition_lin = cc === nothing ? nothing : cc[1],
            composition_lu = cc === nothing ? nothing : cc[2],
            wsplit = _pack_wsplit(split, nl, nr, bw, T),
        )
    end
end

"""
    _gpu_field_has_time_dependent_bc(field) -> Bool

Whether `field` carries a loaded `BoundaryConditionSet` whose inner or outer data
varies in time. `nothing` (disabled field) and fields without boundary-file support
are both `false`.
"""
function _gpu_field_has_time_dependent_bc(field)
    field === nothing && return false
    hasproperty(field, :boundary_condition_set) || return false
    set = field.boundary_condition_set
    set === nothing && return false
    return set.inner_boundary.is_time_dependent || set.outer_boundary.is_time_dependent
end

"""
    _gpu_assert_static_bcs(st)

Reject time-dependent boundary data on the GPU path.

The device bundles bake boundary endpoint VALUES at pack time, whereas the CPU
refreshes them every step (`apply_boundary_conditions!`, bcs/integration.jl). Running
a moving boundary on the device would silently freeze it at its t=0 values, so this
errors the same way the conducting-inner-core, `:ball`, and topography scope limits do.
"""
function _gpu_assert_static_bcs(st)
    offenders = String[]
    for (name, field) in (("temperature", st.fields.temperature),
                          ("velocity", st.fields.velocity),
                          ("magnetic", st.fields.magnetic),
                          ("composition", st.fields.composition))
        _gpu_field_has_time_dependent_bc(field) && push!(offenders, name)
    end
    isempty(offenders) || error(
        "GPU solver path does not support time-dependent boundary conditions " *
        "($(join(offenders, ", "))); the device bundle bakes boundary endpoint values " *
        "at pack time, so the boundary would be silently frozen at its initial values. " *
        "Use the CPU path for time-dependent boundary data.")
    return nothing
end

"""
    _gpu_assert_single_rank(caller)

Reject the dense device-state path under MPI with more than one rank.

The bundle is a whole-domain copy; nothing in `src/gpu/` is pencil- or
rank-aware. Without this the failure surfaces as an opaque `DimensionMismatch`
part-way through a step rather than as a scope limit.
"""
function _gpu_assert_single_rank(caller::AbstractString)
    nprocs = get_nprocs()
    nprocs == 1 || error(
        "$caller: the GPU solver path is single-rank only (got $nprocs MPI ranks). " *
        "The device bundle is a whole-domain dense copy — it has no pencil/halo " *
        "awareness, so a distributed state would be silently truncated to the modes " *
        "and radial slices owned by each rank. Run the GPU path on one rank, or use " *
        "the CPU path for distributed runs.")
    return nothing
end

"""
    build_gpu_solver_state(cpu_state) -> NamedTuple

Assemble the `gpu_solver_step!` device-state bundle from a CPU `SolverState`
(insulating magnetic, homogeneous BCs).  Arrays are on the CPU (Array) backend;
move to a device with `on_architecture(GPU(), …)` for a GPU run.  The returned
NamedTuple matches the layout consumed by [`gpu_solver_step!`](@ref): per-field
`tor`/`pol`/scalar bundles (dense spectral + prev_nl + lin/lu + BC rows), shared
operators (`nlops_vel`/`nlops_mag`, `d1`/`mvals`/`rinv`/`r_vec`), coupling factors,
the velocity-poloidal influence pack, and the persistent lagged physical buffers
(`T_phys`/`C_phys`/`B_*`/`J_*`).

The CPU spectral storage is slot-packed; [`cpu_spectral_to_dense`](@ref) scatters
it to the dense `(lmax+1, mmax+1, nr)` layout the GPU kernels use.  The physical
lag buffers are read from the CPU physical fields AS THEY STAND — to reproduce the
one-step velocity lag, build the device state AFTER one warm-up `solver_step!`.

Scope limits, each rejected loudly rather than silently approximated: a conducting
inner core, `:ball` geometry, enabled topography coupling, and time-dependent
boundary data (endpoint VALUES are baked into the packs at build time).
"""
function build_gpu_solver_state(st)
    T = Float64
    cfg = st.backend.shtns_config
    nl = cfg.lmax + 1
    nm = cfg.mmax + 1
    dom = st.runtime.outer_core_domain
    nr = dom.N
    p = st.parameters
    vel = st.fields.velocity

    # Builder scope: insulating magnetic + CNAB2 only. A conducting inner core sets
    # `magnetic_ic_admittance`; `gpu_solver_step!` always runs the insulating magnetic
    # path (ic=nothing), so error loudly rather than silently drop the φ0 history-flux BC.
    if st.fields.magnetic !== nothing && st.magnetic_ic_admittance !== nothing
        error("build_gpu_solver_state: only insulating magnetic is supported; " *
              "magnetic_ic_admittance is set (conducting inner core not yet wired into gpu_solver_step!)")
    end
    # The GPU kernels hard-code the spherical-shell layout (shell poloidal recovery,
    # shell BC rows); a :ball config would silently integrate the wrong operators.
    p.geometry === :shell || error(
        "build_gpu_solver_state: GPU solver supports only :shell geometry, got $(p.geometry).")
    # Topography core-mantle coupling has no GPU counterpart — the CPU applies
    # apply_solver_topography! after each nonlinear pass; the GPU step never does.
    # Error rather than silently drop the coupling when it is enabled.
    st.topography.config.enabled && error(
        "build_gpu_solver_state: topography coupling is enabled but the GPU step path " *
        "does not apply it (no GPU port of apply_solver_topography!).")
    # Boundary endpoint values are baked into the packs below — reject anything that
    # would move underneath them.
    _gpu_assert_static_bcs(st)
    # The device bundle is a WHOLE-DOMAIN dense copy (see src/gpu/fields.jl: "single
    # GPU, no MPI/pencils"). Under >1 rank `cpu_spectral_to_dense` silently drops the
    # modes this rank does not own and fills radius by min(nr, nr_local), and the
    # lagged physical buffers below are sized from the rank-LOCAL field — which
    # surfaces much later as a bare DimensionMismatch inside gpu_solver_step!
    # (`state.T_phys .= Tn.data`). Fail here instead, like the other scope limits.
    _gpu_assert_single_rank("build_gpu_solver_state")

    # --- shared operators (host-side) ---
    d1 = Array{T}(vel.∂r.data)
    d2 = Array{T}(vel.∂²r.data)
    lfac = T[l * (l + 1) for l in 0:cfg.lmax]
    rinv = T[dom.r[k, 3] for k in 1:nr]
    rinv2 = T[dom.r[k, 2] for k in 1:nr]
    r_vec = T[dom.r[k, 4] for k in 1:nr]
    r2 = T[dom.r[k, 6] for k in 1:nr]         # r² — Q-based poloidal analysis (Stage-2)
    rscale = copy(rinv2)                      # Stage-2 solenoidal: v_r = l(l+1)·P/r² ⇒ 1/r² (was 1/r pre-Stage-2)
    sinθ = T[vel.coriolis_factors[1, i] for i in 1:cfg.nlat]
    cosθ = T[vel.coriolis_factors[2, i] for i in 1:cfg.nlat]
    mvals = T[m for m in 0:cfg.mmax]
    bw = st.implicit_matrices[:temperature].system_matrices[1].bandwidth

    # --- CNAB2 weights ---
    θ = st.implicit_matrices[:temperature].theta
    linw = one(T) - T(θ)

    # --- per-field bundle builders ---
    function vbundle(spec, prev_field, key)
        sr, si = cpu_spectral_to_dense(spec, cfg, nr, T)
        pr, pim = cpu_spectral_to_dense(prev_field, cfg, nr, T)
        lin, lu, _ = _pack_implicit(st.implicit_matrices[key], nl, T)
        z = zeros(T, nl, nm)
        (; spec_r = sr, spec_i = si, prev_nl_r = pr, prev_nl_i = pim, lin = lin, lu = lu,
            bc_in_r = copy(z), bc_in_i = copy(z), bc_out_r = copy(z), bc_out_i = copy(z))
    end
    # scalar bundle with extracted BC rows (homogeneous by default, scattered for safety)
    function sbundle(field, spec, prev_field, key)
        b = vbundle(spec, prev_field, key)
        bc = get_bc_vectors(field)
        bc_in_r = cpu_bc_to_dense(bc.inner_real, cfg, T)
        bc_in_i = cpu_bc_to_dense(bc.inner_imag, cfg, T)
        bc_out_r = cpu_bc_to_dense(bc.outer_real, cfg, T)
        bc_out_i = cpu_bc_to_dense(bc.outer_imag, cfg, T)
        # Internal source (internal_heating / compositional_source): a per-radial-level
        # profile the CPU adds into the advection physical field before the spectral
        # analysis (solver_add_internal_sources_local!). Carry the nr-vector so the GPU
        # scalar nonlinear can do the same; `nothing` when no source is configured.
        src = all(iszero, field.internal_sources) ? nothing : collect(T, field.internal_sources)
        (; b..., bc_in_r = bc_in_r, bc_in_i = bc_in_i, bc_out_r = bc_out_r, bc_out_i = bc_out_i,
            internal_source = src)
    end
    function mbundle(spec, prev_field, key)
        sr, si = cpu_spectral_to_dense(spec, cfg, nr, T)
        pr, pim = cpu_spectral_to_dense(prev_field, cfg, nr, T)
        lin, lu, _ = _pack_implicit(st.implicit_matrices[key], nl, T)
        (; spec_r = sr, spec_i = si, prev_nl_r = pr, prev_nl_i = pim, lin = lin, lu = lu)
    end

    velocity = (;
        tor = vbundle(vel.toroidal, vel.prev_nl_toroidal, :velocity_tor),
        pol = vbundle(vel.poloidal, vel.prev_nl_poloidal, :velocity_pol))

    mag = st.fields.magnetic
    magnetic = mag === nothing ? nothing : (;
        tor = mbundle(mag.toroidal, mag.prev_nl_toroidal, :magnetic_tor),
        pol = mbundle(mag.poloidal, mag.prev_nl_poloidal, :magnetic_pol))

    tmp = st.fields.temperature
    temperature = sbundle(tmp, tmp.spectral, tmp.prev_nonlinear, :temperature)

    cmp_ = st.fields.composition
    composition = cmp_ === nothing ? nothing :
                  sbundle(cmp_, cmp_.spectral, cmp_.prev_nonlinear, :composition)

    influence = _build_influence_pack(st, nl, nr, T)
    wsplit = _build_wsplit_pack(st, nl, nr, bw, T)
    cb3 = st.parameters.timestepper isa RungeKutta3 ? _build_cb3_stage_pack(st, nl, nr, bw, T) : nothing

    # NOTE: d1/d2/lfac/rinv/rinv2/rscale/r/r2 are SHARED (same backing array)
    # across nlops_vel, nlops_mag, and the top-level d1/rinv/r_vec fields — safe
    # because gpu_solver_step! treats all operator arrays read-only.

    # --- physical lag buffers (current state of the CPU physical fields) ---
    T_phys = _phys_scalar(tmp.temperature)
    C_phys = cmp_ === nothing ? nothing : _phys_scalar(cmp_.composition)
    Bp = mag === nothing ? (nothing, nothing, nothing) : _phys_vector(mag.magnetic)
    Jp = mag === nothing ? (nothing, nothing, nothing) : _phys_vector(mag.current)

    return (;
        config = cfg, lmax = cfg.lmax, bw = bw, linear_weight = linw,
        nlops_vel = (; d1, d2, lfac, rinv, rinv2, rscale, r = r_vec, r2, sinθ, cosθ, E = T(p.Ek)),
        nlops_mag = (; d1, d2, lfac, rinv, rinv2, rscale, r = r_vec, r2),
        influence = influence, wsplit = wsplit,
        d1 = d1, mvals = mvals, rinv = rinv, r_vec = r_vec,
        thermal_factor = T((p.Pm / p.Pr) * p.Ra),
        comp_factor = T((p.Pm / p.Sc) * p.RaC),
        lorentz_coeff = T(1.0 / p.Pm),
        inv_dt_vel = T(p.Ek / p.timestep),
        inv_dt_mag = T(1.0 / p.timestep),
        # Scalar mass coefficient is 1 (not Pm/Pr or Pm/Sc): the CPU scalar implicit
        # matrices fold the diffusivity into L and use a 1/dt mass term (scalar_bc.jl,
        # mass_coeff=1). The GPU reuses those CPU LUs, so the RHS mass term MUST match.
        inv_dt_temp = T(1.0 / p.timestep),
        inv_dt_comp = T(1.0 / p.timestep),
        cb3 = cb3,
        velocity = velocity, magnetic = magnetic,
        temperature = temperature, composition = composition,
        work = GPUWorkspace(),
        T_phys = T_phys, C_phys = C_phys,
        B_r = Bp[1], B_θ = Bp[2], B_φ = Bp[3],
        J_r = Jp[1], J_θ = Jp[2], J_φ = Jp[3])
end

# Recursively move a value to `arch`'s backend: arrays via on_architecture,
# NamedTuples element-wise (preserving keys), everything else (scalars, config,
# nothing, Symbols) passes through unchanged.
function _to_device(x, arch)
    if x isa GPUWorkspace
        # scratch pools are backend-specific; reset so buffers are recreated
        # lazily on the destination backend
        return GPUWorkspace()
    elseif x isa AbstractArray
        return on_architecture(arch, x)
    elseif x isa NamedTuple
        return map(v -> _to_device(v, arch), x)
    elseif x isa Tuple
        # e.g. `cb3` is a Tuple of per-stage NamedTuple packs; without this branch
        # it would fall through unchanged and its LU/W-split arrays would stay on
        # the host while the rest of the state moves to the device.
        return map(v -> _to_device(v, arch), x)
    else
        return x
    end
end

"""
    gpu_to_device(state, arch) -> NamedTuple

Deep-copy a [`build_gpu_solver_state`](@ref) bundle to `arch`'s backend: every array
(nested per-field bundles, `nlops_*`, `influence`, the physical buffers) is moved via
`on_architecture(arch, …)`; scalars, `config`, and `nothing` pass through.  Use this to
move a CPU-built state to `GPU()` for the GPU≈CPU hardware gate, then run
[`gpu_solver_step!`](@ref) on the device copy.  `gpu_to_device(state, CPU())` returns an
independent deep copy on the host.
"""
gpu_to_device(state, arch::AbstractArchitecture) = _to_device(state, arch)

"""
    dense_to_cpu_spectral!(field_spec, dense_r, dense_i, config, nr) -> field_spec

Inverse of [`cpu_spectral_to_dense`](@ref): scatter dense `(lmax+1, mmax+1, nr)`
real/imag arrays back into the CPU slot-packed spectral storage of `field_spec`
(`(l+1, m+1)` → mode slot).  `dense_*` may be host or device arrays (host-copied here).
"""
function dense_to_cpu_spectral!(field_spec, dense_r, dense_i, config, nr::Int)
    dr = dense_r isa Array ? dense_r : Array(dense_r)
    di = dense_i isa Array ? dense_i : Array(dense_i)
    pr = parent(field_spec.data_real)
    pim = parent(field_spec.data_imag)
    nr_local = size(pr, 3)
    @inbounds for lm_idx in 1:config.nlm
        slot = local_spectral_storage_slot(config, lm_idx)
        slot === nothing && continue
        l = config.l_values[lm_idx]
        m = config.m_values[lm_idx]
        (0 <= l <= config.lmax && 0 <= m <= config.mmax) || continue
        for k in 1:min(nr, nr_local)
            set_local_spectral_value!(pr, slot, k, dr[l + 1, m + 1, k])
            set_local_spectral_value!(pim, slot, k, di[l + 1, m + 1, k])
        end
    end
    return field_spec
end

# Copy a (possibly device) dense physical array into a CPU physical field's storage.
function _sync_phys!(field, src)
    src === nothing && return field
    parent(field.data) .= (src isa Array ? src : Array(src))
    return field
end

"""
    sync_gpu_state_to_cpu!(cpu_state, gpu_state) -> cpu_state

Write the GPU device-state back into the CPU `SolverState` so CPU stepping /
diagnostics / output / restart can continue from the GPU-evolved state.  Three
categories are synced (the inverse of what [`build_gpu_solver_state`](@ref) reads):

1. spectral fields (velocity tor/pol, temperature, and — when present — magnetic
   tor/pol, composition), into the slot-packed CPU storage;
2. the CNAB2 `prev_nl` histories of the same fields;
3. the lagged physical buffers (`T_phys`/`C_phys`, `B_*`/`J_*`) into the CPU
   physical fields they were built from.

The `gpu_state` arrays may be host or device.
"""
function sync_gpu_state_to_cpu!(st, gst)
    cfg = st.backend.shtns_config
    nr = st.runtime.outer_core_domain.N
    vel = st.fields.velocity
    dense_to_cpu_spectral!(vel.toroidal, gst.velocity.tor.spec_r, gst.velocity.tor.spec_i, cfg, nr)
    dense_to_cpu_spectral!(vel.poloidal, gst.velocity.pol.spec_r, gst.velocity.pol.spec_i, cfg, nr)
    dense_to_cpu_spectral!(vel.prev_nl_toroidal, gst.velocity.tor.prev_nl_r, gst.velocity.tor.prev_nl_i, cfg, nr)
    dense_to_cpu_spectral!(vel.prev_nl_poloidal, gst.velocity.pol.prev_nl_r, gst.velocity.pol.prev_nl_i, cfg, nr)
    tmp = st.fields.temperature
    dense_to_cpu_spectral!(tmp.spectral, gst.temperature.spec_r, gst.temperature.spec_i, cfg, nr)
    dense_to_cpu_spectral!(tmp.prev_nonlinear, gst.temperature.prev_nl_r, gst.temperature.prev_nl_i, cfg, nr)
    _sync_phys!(tmp.temperature, gst.T_phys)
    mag = st.fields.magnetic
    if mag !== nothing && gst.magnetic !== nothing
        dense_to_cpu_spectral!(mag.toroidal, gst.magnetic.tor.spec_r, gst.magnetic.tor.spec_i, cfg, nr)
        dense_to_cpu_spectral!(mag.poloidal, gst.magnetic.pol.spec_r, gst.magnetic.pol.spec_i, cfg, nr)
        dense_to_cpu_spectral!(mag.prev_nl_toroidal, gst.magnetic.tor.prev_nl_r, gst.magnetic.tor.prev_nl_i, cfg, nr)
        dense_to_cpu_spectral!(mag.prev_nl_poloidal, gst.magnetic.pol.prev_nl_r, gst.magnetic.pol.prev_nl_i, cfg, nr)
        _sync_phys!(mag.magnetic.r_component, gst.B_r)
        _sync_phys!(mag.magnetic.θ_component, gst.B_θ)
        _sync_phys!(mag.magnetic.φ_component, gst.B_φ)
        _sync_phys!(mag.current.r_component, gst.J_r)
        _sync_phys!(mag.current.θ_component, gst.J_θ)
        _sync_phys!(mag.current.φ_component, gst.J_φ)
    end
    cmp_ = st.fields.composition
    if cmp_ !== nothing && gst.composition !== nothing
        dense_to_cpu_spectral!(cmp_.spectral, gst.composition.spec_r, gst.composition.spec_i, cfg, nr)
        dense_to_cpu_spectral!(cmp_.prev_nonlinear, gst.composition.prev_nl_r, gst.composition.prev_nl_i, cfg, nr)
        _sync_phys!(cmp_.composition, gst.C_phys)
    end
    return st
end
