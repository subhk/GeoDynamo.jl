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

    # --- shared operators (host-side) ---
    d1 = Array{T}(vel.∂r.data)
    d2 = Array{T}(vel.∂²r.data)
    lfac = T[l * (l + 1) for l in 0:cfg.lmax]
    rinv = T[dom.r[k, 3] for k in 1:nr]
    rinv2 = T[dom.r[k, 2] for k in 1:nr]
    r_vec = T[dom.r[k, 4] for k in 1:nr]
    rscale = copy(rinv)                       # 1/r scaling for v_r; lfac=l(l+1) multiplied at the call site
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
        (; b..., bc_in_r = bc_in_r, bc_in_i = bc_in_i, bc_out_r = bc_out_r, bc_out_i = bc_out_i)
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

    # NOTE: d1/d2/lfac/rinv/rinv2/rscale are SHARED (same backing array) across
    # nlops_vel, nlops_mag, and the top-level d1/rinv fields — safe because
    # gpu_solver_step! treats all operator arrays read-only.

    # --- physical lag buffers (current state of the CPU physical fields) ---
    T_phys = _phys_scalar(tmp.temperature)
    C_phys = cmp_ === nothing ? nothing : _phys_scalar(cmp_.composition)
    Bp = mag === nothing ? (nothing, nothing, nothing) : _phys_vector(mag.magnetic)
    Jp = mag === nothing ? (nothing, nothing, nothing) : _phys_vector(mag.current)

    return (;
        config = cfg, lmax = cfg.lmax, bw = bw, linear_weight = linw,
        nlops_vel = (; d1, d2, lfac, rinv, rinv2, rscale, sinθ, cosθ, E = T(p.Ek)),
        nlops_mag = (; d1, d2, lfac, rinv, rinv2, rscale),
        influence = influence,
        d1 = d1, mvals = mvals, rinv = rinv, r_vec = r_vec,
        thermal_factor = T((p.Pm / p.Pr) * p.Ra),
        comp_factor = T((p.Pm / p.Sc) * p.RaC),
        lorentz_coeff = T(1.0 / p.Pm),
        inv_dt_vel = T(p.Ek / p.timestep),
        inv_dt_mag = T(1.0 / p.timestep),
        inv_dt_temp = T((p.Pm / p.Pr) / p.timestep),
        inv_dt_comp = T((p.Pm / p.Sc) / p.timestep),
        velocity = velocity, magnetic = magnetic,
        temperature = temperature, composition = composition,
        T_phys = T_phys, C_phys = C_phys,
        B_r = Bp[1], B_θ = Bp[2], B_φ = Bp[3],
        J_r = Jp[1], J_θ = Jp[2], J_φ = Jp[3])
end

# Recursively move a value to `arch`'s backend: arrays via on_architecture,
# NamedTuples element-wise (preserving keys), everything else (scalars, config,
# nothing, Symbols) passes through unchanged.
function _to_device(x, arch)
    if x isa AbstractArray
        return on_architecture(arch, x)
    elseif x isa NamedTuple
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
