# ================================================================================
# Public API: Initial Condition types and set_initial_condition! dispatch
# ================================================================================
#
# These types wrap the lower-level helpers in src/core/initial_conditions.jl so
# that callers can write:
#
#   model = GeodynamoModel(grid; initial_conditions = (
#       temperature = RandomPerturbation(amplitude=0.1, lmax=10),
#       magnetic    = AnalyticIC(:dipole; amplitude=1.0),
#   ))
#
# or call set_initial_condition! directly after model construction.
# ================================================================================

# ────────────────────────────────────────────────────────────────────────────────
# IC type definitions
# ────────────────────────────────────────────────────────────────────────────────

"""
    RandomPerturbation(; amplitude, lmax, domain=nothing, seed=nothing)

Superimpose random spectral perturbations up to degree `lmax` ON a field. The
perturbation is *added* to the field's existing spectral content (it does not
clear the field first), so a base state applied earlier — e.g.
`set!(model, :temperature, AnalyticIC(:conductive))` — survives. Apply the base
IC first, then the `RandomPerturbation`.

- `amplitude` – overall scale of the perturbation (required)
- `lmax`      – maximum spherical-harmonic degree to excite (required)
- `domain`    – DEPRECATED / unused. Accepted and forwarded for API symmetry,
                but the underlying randomizers currently ignore it (the off-center
                ball grid has no r=0 node, so no field-plane regularity zeroing is
                needed). Kept to avoid breaking existing call sites.
- `seed`      – optional `Int` random seed for reproducibility
"""
struct RandomPerturbation
    amplitude::Float64
    lmax::Int
    domain::Any          # Union{RadialDomainType, Nothing} – kept as Any to
    # avoid a hard dependency on the internal type alias
    seed::Union{Int, Nothing}
end

function RandomPerturbation(;
        amplitude::Real,
        lmax::Int,
        domain = nothing,
        seed::Union{Int, Nothing} = nothing
)
    lmax >= 0 || throw(ArgumentError(
        "RandomPerturbation: lmax = $lmax must be >= 0 " *
        "(a negative lmax excites no modes and silently produces no perturbation)"))
    return RandomPerturbation(Float64(amplitude), lmax, domain, seed)
end

# ────────────────────────────────────────────────────────────────────────────────

"""
    AnalyticIC(pattern::Symbol; amplitude=1.0, parameters...)

Set a field to an analytical pattern defined in `InitialConditions`.

Supported patterns (see `set_analytical_initial_conditions!` for details):
- `:conductive`     – conductive temperature profile
- `:dipole`         – dipolar magnetic field
- `:convective`     – small convective velocity perturbations
- `:stratified`     – stratified composition profile
- `:hot_blob`       – hot thermal blob
- `:uniform_field`  – uniform magnetic field
- `:blob`           – compositional blob

Any extra keyword arguments are forwarded verbatim to the underlying
`set_analytical_initial_conditions!` call.
"""
struct AnalyticIC{P}
    pattern::Symbol
    amplitude::Float64
    parameters::P
end

function AnalyticIC(pattern::Symbol; amplitude::Real = 1.0, parameters...)
    return AnalyticIC{typeof(parameters)}(pattern, Float64(amplitude), parameters)
end

# ────────────────────────────────────────────────────────────────────────────────

"""
    FileIC(file_path::String)

Load initial conditions from a NetCDF file.

The file is read by `InitialConditions.load_initial_conditions!(field, field_type,
file_path)`.  Note that full NetCDF I/O is not yet implemented in the underlying
module; at runtime a warning is printed and an analytical fallback is used instead.

`field_type` is inferred automatically from the `field` symbol passed to
`set_initial_condition!` (`:velocity`, `:temperature`, `:magnetic`, or
`:composition`).
"""
struct FileIC
    file_path::String
end

# ────────────────────────────────────────────────────────────────────────────────

"""
    ZeroIC()

Leave the field at its default (zero) initial state.  This is a no-op: after
`initialize_solver_state` all fields are already zero-initialized.
"""
struct ZeroIC end

# ────────────────────────────────────────────────────────────────────────────────
# Helper: extract the concrete field object from a GeodynamoModel
# ────────────────────────────────────────────────────────────────────────────────

function _get_field(model::GeodynamoModel, field::Symbol)
    fields = model.state.fields
    if field === :velocity
        return fields.velocity
    elseif field === :temperature
        return fields.temperature
    elseif field === :magnetic
        f = fields.magnetic
        isnothing(f) && throw(ArgumentError(
            "Magnetic field requested but model was built with include_magnetic=false"))
        return f
    elseif field === :composition
        f = fields.composition
        isnothing(f) && throw(ArgumentError(
            "Composition field requested but model was built with include_composition=false"))
        return f
    else
        throw(ArgumentError(
            "Unknown field symbol :$field.  " *
            "Valid options are :velocity, :temperature, :magnetic, :composition."))
    end
end

# ────────────────────────────────────────────────────────────────────────────────
# set_initial_condition! – public dispatch
# ────────────────────────────────────────────────────────────────────────────────

"""
    set_initial_condition!(model::GeodynamoModel, field::Symbol, ic)

Apply initial condition `ic` to the named `field` of `model`.

# Arguments
- `model`  – a `GeodynamoModel` returned by `GeodynamoModel(...)`
- `field`  – one of `:velocity`, `:temperature`, `:magnetic`, `:composition`
- `ic`     – an IC descriptor (`RandomPerturbation`, `AnalyticIC`, `FileIC`, or
             `ZeroIC`), or — for the scalar fields `:temperature` and
             `:composition` only — a direct value: a `Real` (uniform), a
             function `(r, θ, φ) -> value` (radius, colatitude, longitude), or
             an `AbstractArray{<:Real,3}` matching the local physical grid

# Examples
```julia
set_initial_condition!(model, :temperature, RandomPerturbation(amplitude=0.1, lmax=10))
set_initial_condition!(model, :magnetic,    AnalyticIC(:dipole; amplitude=1.0))
set_initial_condition!(model, :velocity,    FileIC("/path/to/checkpoint.nc"))
set_initial_condition!(model, :composition, ZeroIC())
set_initial_condition!(model, :temperature, (r, θ, φ) -> 1 - r)
```
"""
function set_initial_condition!(model::GeodynamoModel, field::Symbol, ic)
    _apply_initial_condition!(model, field, ic)
    # The user has supplied an IC, so mark the state initialized. Otherwise the
    # first solver_step!/run_solver! would run initialize_solver_fields!
    # (`state.is_initialized || ...`, solver/mainloop.jl) and overwrite this IC
    # with the default field initialization.
    model.state.is_initialized = true
    return model
end

# --- RandomPerturbation --------------------------------------------------------

function _apply_initial_condition!(
        model::GeodynamoModel,
        field::Symbol,
        ic::RandomPerturbation
)
    if ic.seed !== nothing
        Random.seed!(ic.seed)
    end

    f = _get_field(model, field)
    domain = ic.domain

    if field === :velocity
        InitialConditions.randomize_vector_field!(f;
            amplitude = ic.amplitude,
            lmax = ic.lmax,
            domain = domain
        )
    elseif field === :temperature || field === :composition
        InitialConditions.randomize_scalar_field!(f;
            amplitude = ic.amplitude,
            lmax = ic.lmax,
            domain = domain
        )
    elseif field === :magnetic
        InitialConditions.randomize_magnetic_field!(f;
            amplitude = ic.amplitude,
            lmax = ic.lmax,
            domain = domain
        )
    end

    return model
end

# --- AnalyticIC ----------------------------------------------------------------

function _apply_initial_condition!(
        model::GeodynamoModel,
        field::Symbol,
        ic::AnalyticIC
)
    f = _get_field(model, field)
    p = model.state.parameters
    dom = model.state.backend.outer_core_domain
    # Source defaults to the model's configured field source; an explicit
    # `:source` in the IC parameters overrides it. We consume `:source` here and
    # drop it from the splat below so the explicit `source = src` kwarg does not
    # clash with a `source` key in `ic.parameters...`.
    src = get(ic.parameters, :source,
        field === :composition ? p.compositional_source : p.internal_heating)
    diff = field === :composition ? p.Pm / p.Sc : p.Pm / p.Pr
    # `ic.parameters` is a kwargs container; drop `:source` (consumed above as the
    # explicit `source = src` kwarg) so it does not clash when splatted.
    rest = Base.structdiff(NamedTuple(ic.parameters), NamedTuple{(:source,)})
    InitialConditions.set_analytical_initial_conditions!(f, field, ic.pattern;
        amplitude = ic.amplitude, geometry = p.geometry, source = src,
        diffusivity = diff, domain = dom, rest...)
    return model
end

# --- FileIC --------------------------------------------------------------------
#
# `load_initial_conditions!(field, field_type, file_path)` accepts a `field_type`
# Symbol as its second argument, so we forward the `field` symbol directly.
# At the time of writing, NetCDF loading is not yet implemented in the underlying
# module; `load_initial_conditions!` will warn and fall back to an analytical
# profile.

function _apply_initial_condition!(
        model::GeodynamoModel,
        field::Symbol,
        ic::FileIC
)
    f = _get_field(model, field)
    InitialConditions.load_initial_conditions!(f, field, ic.file_path)
    return model
end

# --- ZeroIC --------------------------------------------------------------------
#
# Fields are zero-initialised by `initialize_solver_state`, so this is a no-op.

function _apply_initial_condition!(
        model::GeodynamoModel,
        field::Symbol,
        ::ZeroIC
)
    return model
end

# ── Oceananigans-style direct values: number / function / array ──────────────
#
# Scalar fields only (temperature, composition). The value is realized on the
# field's physical grid, then transformed to spectral with the existing
# analysis path. Vector fields (velocity, magnetic) keep descriptor ICs — a
# pointwise function is ambiguous for a toroidal/poloidal decomposition.

const _SCALAR_IC_FIELDS = (:temperature, :composition)

function _check_scalar_ic_field(field::Symbol)
    field in _SCALAR_IC_FIELDS || throw(ArgumentError(
        "set!: direct values (numbers, functions, arrays) are only supported " *
        "for scalar fields $(_SCALAR_IC_FIELDS); :$field needs a descriptor " *
        "IC (RandomPerturbation, AnalyticIC, FileIC, ZeroIC) because of its " *
        "toroidal–poloidal decomposition"))
    return nothing
end

# Fill the field's main physical storage from fn(r, θ, φ) at the LOCAL grid
# points (works under MPI: axes_local gives this rank's global index ranges),
# then run the analysis transform into the spectral field.
function _set_scalar_from_function!(model::GeodynamoModel, field::Symbol, fn)
    f = _get_field(model, field)
    phys = get_main_physical_field(f)
    backend = model.state.backend
    cfg = backend.shtns_config
    domain = backend.outer_core_domain
    θs = cfg.theta_grid
    φs = cfg.phi_grid
    data = parent(phys.data)
    ax = phys.pencil.axes_local
    for (kl, kg) in enumerate(ax[3])
        for (jl, jg) in enumerate(ax[2])
            for (il, ig) in enumerate(ax[1])
                data[il, jl, kl] = fn(domain.r[kg, 4], θs[ig], φs[jg])
            end
        end
    end
    shtnskit_physical_to_spectral!(phys, f.spectral)
    return model
end

function _apply_initial_condition!(model::GeodynamoModel, field::Symbol, value::Real)
    _check_scalar_ic_field(field)
    return _set_scalar_from_function!(model, field, (r, θ, φ) -> Float64(value))
end

function _apply_initial_condition!(model::GeodynamoModel, field::Symbol, fn::Function)
    _check_scalar_ic_field(field)
    return _set_scalar_from_function!(model, field, fn)
end

function _apply_initial_condition!(model::GeodynamoModel, field::Symbol,
        arr::AbstractArray{<:Real, 3})
    _check_scalar_ic_field(field)
    f = _get_field(model, field)
    phys = get_main_physical_field(f)
    data = parent(phys.data)
    size(arr) == size(data) || throw(ArgumentError(
        "set!: array size $(size(arr)) does not match the local physical grid " *
        "$(size(data)) for field :$field"))
    copyto!(data, arr)
    shtnskit_physical_to_spectral!(phys, f.spectral)
    return model
end

# --- Catch-all for unknown IC types --------------------------------------------

function _apply_initial_condition!(
        model::GeodynamoModel,
        field::Symbol,
        ic
)
    throw(ArgumentError(
        "Unrecognised initial condition type $(typeof(ic)).  " *
        "Expected one of: RandomPerturbation, AnalyticIC, FileIC, ZeroIC."))
end
