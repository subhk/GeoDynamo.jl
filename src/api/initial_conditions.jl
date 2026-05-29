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

Superimpose random spectral perturbations up to degree `lmax` on a field.

- `amplitude` – overall scale of the perturbation (required)
- `lmax`      – maximum spherical-harmonic degree to excite (required)
- `domain`    – radial domain, forwarded to the underlying randomizer so that
                ball-regularity conditions are enforced when r=0 is included
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
- `ic`     – an IC descriptor: `RandomPerturbation`, `AnalyticIC`, `FileIC`, or
             `ZeroIC`

# Examples
```julia
set_initial_condition!(model, :temperature, RandomPerturbation(amplitude=0.1, lmax=10))
set_initial_condition!(model, :magnetic,    AnalyticIC(:dipole; amplitude=1.0))
set_initial_condition!(model, :velocity,    FileIC("/path/to/checkpoint.nc"))
set_initial_condition!(model, :composition, ZeroIC())
```
"""
function set_initial_condition! end

# --- RandomPerturbation --------------------------------------------------------

function set_initial_condition!(
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

function set_initial_condition!(
        model::GeodynamoModel,
        field::Symbol,
        ic::AnalyticIC
)
    f = _get_field(model, field)
    InitialConditions.set_analytical_initial_conditions!(f, field, ic.pattern;
        amplitude = ic.amplitude,
        ic.parameters...
    )
    return model
end

# --- FileIC --------------------------------------------------------------------
#
# `load_initial_conditions!(field, field_type, file_path)` accepts a `field_type`
# Symbol as its second argument, so we forward the `field` symbol directly.
# At the time of writing, NetCDF loading is not yet implemented in the underlying
# module; `load_initial_conditions!` will warn and fall back to an analytical
# profile.

function set_initial_condition!(
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

function set_initial_condition!(
        model::GeodynamoModel,
        field::Symbol,
        ::ZeroIC
)
    return model
end

# --- Catch-all for unknown IC types --------------------------------------------

function set_initial_condition!(
        model::GeodynamoModel,
        field::Symbol,
        ic
)
    throw(ArgumentError(
        "Unrecognised initial condition type $(typeof(ic)).  " *
        "Expected one of: RandomPerturbation, AnalyticIC, FileIC, ZeroIC."))
end
