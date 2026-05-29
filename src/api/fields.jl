"""
    fields(model::GeodynamoModel)

Return a `NamedTuple` of all model fields:
`(velocity, temperature, magnetic, composition)`. Disabled fields are `nothing`.
"""
function fields(model::GeodynamoModel)
    f = model.state.fields
    return (velocity = f.velocity,
        temperature = f.temperature,
        magnetic = f.magnetic,
        composition = f.composition)
end

"""
    prognostic_fields(model::GeodynamoModel)

Like [`fields`](@ref) but with disabled (`nothing`) fields removed.
"""
function prognostic_fields(model::GeodynamoModel)
    nt = fields(model)
    return (; (k => v for (k, v) in pairs(nt) if v !== nothing)...)
end
