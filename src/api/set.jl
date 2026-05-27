"""
    set!(model::GeodynamoModel; field = ic, ...)

Oceananigans-canonical initial-condition setter. Each keyword names a field
(`velocity`, `temperature`, `magnetic`, `composition`) and its value is an IC
descriptor (`RandomPerturbation`, `AnalyticIC`, `FileIC`, `ZeroIC`). Dispatches
each to `set_initial_condition!`. Returns `model`.

```julia
set!(model; temperature = RandomPerturbation(amplitude=0.1, lmax=10),
            magnetic    = AnalyticIC(:dipole; amplitude=1.0))
```
"""
function set!(model::GeodynamoModel; kwargs...)
    for (field, ic) in kwargs
        set_initial_condition!(model, field, ic)
    end
    return model
end
