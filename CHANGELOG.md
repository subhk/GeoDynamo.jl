# Changelog

## 2.0.0

SciML-style cleanup. **Breaking**: several public identifiers were renamed from
Unicode "script" glyphs to ASCII. Conventional math Unicode that aids
readability (`α β η κ ν`, `θ φ`, `∇θ ∇φ`, `∂r ∂²r`, `r⁻¹ r⁻²`, `nₙ nₙ₋₁ uₙ`) is
intentionally retained.

### Breaking — renamed public identifiers
- `SHTnsMagneticFields` / `SHTnsVelocityFields` fields:
  - `𝒯 → toroidal`, `𝒫 → poloidal`
  - `nlᵀ → nl_toroidal`, `nlᴾ → nl_poloidal` (and `prev_nl_*`)
  - inner core: `𝒯ⁱᶜ → toroidal_ic`, `𝒫ⁱᶜ → poloidal_ic`
- Domain accessors: `𝒟ᵒᶜ → outer_core_domain`, `𝒟ⁱᶜ → inner_core_domain`
- Harmonic-degree factor field: `ℓ_factors → l_factors`
- Keyword argument: `last_Δt → last_dt`
- Internal/loop usage: `ℓ → l`, `Δt → dt` throughout

### Non-breaking
- Codebase auto-formatted with `JuliaFormatter` (`SciMLStyle`); `.JuliaFormatter.toml`
  added. Static source-contract tests made whitespace-insensitive so future
  formatting does not break them.

### Migration
Replace field/keyword accesses accordingly, e.g.:
```julia
magnetic.𝒯        →  magnetic.toroidal
velocity.𝒫        →  velocity.poloidal
Clock(; last_Δt=…) →  Clock(; last_dt=…)
```

## 1.0.10 and earlier
See git history.
