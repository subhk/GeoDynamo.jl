# ================================================================================
# Force-field QST analysis and momentum-equation curl projections (Stage 1 of the
# double-curl poloidal formulation; see
# docs/superpowers/specs/2026-06-10-poloidal-momentum-double-curl-design.md).
# NOT yet wired into the dynamics — verified standalone against an independent
# curl reference in test/force_projection_reference.jl.
# ================================================================================

"""
    force_physical_to_qst!(force, Q, S, T)

Three-component spectral analysis of a (generally non-solenoidal) force field.
`Q` <- scalar analysis of the radial component; `(S, T)` <- spheroidal/toroidal
scalars of the tangential components (the same sphtor analysis the velocity
path uses -- S is what that path stores in its "poloidal" output).
"""
function force_physical_to_qst!(
        force::VectorFieldType{T},
        Q::SpectralFieldType{T},
        S::SpectralFieldType{T},
        T_out::SpectralFieldType{T}
) where {T}
    scalar_physical_to_spectral!(force.r_component, Q)
    vector_physical_to_spectral!(force, T_out, S)
    return Q, S, T_out
end
