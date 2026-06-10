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
    # raw_spheroidal: a force is not solenoidal — we want the RAW tangential
    # sphtor scalars (S, T), not the Q-based poloidal recovery the default
    # (velocity) analysis performs under the Stage-2 convention.
    vector_physical_to_spectral!(force, T_out, S; raw_spheroidal = true)
    return Q, S, T_out
end

"""
    force_curl_projections!(R_tor, R_pol, Q, S, T_in, domain)

Momentum-equation projections of a force field from its QST scalars
(see `force_physical_to_qst!`):

    R_tor[lm](r) = [r̂·∇×F]_lm          = −λ/r · T_lm(r)
    R_pol[lm](r) = [r̂·∇×∇×F]_lm = (λ/r²) · (Q_lm(r) − ∂_r(r · S_lm(r)))

where λ = l(l+1). Verified against closed-form analytic references
(pure-radial, spheroidal, toroidal families) and an exact single-curl
numerical reference in test/force_projection_reference.jl.
"""
function force_curl_projections!(
        R_tor::SpectralFieldType{T},
        R_pol::SpectralFieldType{T},
        Q::SpectralFieldType{T},
        S::SpectralFieldType{T},
        T_in::SpectralFieldType{T},
        domain::RadialDomainType
) where {T}
    cfg      = Q.config
    D1       = create_derivative_matrix(T, 1, domain)
    nr       = domain.N
    r_range  = local_range(Q.pencil, 3)
    nr_local = length(r_range)

    rS  = Vector{T}(undef, nr)
    drS = Vector{T}(undef, nr)

    for (dst_t, dst_p, src_q, src_s, src_t) in (
        (parent(R_tor.data_real), parent(R_pol.data_real),
         parent(Q.data_real),     parent(S.data_real),     parent(T_in.data_real)),
        (parent(R_tor.data_imag), parent(R_pol.data_imag),
         parent(Q.data_imag),     parent(S.data_imag),     parent(T_in.data_imag)),
    )
        for lm in 1:cfg.nlm
            slot = local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            l = cfg.l_values[lm]
            λ = T(l * (l + 1))
            if l == 0
                for lr in 1:nr_local
                    set_local_spectral_value!(dst_t, slot, lr, zero(T))
                    set_local_spectral_value!(dst_p, slot, lr, zero(T))
                end
                continue
            end
            # Build r·S radial profile on the full global grid
            fill!(rS, zero(T))
            for r_idx in r_range
                lr = r_idx - first(r_range) + 1
                rS[r_idx] = T(domain.r[r_idx, 4]) * local_spectral_value(src_s, slot, lr)
            end
            mul!(drS, D1, rS)
            # Write projections
            for r_idx in r_range
                lr   = r_idx - first(r_range) + 1
                r    = T(domain.r[r_idx, 4])
                tval = local_spectral_value(src_t, slot, lr)
                qval = local_spectral_value(src_q, slot, lr)
                set_local_spectral_value!(dst_t, slot, lr, -λ / r * tval)
                set_local_spectral_value!(dst_p, slot, lr, λ / r^2 * (qval - drS[r_idx]))
            end
        end
    end
    return R_tor, R_pol
end
