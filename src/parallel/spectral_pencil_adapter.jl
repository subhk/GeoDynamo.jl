# Adapter seams between GeoDynamo's dense (l,m) coefficient matrix (one radial
# level) and the DENSE (lmax+1, mmax+1) matrix SHTnsKit's distributed transform
# consumes/produces. An API spike pinned that dist_analysis returns a dense
# Matrix{ComplexF64} replicated on every rank and dist_synthesis consumes an
# AbstractMatrix — so no spectral PencilArray packing is needed. These are named
# seams (currently identity) kept so the transform call sites read symmetrically
# and so a future distributed-spectral layout (Phase 2) can be slotted in here.

"""
    to_sht_spectral(cfg, alm_mat::AbstractMatrix{ComplexF64})

Hand a full (lmax+1, mmax+1) coefficient matrix to the distributed synthesis. The
matrix is already the dense form `dist_synthesis` expects, so this is the identity.
"""
to_sht_spectral(cfg, alm_mat::AbstractMatrix{ComplexF64}) = alm_mat

"""
    from_sht_spectral(cfg, alm::AbstractMatrix{ComplexF64})

`dist_analysis` already returns the complete dense matrix replicated on every rank,
so no gather is needed — return it (identity). Symmetric with `to_sht_spectral`.
"""
from_sht_spectral(cfg, alm::AbstractMatrix{ComplexF64}) = alm
