using Test
using MPI

const FINALIZE_MPI_CNAB2 = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

# Distributed-equivalence contract for the CNAB2 RHS assembly.
#
# Each spherical-harmonic mode (l,m) owns a COMPLETE radial profile on a single
# rank (the spectral pencil distributes (l,m) and keeps r LOCAL). The CNAB2 RHS
#     b = (mass/dt) u + (3/2) N - (1/2) N_prev + (1-θ) (L_l · u)
# is per-mode and couples only in r, so its assembly needs NO inter-rank
# communication: the owning rank already holds everything.
#
# This test pins that contract. It fills u/N/N_prev with values that depend only
# on the GLOBAL mode index and radius (decomposition-independent), builds the RHS,
# and checks every owned mode against a direct local recomputation of the formula
# (the linear operator applied via the independent `BandedMatrix *`). Run on >=2
# ranks it also verifies the owned modes partition the full nlm set exactly.
#
# Passing on multiple ranks is the proof that any per-mode Allreduce in the
# assembly is redundant — the distributed result equals the purely-local one.

@testset "CNAB2 + ERK2 distributed equivalence (radial-local ⇒ comm-free)" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping CNAB2 RHS equivalence test"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    lmax = 6
    mmax = 6
    nlat = max(lmax + 2, 10)
    nlon = max(2lmax + 1, 16)
    nr   = 16

    cfg = GeoDynamo.create_shtnskit_config(lmax=lmax, mmax=mmax, nlat=nlat, nlon=nlon, nr=nr)
    dom = GeoDynamo.create_radial_domain(nr)

    dt          = 1.0e-3
    θ           = 0.5            # nonzero (1-θ) ⇒ the explicit linear term is active
    diffusivity = 0.7
    matrices = GeoDynamo.create_temperature_matrices(cfg, dom, diffusivity, dt;
                                                     temperature_bc_code=1, theta=θ)
    # The live solver step consumes the solver-layer ImplicitMatrixSet (same
    # operators, BandedOperator-typed). Reference uses the original BandedMatrix.
    smats = GeoDynamo.Solver.ImplicitMatrixSet(matrices)

    pencil = cfg.pencils.spec
    u   = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, pencil)
    nl  = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, pencil)
    pv  = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, pencil)
    rhs = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, pencil)

    # Deterministic, decomposition-independent fill: value = f(global lm_idx, r).
    function fill_det!(field, salt)
        fr = parent(field.data_real)
        fi = parent(field.data_imag)
        for lm in GeoDynamo.local_spectral_mode_indices(cfg)
            slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            for r in 1:nr
                GeoDynamo.set_local_spectral_value!(fr, slot, r, sinpi(0.05 * (lm + salt) + 0.013 * r))
                GeoDynamo.set_local_spectral_value!(fi, slot, r, cospi(0.04 * (lm - salt) + 0.017 * r))
            end
        end
        return field
    end
    fill_det!(u, 0); fill_det!(nl, 1); fill_det!(pv, 2)

    GeoDynamo.Solver.solver_build_rhs_cnab2!(rhs, u, nl, pv, dt, smats)

    inv_dt        = 1.0 / dt
    linear_weight = 1.0 - θ

    ur = parent(u.data_real);   ui = parent(u.data_imag)
    nr_ = parent(nl.data_real); ni_ = parent(nl.data_imag)
    pr = parent(pv.data_real);  pi_ = parent(pv.data_imag)
    rr = parent(rhs.data_real); ri = parent(rhs.data_imag)

    maxerr  = 0.0
    n_owned = 0
    uvec    = zeros(Float64, nr)

    for lm in GeoDynamo.local_spectral_mode_indices(cfg)
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue
        l   = cfg.l_values[lm]
        idx = get(matrices.lookup, l, nothing)
        idx === nothing && continue
        n_owned += 1
        L = matrices.linear_matrices[idx]

        # Real component
        for r in 1:nr
            uvec[r] = GeoDynamo.local_spectral_value(ur, slot, r)
        end
        Lu = L * uvec   # independent banded matvec (different code path than build_rhs)
        for r in 1:nr
            ref = inv_dt * GeoDynamo.local_spectral_value(ur, slot, r) +
                  1.5    * GeoDynamo.local_spectral_value(nr_, slot, r) -
                  0.5    * GeoDynamo.local_spectral_value(pr, slot, r) +
                  linear_weight * Lu[r]
            maxerr = max(maxerr, abs(GeoDynamo.local_spectral_value(rr, slot, r) - ref))
        end

        # Imag component
        for r in 1:nr
            uvec[r] = GeoDynamo.local_spectral_value(ui, slot, r)
        end
        Lu = L * uvec
        for r in 1:nr
            ref = inv_dt * GeoDynamo.local_spectral_value(ui, slot, r) +
                  1.5    * GeoDynamo.local_spectral_value(ni_, slot, r) -
                  0.5    * GeoDynamo.local_spectral_value(pi_, slot, r) +
                  linear_weight * Lu[r]
            maxerr = max(maxerr, abs(GeoDynamo.local_spectral_value(ri, slot, r) - ref))
        end
    end

    @test maxerr < 1.0e-10

    # Owned modes must partition the full spectral set exactly (none dropped or
    # double-counted) — the property the global-loop + Allreduce was protecting.
    comm   = GeoDynamo.get_comm()
    nprocs = GeoDynamo.get_nprocs()
    total_owned = nprocs > 1 ? MPI.Allreduce(n_owned, MPI.SUM, comm) : n_owned
    @test total_owned == cfg.nlm

    # ---- Legacy GeoDynamo.build_rhs_cnab2! (SHTnsImplicitMatrices) must assemble
    # every owned mode correctly. Guards both the redundant per-mode Allreduce and
    # an ownership bug where modes were selected by the rectangular l-index range
    # instead of the compact lm index (silently dropping modes lm > lmax+1). ----
    rhs_legacy = GeoDynamo.create_shtns_spectral_field(Float64, cfg, dom, pencil)
    GeoDynamo.build_rhs_cnab2!(rhs_legacy, u, nl, pv, dt, matrices)
    lgr = parent(rhs_legacy.data_real); lgi = parent(rhs_legacy.data_imag)
    legacy_maxerr = 0.0

    for lm in GeoDynamo.local_spectral_mode_indices(cfg)
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue
        l   = cfg.l_values[lm]
        idx = get(matrices.lookup, l, nothing)
        idx === nothing && continue
        L = matrices.linear_matrices[idx]
        for r in 1:nr
            uvec[r] = GeoDynamo.local_spectral_value(ur, slot, r)
        end
        Lu = L * uvec
        for r in 1:nr
            ref = inv_dt * GeoDynamo.local_spectral_value(ur, slot, r) +
                  1.5 * GeoDynamo.local_spectral_value(nr_, slot, r) -
                  0.5 * GeoDynamo.local_spectral_value(pr, slot, r) +
                  linear_weight * Lu[r]
            legacy_maxerr = max(legacy_maxerr, abs(GeoDynamo.local_spectral_value(lgr, slot, r) - ref))
        end
        for r in 1:nr
            uvec[r] = GeoDynamo.local_spectral_value(ui, slot, r)
        end
        Lu = L * uvec
        for r in 1:nr
            ref = inv_dt * GeoDynamo.local_spectral_value(ui, slot, r) +
                  1.5 * GeoDynamo.local_spectral_value(ni_, slot, r) -
                  0.5 * GeoDynamo.local_spectral_value(pi_, slot, r) +
                  linear_weight * Lu[r]
            legacy_maxerr = max(legacy_maxerr, abs(GeoDynamo.local_spectral_value(lgi, slot, r) - ref))
        end
    end

    @test legacy_maxerr < 1.0e-10

    # ---- ERK2 stage preparation obeys the same radial-local ⇒ comm-free contract ----
    # prepare_solver_erk2_field! applies the per-degree stage operators (E, φ₁) to
    # each mode's full local radial profile. Check every owned mode against a direct
    # recompute using the cache's own dense operators (bc_spec=nothing ⇒ zero
    # endpoints on the provisional stage).
    erk2_cache = GeoDynamo.create_erk2_cache(Float64, cfg, dom, diffusivity, dt)
    erk2_buf   = GeoDynamo.ERK2FieldBuffers(u, nl, erk2_cache)
    GeoDynamo.erk2_prepare_field!(erk2_buf, u, nl, erk2_cache, cfg, dt)

    lin_r = erk2_buf.linear_real; lin_i = erk2_buf.linear_imag
    k1r   = erk2_buf.k1_real;     k1i   = erk2_buf.k1_imag
    str   = erk2_buf.stage_real;  sti   = erk2_buf.stage_imag
    half_dt = dt / 2
    erk2_maxerr = 0.0
    uv = zeros(Float64, nr); nv = zeros(Float64, nr)

    for lm in GeoDynamo.local_spectral_mode_indices(cfg)
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue
        l = cfg.l_values[lm]
        ci = get(erk2_buf.cache_lookup, l, 0)
        ci == 0 && continue
        Ef = erk2_cache.E_full[ci]; Eh = erk2_cache.E_half[ci]
        p1f = erk2_cache.phi1_full[ci]; p1h = erk2_cache.phi1_half[ci]

        for (src, lin_b, k1_b, st_b) in ((ur, lin_r, k1r, str), (ui, lin_i, k1i, sti))
            nsrc = src === ur ? nr_ : ni_
            for r in 1:nr
                uv[r] = GeoDynamo.local_spectral_value(src, slot, r)
                nv[r] = GeoDynamo.local_spectral_value(nsrc, slot, r)
            end
            lref = Ef * uv
            kref = p1f * nv
            sref = Eh * uv .+ half_dt .* (p1h * nv)
            sref[1] = 0.0; sref[nr] = 0.0
            for r in 1:nr
                erk2_maxerr = max(erk2_maxerr, abs(GeoDynamo.local_spectral_value(lin_b, slot, r) - lref[r]))
                erk2_maxerr = max(erk2_maxerr, abs(GeoDynamo.local_spectral_value(k1_b, slot, r) - kref[r]))
                erk2_maxerr = max(erk2_maxerr, abs(GeoDynamo.local_spectral_value(st_b, slot, r) - sref[r]))
            end
        end
    end

    @test erk2_maxerr < 1.0e-10

    # ---- ERK2 finalize: run the full prepare→stage→finalize cycle and check the
    # accepted update equals  E·u + dt·φ₁·Nₙ + 2dt·φ₂·(N_stage − Nₙ)  per owned mode
    # (endpoints zeroed). Same radial-local ⇒ comm-free contract as prep. ----
    GeoDynamo.erk2_apply_stage!(erk2_buf, u)             # u ← provisional stage
    GeoDynamo.erk2_store_stage_nonlinear!(erk2_buf, nl)  # buffers.stage_nl ← N_stage
    GeoDynamo.erk2_finalize_field!(erk2_buf, u, erk2_cache, cfg, dt)  # u ← accepted update

    fr_u = parent(u.data_real); fi_u = parent(u.data_imag)
    fin_maxerr = 0.0
    dvec = zeros(Float64, nr)
    for lm in GeoDynamo.local_spectral_mode_indices(cfg)
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue
        l = cfg.l_values[lm]
        ci = get(erk2_buf.cache_lookup, l, 0)
        ci == 0 && continue
        phi2 = erk2_cache.phi2_full[ci]
        for (lin_b, k1_b, ncur_b, snl_b, out_b) in (
                (erk2_buf.linear_real, erk2_buf.k1_real, erk2_buf.n_current_real, erk2_buf.stage_nl_real, fr_u),
                (erk2_buf.linear_imag, erk2_buf.k1_imag, erk2_buf.n_current_imag, erk2_buf.stage_nl_imag, fi_u))
            for r in 1:nr
                dvec[r] = GeoDynamo.local_spectral_value(snl_b, slot, r) -
                          GeoDynamo.local_spectral_value(ncur_b, slot, r)
            end
            corr = phi2 * dvec
            for r in 1:nr
                ref = GeoDynamo.local_spectral_value(lin_b, slot, r) +
                      dt * GeoDynamo.local_spectral_value(k1_b, slot, r) +
                      2 * dt * corr[r]
                (r == 1 || r == nr) && (ref = 0.0)
                fin_maxerr = max(fin_maxerr, abs(GeoDynamo.local_spectral_value(out_b, slot, r) - ref))
            end
        end
    end
    @test fin_maxerr < 1.0e-10

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_CNAB2 && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
