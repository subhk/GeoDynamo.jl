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

@testset "CNAB2 RHS distributed equivalence (radial-local ⇒ comm-free)" begin
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

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_CNAB2 && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
