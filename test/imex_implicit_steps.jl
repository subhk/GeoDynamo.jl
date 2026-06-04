using Test
using LinearAlgebra

# Coverage for the matrix-embedded implicit-step solvers in src/timestep/imex.jl:
#   * _solver_solve_scalar_implicit_step!         (core scalar solve)
#   * solver_solve_temperature_implicit_step!     (scalar wrapper)
#   * solver_solve_composition_implicit_step!     (scalar wrapper)
#   * solver_solve_velocity_implicit_step!        (toroidal/poloidal)
#   * solver_solve_magnetic_implicit_step!        (toroidal/poloidal, inner-increment)
#
# Each solver, per locally-owned spherical-harmonic mode l, solves the banded
# system  system_matrix[l] · x = b'  where b' is the gathered RHS with its two
# endpoint rows overwritten by the boundary values (homogeneous when no BC vector
# is supplied). The Dirichlet boundary rows embedded by create_scalar_matrices
# make rows 1 and nr identity rows, so b'[1]=x[1] and b'[nr]=x[nr].
#
# The test pins this with a fully-controlled identity: build a KNOWN per-mode
# solution x, form the exact RHS  b = system_matrix · x  via the independent
# banded matvec apply_banded_full!, run the solver, and require the recovered
# field to equal x to machine precision (bar is 1e-8). It exercises both the
# homogeneous-endpoint path and the boundary-injection branches (Dirichlet inner
# value; magnetic mag_bc_inner − prev_bc_inner increment).
#
# Skips: the toroidal-velocity inner-core rotation correction (needs a live
# rot_omega + current_field coupling) is not isolated here; the homogeneous
# toroidal/poloidal paths and the magnetic inner-increment branch ARE covered.
# Single-rank only — the per-mode solve is radial-local and comm-free, so one
# rank exercises the full numerical path (the CNAB2 equivalence suite covers the
# multi-rank partitioning separately).

const _GI = GeoDynamo

# Build an ImplicitMatrixSet (solver layer) plus a fresh config/domain.
function _imex_setup(; lmax, nr, dt = 1.0e-3, theta = 0.5, diffusivity = 0.7,
        bc_code = 1)
    nlat = max(lmax + 2, 10)
    nlon = max(2lmax + 1, 12)
    cfg = _GI.create_shtnskit_config(
        lmax = lmax, mmax = lmax, nlat = nlat, nlon = nlon,
        nr = nr, optimize_decomp = false)
    dom = _GI.create_radial_domain(nr)
    matrices = _GI.create_temperature_matrices(
        cfg, dom, diffusivity, dt; temperature_bc_code = bc_code, theta = theta)
    smats = _GI.Solver.ImplicitMatrixSet(matrices)
    return cfg, dom, smats
end

# Fill `rhs` so that  system_matrix · x = rhs  for a known per-mode solution x.
# Endpoint values of x are taken from inner_vals/outer_vals (mode-indexed); the
# Dirichlet rows make the resulting RHS endpoints equal those same values.
# Returns the stored solutions keyed by lm index (real, imag).
function _set_known_system_rhs!(rhs, cfg, smats, nr;
        inner_vals = nothing, outer_vals = nothing,
        salt_r = 0.07, salt_i = 0.06)
    rr = parent(rhs.data_real)
    ri = parent(rhs.data_imag)
    xr_by_lm = Dict{Int, Vector{Float64}}()
    xi_by_lm = Dict{Int, Vector{Float64}}()
    for lm in _GI.local_spectral_mode_indices(cfg)
        slot = _GI.local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue
        l = cfg.l_values[lm]
        idx = get(smats.lookup, l, nothing)
        idx === nothing && continue
        Asys = smats.system_matrices[idx]
        xr = [sinpi(salt_r * (lm + 1) + 0.05 * r) for r in 1:nr]
        xi = [cospi(salt_i * (lm + 2) + 0.04 * r) for r in 1:nr]
        xr[1] = inner_vals === nothing ? 0.0 : inner_vals[lm]
        xr[nr] = outer_vals === nothing ? 0.0 : outer_vals[lm]
        xi[1] = 0.0
        xi[nr] = 0.0
        xr_by_lm[lm] = xr
        xi_by_lm[lm] = xi
        br = zeros(Float64, nr)
        bi = zeros(Float64, nr)
        _GI.apply_banded_full!(br, Asys, xr)
        _GI.apply_banded_full!(bi, Asys, xi)
        for r in 1:nr
            _GI.set_local_spectral_value!(rr, slot, r, br[r])
            _GI.set_local_spectral_value!(ri, slot, r, bi[r])
        end
    end
    return xr_by_lm, xi_by_lm
end

# Max abs error between solved field and the stored known solutions.
function _solution_maxerr(sol, cfg, smats, nr, xr_by_lm, xi_by_lm)
    sr = parent(sol.data_real)
    si = parent(sol.data_imag)
    maxerr = 0.0
    for lm in _GI.local_spectral_mode_indices(cfg)
        slot = _GI.local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue
        l = cfg.l_values[lm]
        haskey(smats.lookup, l) || continue
        xr = xr_by_lm[lm]
        xi = xi_by_lm[lm]
        for r in 1:nr
            maxerr = max(maxerr, abs(_GI.local_spectral_value(sr, slot, r) - xr[r]))
            maxerr = max(maxerr, abs(_GI.local_spectral_value(si, slot, r) - xi[r]))
        end
    end
    return maxerr
end

@testset "imex matrix-embedded implicit-step solvers" begin
    nr = 12
    lmax = 4

    @testset "ImplicitMatrixSet structure" begin
        cfg, _, smats = _imex_setup(lmax = lmax, nr = nr)
        # One operator per unique spherical-harmonic degree 0..lmax.
        @test smats isa _GI.ImplicitMatrixSet{Float64}
        @test length(smats.system_matrices) == lmax + 1
        @test length(smats.factorizations) == lmax + 1
        @test sort(smats.l_values) == collect(0:lmax)
        @test all(A -> A.size == nr, smats.system_matrices)
        @test smats.theta == 0.5
    end

    @testset "temperature implicit solve recovers known solution" begin
        cfg, _, smats = _imex_setup(lmax = lmax, nr = nr)
        pencil = cfg.pencils.spec
        rhs = _GI.create_shtns_spectral_field(Float64, cfg,
            _GI.create_radial_domain(nr), pencil)
        sol = _GI.create_shtns_spectral_field(Float64, cfg,
            _GI.create_radial_domain(nr), pencil)
        xr, xi = _set_known_system_rhs!(rhs, cfg, smats, nr)
        _GI.Solver.solver_solve_temperature_implicit_step!(sol, rhs, smats)
        @test _solution_maxerr(sol, cfg, smats, nr, xr, xi) < 1.0e-8
    end

    @testset "composition implicit solve recovers known solution" begin
        cfg, _, smats = _imex_setup(lmax = lmax, nr = nr)
        pencil = cfg.pencils.spec
        rhs = _GI.create_shtns_spectral_field(Float64, cfg,
            _GI.create_radial_domain(nr), pencil)
        sol = _GI.create_shtns_spectral_field(Float64, cfg,
            _GI.create_radial_domain(nr), pencil)
        xr, xi = _set_known_system_rhs!(rhs, cfg, smats, nr; salt_r = 0.09, salt_i = 0.05)
        _GI.Solver.solver_solve_composition_implicit_step!(sol, rhs, smats)
        @test _solution_maxerr(sol, cfg, smats, nr, xr, xi) < 1.0e-8
    end

    @testset "temperature solve with Dirichlet boundary injection" begin
        cfg, _, smats = _imex_setup(lmax = lmax, nr = nr)
        pencil = cfg.pencils.spec
        rhs = _GI.create_shtns_spectral_field(Float64, cfg,
            _GI.create_radial_domain(nr), pencil)
        sol = _GI.create_shtns_spectral_field(Float64, cfg,
            _GI.create_radial_domain(nr), pencil)
        nlm = cfg.nlm
        bc_in = [0.1 * lm + 1.0 for lm in 1:nlm]
        bc_out = [-0.2 * lm + 2.0 for lm in 1:nlm]
        xr, xi = _set_known_system_rhs!(
            rhs, cfg, smats, nr; inner_vals = bc_in, outer_vals = bc_out)
        # Pass the SAME BC vectors so the solver restores the endpoints it zeroes.
        _GI.Solver.solver_solve_temperature_implicit_step!(
            sol, rhs, smats; bc_inner = bc_in, bc_outer = bc_out)
        @test _solution_maxerr(sol, cfg, smats, nr, xr, xi) < 1.0e-8

        # Endpoints must equal the prescribed Dirichlet values exactly.
        sr = parent(sol.data_real)
        endpoint_err = 0.0
        for lm in _GI.local_spectral_mode_indices(cfg)
            slot = _GI.local_spectral_storage_slot(cfg, lm)
            slot === nothing && continue
            haskey(smats.lookup, cfg.l_values[lm]) || continue
            endpoint_err = max(endpoint_err,
                abs(_GI.local_spectral_value(sr, slot, 1) - bc_in[lm]))
            endpoint_err = max(endpoint_err,
                abs(_GI.local_spectral_value(sr, slot, nr) - bc_out[lm]))
        end
        @test endpoint_err < 1.0e-8
    end

    @testset "velocity poloidal implicit solve recovers known solution" begin
        cfg, _, smats = _imex_setup(lmax = lmax, nr = nr)
        pencil = cfg.pencils.spec
        rhs = _GI.create_shtns_spectral_field(Float64, cfg,
            _GI.create_radial_domain(nr), pencil)
        sol = _GI.create_shtns_spectral_field(Float64, cfg,
            _GI.create_radial_domain(nr), pencil)
        xr, xi = _set_known_system_rhs!(rhs, cfg, smats, nr; salt_r = 0.03, salt_i = 0.05)
        _GI.Solver.solver_solve_velocity_implicit_step!(sol, rhs, smats, :poloidal)
        @test _solution_maxerr(sol, cfg, smats, nr, xr, xi) < 1.0e-8
    end

    @testset "velocity toroidal implicit solve (homogeneous endpoints)" begin
        cfg, _, smats = _imex_setup(lmax = lmax, nr = nr)
        pencil = cfg.pencils.spec
        rhs = _GI.create_shtns_spectral_field(Float64, cfg,
            _GI.create_radial_domain(nr), pencil)
        sol = _GI.create_shtns_spectral_field(Float64, cfg,
            _GI.create_radial_domain(nr), pencil)
        # rot_omega = 0 ⇒ toroidal endpoints stay homogeneous (no rotation term).
        xr, xi = _set_known_system_rhs!(rhs, cfg, smats, nr; salt_r = 0.04, salt_i = 0.06)
        _GI.Solver.solver_solve_velocity_implicit_step!(
            sol, rhs, smats, :toroidal; velocity_bc_code = 1, rot_omega = 0.0)
        @test _solution_maxerr(sol, cfg, smats, nr, xr, xi) < 1.0e-8
    end

    @testset "magnetic toroidal implicit solve recovers known solution" begin
        cfg, _, smats = _imex_setup(lmax = lmax, nr = nr)
        pencil = cfg.pencils.spec
        rhs = _GI.create_shtns_spectral_field(Float64, cfg,
            _GI.create_radial_domain(nr), pencil)
        sol = _GI.create_shtns_spectral_field(Float64, cfg,
            _GI.create_radial_domain(nr), pencil)
        xr, xi = _set_known_system_rhs!(rhs, cfg, smats, nr; salt_r = 0.05, salt_i = 0.04)
        _GI.Solver.solver_solve_magnetic_implicit_step!(sol, rhs, smats, :toroidal)
        @test _solution_maxerr(sol, cfg, smats, nr, xr, xi) < 1.0e-8
    end

    @testset "magnetic poloidal solve with inner-increment injection" begin
        cfg, _, smats = _imex_setup(lmax = lmax, nr = nr)
        pencil = cfg.pencils.spec
        rhs = _GI.create_shtns_spectral_field(Float64, cfg,
            _GI.create_radial_domain(nr), pencil)
        sol = _GI.create_shtns_spectral_field(Float64, cfg,
            _GI.create_radial_domain(nr), pencil)
        nlm = cfg.nlm
        mag_in = [0.3 * lm + 0.5 for lm in 1:nlm]
        prev_in = [0.1 * lm + 0.2 for lm in 1:nlm]
        eff_inner = mag_in .- prev_in   # the effective inner endpoint value
        # Build known solutions whose inner endpoint is the effective increment.
        xr, xi = _set_known_system_rhs!(rhs, cfg, smats, nr; inner_vals = eff_inner)
        _GI.Solver.solver_solve_magnetic_implicit_step!(
            sol, rhs, smats, :poloidal; mag_bc_inner = mag_in, prev_bc_inner = prev_in)
        @test _solution_maxerr(sol, cfg, smats, nr, xr, xi) < 1.0e-8
    end
end
