# ================================================================================
# Review batch G — control-plane I/O guards + topography correctness
# ================================================================================
#
# Serial regressions for the 2026-08-18 review. The multi-rank half of the same
# batch lives in `test/mpi_control_plane_invariants.jl` (run by
# `test/run_mpi_control_plane.sh`), because the defects there are hangs rather
# than failures and need >= 2 ranks to bite.
# ================================================================================

using Test
using GeoDynamo

# Shorthand for the internal topography submodule (matches test/topography_data.jl)
const topo = GeoDynamo.bcs.topography

# Stand-in for `SolverFields`: the point is that every slot is UNION-typed, so
# `hasfield(typeof(x), :magnetic)` answers true whether or not a magnetic field is
# actually present. Declared at top level because Julia only allows struct
# definitions there.
mutable struct _FieldsG{V, M, T, C}
    velocity::V
    magnetic::M
    temperature::T
    composition::C
end

@testset "Review batch G fixes" begin

    # ── G1: a rank-0-only directory scan must not throw into a collective ──────
    @testset "_scan_output_count survives an unreadable directory" begin
        # `_scan_output_count` runs on rank 0 ONLY, immediately in front of an
        # `MPI.Bcast!` (io/restart.jl `_persisted_output_count`,
        # api/output_writers.jl `_existing_writer_count`). `isdir` passing does not
        # mean `readdir` succeeds: a stale NFS handle, an EIO, or a directory whose
        # execute bit was dropped all raise `SystemError` from `readdir`. Rank 0 then
        # unwinds out of the collective alone and every other rank blocks in the
        # broadcast forever. The same class of failure one function below is already
        # guarded (`_restart_path_for_all_ranks` wraps `find_restart_files`).
        dir = mktempdir()
        touch(joinpath(dir, "geodynamo_shell_hist_7.nc"))
        @test GeoDynamo._scan_output_count(dir, "geodynamo", :shell, "hist") == 7

        chmod(dir, 0o000)
        try
            # Root ignores the mode bits, so only assert when the mode really bites.
            if !isreadable(dir)
                @test GeoDynamo._scan_output_count(dir, "geodynamo", :shell, "hist") == 0
            end
        finally
            chmod(dir, 0o700)
        end

        # A directory that is not there at all was already handled; keep it pinned.
        @test GeoDynamo._scan_output_count(joinpath(dir, "nope"), "geodynamo", :shell,
            "hist") == 0
    end

    # ── G9: index_to_lm must invert lm_to_index at TRUNCATED mmax too ─────────
    @testset "index_to_lm inverts lm_to_index when mmax < lmax" begin
        # `lm_to_index` walks the truncated triangle (degree l' contributes
        # min(l',mmax)+1 slots) and its own docstring says the classic full-triangle
        # formula is wrong once mmax < lmax. `index_to_lm` nevertheless inverted
        # exactly that full-triangle formula and took no mmax at all, so the two
        # disagreed for every index past l = mmax. `stefan_condition.jl` labels
        # `heat_flux_*` / `normal_velocity` entries with one and re-reads them with the
        # other, so the Stefan flux was attributed to the wrong modes.
        lmax, mmax = 8, 4
        for l in 0:lmax, m in 0:min(l, mmax)
            idx = topo.lm_to_index(l, m, lmax, mmax)
            @test topo.index_to_lm(idx, lmax, mmax) == (l, m)
        end
        # the untruncated default must keep behaving exactly as before
        for l in 0:6, m in 0:l
            @test topo.index_to_lm(topo.lm_to_index(l, m, 6), 6) == (l, m)
        end
    end

    # ── G6: a saved mmax must survive the round trip ──────────────────────────
    @testset "load_topography_from_file honours the stored mmax" begin
        # `save_topography_to_file` writes an `mmax` attribute; the loader read only
        # `lmax` and rebuilt the field as TopographyField(lmax, lmax, ...). A field
        # saved with mmax < lmax therefore came back with a LARGER nlm, and the
        # coefficient vector — stored positionally — was reinterpreted against the
        # untruncated layout, so every slot past l = mmax named the wrong harmonic.
        src = topo.TopographyField{Float64}(8, 4, 1.0, GeoDynamo.bcs.OUTER_BOUNDARY)
        src.coeffs_real[src.nlm] = 0.25          # last slot: the one that moves
        tmp = tempname() * ".nc"
        try
            topo.save_topography_to_file(src, tmp)
            back = GeoDynamo.load_topography_from_file(tmp, GeoDynamo.bcs.OUTER_BOUNDARY)
            @test back.lmax == 8
            @test back.mmax == 4
            @test back.nlm == src.nlm
            @test back.coeffs_real[back.nlm] ≈ 0.25
        finally
            rm(tmp, force = true)
        end
    end

    # ── G7: the array loader must build the field at the config's mmax ────────
    @testset "load_topography_from_array truncates mmax to the config" begin
        # The field was built with mmax = lmax_use while the storage loop walked
        # `m in 0:min(l, config.mmax)` with a running counter, so with config.mmax <
        # lmax the counter desynchronised from the field's own layout at l = mmax+1
        # and scrambled every higher-degree coefficient.
        cfg = (lmax = 8, mmax = 4)               # transform is irrelevant here
        h = zeros(Float64, 4, 8)
        field = topo.load_topography_from_array(h, 1.0,
            GeoDynamo.bcs.OUTER_BOUNDARY, cfg)
        @test field.lmax == 8
        @test field.mmax == 4
        @test field.nlm == topo.TopographyField{Float64}(8, 4, 1.0,
            GeoDynamo.bcs.OUTER_BOUNDARY).nlm
    end

    # ── G8: random topography must be a real field, and reproducible ──────────
    @testset "create_random_topography leaves m = 0 real" begin
        # A random phase was applied to EVERY mode including m = 0, so coeffs_imag
        # was non-zero at m = 0 and the synthesised h(θ,φ) was not real-valued.
        f = topo.create_random_topography(l -> 1.0 / max(l, 1)^2, 1.0,
            GeoDynamo.bcs.OUTER_BOUNDARY; lmax = 6, seed = 12345)
        for l in 0:6
            idx = topo.lm_to_index(l, 0, 6)
            @test f.coeffs_imag[idx] == 0.0
        end
        @test any(!=(0.0), f.coeffs_imag)        # m > 0 still carries a phase
    end

    # ── G3 / G13: field presence must be a VALUE check, not a type check ──────
    @testset "topography dispatch skips absent fields and covers composition" begin
        # `SolverFields` declares `magnetic::M` with `M <: Union{...,Nothing}`, so
        # `hasfield(typeof(fields), :magnetic)` is true even on a hydro-only run:
        # the magnetic correction was invoked with `nothing` every step, fell through
        # both component branches and emitted an unthrottled `@warn` — once per
        # timestep for the life of the run. Presence has to be read from the VALUE.
        # Composition was simply never dispatched at all, though its correction exists.
        cfg = topo.TopographyCouplingConfig(; enabled = true, epsilon = 0.01)
        data = topo.TopographyData()
        empty = _FieldsG{Nothing, Nothing, Nothing, Nothing}(nothing, nothing,
            nothing, nothing)

        # Nothing present ⇒ nothing attempted ⇒ not a single log record.
        @test_logs topo.apply_all_topography_corrections!(empty, data; config = cfg)

        # Presence is a value question, and composition has to be asked it too — the
        # dispatch previously had no composition branch at all.
        present = _FieldsG{Nothing, Nothing, Nothing, Int}(nothing, nothing, nothing, 1)
        @test topo._field_present(empty, :composition) == false
        @test topo._field_present(present, :composition) == true
        @test topo._field_present(present, :magnetic) == false
        @test topo._field_present(present, :nonexistent) == false
    end

    # ── G1: the cross-Gaunt cache must memoize ZEROS as well ─────────────────
    @testset "get_cross_gaunt caches every key it computes" begin
        # `precompute_gaunt_tensors!` fills `G_cross` only in its `!use_wigner`
        # branch, and the solver precomputes with `use_wigner = true` — so `G_cross`
        # reaches the coupling kernels EMPTY and every value is computed lazily.
        # `get_cross_gaunt` then stored only results with |G| > 1e-14, so every
        # zero-valued key was recomputed on every visit: 5 SHTnsKit syntheses plus a
        # full nlat×nlon quadrature, in the innermost loop of the impermeability and
        # insulating corrections. The lazy path is fine; the selective caching is not.
        cache = topo.GauntTensorCache{Float64}(4, 4)
        @test isempty(cache.G_cross)

        # keys that clear the analytic early-out (l2 > 0, L > 0, m1 == m2 + M), so
        # each one is genuinely computed; most of them evaluate to zero.
        keys = [(1, 0, 1, 0, 1, 0), (1, 0, 2, 0, 1, 0), (2, 1, 1, 1, 2, 0),
            (2, 0, 2, 0, 2, 0), (3, 1, 2, 1, 1, 0)]
        for k in keys
            topo.get_cross_gaunt(cache, k...)
            @test haskey(cache.G_cross, k)
        end

        # and a repeat visit must be a pure hit, not a recompute
        n = length(cache.G_cross)
        for k in keys
            topo.get_cross_gaunt(cache, k...)
        end
        @test length(cache.G_cross) == n

        # the analytic early-out stays free: rejected keys are never cached
        @test topo.get_cross_gaunt(cache, 1, 0, 0, 0, 1, 0) == 0.0
        @test !haskey(cache.G_cross, (1, 0, 0, 0, 1, 0))
    end

    # ── G11: the fallback evaluator must use the SAME pole convention ─────────
    @testset "evaluate_topography_fallback is not pole-flipped" begin
        # `gauss_legendre_nodes` returns μ ASCENDING (-0.96 … +0.96), so
        # `acos.(nodes)` gives θ descending from ~π: row 1 of the fallback grid sat at
        # the SOUTH pole. The primary path (`_gauss_legendre_point`, used to build the
        # grid the transform and the Gaunt quadrature share) gives θ ascending from
        # ~0. When an SHTnsKit synthesis failed and the fallback fired, the topography
        # it returned was mirrored top-to-bottom against the field grid.
        nlat, nlon = 8, 16
        cfg = (nlat = nlat, nlon = nlon)

        f = topo.TopographyField{Float64}(1, 1, 1.0, GeoDynamo.bcs.OUTER_BOUNDARY)
        f.coeffs_real[topo.lm_to_index(1, 0, 1)] = 1.0     # h ∝ cos θ

        h = topo.evaluate_topography_fallback(f, cfg)
        @test h[1, 1] > 0                     # row 1 is the NORTH pole side
        @test h[nlat, 1] < 0

        # and it must agree with the primary grid pointwise, not just in sign
        theta = [acos(topo._gauss_legendre_point(nlat, i)[1]) for i in 1:nlat]
        phi = [2π * (j - 1) / nlon for j in 1:nlon]
        @test h ≈ topo.evaluate_topography(f, theta, phi)
    end

    # ── G4: the base snapshot must not overwrite someone else's update ────────
    @testset "reset_boundary_to_base! adopts an external boundary update" begin
        # The snapshot was taken on the FIRST call and `copyto!`d back on every later
        # one, so any legitimate write by another owner of the same array was silently
        # reverted from step 2 onwards — `update_time_dependent_boundaries!`
        # (bcs/bcs.jl) and `apply_temperature_boundaries!` both rewrite
        # `field.boundary_values`, so a time-dependent thermal BC ran frozen at its
        # t = 0 value for the rest of the run. Only topography's OWN correction may be
        # rolled back; anything else is the new base.
        topo.clear_boundary_value_base_cache!()
        bv = zeros(Float64, 2, 4)

        topo.reset_boundary_to_base!(bv)          # first visit: base = zeros
        bv[1, 1] = 5.0                            # topography's own correction
        topo.mark_boundary_applied!(bv)
        topo.reset_boundary_to_base!(bv)
        @test bv[1, 1] == 0.0                     # rolled back, as intended

        bv[2, 1] = 7.0                            # an EXTERNAL update, not ours
        topo.reset_boundary_to_base!(bv)
        @test bv[2, 1] == 7.0                     # must survive

        bv[1, 2] = 3.0                            # our correction on the new base
        topo.mark_boundary_applied!(bv)
        topo.reset_boundary_to_base!(bv)
        @test bv[1, 2] == 0.0                     # ours rolled back
        @test bv[2, 1] == 7.0                     # theirs still there
        topo.clear_boundary_value_base_cache!()
    end

    # ── G10: Stefan arrays sized from two different nlm must not overrun ──────
    @testset "compute_stefan_flux tolerates a larger topography nlm" begin
        # `heat_flux_ic/oc` are sized from the SPECTRAL nlm, but the loop is sized from
        # `state.topography.nlm` — and `create_solver_topography_state` assigns
        # `stefan.topography = data.icb`, whose nlm comes from the TOPOGRAPHY FILE's
        # lmax. Loading an ICB file at lmax = 8 under an lmax = 4 run made the loop run
        # 45 times over 15-element vectors: a BoundsError from inside the timestep.
        st = topo.StefanState{Float64}(lmax = 4)
        n_small = length(st.heat_flux_ic)
        st.topography = topo.TopographyField{Float64}(8, 8, 0.35,
            GeoDynamo.bcs.INNER_BOUNDARY)
        @test st.topography.nlm > n_small

        flux = topo.compute_stefan_flux(st)
        @test length(flux) == st.topography.nlm
        @test all(iszero, flux[(n_small + 1):end])   # nothing invented past the data
    end

    @testset "create_random_topography is seeded on every rank by default" begin
        # `seed = 0` skipped seeding entirely, so under MPI each rank drew from its
        # own stream and built a DIFFERENT boundary — the ranks then solved against
        # inconsistent geometry with nothing to flag it.
        a = topo.create_random_topography(l -> 1.0, 1.0, GeoDynamo.bcs.OUTER_BOUNDARY;
            lmax = 4)
        b = topo.create_random_topography(l -> 1.0, 1.0, GeoDynamo.bcs.OUTER_BOUNDARY;
            lmax = 4)
        @test a.coeffs_real == b.coeffs_real
        @test a.coeffs_imag == b.coeffs_imag
    end
end
