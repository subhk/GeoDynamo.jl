# ================================================================================
# Extended tail-coverage tests
# ================================================================================
# Targets five high-yield, under-covered source files with concrete, known-value
# assertions:
#   - src/bcs/interpolation.jl       (grid interpolation math)
#   - src/core/parameters.jl         (parameter parsing / validation / IO)
#   - src/core/initial_conditions.jl (analytical + random IC builders)
#   - src/bcs/integration.jl         (field <-> boundary integration helpers)
#   - src/physics/composition/field.jl (composition field ops + diagnostics)
#
# MPI is initialized once (some testsets construct fields, which require it) and,
# matching the rest of the suite, finalize is guarded behind the
# GEODYNAMO_TEST_MPI_FINALIZE env var so the harness controls ordering.

using Test
using MPI
using LinearAlgebra
using Random

const FINALIZE_MPI_TAIL_EXT = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

@testset "Tail Coverage Extended" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping extended tail-coverage tests"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    # ----------------------------------------------------------------------------
    # Shared small grid / config / domain (mirrors fixture construction)
    # ----------------------------------------------------------------------------
    lmax = 4
    mmax = 4
    nlat = max(lmax + 2, 10)
    nlon = max(2lmax + 1, 16)
    nr = 6

    cfg = GeoDynamo.create_shtnskit_config(
        lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr)
    shell = GeoDynamo.create_radial_domain(nr)

    # ============================================================================
    # 1. src/bcs/interpolation.jl
    # ============================================================================
    @testset "interpolation.jl" begin
        bcs = GeoDynamo.bcs

        # ---- find_grid_indices: non-periodic ----
        coords = Float64[0.0, 1.0, 2.0, 3.0, 4.0]
        # interior point between coords[3]=2.0 and coords[4]=3.0
        @test bcs.find_grid_indices(coords, 2.5; is_periodic = false) == (3, 4)
        # exact interior node -> low side picks that index
        @test bcs.find_grid_indices(coords, 2.0; is_periodic = false) == (3, 4)
        # at/below first
        @test bcs.find_grid_indices(coords, -1.0; is_periodic = false) == (1, 2)
        @test bcs.find_grid_indices(coords, 0.0; is_periodic = false) == (1, 2)
        # at/above last
        @test bcs.find_grid_indices(coords, 4.0; is_periodic = false) == (4, 5)
        @test bcs.find_grid_indices(coords, 9.0; is_periodic = false) == (4, 5)
        # too-short array throws
        @test_throws ArgumentError bcs.find_grid_indices([1.0], 0.5; is_periodic = false)

        # ---- find_grid_indices: periodic (longitude-like) ----
        phi = collect(range(0.0, step = π / 2, length = 4))  # 0, π/2, π, 3π/2 (spacing π/2)
        # interior
        @test bcs.find_grid_indices(phi, π / 4; is_periodic = true) == (1, 2)
        # beyond last but within one spacing -> wrap to (n, 1)
        @test bcs.find_grid_indices(phi, 7π / 4; is_periodic = true) == (4, 1)

        # ---- get_interpolation_weights ----
        # midpoint between coords[3] and coords[4] -> equal weights
        w1, w2 = bcs.get_interpolation_weights(coords, 2.5, (3, 4); is_periodic = false)
        @test w1 ≈ 0.5
        @test w2 ≈ 0.5
        @test w1 + w2 ≈ 1.0
        # 1/4 of the way -> (0.75, 0.25)
        wa, wb = bcs.get_interpolation_weights(coords, 2.25, (3, 4); is_periodic = false)
        @test wa ≈ 0.75
        @test wb ≈ 0.25
        # equal indices -> degenerate (1, 0)
        @test bcs.get_interpolation_weights(coords, 2.0, (3, 3)) == (1.0, 0.0)
        # periodic wrap-around weights sum to 1 and are in [0,1]
        pw1, pw2 = bcs.get_interpolation_weights(phi, 7π / 4, (4, 1); is_periodic = true)
        @test pw1 + pw2 ≈ 1.0
        @test 0.0 <= pw1 <= 1.0 && 0.0 <= pw2 <= 1.0

        # ---- bilinear_interpolate ----
        # data[i,j] = i so the result depends only on theta weights
        data = Float64[1 1 1 1; 2 2 2 2; 3 3 3 3]   # 3x4
        # theta between rows 2 and 3, midpoint -> 2.5 ; phi exactly on a node
        r = bcs.bilinear_interpolate(data, (2, 3), (1, 1), (0.5, 0.5), (1.0, 0.0))
        @test r ≈ 2.5
        # corner-exact: weights (1,0),(1,0) -> data[1,1] = 1
        @test bcs.bilinear_interpolate(data, (1, 2), (1, 2), (1.0, 0.0), (1.0, 0.0)) ≈ 1.0
        # full average of a constant block stays constant
        cdata = fill(7.0, 3, 4)
        @test bcs.bilinear_interpolate(cdata, (1, 2), (1, 2), (0.3, 0.7), (0.4, 0.6)) ≈ 7.0
        # phi index wrap (j2 > nlon -> 1)
        @test bcs.bilinear_interpolate(data, (2, 2), (4, 5), (1.0, 0.0), (0.5, 0.5)) ≈ 2.0

        # ---- interpolate_boundary_to_grid (identity grid round-trip) ----
        src_theta = collect(range(0.0, π, length = 6))
        src_phi = collect(range(0.0, 2π * 5 / 6, length = 6))  # 6 pts, spacing 2π/6
        src_vals = Float64[sin(t) + 0.1 * p for t in src_theta, p in src_phi]
        bd = bcs.create_boundary_data(src_vals, "temperature";
            theta = src_theta, phi = src_phi)
        # interpolate back onto the SAME grid -> should reproduce values
        out = bcs.interpolate_boundary_to_grid(bd, src_theta, src_phi)
        @test size(out) == (6, 6)
        @test out ≈ src_vals atol = 1e-10

        # empty target throws
        @test_throws ArgumentError bcs.interpolate_boundary_to_grid(bd, Float64[], src_phi)
        # missing coordinates throws
        bd_nocoord = bcs.create_boundary_data(src_vals, "temperature")
        @test_throws ArgumentError bcs.interpolate_boundary_to_grid(bd_nocoord, src_theta, src_phi)

        # ---- create_interpolation_cache + interpolate_with_cache ----
        cache = bcs.create_interpolation_cache(bd, src_theta, src_phi)
        @test cache["target_shape"] == (6, 6)
        @test length(cache["theta_indices"]) == 6
        @test length(cache["phi_indices"]) == 6
        cached_out = bcs.interpolate_with_cache(bd, cache)
        @test cached_out ≈ src_vals atol = 1e-10
        # cached result equals direct result
        @test cached_out ≈ out atol = 1e-12

        # empty cache (no coordinates) returns data as-is
        empty_cache = bcs.create_interpolation_cache(bd_nocoord, src_theta, src_phi)
        @test isempty(empty_cache)
        passthrough = bcs.interpolate_with_cache(bd_nocoord, empty_cache)
        @test passthrough == src_vals

        # ---- validate_interpolation_grids ----
        @test bcs.validate_interpolation_grids(src_theta, src_phi, src_theta, src_phi) == true
        # target theta exceeding source range throws
        @test_throws ArgumentError bcs.validate_interpolation_grids(
            src_theta, src_phi, collect(range(-1.0, π, length = 6)), src_phi)
        # non-monotonic source theta throws
        @test_throws ArgumentError bcs.validate_interpolation_grids(
            reverse(src_theta), src_phi, src_theta, src_phi)

        # ---- check_interpolation_bounds (warns on out-of-range theta) ----
        @test bcs.check_interpolation_bounds(bd, src_theta, src_phi) === nothing
        @test_logs (:warn, r"Target theta range") bcs.check_interpolation_bounds(
            bd, [-1.0, 0.0, π], src_phi)
        # no-coordinate data returns nothing silently
        @test bcs.check_interpolation_bounds(bd_nocoord, src_theta, src_phi) === nothing

        # ---- get_interpolation_statistics ----
        stats = bcs.get_interpolation_statistics(bd, out)
        smin, smax = stats["source_range"]
        @test smin ≈ minimum(src_vals)
        @test smax ≈ maximum(src_vals)
        @test stats["source_mean"] ≈ sum(src_vals) / length(src_vals)
        @test stats["range_preservation"] == true   # identity interp preserves range

        # ---- estimate_interpolation_error ----
        # identical grids -> zero relative error, "good" quality
        err = bcs.estimate_interpolation_error(bd, src_theta, src_phi)
        te, pe = err["relative_error_estimate"]
        @test te ≈ 0.0 atol = 1e-12
        @test pe ≈ 0.0 atol = 1e-12
        @test err["interpolation_quality"] == "good"
        # no-coordinate data returns the error sentinel
        err2 = bcs.estimate_interpolation_error(bd_nocoord, src_theta, src_phi)
        @test haskey(err2, "error")
    end

    # ============================================================================
    # 2. src/core/parameters.jl
    # ============================================================================
    @testset "parameters.jl" begin
        # ---- safe_parse_value: scalars / symbols / strings / bools ----
        pd = Dict{Symbol, Any}()
        @test GeoDynamo.safe_parse_value("true", pd) === true
        @test GeoDynamo.safe_parse_value("false", pd) === false
        @test GeoDynamo.safe_parse_value("42", pd) === 42
        @test GeoDynamo.safe_parse_value("3.5", pd) === 3.5
        @test GeoDynamo.safe_parse_value(":ball", pd) === :ball
        @test GeoDynamo.safe_parse_value("\"hello\"", pd) == "hello"
        @test GeoDynamo.safe_parse_value("pi", pd) == π
        @test GeoDynamo.safe_parse_value("π", pd) == π
        # arithmetic expression evaluated via safe_eval_expr
        @test GeoDynamo.safe_parse_value("2 * 3 + 4", pd) == 10
        @test GeoDynamo.safe_parse_value("sqrt(16.0)", pd) == 4.0

        # ---- safe_eval_expr directly ----
        @test GeoDynamo.safe_eval_expr(Meta.parse("max(2, 7)"), pd) == 7
        @test GeoDynamo.safe_eval_expr(Meta.parse("2^10"), pd) == 1024
        # parameter reference resolution
        pd2 = Dict{Symbol, Any}(:lmax => 8)
        @test GeoDynamo.safe_eval_expr(Meta.parse("lmax * 2"), pd2) == 16
        # unknown reference throws
        @test_throws ArgumentError GeoDynamo.safe_eval_expr(Meta.parse("nope"), pd)
        # disallowed operation throws
        @test_throws ArgumentError GeoDynamo.safe_eval_expr(Meta.parse("rm(\"x\")"), pd)
        # safe constructor (CNAB2) is reconstructed
        ts = GeoDynamo.safe_eval_expr(Meta.parse("CNAB2()"), pd)
        @test ts isa GeoDynamo.CNAB2

        # ---- _parameter_errors_warnings: clean defaults ----
        e0, w0 = GeoDynamo._parameter_errors_warnings(GeoDynamo.SolverParameters())
        @test isempty(e0)
        @test isempty(w0)
        # coarse nr (8 <= nr < 16) triggers a warning, not an error
        _, wcoarse = GeoDynamo._parameter_errors_warnings(GeoDynamo.SolverParameters(nr = 15))
        @test any(contains(w, "coarse") for w in wcoarse)
        # mmax > lmax error
        emm, _ = GeoDynamo._parameter_errors_warnings(
            GeoDynamo.SolverParameters(lmax = 4, mmax = 8))
        @test any(contains(e, "mmax") for e in emm)
        # conducting inner core needs shell geometry
        ebc, _ = GeoDynamo._parameter_errors_warnings(GeoDynamo.SolverParameters(
            geometry = :ball, radius_ratio = 0.0, nr_inner = 0,
            magnetic_inner_bc = :conducting_inner_core))
        @test any(contains(e, "conducting_inner_core") for e in ebc)
        # bad output precision
        eop, _ = GeoDynamo._parameter_errors_warnings(
            GeoDynamo.SolverParameters(output_precision = :float16))
        @test any(contains(e, "output_precision") for e in eop)

        # ---- show (text/plain) renders sections ----
        io = IOBuffer()
        show(io, MIME"text/plain"(), GeoDynamo.SolverParameters())
        txt = String(take!(io))
        @test occursin("SolverParameters", txt)
        @test occursin("Grid", txt)
        @test occursin("Physics", txt)
        @test occursin("Timestepping", txt)

        # ---- save_parameters + load_parameters_from_file round-trip ----
        params = GeoDynamo.SolverParameters(nr = 48, lmax = 16, Ra = 2.5e6, Ek = 3e-4)
        fname = joinpath(mktempdir(), "params_roundtrip.jl")
        @test GeoDynamo.save_parameters(params, fname) == fname
        @test isfile(fname)
        loaded = GeoDynamo.load_parameters_from_file(fname)
        @test loaded.nr == 48
        @test loaded.lmax == 16
        @test loaded.Ra == 2.5e6
        @test loaded.Ek == 3e-4
        @test loaded.geometry == :shell

        # ---- create_parameter_template writes a loadable file ----
        tmpl = joinpath(mktempdir(), "template_params.jl")
        @test GeoDynamo.create_parameter_template(tmpl) == tmpl
        @test isfile(tmpl)
        tmpl_loaded = GeoDynamo.load_parameters_from_file(tmpl)
        @test tmpl_loaded.nr == GeoDynamo.SolverParameters().nr

        # ---- load_parameters_from_file on a missing file -> defaults + warn ----
        missing_params = @test_logs (:warn, r"Parameter file not found") GeoDynamo.load_parameters_from_file(
            joinpath(mktempdir(), "does_not_exist.jl"))
        @test missing_params.nr == GeoDynamo.SolverParameters().nr

        # ---- legacy alias max_steps -> stop_iteration (with deprecation warn) ----
        alias_file = joinpath(mktempdir(), "alias_params.jl")
        open(alias_file, "w") do f
            println(f, "# legacy parameter file")
            println(f, "max_steps = 1234")
            println(f, "nr = 40")
        end
        alias_params = GeoDynamo.load_parameters_from_file(alias_file)
        @test alias_params.stop_iteration == 1234
        @test alias_params.nr == 40

        # ---- set_parameters! / get_parameters round-trip on the global ----
        previous = GeoDynamo.get_parameters()
        try
            custom = GeoDynamo.SolverParameters(nr = 72, lmax = 24)
            GeoDynamo.set_parameters!(custom; validate = false)
            got = GeoDynamo.get_parameters()
            @test got.nr == 72
            @test got.lmax == 24
        finally
            GeoDynamo.set_parameters!(previous; validate = false)
        end
        @test GeoDynamo.get_parameters().nr == previous.nr
    end

    # ============================================================================
    # 3. src/core/initial_conditions.jl
    # ============================================================================
    @testset "initial_conditions.jl" begin
        s4pi = sqrt(4 * π)

        # Helper: read the l=0 (lm index 1) coefficient along radius from a scalar field
        function l0_profile(field)
            spec = field.spectral
            real_data = parent(spec.data_real)
            r_range = GeoDynamo.range_local(spec.pencil, 3)
            slot = GeoDynamo.local_spectral_storage_slot(spec.config, 1)
            slot === nothing && return nothing
            prof = Float64[]
            for r_idx in r_range
                lr = r_idx - first(r_range) + 1
                lr <= size(real_data, 3) || continue
                push!(prof, real_data[slot[1], slot[2], lr])
            end
            return prof
        end

        # ---- set_analytical_initial_conditions! :conductive (known l=0 coeffs) ----
        temp = GeoDynamo.create_shtns_temperature_field(Float64, cfg, shell)
        GeoDynamo.set_analytical_initial_conditions!(temp, :temperature, :conductive;
            amplitude = 2.0)
        prof = l0_profile(temp)
        @test prof !== nothing
        # conductive: value = sqrt(4π) * amplitude * (1 - r_frac), r_frac=(r-1)/(nr-1)
        expected_cond = [s4pi * 2.0 * (1.0 - (r - 1) / (nr - 1)) for r in 1:nr]
        @test prof ≈ expected_cond atol = 1e-12
        # imaginary part is all zeros
        @test all(parent(temp.spectral.data_imag) .== 0.0)
        # non-l0 modes are exactly zero for a pure conductive profile
        spec_real_t = parent(temp.spectral.data_real)
        @test all(spec_real_t[2:end, :, :] .== 0.0)

        # ---- :hot_blob temperature (l=0 only, finite, background+bump) ----
        temp2 = GeoDynamo.create_shtns_temperature_field(Float64, cfg, shell)
        GeoDynamo.set_analytical_initial_conditions!(temp2, :temperature, :hot_blob;
            amplitude = 1.0, r_center = 0.5, blob_width = 0.2)
        prof2 = l0_profile(temp2)
        # value = sqrt(4π) * (0.5*(1-r_frac) + amplitude * exp(-0.5*((r_frac-0.5)/0.2)^2))
        expected_hb = [s4pi * (0.5 * (1.0 - (r - 1) / (nr - 1)) +
                               1.0 * exp(-0.5 * (((r - 1) / (nr - 1) - 0.5) / 0.2)^2))
                       for r in 1:nr]
        @test prof2 ≈ expected_hb atol = 1e-12
        @test all(isfinite, parent(temp2.spectral.data_real))

        # unknown temperature pattern throws
        @test_throws ArgumentError GeoDynamo.set_analytical_initial_conditions!(
            temp2, :temperature, :bogus_pattern)

        # ---- :stratified composition (known l=0 coeffs) ----
        comp = GeoDynamo.create_shtns_composition_field(Float64, cfg, shell)
        GeoDynamo.set_analytical_initial_conditions!(comp, :composition, :stratified;
            amplitude = 1.0, bottom_composition = 0.3, top_composition = 0.1)
        cprof = l0_profile(comp)
        expected_strat = [s4pi * (0.3 + (0.1 - 0.3) * (r - 1) / (nr - 1)) for r in 1:nr]
        @test cprof ≈ expected_strat atol = 1e-12

        # ---- :dipole magnetic (l=1 poloidal = amp*sin(pi*r_frac), toroidal scaled) ----
        mag = GeoDynamo.create_shtns_magnetic_fields(Float64, cfg, shell, shell)
        GeoDynamo.set_analytical_initial_conditions!(mag, :magnetic, :dipole; amplitude = 1.0)
        # find an l=1 lm index and read its poloidal radial profile
        l1_idx = findfirst(==(1), cfg.l_values)
        @test l1_idx !== nothing
        pol_real = parent(mag.poloidal.data_real)
        pol_slot = GeoDynamo.local_spectral_storage_slot(cfg, l1_idx)
        if pol_slot !== nothing
            r_range = GeoDynamo.range_local(mag.poloidal.pencil, 3)
            pol_prof = Float64[]
            for r_idx in r_range
                lr = r_idx - first(r_range) + 1
                lr <= size(pol_real, 3) || continue
                push!(pol_prof, pol_real[pol_slot[1], pol_slot[2], lr])
            end
            expected_dip = [sin(π * (r - 1) / (nr - 1)) for r in 1:nr]
            @test pol_prof ≈ expected_dip atol = 1e-12
        end
        @test all(isfinite, parent(mag.poloidal.data_real))
        @test all(isfinite, parent(mag.toroidal.data_real))

        # unknown magnetic pattern throws
        @test_throws ArgumentError GeoDynamo.set_analytical_initial_conditions!(
            mag, :magnetic, :no_such_pattern)

        # ---- :convective velocity (low-order modes, finite) ----
        vel = GeoDynamo.create_shtns_velocity_fields(Float64, cfg, shell)
        GeoDynamo.set_analytical_initial_conditions!(vel, :velocity, :convective; amplitude = 0.5)
        @test all(isfinite, parent(vel.toroidal.data_real))
        @test all(isfinite, parent(vel.poloidal.data_real))
        # l=0 mode must remain zero for a vector field
        v_tor_real = parent(vel.toroidal.data_real)
        l0v = GeoDynamo.local_spectral_storage_slot(cfg, 1)
        if l0v !== nothing
            @test all(v_tor_real[l0v[1], l0v[2], :] .== 0.0)
        end

        # unknown field type throws
        @test_throws ArgumentError GeoDynamo.set_analytical_initial_conditions!(
            temp, :pressure, :conductive)

        # ---- generate_random_initial_conditions! reproducibility (seeded) ----
        ta = GeoDynamo.create_shtns_temperature_field(Float64, cfg, shell)
        tb = GeoDynamo.create_shtns_temperature_field(Float64, cfg, shell)
        GeoDynamo.generate_random_initial_conditions!(ta, :temperature;
            amplitude = 0.3, modes_range = 1:3, seed = 123)
        GeoDynamo.generate_random_initial_conditions!(tb, :temperature;
            amplitude = 0.3, modes_range = 1:3, seed = 123)
        @test parent(ta.spectral.data_real) == parent(tb.spectral.data_real)
        @test all(isfinite, parent(ta.spectral.data_real))

        # different seeds -> different fields (with very high probability)
        tc = GeoDynamo.create_shtns_temperature_field(Float64, cfg, shell)
        GeoDynamo.generate_random_initial_conditions!(tc, :temperature;
            amplitude = 0.3, modes_range = 1:3, seed = 999)
        @test parent(tc.spectral.data_real) != parent(ta.spectral.data_real)

        # random for vector + composition fields (finite, reproducible)
        ma = GeoDynamo.create_shtns_magnetic_fields(Float64, cfg, shell, shell)
        GeoDynamo.generate_random_initial_conditions!(ma, :magnetic;
            amplitude = 0.01, modes_range = 1:5, seed = 42)
        @test all(isfinite, parent(ma.poloidal.data_real))

        va = GeoDynamo.create_shtns_velocity_fields(Float64, cfg, shell)
        GeoDynamo.generate_random_initial_conditions!(va, :velocity;
            amplitude = 0.02, modes_range = 1:4, seed = 7)
        @test all(isfinite, parent(va.toroidal.data_real))

        ca = GeoDynamo.create_shtns_composition_field(Float64, cfg, shell)
        GeoDynamo.generate_random_initial_conditions!(ca, :composition;
            amplitude = 0.05, modes_range = 1:3, seed = 11)
        @test all(isfinite, parent(ca.spectral.data_real))

        # unknown field type for the random generator throws
        @test_throws ArgumentError GeoDynamo.generate_random_initial_conditions!(
            ta, :pressure)

        # ---- load_initial_conditions! real NetCDF behavior ----
        tload = GeoDynamo.create_shtns_temperature_field(Float64, cfg, shell)
        # missing file throws
        @test_throws ArgumentError GeoDynamo.load_initial_conditions!(
            tload, :temperature, joinpath(mktempdir(), "missing.nc"))
        # an existing non-NetCDF file errors instead of silently falling back
        existing, ioh = mktemp()
        close(ioh)
        @test_throws Exception GeoDynamo.load_initial_conditions!(
            tload, :temperature, existing)

        # ---- save_initial_conditions writes a real file that loads back ----
        GeoDynamo.set_analytical_initial_conditions!(
            tload, :temperature, :conductive; amplitude = 1.0)
        savepath = joinpath(mktempdir(), "save_ic.nc")
        saved = GeoDynamo.save_initial_conditions(tload, :temperature, savepath)
        @test saved == savepath
        @test isfile(savepath)
        treload = GeoDynamo.create_shtns_temperature_field(Float64, cfg, shell)
        GeoDynamo.load_initial_conditions!(treload, :temperature, savepath)
        @test parent(treload.spectral.data_real) == parent(tload.spectral.data_real)
    end

    # ============================================================================
    # 4. src/bcs/integration.jl
    # ============================================================================
    @testset "integration.jl" begin
        bcs = GeoDynamo.bcs
        Shell = GeoDynamo.GeoDynamoShell

        # ---- get_current_simulation_time: fallback chain ----
        @test bcs.get_current_simulation_time((; time = 3.5)) == 3.5
        @test bcs.get_current_simulation_time((; t = 1.25)) == 1.25
        @test bcs.get_current_simulation_time((; current_time = 9.0)) == 9.0
        # nested timestep_state with time
        @test bcs.get_current_simulation_time((; timestep_state = (; time = 4.0))) == 4.0
        # nested timestep_state with step + dt -> step*dt
        @test bcs.get_current_simulation_time(
            (; timestep_state = (; step = 5, dt = 0.1))) ≈ 0.5
        # no time info -> 0.0
        @test bcs.get_current_simulation_time((; foo = 1)) == 0.0

        # ---- apply_temperature_boundaries! transforms physical -> spectral ----
        temp_field = Shell.create_shell_temperature_field(Float64, cfg; nr = nr)
        pbs = bcs.create_programmatic_temperature_boundaries(
            (:uniform, 1.0), (:uniform, 0.5), cfg)
        bcs.apply_temperature_boundaries!(temp_field, pbs)
        @test temp_field.boundary_condition_set === pbs.boundary_set
        @test all(temp_field.bc_type_inner .== Int(bcs.DIRICHLET))
        @test all(temp_field.bc_type_outer .== Int(bcs.DIRICHLET))
        # A uniform physical field excites only the (0,0) coefficient; verify the
        # stored spectral inner row matches the direct transform of the input.
        expected_inner = bcs.shtns_physical_to_spectral(
            pbs.boundary_set.inner_boundary.values, cfg)
        nlm_chk = min(length(expected_inner), size(temp_field.boundary_values, 2))
        @test temp_field.boundary_values[1, 1:nlm_chk] ≈ expected_inner[1:nlm_chk]
        @test count(!iszero, temp_field.boundary_values[1, :]) > 0
        @test count(!iszero, temp_field.boundary_values[2, :]) > 0

        # ---- get_boundary_condition_summary (after applying boundaries) ----
        summ = bcs.get_boundary_condition_summary(temp_field, GeoDynamo.TEMPERATURE)
        @test summ["field_type"] == string(GeoDynamo.TEMPERATURE)
        @test summ["has_boundary_fields"] == true
        @test summ["has_boundary_conditions"] == true
        @test haskey(summ, "inner_boundary")
        @test haskey(summ, "outer_boundary")
        @test summ["boundary_spectral_coefficients"]["total_modes"] == cfg.nlm

        # ---- reset_boundary_conditions! clears everything ----
        bcs.reset_boundary_conditions!(temp_field, GeoDynamo.TEMPERATURE)
        @test temp_field.boundary_condition_set === nothing
        @test all(temp_field.boundary_values .== 0.0)
        @test temp_field.boundary_time_index[] == 1
        # summary after reset reports no boundary conditions
        summ2 = bcs.get_boundary_condition_summary(temp_field, GeoDynamo.TEMPERATURE)
        @test summ2["has_boundary_conditions"] == false
        @test haskey(summ2, "reason")

        # ---- apply_composition_boundaries! ----
        comp_field = GeoDynamo.create_shtns_composition_field(Float64, cfg, shell)
        cpbs = bcs.create_programmatic_composition_boundaries(
            (:uniform, 0.2), (:neumann, 0.0), cfg)
        bcs.apply_composition_boundaries!(comp_field, cpbs)
        @test all(comp_field.bc_type_inner .== Int(bcs.DIRICHLET))
        @test all(comp_field.bc_type_outer .== Int(bcs.NEUMANN))
        @test count(!iszero, comp_field.boundary_values[1, :]) > 0

        # ---- initialize_boundary_conditions! resets scalar BC arrays to DIRICHLET ----
        comp_field2 = GeoDynamo.create_shtns_composition_field(Float64, cfg, shell)
        # composition defaults are NEUMANN; initialize must flip to DIRICHLET defaults
        bcs.initialize_boundary_conditions!(comp_field2, GeoDynamo.COMPOSITION)
        @test all(comp_field2.bc_type_inner .== Int(bcs.DIRICHLET))
        @test all(comp_field2.bc_type_outer .== Int(bcs.DIRICHLET))
        @test all(comp_field2.boundary_values .== 0.0)
        @test comp_field2.boundary_time_index[] == 1

        # ---- validate_field_boundary_compatibility ----
        # compatible scalar boundary set passes
        inner = bcs.create_boundary_data(zeros(Float64, cfg.nlat, cfg.nlon), "temperature")
        outer = bcs.create_boundary_data(zeros(Float64, cfg.nlat, cfg.nlon), "temperature")
        good_set = bcs.BoundaryConditionSet{Float64}(
            inner, outer, "temperature", GeoDynamo.TEMPERATURE, 0.0)
        @test bcs.validate_field_boundary_compatibility(
            comp_field2, GeoDynamo.TEMPERATURE, good_set) == true
        # field-type mismatch throws (field is COMPOSITION-shaped but set is VELOCITY)
        vel_inner = bcs.create_boundary_data(
            zeros(Float64, cfg.nlat, cfg.nlon, 3), "velocity")
        vel_outer = bcs.create_boundary_data(
            zeros(Float64, cfg.nlat, cfg.nlon, 3), "velocity")
        vel_set = bcs.BoundaryConditionSet{Float64}(
            vel_inner, vel_outer, "velocity", GeoDynamo.VELOCITY, 0.0)
        @test_throws ArgumentError bcs.validate_field_boundary_compatibility(
            comp_field2, GeoDynamo.TEMPERATURE, vel_set)

        # ---- copy_boundary_conditions! duplicates BC state across fields ----
        src_field = GeoDynamo.create_shtns_composition_field(Float64, cfg, shell)
        bcs.apply_composition_boundaries!(src_field, cpbs)
        dst_field = GeoDynamo.create_shtns_composition_field(Float64, cfg, shell)
        bcs.copy_boundary_conditions!(dst_field, src_field, GeoDynamo.COMPOSITION)
        @test dst_field.boundary_condition_set === src_field.boundary_condition_set
        @test dst_field.boundary_values ≈ src_field.boundary_values
        @test dst_field.bc_type_inner == src_field.bc_type_inner
        @test dst_field.bc_type_outer == src_field.bc_type_outer
        # copying from a field with no BC set is a no-op returning dest
        clean_src = GeoDynamo.create_shtns_composition_field(Float64, cfg, shell)
        returned = bcs.copy_boundary_conditions!(dst_field, clean_src, GeoDynamo.COMPOSITION)
        @test returned === dst_field
    end

    # ============================================================================
    # 5. src/physics/composition/field.jl
    # ============================================================================
    @testset "composition/field.jl" begin
        # ---- defaults on a fresh composition field ----
        comp = GeoDynamo.create_shtns_composition_field(Float64, cfg, shell)
        @test size(comp.boundary_values) == (2, cfg.nlm)
        @test all(comp.boundary_values .== 0.0)
        @test all(comp.bc_type_inner .== Int(GeoDynamo.NEUMANN))
        @test all(comp.bc_type_outer .== Int(GeoDynamo.NEUMANN))
        @test length(comp.l_factors) == cfg.nlm
        # l_factors hold l*(l+1)
        @test comp.l_factors ≈ Float64[l * (l + 1) for l in cfg.l_values]
        @test all(comp.internal_sources .== 0.0)

        # ---- validate_composition_field passes for a well-formed field ----
        @test GeoDynamo.validate_composition_field(comp, shell) == true

        # ---- set_composition_boundary_conditions! (:fixed / :no_flux) ----
        GeoDynamo.set_composition_boundary_conditions!(comp, :fixed, :no_flux, 2.5, -1.0)
        @test all(comp.bc_type_inner .== Int(GeoDynamo.DIRICHLET))
        @test all(comp.bc_type_outer .== Int(GeoDynamo.NEUMANN))
        @test all(comp.boundary_values[1, :] .== 2.5)
        # outer no_flux leaves outer boundary_values untouched (still zero)
        @test all(comp.boundary_values[2, :] .== 0.0)
        # the other direction: outer :fixed sets outer row, inner :no_flux only sets type
        GeoDynamo.set_composition_boundary_conditions!(comp, :no_flux, :fixed, 0.0, 4.0)
        @test all(comp.bc_type_inner .== Int(GeoDynamo.NEUMANN))
        @test all(comp.bc_type_outer .== Int(GeoDynamo.DIRICHLET))
        @test all(comp.boundary_values[2, :] .== 4.0)

        # ---- set_composition_ic! :uniform sets l=0 coefficient to 1 along radius ----
        comp_u = GeoDynamo.create_shtns_composition_field(Float64, cfg, shell)
        GeoDynamo.set_composition_ic!(comp_u, :uniform, shell)
        sru = parent(comp_u.spectral.data_real)
        slot1 = GeoDynamo.local_spectral_storage_slot(cfg, 1)
        @test slot1 !== nothing
        @test all(sru[slot1[1], slot1[2], :] .== 1.0)
        @test all(parent(comp_u.spectral.data_imag) .== 0.0)
        # higher modes remain zero
        @test all(sru[2:end, :, :] .== 0.0)

        # ---- set_composition_ic! :linear sets l=0 coefficient = normalized radius ----
        comp_l = GeoDynamo.create_shtns_composition_field(Float64, cfg, shell)
        # pre-fill with garbage to confirm the IC clears stale data
        fill!(parent(comp_l.spectral.data_real), 5.0)
        fill!(parent(comp_l.spectral.data_imag), -3.0)
        GeoDynamo.set_composition_ic!(comp_l, :linear, shell)
        srl = parent(comp_l.spectral.data_real)
        r_range = GeoDynamo.range_local(cfg.pencils.spec, 3)
        for r_idx in r_range
            lr = r_idx - first(r_range) + 1
            lr <= size(srl, 3) || continue
            @test srl[slot1[1], slot1[2], lr] ≈ shell.r[r_idx, 4]
        end
        @test all(parent(comp_l.spectral.data_imag) .== 0.0)
        @test all(srl[2:end, :, :] .== 0.0)

        # ---- set_composition_ic! :random_perturbation (reproducible, l=0 untouched) ----
        comp_r = GeoDynamo.create_shtns_composition_field(Float64, cfg, shell)
        Random.seed!(2024)
        GeoDynamo.set_composition_ic!(comp_r, :random_perturbation, shell)
        srr = parent(comp_r.spectral.data_real)
        # l=0,m=0 mode (lm index 1) must remain zero
        @test all(srr[slot1[1], slot1[2], :] .== 0.0)
        @test all(isfinite, srr)

        # ---- compute_composition_rms / energy on a known uniform field ----
        # zero field -> zero RMS and zero energy
        comp_zero = GeoDynamo.create_shtns_composition_field(Float64, cfg, shell)
        @test GeoDynamo.compute_composition_rms(comp_zero, shell) ≈ 0.0 atol = 1e-14
        @test GeoDynamo.compute_composition_energy(comp_zero, shell) ≈ 0.0 atol = 1e-14

        # uniform l=0 spectral field -> positive, finite RMS
        rms_uniform = GeoDynamo.compute_composition_rms(comp_u, shell)
        @test rms_uniform > 0.0
        @test isfinite(rms_uniform)
        # RMS of the all-ones l=0 coefficient over (nr * nlm) slots:
        # sum of squares = nr (one nonzero coeff per radial level), normalized by nr*nlm
        @test rms_uniform ≈ sqrt(nr / (shell.N * cfg.nlm)) atol = 1e-12

        # ---- get_composition_statistics returns consistent named tuple ----
        # put a constant in physical space, then check min == max == that constant
        fill!(parent(comp.composition.data), 1.5)
        stats = GeoDynamo.get_composition_statistics(comp, shell)
        @test stats.min ≈ 1.5
        @test stats.max ≈ 1.5
        @test stats.rms ≈ 1.5 atol = 1e-12   # RMS of a constant equals the constant
        @test isfinite(stats.energy)

        # ---- zero_composition_work_arrays! clears work buffers ----
        fill!(parent(comp.work_spectral.data_real), 9.0)
        fill!(parent(comp.work_physical.data), 9.0)
        fill!(parent(comp.advection_physical.data), 9.0)
        GeoDynamo.zero_composition_work_arrays!(comp)
        @test all(parent(comp.work_spectral.data_real) .== 0.0)
        @test all(parent(comp.work_physical.data) .== 0.0)
        @test all(parent(comp.advection_physical.data) .== 0.0)
    end

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_TAIL_EXT && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
