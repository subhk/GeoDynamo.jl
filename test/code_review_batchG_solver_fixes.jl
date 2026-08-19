# ================================================================================
# Review batch G — fixes that need a real solver state
# ================================================================================
#
# The rest of the batch lives in `test/code_review_batchG_fixes.jl`; these two need
# constructed solver fields (a scalar field with its interpolation cache, and the
# boundary rows the implicit solve actually reads), so they build a small state the
# same way `test/topography_coupling.jl` does.
# ================================================================================

using Test
using MPI

const topoG = GeoDynamo.bcs.topography

@testset "Review batch G — solver-field fixes" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping batch G solver-field fixes"
        return
    end
    MPI.Initialized() || MPI.Init()

    L = 4
    params = GeoDynamo.SolverParameters(
        architecture = :cpu, geometry = :shell,
        nr = 16, nr_inner = 4, lmax = L, mmax = L, nlat = 12, nlon = 16,
        Ra = 1e4, Ek = 1e-2, Pr = 1.0, Pm = 1.0,
        timestep = 1e-4, start_time = 0.0, end_time = 1e-3, stop_iteration = 10,
        include_magnetic = false, include_composition = true,
        timestepper = GeoDynamo.CNAB2(),
        topography_enabled = false, stefan_enabled = false)
    state = GeoDynamo.initialize_simulation(Float64, params)
    GeoDynamo.initialize_fields!(state)

    # ── G5: the scalar boundary rows must have an imaginary half ─────────────
    @testset "scalar fields carry an imaginary boundary row" begin
        # The scalar solve already plumbs `bc_inner_imag` / `bc_outer_imag`
        # (physics/scalar_field_solver_common.jl), but `SHTnsTemperatureField` had no
        # `boundary_values_imag`, so `get_bc_vectors` handed it `nothing` and the
        # imaginary half of every m > 0 boundary correction — the entire
        # non-axisymmetric part of the thermal topography coupling — was discarded.
        for field in (state.fields.temperature, state.fields.composition)
            bc = GeoDynamo.get_bc_vectors(field)
            @test bc.inner_real !== nothing
            @test bc.inner_imag !== nothing
            @test bc.outer_imag !== nothing
            @test length(bc.outer_imag) == length(bc.outer_real)
        end

        # and a write to it must be visible through the reader
        temp = state.fields.temperature
        temp.boundary_values_imag[2, 1] = 0.125
        @test GeoDynamo.get_bc_vectors(temp).outer_imag[1] == 0.125
        temp.boundary_values_imag[2, 1] = 0.0
    end

    # ── G2: a correction must land where the SOLVE reads, cache or not ───────
    @testset "topography corrections reach the loaded BC cache" begin
        # `get_bc_vectors` PREFERS `boundary_interpolation_cache.bc_real/bc_imag`
        # whenever a spectral BC file has been loaded. The topography couplings wrote
        # unconditionally to `field.boundary_values`, so in exactly those runs the
        # implicit solve never saw a single correction — a silent no-op with no
        # warning anywhere.
        temp = state.fields.temperature
        nlm = size(temp.boundary_values, 2)
        cache = temp.boundary_interpolation_cache
        cache.bc_real = zeros(Float64, 2, nlm)
        cache.bc_imag = zeros(Float64, 2, nlm)
        cache.bc_loaded = true
        try
            # the reader now points at the cache, not at boundary_values
            @test GeoDynamo.get_bc_vectors(temp).outer_real === view(cache.bc_real, 2, :)

            # ...and so must the writer
            bv, bv_imag = GeoDynamo.bcs.active_boundary_arrays(temp)
            @test bv === cache.bc_real
            @test bv_imag === cache.bc_imag
        finally
            cache.bc_loaded = false
            cache.bc_real = nothing
            cache.bc_imag = nothing
        end

        # with no BC file loaded the field's own rows are authoritative again
        bv, bv_imag = GeoDynamo.bcs.active_boundary_arrays(temp)
        @test bv === temp.boundary_values
        @test bv_imag === temp.boundary_values_imag
    end
end
