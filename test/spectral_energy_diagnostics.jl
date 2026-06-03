using Test
using MPI

const FINALIZE_MPI_SPEC_DIAG = get(ENV, "GEODYNAMO_TEST_MPI_FINALIZE", "true") == "true"

# Tests for compute_spectral_energy_diagnostics! correctness.
#
# Bug 1 (indexing): compute_spectral_energy_diagnostics! iterated
# `enumerate(config.l_values)` (packed nlm mode-list, length nlm) and indexed
# `real_part[idx, j, k]` where dim-1 of real_part is the spec pencil's l-slot
# axis (size lmax+1).  The `idx <= size(real_part,1)` guard truncated to the
# first lmax+1 nlm-list entries.  In the Phase-3 (lmax+1, mmax+1, nr) storage
# layout the first lmax+1 entries happen to be the m=0 block, so the l-labeling
# accidentally stays correct but all m>0 modes get their energy summed under the
# WRONG l (or truncated on deeper architectures).
#
# Bug 2 (MPI): l_energies was computed locally per rank and never MPI-reduced.
# In multi-rank runs each rank only sees its own m-slab, so peak_l/centroid
# are wrong (or missing) on ranks that don't own the highest-energy m-slots.
# Observable: put energy at l=5,m=5 in a 2x1 run; rank 0 (m=0..3) sees NO
# energy -> peak_l missing; rank 1 (m=4..8) sees it correctly -> peak_l=5.
#
# Fix: map storage slots via local_spectral_lm_map (correct indexing for every
# layout) + Allreduce l_energies over get_comm() (correct multi-rank reduction).

@testset "spectral_energy_diagnostics correctness" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping spectral_energy_diagnostics tests"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    comm = GeoDynamo.get_comm()
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    lmax = 8
    mmax = 8
    nlat = lmax + 2
    nlon = 2 * lmax + 2
    nr = 4

    cfg = GeoDynamo.create_shtnskit_config(
        lmax = lmax, mmax = mmax, nlat = nlat, nlon = nlon, nr = nr)

    lm_map = GeoDynamo.local_spectral_lm_map(cfg)
    loc_shape = size(lm_map)  # (lmax+1, local_mmax+1, nr) locally

    # ----------------------------------------------------------------
    # Helper: build a local spec-storage array with energy at a given
    # target (l_target, m_target) pair.  Uses lm_map for layout-correct
    # slot placement on every rank / Phase.  Returns (real_part, imag_part,
    # n_placed) where n_placed is the count of modes placed by THIS rank.
    # ----------------------------------------------------------------
    function make_spec_at_lm(cfg, nr, l_target, m_target; amplitude = 1.0)
        lm_map = GeoDynamo.local_spectral_lm_map(cfg)
        ls = size(lm_map)
        rp = zeros(Float64, ls[1], ls[2], nr)
        ip = zeros(Float64, ls[1], ls[2], nr)
        placed = 0
        for idx in CartesianIndices(lm_map)
            mode = lm_map[idx]
            mode == 0 && continue
            l = cfg.l_values[mode]
            m = cfg.m_values[mode]
            if l == l_target && m == m_target
                for k in 1:nr
                    rp[idx[1], idx[2], k] = amplitude
                end
                placed += 1
            end
        end
        return rp, ip, placed
    end

    # ----------------------------------------------------------------
    # Helper: total energy at l_target across all ranks via Allreduce.
    # Counts (real^2 + imag^2) for modes whose lm_map slot maps to l_target.
    # This is the ground truth we use to validate diagnostics.
    # ----------------------------------------------------------------
    function true_total_energy_at_l(cfg, nr, real_part, imag_part, l_target)
        lm_map = GeoDynamo.local_spectral_lm_map(cfg)
        local_energy = 0.0
        for idx in CartesianIndices(lm_map)
            mode = lm_map[idx]
            mode == 0 && continue
            l = cfg.l_values[mode]
            l == l_target || continue
            for k in 1:nr
                local_energy += real_part[idx[1], idx[2], k]^2 +
                                imag_part[idx[1], idx[2], k]^2
            end
        end
        comm = GeoDynamo.get_comm()
        return (comm !== nothing && MPI.Comm_size(comm) > 1) ?
               MPI.Allreduce(local_energy, MPI.SUM, comm) : local_energy
    end

    # ----------------------------------------------------------------
    # Test 1: Energy at l=5, m=5 (a high-m mode).
    # In 2x1 layout (θ_ranks=2) rank 0 owns m-slots 0..3, rank 1 owns 4..8.
    # So l=5,m=5 lives entirely on rank 1.  Without the MPI reduction,
    # rank 0 sees no energy -> no diagnostic keys.  With the fix, Allreduce
    # makes the result consistent across all ranks: peak_l=5 everywhere.
    # In single-rank mode this also tests the lm_map-based indexing path.
    # ----------------------------------------------------------------
    @testset "multi-rank: l=5,m=5 energy detected on all ranks after Allreduce" begin
        l_target = 5
        m_target = 5
        real_part, imag_part, n_placed_local = make_spec_at_lm(cfg, nr, l_target, m_target)

        # Reduce to check at least one rank placed the mode
        total_placed = (nranks > 1) ?
                       MPI.Allreduce(n_placed_local, MPI.SUM, comm) : n_placed_local
        @test total_placed >= 1

        fields = Dict{String, Any}(
            "velocity_toroidal" => Dict(
                "real" => real_part,
                "imag" => imag_part
            )
        )
        fi = GeoDynamo.extract_field_info(fields, cfg, nothing; radius_ratio = 0.35)

        diagnostics = Dict{String, Float64}()
        GeoDynamo.compute_spectral_energy_diagnostics!(
            diagnostics, "velocity_toroidal", real_part, imag_part, fi)

        # All ranks must agree: peak_l == 5 and centroid ≈ 5.
        # Without the Allreduce, rank 0 in 2x1 layout would have MISSING keys.
        @test haskey(diagnostics, "velocity_toroidal_peak_l")
        @test diagnostics["velocity_toroidal_peak_l"] == Float64(l_target)
        @test haskey(diagnostics, "velocity_toroidal_spectral_centroid")
        @test diagnostics["velocity_toroidal_spectral_centroid"] ≈ Float64(l_target) atol = 1e-10

        # Verify the total energy sums to the expected amount.
        # Each placed slot has amplitude=1 in real, 0 in imag, over nr levels.
        expected_total_energy = Float64(nr * total_placed)  # nr levels * n_modes_at_l5m5
        @test diagnostics["velocity_toroidal_low_mode_fraction"] ≈ 1.0 atol = 1e-10
    end

    # ----------------------------------------------------------------
    # Test 2: Energy at l=5, m=3 (another m>0 mode).
    # Checks that the lm_map indexing assigns energy to the correct l.
    # ----------------------------------------------------------------
    @testset "lm_map indexing: l=5,m=3 energy attributed to l=5" begin
        l_target = 5
        m_target = 3
        real_part, imag_part, n_placed_local = make_spec_at_lm(cfg, nr, l_target, m_target)
        total_placed = (nranks > 1) ?
                       MPI.Allreduce(n_placed_local, MPI.SUM, comm) : n_placed_local
        @test total_placed >= 1

        fields = Dict{String, Any}(
            "velocity_poloidal" => Dict("real" => real_part, "imag" => imag_part)
        )
        fi = GeoDynamo.extract_field_info(fields, cfg, nothing; radius_ratio = 0.35)

        diagnostics = Dict{String, Float64}()
        GeoDynamo.compute_spectral_energy_diagnostics!(
            diagnostics, "velocity_poloidal", real_part, imag_part, fi)

        @test haskey(diagnostics, "velocity_poloidal_peak_l")
        @test diagnostics["velocity_poloidal_peak_l"] == Float64(l_target)
        @test diagnostics["velocity_poloidal_spectral_centroid"] ≈ Float64(l_target) atol = 1e-10
    end

    # ----------------------------------------------------------------
    # Test 3: Energy at l=5 across ALL m (all modes locally owned),
    # compared to a ground-truth Allreduce to verify the diagnostics
    # sum is globally consistent.
    # ----------------------------------------------------------------
    @testset "all-m energy at l=5: diagnostics match global ground truth" begin
        l_target = 5
        # Build field with all l=5 modes set to amplitude 1.0
        lm_map_local = GeoDynamo.local_spectral_lm_map(cfg)
        ls = size(lm_map_local)
        real_part = zeros(Float64, ls[1], ls[2], nr)
        imag_part = zeros(Float64, ls[1], ls[2], nr)
        for idx in CartesianIndices(lm_map_local)
            mode = lm_map_local[idx]
            mode == 0 && continue
            l = cfg.l_values[mode]
            if l == l_target
                for k in 1:nr
                    real_part[idx[1], idx[2], k] = 1.0
                end
            end
        end

        # Ground-truth total energy at l_target (globally reduced)
        e_true = true_total_energy_at_l(cfg, nr, real_part, imag_part, l_target)

        fields = Dict{String, Any}(
            "temperature_spectral" => Dict("real" => real_part, "imag" => imag_part)
        )
        fi = GeoDynamo.extract_field_info(fields, cfg, nothing; radius_ratio = 0.35)

        diagnostics = Dict{String, Float64}()
        GeoDynamo.compute_spectral_energy_diagnostics!(
            diagnostics, "temperature_spectral", real_part, imag_part, fi)

        @test haskey(diagnostics, "temperature_spectral_peak_l")
        @test diagnostics["temperature_spectral_peak_l"] == Float64(l_target)
        @test diagnostics["temperature_spectral_spectral_centroid"] ≈ Float64(l_target) atol = 1e-10
        # All energy is at l=5, low_mode_fraction should be 1 (5 <= 10)
        @test diagnostics["temperature_spectral_low_mode_fraction"] ≈ 1.0 atol = 1e-10
    end

    # ----------------------------------------------------------------
    # Test 4: zero field -> no diagnostic keys (no division by zero)
    # ----------------------------------------------------------------
    @testset "zero field produces no diagnostic keys" begin
        real_part = zeros(Float64, loc_shape[1], loc_shape[2], nr)
        imag_part = zeros(Float64, loc_shape[1], loc_shape[2], nr)

        fields = Dict{String, Any}(
            "velocity_toroidal" => Dict("real" => real_part, "imag" => imag_part)
        )
        fi = GeoDynamo.extract_field_info(fields, cfg, nothing; radius_ratio = 0.35)

        diagnostics = Dict{String, Float64}()
        GeoDynamo.compute_spectral_energy_diagnostics!(
            diagnostics, "velocity_toroidal", real_part, imag_part, fi)

        @test !haskey(diagnostics, "velocity_toroidal_peak_l")
        @test !haskey(diagnostics, "velocity_toroidal_spectral_centroid")
    end

    # ----------------------------------------------------------------
    # Test 5: two l-values with different mode counts.
    # l=3 has 4 modes (m=0..3), l=1 has 2 modes (m=0..1).
    # Equal amplitude -> l=3 has 2x energy -> peak_l == 3.
    # ----------------------------------------------------------------
    @testset "peak_l picks the l with higher total energy (across ranks)" begin
        lm_map_local = GeoDynamo.local_spectral_lm_map(cfg)
        ls = size(lm_map_local)
        real_part = zeros(Float64, ls[1], ls[2], nr)
        imag_part = zeros(Float64, ls[1], ls[2], nr)
        for idx in CartesianIndices(lm_map_local)
            mode = lm_map_local[idx]
            mode == 0 && continue
            l = cfg.l_values[mode]
            if l == 1 || l == 3
                for k in 1:nr
                    real_part[idx[1], idx[2], k] = 1.0
                end
            end
        end

        fields = Dict{String, Any}(
            "composition_spectral" => Dict("real" => real_part, "imag" => imag_part)
        )
        fi = GeoDynamo.extract_field_info(fields, cfg, nothing; radius_ratio = 0.35)

        diagnostics = Dict{String, Float64}()
        GeoDynamo.compute_spectral_energy_diagnostics!(
            diagnostics, "composition_spectral", real_part, imag_part, fi)

        # l=3 has 4 modes vs l=1's 2 -> l=3 dominates
        @test diagnostics["composition_spectral_peak_l"] == 3.0

        # centroid: l=3 energy = 4 modes * nr; l=1 energy = 2 modes * nr
        e1 = true_total_energy_at_l(cfg, nr, real_part, imag_part, 1)
        e3 = true_total_energy_at_l(cfg, nr, real_part, imag_part, 3)
        expected_centroid = (1.0 * e1 + 3.0 * e3) / (e1 + e3)
        @test diagnostics["composition_spectral_spectral_centroid"] ≈ expected_centroid atol = 1e-10
    end

    if MPI.Initialized()
        MPI.Barrier(GeoDynamo.get_comm())
        if FINALIZE_MPI_SPEC_DIAG && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end
