using Test, GeoDynamo, MPI, PencilArrays

# Test that the LEGACY field-level scalar transforms (shtnskit_spectral_to_physical! /
# shtnskit_physical_to_spectral!) work correctly under r-distribution (r_ranks > 1).
# These predate the Phase-2 r×θ path and were previously guarded by
# _assert_not_r_distributed which errored them at r_ranks > 1.
# After the migration (delegate → scalar_spectral_to_physical! / scalar_physical_to_spectral!)
# they must roundtrip cleanly at all process-grid configurations.

MPI.Initialized() || MPI.Init()

@testset "legacy shtnskit scalar transforms roundtrip under r-distribution" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=20, nr=6)
    dom = GeoDynamo.create_radial_domain(6)
    tf  = GeoDynamo.create_shtns_temperature_field(Float64, cfg, dom)

    sr = parent(tf.spectral.data_real)
    si = parent(tf.spectral.data_imag)
    fill!(sr, 0.0)
    fill!(si, 0.0)

    # Seed deterministic spectral modes at valid (l >= m) positions only.
    # The spec pencil stores a (lmax+1)×(mmax+1)×nr sub-block per rank (contiguous l/m).
    l_range_spec = PencilArrays.range_local(cfg.pencils.spec)[1]  # global l-indices (1-based, l_global = l+1)
    m_range_spec = PencilArrays.range_local(cfg.pencils.spec)[2]  # global m-indices

    for k in 1:size(sr, 3)
        for (il, lg) in enumerate(l_range_spec)
            l = lg - 1   # 0-based degree
            for (im, mg) in enumerate(m_range_spec)
                m = mg - 1   # 0-based order
                l < m && continue   # skip invalid (l < m) slots
                if l == 2 && m == 0
                    sr[il, im, k] = 0.7
                elseif l == 3 && m == 1
                    sr[il, im, k] = 0.3
                    si[il, im, k] = 0.1
                end
            end
        end
    end

    sr0 = copy(sr)
    si0 = copy(si)

    # Call LEGACY transforms (these used to error under r_ranks > 1 via _assert_not_r_distributed).
    GeoDynamo.shtnskit_spectral_to_physical!(tf.spectral, tf.temperature)
    GeoDynamo.shtnskit_physical_to_spectral!(tf.temperature, tf.spectral)

    err_real = maximum(abs.(parent(tf.spectral.data_real) .- sr0))
    err_imag = maximum(abs.(parent(tf.spectral.data_imag) .- si0))

    @test err_real < 1e-10
    @test err_imag < 1e-10
end
