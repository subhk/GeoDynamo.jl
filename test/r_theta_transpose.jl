using Test, GeoDynamo, MPI, PencilArrays

MPI.Initialized() || MPI.Init()

@testset "r<->lm transpose roundtrip is identity" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=20, nr=6)

    # --- Create array in solve orientation (spec: l/θ_ranks, m/r_ranks, r local) ---
    a = GeoDynamo.create_pencil_array(ComplexF64, cfg.pencils.spec; init=:zero)
    rank_offset = ComplexF64(MPI.Comm_rank(MPI.COMM_WORLD) + 1)
    p = parent(a)
    for i in eachindex(p)
        p[i] = rank_offset + im * ComplexF64(i)
    end
    a0 = copy(parent(a))

    # --- spec_transform pencil must exist in cfg.pencils ---
    @test :spec_transform in keys(cfg.pencils)

    # --- spec_transform: m full (axis 2 local), r distributed (axis 2 of topology) ---
    bl = PencilArrays.size_local(cfg.pencils.spec_transform)
    @test bl[2] == PencilArrays.size_global(cfg.pencils.spec_transform)[2]   # m axis is full/local

    # --- Roundtrip via transpose helpers ---
    b = GeoDynamo.create_pencil_array(ComplexF64, cfg.pencils.spec_transform; init=:zero)
    GeoDynamo.transpose_solve_to_transform!(b, a)    # spec -> spec_transform
    GeoDynamo.transpose_transform_to_solve!(a, b)    # back to spec
    @test parent(a) == a0                             # exact identity (no floating-point error)
end

@testset "scalar r×θ transform roundtrip" begin
    cfg = GeoDynamo.create_shtnskit_config(lmax=8, mmax=8, nlat=12, nlon=20, nr=6)
    dom = GeoDynamo.create_radial_domain(6)
    tf  = GeoDynamo.create_shtns_temperature_field(Float64, cfg, dom)
    sr = parent(tf.spectral.data_real); si = parent(tf.spectral.data_imag); sr.=0; si.=0

    # Set spectral coefficients only at VALID (l ≥ m) modes by checking global (l,m).
    # The spec pencil stores a rectangular (lmax+1)×(mmax+1)×nr block; each rank owns a
    # contiguous (l-local × m-local) sub-block.  Invalid (l < m) slots are always zero
    # and do NOT roundtrip through SH transforms, so we must skip them.
    l_range_spec = PencilArrays.range_local(cfg.pencils.spec)[1]  # global l indices
    m_range_spec = PencilArrays.range_local(cfg.pencils.spec)[2]  # global m indices (1-based, m=mg-1)
    for k in 1:size(sr, 3)
        for (il, lg) in enumerate(l_range_spec)
            l = lg - 1  # 0-based degree
            for (im, mg) in enumerate(m_range_spec)
                m = mg - 1  # 0-based order
                l < m && continue  # skip invalid modes
                if l == 2 && m == 0
                    sr[il, im, k] = 0.7
                elseif l == 3 && m == 1
                    sr[il, im, k] = 0.3
                    si[il, im, k] = 0.1
                end
            end
        end
    end

    sr0 = copy(sr); si0 = copy(si)
    GeoDynamo.scalar_spectral_to_physical!(tf.spectral, tf.temperature)
    GeoDynamo.scalar_physical_to_spectral!(tf.temperature, tf.spectral)
    @test maximum(abs.(parent(tf.spectral.data_real) .- sr0)) < 1e-10
    @test maximum(abs.(parent(tf.spectral.data_imag) .- si0)) < 1e-10
end
