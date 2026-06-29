using Test
using GeoDynamo
using MPI

MPI.Initialized() || MPI.Init()

# Topography boundary corrections are written into each field's `boundary_values`
# (and, for magnetic, `boundary_values_imag`). These tests verify the corrections
# are actually CONSUMED by the implicit solves — they were previously computed and
# silently dropped. We drive the boundary rows directly (bypassing the topography
# math) and assert the imposed boundary value appears in the solved field.

function _mode_idx(cfg, l, m)
    idx = findfirst(i -> cfg.l_values[i] == l && cfg.m_values[i] == m, 1:cfg.nlm)
    @assert idx !== nothing
    return idx
end

_inner_val(spec, cfg, idx) = begin
    slot = GeoDynamo.local_spectral_storage_slot(cfg, idx)
    GeoDynamo.local_spectral_value(parent(spec.data_real), slot, 1)
end

function _make_state()
    params = GeoDynamo.SolverParameters(
        geometry = :shell, lmax = 4, mmax = 4, nlat = 10, nlon = 20, nr = 12,
        nr_inner = 4, radial_bandwidth = 3, radius_ratio = 0.35,
        Ek = 1e-3, Ra = 1e4, Pm = 1.0, Pr = 1.0, timestep = 1e-4,
        include_magnetic = true, include_composition = true,
        timestepper = GeoDynamo.CNAB2(),
    )
    st = GeoDynamo.initialize_solver_state(Float64; params)
    GeoDynamo.initialize_solver_fields!(st)
    return st
end

@testset "topography boundary-value injection reaches the solve" begin
    @testset "velocity toroidal" begin
        st = _make_state()
        v = st.fields.velocity
        cfg = v.toroidal.config
        idx = _mode_idx(cfg, 2, 1)          # not the l=1,m=0 rotation mode
        V = 0.137

        for f in (v.toroidal, v.poloidal, v.nl_toroidal, v.prev_nl_toroidal,
            v.nl_poloidal, v.prev_nl_poloidal)
            parent(f.data_real) .= 0.0
            parent(f.data_imag) .= 0.0
        end
        v.toroidal.boundary_values[1, idx] = V   # inner-boundary value for this mode

        GeoDynamo.apply_velocity_toroidal_implicit_update!(st)

        # No-slip toroidal inner row is identity ⇒ T[inner] == imposed value.
        @test _inner_val(v.toroidal, cfg, idx) ≈ V
    end

    @testset "velocity poloidal (W-split impermeability)" begin
        st = _make_state()
        v = st.fields.velocity
        cfg = v.poloidal.config
        idx = _mode_idx(cfg, 2, 0)
        V = 0.091

        for f in (v.toroidal, v.poloidal, v.nl_toroidal, v.prev_nl_toroidal,
            v.nl_poloidal, v.prev_nl_poloidal)
            parent(f.data_real) .= 0.0
            parent(f.data_imag) .= 0.0
        end
        v.poloidal.boundary_values[1, idx] = V   # inner P wall value

        GeoDynamo.apply_velocity_poloidal_implicit_update!(st)

        # P-recovery inner row is Dirichlet ⇒ P[inner] == imposed value.
        @test _inner_val(v.poloidal, cfg, idx) ≈ V
    end

    @testset "magnetic toroidal (real + imaginary)" begin
        st = _make_state()
        b = st.fields.magnetic
        cfg = b.toroidal.config
        idx = _mode_idx(cfg, 2, 1)
        Vr_in, Vr_out, Vi_in, Vi_out = 0.11, -0.07, 0.05, 0.13

        for f in (b.toroidal, b.poloidal, b.nl_toroidal, b.prev_nl_toroidal,
            b.nl_poloidal, b.prev_nl_poloidal)
            parent(f.data_real) .= 0.0
            parent(f.data_imag) .= 0.0
        end
        b.toroidal.boundary_values[1, idx] = Vr_in
        b.toroidal.boundary_values[2, idx] = Vr_out
        b.toroidal.boundary_values_imag[1, idx] = Vi_in
        b.toroidal.boundary_values_imag[2, idx] = Vi_out

        GeoDynamo.apply_magnetic_toroidal_implicit_update!(st)

        slot = GeoDynamo.local_spectral_storage_slot(cfg, idx)
        sr = parent(b.toroidal.data_real)
        si = parent(b.toroidal.data_imag)
        nr = size(sr, 3)
        # Insulating toroidal BC rows are identity ⇒ endpoints equal imposed values.
        @test GeoDynamo.local_spectral_value(sr, slot, 1) ≈ Vr_in
        @test GeoDynamo.local_spectral_value(sr, slot, nr) ≈ Vr_out
        @test GeoDynamo.local_spectral_value(si, slot, 1) ≈ Vi_in
        @test GeoDynamo.local_spectral_value(si, slot, nr) ≈ Vi_out
    end
end
