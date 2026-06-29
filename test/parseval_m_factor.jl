using Test
using GeoDynamo
using MPI

MPI.Initialized() || MPI.Init()

# Parseval factor of 2 for m>0 modes.
#
# Spectral storage keeps only m>=0 of a REAL field. For orthonormal harmonics the
# angular integral is  ∫|f|² dΩ = Σ_lm (2 - δ_{m,0}) |c_lm|² : every m>0 coefficient
# carries DOUBLE the energy of an m=0 coefficient of the same magnitude (its -m
# conjugate partner is not stored separately). The spectral energy diagnostics
# omitted this, under-counting all non-axisymmetric energy by 2×.
#
# Test the relationship directly: the SAME coefficient placed in an m>0 mode must
# contribute exactly twice the energy it contributes in an m=0 mode of the same
# degree l (identical l(l+1) and radial weights).

const _PF_L = 2

function _find_mode(cfg, l, m)
    idx = findfirst(i -> cfg.l_values[i] == l && cfg.m_values[i] == m, 1:cfg.nlm)
    @assert idx !== nothing "mode (l=$l, m=$m) not present"
    return idx
end

function _set_single_real_mode!(spec, cfg, l, m, A)
    parent(spec.data_real) .= 0.0
    parent(spec.data_imag) .= 0.0
    idx = _find_mode(cfg, l, m)
    slot = GeoDynamo.local_spectral_storage_slot(cfg, idx)
    if slot !== nothing
        sr = parent(spec.data_real)
        for r in 1:size(sr, 3)
            GeoDynamo.set_local_spectral_value!(sr, slot, r, A)
        end
    end
    return spec
end

@testset "Parseval m>0 factor of 2 in spectral energies" begin
    params = GeoDynamo.SolverParameters(
        geometry = :shell, lmax = 4, mmax = 4, nlat = 10, nlon = 20, nr = 12,
        nr_inner = 4, radial_bandwidth = 3, radius_ratio = 0.35,
        Ek = 1e-3, Ra = 1e4, Pm = 1.0, Pr = 1.0, timestep = 1e-4,
        include_magnetic = true, include_composition = true,
        timestepper = GeoDynamo.CNAB2(),
    )
    st = GeoDynamo.initialize_solver_state(Float64; params)
    GeoDynamo.initialize_solver_fields!(st)
    dom = st.backend.outer_core_domain
    A = 0.3

    @testset "compute_thermal_energy" begin
        cfg = st.fields.temperature.spectral.config
        _set_single_real_mode!(st.fields.temperature.spectral, cfg, _PF_L, 0, A)
        E0 = GeoDynamo.compute_thermal_energy(st.fields.temperature)
        _set_single_real_mode!(st.fields.temperature.spectral, cfg, _PF_L, 1, A)
        E1 = GeoDynamo.compute_thermal_energy(st.fields.temperature)
        @test E0 > 0
        @test E1 ≈ 2 * E0
    end

    @testset "compute_kinetic_energy" begin
        cfg = st.fields.velocity.toroidal.config
        parent(st.fields.velocity.poloidal.data_real) .= 0.0
        parent(st.fields.velocity.poloidal.data_imag) .= 0.0
        _set_single_real_mode!(st.fields.velocity.toroidal, cfg, _PF_L, 0, A)
        E0 = GeoDynamo.compute_kinetic_energy(st.fields.velocity, dom)
        _set_single_real_mode!(st.fields.velocity.toroidal, cfg, _PF_L, 1, A)
        E1 = GeoDynamo.compute_kinetic_energy(st.fields.velocity, dom)
        @test E0 > 0
        @test E1 ≈ 2 * E0
    end

    @testset "compute_magnetic_energy" begin
        cfg = st.fields.magnetic.toroidal.config
        parent(st.fields.magnetic.poloidal.data_real) .= 0.0
        parent(st.fields.magnetic.poloidal.data_imag) .= 0.0
        _set_single_real_mode!(st.fields.magnetic.toroidal, cfg, _PF_L, 0, A)
        E0 = GeoDynamo.compute_magnetic_energy(st.fields.magnetic, dom)
        _set_single_real_mode!(st.fields.magnetic.toroidal, cfg, _PF_L, 1, A)
        E1 = GeoDynamo.compute_magnetic_energy(st.fields.magnetic, dom)
        @test E0 > 0
        @test E1 ≈ 2 * E0
    end
end
