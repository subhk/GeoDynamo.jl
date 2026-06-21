using Test
using GeoDynamo
using MPI

MPI.Initialized() || MPI.Init()

# The solver energy diagnostic (compute_total_energy! → report_energy_conservation)
# must report the physical volume-integral energy E = ½∫|f|² dV, NOT a raw
# count-weighted sum over grid points. The verified per-field reference is
# compute_kinetic_energy / compute_magnetic_energy (spectral: angular integral via
# l(l+1) Parseval, radial via r²·integration_weights). The buggy version summed
# the physical arrays with uniform (==1) weight.
#
# Contract: with temperature and composition zeroed, the recorded total energy
# must equal compute_kinetic_energy + compute_magnetic_energy.

function _seed_smooth!(field)
    sr = parent(field.data_real)
    nr = size(sr, 3)
    cfg = field.config
    for lm in 1:cfg.nlm
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue
        for r in 1:nr
            GeoDynamo.set_local_spectral_value!(sr, slot, r,
                sin(pi * (r - 1) / (nr - 1)) * 1e-3 / lm)
        end
    end
    return field
end

@testset "solver energy diagnostic is the volume integral (matches verified energies)" begin
    params = GeoDynamo.SolverParameters(
        geometry = :shell, lmax = 4, mmax = 4, nlat = 10, nlon = 20, nr = 12,
        nr_inner = 4, radial_bandwidth = 3, radius_ratio = 0.35,
        Ek = 1e-3, Ra = 1e4, Pm = 1.0, Pr = 1.0, timestep = 1e-4,
        include_magnetic = true, include_composition = true,
        timestepper = GeoDynamo.CNAB2(),
    )
    st = GeoDynamo.initialize_solver_state(Float64; params)
    GeoDynamo.initialize_solver_fields!(st)

    # Zero temperature + composition so total = kinetic + magnetic only.
    for f in (st.fields.temperature.spectral, st.fields.composition.spectral)
        parent(f.data_real) .= 0.0
        parent(f.data_imag) .= 0.0
    end
    # Seed nonzero, non-uniform velocity + magnetic so the weighting matters.
    _seed_smooth!(st.fields.velocity.toroidal)
    _seed_smooth!(st.fields.velocity.poloidal)
    _seed_smooth!(st.fields.magnetic.toroidal)
    _seed_smooth!(st.fields.magnetic.poloidal)

    dom = st.backend.outer_core_domain
    expected = GeoDynamo.compute_kinetic_energy(st.fields.velocity, dom) +
               GeoDynamo.compute_magnetic_energy(st.fields.magnetic, dom)

    GeoDynamo.compute_total_energy!(st)
    total = st.energy_tracker.total_energy[end]

    @test expected > 0
    @test total ≈ expected rtol = 1e-10
end
