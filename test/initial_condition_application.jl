using Test
using MPI
using Random
using LinearAlgebra
using GeoDynamo
const G = GeoDynamo

# Regression: a user-supplied initial condition (via set! / set_initial_condition!
# or the model `initial_conditions=` kwarg) must NOT be overwritten by the solver's
# default field initialization on the first step. Historically set_initial_condition!
# never set state.is_initialized, so the first solver_step! ran
# initialize_solver_fields! (fill!(0) + default conductive IC), erasing the user IC.
# The default temperature IC is axisymmetric (m=0 only), so a user RandomPerturbation
# carrying m>0 content is the discriminator: it survives one step iff it was not
# clobbered.

function _scalar_nonaxisym_energy(cfg, spec)
    real3 = parent(spec.data_real)
    imag3 = parent(spec.data_imag)
    nr = size(real3, 3)
    acc = 0.0
    for lm in 1:cfg.nlm
        cfg.m_values[lm] > 0 || continue
        slot = G.local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue
        for r in 1:nr
            acc += abs(G.local_spectral_value(real3, slot, r)) +
                   abs(G.local_spectral_value(imag3, slot, r))
        end
    end
    return acc
end

@testset "user IC survives the first step (not clobbered by default init)" begin
    MPI.Initialized() || MPI.Init()

    grid = G.SphericalShellGrid(G.CPU(); lmax = 4, mmax = 4,
                                nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
    model = G.GeodynamoModel(grid; Ek = 1e-2, Ra = 1e4,
                             include_magnetic = false, include_composition = false)

    Random.seed!(99)
    G.set!(model; temperature = G.RandomPerturbation(amplitude = 0.5, lmax = 4))

    cfg = model.state.backend.shtns_config
    spec = model.state.fields.temperature.spectral
    e_set = _scalar_nonaxisym_energy(cfg, spec)
    @test e_set > 0.0                  # set! actually wrote m>0 content
    @test model.state.is_initialized   # set! must mark the state initialized

    # One step must evolve the user IC CONTINUOUSLY (change ~ O(dt)). A clobber
    # would replace it with the default conductive profile (an O(1) jump) before
    # stepping, so the relative change discriminates fix from bug.
    snapshot = copy(parent(spec.data_real))
    G.time_step!(model, 1e-4)
    after = parent(spec.data_real)
    rel = norm(after .- snapshot) / max(norm(snapshot), eps())
    @test rel < 0.5                    # user IC survived; not replaced by default init
end
