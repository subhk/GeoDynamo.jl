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

function _seed_spec_nonzero!(field)
    fill!(parent(field.data_real), 1.0)
    fill!(parent(field.data_imag), -1.0)
    return field
end

function _spec_is_zero(field)
    return all(iszero, parent(field.data_real)) &&
           all(iszero, parent(field.data_imag))
end

function _seed_phys_nonzero!(field)
    fill!(parent(field.data), 1.0)
    return field
end

_phys_is_zero(field) = all(iszero, parent(field.data))

function _seed_vector_nonzero!(field)
    _seed_phys_nonzero!(field.r_component)
    _seed_phys_nonzero!(field.θ_component)
    _seed_phys_nonzero!(field.φ_component)
    return field
end

function _vector_is_zero(field)
    return _phys_is_zero(field.r_component) &&
           _phys_is_zero(field.θ_component) &&
           _phys_is_zero(field.φ_component)
end

@testset "ZeroIC validates and clears the complete selected field state" begin
    MPI.Initialized() || MPI.Init()

    grid = G.SphericalShellGrid(G.CPU(); lmax = 4, mmax = 4,
                                nlat = 12, nlon = 16, nr = 16, nr_inner = 4)
    model = G.GeodynamoModel(grid; Ek = 1e-2, Ra = 1e4,
                             include_magnetic = true, include_composition = true)
    G.initialize_fields!(model.state)

    temperature = model.temperature
    _seed_spec_nonzero!(temperature.spectral)
    _seed_spec_nonzero!(temperature.prev_nonlinear)
    _seed_phys_nonzero!(temperature.temperature)
    G.set_initial_condition!(model, :temperature, G.ZeroIC())
    @test _spec_is_zero(temperature.spectral)
    @test _spec_is_zero(temperature.prev_nonlinear)
    @test _phys_is_zero(temperature.temperature)

    composition = model.composition
    _seed_spec_nonzero!(composition.spectral)
    _seed_spec_nonzero!(composition.prev_nonlinear)
    _seed_phys_nonzero!(composition.composition)
    G.set_initial_condition!(model, :composition, G.ZeroIC())
    @test _spec_is_zero(composition.spectral)
    @test _spec_is_zero(composition.prev_nonlinear)
    @test _phys_is_zero(composition.composition)

    velocity = model.velocity
    _seed_spec_nonzero!(velocity.toroidal)
    _seed_spec_nonzero!(velocity.poloidal)
    _seed_spec_nonzero!(velocity.prev_nl_toroidal)
    _seed_vector_nonzero!(velocity.velocity)
    G.set_initial_condition!(model, :velocity, G.ZeroIC())
    @test _spec_is_zero(velocity.toroidal)
    @test _spec_is_zero(velocity.poloidal)
    @test _spec_is_zero(velocity.prev_nl_toroidal)
    @test _vector_is_zero(velocity.velocity)

    magnetic = model.magnetic
    _seed_spec_nonzero!(magnetic.toroidal)
    _seed_spec_nonzero!(magnetic.poloidal)
    _seed_spec_nonzero!(magnetic.toroidal_ic)
    _seed_spec_nonzero!(magnetic.prev_nl_poloidal)
    _seed_vector_nonzero!(magnetic.magnetic)
    G.set_initial_condition!(model, :magnetic, G.ZeroIC())
    @test _spec_is_zero(magnetic.toroidal)
    @test _spec_is_zero(magnetic.poloidal)
    @test _spec_is_zero(magnetic.toroidal_ic)
    @test _spec_is_zero(magnetic.prev_nl_poloidal)
    @test _vector_is_zero(magnetic.magnetic)

    for (field, kwargs) in ((:unknown, (;)),
                            (:magnetic, (; include_magnetic = false)),
                            (:composition, (; include_composition = false)))
        invalid_model = G.GeodynamoModel(grid; Ek = 1e-2, Ra = 1e4, kwargs...)
        @test !invalid_model.state.is_initialized
        @test_throws ArgumentError G.set_initial_condition!(invalid_model, field, G.ZeroIC())
        @test !invalid_model.state.is_initialized
    end
end
