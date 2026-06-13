using Test
using MPI
using Random
using GeoDynamo
const G = GeoDynamo

# Regression: randomize_scalar_field! / randomize_vector_field! /
# randomize_magnetic_field! must populate the FULL (l, m) spectrum up to `lmax`,
# not only the m = 0 column. The legacy implementation indexed spectral storage
# with the stale `[idx, 1, r]` idiom (dim-2 hard-pinned to slot 1 = m = 0), so a
# "random" perturbation was silently axisymmetric: every m > 0 coefficient stayed
# exactly zero. These tests fail against that bug and pass once the functions
# index modes via local_spectral_storage_slot(cfg, lm).

# Sum of |coeff| over every owned mode with m > 0 (real part; randomize keeps
# imag = 0). Nonzero ⇒ the non-axisymmetric spectrum was actually populated.
function _nonaxisymmetric_energy(cfg, spec)
    real3 = parent(spec.data_real)
    nr = size(real3, 3)
    acc = 0.0
    for lm in 1:cfg.nlm
        cfg.m_values[lm] > 0 || continue          # only m > 0 modes
        slot = G.local_spectral_storage_slot(cfg, lm)
        slot === nothing && continue
        for r in 1:nr
            acc += abs(G.local_spectral_value(real3, slot, r))
        end
    end
    return acc
end

@testset "randomize_* populates the full (l,m) spectrum, not just m=0" begin
    MPI.Initialized() || MPI.Init()

    lmax = 4; mmax = 4
    nlat = max(lmax + 2, 10); nlon = max(2lmax + 1, 16); nr = 8
    cfg = G.create_shtnskit_config(lmax = lmax, mmax = mmax,
                                   nlat = nlat, nlon = nlon, nr = nr)
    dom = G.create_radial_domain(nr)

    @testset "scalar field" begin
        Random.seed!(20260613)
        comp = G.create_shtns_composition_field(Float64, cfg, dom)
        G.randomize_scalar_field!(comp; amplitude = 1.0, lmax = lmax)
        @test _nonaxisymmetric_energy(cfg, comp.spectral) > 0.0
    end

    @testset "vector field (toroidal + poloidal)" begin
        Random.seed!(20260613)
        vel = G.create_shtns_velocity_fields(Float64, cfg, dom)
        G.randomize_vector_field!(vel; amplitude = 1.0, lmax = lmax)
        @test _nonaxisymmetric_energy(cfg, vel.toroidal) > 0.0
        @test _nonaxisymmetric_energy(cfg, vel.poloidal) > 0.0
    end

    @testset "magnetic field (toroidal + poloidal)" begin
        Random.seed!(20260613)
        mag = G.create_shtns_magnetic_fields(Float64, cfg, dom, dom)
        G.randomize_magnetic_field!(mag; amplitude = 1.0, lmax = lmax)
        @test _nonaxisymmetric_energy(cfg, mag.toroidal) > 0.0
        @test _nonaxisymmetric_energy(cfg, mag.poloidal) > 0.0
    end

    # Same m=0-only bug class in the generate_random_*! family (core/initial_conditions.jl).
    # These looped over get_local_range(pencil, 1) (the l_slot axis 1:lmax+1) and fed those
    # as mode indices, so only m=0 modes were touched. Random ICs were silently axisymmetric.
    @testset "generate_random_*! populates the full (l,m) spectrum" begin
        IC = G.InitialConditions
        @testset "temperature" begin
            Random.seed!(7)
            temp = G.create_shtns_temperature_field(Float64, cfg, dom)
            IC.generate_random_temperature!(temp, 0.5, 1:lmax)
            @test _nonaxisymmetric_energy(cfg, temp.spectral) > 0.0
        end
        @testset "composition" begin
            Random.seed!(7)
            comp = G.create_shtns_composition_field(Float64, cfg, dom)
            IC.generate_random_composition!(comp, 0.5, 1:lmax)
            @test _nonaxisymmetric_energy(cfg, comp.spectral) > 0.0
        end
        @testset "velocity" begin
            Random.seed!(7)
            vel = G.create_shtns_velocity_fields(Float64, cfg, dom)
            IC.generate_random_velocity!(vel, 0.5, 1:lmax)
            @test _nonaxisymmetric_energy(cfg, vel.toroidal) > 0.0
            @test _nonaxisymmetric_energy(cfg, vel.poloidal) > 0.0
        end
        @testset "magnetic" begin
            Random.seed!(7)
            mag = G.create_shtns_magnetic_fields(Float64, cfg, dom, dom)
            IC.generate_random_magnetic!(mag, 0.5, 1:lmax)
            @test _nonaxisymmetric_energy(cfg, mag.toroidal) > 0.0
            @test _nonaxisymmetric_energy(cfg, mag.poloidal) > 0.0
        end
    end
end
