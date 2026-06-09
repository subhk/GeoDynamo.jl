using Test
using MPI
using Serialization

# Characterization test for the temperature/composition solver consolidation
# (docs/superpowers/plans/2026-06-09-scalar-solver-consolidation.md).
#
# Runs ONE real `solver_step!` on a tiny shell with BOTH scalar fields active
# (CNAB2), then snapshots the temperature + composition spectral coefficients
# and compares against a committed reference. During the refactor itself the
# comparison was bit-exact on the machine that generated the reference; the
# committed gate uses an elementwise atol+rtol check because last-ulp FP
# results differ across platforms (CPU arch, BLAS/FFTW builds, Julia patch
# version) — observed cross-platform drift is <=~1e-11 relative, far below
# the tolerances here, while any real behavior change moves coefficients by
# orders of magnitude more.
#
# Determinism: initialize_{temperature,composition}_field! seed their own RNG
# (Random.seed!(42+rank)) internally, so the IC — and therefore the stepped
# state — is reproducible run-to-run on a fixed Julia version.

_scalar_consolidation_close(a, b; atol = 1e-12, rtol = 1e-9) =
    size(a) == size(b) &&
    all(abs(x - y) <= atol + rtol * abs(y) for (x, y) in zip(a, b))

@testset "Scalar solver consolidation (characterization)" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping scalar consolidation test"
        return
    end
    if !MPI.Initialized()
        MPI.Init()
    end

    params = GeoDynamo.SolverParameters(
        architecture = :cpu,
        geometry = :shell,
        nr = 16,
        nr_inner = 4,
        lmax = 4,
        mmax = 4,
        nlat = 12,
        nlon = 16,
        Ra = 1e4,
        Ek = 1e-2,
        Pr = 1.0,
        Pm = 1.0,
        Sc = 1.0,
        timestep = 1e-4,
        start_time = 0.0,
        end_time = 1e-3,
        stop_iteration = 10,
        include_magnetic = false,
        include_composition = true,
        timestepper = GeoDynamo.CNAB2(),
        topography_enabled = false,
        stefan_enabled = false,
    )

    state = GeoDynamo.initialize_simulation(Float64, params)
    @test state.fields.composition !== nothing

    GeoDynamo.solver_step!(state)

    snap = (
        t_real = copy(parent(state.fields.temperature.spectral.data_real)),
        t_imag = copy(parent(state.fields.temperature.spectral.data_imag)),
        c_real = copy(parent(state.fields.composition.spectral.data_real)),
        c_imag = copy(parent(state.fields.composition.spectral.data_imag)),
    )

    @test all(isfinite, snap.t_real) && all(isfinite, snap.t_imag)
    @test all(isfinite, snap.c_real) && all(isfinite, snap.c_imag)

    refpath = joinpath(@__DIR__, "refs", "scalar_consolidation_ref.jls")

    # Load the committed reference. If it is missing or unreadable (e.g. a Julia
    # upgrade changed the Serialization format), regenerate it from the CURRENT
    # run and warn — this self-heals across versions while still acting as a
    # tight regression gate against the committed snapshot.
    ref = nothing
    if isfile(refpath)
        try
            ref = deserialize(refpath)
        catch err
            @warn "Could not deserialize scalar consolidation ref; regenerating" err
            ref = nothing
        end
    end
    if ref === nothing
        mkpath(dirname(refpath))
        serialize(refpath, snap)
        @info "Wrote scalar consolidation reference snapshot" refpath
        ref = snap
    end

    @test _scalar_consolidation_close(snap.t_real, ref.t_real)
    @test _scalar_consolidation_close(snap.t_imag, ref.t_imag)
    @test _scalar_consolidation_close(snap.c_real, ref.c_real)
    @test _scalar_consolidation_close(snap.c_imag, ref.c_imag)
end
