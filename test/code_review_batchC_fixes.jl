# ================================================================================
# Regression tests for batch C of the max-effort src/ review — the cleanup findings
# that are really correctness bugs, plus the mechanically safe duplications.
# ================================================================================
#
#   C29 physics/nonlinear.jl:18       theta-gradient recurrence uses A_±(l) — the WRONG
#                                    l-argument — where the corrected sibling in
#                                    scalar_operators.jl uses A_∓(l±1)
#   C28 gpu/scalar_gradient.jl:42     the GPU kernel mirrors the same wrong coefficients
#   C50 fields/transforms.jl:1074     `::Matrix{Float64}` assertion on a buffer created
#                                    with eltype(phys_data) throws for Float32 fields
#   C36 gpu/device_state.jl:136       _build_wsplit_pack duplicates all of _pack_wsplit
#   C34 fields/transforms.jl:1090     extract_physical_slice_generic! is a copy of
#                                    extract_physical_slice_phi_local!
#   C40 bcs/bcs.jl:112                hand-rolled _mean/_std + a fake `module _Statistics`
#   C37 core/initial_conditions.jl:133 randomize_magnetic_field! is a verbatim copy of
#                                    randomize_vector_field!
#   C38 core/initial_conditions.jl:95 `imag3` bound and never read in three functions
# ================================================================================

using Test
using MPI
using Random
using Statistics
using GeoDynamo

_crc_wsn(s) = replace(s, r"\s+" => "")
_crc_occ(pat::AbstractString, src) = occursin(_crc_wsn(pat), _crc_wsn(src))
const CRC_IC_SRC = read(
    joinpath(normpath(joinpath(@__DIR__, "..")), "src", "core", "initial_conditions.jl"),
    String)

# The standard recurrence sinθ·∂θY_l^m = A₊(l)·Y_{l+1}^m + A₋(l)·Y_{l-1}^m.
# Collecting the Y_l term of Σ a_{l'}·sinθ∂θY_{l'} gives the OUTPUT coefficient
#   b_l = A₊(l-1)·a_{l-1} + A₋(l+1)·a_{l+1},
# so a single input mode a_L = 1 must produce exactly b_{L+1} = A₊(L) and
# b_{L-1} = A₋(L). That is an anchor independent of either implementation.
_A_plus(l, m) = l * sqrt(((l + abs(m) + 1) * (l - abs(m) + 1)) / ((2l + 1) * (2l + 3)))
_A_minus(l, m) = -(l + 1) * sqrt(((l + abs(m)) * (l - abs(m))) / ((2l - 1) * (2l + 1)))

@testset "Max-effort review batch C" begin
    if MPI.Finalized()
        @warn "MPI already finalized; skipping batch C fix tests"
        return
    end
    MPI.Initialized() || MPI.Init()

    # ── C28: the GPU theta-gradient kernel coefficients ───────────────────────
    @testset "C28 gpu_theta_gradient! uses A∓(l±1)" begin
        lmax = 6
        nl = lmax + 1
        nm = lmax + 1
        nr = 1
        L = 3
        m = 0
        s_r = zeros(Float64, nl, nm, nr)
        s_i = zeros(Float64, nl, nm, nr)
        s_r[L + 1, m + 1, 1] = 1.0          # single input mode a_L = 1
        g_r = zeros(Float64, nl, nm, nr)
        g_i = zeros(Float64, nl, nm, nr)

        GeoDynamo.gpu_theta_gradient!(g_r, g_i, s_r, s_i, lmax)

        @test g_r[L + 2, m + 1, 1] ≈ _A_plus(L, m)     # b_{L+1} = A₊(L)
        @test g_r[L, m + 1, 1] ≈ _A_minus(L, m)        # b_{L-1} = A₋(L)
        # no other output mode is touched by a single input mode
        for li in 1:nl
            (li == L || li == L + 2) && continue
            @test g_r[li, m + 1, 1] == 0.0
        end
    end

    # ── C29: the CPU (solver-workspace) theta-gradient coefficients ───────────
    @testset "C29 compute_theta_gradient_spectral! uses A∓(l±1)" begin
        params = GeoDynamo.SolverParameters(
            geometry = :shell, lmax = 6, mmax = 6, nlat = 16, nlon = 32, nr = 8,
            nr_inner = 4, radial_bandwidth = 3, radius_ratio = 0.35,
            include_magnetic = false, include_composition = false)
        st = GeoDynamo.initialize_solver_state(Float64; params = params)
        field = st.fields.temperature
        ws = st.runtime.gradient_workspace
        cfg = field.config

        L = 3
        m = 0
        spec_real = parent(field.spectral.data_real)
        fill!(spec_real, 0.0)
        fill!(parent(field.spectral.data_imag), 0.0)
        lm_L = GeoDynamo.get_mode_index(cfg, L, m)
        @test lm_L > 0
        slot = GeoDynamo.local_spectral_storage_slot(cfg, lm_L)
        GeoDynamo.set_local_spectral_value!(spec_real, slot, 1, 1.0)

        GeoDynamo.zero_gradient_workspace!(ws)
        GeoDynamo.compute_theta_gradient_spectral!(field, ws)

        out = parent(ws.∇θ_spec.data_real)
        readmode(l) = begin
            idx = GeoDynamo.get_mode_index(cfg, l, m)
            idx <= 0 && return 0.0
            GeoDynamo.local_spectral_value(out, GeoDynamo.local_spectral_storage_slot(cfg, idx), 1)
        end
        @test readmode(L + 1) ≈ _A_plus(L, m)
        @test readmode(L - 1) ≈ _A_minus(L, m)
    end

    # ── C50: the slice buffer must work for a Float32 field ──────────────────
    @testset "C50 extract_physical_slice_phi_local accepts Float32" begin
        cfg = GeoDynamo.create_shtnskit_config(
            lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 4)
        phys32 = zeros(Float32, cfg.nlat, cfg.nlon, 2)
        phys32[2, 3, 1] = 1.5f0
        out = GeoDynamo.extract_physical_slice_phi_local(phys32, 1, cfg)
        @test eltype(out) == Float32
        @test out[2, 3] == 1.5f0

        # Float64 keeps working
        phys64 = zeros(Float64, cfg.nlat, cfg.nlon, 2)
        phys64[4, 5, 1] = -2.25
        out64 = GeoDynamo.extract_physical_slice_phi_local(phys64, 1, cfg)
        @test eltype(out64) == Float64
        @test out64[4, 5] == -2.25
    end

    # ── C34: the generic slice extractor must not be a second copy ───────────
    @testset "C34 generic slice extraction matches the phi-local one" begin
        cfg = GeoDynamo.create_shtnskit_config(
            lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 4)
        phys = reshape(collect(1.0:(cfg.nlat * cfg.nlon * 2)), cfg.nlat, cfg.nlon, 2)
        a = zeros(Float64, cfg.nlat, cfg.nlon)
        b = zeros(Float64, cfg.nlat, cfg.nlon)
        GeoDynamo.extract_physical_slice_phi_local!(a, phys, 2, cfg)
        GeoDynamo.extract_physical_slice_generic!(b, phys, 2, cfg)
        @test a == b
        @test b[3, 4] == phys[3, 4, 2]
    end

    # ── C36: the two W-split packers must agree ──────────────────────────────
    @testset "C36 _build_wsplit_pack delegates to _pack_wsplit" begin
        params = GeoDynamo.SolverParameters(
            geometry = :shell, lmax = 4, mmax = 4, nlat = 12, nlon = 24, nr = 8,
            nr_inner = 4, radial_bandwidth = 3, radius_ratio = 0.35,
            Ek = 1e-3, timestep = 1e-4,
            include_magnetic = false, include_composition = false)
        st = GeoDynamo.initialize_solver_state(Float64; params = params)
        GeoDynamo.initialize_fields!(st)
        nl = params.lmax + 1
        nr = st.runtime.outer_core_domain.N
        bc = GeoDynamo._velocity_bc_code(params.velocity_bcs)
        split = GeoDynamo._get_or_build_poloidal_split!(st, bc)
        bw = split.dpol_op[1].bandwidth

        built = GeoDynamo._build_wsplit_pack(st, nl, nr, bw, Float64)
        packed = GeoDynamo._pack_wsplit(split, nl, nr, bw, Float64)
        @test keys(built) == keys(packed)
        for k in keys(built)
            @test getproperty(built, k) == getproperty(packed, k)
        end
    end

    # ── C40: statistics helpers must be the real ones ────────────────────────
    @testset "C40 bcs statistics match Statistics" begin
        x = [1.0, 2.0, 4.0, 8.0, 16.0]
        @test GeoDynamo.bcs._mean(x) ≈ mean(x)
        @test GeoDynamo.bcs._std(x) ≈ std(x)
        # single element: std is defined as zero here, not NaN
        @test GeoDynamo.bcs._std([3.0]) == 0.0
        # the fake shim module must be gone
        @test !isdefined(GeoDynamo.bcs, :_Statistics)
    end

    # ── C37/C38: the randomize twins ─────────────────────────────────────────
    @testset "C37 randomize_magnetic_field! shares the vector implementation" begin
        params = GeoDynamo.SolverParameters(
            geometry = :shell, lmax = 4, mmax = 4, nlat = 12, nlon = 24, nr = 8,
            nr_inner = 4, radial_bandwidth = 3, radius_ratio = 0.35,
            include_magnetic = true, include_composition = false)
        sa = GeoDynamo.initialize_solver_state(Float64; params = params)
        sb = GeoDynamo.initialize_solver_state(Float64; params = params)
        a = sa.fields.velocity
        b = sb.fields.magnetic

        Random.seed!(1234)
        GeoDynamo.randomize_vector_field!(a; amplitude = 1e-3, lmax = 4)
        Random.seed!(1234)
        GeoDynamo.randomize_magnetic_field!(b; amplitude = 1e-3, lmax = 4)
        @test parent(a.toroidal.data_real) == parent(b.toroidal.data_real)
        @test parent(a.poloidal.data_imag) == parent(b.poloidal.data_imag)

        # C38: the dead `imag3` bindings are gone from the randomize_* functions
        @test !_crc_occ("imag3 = parent(spectral.data_imag)", CRC_IC_SRC)
    end
end
