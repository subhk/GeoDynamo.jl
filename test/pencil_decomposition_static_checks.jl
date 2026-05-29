# Whitespace-insensitive source matching: SciML auto-formatting (spacing,
# line wraps) must not break these static source-contract checks.
_sc_wsn(s) = replace(s, r"\s+" => "")
_sc_occ(pat, src) = occursin(_sc_wsn(pat), _sc_wsn(src))

using Test

const ROOT = normpath(joinpath(@__DIR__, ".."))
const PARALLEL_PENCILS_SRC = read(joinpath(ROOT, "src", "parallel", "pencils.jl"), String)
const SPECTRAL_SRC = read(joinpath(ROOT, "src", "transforms", "spectral.jl"), String)
const FIELD_TRANSFORMS_SRC = read(joinpath(ROOT, "src", "fields", "transforms.jl"), String)
const FIELD_INFO_SRC = read(joinpath(ROOT, "src", "io", "field_info.jl"), String)
const NETCDF_SRC = read(joinpath(ROOT, "src", "io", "netcdf.jl"), String)
const RESTART_SRC = read(joinpath(ROOT, "src", "io", "restart.jl"), String)

@testset "2D pencil decomposition static contract" begin
    @test _sc_occ("spectral_mode_grid_dims", PARALLEL_PENCILS_SRC)
    @test _sc_occ("spectral_mode_grid_dims", SPECTRAL_SRC)

    @test !_sc_occ("shared_spectral_compatible_proc_dims(optimize_process_topology", PARALLEL_PENCILS_SRC)
    @test !_sc_occ(
        "shared_spectral_compatible_proc_dims(\n            optimize_process_topology_shtnskit",
        SPECTRAL_SRC)

    @test _sc_occ("mode_lookup", FIELD_TRANSFORMS_SRC)
    @test _sc_occ("global_l = global_l_slot - 1", FIELD_TRANSFORMS_SRC)
    @test _sc_occ("global_m = global_m_slot - 1", FIELD_TRANSFORMS_SRC)
    @test !_sc_occ("only supports the shared-radial layout", FIELD_TRANSFORMS_SRC)

    @test _sc_occ("local_spectral_mode_indices(config)", NETCDF_SRC)
    @test _sc_occ("local_spectral_mode_indices(shtns_config)", RESTART_SRC)

    @test _sc_occ("spectral_dims", PARALLEL_PENCILS_SRC)
    @test _sc_occ("spectral_dims", SPECTRAL_SRC)
    @test _sc_occ("spectral_dims[1] ÷ p1 < 1", PARALLEL_PENCILS_SRC)
    @test _sc_occ("spectral_dims[2] ÷ p2 < 1", PARALLEL_PENCILS_SRC)
    @test !_sc_occ("optimize_process_topology_shtnskit(nprocs, nlat, nlon)", SPECTRAL_SRC)
    @test !_sc_occ("mixed_dims = (nlm, nlat, nr)", SPECTRAL_SRC)
    @test !_sc_occ("mixed_dims = (config.nlm, config.nlat, nr)", PARALLEL_PENCILS_SRC)
    @test !_sc_occ("mixed_dims = (sht_config.nlm, nlat, nr)", SPECTRAL_SRC)
    @test _sc_occ("mixed = pencil_spec", PARALLEL_PENCILS_SRC)
    @test _sc_occ("mixed=pencil_spec", SPECTRAL_SRC)

    @test !_sc_occ("local_ranges[:spec] = range_local(pencils.spec, 1)", FIELD_INFO_SRC)
    @test _sc_occ("local_spectral_mode_indices(config)", FIELD_INFO_SRC)
end
