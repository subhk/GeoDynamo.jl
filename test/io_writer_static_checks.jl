# Whitespace-insensitive source matching: SciML auto-formatting (spacing,
# line wraps) must not break these static source-contract checks.
_sc_wsn(s) = replace(s, r"\s+" => "")
_sc_occ(pat, src) = occursin(_sc_wsn(pat), _sc_wsn(src))

using Test

const IO_STATIC_ROOT = normpath(joinpath(@__DIR__, ".."))
const IO_STATIC_NETCDF_SRC = read(joinpath(IO_STATIC_ROOT, "src", "io", "netcdf.jl"), String)
const IO_STATIC_RESTART_SRC = read(joinpath(IO_STATIC_ROOT, "src", "io", "restart.jl"), String)
const IO_STATIC_STATE_SRC = read(joinpath(IO_STATIC_ROOT, "src", "solver", "state.jl"), String)
const IO_STATIC_SIMULATION_SRC = read(joinpath(IO_STATIC_ROOT, "src", "api", "simulation.jl"), String)

@testset "I/O writer static contract" begin
    @test _sc_occ("should_define_physical_field_variables(config)", IO_STATIC_NETCDF_SRC)
    @test _sc_occ("should_define_spectral_field_variables(config)", IO_STATIC_NETCDF_SRC)
    @test _sc_occ(
        r"(?s)if should_define_physical_field_variables\(config\).*?defVar\(ds, \"temperature\"",
        IO_STATIC_NETCDF_SRC)
    @test _sc_occ(
        r"(?s)if should_define_physical_field_variables\(config\).*?defVar\(ds, \"composition\"",
        IO_STATIC_NETCDF_SRC)
    @test _sc_occ(
        r"(?s)if should_define_spectral_field_variables\(config\).*?velocity_toroidal",
        IO_STATIC_NETCDF_SRC)

    @test _sc_occ("sync_output_physical_scalars!(state)", IO_STATIC_STATE_SRC)
    @test _sc_occ(
        "fields[\"temperature\"] = copy(parent(state.fields.temperature.temperature.data))",
        IO_STATIC_STATE_SRC)
    @test _sc_occ("\"temperature_spectral\"", IO_STATIC_STATE_SRC)
    @test _sc_occ(
        "fields[\"composition\"] = copy(parent(state.fields.composition.composition.data))",
        IO_STATIC_STATE_SRC)
    @test _sc_occ("\"composition_spectral\"", IO_STATIC_STATE_SRC)

    @test _sc_occ("_legacy_linear_spectral_io_ranges", IO_STATIC_NETCDF_SRC)
    @test _sc_occ("requires shtns_config", IO_STATIC_NETCDF_SRC)
    @test !_sc_occ("lm_range = range_local(pencils.spec, 1)", IO_STATIC_RESTART_SRC)

    @test _sc_occ("read_restart!(tracker, restart_from,", IO_STATIC_SIMULATION_SRC)
    @test _sc_occ("model.state.runtime.shtns_config.pencils", IO_STATIC_SIMULATION_SRC)
    @test _sc_occ("shtns_config=model.state.runtime.shtns_config", IO_STATIC_SIMULATION_SRC)
    @test _sc_occ("restore_fields_from_restart!(model.state, restart_data)", IO_STATIC_SIMULATION_SRC)
    @test _sc_occ("reset_solver_clock!(model.state;", IO_STATIC_SIMULATION_SRC)
    @test !_sc_occ("model.state.time = Float64(metadata[\"current_time\"])", IO_STATIC_SIMULATION_SRC)
end
