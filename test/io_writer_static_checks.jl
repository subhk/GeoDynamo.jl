using Test

const IO_STATIC_ROOT = normpath(joinpath(@__DIR__, ".."))
const IO_STATIC_NETCDF_SRC = read(joinpath(IO_STATIC_ROOT, "src", "io", "netcdf.jl"), String)
const IO_STATIC_RESTART_SRC = read(joinpath(IO_STATIC_ROOT, "src", "io", "restart.jl"), String)
const IO_STATIC_STATE_SRC = read(joinpath(IO_STATIC_ROOT, "src", "solver", "state.jl"), String)
const IO_STATIC_SIMULATION_SRC = read(joinpath(IO_STATIC_ROOT, "src", "api", "simulation.jl"), String)

@testset "I/O writer static contract" begin
    @test occursin("should_define_physical_field_variables(config)", IO_STATIC_NETCDF_SRC)
    @test occursin("should_define_spectral_field_variables(config)", IO_STATIC_NETCDF_SRC)
    @test occursin(r"(?s)if should_define_physical_field_variables\(config\).*?defVar\(ds, \"temperature\"", IO_STATIC_NETCDF_SRC)
    @test occursin(r"(?s)if should_define_physical_field_variables\(config\).*?defVar\(ds, \"composition\"", IO_STATIC_NETCDF_SRC)
    @test occursin(r"(?s)if should_define_spectral_field_variables\(config\).*?velocity_toroidal", IO_STATIC_NETCDF_SRC)

    @test occursin("sync_output_physical_scalars!(state)", IO_STATIC_STATE_SRC)
    @test occursin("fields[\"temperature\"] = copy(parent(state.fields.temperature.temperature.data))", IO_STATIC_STATE_SRC)
    @test occursin("\"temperature_spectral\"", IO_STATIC_STATE_SRC)
    @test occursin("fields[\"composition\"] = copy(parent(state.fields.composition.composition.data))", IO_STATIC_STATE_SRC)
    @test occursin("\"composition_spectral\"", IO_STATIC_STATE_SRC)

    @test occursin("__legacy_linear_spectral_io_ranges", IO_STATIC_NETCDF_SRC)
    @test occursin("requires shtns_config", IO_STATIC_NETCDF_SRC)
    @test !occursin("lm_range = range_local(pencils.spec, 1)", IO_STATIC_RESTART_SRC)

    @test occursin("read_restart!(tracker, restart_from,", IO_STATIC_SIMULATION_SRC)
    @test occursin("model.state.runtime.shtns_config.pencils", IO_STATIC_SIMULATION_SRC)
    @test occursin("shtns_config=model.state.runtime.shtns_config", IO_STATIC_SIMULATION_SRC)
    @test occursin("restore_fields_from_restart!(model.state, restart_data)", IO_STATIC_SIMULATION_SRC)
    @test occursin("reset_solver_clock!(model.state;", IO_STATIC_SIMULATION_SRC)
    @test !occursin("model.state.time = Float64(metadata[\"current_time\"])", IO_STATIC_SIMULATION_SRC)
end
