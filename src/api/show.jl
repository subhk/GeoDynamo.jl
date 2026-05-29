# Oceananigans-style summaries and tree-style show methods.

__arch_name(arch) = arch isa CPU ? "CPU" : "GPU"

Base.summary(g::SphericalShellGrid) =
    "SphericalShellGrid($(__arch_name(g.arch)), lmax=$(g.lmax), mmax=$(g.mmax), nr=$(g.nr))"

Base.summary(g::SphericalBallGrid) =
    "SphericalBallGrid($(__arch_name(g.arch)), lmax=$(g.lmax), mmax=$(g.mmax), nr=$(g.nr))"

Base.summary(c::Clock) = "Clock(time=$(c.time), iteration=$(c.iteration))"

Base.summary(m::GeodynamoModel{T}) where {T} =
    "GeodynamoModel{$T}(time=$(m.clock.time), iteration=$(m.clock.iteration))"

Base.summary(s::Simulation) =
    "Simulation(dt=$(s.dt), stop_time=$(s.stop_time), stop_iteration=$(s.stop_iteration))"

function Base.show(io::IO, ::MIME"text/plain", m::GeodynamoModel{T}) where {T}
    p = m.state.parameters
    println(io, "GeodynamoModel{$T}")
    println(io, "├── grid: ", summary(m.grid))
    println(io, "├── clock: time=", m.clock.time, ", iteration=", m.clock.iteration)
    println(io, "├── physics: Ek=$(p.Ek), Pr=$(p.Pr), Pm=$(p.Pm), Sc=$(p.Sc), Ra=$(p.Ra)")
    print(io,   "└── active: magnetic=$(p.include_magnetic_field), composition=$(p.include_composition)")
end

function Base.show(io::IO, ::MIME"text/plain", sim::Simulation)
    c = sim.model.clock
    println(io, "Simulation of ", summary(sim.model))
    println(io, "├── dt: ", sim.dt)
    println(io, "├── stop_time: ", sim.stop_time)
    println(io, "├── stop_iteration: ", sim.stop_iteration)
    println(io, "├── wall_time_limit: ", sim.wall_time_limit)
    println(io, "├── clock: time=", c.time, ", iteration=", c.iteration)
    println(io, "├── callbacks: ", isempty(sim.callbacks) ? "(none)" : join(keys(sim.callbacks), ", "))
    print(io,   "└── output_writers: ", isempty(sim.output_writers) ? "(none)" : join(keys(sim.output_writers), ", "))
end
