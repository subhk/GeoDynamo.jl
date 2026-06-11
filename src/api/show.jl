# Oceananigans-style summaries and tree-style show methods.
# Model time is nondimensional → prettysummary; wall-clock → prettytime.

_arch_name(arch) = arch isa CPU ? "CPU" : "GPU"

# ── Grids ─────────────────────────────────────────────────────────────────────
function Base.summary(g::SphericalShellGrid)
    "SphericalShellGrid($(_arch_name(g.arch)), lmax=$(g.lmax), mmax=$(g.mmax), nr=$(g.nr))"
end
function Base.summary(g::SphericalBallGrid)
    "SphericalBallGrid($(_arch_name(g.arch)), lmax=$(g.lmax), mmax=$(g.mmax), nr=$(g.nr))"
end

# ── Schedules ────────────────────────────────────────────────────────────────
Base.summary(s::IterationInterval) = "IterationInterval($(s.interval))"
Base.summary(s::TimeInterval) = "TimeInterval($(prettysummary(s.interval)))"
Base.summary(s::WallTimeInterval) = "WallTimeInterval($(prettysummary(s.interval)))"
Base.summary(s::SpecifiedTimes) =
    "SpecifiedTimes($(join(map(prettysummary, s.times), ", ")))"

# ── Callbacks / writers ──────────────────────────────────────────────────────
_callable_name(f::Function) = string(nameof(f))
_callable_name(f) = string(nameof(typeof(f)))
Base.summary(cb::Callback) =
    "Callback of $(_callable_name(cb.func)) on $(summary(cb.schedule))"
Base.summary(cb::EnergyDiagnostics) = "EnergyDiagnostics on $(summary(cb.schedule))"
Base.summary(cb::SolenoidalMonitor) =
    "SolenoidalMonitor (threshold=$(prettysummary(cb.threshold))) on $(summary(cb.schedule))"
Base.summary(cb::SimulationProgress) = "SimulationProgress on $(summary(cb.schedule))"
Base.summary(cb::HealthCheck) = "HealthCheck on $(summary(cb.schedule))"
Base.summary(ow::FieldWriter) =
    "FieldWriter writing ($(join(ow.fields, ", "))) to $(ow.path) on $(summary(ow.schedule))"
Base.summary(ow::CheckpointWriter) =
    "CheckpointWriter writing to $(ow.path) on $(summary(ow.schedule))"

# ── Timesteppers ─────────────────────────────────────────────────────────────
Base.summary(ts::CNAB2) = "CNAB2(theta=$(prettysummary(ts.implicit_theta)))"
Base.summary(ts::ERK2) = "ERK2()"
Base.summary(ts::EAB2) =
    "EAB2(krylov_dimension=$(ts.krylov_dimension), tolerance=$(prettysummary(ts.tolerance)))"
Base.summary(ts::ETD) =
    "ETD(krylov_dimension=$(ts.krylov_dimension), tolerance=$(prettysummary(ts.tolerance)))"
Base.summary(ts::ThetaMethod) = "ThetaMethod(theta=$(prettysummary(ts.theta)))"

# ── Clock ────────────────────────────────────────────────────────────────────
Base.summary(c::Clock) =
    "Clock(time = $(prettysummary(c.time)), iteration = $(c.iteration), " *
    "last_Δt = $(prettysummary(c.last_dt)))"
Base.show(io::IO, ::MIME"text/plain", c::Clock) = print(io, summary(c))

# ── Model ────────────────────────────────────────────────────────────────────
function Base.summary(m::GeodynamoModel{T, A}) where {T, A}
    arch = _arch_name(m.grid.arch)
    "GeodynamoModel{$A, $T}(time = $(prettysummary(m.clock.time)), " *
    "iteration = $(m.clock.iteration))"
end

function Base.show(io::IO, ::MIME"text/plain", m::GeodynamoModel)
    p = m.state.parameters
    println(io, summary(m))
    println(io, "├── grid: ", summary(m.grid))
    println(io, "├── timestepper: ", summary(p.timestepper))
    println(io, "├── physics: Ek=", prettysummary(p.Ek), ", Pr=", prettysummary(p.Pr),
        ", Pm=", prettysummary(p.Pm), ", Sc=", prettysummary(p.Sc),
        ", Ra=", prettysummary(p.Ra))
    print(io, "└── active: magnetic=", p.include_magnetic,
        ", composition=", p.include_composition)
end

# ── Simulation ───────────────────────────────────────────────────────────────
function Base.summary(s::Simulation)
    "Simulation(Δt=$(prettysummary(s.dt)), stop_time=$(prettysummary(s.stop_time)), " *
    "stop_iteration=$(prettysummary(s.stop_iteration)))"
end

function _show_ordered_tree(io, label, dict; connector = "├──")
    if isempty(dict)
        println(io, connector, " ", label, ": OrderedDict with no entries")
    else
        n = length(dict)
        println(io, connector, " ", label, ": OrderedDict with ", n,
            n == 1 ? " entry:" : " entries:")
        for (i, (k, v)) in enumerate(dict)
            conn = i == n ? "└──" : "├──"
            println(io, "│   ", conn, " ", k, " => ", summary(v))
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", sim::Simulation)
    wall = sim._wall_start > 0 ? time() - sim._wall_start : 0.0
    println(io, "Simulation of ", summary(sim.model))
    println(io, "├── Next time step: ", prettysummary(sim.dt))
    println(io, "├── Elapsed wall time: ", prettytime(wall))
    println(io, "├── Stop time: ", prettysummary(sim.stop_time))
    println(io, "├── Stop iteration: ", prettysummary(sim.stop_iteration))
    println(io, "├── Wall time limit: ",
        isfinite(sim.wall_time_limit) ? prettytime(sim.wall_time_limit) : "Inf")
    _show_ordered_tree(io, "Callbacks", sim.callbacks)
    if isempty(sim.output_writers)
        print(io, "└── Output writers: OrderedDict with no entries")
    else
        n = length(sim.output_writers)
        println(io, "└── Output writers: OrderedDict with ", n,
            n == 1 ? " entry:" : " entries:")
        for (i, (k, v)) in enumerate(sim.output_writers)
            conn = i == n ? "└──" : "├──"
            if i == n
                print(io, "    ", conn, " ", k, " => ", summary(v))
            else
                println(io, "    ", conn, " ", k, " => ", summary(v))
            end
        end
    end
end
