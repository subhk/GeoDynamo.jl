# Solver include spine.
#
# The order here is intentional: low-level interop and parameter types are
# loaded first, then runtime/state containers, numerical kernels, physics
# updates, timesteppers, and finally the public main loop. Keeping this file as
# the single include spine makes the solver layering visible in one place.
include("solver/interop.jl")
include("solver/parameters.jl")
include("solver/backend.jl")
include("solver/state.jl")
include("solver/numerics.jl")
include("diagnostics/solver.jl")
include("timestep/imex.jl")
include("physics/topography.jl")
include("physics/velocity/solver.jl")
include("physics/scalar_field_solver_common.jl")
include("physics/temperature/solver.jl")
include("physics/composition/solver.jl")
include("physics/magnetic/solver.jl")
include("physics/nonlinear.jl")
include("timestep/erk2.jl")
include("timestep/driver.jl")
include("solver/mainloop.jl")
