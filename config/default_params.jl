# GeoDynamo.jl default solver parameters

architecture = :cpu
geometry = :shell

nr = 64
nr_inner = 16
lmax = 32
mmax = 32
nlat = 64
nlon = 128
radial_bandwidth = 4
radius_ratio = 0.35
r_outer = 1.0

Ek = 1e-4
Ra = 1e6
RaC = 1e6
Pr = 1.0
Pm = 1.0
Sc = 1.0

timestep = 5e-5
start_time = 0.0
end_time = 1.0
stop_iteration = 10_000
timestepper = CNAB2(theta = 0.5)
timestep_error = 1e-8
courant = 0.5

output_precision = :float64
independent_output_files = true
output_interval = 1.0
restart_interval = 0.0

include_magnetic = false
include_composition = true
impose_magnetic_field = false

velocity_bcs = BoundaryConditions(inner = NoSlip(), outer = NoSlip())
temperature_bcs = BoundaryConditions(inner = FixedTemperature(1.0), outer = FixedTemperature(0.0))
composition_bcs = BoundaryConditions(inner = FixedTemperature(0.0), outer = FixedTemperature(0.0))
poloidal_stress_iterations = 2
temperature_bc_file = ""
composition_bc_file = ""
temperature_bc_format = :physical
composition_bc_format = :physical

topography_enabled = false
topography_epsilon = 0.01
topography_degree = -1
include_topography_velocity = true
include_topography_magnetic = true
include_topography_thermal = true
include_topography_slope_terms = true
include_topography_shift_terms = true

stefan_enabled = false
stefan_number = 1.0
inner_core_conductivity_ratio = 1.0
latent_heat = 1.0

icb_topography_file = ""
ocb_topography_file = ""

restart_file = ""
restart_dir = ""
restart_time = 0.0
