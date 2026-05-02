# Test parameter file with modified values

architecture = :cpu
geometry = :shell

nr = 128
nr_inner = 16
lmax = 64
mmax = 32
nlat = 128
nlon = 256
radial_bandwidth = 4
radius_ratio = 0.35
r_outer = 1.0

Ek = 5e-5
Ra = 2e6
RaC = 1e6
Pr = 1.0
Pm = 1.0
Sc = 1.0

timestep = 5e-5
start_time = 0.0
end_time = 1.0
max_steps = 10_000
timestepper = CNAB2(theta = 0.5)
timestep_error = 1e-8
courant = 0.5

output_precision = :float64
independent_output_files = true

velocity_bcs = BoundaryConditions(inner = NoSlip(), outer = NoSlip())
temperature_bcs = BoundaryConditions(inner = FixedTemperature(1.0), outer = FixedTemperature(0.0))
composition_bcs = BoundaryConditions(inner = FixedTemperature(0.0), outer = FixedTemperature(0.0))
