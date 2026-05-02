# ================================================================================
# NetCDF Output Writer - Parallel I/O (MPI-IO) for GeoDynamo Simulation
# All ranks write concurrently to a single shared NetCDF file per timestep.
# Uses NCDatasets.jl with parallel HDF5 backend.
# ================================================================================

using MPI
using NCDatasets
using LinearAlgebra
using Statistics
using Dates
using Printf

include("config.jl")
include("field_info.jl")
include("netcdf.jl")
include("diagnostics.jl")
include("history.jl")
include("restart.jl")
include("utilities.jl")
