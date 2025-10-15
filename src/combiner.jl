"""
Combiner APIs for Geodynamo.jl

This file centralizes the distributed-output combiner so it is available
under the `Geodynamo` module. The implementation lives in `extras/combine_file.jl`
and is included here to avoid duplication and drift.
"""

include(joinpath(@__DIR__, "..", "extras", "combine_file.jl"))

