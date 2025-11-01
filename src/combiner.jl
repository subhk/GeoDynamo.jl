"""
Combiner APIs for GeoDynamo.jl

This file centralizes the distributed-output combiner so it is available
under the `GeoDynamo` module. The implementation lives in `extras/combine_file.jl`
and is included here to avoid duplication and drift.
"""

include(joinpath(@__DIR__, "..", "extras", "combine_file.jl"))

