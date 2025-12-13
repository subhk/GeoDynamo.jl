#!/usr/bin/env julia
# ================================================================================
# GeoDynamo.jl Documentation Build Script
# ================================================================================
#
# This script builds and deploys documentation using Documenter.jl.
#
# Versioning:
#   - `dev/`    : Built from the main branch (latest development)
#   - `stable/` : Built from the latest tagged release
#   - `vX.Y.Z/` : Built from specific version tags
#
# To build locally:
#   cd docs
#   julia --project=. make.jl
#
# The built documentation will be in docs/build/
#
# ================================================================================

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Ensure the local package is available without registering it
pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter
using GeoDynamo

DocMeta.setdocmeta!(GeoDynamo, :DocTestSetup, :(using GeoDynamo); recursive=true)

pages = [
    "Home" => "index.md",
    "Getting Started" => "getting-started.md",
    "Configuration" => "configuration.md",
    "Time Integration" => "timestepping.md",
    "Spherical Harmonics" => "shtnskit.md",
    "Data Output" => "io.md",
    "Developer Guide" => "developer.md",
    "API Reference" => "api.md",
]

format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://subhk.github.io/GeoDynamo.jl/stable/",
    assets = String[],
    size_threshold = 500 * 1024,
    size_threshold_warn = 200 * 1024,
    # Show version selector with dev and stable
    ansicolor = true,
)

makedocs(
    modules = [GeoDynamo, GeoDynamo.BoundaryConditions, GeoDynamo.InitialConditions, GeoDynamo.GeoDynamoShell, GeoDynamo.GeoDynamoBall],
    sitename = "GeoDynamo.jl",
    format = format,
    pages = pages,
    checkdocs = :all,
    warnonly = true,  # Don't fail on docstring warnings
)

# Deploy documentation with proper versioning
# - main branch → dev/
# - tagged releases → stable/ and vX.Y.Z/
deploydocs(
    repo = "github.com/subhk/GeoDynamo.jl.git",
    devbranch = "main",
    push_preview = true,
    # Explicitly set versions to show in selector
    versions = ["stable" => "v^", "dev" => "dev"],
)
