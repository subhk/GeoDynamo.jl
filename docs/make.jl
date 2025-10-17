#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Add parent directory to load path to find GeoDynamo
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter
using GeoDynamo

DocMeta.setdocmeta!(GeoDynamo, :DocTestSetup, :(using GeoDynamo); recursive=true)

pages = [
    "Home" => "index.md",
    "Getting Started" => "getting-started.md",
    "Configuration" => "configuration.md",
    "Time Integration" => "timestepping.md",
    "Data Output" => "io.md",
    "Developer Guide" => "developer.md",
    "API Reference" => "api.md",
]

format = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true",
                             canonical = "https://subhk.github.io/GeoDynamo.jl/stable/",
                             assets = String[],
                             size_threshold = 500*1024)  # Increase threshold to 500KB

makedocs(
    modules = [GeoDynamo, GeoDynamo.BoundaryConditions, GeoDynamo.InitialConditions, GeoDynamo.GeoDynamoShell, GeoDynamo.GeoDynamoBall],
    sitename = "GeoDynamo.jl",
    format = format,
    pages = pages,
    checkdocs = :all,  # Include all docstrings, including internal implementation
)

deploydocs(
    repo = "github.com/subhk/GeoDynamo.jl.git",
    devbranch = "main",
    push_preview = true,
    versions = ["stable" => "v^", "v#.#", "dev" => "main"],
)
