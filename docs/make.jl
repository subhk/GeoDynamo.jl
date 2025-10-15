#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Documenter
using Geodynamo

DocMeta.setdocmeta!(Geodynamo, :DocTestSetup, :(using Geodynamo); recursive=true)

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
                             canonical = "https://subhk.github.io/Geodynamo.jl/",
                             assets = String[],
                             size_threshold = 500*1024)  # Increase threshold to 500KB

makedocs(
    modules = [Geodynamo, Geodynamo.BoundaryConditions, Geodynamo.InitialConditions, Geodynamo.GeodynamoShell, Geodynamo.GeodynamoBall],
    sitename = "Geodynamo.jl",
    format = format,
    pages = pages,
    checkdocs = :all,  # Include all docstrings, including internal implementation
)

deploydocs(
    repo = "github.com/subhk/Geodynamo.jl.git",
    devbranch = "main",
    push_preview = true,
)
