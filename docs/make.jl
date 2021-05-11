push!(LOAD_PATH, "../src/")

using RPCircuits
using Documenter

makedocs(;
    modules=[RPCircuits],
    authors="Renato Lui Geh <renatogeh@gmail.com>",
    repo="https://github.com/RenatoGeh/RPCircuits.jl/blob/{commit}{path}#L{line}",
    sitename="RPCircuits.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://renatogeh.github.io/RPCircuits.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/RenatoGeh/RPCircuits.jl.git",
)
