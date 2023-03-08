using src
using Documenter

DocMeta.setdocmeta!(src, :DocTestSetup, :(using src); recursive=true)

makedocs(;
    modules=[src],
    authors="FedeClaudi <federicoclaudi@protonmail.com> and contributors",
    repo="https://github.com/FedeClaudi/src.jl/blob/{commit}{path}#{line}",
    sitename="src.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://FedeClaudi.github.io/src.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/FedeClaudi/src.jl",
    devbranch="main",
)
