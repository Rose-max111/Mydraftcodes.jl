using Mydraftcodes
using Documenter

DocMeta.setdocmeta!(Mydraftcodes, :DocTestSetup, :(using Mydraftcodes); recursive=true)

makedocs(;
    modules=[Mydraftcodes],
    authors="Yiming Lu <luyimingboy@163.com> and contributors",
    sitename="Mydraftcodes.jl",
    format=Documenter.HTML(;
        canonical="https://Rose_max111.github.io/Mydraftcodes.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Rose_max111/Mydraftcodes.jl",
    devbranch="main",
)
