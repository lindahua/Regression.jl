examples = [
    "linearreg",
    "logireg",
    "mlogireg",
    "lasso"
]

for e in examples
    f = "$e.jl"
    println("Running $f ...")
    include(f)
    println()
end
