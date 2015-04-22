examples = [
    "linearreg",
    "logireg",
    "mlogireg"
]

for e in examples
    f = "$e.jl"
    println("Running $f ...")
    include(f)
    println()
end
