examples = [
    "linearreg",
    "logireg",
    "mnlogireg"
]

for e in examples
    f = "$e.jl"
    println("Running $f ...")
    include(f)
    println()
end
