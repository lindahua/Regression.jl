tests = ["common", "linearreg", "logisticreg", "mclogisticreg"]

for t in tests
	f = joinpath("test", "test_$t.jl")
	println("$f ...")
	include(f)
end
