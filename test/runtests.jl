tests = [
	"common",
	"linearreg",
	"regproblems",
	"solve",
	"lasso"
]

for t in tests
	f = "$t.jl"
	println("* running $f ...")
	include(f)
end
