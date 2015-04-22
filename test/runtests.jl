tests = [
	"common",
	"linearreg",
	"solve"
]

for t in tests
	f = "$t.jl"
	println("* running $f ...")
	include(f)
end
