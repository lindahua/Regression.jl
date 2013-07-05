# Benchmark different algorithms for logistic regression

using Regression

# generate data

function generate_data(n::Int)
	r = rand() * (2pi)
	rot = [cos(r) -sin(r); sin(r) cos(r)]
	xp = randn(2, n) .* [0.5, 1.0] .+ [1.0; 0.0]
	xn = randn(2, n) .* [0.5, 1.0] .- [1.0; 0.0]
	xp = rot * xp
	xn = rot * xn

	x = [xp xn]
	y = [ones(n); -ones(n)]
	return x, y
end

x0, y0 = generate_data(100)
x, y = generate_data(5000)

# warm up

println("Warming up ...")

logisticreg(x, y, 0., zeros(2); by_columns=true, method=:gradient_descent, iterations=20)
logisticreg(x, y, 0., zeros(2); by_columns=true, method=:cg,               iterations=20)
logisticreg(x, y, 0., zeros(2); by_columns=true, method=:bfgs,             iterations=20)
logisticreg(x, y, 0., zeros(2); by_columns=true, method=:l_bfgs,           iterations=20)

# benchmark

println("Benchmarking ...")
const rp = 10

t = @elapsed for i = 1 : rp
	logisticreg(x, y, 0., zeros(2); by_columns=true, method=:gradient_descent, iterations=1000)
end
@printf("%-18s:  %12.6fsec\n", "gradient_descent", t)

t = @elapsed for i = 1 : rp
	logisticreg(x, y, 0., zeros(2); by_columns=true, method=:cg,               iterations=1000)
end
@printf("%-18s:  %12.6fsec\n", "cg", t)

t = @elapsed for i = 1 : rp
	logisticreg(x, y, 0., zeros(2); by_columns=true, method=:bfgs,             iterations=1000)
end
@printf("%-18s:  %12.6fsec\n", "bfgs", t)

t = @elapsed for i = 1 : rp
	logisticreg(x, y, 0., zeros(2); by_columns=true, method=:l_bfgs,           iterations=1000)
end
@printf("%-18s:  %12.6fsec\n", "l_bfgs", t)

