# Multiclass logistic regression

println("Import packages ...")

using Regression

println("Generate data ...")

function generate_data(r::Float64, n::Int)
	rot = [cos(r) -sin(r); sin(r) cos(r)]
	xp = randn(2, n) .* [0.5, 1.0] .+ [1.0; 0.0]
	xn = randn(2, n) .* [0.5, 1.0] .- [1.0; 0.0]
	xp = rot * xp
	xn = rot * xn

	x = [xp xn]
	y = [fill(1, n); fill(2, n)]
	return x, y
end

rad = rand() * (2pi)

xtrain, ytrain = generate_data(rad, 1000)
xtest, ytest  = generate_data(rad, 500)

println("Training ...")

K = 2  # number of classes
theta0 = zeros(3, K)
theta, objv = multiclass_logisticreg(K, xtrain, ytrain, 1.0e-3, theta0; by_columns=true, bias=true)

println("solution:")
println(theta)
println("objective value = $objv")

println("Testing ...")

u = theta'append_ones(xtest, 1)
nt = size(xtest, 2)
@assert size(u) == (K, nt)

pred = zeros(Int, nt)
for i in 1 : nt
	pred[i] = u[1,i] > u[2,i] ? 1 : 2
end

correct_rate = sum(pred .== ytest) / nt
@printf("correct rate = %.4f\n", correct_rate)
println()

