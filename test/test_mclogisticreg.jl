# Test of multi-class logistic regression

using NumericExtensions
using Regression
using Calculus
using Base.Test

# test the functor

mclogistic_term(u, y) = logsumexp(u) - u[y]

rf = MultiClassLogisticRegressFunctor()
K = 3
d = 4
n = 20

u = randn(K, n)
y = rand(1:K, n)

v0 = zeros(n)
for i in 1 : n
	v0[i] = mclogistic_term(u[:,i], y[i])
end

v = zeros(n)
evaluate_values!(rf, u, y, v)
@test_approx_eq v v0

g0 = zeros(K, n)
for i in 1 : n
	g0[:,i] = gradient(x -> mclogistic_term(x, y[i]), u[:,i])
end

g = zeros(K, n)
evaluate_derivs!(rf, u, y, g)
@test_approx_eq_eps g g0 1.0e-4

v2 = zeros(n)
g2 = zeros(K, n)
evaluate_values_and_derivs!(rf, u, y, v2, g2)
@test_approx_eq v2 v
@test_approx_eq g2 g


# by-columns

x = randn(d, n)
y = rand(1:K, n)

# without bias

objfun = multiclass_logisticreg_objfun(K, x, y, 1.0; by_columns=true)

for t = 1 : 10
	theta = randn(d, K)
	vtheta = vec(theta)
	u = theta'x

	v = zeros(n)
	evaluate_values!(rf, u, y, v)
	objv0 = sum(v) + 0.5 * sumsq(theta)

	@test_approx_eq objfun.f(vtheta) objv0

	g0 = gradient(objfun.f, vtheta)
	g = zeros(d * K)
	objfun.g!(vtheta, g)

	@test_approx_eq_eps g g0 1.0e-4

	g2 = zeros(d * K)
	objv2 = objfun.fg!(vtheta, g2)

	@test_approx_eq objv2 objv0
	@test_approx_eq g2 g
end

# with bias

objfun = multiclass_logisticreg_objfun(K, x, y, 1.0; by_columns=true, bias=true)

for t = 1 : 10
	theta = randn(d+1, K)
	vtheta = vec(theta)
	u = theta' * append_ones(x, 1)

	v = zeros(n)
	evaluate_values!(rf, u, y, v)
	objv0 = sum(v) + 0.5 * sumsq(theta[1:d,:])

	@test_approx_eq objfun.f(vtheta) objv0

	g0 = gradient(objfun.f, vtheta)
	g = zeros((d+1) * K)
	objfun.g!(vtheta, g)

	@test_approx_eq_eps g g0 1.0e-4

	g2 = zeros((d+1) * K)
	objv2 = objfun.fg!(vtheta, g2)

	@test_approx_eq objv2 objv0
	@test_approx_eq g2 g
end


# by-rows

x = randn(n, d)
y = rand(1:K, n)

# without bias

objfun = multiclass_logisticreg_objfun(K, x, y, 1.0; by_columns=false)

for t = 1 : 10
	theta = randn(d, K)
	vtheta = vec(theta)
	u = theta' * x'

	v = zeros(n)
	evaluate_values!(rf, u, y, v)
	objv0 = sum(v) + 0.5 * sumsq(theta)

	@test_approx_eq objfun.f(vtheta) objv0

	g0 = gradient(objfun.f, vtheta)
	g = zeros(d * K)
	objfun.g!(vtheta, g)

	@test_approx_eq_eps g g0 1.0e-4

	g2 = zeros(d * K)
	objv2 = objfun.fg!(vtheta, g2)

	@test_approx_eq objv2 objv0
	@test_approx_eq g2 g
end

# with bias

objfun = multiclass_logisticreg_objfun(K, x, y, 1.0; by_columns=false, bias=true)

for t = 1 : 10
	theta = randn(d+1, K)
	vtheta = vec(theta)
	u = theta' * append_ones(x', 1)

	v = zeros(n)
	evaluate_values!(rf, u, y, v)
	objv0 = sum(v) + 0.5 * sumsq(theta[1:d,:])

	@test_approx_eq objfun.f(vtheta) objv0

	g0 = gradient(objfun.f, vtheta)
	g = zeros((d+1) * K)
	objfun.g!(vtheta, g)

	@test_approx_eq_eps g g0 1.0e-4

	g2 = zeros((d+1) * K)
	objv2 = objfun.fg!(vtheta, g2)

	@test_approx_eq objv2 objv0
	@test_approx_eq g2 g
end


