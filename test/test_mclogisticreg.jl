# Test of multi-class logistic regression

using NumericExtensions
using Regression
using Calculus
using Base.Test

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


