# Test logistic regression

using Regression
using Calculus
using Base.Test

my_logistic(x) = log(1 + exp(-x))

rf = LogisticRegressFunctor()

n = 200
u = rand(n) * 6. - 3.  # u ~ U[-3, 3]
y = randbool(n) * 2. - 1.

v0 = my_logistic(u .* y)

v = zeros(n)
evaluate_values!(rf, u, y, v)

@test_approx_eq v v0

gp = derivative(u -> my_logistic(u))
gn = derivative(u -> my_logistic(-u))

g0 = zeros(n)
for i in 1 : n
	g0[i] = y[i] > 0 ? gp(u[i]) : gn(u[i])
end

g = zeros(n)
evaluate_derivs!(rf, u, y, g)

@test_approx_eq g g0

v2 = zeros(n)
g2 = zeros(n)

evaluate_values_and_derivs!(rf, u, y, v2, g2)
@test v2 == v
@test g2 == g


# by-columns

d = 5
x = randn(d, n)
y = randbool(n) * 2. - 1.

# without bias

objfun = logisticreg_objfun(x, y, 1.0; by_columns=true)

for t = 1 : 10
	theta = randn(d)
	u = x'theta

	v = zeros(n)
	evaluate_values!(rf, u, y, v)
	objv0 = sum(v) + 0.5 * abs2(norm(theta, 2))

	@test_approx_eq objfun.f(theta) objv0

	g0 = gradient(objfun.f, theta)
	g = zeros(d)
	objfun.g!(theta, g)

	@test_approx_eq_eps g g0 1.0e-4

	g2 = zeros(d)
	objv2 = objfun.fg!(theta, g2)

	@test_approx_eq objv2 objv0
	@test_approx_eq g2 g
end

# with bias

objfun = logisticreg_objfun(x, y, 1.0; by_columns=true, bias=true)

for t = 1 : 10
	theta = randn(d+1)
	u = x'theta[1:d] + theta[d+1]

	v = zeros(n)
	evaluate_values!(rf, u, y, v)
	objv0 = sum(v) + 0.5 * abs2(norm(theta[1:d], 2))

	@test_approx_eq objfun.f(theta) objv0

	g0 = gradient(objfun.f, theta)
	g = zeros(d+1)
	objfun.g!(theta, g)

	@test_approx_eq_eps g g0 1.0e-4

	g2 = zeros(d+1)
	objv2 = objfun.fg!(theta, g2)

	@test_approx_eq objv2 objv0
	@test_approx_eq g2 g
end



# by-rows

x = rand(n, d)
y = randbool(n) * 2. - 1.

# without bias

objfun = logisticreg_objfun(x, y, 1.0; by_columns=false)

for t = 1 : 10
	theta = randn(d)
	u = x * theta

	v = zeros(n)
	evaluate_values!(rf, u, y, v)
	objv0 = sum(v) + 0.5 * abs2(norm(theta, 2))

	@test_approx_eq objfun.f(theta) objv0

	g0 = gradient(objfun.f, theta)
	g = zeros(d)
	objfun.g!(theta, g)

	@test_approx_eq_eps g g0 1.0e-4

	g2 = zeros(d)
	objv2 = objfun.fg!(theta, g2)

	@test_approx_eq objv2 objv0
	@test_approx_eq g2 g
end

# with bias

objfun = logisticreg_objfun(x, y, 1.0; by_columns=false, bias=true)

for t = 1 : 10
	theta = randn(d+1)
	u = x * theta[1:d] + theta[d+1]

	v = zeros(n)
	evaluate_values!(rf, u, y, v)
	objv0 = sum(v) + 0.5 * abs2(norm(theta[1:d], 2))

	@test_approx_eq objfun.f(theta) objv0

	g0 = gradient(objfun.f, theta)
	g = zeros(d+1)
	objfun.g!(theta, g)

	@test_approx_eq_eps g g0 1.0e-4

	g2 = zeros(d+1)
	objv2 = objfun.fg!(theta, g2)

	@test_approx_eq objv2 objv0
	@test_approx_eq g2 g
end

