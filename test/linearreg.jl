# Test linear regression

using Regression
using Base.Test

### linear least square

A = randn(8, 3)
y = randn(8)
Y = randn(8, 5)

for m in [:qrlq, :orth, :svd]
	x = llsq(A, y; method=m)
	@test size(x) == (3,)
	u = A * x
	@test_approx_eq_eps 0.0 dot(y - u, u) 1.0e-10

	At = A'
	x2 = llsq(At, y; method=m, trans=true)
	@test size(x2) == (3,)
	@test_approx_eq_eps x x2 1.0e-12

	x = llsq(A, y; method=m, bias=2.0)
	@test size(x) == (4,)
	u = A * x[1:3] + 2.0 * x[4]
	@test_approx_eq_eps 0.0 dot(y - u, u) 1.0e-10

	x2 = llsq(At, y; method=m, bias=2.0, trans=true)
	@test size(x2) == (4,)
	@test_approx_eq_eps x x2 1.0e-12
end
