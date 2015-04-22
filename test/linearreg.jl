# Test linear regression

using Regression
using Base.Test

### auxiliary

function _affmul(A::Matrix, x::VecOrMat, bias::Float64)
	d = size(x,1) - 1
	if ndims(x) == 1
		A * x[1:d] + 2.0 * x[d+1]
	else
		A * x[1:d,:] .+ 2.0 * x[d+1,:]
	end
end


### linear least square

A = randn(8, 3)
y = randn(8)
Y = randn(8, 5)

function verify_llsq(A::Matrix, y::VecOrMat, m::Symbol)
	x = llsq(A, y; method=m)
	u = A * x
	@test_approx_eq_eps 0.0 dot(vec(y - u), vec(u)) 1.0e-10

	At = A'
	x2 = llsq(At, y; method=m, trans=true)
	@test_approx_eq_eps x x2 1.0e-12

	x = llsq(A, y; method=m, bias=2.0)
	u = _affmul(A, x, 2.0)
	@test_approx_eq_eps 0.0 dot(vec(y - u), vec(u)) 1.0e-10

	x2 = llsq(At, y; method=m, bias=2.0, trans=true)
	@test_approx_eq_eps x x2 1.0e-12
end

println("\ton llsq")
for m in [:qrlq, :orth, :svd]
	verify_llsq(A, y, m)
	verify_llsq(A, Y, m)
end


### Ridge regression

function verify_ridgereg(A::Matrix, y::VecOrMat, r::Float64)
	d = size(A, 2)
	At = A'

	x = ridgereg(A, y, r)
	u = A * x
	z = A' * (u - y) + r * x
	@test_approx_eq_eps zeros(size(x)) z 1.0e-10

	x2 = ridgereg(At, y, r; trans=true)
	@test_approx_eq_eps x x2 1.0e-12

	x = ridgereg(A, y, r; bias=2.0)
	u = _affmul(A, x, 2.0)
	_x = ndims(x) == 1 ? x[1:d] : x[1:d,:]
	z = A' * (u - y) + r * _x
	@test_approx_eq_eps zeros(size(z)) z 1.0e-10

	x2 = ridgereg(At, y, r; bias=2.0, trans=true)
	@test_approx_eq_eps x x2 1.0e-10
end

println("\ton ridgereg")
verify_ridgereg(A, y, 0.6)
verify_ridgereg(A, Y, 0.6)
