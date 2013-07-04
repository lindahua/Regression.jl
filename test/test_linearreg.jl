# Test linear regression

using Regression
using Base.Test

### ordinary least square

for m in [:qrlq, :orth, :svd]

	println("testing $m ...")

	# by rows

	a = randn(5, 3)
	b = randn(3)
	x = a * b

	r = linearreg_lsq(a, x; method=m, by_columns=false)
	@test size(r) == size(b)
	@test_approx_eq r b

	b = randn(4)
	x = a * b[1:3] + b[4]

	r = linearreg_lsq(a, x; method=m, by_columns=false, bias=true)
	@test size(r) == size(b)
	@test_approx_eq r b

	b = randn(3, 4)
	x = a * b

	r = linearreg_lsq(a, x; method=m, by_columns=false)
	@test size(r) == size(b)
	@test_approx_eq r b

	b = randn(4, 4)
	x = a * b[1:3, :] .+ b[4, :]

	r = linearreg_lsq(a, x; method=m, by_columns=false, bias=true)
	@test size(r) == size(b)
	@test_approx_eq r b

	# by columns

	a = randn(3, 5)
	b = randn(3)
	x = vec(b'a)

	r = linearreg_lsq(a, x; method=m, by_columns=true)
	@test size(r) == size(b)
	@test_approx_eq r b

	b = randn(4)
	x = a'b[1:3] + b[4]

	r = linearreg_lsq(a, x; method=m, by_columns=true, bias=true)
	@test size(r) == size(b)
	@test_approx_eq r b

	b = randn(3, 2)
	x = b'a

	r = linearreg_lsq(a, x; method=m, by_columns=true)
	@test size(r) == size(b)
	@test_approx_eq r b

	b = randn(4, 2)
	x = b[1:3,:]'a .+ b[4,:]'

	r = linearreg_lsq(a, x; method=m, by_columns=true, bias=true)
	@test size(r) == size(b)
	@test_approx_eq r b

end

