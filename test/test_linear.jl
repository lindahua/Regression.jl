# Test linear regression

using Regression
using Base.Test

# ordinary least square

a = randn(5, 3)
b = randn(3)
x = a * b

r = llsq(a, x)
@test size(r) == size(b)
@test_approx_eq r b

r = llsq(a, x; method=:qrlq)
@test size(r) == size(b)
@test_approx_eq r b

r = llsq(a, x; method=:orth)
@test size(r) == size(b)
@test_approx_eq r b

r = llsq(a, x; method=:svd)
@test size(r) == size(b)
@test_approx_eq r b


a = randn(3, 5)
b = randn(3)
x = a'b

r = llsq(a, x; by_columns=true, method=:qrlq)
@test size(r) == size(b)
@test_approx_eq r b

r = llsq(a, x; by_columns=true, method=:orth)
@test size(r) == size(b)
@test_approx_eq r b

r = llsq(a, x; by_columns=true, method=:svd)
@test size(r) == size(b)
@test_approx_eq r b


a = randn(5, 3)
b = randn(3, 4)
x = a * b

r = llsq(a, x)
@test size(r) == size(b)
@test_approx_eq r b

r = llsq(a, x; method=:qrlq)
@test size(r) == size(b)
@test_approx_eq r b

r = llsq(a, x; method=:orth)
@test size(r) == size(b)
@test_approx_eq r b

r = llsq(a, x; method=:svd)
@test size(r) == size(b)
@test_approx_eq r b


a = randn(3, 5)
b = randn(3, 2)
x = b'a

r = llsq(a, x; by_columns=true, method=:qrlq)
@test size(r) == size(b)
@test_approx_eq r b

r = llsq(a, x; by_columns=true, method=:orth)
@test size(r) == size(b)
@test_approx_eq r b

r = llsq(a, x; by_columns=true, method=:svd)
@test size(r) == size(b)
@test_approx_eq r b

