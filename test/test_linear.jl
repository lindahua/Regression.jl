# Test linear regression

using Regression
using Base.Test


k = 3
d = 5
n = 20
x = randn(d, n)

# ordinary least square

a = randn(d)
y = vec(a'x)

ra = ordinary_least_squares(x, y)
@test size(ra) == (d,)
@test_approx_eq ra a

a = randn(d, k)
y = a'x

ra = ordinary_least_squares(x, y)
@test size(ra) == (d, k)
@test_approx_eq ra a

