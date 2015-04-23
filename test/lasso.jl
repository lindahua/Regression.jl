using Regression
using Base.Test

import Regression: Options

srand(654321)

## auxiliary functions

_zoverlap(w_e, w) = countnz((w_e .== 0) & (w .== 0)) / max(countnz(w .== 0), countnz(w_e .== 0))

## prepare for simulation

d = 100
num_zeros = 60
w = randn(d)
iz = sort(randperm(d)[1:num_zeros])
w[iz] = 0.0
@assert countnz(w) == d - num_zeros

n = 5000
X = randn(d, n)
y = X'w + 0.1 * randn(n)


## solve w using proximal gradient descent (i.e. ISTA)

println("    with solver ProxGD")

ret = Regression.solve(linearreg(X, y);
            reg=L1Reg(0.01 * n),
            solver=ProxGD(),
            options=Options(armijo=0.2, ftol=1.0e-8 * n))

w_e = ret.sol
relerr = sumabs2(w_e - w) / sumabs2(w)

@test relerr < 1.0e-3
@test _zoverlap(w_e, w) > 0.95
