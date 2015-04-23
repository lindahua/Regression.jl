# Proximal methods for LASSO

using Regression
import Regression: Options, solve

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

ret = solve(linearreg(X, y);
            reg=L1Reg(0.01 * n),
            solver=ProxGD(),
            options=Options(verbosity=:iter, armijo=0.2, ftol=1.0e-8 * n))

println()


## examine solution

w_e = ret.sol
@printf("relative error   = %.4e\n", sumabs2(w_e - w) / sumabs2(w))

numz_w = d - countnz(w)
numz_we = d - countnz(w_e)
@printf("w:   num_zeros = %d\n", numz_w)
@printf("w_e: num_zeros = %d\n", numz_we)
@printf("overlap of zeros = %.1f%%\n",
    100.0 * countnz((w .== 0) & (w_e .== 0)) / max(numz_w, numz_we))
