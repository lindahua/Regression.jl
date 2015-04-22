using Regression
import Regression: solve, Options

# Data

k = 3
d = 5
n = 1000

W = randn(k, d+1)
X = randn(d, n)
U = (W[:,1:d] * X .+ W[:,d+1]) + 0.2 * randn(k, n)
y = vec(mapslices(indmax, U, 1))

# Solve

ret = solve(mlogisticreg(X, y, k; bias=1.0);
            reg=SqrL2Reg(1.0e-3),
            options=Options(verbosity=:iter, ftol=1.0e-5 * n))

println()

# Print results

W_est = ret.sol

@show size(W_est)
U_r = predict(MvAffinePred(d, k), W_est, X)
y_r = vec(mapslices(indmax, U_r, 1))

@printf("correct rate = %.4f\n", countnz(y .== y_r) / n)
