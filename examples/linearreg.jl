# Linear regression (using iterative update)

using Regression
import Regression: solve, Options


# Data

d = 5
k = 3
n = 1000

W = randn(k, d+1)
X = randn(d, n)
y = (W[:, 1:d] * X .+ W[:,d+1]) + 0.2 * randn(k, n)

# Solve

ret = solve(SumSqrLoss(), k, X, y;
            bias=1.0,
            reg=SqrL2Reg(1.0e-3),
            options=Options(verbosity=:iter, grtol=1.0e-6 * n))

println()

# Print results

W_est = ret.sol
@printf("relative error = %.4e\n", sumabs2(W_est - W) / sumabs2(W))
