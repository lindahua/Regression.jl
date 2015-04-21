# Linear regression (using iterative update)

using Regression
import Regression: solve, Options


# Data

d = 3
n = 1000

w = randn(d)
X = randn(d, n)
b = randn()
y = (X'w .+ b) + 0.2 * randn(n)

# Solve

ret = solve(SqrLoss(), X, y;
            bias=1.0,
            reg=SqrL2Reg(1.0e-3),
            options=Options(verbosity=:iter, ftol=1.0e-8 * n))

println()

# Print results

w_g = [w; b]
w_e = ret.sol
@printf("w_g = [%7.4f, %7.4f, %7.4f, %7.4f]\n", w_g[1], w_g[2], w_g[3], w_g[4])
@printf("w_e = [%7.4f, %7.4f, %7.4f, %7.4f]\n", w_e[1], w_e[2], w_e[3], w_e[4])

@printf("relative error = %.4e\n", sumabs2(w_g - w_e) / sumabs2(w_g))
