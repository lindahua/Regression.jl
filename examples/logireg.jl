# Logistic regression

using Regression
import Regression: solve, Options


# Data

d = 3
n = 1000

w = randn(d+1)
X = randn(d, n)
y = sign(X'w[1:d] + w[d+1] + 0.2 * randn(n))

# Solve

ret = solve(LogisticLoss(), X, y;
            bias=1.0,
            reg=SqrL2Reg(1.0e-2),
            options=Options(verbosity=:iter, ftol=1.0e-5 * n))

println()

# Print results

w_e = ret.sol

@printf("corr(truth, estimated) = %.6f\n", dot(w, w_e) / (vecnorm(w) * vecnorm(w_e)))
