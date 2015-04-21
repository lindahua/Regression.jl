# Logistic regression

using Regression

# Data

d = 3
n = 1000

w = randn(d)
X = randn(d, n)
y = sign(X'w + 0.2 * randn(n))

# Solve

ret = Regression.solve(
    riskmodel(LinearPred(d), LogisticLoss()),
    SqrL2Reg(1.0e-2),
    zeros(3), X, y;
    options=RiskMinOptions(verbosity=:iter, ftol=1.0e-5 * n))
    
println()

# Print results

w_e = ret.sol

@printf("corr(truth, estimated) = %.6f\n", dot(w, w_e) / (vecnorm(w) * vecnorm(w_e)))
