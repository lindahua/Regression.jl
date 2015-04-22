using Regression
using Base.Test

import Regression: loss, has_bias, pred_without_bias, pred_with_bias

d = 5
k = 3
n = 100
X = randn(d, n)
y = randn(n)
Y = randn(k, n)
c = rand(1:k, n)
b = 2.0

pb = linearreg(X, y; bias=b)
@test isa(pb, UnivariateRegression{SqrLoss})
@test has_bias(pb)
@test loss(pb) == SqrLoss()
@test pred_without_bias(pb) == LinearPred(d)
@test pred_with_bias(pb) == AffinePred(d, b)

pb = linearreg(X, Y; bias=b)
@test isa(pb, MultivariateRegression{SumSqrLoss})
@test has_bias(pb)
@test loss(pb) == SumSqrLoss()
@test pred_without_bias(pb) == MvLinearPred(d, k)
@test pred_with_bias(pb) == MvAffinePred(d, k, b)

pb = logisticreg(X, sign(y); bias=b)
@test isa(pb, UnivariateRegression{LogisticLoss})
@test has_bias(pb)
@test loss(pb) == LogisticLoss()
@test pred_without_bias(pb) == LinearPred(d)
@test pred_with_bias(pb) == AffinePred(d, b)

pb = mlogisticreg(X, c, k; bias=b)
@test isa(pb, MultivariateRegression{MultiLogisticLoss})
@test has_bias(pb)
@test loss(pb) == MultiLogisticLoss()
@test pred_without_bias(pb) == MvLinearPred(d, k)
@test pred_with_bias(pb) == MvAffinePred(d, k, b)
