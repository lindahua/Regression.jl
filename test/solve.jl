using Regression
using Base.Test


srand(54321)  # ensure repeatable experiment
              # avoid CI issues

## auxiliary functions

_relerr(x::Array, x0::Array) = sumabs2(x - x0) / sumabs2(x0)
_corr(x::Array, x0::Array) = dot(vec(x), vec(x0)) / (vecnorm(x) * vecnorm(x0))

_classify(U::Matrix) = vec(mapslices(indmax, U, 1))

_errrate(y::Array{Int}, y0::Array{Int}) = countnz(y .!= y0) / length(y0)

## Univariate loss

# data
d = 5
n = 2000
w = randn(d)
wa = [w; randn()]

X = randn(d, n)
u = X'w + 0.01 * randn(n)
ua = u .+ wa[d+1]

# test cases

println("\twith LinearPred + SqrLoss")

ret = Regression.solve(SqrLoss(), X, u; reg=SqrL2Reg(1.0e-4))
@test isa(ret, Regression.Solution{Vector{Float64}})
r = ret.sol
@test size(r) == size(w)
@test _relerr(r, w) < 1.0e-4

println("\twith AffinePred + SqrLoss")

ret = Regression.solve(SqrLoss(), X, ua; bias=1.0, reg=SqrL2Reg(1.0e-4))
@test isa(ret, Regression.Solution{Vector{Float64}})
r = ret.sol
@test size(r) == size(wa)
@test _relerr(r, wa) < 1.0e-4

println("\twith LinearPred + LogisticLoss")

ret = Regression.solve(LogisticLoss(), X, sign(u); reg=SqrL2Reg(1.0e-4))
@test isa(ret, Regression.Solution{Vector{Float64}})
r = ret.sol
@test size(r) == size(w)
@test abs(1.0 - _corr(r, w)) < 1.0e-3

println("\twith AffinePred + LogisticLoss")

ret = Regression.solve(LogisticLoss(), X, sign(ua); bias=1.0, reg=SqrL2Reg(1.0e-4))
@test isa(ret, Regression.Solution{Vector{Float64}})
r = ret.sol
@test size(r) == size(wa)
@test abs(1.0 - _corr(r, wa)) < 1.0e-3


## Multivariate loss

# data
d = 5
k = 3
W = randn(k, d)
Wa = [W randn(k)]

X = randn(d, n)
U = W * X + 0.01 * randn(k, n)
Ua = U .+ Wa[:,d+1]

# test cases

println("\twith MvLinearPred + SumSqrLoss")

ret = Regression.solve(SumSqrLoss(), k, X, U; reg=SqrL2Reg(1.0e-4))
@test isa(ret, Regression.Solution{Matrix{Float64}})
R = ret.sol
@test size(R) == size(W)
@test _relerr(R, W) < 1.0e-4

println("\twith MvAffinePred + SumSqrLoss")

ret = Regression.solve(SumSqrLoss(), k, X, Ua; bias=1.0, reg=SqrL2Reg(1.0e-4))
@test isa(ret, Regression.Solution{Matrix{Float64}})
R = ret.sol
@test size(R) == size(Wa)
@test _relerr(R, Wa) < 1.0e-4

println("\twith MvLinearPred + MultiLogisticLoss")

Y = _classify(U)
ret = Regression.solve(MultiLogisticLoss(), k, X, Y; reg=SqrL2Reg(1.0e-4))
R = ret.sol
@test size(R) == size(W)
Yr = _classify(predict(MvLinearPred(d, k), R, X))
@test _errrate(Yr, Y) < 0.03

println("\twith MvAffinePred + MultiLogisticLoss")

Y = _classify(Ua)
ret = Regression.solve(MultiLogisticLoss(), k, X, Y; bias=1.0, reg=SqrL2Reg(1.0e-4))
R = ret.sol
@test size(R) == size(Wa)
Yr = _classify(predict(MvAffinePred(d, k), R, X))
@test _errrate(Yr, Y) < 0.03
