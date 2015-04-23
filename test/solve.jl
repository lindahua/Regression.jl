using Regression
using Base.Test


srand(54321)  # ensure repeatable experiment
              # avoid CI issues

## auxiliary functions

_relerr(x::Array, x0::Array) = sumabs2(x - x0) / sumabs2(x0)
_corrdist(x::Array, x0::Array) = 1.0 - dot(vec(x), vec(x0)) / (vecnorm(x) * vecnorm(x0))

_classify(U::Matrix) = vec(mapslices(indmax, U, 1))

function _errrate(pred::PredictionModel, X::Matrix, θ::Array, θ0::Array)
    u0 = _classify(predict(pred, θ0, X))
    u  = _classify(predict(pred, θ, X))
    countnz(u .!= u0) / length(u0)
end


const _solvers = [GD(), BFGS(), LBFGS(6)]

function verify_solver(title, loss::Loss, data,
                       θg::Array, vcond::Function, thres::Real)
    println("    $title")

    X = data[1]
    dx = size(X, 1)
    dθ = size(θg, ndims(θg))
    b = dθ == dx ? 0.0 :
        dθ == dx + 1 ? 1.0 :
        error("Unmatched dimensions.")

    pb = ndims(θg) == 1 ? UnivariateRegression(loss, data...; bias=b) :
                          MultivariateRegression(loss, data...; bias=b)

    for solv in _solvers
        println("      - with solver $(solv)")
        n = size(X, ndims(X))
        ret = Regression.solve(pb;
                               reg=SqrL2Reg(1.0e-4),
                               solver=solv,
                               options=Regression.Options(grtol=n * 1.0e-6))

        @test isa(ret, Regression.Solution)
        θe = ret.sol
        @test vcond(θe, θg) < thres
    end
end


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

verify_solver("LinearPred + SqrLoss",
    SqrLoss(), (X, u), w, _relerr, 1.0e-4)

verify_solver("AffinePred + SqrLoss",
    SqrLoss(), (X, ua), wa, _relerr, 1.0e-4)

verify_solver("LinearPred + LogisticLoss",
    SqrLoss(), (X, u), w, _corrdist, 1.0e-3)

verify_solver("AffinePred + LogisticLoss",
    SqrLoss(), (X, ua), wa, _corrdist, 1.0e-3)


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

verify_solver("MvLinearPred + SumSqrLoss",
    SumSqrLoss(), (X, U, k), W, _relerr, 1.0e-4)

verify_solver("MvAffinePred + SumSqrLoss",
    SumSqrLoss(), (X, Ua, k), Wa, _relerr, 1.0e-4)

y = _classify(U)
verify_solver("MvLinearPred + MultiLogisticLoss",
    MultiLogisticLoss(), (X, y, k), W,
        (θ, θ0) -> _errrate(MvLinearPred(d, k), X, θ, θ0), 0.02)

ya = _classify(Ua)
verify_solver("MvAffinePred + MultiLogisticLoss",
    MultiLogisticLoss(), (X, ya, k), Wa,
        (θ, θ0) -> _errrate(MvAffinePred(d, k), X, θ, θ0), 0.02)
