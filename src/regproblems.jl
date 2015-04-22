# Common regression problems

abstract Problem

### Univariate Prediction Problem

immutable UnivariateRegression{L<:UnivariateLoss,
                               T<:FloatingPoint,
                               XT<:StridedMatrix,
                               YT<:StridedVector} <: Problem
    loss::L
    d::Int
    n::Int
    bias::T
    X::XT
    Y::YT
end

function UnivariateRegression{T<:FloatingPoint}(loss::UnivariateLoss,
                                                X::StridedMatrix{T},
                                                Y::StridedVector;
                                                bias::Real=0.0)
    d, n = size(X)
    length(Y) == n || throw(DimensionMismatch())
    UnivariateRegression{typeof(loss), T, typeof(X), typeof(Y)}(
        loss, d, n, convert(T, bias), X, Y)
end

nsamples(pb::UnivariateRegression) = pb.n
inputs(pb::UnivariateRegression) = pb.X
outputs(pb::UnivariateRegression) = pb.Y

loss(pb::UnivariateRegression) = pb.loss

has_bias(pb::UnivariateRegression) = (pb.bias != 0)
pred_without_bias(pb::UnivariateRegression) = LinearPred(pb.d)
pred_with_bias(pb::UnivariateRegression) = AffinePred(pb.d, pb.bias)

# specific problems

linearreg{T<:FloatingPoint}(X::StridedMatrix{T}, y::StridedVector{T}; bias::Real=0.0) =
    UnivariateRegression(SqrLoss(), X, y; bias=bias)

logisticreg{T<:FloatingPoint}(X::StridedMatrix{T}, y::StridedVector; bias::Real=0.0) =
    UnivariateRegression(LogisticLoss(), X, convert(Vector{T}, y); bias=bias)


### Multivariate Prediction Problem

immutable MultivariateRegression{L<:MultivariateLoss,
                                 T<:FloatingPoint,
                                 XT<:StridedMatrix,
                                 YT<:StridedVecOrMat} <: Problem
    loss::L
    d::Int
    k::Int
    n::Int
    bias::T
    X::XT
    Y::YT
end

function MultivariateRegression{T<:FloatingPoint}(loss::MultivariateLoss,
                                                  X::StridedMatrix{T},
                                                  Y::StridedVecOrMat,
                                                  k::Int;
                                                  bias::Real=0.0)
    d, n = size(X)
    size(Y, ndims(Y)) == n || throw(DimensionMismatch())
    MultivariateRegression{typeof(loss), T, typeof(X), typeof(Y)}(
        loss, d, k, n, convert(T, bias), X, Y)
end


nsamples(pb::MultivariateRegression) = pb.n
inputs(pb::MultivariateRegression) = pb.X
outputs(pb::MultivariateRegression) = pb.Y

loss(pb::MultivariateRegression) = pb.loss

has_bias(pb::MultivariateRegression) = (pb.bias != 0)
pred_without_bias(pb::MultivariateRegression) = MvLinearPred(pb.d, pb.k)
pred_with_bias(pb::MultivariateRegression) = MvAffinePred(pb.d, pb.k, pb.bias)


# specific problems

linearreg{T}(X::StridedMatrix{T}, Y::StridedMatrix{T}; bias::Real=0.0) =
    MultivariateRegression(SumSqrLoss(), X, Y, size(Y,1); bias=bias)

mlogisticreg{T}(X::StridedMatrix{T}, Y::StridedVector{Int}, k::Int; bias::Real=0.0) =
    MultivariateRegression(MultiLogisticLoss(), X, Y, k; bias=bias)
