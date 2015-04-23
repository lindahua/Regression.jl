
### Options

type Options
    maxiter::Int        # maximum number of iterations
    ftol::Float64       # function value change tolerance
    xtol::Float64       # solution change tolerance
    grtol::Float64      # gradient norm tolerance
    armijo::Float64     # Armijo coefficient for line search
    beta::Float64       # backtracking ratio
    verbosity::Symbol   # verbosity (:none | :final | :iter)
end

function Options(;maxiter::Integer=200,
                  ftol::Real=1.0e-6,
                  xtol::Real=1.0e-8,
                  grtol::Real=1.0e-8,
                  armijo::Real=0.5,
                  beta::Real=0.5,
                  verbosity::Symbol=:none)

     maxiter > 1 || error("maxiter must be an integer greater than 1.")
     ftol > 0 || error("ftol must be a positive real value.")
     xtol > 0 || error("xtol must be a positive real value.")
     grtol > 0 || error("grtol must be a positive real value.")
     0 < armijo < 1 || error("armijo must be a real value in (0, 1).")
     0 < beta < 1 || error("beta must be a real value in (0, 1).")
     (verbosity == :none || verbosity == :final || verbosity == :iter) ||
         error("verbosity must be either :none, :final, or :iter.")

     Options(convert(Int, maxiter),
             convert(Float64, ftol),
             convert(Float64, xtol),
             convert(Float64, grtol),
             convert(Float64, armijo),
             convert(Float64, beta),
             verbosity)
end


### Objective Function

abstract Functional{T}

immutable RiskFun{T,
                  XT<:StridedArray,
                  YT<:StridedArray,
                  RModel<:RiskModel} <: Functional{T}

    rmodel::RModel
    X::XT
    Y::YT
end

RiskFun{T<:FloatingPoint}(rmodel::RiskModel, X::StridedArray{T}, Y::StridedArray) =
    RiskFun{T, typeof(X), typeof(Y), typeof(rmodel)}(rmodel, X, Y)

value{T<:FloatingPoint}(f::RiskFun{T}, θ::StridedArray{T}) = value(f.rmodel, θ, f.X, f.Y)

value_and_grad!{T<:FloatingPoint}(f::RiskFun{T}, g::StridedArray{T}, θ::StridedArray{T}) =
    value_and_addgrad!(f.rmodel, zero(T), g, one(T), θ, f.X, f.Y)


immutable RegRiskFun{T,
                     XT<:StridedArray,
                     YT<:StridedArray,
                     RModel<:RiskModel,
                     Reg<:Regularizer} <: Functional{T}
    rmodel::RModel
    reg::Reg
    X::XT
    Y::YT
end

RegRiskFun{T<:FloatingPoint}(rmodel::RiskModel, reg::Regularizer, X::StridedArray{T}, Y::StridedArray) =
    RegRiskFun{T, typeof(X), typeof(Y), typeof(rmodel), typeof(reg)}(rmodel, reg, X, Y)

value{T<:FloatingPoint}(f::RegRiskFun{T}, θ::StridedArray{T}) =
    value(f.rmodel, θ, f.X, f.Y) + value(f.reg, θ)

function value_and_grad!{T<:FloatingPoint}(f::RegRiskFun{T}, g::StridedArray{T}, θ::StridedArray{T})
    v_risk, _ = value_and_addgrad!(f.rmodel, zero(T), g, one(T), θ, f.X, f.Y)
    v_regr, _ = value_and_addgrad!(f.reg, one(T), g, one(T), θ)
    return (v_risk + v_regr, g)
end


### Solution

immutable Solution{Sol<:StridedArray}
    sol::Sol
    fval::Float64
    niters::Int
    converged::Bool
end

function Base.show(io::IO, r::Solution)
    println(io, "RiskMinSolution:")
    println(io, "- sol:       $(size(r.sol)) $(typeof(r.sol))")
    println(io, "- fval:      $(r.fval)")
    println(io, "- niters:    $(r.niters)")
    println(io, "- converged: $(r.converged)")
end


### Solver

abstract Solver
abstract DescentSolver <: Solver



### Line search

function backtrack!{T<:FloatingPoint}(
    f::Functional{T},       # objective function
    θr::Array{T},           # destination solution
    θ::Array{T},            # last solution
    v::T,                   # objective value at θ
    g::Array{T},            # gradient at θ
    p::Array{T},            # descent direction
    α::T,                   # initial step size
    opts::Options)          # options

    armijo = convert(T, opts.armijo)
    β = convert(T, opts.beta)
    dv = dot(vec(p), vec(g))
    dv > zero(T) || error("The descent direction is invalid.")

    _xmcy!(θr, θ, α, p)   # θr <- θ - α p
    v2 = value(f, θr)
    while v2 > v - armijo * α * dv
        α > eps(T) || error("Failed to find a proper step size.")
        α *= β
        _xmcy!(θr, θ, α, p)   # θr <- θ - α p
        v2 = value(f, θr)
    end
    return α
end


### test convergence

test_convergence{T<:FloatingPoint}(θ::Array{T}, θpre::Array{T}, v::T, vpre::T, g::Array{T}, opt::Options) =
    abs(v - vpre) < convert(T, opt.ftol) ||
    vecnorm(g) < convert(T, opt.grtol) ||
    _l2diff(θ, θpre) < convert(T, opt.xtol)

test_convergence{T<:FloatingPoint}(θ::Array{T}, θpre::Array{T}, v::T, vpre::T, opt::Options) =
    abs(v - vpre) < convert(T, opt.ftol) ||
    _l2diff(θ, θpre) < convert(T, opt.xtol)


### auxiliary functions

function _l2diff{T<:FloatingPoint}(x::StridedArray{T}, y::StridedArray{T})
    @assert length(x) == length(y)
    s = zero(T)
    @inbounds for i = 1:length(x)
        s += abs2(x[i] - y[i])
    end
    return sqrt(s)
end

function _xmcy!{T<:Real}(r::Array{T}, x::Array{T}, c::T, y::Array{T})
    @inbounds for i = 1:length(x)
        r[i] = x[i] - c * y[i]
    end
    r
end
