
# include algorithms

include("algorithms/gd.jl")
include("algorithms/bfgs.jl")

const default_solver = BFGSSolver()


### higher level problem-solve functions

function solve{T<:FloatingPoint}(
    pb::Problem{T};
    reg::Regularizer=ZeroReg(),
    init::StridedArray{T}=T[],
    solver::Solver=default_solver,
    options::Options=Options(),
    callback::Function=no_op)

    θ = isempty(init) ? initsol(pb) : copy(init)

    if has_bias(pb)
        rmodel = riskmodel(pred_with_bias(pb), loss(pb))
        f = RegularizedRiskFun(rmodel, reg, inputs(pb), outputs(pb))
        solve!(solver, f, θ, options, callback)

    else
        rmodel = riskmodel(pred_without_bias(pb), loss(pb))
        f = RegularizedRiskFun(rmodel, reg, inputs(pb), outputs(pb))
        solve!(solver, f, θ, options, callback)

    end::Solution{typeof(θ)}
end
