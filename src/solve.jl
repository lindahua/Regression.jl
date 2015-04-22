# Numerical solver

default_solver() = BFGSSolver()


## Solution

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


## core solve! function

function solve!{T<:FloatingPoint}(
    f::ObjectiveFun{T},        # the objective function
    θ::Array{T},               # the solution (which would be updated inplace)
    solver::RiskMinSolver,     # solver
    options::Options,          # options to control the procedure
    callback::Function)        # callback function

    ## extract arguments and options

    maxiter = options.maxiter
    vbose = verbosity_level(options.verbosity)::Int

    ## prepare storage

    θ2 = similar(θ)    # tempoarily new parameter (linear search)
    g = similar(θ)     # gradient
    g2 = similar(θ)    # another gradient
    p = _prep_searchdir(solver, g)
    states = init_states(solver, θ)   # solver state

    ## main loop
    t = 0
    converged = false
    v, _ = value_and_grad!(f, g, θ)

    if vbose >= VERBOSE_ITER
        print_iter_head()
        print_iter(t, v)
    end

    while !converged && t < maxiter
        t += 1
        v_pre = v

        # compute descent direction
        descent_dir!(solver, states, θ, g, p)

        # backtracking
        α = backtrack!(f, θ2, θ, v, g, p, one(T), options)
        θ, θ2 = θ2, θ  # swap current solution and previous solution

        # compute new gradient
        v, _ = value_and_grad!(f, g2, θ)
        g, g2 = g2, g  # swap current gradient with previous gradient

        # test convergence
        converged = test_convergence(θ, θ2, v, v_pre, g, options)

        # update states (if necessary)
        if !converged
            update_states!(solver, states, θ, θ2, g, g2)
        end

        # print iteration
        if vbose >= VERBOSE_ITER
            print_iter(t, v, v_pre, g, α)
        end

        # invoke callback (when requested)
        callback(t, θ, v, g)
    end

    if vbose >= VERBOSE_FINAL
        print_final(t, v, converged)
    end

    return Solution(θ, v, t, converged)
end


### higher level problem-solve functions

function solve{T<:FloatingPoint}(
    pb::Problem{T};
    reg::Regularizer=ZeroReg(),
    init::StridedArray{T}=T[],
    solver::RiskMinSolver=default_solver(),
    options::Options=Options(),
    callback::Function=no_op)

    θ = isempty(init) ? initsol(pb) : copy(init)

    if has_bias(pb)
        rmodel = riskmodel(pred_with_bias(pb), loss(pb))
        f = RegularizedRiskFun(rmodel, reg, inputs(pb), outputs(pb))
        solve!(f, θ, solver, options, callback)

    else
        rmodel = riskmodel(pred_without_bias(pb), loss(pb))
        f = RegularizedRiskFun(rmodel, reg, inputs(pb), outputs(pb))
        solve!(f, θ, solver, options, callback)

    end::Solution{typeof(θ)}
end
