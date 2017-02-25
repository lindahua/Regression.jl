
## solver

immutable ProximalDescent{S<:DescentSolver} <: Solver
    dsolver::DescentSolver
end

ProximalDescent{S<:DescentSolver}(solver::S) = ProximalDescent{S}(solver)

typealias ProxGD ProximalDescent{GD}
typealias ProxAGD ProximalDescent{AGD}

ProxGD() = ProximalDescent(GD())::ProxGD
ProxAGD() = ProximalDescent(AGD())::ProxAGD

## higher level

function _solve{T<:AbstractFloat}(
    pb::Problem{T},
    reg::Regularizer,
    θ::Array{T},
    solver::ProximalDescent,
    options::Options,
    callback::Function)

    if has_bias(pb)
        rmodel = riskmodel(pred_with_bias(pb), loss(pb))
        f = RiskFun(rmodel, inputs(pb), outputs(pb))
        solve!(solver, f, reg, θ, options, callback)::Solution{typeof(rmodel.predmodel),typeof(θ)}

    else
        rmodel = riskmodel(pred_without_bias(pb), loss(pb))
        f = RiskFun(rmodel, inputs(pb), outputs(pb))
        solve!(solver, f, reg, θ, options, callback)::Solution{typeof(rmodel.predmodel),typeof(θ)}

    end
end


## core skeleton

function solve!{T<:AbstractFloat}(solver::ProximalDescent,
    f::Functional{T},       # the objective function (smooth part)
    reg::Regularizer,       # the regularizer (non-smooth part)
    θ::Array{T},            # the solution (which would be updated inplace)
    options::Options,       # options to control the procedure
    callback::Function)     # callback function

    ## extract arguments and options

    dsolver = solver.dsolver
    maxiter = options.maxiter
    vbose = verbosity_level(options.verbosity)::Int
    β = options.beta

    ## prepare storage

    θ2 = similar(θ)    # tempoarily new parameter (linear search)
    g = similar(θ)     # gradient w.r.t. f
    g2 = similar(θ)    # another gradient w.r.t. f
    p = prep_dir(dsolver, g)

    ## main loop
    t = 0
    converged = false
    vf, _ = value_and_grad!(f, g, θ)
    vr = value(reg, θ)
    v = vf + vr

    if vbose >= VERBOSE_ITER
        print_iter_head(with_gnorm=false)
        print_iter(t, v)
    end

    states = init_states(dsolver, θ, g)
    α = one(T)
    α_cnt = 0

    while !converged && t < maxiter
        t += 1
        v_pre = v

        # accelerate solution
        if has_accelerate(dsolver)
            accelerate!(dsolver, t, states, θ)
            vf, _ = value_and_grad!(f, g, θ)
            vr = value(reg, θ)
            v = vf + vr
        end

        # descent direction: g
        descent_dir!(dsolver, t, states, p, θ, θ2, g, g2)

        # backtracking
        α_pre = α
        if α_cnt > 3  # if α stays there for a while, we may begin with a bigger step
            α_pre = max(α_pre / β, one(T))
            α_cnt = 0
        end

        # Note: proximal operator has been applied during backtrack
        α = prox_backtrack!(f, reg, θ2, θ, vf, vr, g, p, α_pre, options)
        if α == α_pre
            α_cnt += 1
        end
        θ, θ2 = θ2, θ  # swap current solution and previous solution

        # compute new gradient
        vf, _ = value_and_grad!(f, g2, θ)
        vr = value(reg, θ)
        v = vf + vr
        g, g2 = g2, g  # swap current gradient with previous gradient

        # test convergence
        converged = test_convergence(θ, θ2, v, v_pre, options)

        # print iteration
        if vbose >= VERBOSE_ITER
            print_iter(t, v, v_pre, α)
        end

        # invoke callback (when requested)
        callback(t, θ, v, g)
    end

    if vbose >= VERBOSE_FINAL
        print_final(t, v, converged)
    end

    return Solution(f.rmodel.predmodel, θ, v, t, converged)
end
