# Numerical solver

## Options

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

default_solver() = GDSolver()


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


## Solve

function solve!{T<:FloatingPoint}(rmodel::SupervisedRiskModel,    # the risk model
                                  reg::Regularizer,               # the regularizer
                                  θ::Array{T},                    # the solution (which would be updated inplace)
                                  X::StridedArray{T},             # array of inputs
                                  y::StridedArray,                # array of outputs
                                  solver::RiskMinSolver,          # solver
                                  options::Options,               # options to control the procedure
                                  callback::Function)             # callback function

    ## extract arguments and options

    maxiter = options.maxiter
    ftol = convert(T, options.ftol)
    xtol = convert(T, options.xtol)
    grtol = convert(T, options.grtol)
    armijo = convert(T, options.armijo)
    β = convert(T, options.beta)
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
    v, _ = value_and_grad!(rmodel, reg, g, θ, X, y)

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
        dv = dot(vec(p), vec(g))
        dv > zero(T) || error("The descent direction is invalid.")
        α = one(T)
        _xmcy!(θ2, θ, α, p)   # θ2 <- θ - α p
        v2 = value(rmodel, θ2, X, y) + value(reg, θ2)
        while v2 > v - armijo * α * dv
            α > eps(T) || error("Failed to find a proper step size.")
            α *= β
            _xmcy!(θ2, θ, α, p)   # θ2 <- θ - α p
            v2 = value(rmodel, θ2, X, y) + value(reg, θ2)
        end
        θ, θ2 = θ2, θ  # swap current solution and previous solution

        # compute new gradient
        v, _ = value_and_grad!(rmodel, reg, g2, θ, X, y)
        g, g2 = g2, g  # swap current gradient with previous gradient

        # test convergence
        converged = abs(v - v_pre) < ftol ||
                    vecnorm(g) < grtol ||
                    _l2diff(θ, θ2) < xtol

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
