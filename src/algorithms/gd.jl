
type GDSolver <: DescentSolver
end

function solve!{T<:FloatingPoint}(::GDSolver,
    f::ObjectiveFun{T},        # the objective function
    θ::Array{T},               # the solution (which would be updated inplace)
    options::Options,          # options to control the procedure
    callback::Function)        # callback function

    ## extract arguments and options

    maxiter = options.maxiter
    vbose = verbosity_level(options.verbosity)::Int

    ## prepare storage

    θ2 = similar(θ)    # tempoarily new parameter (linear search)
    g = similar(θ)     # gradient
    g2 = similar(θ)    # another gradient

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

        # descent direction: g
        p = g

        # backtracking
        α = backtrack!(f, θ2, θ, v, g, p, one(T), options)
        θ, θ2 = θ2, θ  # swap current solution and previous solution

        # compute new gradient
        v, _ = value_and_grad!(f, g2, θ)
        g, g2 = g2, g  # swap current gradient with previous gradient

        # test convergence
        converged = test_convergence(θ, θ2, v, v_pre, g, options)

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
