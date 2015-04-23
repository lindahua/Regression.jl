
type BFGSSolver <: DescentSolver
end

function solve!{T<:FloatingPoint}(::BFGSSolver,
    f::Functional{T},       # the objective function
    θ::Array{T},            # the solution (which would be updated inplace)
    options::Options,       # options to control the procedure
    callback::Function)     # callback function

    ## extract arguments and options

    maxiter = options.maxiter
    vbose = verbosity_level(options.verbosity)::Int

    ## prepare storage

    θ2 = similar(θ)    # tempoarily new parameter (linear search)
    g = similar(θ)     # gradient
    g2 = similar(θ)    # another gradient
    p = similar(θ)     # descent direction

    d = length(θ)
    Λ = eye(T, d)   # B^{-1}
    s = Array(T, d)   # x_{t+1} - x_t
    y = Array(T, d)   # g_{t+1} - g_t
    z = Array(T, d)   # Λ * y

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

        # descent direction: p = Λ * g
        A_mul_B!(vec(p), Λ, vec(g))

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

        # update states
        if !converged

            # compute s <- θ - θ_pre
            @inbounds for i = 1:d
                s[i] = θ[i] - θ2[i]
            end

            # compute y <- g - g_pre
            @inbounds for i = 1:d
                y[i] = g[i] - g2[i]
            end
            A_mul_B!(z, Λ, y)

            u = dot(s, y)
            c1 = (u + dot(y, z)) / (u * u)
            c2 = one(T) / u

            # update Λ
            @inbounds for j = 1:d, i = 1:d
                s_i = s[i]
                s_j = s[j]
                z_i = z[i]
                z_j = z[j]

                Λ[i,j] += (c1 * s_i * s_j - c2 *(z_i * s_j + z_j * s_i))
            end
        end
    end

    if vbose >= VERBOSE_FINAL
        print_final(t, v, converged)
    end

    return Solution(θ, v, t, converged)
end
