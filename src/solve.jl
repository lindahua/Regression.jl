

### higher level problem-solve functions

function _solve{T<:FloatingPoint}(
    pb::Problem{T},
    reg::Regularizer,
    θ::Array{T},
    solver::DescentSolver,
    options::Options,
    callback::Function)

    if has_bias(pb)
        rmodel = riskmodel(pred_with_bias(pb), loss(pb))
        f = RegRiskFun(rmodel, reg, inputs(pb), outputs(pb))
        solve!(solver, f, θ, options, callback)

    else
        rmodel = riskmodel(pred_without_bias(pb), loss(pb))
        f = RegRiskFun(rmodel, reg, inputs(pb), outputs(pb))
        solve!(solver, f, θ, options, callback)

    end::Solution{typeof(θ)}
end


function solve{T<:FloatingPoint}(
    pb::Problem{T};
    reg::Regularizer=ZeroReg(),
    init::StridedArray{T}=T[],
    solver::Solver=default_solver,
    options::Options=Options(),
    callback::Function=no_op)

    θ = isempty(init) ? initsol(pb) : copy(init)
    _solve(pb, reg, θ, solver, options, callback)
end


### Specific solvers

prep_dir(::Solver, g::Array) = similar(g)

function solve!{T<:FloatingPoint}(solver::DescentSolver,
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
    p = prep_dir(solver, g)

    ## main loop
    t = 0
    converged = false
    v, _ = value_and_grad!(f, g, θ)

    if vbose >= VERBOSE_ITER
        print_iter_head()
        print_iter(t, v)
    end

    states = init_states(solver, θ, g)

    while !converged && t < maxiter
        t += 1
        v_pre = v

        # descent direction: g
        descent_dir!(solver, t, states, p, θ, θ2, g, g2)

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

## GDSolver

type GDSolver <: DescentSolver
end

prep_dir(::GDSolver, g::Array) = g

init_states(::GDSolver, θ, g) = nothing

descent_dir!{T<:Real}(::GDSolver, t::Int, states, p::Array{T},
                      θ::Array{T}, θp::Array{T}, g::Array{T}, gp::Array{T}) =
    is(p, g) || copy!(p, g)


## BFGSSolver

type BFGSSolver <: DescentSolver
end

immutable BFGSStates{T}
    d::Int
    Λ::Matrix{T}
    s::Vector{T}
    y::Vector{T}
    z::Vector{T}
end

function init_states{T<:Real}(::BFGSSolver, θ::Array{T}, g::Array{T})
    d = length(θ)
    BFGSStates{T}(d, eye(T, d), Array(T, d), Array(T, d), Array(T, d))
end

function descent_dir!{T<:Real}(::BFGSSolver, t::Int, states::BFGSStates{T}, p::Array{T},
                               θ::Array{T}, θp::Array{T}, g::Array{T}, gp::Array{T})
    if t > 1
        d = states.d
        Λ = states.Λ
        s = states.s
        y = states.y
        z = states.z

        # compute s <- θ - θ_pre
        @inbounds for i = 1:d
            s[i] = θ[i] - θp[i]
        end

        # compute y <- g - g_pre
        @inbounds for i = 1:d
            y[i] = g[i] - gp[i]
        end
        A_mul_B!(z, Λ, y)

        u = dot(s, y)
        c1 = (u + dot(y, z)) / (u * u)
        c2 = one(T) / u

        # update Λ
        @inbounds for j = 1:d
            s_j, z_j = s[j], z[j]
            for i = 1:d
                s_i, z_i = s[i], z[i]
                Λ[i,j] += (c1 * s_i * s_j - c2 *(z_i * s_j + z_j * s_i))
            end
        end

        # compute direction
        A_mul_B!(vec(p), Λ, vec(g))
    else
        copy!(p, g)
    end
end
