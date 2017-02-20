

### higher level problem-solve functions

function _solve{T<:AbstractFloat}(
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

function solve{T<:AbstractFloat}(
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
has_accelerate(::Solver) = false

function solve!{T<:AbstractFloat}(solver::DescentSolver,
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

    buffer = ndims(θ) == 2 ? zeros(size(θ,1), size(f.X,2)) : zeros(size(f.X,2))

    ## main loop
    t = 0
    converged = false
    v, _ = value_and_grad!(buffer, f, g, θ)

    if vbose >= VERBOSE_ITER
        print_iter_head()
        print_iter(t, v)
    end

    states = init_states(solver, θ, g)

    while !converged && t < maxiter
        t += 1
        v_pre = v

        # accelerate solution
        if has_accelerate(solver)
            accelerate!(solver, t, states, θ)
            v, _ = value_and_grad!(buffer, f, g, θ)
        end

        # descent direction: g
        descent_dir!(solver, t, states, p, θ, θ2, g, g2)

        # backtracking
        α = backtrack!(buffer, f, θ2, θ, v, g, p, one(T), options)
        θ, θ2 = θ2, θ  # swap current solution and previous solution

        # compute new gradient
        v, _ = value_and_grad!(buffer, f, g2, θ)
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


## Gradient Descent Solver

type GD <: DescentSolver
end

Base.show(io::IO, ::GD) = print(io, "GD")
init_states(::GD, θ, g) = nothing

type AGD <: DescentSolver   # Accelerated GD
end

type AGDStates{T}
    d::Int
    τ::T
    xpre::Vector{T}
    y::Vector{T}
end

Base.show(io::IO, ::AGD) = print(io, "AGD")
function init_states{T<:Real}(::AGD, θ::Array{T}, g::Array{T})
    d = length(θ)
    length(g) == d || throw(DimensionMismatch())
    AGDStates{T}(d, one(T), Array(T, d), Array(T, d))
end

prep_dir(::Union{GD,AGD}, g::Array) = g

descent_dir!{T<:Real}(::Union{GD,AGD}, t::Int, states, p::Array{T},
                      θ::Array{T}, θp::Array{T}, g::Array{T}, gp::Array{T}) =
    (is(p, g) || copy!(p, g); p)

has_accelerate(::AGD) = true
function accelerate!{T}(::AGD, t::Int, states::AGDStates{T}, x::Array{T})
    τ = states.τ
    xpre = states.xpre
    y = states.y

    # compute y
    if t > 1
        d = states.d
        τ2 = (one(T) + sqrt(one(T) + 4 * abs2(τ))) * convert(T, 0.5)
        c = (τ - one(T)) / τ2
        @inbounds for i = 1:d
            x_i = x[i]
            y[i] = x_i + c * (x_i - xpre[i])
        end
        # @show τ2
        states.τ = τ2
    end

    # store current x as xpre
    copy!(xpre, x)

    # x <- y
    if t > 1
        copy!(x, y)
    end
    x
end


## BFGS Solver

type BFGS <: DescentSolver
end

Base.show(io::IO, ::BFGS) = print(io, "BFGS")

immutable BFGSStates{T}
    d::Int
    Λ::Matrix{T}
    s::Vector{T}
    y::Vector{T}
    z::Vector{T}
end

function init_states{T<:Real}(::BFGS, θ::Array{T}, g::Array{T})
    d = length(θ)
    length(g) == d || throw(DimensionMismatch())
    BFGSStates{T}(d, eye(T, d), Array(T, d), Array(T, d), Array(T, d))
end

function descent_dir!{T<:Real}(::BFGS, t::Int, states::BFGSStates{T}, p::Array{T},
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
    p
end


## L-BFGS Solver

immutable LBFGS <: DescentSolver
    m::Int    # the numbe of memorized steps
end

Base.show(io::IO, s::LBFGS) = print(io, "LBFGS($(s.m))")

immutable LBFGSStates{T}
    d::Int
    Δx::Matrix{T}
    Δg::Matrix{T}
    ρ::Vector{T}
    α::Vector{T}
    q::Vector{T}
end

function init_states{T<:Real}(solver::LBFGS, θ::Array{T}, g::Array{T})
    d = length(θ)
    length(g) == d || throw(DimensionMismatch())
    m = solver.m
    LBFGSStates{T}(d,
        zeros(T, d, m),     # Δx
        zeros(T, d, m),     # Δg
        zeros(T, m),        # ρ
        zeros(T, m),        # α
        zeros(T, d))        # q
end

function descent_dir!{T<:Real}(solver::LBFGS, t::Int, states::LBFGSStates{T}, p::Array{T},
                               θ::Array{T}, θp::Array{T}, g::Array{T}, gp::Array{T})

    if t > 1
        # extract fields
        m = solver.m
        d = states.d
        Δx = states.Δx
        Δg = states.Δg
        ρ = states.ρ
        α = states.α
        q = states.q
        p_ = vec(p)

        # q <- g
        copy!(q, vec(g))

        # backward pass
        h = min(t-1, m)
        for idx = t-1:-1:t-h
            i = mod1(idx, m)
            α[i] = ρ[i] * dot(Base.view(Δx,:,i), q)
            axpy!(-α[i], Base.view(Δg,:,i), q)
        end

        # p <- q
        copy!(p_, q)

        # forward pass
        for idx = t-h:t-1
            i = mod1(idx, m)
            β = ρ[i] * dot(Base.view(Δg,:,i), p_)
            axpy!(α[i] - β, p_, Base.view(Δx, :, i))
        end
    else
        copy!(p, g)
    end
    p
end
