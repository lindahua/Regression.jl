## Abstract type

abstract RiskMinSolver

## Generic functions

_prep_searchdir(::RiskMinSolver, g::StridedArray) = similar(g)


### Solver: Gradient descent

type GDSolver <: RiskMinSolver
end

init_states(::GDSolver, θ) = nothing

descent_dir!{T<:Real}(::GDSolver, st::Nothing, θ::Array{T}, g::Array{T}, p::Array{T}) =
    (is(p, g) || copy!(p, g); p)

update_states!{T<:Real}(::GDSolver, st::Nothing,
    θ::Array{T}, θpre::Array{T}, g::Array{T}, gpre::Array{T}) = nothing

_prep_searchdir(::GDSolver, g::StridedArray) = g


### Solver: BFGS
#
# The implementation follows Wikipedia
#

type BFGSSolver <: RiskMinSolver
end

type BFGSState{T<:BlasReal}
    Λ::Matrix{T}   # B^{-1}
    s::Vector{T}   # x_{t+1} - x_t
    y::Vector{T}   # g_{t+1} - g_t
    z::Vector{T}   # Λ * y

    BFGSState(n::Int) = new(eye(T, n), Array(T, n), Array(T, n), Array(T, n))
end

init_states{T<:BlasReal}(::BFGSSolver, θ::Array{T}) = BFGSState{T}(length(θ))

function descent_dir!{T<:BlasReal}(::BFGSSolver, st::BFGSState{T},
                                   θ::Array{T}, g::Array{T}, p::Array{T})
    A_mul_B!(vec(p), st.Λ, vec(g))
end

function update_states!{T<:BlasReal}(::BFGSSolver, st::BFGSState{T},
                                     θ::Array{T}, θpre::Array{T},
                                     g::Array{T}, gpre::Array{T})

    # extract fields
    Λ = st.Λ
    s = st.s
    y = st.y
    z = st.z

    # compute intermediate quantities
    n = length(θ)
    @inbounds for i = 1:n
        s[i] = θ[i] - θpre[i]
    end
    @inbounds for i = 1:n
        y[i] = g[i] - gpre[i]
    end
    A_mul_B!(z, Λ, y)

    u = dot(s, y)
    c1 = (u + dot(y, z)) / (u * u)
    c2 = one(T) / u

    # update Λ
    @inbounds for j = 1:n, i = 1:n
        s_i = s[i]
        s_j = s[j]
        z_i = z[i]
        z_j = z[j]

        Λ[i,j] += (c1 * s_i * s_j - c2 *(z_i * s_j + z_j * s_i))
    end
    st
end
