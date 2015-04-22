
function backtrack!{T<:FloatingPoint}(f::OvjectiveFun{T},
    r::Array{T}, θ::Array{T}, α::T, armijo::T, )

    dv = dot(vec(p), vec(g))
    dv > zero(T) || error("The descent direction is invalid.")
    α = one(T)
    _xmcy!(θ2, θ, α, p)   # θ2 <- θ - α p
    v2 = value(f, θ2)
    while v2 > v - armijo * α * dv
        α > eps(T) || error("Failed to find a proper step size.")
        α *= β
        _xmcy!(θ2, θ, α, p)   # θ2 <- θ - α p
        v2 = value(f, θ2)
    end
end
