# Specific problems


function solve{T<:BlasReal}(loss::UnivariateLoss,
                            X::StridedMatrix{T},
                            y::StridedVector{T};
                            bias::Real=zero(T),
                            reg::Regularizer=ZeroReg(),
                            init::StridedArray{T}=T[],
                            options::Options=Options(),
                            solver::RiskMinSolver=default_solver(),
                            callback::Union(Nothing,Function)=nothing)

    size(X,2) == length(y) ||
        error("Mismatched input dimensions.")

    d = size(X,1)
    dθ = bias == zero(bias) ? d : d+1

    θ = isempty(init) ? zeros(T, dθ) : copy(init)

    if bias == zero(bias)
        solve!(riskmodel(LinearPred(d), loss), reg,
            θ, X, y, solver, options, _nullable_callback(callback))

    else
        solve!(riskmodel(AffinePred(d, bias), loss), reg,
            θ, X, y, solver, options, _nullable_callback(callback))

    end::Solution{Vector{Float64}}
end
