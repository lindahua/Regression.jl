# Specific problems


function solve{T<:BlasReal}(loss::UnivariateLoss,
                            X::StridedMatrix{T},
                            y::StridedVector;
                            bias::Real=zero(T),
                            reg::Regularizer=ZeroReg(),
                            init::StridedArray{T}=T[],
                            options::Options=Options(),
                            solver::RiskMinSolver=default_solver(),
                            callback::Function=no_op)

    size(X,2) == length(y) ||
        error("Mismatched input dimensions.")

    d = size(X,1)
    dθ = bias == zero(bias) ? d : d+1

    θ = isempty(init) ? zeros(T, dθ) : copy(init)

    if bias == zero(bias)
        solve!(riskmodel(LinearPred(d), loss), reg,
            θ, X, y, solver, options, callback)

    else
        solve!(riskmodel(AffinePred(d, bias), loss), reg,
            θ, X, y, solver, options, callback)

    end::Solution{Vector{T}}
end


function solve{T<:BlasReal}(loss::MultivariateLoss, k::Int,
                            X::StridedMatrix{T},
                            Y::StridedArray;
                            bias::Real=zero(T),
                            reg::Regularizer=ZeroReg(),
                            init::StridedArray{T}=T[],
                            options::Options=Options(),
                            solver::RiskMinSolver=default_solver(),
                            callback::Function=no_op)

    size(X,2) == size(Y,ndims(Y)) ||
        error("Mismatched input dimensions.")

    d = size(X,1)
    dθ = bias == zero(bias) ? d : d+1

    θ = isempty(init) ? zeros(T, k, dθ) : copy(init)

    if bias == zero(bias)
        solve!(riskmodel(MvLinearPred(d, k), loss), reg,
            θ, X, Y, solver, options, callback)

    else
        solve!(riskmodel(MvAffinePred(d, k, bias), loss), reg,
            θ, X, Y, solver, options, callback)

    end::Solution{Matrix{T}}
end
