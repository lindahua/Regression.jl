# Multi-class logistic regression

immutable MultiClassLogisticRegressFunctor <: DifferentiableRegressFunctor end


####
#
#  rf(u, y) = log(sum_k exp(u_k)) - u_y
#
#  rf'(u, y) = p_k - I(y == k) w.r.t. u
#  

function evaluate_values!(
	rf::MultiClassLogisticRegressFunctor, 
	u::Matrix{Float64}, 
	y::Vector{Int}, 
	v::Vector{Float64})

	K = size(u, 1)
	n = size(u, 2)

	for j in 1 : n		
		yj = y[j]

		maxu = u[1, j] 
		for k in 2 : K
			uk = u[k, j]
			if uk > maxu
				maxu = uk
			end
		end

		sj = 0.
		for k in 1 : K
			sj += exp(u[k, j] - maxu)
		end

		v[j] = log(sj) + maxu - u[yj, j]
	end
end

function evaluate_derivs!(
	rf::MultiClassLogisticRegressFunctor, 
	u::Matrix{Float64}, 
	y::Vector{Int}, 
	g::Matrix{Float64})

	K = size(u, 1)
	n = size(u, 2)

	for j in 1 : n
		yj = y[j]

		maxu = u[1, j]
		for k in 2 : K
			uk = u[k, j]
			if uk > maxu
				maxu = uk
			end
		end

		sj = 0.
		for k in 1 : K
			uk = u[k, j]
			sj += (g[k, j] = exp(uk - maxu))
		end

		inv_sj = 1.0 / sj
		for k in 1 : K
			g[k, j] *= inv_sj
		end
		g[yj, j] -= 1.0
	end
end

function evaluate_values_and_derivs!(
	rf::MultiClassLogisticRegressFunctor, 
	u::Matrix{Float64}, 
	y::Vector{Int}, 
	v::Vector{Float64},
	g::Matrix{Float64})

	K = size(u, 1)
	n = size(u, 2)

	for j in 1 : n		
		yj = y[j]

		maxu = u[1, j] 
		for k in 2 : K
			uk = u[k, j]
			if uk > maxu
				maxu = uk
			end
		end

		sj = 0.
		for k in 1 : K
			sj += (g[k, j] = exp(u[k, j] - maxu))
		end

		inv_sj = 1.0 / sj
		for k in 1 : K
			g[k, j] *= inv_sj
		end
		g[yj, j] -= 1.0

		v[j] = log(sj) + maxu - u[yj, j]
	end
end


function multiclass_logisticreg_objfun(K::Int, x::Matrix{Float64}, y::Vector{Int}, r::Regularizer; 
	by_columns::Bool=false, bias::Bool=false)

	generic_regress_objfun(MultiClassLogisticRegressFunctor(), K, x, y, 1.0; 
		by_columns=by_columns, bias=bias)
end

function multiclass_logisticreg(K::Int,
								x::Matrix{Float64}, 
                                y::Vector{Int}, 
                     			r::Regularizer,
                     			theta0::Matrix{Float64};
                     			by_columns::Bool=false,
                     			bias::Bool=false,
                     			method::Symbol = :bfgs,
                     			xtol::Float64 = 1.0e-12,
                     			ftol::Real = 1.0e-12,
                     			grtol::Real = 1.0e-8,
                     			iterations::Integer = 200, 
                     			show_trace::Bool=false)

	dt = size(x, by_columns ? 1 : 2) + int(bias)
	if size(theta0) != (dt, K)
		throw(ArgumentError("The dimension of theta0 is inconsistent with the problem."))
	end

	n = size(x, by_columns ? 2 : 1)
	if length(y) != n
		throw(ArgumentError("The size of y does not match the number of samples."))
	end

	objfun = multiclass_logisticreg_objfun(K, x, y, r; by_columns=by_columns, bias=bias)
	res = optimize(objfun, vec(theta0); method=method, 
		xtol=xtol, ftol=ftol, grtol=grtol, iterations=iterations, show_trace=show_trace)

	theta = reshape(res.minimum, dt, K)

	return (theta, res.f_minimum)
end



