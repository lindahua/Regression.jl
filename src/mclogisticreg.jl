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


