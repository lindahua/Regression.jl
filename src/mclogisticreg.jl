# Multi-class logistic regression


function mc_uvalues!(x::Matrix, theta::Matrix, u::Matrix, by_columns::Bool)
	if by_columns
		gemm!('T', 'N', 1.0, theta, x, 0.0, u)
	else
		gemm!('T', 'T', 1.0, theta, x, 0.0, u)
	end
end

function mclogireg_evalobjv(
	u::Matrix{Float64}, 
	y::Vector{Int}, 
	theta::Matrix{Float64}, 
	r::Float64)

	K = size(u, 1)
	n = size(u, 2)

	v = 0.
	o = 0
	for i in 1 : n
		yi = y[i]

		maxu = u[o + 1]
		for k in 2 : K
			uk = u[o + k]
			if uk > maxu
				maxu = uk
			end
		end

		s = 0.
		for k in 1 : K
			uk = u[o + k]
			s += exp(uk - maxu)
		end
		vi = maxu + log(s) - u[o + yi]
		v += vi

		o += K
	end

	if r > 0
		v += 0.5 * r * sqsum(theta)
	end
	v
end


function mclogireg_evalgrad!(
	u::Matrix{Float64}, 
	y::Vector{Int}, 
	theta::Matrix{Float64}, 
	r::Float64
	gv::Matrix{Float64})

	K = size(u, 1)
	n = size(u, 2)

	o = 0
	for i in 1 : n
		yi = y[i]

		maxu = u[o + 1]
		for k in 2 : K
			uk = u[o + k]
			if uk > maxu
				maxu = uk
			end
		end

		s = 0.
		for k in 1 : K
			uk = u[o + k]
			s += exp(uk - maxu)
		end
	end
end





function multiclass_logisticreg_objfun(
	K::Int
	x::Matrix{Float64}, 
	y::Vector{Int}, 
	r::Float64;
	by_columns::Bool=false,
	bias::Bool=false)

	if by_columns
		d = size(x, 1)
		n = size(x, 2)
	else
		n = size(x, 1)
		d = size(x, 2)
	end
	
	td = bias ? d + 1 : d 

	# prepare storage
	u = Array(Float64, K, n)

	function f(theta::Vector{Float64})
		theta = reshape(theta, td, K)
		if bias
		else
			mc_uvalues!(x, theta, u, by_columns)
			mclogireg_evalobjv(u, y, theta, r)
		end
	end

	function g!(theta::Vector{Float64}, g::Vector{Float64})
		theta = reshape(theta, td, K)
		gv = reshape(g, td, K)
		@assert pointer(gv) == pointer(g)
		if bias
		else
			mc_uvalues!(x, theta, u, by_columns)
			mclogireg_evalgrad!(u, y, theta, r, gv)
		end
	end

	function fg!(theta::Vector{Float64}, g::Vector{Float64})
		theta = reshape(theta, td, K)
		gv = reshape(g, td, K)
		@assert pointer(gv) == pointer(g)
		if bias
		else
			mc_uvalues!(x, theta, u, by_columns)
			mclogireg_evalobjv_and_grad!(u, y, theta, r, gv)
		end
	end

	DifferentiableFunction(f, g!, fg!)
end



