# Logistic regression

immutable LogisticRegressFunctor <: DifferentiableRegressFunctor end


####
#
#  rf(u, y) = log(1 + exp(-y * u))             ... (1)
#           = -(y * u) + log(1 + exp(y * u))   ... (2)
#
#  For numerical stability, use (1) when y * u > 0, or (2) otherwise
#  

function evaluate_values!(
	rf::LogisticRegressFunctor, 
	u::Vector{Float64}, 
	y::Vector{Float64}, 
	v::Vector{Float64})

	for i in 1 : length(u)
		x = y[i] * u[i]
		v[i] = x > 0 ? log1p(exp(-x)) : -x + log1p(exp(x))
	end
end

function evaluate_derivs!(
	rf::LogisticRegressFunctor, 
	u::Vector{Float64}, 
	y::Vector{Float64}, 
	g::Vector{Float64})

	for i in 1 : length(u)
		yi = y[i]
		x = yi * u[i]
		if x > 0
			t = exp(-x)
			g[i] = - yi * t / (1 + t)
		else
			g[i] = - yi / (1 + exp(x))
		end
	end
end

function evaluate_values_and_derivs!(
	rf::LogisticRegressFunctor, 
	u::Vector{Float64}, 
	y::Vector{Float64}, 
	v::Vector{Float64},
	g::Vector{Float64})

	for i in 1 : length(u)
		yi = y[i]
		x = yi * u[i]
		if x > 0
			t = exp(-x)
			v[i] = log1p(t)
			g[i] = - yi * t / (1 + t)
		else
			t = exp(x)
			v[i] = -x + log1p(t)
			g[i] = - yi / (1 + t)
		end
	end
end


####
#
#  driver function
#
####

function logisticreg_objfun(x::Matrix{Float64}, y::Vector{Float64}, r::Regularizer; 
	by_columns::Bool=false, bias::Bool=false)

	generic_regress_objfun(LogisticRegressFunctor(), x, y, 1.0; by_columns=by_columns, bias=bias)
end

function logisticreg(x::Matrix{Float64}, 
                     y::Vector{Float64}, 
                     r::Regularizer,
                     theta0::Vector{Float64};
                     by_columns::Bool=false,
                     bias::Bool=false,
                     method::Symbol = :bfgs,
                     xtol::Float64 = 1.0e-12,
                     ftol::Real = 1.0e-12,
                     grtol::Real = 1.0e-8,
                     iterations::Integer = 200, 
                     show_trace::Bool=false)

	dt = size(x, by_columns ? 1 : 2) + int(bias)
	if length(theta0) != dt
		throw(ArgumentError("The dimension of theta0 is inconsistent with the problem."))
	end

	n = size(x, by_columns ? 2 : 1)
	if length(y) != n
		throw(ArgumentError("The size of y does not match the number of samples."))
	end

	objfun = logisticreg_objfun(x, y, r; by_columns=by_columns, bias=bias)
	res = optimize(objfun, theta0; method=method, 
		xtol=xtol, ftol=ftol, grtol=grtol, iterations=iterations, show_trace=show_trace)

	return (res.minimum, res.f_minimum)
end


