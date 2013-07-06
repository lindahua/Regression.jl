# Generic regression

##########################################################################
#
#  Generic regression formulation
#
#  Without bias:
#
#  minimize sum_i f(theta' * x_i, y_i) + (r/2) * ||theta||^2
#
#  With bias:
#
#  minimize sum_i f(theta' * x_i + theta0, y_i) + (r/2) * ||theta||^2
#
##########################################################################

abstract RegressFunctor

abstract DifferentiableRegressFunctor <: RegressFunctor

macro check_thetadim(d, theta)
	quote
		if length($theta) != ($d)
			throw(ArgumentError("The dimension of theta is inconsistent with the problem."))
		end
	end
end


function generic_regress_objfun(
	rf::DifferentiableRegressFunctor, 
	x::Matrix{Float64}, 
	y::Array, 
	r::Regularizer; 
	by_columns::Bool=false, 
	bias::Bool=false)

	# prepare meta info

	d::Int = 0
	n::Int = 0

	if by_columns		
		d = size(x, 1)
		n = size(x, 2)
		uch = 'T'
		gch = 'N'
		dim = 1
	else
		n = size(x, 1)
		d = size(x, 2)
		uch = 'N'
		gch = 'T'
		dim = 2
	end

	# preprocess x and r

	dt::Int = d
	if bias
		x::Matrix{Float64} = append_ones(x, dim)
		dt += 1
	end

	r = check_regularizer(d, r, bias)

	# prepare storage

	u = Array(Float64, n)
	values = Array(Float64, n)
	derivs = Array(Float64, n)

	# functions to evaluate objective and gradient

	function f(theta::Vector{Float64})
		@check_thetadim(dt, theta)
		gemv!(uch, 1.0, x, theta, 0.0, u)
		evaluate_values!(rf, u, y, values)
		sum(values) + regularize_cost(theta, r)
	end

	function g!(theta::Vector{Float64}, g::Vector{Float64})
		@check_thetadim(dt, theta)
		gemv!(uch, 1.0, x, theta, 0.0, u)
		evaluate_derivs!(rf, u, y, derivs)
		gemv!(gch, 1.0, x, derivs, 0.0, g)
		add_regularize_grad!(theta, r, g)
	end

	function fg!(theta::Vector{Float64}, g::Vector{Float64})
		@check_thetadim(dt, theta)
		gemv!(uch, 1.0, x, theta, 0.0, u)
		evaluate_values_and_derivs!(rf, u, y, values, derivs)
		gemv!(gch, 1.0, x, derivs, 0.0, g)
		add_regularize_grad!(theta, r, g)
		sum(values) + regularize_cost(theta, r)
	end

	DifferentiableFunction(f, g!, fg!)
end




