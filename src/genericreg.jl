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


function _check_thetadim(d::Int, theta::Vector)
	if length(theta) != d
		throw(ArgumentError("The dimension of theta is inconsistent with the problem."))
	end
end

function _genreg_objvalue(theta::Vector, values::Vector, r::Float64)
	v = sum(values) 
	if r > 0
		v += 0.5 * r * sqsum(theta) 
	end
	v::Float64
end

function _genreg_grad!(g::Vector, bycol::Bool, x::Matrix, theta::Vector, derivs::Vector, r::Float64)
	gemv!(bycol ? 'N' : 'T', 1.0, x, derivs, 0.0, g)
	if r > 0
		axpy!(r, theta, g)
	end
end


function generic_regress_objfun(
	rf::DifferentiableRegressFunctor, 
	x::Matrix{Float64}, 
	y::Vector{Float64}, 
	r::Float64; 
	by_columns::Bool=false)

	d::Int = 0
	n::Int = 0
	if by_columns
		d = size(x, 1)
		n = size(x, 2)
	else
		n = size(x, 1)
		d = size(x, 2)
	end

	if length(y) != n
		throw(ArgumentError("Argument dimensions must match."))
	end
	u = Array(Float64, n)
	values = Array(Float64, n)
	derivs = Array(Float64, n)

	uch = by_columns ? 'T' : 'N'

	function f(theta::Vector{Float64})
		_check_thetadim(d, theta)
		gemv!(uch, 1.0, x, theta, 0.0, u)
		evaluate_values!(rf, u, y, values)
		_genreg_objvalue(theta, values, r)
	end

	function g!(theta::Vector{Float64}, g::Vector{Float64})
		_check_thetadim(d, theta)
		gemv!(uch, 1.0, x, theta, 0.0, u)
		evaluate_derivs!(rf, u, y, derivs)
		_genreg_grad!(g, by_columns, x, theta, derivs, r)
	end

	function fg!(theta::Vector{Float64}, g::Vector{Float64})
		_check_thetadim(d, theta)
		gemv!(uch, 1.0, x, theta, 0.0, u)
		evaluate_values_and_derivs!(rf, u, y, values, derivs)
		_genreg_grad!(g, by_columns, x, theta, derivs, r)
		_genreg_objvalue(theta, values, r)
	end

	DifferentiableFunction(f, g!, fg!)
end




