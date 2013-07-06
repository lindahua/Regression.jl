
# import useful linear algebra tools

import Base.LinAlg.BLAS.axpy!
import Base.LinAlg.BLAS.gemv, Base.LinAlg.BLAS.gemv! 
import Base.LinAlg.BLAS.gemm, Base.LinAlg.BLAS.gemm! 

function append_zeros{T<:Number}(x::Matrix{T}, dim::Int)
	if dim == 1
		[x; zeros(T, 1, size(x, 2))]
	elseif dim == 2
		[x zeros(T, size(x, 1))]
	else
		error("The value for dim must be either 1 or 2.")
	end
end

function append_ones{T<:Number}(x::Matrix{T}, dim::Int)
	if dim == 1
		[x; ones(T, 1, size(x, 2))]
	elseif dim == 2
		[x ones(T, size(x, 1))]
	else
		error("The value for dim must be either 1 or 2.")
	end
end


typealias Regularizer Union(Float64, Vector{Float64})

function regularize_cost(a::Array{Float64}, r::Float64)
	0.5 * r * sqsum(a)
end

function regularize_cost(a::Vector{Float64}, r::Vector{Float64})
	0.5 * wsqsum(r, a)
end

function regularize_cost(a::Matrix{Float64}, r::Vector{Float64})
	s = 0.
	d = size(a, 1)
	for j in 1 : size(a, 2)
		sj = 0.
		for i in 1 : d
			@inbounds ai = a[o + i]
			@inbounds sj += abs2(ai) * r[i]
		end
		s += sj
		o += d
	end
	0.5 * s
end


function add_regularize_grad!(a::Array{Float64}, r::Float64, g::Array{Float64})
	if r != 0
		axpy!(r, a, g)
	end
end

function add_regularize_grad!(a::Vector{Float64}, r::Vector{Float64}, g::Vector{Float64})
	for i in 1 : length(g)
		@inbounds g[i] += a[i] * r[i]
	end
end

function add_regularize_grad!(a::Matrix{Float64}, r::Vector{Float64}, g::Matrix{Float64})
	d = size(a, 1)
	for j in 1 : size(a, 2)
		for i in 1 : d
			@inbounds g[o + i] += a[o + i] * r[i]
		end
		o += d
	end
end

function check_regularizer(d::Int, r::Float64, bias::Bool)
	if bias
		rv = Array(Float64, d+1)
		for i in 1 : d
			@inbounds rv[i] = r
		end 
		@inbounds rv[d+1] = 0.
		return rv
	else
		return r
	end
end

function check_regularizer(d::Int, r::Vector{Float64}, bias::Bool)
	if bias 
		d += 1
	end
	if length(r) != d
		throw(ArgumentError("The dimension of the regularizers is inconsistent with the solution."))
	end
	r
end

