# Linear regression

import Base.LinAlg.LAPACK.gels!
import Base.LinAlg.LAPACK.gelsy!
import Base.LinAlg.LAPACK.gelsd!


#################################################
#
#  Specific methods
#
#################################################

# qrlq

function llsq_qrlq{T<:FloatingPoint}(x::Matrix{T}, y::Vector{T}; by_columns::Bool=false, bias::Bool=false)
	xc = bias ? append_ones(x, by_columns ? 1 : 2) : copy(x)
	gels!(by_columns ? 'T' : 'N', xc, copy(y))[2]
end

function llsq_qrlq{T<:FloatingPoint}(x::Matrix{T}, y::Matrix{T}; by_columns::Bool=false, bias::Bool=false)
	if by_columns
		gels!('T', bias ? append_ones(x, 1) : copy(x), y')[2]
	else
		gels!('N', bias ? append_ones(x, 2) : copy(x), copy(y))[2]
	end
end

function wllsq_qrlq{T<:FloatingPoint}(x::Matrix{T}, y::Vector{T}, w::Vector{T}; by_columns::Bool=false, bias::Bool=false)
	sqw = sqrt(w)
	if by_columns
		xw = bmultiply(x, sqw, 2)
		gels!('T', bias ? vcat(xw, sqw') : xw, y .* sqw)[2]
	else
		xw = bmultiply(x, sqw, 1)
		gels!('N', bias ? hcat(xw, sqw) : xw, y .* sqw)[2]
	end
end

function wllsq_qrlq{T<:FloatingPoint}(x::Matrix{T}, y::Matrix{T}, w::Vector{T}; by_columns::Bool=false, bias::Bool=false)
	sqw = sqrt(w)
	if by_columns
		xw = bmultiply(x, sqw, 2)
		gels!('T', bias ? vcat(xw, sqw') : xw, bmultiply(y', sqw, 2))[2]
	else
		xw = bmultiply(x, sqw, 1)
		gels!('N', bias ? hcat(xw, sqw) : xw, bmultiply(y, sqw, 1))[2]
	end
end


# orth & svd

for (fun, lapackfun!) in [(:llsq_orth, :gelsy!), (:llsq_svd, :gelsd!)]

	@eval function ($fun){T<:FloatingPoint}(x::Matrix{T}, y::Vector{T}, rcond::T; 
		by_columns::Bool=false, bias::Bool=false)

		if by_columns
			($lapackfun!)(bias ? append_ones(x', 2) : x', copy(y), rcond)[1]
		else
			($lapackfun!)(bias ? append_ones(x, 2) : copy(x), copy(y), rcond)[1]
		end
	end

	@eval function ($fun){T<:FloatingPoint}(x::Matrix{T}, y::Matrix{T}, rcond::T; 
		by_columns::Bool=false, bias::Bool=false)

		if by_columns
			($lapackfun!)(bias ? append_ones(x', 2) : x', y', rcond)[1]
		else
			($lapackfun!)(bias ? append_ones(x, 2) : copy(x), copy(y), rcond)[1]
		end
	end
end


#################################################
#
#  Linear least square
#
#################################################

default_rcond{T<:FloatingPoint}(x::Matrix{T}) = eps(convert(T, length(x)))

function linearreg_lsq{T<:FloatingPoint}(
	x::Matrix{T}, 
	y::VecOrMat{T}; 
	method::Symbol=:qrlq, 
	by_columns::Bool=false,
	bias::Bool=false,
	weights::Union(Vector{T},Nothing)=nothing,
	rcond::T=-one(T))

	no_weights = weights == nothing

	if method == :qrlq
		if no_weights
			llsq_qrlq(x, y; by_columns=by_columns, bias=bias)
		else
			wllsq_qrlq(x, y, weights; by_columns=by_columns, bias=bias)
		end

	elseif method == :orth
		if rcond < 0 rcond = default_rcond(x) end
		llsq_orth(x, y, rcond; by_columns=by_columns, bias=bias)

	elseif method == :svd
		if rcond < 0 rcond = default_rcond(x) end
		llsq_svd(x, y, rcond; by_columns=by_columns, bias=bias)

	else
		throw(ArgumentError("Invalid method for linearreg_lsq."))
	end
end


