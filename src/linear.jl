# Linear regression

import Base.LinAlg.LAPACK.gels!
import Base.LinAlg.LAPACK.gelsy!
import Base.LinAlg.LAPACK.gelsd!

#################################################
#
#  Linear least square
#
#################################################

function llsq{T<:FloatingPoint}(
	x::Matrix{T}, 
	y::Vector{T}; 
	method::Symbol=:qrlq, 
	by_columns::Bool=false,
	rcond::T=-one(T))

	if rcond < 0
		rcond = eps(convert(T, length(x)))
	end

	if method == :qrlq
		gels!(by_columns ? 'T' : 'N', copy(x), copy(y))[2]

	elseif method == :orth
		if by_columns
			gelsy!(x', copy(y), rcond)[1]
		else
			gelsy!(copy(x), copy(y), rcond)[1]
		end

	elseif method == :svd
		if by_columns
			gelsd!(x', copy(y), rcond)[1]
		else
			gelsd!(copy(x), copy(y), rcond)[1]
		end

	else
		throw(ArgumentError("Invalid method for llsq."))
	end
end

function llsq{T<:FloatingPoint}(
	x::Matrix{T}, 
	y::Matrix{T}; 
	method::Symbol=:qrlq, 
	by_columns::Bool=false,
	rcond::T=-one(T))

	if rcond < 0
		rcond = eps(convert(T, length(x)))
	end

	if method == :qrlq
		if by_columns
			gels!('T', copy(x), y')[2]
		else
			gels!('N', copy(x), copy(y))[2]
		end

	elseif method == :orth
		if by_columns
			gelsy!(x', y', rcond)[1]
		else
			gelsy!(copy(x), copy(y), rcond)[1]
		end

	elseif method == :svd
		if by_columns
			gelsd!(x', y', rcond)[1]
		else
			gelsd!(copy(x), copy(y), rcond)[1]
		end

	else
		throw(ArgumentError("Invalid method for llsq."))
	end
end

