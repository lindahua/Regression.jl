# Linear regression

#################################################
#
#  Specific methods
#
#################################################

# qrlq

function llsq_qrlq{T<:BlasReal}(x::Matrix{T}, y::Vector{T}; by_columns::Bool=false, bias::Bool=false)
	xc = bias ? append_ones(x, by_columns ? 1 : 2) : copy(x)
	gels!(by_columns ? 'T' : 'N', xc, copy(y))[2]
end

function llsq_qrlq{T<:BlasReal}(x::Matrix{T}, y::Matrix{T}; by_columns::Bool=false, bias::Bool=false)
	if by_columns
		gels!('T', bias ? append_ones(x, 1) : copy(x), y')[2]
	else
		gels!('N', bias ? append_ones(x, 2) : copy(x), copy(y))[2]
	end
end

function wllsq_qrlq{T<:BlasReal}(x::Matrix{T}, y::Vector{T}, w::Vector{T}; by_columns::Bool=false, bias::Bool=false)
	sqw = sqrt(w)
	if by_columns
		xw = bmultiply(x, sqw, 2)
		gels!('T', bias ? vcat(xw, sqw') : xw, y .* sqw)[2]
	else
		xw = bmultiply(x, sqw, 1)
		gels!('N', bias ? hcat(xw, sqw) : xw, y .* sqw)[2]
	end
end

function wllsq_qrlq{T<:BlasReal}(x::Matrix{T}, y::Matrix{T}, w::Vector{T}; by_columns::Bool=false, bias::Bool=false)
	sqw = sqrt(w)
	if by_columns
		xw = bmultiply(x, sqw, 2)
		gels!('T', bias ? vcat(xw, sqw') : xw, bmultiply(y', sqw, 1))[2]
	else
		xw = bmultiply(x, sqw, 1)
		gels!('N', bias ? hcat(xw, sqw) : xw, bmultiply(y, sqw, 1))[2]
	end
end


# orth & svd

for (fun, wfun, lapackfun!) in [(:llsq_orth, :wllsq_orth, :gelsy!), (:llsq_svd, :wllsq_svd, :gelsd!)]

	@eval function ($fun){T<:BlasReal}(x::Matrix{T}, y::Vector{T}, rcond::T;
		by_columns::Bool=false, bias::Bool=false)

		if by_columns
			($lapackfun!)(bias ? append_ones(x', 2) : x', copy(y), rcond)[1]
		else
			($lapackfun!)(bias ? append_ones(x, 2) : copy(x), copy(y), rcond)[1]
		end
	end

	@eval function ($fun){T<:BlasReal}(x::Matrix{T}, y::Matrix{T}, rcond::T;
		by_columns::Bool=false, bias::Bool=false)

		if by_columns
			($lapackfun!)(bias ? append_ones(x', 2) : x', y', rcond)[1]
		else
			($lapackfun!)(bias ? append_ones(x, 2) : copy(x), copy(y), rcond)[1]
		end
	end

	@eval function ($wfun){T<:BlasReal}(x::Matrix{T}, y::Vector{T}, w::Vector{T}, rcond::T;
		by_columns::Bool=false, bias::Bool=false)

		sqw = sqrt(w)

		if by_columns
			xwt = bmultiply(x', sqw, 1)
			($lapackfun!)(bias ? hcat(xwt, sqw) : xwt, y .* sqw, rcond)[1]
		else
			xw = bmultiply(x, sqw, 1)
			($lapackfun!)(bias ? hcat(xw, sqw) : xw, y .* sqw, rcond)[1]
		end
	end

	@eval function ($wfun){T<:BlasReal}(x::Matrix{T}, y::Matrix{T}, w::Vector{T}, rcond::T;
		by_columns::Bool=false, bias::Bool=false)

		sqw = sqrt(w)

		if by_columns
			xwt = bmultiply(x', sqw, 1)
			($lapackfun!)(bias ? hcat(xwt, sqw) : xwt, bmultiply(y', sqw, 1), rcond)[1]
		else
			xw = bmultiply(x, sqw, 1)
			($lapackfun!)(bias ? hcat(xw, sqw) : xw, bmultiply(y, sqw, 1), rcond)[1]
		end
	end
end
