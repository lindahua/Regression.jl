# Linear regression

import Base.LinAlg.LAPACK.gels!
import Base.LinAlg.LAPACK.gelsy!
import Base.LinAlg.LAPACK.gelsd!
using PDMats


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
		gels!('T', bias ? vcat(xw, sqw') : xw, bmultiply(y', sqw, 1))[2]
	else
		xw = bmultiply(x, sqw, 1)
		gels!('N', bias ? hcat(xw, sqw) : xw, bmultiply(y, sqw, 1))[2]
	end
end


# orth & svd

for (fun, wfun, lapackfun!) in [(:llsq_orth, :wllsq_orth, :gelsy!), (:llsq_svd, :wllsq_svd, :gelsd!)]

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

	@eval function ($wfun){T<:FloatingPoint}(x::Matrix{T}, y::Vector{T}, w::Vector{T}, rcond::T; 
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

	@eval function ($wfun){T<:FloatingPoint}(x::Matrix{T}, y::Matrix{T}, w::Vector{T}, rcond::T; 
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


#################################################
#
#  Linear least square
#
#################################################

default_rcond{T<:FloatingPoint}(x::Matrix{T}) = eps(convert(T, length(x)))

function linreg_chkdims(x::Matrix, y::Vector, w::Nothing, by_columns::Bool)
	size(x, by_columns ? 2 : 1) == length(y)
end

function linreg_chkdims(x::Matrix, y::Vector, w::Vector, by_columns::Bool)
	size(x, by_columns ? 2 : 1) == length(y) == length(w)
end

function linreg_chkdims(x::Matrix, y::Matrix, w::Nothing, by_columns::Bool)
	i = by_columns ? 2 : 1
	size(x, i) == size(y, i)
end

function linreg_chkdims(x::Matrix, y::Matrix, w::Vector, by_columns::Bool)
	i = by_columns ? 2 : 1
	size(x, i) == size(y, i) == length(w)
end


function linearreg_lsq{T<:FloatingPoint}(
	x::Matrix{T}, 
	y::VecOrMat{T}; 
	method::Symbol=:qrlq, 
	by_columns::Bool=false,
	bias::Bool=false,
	weights::Union(Vector{T},Nothing)=nothing,
	rcond::T=-one(T))

	if !linreg_chkdims(x, y, weights, by_columns)
		throw(ArgumentError("Argument dimensions must match."))
	end

	no_weights = weights == nothing

	if method == :qrlq
		if no_weights
			llsq_qrlq(x, y; by_columns=by_columns, bias=bias)
		else
			wllsq_qrlq(x, y, weights; by_columns=by_columns, bias=bias)
		end

	elseif method == :orth
		if rcond < 0 rcond = default_rcond(x) end
		if no_weights
			llsq_orth(x, y, rcond; by_columns=by_columns, bias=bias)
		else
			wllsq_orth(x, y, weights, rcond; by_columns=by_columns, bias=bias)
		end

	elseif method == :svd
		if rcond < 0 rcond = default_rcond(x) end
		if no_weights
			llsq_svd(x, y, rcond; by_columns=by_columns, bias=bias)
		else
			wllsq_svd(x, y, weights, rcond; by_columns=by_columns, bias=bias)
		end

	else
		throw(ArgumentError("Invalid method for linearreg_lsq."))
	end
end


#################################################
#
#  Ridge regression
#
#################################################


function _ridgereg_H(x::Matrix{Float64}, by_columns::Bool, w::Nothing)
	by_columns ? gemm('N', 'T', 1.0, x, x) : gemm('T', 'N', 1.0, x, x)
end

function _ridgereg_H(x::Matrix{Float64}, by_columns::Bool, w::Vector{Float64})
	if by_columns
		symmetrize!(gemm('N', 'T', 1.0, bmultiply(x, w, 2), x))
	else
		symmetrize!(gemm('T', 'N', 1.0, x, bmultiply(x, w, 1)))
	end
end

function _ridgereg_g(x::Matrix{Float64}, y::Vector{Float64}, by_columns::Bool, w::Nothing)
	gemv(by_columns ? 'N' : 'T', 1.0, x, y)
end

function _ridgereg_g(x::Matrix{Float64}, y::Vector{Float64}, by_columns::Bool, w::Vector{Float64})
	gemv(by_columns ? 'N' : 'T', 1.0, x, w .* y)
end

function _ridgereg_g(x::Matrix{Float64}, y::Matrix{Float64}, by_columns::Bool, w::Nothing)
	by_columns ? gemm('N', 'T', 1.0, x, y) : gemm('T', 'N', 1.0, x, y)
end

function _ridgereg_g(x::Matrix{Float64}, y::Matrix{Float64}, by_columns::Bool, w::Vector{Float64})
	if by_columns
		gemm('N', 'T', 1.0, bmultiply(x, w, 2), y)
	else
		gemm('T', 'N', 1.0, x, bmultiply(y, w, 1))
	end
end


function _add_dm1!(H::Matrix{Float64}, q::AbstractPDMat)
	d = size(H, 1) - 1
	H[1:d, 1:d] += full(q)
end

function _add_dm1!(H::Matrix{Float64}, q::ScalMat)
	v::Float64 = q.value
	d = size(H, 1) - 1
	for i in 1 : d
		H[i, i] += v
	end
end

function _add_dm1!(H::Matrix{Float64}, q::PDiagMat)
	v::Vector{Float64} = q.diag
	d = size(H, 1) - 1
	for i in 1 : d
		H[i, i] += v[i]
	end
end

function _add_dm1!(H::Matrix{Float64}, q::PDMat)
	d = size(H, 1) - 1
	H[1:d, 1:d] += q.mat
end


function ridgereg(
	x::Matrix{Float64}, 
	y::VecOrMat{Float64},
	q::AbstractPDMat;
	by_columns::Bool=false,
	bias::Bool=false,
	weights::Union(Vector{Float64},Nothing)=nothing)

	if !linreg_chkdims(x, y, weights, by_columns)
		throw(ArgumentError("Argument dimensions must match."))
	end	
	d = size(x, by_columns ? 1 : 2)
	if dim(q) != d
		throw(ArgumentError("The dimension of q is incorrect."))
	end

	if bias
		xa = append_ones(x, by_columns ? 1 : 2)
		H = _ridgereg_H(xa, by_columns, weights)
		g = _ridgereg_g(xa, y, by_columns, weights)

		@assert size(H) == (d+1, d+1)
		_add_dm1!(H, q)
		cholfact!(H) \ g
	else
		H = _ridgereg_H(x, by_columns, weights)
		g = _ridgereg_g(x, y, by_columns, weights)

		@assert size(H) == (d, d)
		add!(H, q)
		cholfact!(H) \ g
	end	
end

function ridgereg(
	x::Matrix{Float64}, 
	y::VecOrMat{Float64},
	r::Float64;
	by_columns::Bool=false,
	bias::Bool=false,
	weights::Union(Vector{Float64},Nothing)=nothing)

	d = size(x, by_columns ? 1 : 2)
	ridgereg(x, y, ScalMat(d, r), by_columns=by_columns, bias=bias, weights=weights)
end

function ridgereg(
	x::Matrix{Float64}, 
	y::VecOrMat{Float64},
	r::Vector{Float64};
	by_columns::Bool=false,
	bias::Bool=false,
	weights::Union(Vector{Float64},Nothing)=nothing)

	ridgereg(x, y, PDiagMat(r), by_columns=by_columns, bias=bias, weights=weights)
end

function ridgereg(
	x::Matrix{Float64}, 
	y::VecOrMat{Float64},
	r::Matrix{Float64};
	by_columns::Bool=false,
	bias::Bool=false,
	weights::Union(Vector{Float64},Nothing)=nothing)

	ridgereg(x, y, PDMat(r), by_columns=by_columns, bias=bias, weights=weights)
end


