# Linear Least Square

function _llsq{T<:BlasReal}(trans::Bool, X::StridedMatrix{T}, y::StridedVecOrMat{T}, bias::T, method::Symbol)

	X_ = if bias == zero(T)
		copy(X)
	else
		trans ? augment_cols(X, bias) :
				augment_rows(X, bias)
	end::Matrix{T}
	ch = trans ? 'T' : 'N'
	y_ = copy(y)
	rcond = eps(convert(T, length(X)))

	r = if method == :qrlq
		gels!(ch, X_, y_)[2]
	elseif method == :orth
		trans ? gelsy!(X_', y_, rcond)[1] : gelsy!(X_, y_, rcond)[1]
	elseif method == :svd
		trans ? gelsd!(X_', y_, rcond)[1] : gelsd!(X_, y_, rcond)[1]
	else
		error("Invalid method value: $method.")
	end

	return r
end


function llsq{T<:BlasReal}(X::StridedMatrix{T}, y::StridedVecOrMat{T};
						   bias::Real=zero(T),
						   method::Symbol=:qrlq,
						   trans::Bool=false)

	_llsq(trans, X, y, convert(T, bias)::T, method)
end


# Ridge regression

function _ridgereg{T<:BlasReal}(trans::Bool, X::StridedMatrix{T}, y::StridedVecOrMat{T}, bias::T, r::T)
	X_ = if bias == zero(T)
		X
	else
		trans ? augment_cols(X, bias) :
				augment_rows(X, bias)
	end

	H = trans ? A_mul_Bt(X_, X_) : At_mul_B(X_, X_)
	n = trans ? size(X, 1) : size(X, 2)
	n_ = bias == zero(T) ? n : n+1
	@assert size(H, 1) == size(H, 2) == n_
	for i = 1:n
		H[i,i] += r
	end
	rhs = trans ? X_ * y : X_'y
	cholfact!(H) \ rhs
end

function ridgereg{T<:BlasReal}(X::StridedMatrix{T}, y::StridedVecOrMat{T}, r::Real;
							   bias::Real=zero(T),
							   trans::Bool=false)

	_ridgereg(trans, X, y, convert(T, bias)::T, convert(T, r)::T)
end
