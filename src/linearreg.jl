# Linear regression


function llsq{T<:BlasReal}(trans::Bool, X::StridedMatrix{T}, y::StridedVecOrMat{T}, bias::T, method::Symbol)

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

	llsq(trans, X, y, convert(T, bias)::T, method)
end
