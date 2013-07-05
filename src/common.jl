
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


