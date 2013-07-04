# Linear regression


#################################################
#
#  Least square regression
#
#################################################

function ordinary_least_squares(x::Matrix{Float64}, y::Vector{Float64})
	H = gemm('N', 'T', 1.0, x, x)
	g = gemv('N', 1.0, x, y)
	cholfact!(H) \ g
end

function ordinary_least_squares(x::Matrix{Float64}, y::Matrix{Float64})
	H = gemm('N', 'T', 1.0, x, x)
	g = gemm('N', 'T', 1.0, x, y)
	cholfact!(H) \ g	
end



