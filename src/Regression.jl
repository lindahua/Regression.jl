module Regression
	using NumericExtensions
	using Optim

	export

	# common
	append_zeros, append_ones,

	# linear
	llsq_qrlq, wllsq_qrlq, llsq_orth, wllsq_orth, llsq_svd, wllsq_svd,
	linearreg_lsq

	# sources

	include("common.jl")
	include("linearreg.jl")
end

