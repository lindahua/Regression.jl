module Regression
	using NumericExtensions
	using Optim

	export

	# common
	append_zeros, append_ones,

	# linear
	llsq

	# sources

	include("common.jl")
	include("linear.jl")
end

