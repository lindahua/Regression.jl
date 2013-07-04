module Regression
	using NumericExtensions
	using Optim

	export

	# linear
	ordinary_least_squares


	# sources

	include("common.jl")
	include("linear.jl")
end

